"""Sustained-vowel feature extraction with a quality gate (task VOICE-20).

``extract_voice_features(path)`` is the ONE function that turns an audio file of
a sustained "aaah" into named voice measurements. Training data and uploaded
files both go through it, so the model always sees measurements made the same
way.

Steps:

1. ``audio_io.load_audio``: any format, mono, 8 kHz.
2. Find the vowel: the longest stretch where Praat finds a pitch, then drop its
   first and last 10%, keeping the steady middle where voice measures are most
   reliable. The audio around it is the background used for the noise check.
3. Quality gate. Two cheap checks run first, before any pitch tracking: the file
   is refused if it is much longer than one "aaah" (speech, a long conversation
   or a wrong file) or almost silent. Then it is refused, with plain-language
   reasons, if the vowel is too short, the original audio is clipped, the background noise is too
   loud, too little of the vowel has a measurable pitch, or any feature cannot
   be measured. A refused file gets NO features: it is never imputed or
   zero-filled (CLAUDE.md section 3.5).
4. 56 features (``FEATURE_INFO``): Praat measures via parselmouth (pitch, jitter,
   shimmer, HNR/NHR, formants, intensity) and librosa (MFCC 1-13, spectrum).
   The steady vowel is returned too, so ``uci_transfer.py`` can recompute the
   UCI-470 Praat measures from exactly the same stretch of audio.

Absolute loudness is not a feature: every microphone records at a different
volume, so it would measure the microphone, not the person. The vowel is
peak-normalised before the librosa features; jitter, shimmer, HNR, formants,
MFCC 1-13 and intensity variation do not depend on overall gain.

Nothing here learns from data, so it is safe to run before any split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import audio_io
import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call

# Praat settings for adult voices (Praat's own voice-report defaults).
PITCH_FLOOR_HZ = 75.0
PITCH_CEILING_HZ = 500.0
MAX_FORMANT_HZ = 3800.0  # below the 4 kHz limit of 8 kHz audio

# Quality gate thresholds.
MAX_DURATION_S = 30.0  # one "aaah" is 3-10 s; training files are at most 7.2 s
MIN_PEAK = 0.01  # -40 dB full scale: below this the file is nearly silent
MIN_VOWEL_S = 1.0  # steady vowel length after trimming the edges
MAX_CLIPPED_FRACTION = 0.01
MIN_SNR_DB = 10.0  # only checked when there is background around the vowel to measure
MIN_BACKGROUND_S = 0.25
NOISE_PERCENTILE = 20  # quietest fifth of the background frames
MIN_VOICED_FRACTION = 0.5
MAX_VOICE_BREAK_FRAMES = 10  # 0.1 s: shorter breaks do not split the vowel
EDGE_FRACTION = 0.10

# librosa frame settings at 8 kHz: 32 ms windows, 10 ms hop.
N_FFT = 256
HOP = 80

# Praat's voice-report arguments for jitter and shimmer.
JITTER_ARGS = (0, 0, 0.0001, 0.02, 1.3)
SHIMMER_ARGS = (*JITTER_ARGS, 1.6)

FEATURE_INFO: dict[str, tuple[str, str]] = {
    "f0_mean_hz": ("pitch", "average pitch of the voice"),
    "f0_sd_hz": ("pitch", "how much the pitch moves"),
    "f0_range_hz": ("pitch", "gap between the highest and lowest pitch"),
    "f0_cv": ("pitch", "pitch movement relative to the average pitch"),
    "jitter_local": ("jitter", "cycle-to-cycle wobble in pitch"),
    "jitter_local_abs": ("jitter", "cycle-to-cycle wobble in pitch, in seconds"),
    "jitter_rap": ("jitter", "short-term pitch wobble (3 cycles)"),
    "jitter_ppq5": ("jitter", "short-term pitch wobble (5 cycles)"),
    "jitter_ddp": ("jitter", "change in pitch wobble between cycles"),
    "shimmer_local": ("shimmer", "cycle-to-cycle wobble in loudness"),
    "shimmer_local_db": ("shimmer", "cycle-to-cycle wobble in loudness, in dB"),
    "shimmer_apq3": ("shimmer", "short-term loudness wobble (3 cycles)"),
    "shimmer_apq5": ("shimmer", "short-term loudness wobble (5 cycles)"),
    "shimmer_apq11": ("shimmer", "loudness wobble over 11 cycles"),
    "shimmer_dda": ("shimmer", "change in loudness wobble between cycles"),
    "hnr_db": ("noise", "how clear the voice is (higher is clearer)"),
    "nhr": ("noise", "how breathy or hoarse the voice is (higher is noisier)"),
    "f1_mean_hz": ("formants", "first resonance of the mouth and throat"),
    "f2_mean_hz": ("formants", "second resonance of the mouth and throat"),
    "f3_mean_hz": ("formants", "third resonance of the mouth and throat"),
    "f1_bandwidth_hz": ("formants", "width of the first resonance"),
    "f2_bandwidth_hz": ("formants", "width of the second resonance"),
    "f3_bandwidth_hz": ("formants", "width of the third resonance"),
    "intensity_sd_db": ("loudness change", "how much the loudness moves"),
    "intensity_delta_sd_db": ("loudness change", "how fast the loudness changes"),
    **{
        f"mfcc{i}_{stat}": (
            "mfcc",
            f"sound colour band {i}, {'average' if stat == 'mean' else 'variation'}",
        )
        for i in range(1, 14)
        for stat in ("mean", "sd")
    },
    "spectral_centroid_hz": ("spectrum", "where the sound energy is centred"),
    "spectral_bandwidth_hz": ("spectrum", "how spread out the sound energy is"),
    "spectral_rolloff_hz": ("spectrum", "frequency below which most energy sits"),
    "spectral_flatness": ("spectrum", "how noise-like the sound is"),
    "spectral_slope_db_per_khz": ("spectrum", "how fast energy falls at higher pitches"),
}
FEATURE_NAMES: tuple[str, ...] = tuple(FEATURE_INFO)


@dataclass
class QualityReport:
    passed: bool
    problems: list[str]  # plain-language reasons the file was refused
    info: dict[str, Any] = field(default_factory=dict)  # format, durations, SNR, ...


@dataclass
class Extraction:
    features: dict[str, float] | None  # None when the quality gate failed
    quality: QualityReport
    vowel: np.ndarray | None = None  # the steady vowel, kept for further measures
    sample_rate: int = audio_io.TARGET_SR


def _frame_rms(samples: np.ndarray) -> np.ndarray:
    return librosa.feature.rms(y=samples, frame_length=N_FFT, hop_length=HOP)[0]


def _longest_voiced_run(voiced: np.ndarray, max_gap: int) -> tuple[int, int] | None:
    """First and last frame of the longest voiced run, bridging gaps of up to ``max_gap`` frames.

    Short voice breaks inside an "aaah" are common in PD, so they do not split the vowel.
    """
    frames = np.flatnonzero(voiced)
    if len(frames) == 0:
        return None
    runs, start = [], frames[0]
    for previous, current in zip(frames[:-1], frames[1:], strict=True):
        if current - previous > max_gap + 1:
            runs.append((start, previous))
            start = current
    runs.append((start, frames[-1]))
    return max(runs, key=lambda run: run[1] - run[0])


def find_vowel(clip: audio_io.AudioClip) -> tuple[np.ndarray, dict[str, Any]]:
    """The steady middle of the "aaah", plus the background noise level around it.

    The vowel is where Praat finds a pitch (the longest voiced stretch); its first
    and last 10% are dropped. Everything well outside it is background, so the
    noise check works even when the noise is loud.
    """
    samples, sr = clip.samples, clip.sample_rate
    info: dict[str, Any] = {"snr_db": None}
    sound = parselmouth.Sound(samples.astype(np.float64), sampling_frequency=sr)
    pitch = sound.to_pitch(
        time_step=0.01, pitch_floor=PITCH_FLOOR_HZ, pitch_ceiling=PITCH_CEILING_HZ
    )
    run = _longest_voiced_run(pitch.selected_array["frequency"] > 0, max_gap=MAX_VOICE_BREAK_FRAMES)
    if run is None:
        return samples[:0], info

    times = pitch.xs()
    start = max(int((times[run[0]] - 0.005) * sr), 0)
    end = min(int((times[run[1]] + 0.005) * sr), len(samples))
    edge = int((end - start) * EDGE_FRACTION)
    vowel = samples[start + edge : end - edge]

    # Background = everything at least 50 ms away from the vowel. It can still hold
    # voice the pitch tracker skipped (breathy onsets, voice breaks: common in PD),
    # so the noise level is taken from its quietest frames, not its average.
    # Otherwise the gate would refuse PD voices more often than healthy ones.
    margin = int(0.05 * sr)
    background = np.concatenate([samples[: max(start - margin, 0)], samples[end + margin :]])
    if len(background) >= int(MIN_BACKGROUND_S * sr) and len(vowel) > 0:
        noise = float(np.percentile(_frame_rms(background), NOISE_PERCENTILE))
        signal = float(np.sqrt(np.mean(vowel.astype(np.float64) ** 2)))
        if noise > 1e-6:  # digital silence has no measurable noise
            info["snr_db"] = round(20 * np.log10(signal / noise), 1)
    return vowel, info


def _praat_features(sound: parselmouth.Sound) -> tuple[dict[str, float], float]:
    """Praat voice-report measures, and the share of pitch frames that are voiced."""
    pitch = sound.to_pitch(
        time_step=0.01, pitch_floor=PITCH_FLOOR_HZ, pitch_ceiling=PITCH_CEILING_HZ
    )
    f0 = pitch.selected_array["frequency"]
    voiced = f0[f0 > 0]
    voiced_fraction = float(len(voiced) / len(f0)) if len(f0) else 0.0

    points = call(sound, "To PointProcess (periodic, cc)", PITCH_FLOOR_HZ, PITCH_CEILING_HZ)

    harmonicity = sound.to_harmonicity_cc(
        time_step=0.01, minimum_pitch=PITCH_FLOOR_HZ, silence_threshold=0.1, periods_per_window=1.0
    )
    hnr_frames = harmonicity.values[0]
    hnr_frames = hnr_frames[hnr_frames != -200]  # Praat marks unvoiced frames as -200

    formant = sound.to_formant_burg(
        time_step=0.01, max_number_of_formants=4, maximum_formant=MAX_FORMANT_HZ
    )
    voiced_times = pitch.xs()[f0 > 0]

    def formant_mean(index: int, what: str) -> float:
        command = "Get value at time" if what == "value" else "Get bandwidth at time"
        values = [call(formant, command, index, t, "Hertz", "Linear") for t in voiced_times]
        return float(np.nanmean(values)) if values else float("nan")

    intensity = sound.to_intensity(minimum_pitch=PITCH_FLOOR_HZ).values[0]

    mean_f0 = float(np.mean(voiced)) if len(voiced) else float("nan")
    sd_f0 = float(np.std(voiced)) if len(voiced) else float("nan")
    features = {
        "f0_mean_hz": mean_f0,
        "f0_sd_hz": sd_f0,
        "f0_range_hz": float(np.ptp(voiced)) if len(voiced) else float("nan"),
        "f0_cv": sd_f0 / mean_f0 if len(voiced) else float("nan"),
        "jitter_local": call(points, "Get jitter (local)", *JITTER_ARGS),
        "jitter_local_abs": call(points, "Get jitter (local, absolute)", *JITTER_ARGS),
        "jitter_rap": call(points, "Get jitter (rap)", *JITTER_ARGS),
        "jitter_ppq5": call(points, "Get jitter (ppq5)", *JITTER_ARGS),
        "jitter_ddp": call(points, "Get jitter (ddp)", *JITTER_ARGS),
        "shimmer_local": call([sound, points], "Get shimmer (local)", *SHIMMER_ARGS),
        "shimmer_local_db": call([sound, points], "Get shimmer (local_dB)", *SHIMMER_ARGS),
        "shimmer_apq3": call([sound, points], "Get shimmer (apq3)", *SHIMMER_ARGS),
        "shimmer_apq5": call([sound, points], "Get shimmer (apq5)", *SHIMMER_ARGS),
        "shimmer_apq11": call([sound, points], "Get shimmer (apq11)", *SHIMMER_ARGS),
        "shimmer_dda": call([sound, points], "Get shimmer (dda)", *SHIMMER_ARGS),
        "hnr_db": float(np.mean(hnr_frames)) if len(hnr_frames) else float("nan"),
        "nhr": float(np.mean(10 ** (-hnr_frames / 10))) if len(hnr_frames) else float("nan"),
        "f1_mean_hz": formant_mean(1, "value"),
        "f2_mean_hz": formant_mean(2, "value"),
        "f3_mean_hz": formant_mean(3, "value"),
        "f1_bandwidth_hz": formant_mean(1, "bandwidth"),
        "f2_bandwidth_hz": formant_mean(2, "bandwidth"),
        "f3_bandwidth_hz": formant_mean(3, "bandwidth"),
        "intensity_sd_db": float(np.std(intensity)),
        "intensity_delta_sd_db": float(np.std(np.diff(intensity))) if len(intensity) > 1 else 0.0,
    }
    return {k: float(v) for k, v in features.items()}, voiced_fraction


def _librosa_features(vowel: np.ndarray, sr: int) -> dict[str, float]:
    """MFCC 1-13 (mean and SD) and spectral shape, on the peak-normalised vowel."""
    y = vowel / max(float(np.max(np.abs(vowel))), 1e-9)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=14, n_fft=N_FFT, hop_length=HOP, n_mels=40, fmax=sr / 2
    )[1:]  # coefficient 0 is overall loudness: left out on purpose
    spectrum = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    mean_db = librosa.power_to_db(np.mean(spectrum**2, axis=1) + 1e-12)
    slope_db_per_hz = float(np.polyfit(freqs, mean_db, 1)[0])

    features: dict[str, float] = {}
    for i, row in enumerate(mfcc, start=1):
        features[f"mfcc{i}_mean"] = float(np.mean(row))
        features[f"mfcc{i}_sd"] = float(np.std(row))
    features["spectral_centroid_hz"] = float(
        np.mean(librosa.feature.spectral_centroid(S=spectrum, sr=sr))
    )
    features["spectral_bandwidth_hz"] = float(
        np.mean(librosa.feature.spectral_bandwidth(S=spectrum, sr=sr))
    )
    features["spectral_rolloff_hz"] = float(
        np.mean(librosa.feature.spectral_rolloff(S=spectrum, sr=sr, roll_percent=0.85))
    )
    features["spectral_flatness"] = float(np.mean(librosa.feature.spectral_flatness(S=spectrum)))
    features["spectral_slope_db_per_khz"] = slope_db_per_hz * 1000
    return features


def early_problems(clip: audio_io.AudioClip) -> list[str]:
    """Checks that need no pitch tracking, run before the slow steps."""
    problems = []
    if clip.duration_s > MAX_DURATION_S:
        problems.append(
            f"The recording is {clip.duration_s:.0f} s long; one steady 'aaah' of 3 to 10 s is "
            "expected. Upload only the 'aaah', not speech or a longer recording."
        )
    peak = float(np.max(np.abs(clip.samples))) if clip.samples.size else 0.0
    if peak < MIN_PEAK:
        problems.append(
            "The recording is almost silent: no 'aaah' could be heard. Check the microphone "
            "and record closer to it."
        )
    return problems


def extract_voice_features(path: str | Path) -> Extraction:
    """Quality-checked voice features for one sustained-vowel recording.

    Raises ``audio_io.AudioLoadError`` only if the file is not audio at all; any
    audio that is merely unusable comes back with ``features=None`` and reasons.
    """
    clip = audio_io.load_audio(path)
    early = early_problems(clip)
    if early:
        info = {"source_format": clip.source_format, "duration_s": round(clip.duration_s, 2)}
        return Extraction(None, QualityReport(False, early, info))
    vowel, vowel_info = find_vowel(clip)
    vowel_s = len(vowel) / clip.sample_rate
    info: dict[str, Any] = {
        "source_format": clip.source_format,
        "source_sample_rate": clip.source_sample_rate,
        "source_channels": clip.source_channels,
        "duration_s": round(clip.duration_s, 2),
        "vowel_s": round(vowel_s, 2),
        "clipped_fraction": round(clip.clipped_fraction, 4),
        **vowel_info,
    }

    problems: list[str] = []
    if vowel_s < MIN_VOWEL_S:
        problems.append(
            f"The steady 'aaah' is only {vowel_s:.1f} s long; at least {MIN_VOWEL_S:.0f} s "
            "is needed. Record a longer, continuous 'aaah'."
        )
    if clip.clipped_fraction > MAX_CLIPPED_FRACTION:
        problems.append(
            f"{clip.clipped_fraction:.1%} of the recording is clipped (too loud). "
            "Record further from the microphone."
        )
    snr = vowel_info["snr_db"]
    if snr is not None and snr < MIN_SNR_DB:
        problems.append(
            f"Background noise is too loud (voice only {snr:.0f} dB above the noise). "
            "Record in a quieter room."
        )
    if problems:
        return Extraction(None, QualityReport(False, problems, info))

    sound = parselmouth.Sound(vowel.astype(np.float64), sampling_frequency=clip.sample_rate)
    praat, voiced_fraction = _praat_features(sound)
    info["voiced_fraction"] = round(voiced_fraction, 3)
    if voiced_fraction < MIN_VOICED_FRACTION:
        problems.append(
            f"A steady pitch was found in only {voiced_fraction:.0%} of the 'aaah'. "
            "Record a steady, continuous vowel."
        )
        return Extraction(None, QualityReport(False, problems, info))

    features = {**praat, **_librosa_features(vowel, clip.sample_rate)}
    unmeasured = [name for name in FEATURE_NAMES if not np.isfinite(features.get(name, np.nan))]
    if unmeasured:
        problems.append(
            "Some voice measures could not be computed from this recording "
            f"({', '.join(unmeasured[:3])}{', ...' if len(unmeasured) > 3 else ''})."
        )
        return Extraction(None, QualityReport(False, problems, info))

    ordered = {name: features[name] for name in FEATURE_NAMES}
    return Extraction(ordered, QualityReport(True, [], info), vowel, clip.sample_rate)
