"""Tests for audio loading, the quality gate and feature extraction (VOICE-20).

All audio here is synthetic: a harmonic "vowel" with a known pitch, built cycle by
cycle so pitch wobble (jitter) can be set exactly. No real recordings are used.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import audio_features
import audio_io
import contract
import numpy as np
import pytest
import soundfile as sf

SR = 16000


def synthetic_vowel(
    f0: float = 150.0,
    seconds: float = 2.0,
    jitter: float = 0.0,
    amplitude: float = 0.5,
    pad_s: float = 0.0,
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """A voiced 'aaah': one glottal-like cycle at a time, each cycle's length wobbling by ``jitter``."""
    rng = np.random.default_rng(seed)
    cycles, total = [], 0
    while total < seconds * SR:
        period = (1 / f0) * (1 + jitter * rng.standard_normal())
        t = np.arange(int(round(period * SR))) / SR
        cycle = sum(np.sin(2 * np.pi * k * t / period) / k**1.5 for k in range(1, 12))
        cycles.append(cycle)
        total += len(cycle)
    voice = np.concatenate(cycles)[: int(seconds * SR)]
    voice = amplitude * voice / np.max(np.abs(voice))
    pad = np.zeros(int(pad_s * SR))
    signal = np.concatenate([pad, voice, pad])
    return signal + noise * rng.standard_normal(len(signal))


def write_wav(path: Path, samples: np.ndarray, sr: int = SR) -> Path:
    sf.write(path, samples.astype(np.float32), sr, subtype="FLOAT")
    return path


@pytest.fixture
def clean_vowel(tmp_path: Path) -> Path:
    return write_wav(tmp_path / "clean.wav", synthetic_vowel(pad_s=0.3))


# --- feature list -----------------------------------------------------------


def test_feature_list_has_56_named_features_that_pass_the_contract() -> None:
    assert len(audio_features.FEATURE_NAMES) == 56
    assert len(set(audio_features.FEATURE_NAMES)) == 56
    contract.validate_features(audio_features.FEATURE_NAMES)


# --- loading ----------------------------------------------------------------


def test_load_audio_makes_stereo_44k_into_mono_8k(tmp_path: Path) -> None:
    mono = synthetic_vowel()
    stereo = np.stack([mono, mono], axis=1)
    path = tmp_path / "stereo.wav"
    sf.write(path, stereo.astype(np.float32), 44100)
    clip = audio_io.load_audio(path)
    assert clip.sample_rate == audio_io.TARGET_SR == 8000
    assert clip.samples.ndim == 1
    assert (clip.source_sample_rate, clip.source_channels, clip.source_format) == (44100, 2, "wav")


def test_load_audio_refuses_missing_and_non_audio_files(tmp_path: Path) -> None:
    with pytest.raises(audio_io.AudioLoadError, match="no such file"):
        audio_io.load_audio(tmp_path / "missing.wav")
    fake = tmp_path / "fake.wav"
    fake.write_text("this is not audio")
    with pytest.raises(audio_io.AudioLoadError):
        audio_io.load_audio(fake)


# --- extraction on good audio -----------------------------------------------


def test_clean_vowel_passes_with_the_right_pitch(clean_vowel: Path) -> None:
    result = audio_features.extract_voice_features(clean_vowel)
    assert result.quality.passed, result.quality.problems
    assert result.features is not None
    assert list(result.features) == list(audio_features.FEATURE_NAMES)
    assert all(np.isfinite(v) for v in result.features.values())
    assert result.features["f0_mean_hz"] == pytest.approx(150, abs=3)
    assert result.features["jitter_local"] < 0.005
    assert result.vowel is not None and len(result.vowel) > 0


def test_more_pitch_wobble_gives_higher_jitter(tmp_path: Path) -> None:
    steady = write_wav(tmp_path / "steady.wav", synthetic_vowel(jitter=0.0, pad_s=0.3))
    wobbly = write_wav(tmp_path / "wobbly.wav", synthetic_vowel(jitter=0.03, pad_s=0.3))
    steady_jitter = audio_features.extract_voice_features(steady).features["jitter_local"]
    wobbly_jitter = audio_features.extract_voice_features(wobbly).features["jitter_local"]
    assert wobbly_jitter > 3 * steady_jitter
    assert wobbly_jitter > 0.01


# --- quality gate: each check must refuse bad audio --------------------------


def _refused(path: Path) -> list[str]:
    result = audio_features.extract_voice_features(path)
    assert not result.quality.passed
    assert result.features is None  # never zero-filled or imputed
    assert result.quality.problems
    return result.quality.problems


def test_silence_is_refused(tmp_path: Path) -> None:
    problems = _refused(write_wav(tmp_path / "silence.wav", np.zeros(SR * 2)))
    assert "aaah" in problems[0]


def test_too_short_vowel_is_refused(tmp_path: Path) -> None:
    problems = _refused(write_wav(tmp_path / "short.wav", synthetic_vowel(seconds=0.6, pad_s=0.3)))
    assert "only" in problems[0] and "long" in problems[0]


def test_clipped_recording_is_refused(tmp_path: Path) -> None:
    loud = np.clip(synthetic_vowel(amplitude=3.0, pad_s=0.3), -1.0, 1.0)
    problems = _refused(write_wav(tmp_path / "clipped.wav", loud))
    assert any("clipped" in p for p in problems)


def test_noisy_recording_is_refused(tmp_path: Path) -> None:
    noisy = synthetic_vowel(amplitude=0.3, pad_s=0.6, noise=0.15)
    problems = _refused(write_wav(tmp_path / "noisy.wav", noisy))
    assert any("noise" in p for p in problems)


def test_recording_much_longer_than_one_aaah_is_refused_before_pitch_tracking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_pitch_tracking(*args: object) -> None:
        raise AssertionError("the slow steps must not run for a refused file")

    monkeypatch.setattr(audio_features, "find_vowel", no_pitch_tracking)
    long_file = synthetic_vowel(seconds=audio_features.MAX_DURATION_S + 2)
    problems = _refused(write_wav(tmp_path / "long.wav", long_file))
    assert "s long" in problems[0] and "speech" in problems[0]


def test_nearly_silent_recording_is_refused(tmp_path: Path) -> None:
    quiet = synthetic_vowel(amplitude=audio_features.MIN_PEAK / 2, pad_s=0.3)
    problems = _refused(write_wav(tmp_path / "quiet.wav", quiet))
    assert "almost silent" in problems[0]


def test_quiet_but_audible_recording_still_passes(tmp_path: Path) -> None:
    quiet = synthetic_vowel(amplitude=audio_features.MIN_PEAK * 5, pad_s=0.3)
    result = audio_features.extract_voice_features(write_wav(tmp_path / "quiet.wav", quiet))
    assert result.quality.passed, result.quality.problems


def test_unvoiced_noise_is_refused(tmp_path: Path) -> None:
    hiss = np.random.default_rng(1).normal(scale=0.2, size=SR * 2)
    _refused(write_wav(tmp_path / "hiss.wav", hiss))


# --- formats ------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
@pytest.mark.parametrize("extension", ["mp3", "m4a", "flac"])
def test_compressed_copy_gives_close_values(clean_vowel: Path, extension: str) -> None:
    copy = clean_vowel.with_suffix(f".{extension}")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(clean_vowel), "-b:a", "128k", str(copy)],
        check=True,
    )
    original = audio_features.extract_voice_features(clean_vowel)
    converted = audio_features.extract_voice_features(copy)
    assert converted.quality.passed, converted.quality.problems
    assert converted.quality.info["source_format"] == extension
    assert converted.features["f0_mean_hz"] == pytest.approx(
        original.features["f0_mean_hz"], rel=0.02
    )
    assert converted.features["hnr_db"] == pytest.approx(original.features["hnr_db"], abs=3)
