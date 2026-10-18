"""Recompute UCI-470's Praat "Baseline" measures from an audio file (task VOICE-25).

UCI-470's Baseline features were computed with Praat, which ``parselmouth`` runs
too. ``voice_report`` runs Praat's own "Voice report" on the steady vowel that
``audio_features`` finds and returns 16 of them under their UCI-470 column names
and units (Praat prints percentages; UCI-470 stores fractions).

Kept (``UCI_PRAAT_FEATURES``): mean and SD of the period, 5 jitter, 6 shimmer and
3 harmonicity measures. Left out on purpose:
- numPulses, numPeriodsPulses: depend on how long the recording is;
- PPE, DFA, RPDE: not Praat measures (custom algorithms, settings unknown);
- intensity: depends on microphone gain; formants: Praat settings unknown, and
  F4 sits above the 4 kHz limit of telephone audio.

How the UCI-470 model is trained and tested on these is in ``uci_transfer.py``.
"""

from __future__ import annotations

import re

import audio_features
import numpy as np
import parselmouth
from parselmouth.praat import call

# UCI-470 column -> (Voice report line label, scale to UCI units). Percent -> fraction.
_REPORT_LINES: dict[str, tuple[str, float]] = {
    "meanPeriodPulses": ("Mean period", 1.0),
    "stdDevPeriodPulses": ("Standard deviation of period", 1.0),
    "locPctJitter": ("Jitter (local)", 0.01),
    "locAbsJitter": ("Jitter (local, absolute)", 1.0),
    "rapJitter": ("Jitter (rap)", 0.01),
    "ppq5Jitter": ("Jitter (ppq5)", 0.01),
    "ddpJitter": ("Jitter (ddp)", 0.01),
    "locShimmer": ("Shimmer (local)", 0.01),
    "locDbShimmer": ("Shimmer (local, dB)", 1.0),
    "apq3Shimmer": ("Shimmer (apq3)", 0.01),
    "apq5Shimmer": ("Shimmer (apq5)", 0.01),
    "apq11Shimmer": ("Shimmer (apq11)", 0.01),
    "ddaShimmer": ("Shimmer (dda)", 0.01),
    "meanAutoCorrHarmonicity": ("Mean autocorrelation", 1.0),
    "meanNoiseToHarmHarmonicity": ("Mean noise-to-harmonics ratio", 1.0),
    "meanHarmToNoiseHarmonicity": ("Mean harmonics-to-noise ratio", 1.0),
}
UCI_PRAAT_FEATURES: tuple[str, ...] = tuple(_REPORT_LINES)

UCI_FEATURE_INFO: dict[str, tuple[str, str]] = {
    "meanPeriodPulses": ("pitch", "average length of one voice cycle (inverse of pitch)"),
    "stdDevPeriodPulses": ("pitch", "how much the voice-cycle length varies"),
    "locPctJitter": ("jitter", "cycle-to-cycle wobble in pitch"),
    "locAbsJitter": ("jitter", "cycle-to-cycle wobble in pitch, in seconds"),
    "rapJitter": ("jitter", "short-term pitch wobble (3 cycles)"),
    "ppq5Jitter": ("jitter", "short-term pitch wobble (5 cycles)"),
    "ddpJitter": ("jitter", "change in pitch wobble between cycles"),
    "locShimmer": ("shimmer", "cycle-to-cycle wobble in loudness"),
    "locDbShimmer": ("shimmer", "cycle-to-cycle wobble in loudness, in dB"),
    "apq3Shimmer": ("shimmer", "short-term loudness wobble (3 cycles)"),
    "apq5Shimmer": ("shimmer", "short-term loudness wobble (5 cycles)"),
    "apq11Shimmer": ("shimmer", "loudness wobble over 11 cycles"),
    "ddaShimmer": ("shimmer", "change in loudness wobble between cycles"),
    "meanAutoCorrHarmonicity": ("noise", "how regular the voice is (1 = perfectly regular)"),
    "meanNoiseToHarmHarmonicity": ("noise", "how breathy or hoarse the voice is"),
    "meanHarmToNoiseHarmonicity": ("noise", "how clear the voice is (higher is clearer)"),
}

_NUMBER = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"


def voice_report(vowel: np.ndarray, sr: int) -> dict[str, float]:
    """The 16 UCI-470 Baseline measures from Praat's "Voice report" on ``vowel``.

    Missing or undefined values come back as NaN; the caller refuses the file.
    """
    sound = parselmouth.Sound(np.asarray(vowel, dtype=np.float64), sampling_frequency=sr)
    floor, ceiling = audio_features.PITCH_FLOOR_HZ, audio_features.PITCH_CEILING_HZ
    pitch = call(sound, "To Pitch (cc)", 0, floor, 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, ceiling)
    pulses = call([sound, pitch], "To PointProcess (cc)")
    report = call(
        [sound, pitch, pulses], "Voice report", 0, 0, floor, ceiling, 1.3, 1.6, 0.03, 0.45
    )

    values: dict[str, float] = {}
    for column, (label, scale) in _REPORT_LINES.items():
        match = re.search(rf"^\s*{re.escape(label)}:\s*{_NUMBER}", report, re.MULTILINE)
        values[column] = float(match.group(1)) * scale if match else float("nan")
    return values


def uci_features_from(extraction: audio_features.Extraction) -> dict[str, float]:
    """UCI-470 Praat measures for a quality-passed extraction; {} if it was refused."""
    if extraction.vowel is None or extraction.features is None:
        return {}
    return voice_report(extraction.vowel, extraction.sample_rate)
