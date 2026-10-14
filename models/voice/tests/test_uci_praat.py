"""Tests for recomputing UCI-470 Praat measures and the unit check (VOICE-25).

Synthetic voices and synthetic UCI-shaped tables only.
"""

from __future__ import annotations

import contract
import numpy as np
import pandas as pd
import pytest
import uci_praat
import uci_transfer
from test_audio_features import SR, synthetic_vowel

FEATURES = list(uci_praat.UCI_PRAAT_FEATURES)


def test_sixteen_uci_columns_that_pass_the_contract() -> None:
    assert len(FEATURES) == 16
    contract.validate_features(FEATURES)
    assert set(uci_praat.UCI_FEATURE_INFO) == set(FEATURES)


def test_voice_report_gives_uci_units() -> None:
    values = uci_praat.voice_report(synthetic_vowel(f0=150, jitter=0.01), SR)
    assert set(values) == set(FEATURES)
    assert all(np.isfinite(v) for v in values.values())
    assert values["meanPeriodPulses"] == pytest.approx(1 / 150, rel=0.03)  # seconds
    assert 0 < values["locPctJitter"] < 0.1  # a fraction, not a percentage
    assert 0 < values["locShimmer"] < 0.5
    assert 0 < values["meanAutoCorrHarmonicity"] <= 1


def test_more_wobble_gives_more_uci_jitter() -> None:
    steady = uci_praat.voice_report(synthetic_vowel(jitter=0.0), SR)["locPctJitter"]
    wobbly = uci_praat.voice_report(synthetic_vowel(jitter=0.03), SR)["locPctJitter"]
    assert wobbly > 3 * steady


def test_silence_gives_nan_not_zero() -> None:
    values = uci_praat.voice_report(np.zeros(SR), SR)
    assert np.isnan(values["locPctJitter"])


def _table(scale: float, n: int = 20) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(0)
    base = rng.uniform(1, 2, size=(n, len(FEATURES)))
    uci = pd.DataFrame(base, columns=FEATURES)
    external = pd.DataFrame(base.copy(), columns=FEATURES)
    external["locPctJitter"] *= scale
    external["label"] = np.zeros(n, dtype=int)
    return uci, np.zeros(n, dtype=int), external


def test_unit_check_flags_a_measure_on_a_different_scale() -> None:
    uci, y, external = _table(scale=10.0)
    check = uci_transfer.unit_check(uci, y, external)
    assert check["locPctJitter"]["mismatch"]
    assert check["locPctJitter"]["ratio"] == pytest.approx(10, rel=0.01)
    assert not any(row["mismatch"] for name, row in check.items() if name != "locPctJitter")


def test_unit_check_passes_matching_measures() -> None:
    uci, y, external = _table(scale=1.0)
    assert not any(row["mismatch"] for row in uci_transfer.unit_check(uci, y, external).values())
