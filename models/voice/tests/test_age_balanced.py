"""Tests for the age-balanced voice tests. Synthetic data only."""

from __future__ import annotations

import age_balanced as ab
import metrics
import numpy as np
import pandas as pd
import pipeline
import pytest

N = 100
NAMES = [f"voice_{i}" for i in range(8)]


def _people(pd_voice_effect: float, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """PD people are older, voice changes with age, and PD adds ``pd_voice_effect``."""
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (N // 2))
    age = np.where(y == 1, rng.normal(68, 7, N), rng.normal(45, 12, N))
    voice = 0.08 * age[:, None] + rng.normal(size=(N, len(NAMES)))
    voice[:, :2] += pd_voice_effect * y[:, None]
    return pd.DataFrame(voice, columns=NAMES), age, y


def _auc(voice: pd.DataFrame, age: np.ndarray, y: np.ndarray) -> float:
    groups = np.array([f"synthetic:{i}" for i in range(len(y))])
    return metrics.auc(y, ab.nested_cv_adjusted(voice, age, y, groups, seed=1, k_grid=(2, 4)))


def test_adjuster_learns_normal_ageing_from_healthy_people_only() -> None:
    age = np.array([30.0, 40, 50, 60, 70, 35, 45, 55, 65, 75])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    voice = 2.0 * age + 5.0 + 50.0 * y  # PD adds 50 on top of ageing
    X = np.column_stack([voice, age])
    adjuster = ab.AgeAdjuster().fit(X, y)
    assert adjuster.slope_[0] == pytest.approx(2.0)  # the PD offset did not bend the slope
    out = adjuster.transform(X)
    assert out.shape == (10, 1)  # age column dropped
    assert np.allclose(out[y == 0], 0.0) and np.allclose(out[y == 1], 50.0)


def test_a_model_that_only_heard_age_falls_to_chance_once_age_is_removed() -> None:
    """The failing case: voice here carries age and nothing else."""
    voice, age, y = _people(pd_voice_effect=0.0)
    groups = np.array([f"synthetic:{i}" for i in range(N)])
    raw = pipeline.nested_cv_predict(voice, y, groups, seed=1, k_grid=(2, 4)).proba
    assert metrics.auc(y, raw) > 0.75  # the plain model happily learns age
    assert _auc(voice, age, y) < 0.65  # the adjusted model cannot


def test_a_real_voice_effect_survives_the_adjustment() -> None:
    voice, age, y = _people(pd_voice_effect=1.5)
    assert _auc(voice, age, y) > 0.75


def test_blank_age_is_refused_not_imputed() -> None:
    voice, age, y = _people(pd_voice_effect=1.0)
    age[3] = np.nan
    with pytest.raises(ValueError, match="age is blank"):
        _auc(voice, age, y)


def test_more_than_one_recording_per_person_is_refused() -> None:
    table = pd.DataFrame({"person": ["a", "a"], "label": [0, 0], "age": [50.0, 50.0]})
    with pytest.raises(ValueError, match="one recording per person"):
        ab.run(table, n_boot=10)
