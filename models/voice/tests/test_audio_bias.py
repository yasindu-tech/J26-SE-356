"""Tests for the age and sex bias check of the raw-audio model. Synthetic data only."""

from __future__ import annotations

import audio_bias
import numpy as np
import pandas as pd

N = 80


def _table(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "person": [f"synthetic:{i}" for i in range(N)],
            "label": np.array([0, 1] * (N // 2)),
            "age": rng.uniform(20, 85, N),
            "sex": np.where(np.arange(N) % 4 < 2, "F", "M"),
        }
    )


def test_score_that_follows_age_in_healthy_people_is_flagged() -> None:
    table = _table()
    # A model that learned age: healthy scores rise with age.
    proba = np.clip(table.age.to_numpy() / 100 + 0.3 * table.label.to_numpy(), 0, 1)
    report = audio_bias.bias_report(table, proba, n_boot=200, seed=1)
    assert report["age_driven"] is True
    assert report["age_vs_score_within_class"]["healthy"]["lower"] > 0


def test_score_unrelated_to_age_is_not_flagged() -> None:
    table = _table()
    rng = np.random.default_rng(3)
    proba = np.clip(0.3 + 0.4 * table.label.to_numpy() + 0.1 * rng.normal(size=N), 0, 1)
    report = audio_bias.bias_report(table, proba, n_boot=200, seed=1)
    assert report["age_driven"] is False
    assert report["age_and_sex_are_model_inputs"] is False
    assert report["sex"]["note"]


def test_false_referrals_are_counted_per_age_band() -> None:
    age = np.array([30.0, 45.0, 60.0, 70.0, 80.0])
    proba = np.array([0.1, 0.6, 0.7, 0.2, 0.5])
    bands = audio_bias.false_referrals_by_age(age, proba)
    assert bands == {
        "0-50": {"n_healthy": 2, "referred": 1},
        "50-65": {"n_healthy": 1, "referred": 1},
        "65-120": {"n_healthy": 2, "referred": 1},
    }


def test_report_states_that_other_voice_conditions_were_not_checked() -> None:
    table = _table()
    proba = np.full(N, 0.5)
    proba[::3] = 0.4
    report = audio_bias.bias_report(table, proba, n_boot=50, seed=1)
    assert "asthma" in report["other_voice_conditions"]
    assert report["status"] == "exploratory"
