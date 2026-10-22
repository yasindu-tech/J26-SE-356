"""Tests for models/mri/src/explain_checks.py on synthetic people (no real data).

The synthetic cohort has a real group difference in two features (see test_train_mri), so a
faithful explanation must rank those two first. A random 'explanation' is the failing case.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("lightgbm")

import explain_checks as ec  # noqa: E402
from t1_features import feature_names  # noqa: E402
from test_train_mri import T1_COLS, cohort  # noqa: E402

# the two signal columns of the cohort first, then 22 real T1 names that carry only noise
COLS = T1_COLS[:2] + [c for c in feature_names() if c not in T1_COLS[:2]][:22]


@pytest.fixture(scope="module")
def folds():
    data = cohort(n=40, signal=2.5, cols=COLS)
    return data, list(ec.run_folds(data, n_splits=4, n_repeats=3, seed=0))


def random_attribution(member, rows):
    """A fake explanation that ignores the model: random importances."""
    return np.random.default_rng(len(rows)).random((len(rows), len(member.columns)))


def test_shap_explanation_passes_the_deletion_test(folds) -> None:
    data, f = folds
    d = ec.deletion_test(f, data.n)
    assert d["passed"] and d["ci_95"][0] > 0
    assert d["curve"]["1"]["mean_change_top"] > d["curve"]["1"]["mean_change_random"]


def test_random_explanation_fails_the_deletion_test(folds) -> None:
    data, f = folds
    d = ec.deletion_test(f, data.n, attribution=random_attribution)
    assert not d["passed"]


def test_shap_explanation_is_more_stable_than_a_random_one(folds) -> None:
    _, f = folds
    real = ec.stability_test(f)
    fake = ec.stability_test(f, attribution=random_attribution)
    assert real["mean_spearman"] > fake["mean_spearman"]
    assert real["mean_top5_overlap"] > fake["mean_top5_overlap"]
    assert 0 < real["copies_per_person"] <= 3


def test_every_person_is_checked_only_by_models_that_never_saw_them(folds) -> None:
    data, f = folds
    for fold, _, test, _ in f:
        assert len(test) == len(fold.test_idx)
        assert not set(fold.test_idx) & set(fold.train_idx)


def test_report_states_the_rules_and_no_cause(folds) -> None:
    data, f = folds
    r = {"set": "t1", "people": data.n, "folds": 4, "repeats": 3,
         "deletion": ec.deletion_test(f, data.n), "stability": ec.stability_test(f)}
    text = ec.markdown(r)
    assert "not a cause" in text and "Rule:" in text
