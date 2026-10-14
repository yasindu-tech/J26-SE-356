"""Tests for models/mri/src/metrics.py with hand-checkable numbers."""

from __future__ import annotations

import numpy as np
import pytest
from metrics import (
    MetricError,
    auc,
    balanced_accuracy,
    bootstrap_ci,
    paired_difference_ci,
    ppv_at_prevalence,
    sensitivity_specificity,
    summarise,
)


def test_balanced_accuracy_ignores_group_sizes() -> None:
    y = np.array([1] * 9 + [0])
    always_pd = np.ones(10)
    assert np.mean(always_pd == y) == 0.9  # raw accuracy would look good
    assert balanced_accuracy(y, always_pd) == pytest.approx(0.5)  # it is chance


def test_sensitivity_specificity_known_values() -> None:
    y = [1, 1, 1, 1, 0, 0, 0, 0]
    p = [0.9, 0.8, 0.7, 0.2, 0.1, 0.2, 0.6, 0.3]
    assert sensitivity_specificity(y, p) == (0.75, 0.75)


def test_auc_perfect_reversed_and_ties() -> None:
    y = [0, 0, 1, 1]
    assert auc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auc(y, [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert auc(y, [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_auc_matches_sklearn() -> None:
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.random(200) + 0.3 * y
    assert auc(y, p) == pytest.approx(sk.roc_auc_score(y, p))


def test_ppv_at_screening_prevalence() -> None:
    # 80% sensitive, 90% specific: at 1% prevalence most positives are false
    assert ppv_at_prevalence(0.8, 0.9, 0.01) == pytest.approx(0.008 / (0.008 + 0.099))
    assert ppv_at_prevalence(0.8, 0.9, 0.5) == pytest.approx(0.4 / 0.45)


def test_bootstrap_ci_contains_estimate_and_narrows_with_more_people() -> None:
    rng = np.random.default_rng(1)

    def data(n):
        y = np.tile([0, 1], n // 2)
        return y, rng.random(n) + 0.5 * y

    y, p = data(40)
    lo, hi = bootstrap_ci(y, p, auc, n_boot=500)
    assert lo <= auc(y, p) <= hi
    y2, p2 = data(400)
    lo2, hi2 = bootstrap_ci(y2, p2, auc, n_boot=500)
    assert hi2 - lo2 < hi - lo


def test_paired_difference_detects_a_better_model_and_not_an_equal_one() -> None:
    rng = np.random.default_rng(2)
    y = np.tile([0, 1], 100)
    good = y + rng.normal(0, 0.5, 200)
    weak = y + rng.normal(0, 3.0, 200)
    diff, lo, hi = paired_difference_ci(y, good, weak, auc, n_boot=500)
    assert diff > 0 and lo > 0  # interval excludes zero
    diff, lo, hi = paired_difference_ci(y, good, good, auc, n_boot=200)
    assert diff == 0 and lo == 0 and hi == 0


def test_summary_has_every_reported_number() -> None:
    y = np.tile([0, 1], 20)
    p = np.where(y == 1, 0.7, 0.3)
    s = summarise(y, p, n_boot=200)
    assert (s.n, s.n_pd, s.auc, s.balanced_accuracy) == (40, 20, 1.0, 1.0)
    assert set(s.ppv) == {0.01, 0.02, 0.05}


@pytest.mark.parametrize(
    ("y", "p", "match"),
    [
        ([1, 1, 1], [0.2, 0.4, 0.6], "both classes"),
        ([0, 1], [0.2], "lengths"),
        ([0, 2], [0.2, 0.4], "0 or 1"),
        ([0, 1], [0.2, float("nan")], "missing"),
    ],
)
def test_bad_inputs_fail(y, p, match) -> None:
    with pytest.raises(MetricError, match=match):
        auc(y, p)


def test_bad_prevalence_fails() -> None:
    with pytest.raises(MetricError):
        ppv_at_prevalence(0.8, 0.9, 0.0)
