"""Tests for the scores and subject-level bootstrap (GAIT-19)."""

from __future__ import annotations

import gait_metrics as gm
import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


def random_problem(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    logits = rng.normal(size=(n, 3)) + 1.2 * np.eye(3)[y]
    proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return y, proba


# --------------------------------------------------------------------------- the scores
def test_weighted_auc_matches_scikit_learn_with_weights() -> None:
    y, proba = random_problem()
    w = np.random.default_rng(1).uniform(0.2, 3.0, len(y))
    for c in range(3):
        mine = gm.weighted_auc(y == c, proba[:, c], w)
        theirs = roc_auc_score(y == c, proba[:, c], sample_weight=w)
        assert mine == pytest.approx(theirs, abs=1e-12)


def test_weighted_auc_counts_ties_as_half() -> None:
    positive = np.array([1, 1, 0, 0], dtype=bool)
    score = np.array([0.5, 0.5, 0.5, 0.5])
    assert gm.weighted_auc(positive, score, np.ones(4)) == pytest.approx(0.5)
    tied = np.array([0.9, 0.5, 0.5, 0.1])
    assert gm.weighted_auc(positive, tied, np.ones(4)) == pytest.approx(
        roc_auc_score(positive, tied)
    )


def test_weighted_auc_is_nan_without_both_groups() -> None:
    assert np.isnan(gm.weighted_auc(np.ones(5, bool), np.arange(5.0), np.ones(5)))
    assert np.isnan(gm.weighted_auc(np.zeros(5, bool), np.arange(5.0), np.ones(5)))


def test_macro_auc_matches_scikit_learn() -> None:
    y, proba = random_problem()
    assert gm.macro_auc(y, proba) == pytest.approx(
        roc_auc_score(y, proba, multi_class="ovr", average="macro"), abs=1e-12
    )


def test_balanced_accuracy_matches_scikit_learn_and_weights() -> None:
    y, proba = random_problem()
    pred = proba.argmax(axis=1)
    assert gm.balanced_accuracy(y, pred) == pytest.approx(balanced_accuracy_score(y, pred))
    w = np.random.default_rng(2).uniform(0.5, 2.0, len(y))
    assert gm.balanced_accuracy(y, pred, w) == pytest.approx(
        balanced_accuracy_score(y, pred, sample_weight=w)
    )


def test_perfect_and_constant_predictions() -> None:
    y = np.array([0, 1, 2, 0, 1, 2])
    perfect = np.eye(3)[y]
    assert gm.balanced_accuracy(y, perfect.argmax(1)) == 1.0
    assert gm.macro_auc(y, perfect) == 1.0
    constant = np.tile([0.5, 0.3, 0.2], (6, 1))
    assert gm.macro_auc(y, constant) == pytest.approx(0.5)
    assert gm.balanced_accuracy(y, constant.argmax(1)) == pytest.approx(1 / 3)


def test_macro_f1_and_confusion() -> None:
    y = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 1, 1, 1, 2, 0])
    assert gm.macro_f1(y, pred) == pytest.approx(f1_score(y, pred, average="macro"))
    cm = gm.confusion(y, pred)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert cm.sum() == len(y)


def test_subject_weights_give_each_subject_the_same_total() -> None:
    subjects = np.array(["a"] * 6 + ["b"] * 2 + ["c"] * 1)
    w = gm.subject_weights(subjects)
    assert w.mean() == pytest.approx(1.0)
    totals = [w[subjects == s].sum() for s in "abc"]
    assert totals == pytest.approx([totals[0]] * 3)


def test_metric_values_average_the_seeds() -> None:
    y, p1 = random_problem(seed=0)
    _, p2 = random_problem(seed=5)
    w = np.ones(len(y))
    a = gm.metric_values(y, [p1], w)
    b = gm.metric_values(y, [p2], w)
    both = gm.metric_values(y, [p1, p2], w)
    assert both["bal_acc"] == pytest.approx((a["bal_acc"] + b["bal_acc"]) / 2)
    assert both["auc"] == pytest.approx((a["auc"] + b["auc"]) / 2)


# --------------------------------------------------------------------------- the bootstrap
def subject_problem(n_subjects: int = 40, walks: int = 10, seed: int = 0):
    """Walks grouped by subject; each subject has one class most of the time."""
    rng = np.random.default_rng(seed)
    subjects = np.repeat([f"s{i:03d}" for i in range(n_subjects)], walks)
    base = np.repeat(rng.integers(0, 3, n_subjects), walks)
    y = np.where(rng.random(len(base)) < 0.8, base, rng.integers(0, 3, len(base)))
    logits = rng.normal(size=(len(y), 3)) + 1.0 * np.eye(3)[y]
    proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return y, proba, subjects


def test_interval_contains_the_point_estimate_and_is_reproducible() -> None:
    y, proba, subjects = subject_problem()
    point = gm.metric_values(y, [proba], gm.subject_weights(subjects))
    a = gm.bootstrap_intervals(y, [proba], subjects, n_boot=200, seed=3)
    b = gm.bootstrap_intervals(y, [proba], subjects, n_boot=200, seed=3)
    assert a == b
    for m in gm.METRICS:
        lo, hi = a[m]
        assert lo <= point[m] <= hi


def test_resampling_subjects_gives_wider_intervals_than_resampling_walks() -> None:
    """A subject's walks share a class, so subject-level draws vary more than walk-level draws."""
    y, proba, subjects = subject_problem(n_subjects=30, walks=20, seed=1)
    by_subject = gm.bootstrap_intervals(y, [proba], subjects, n_boot=300, seed=0)
    by_walk = gm.bootstrap_intervals(y, [proba], np.arange(len(y)).astype(str), n_boot=300, seed=0)
    width = lambda d: d["bal_acc"][1] - d["bal_acc"][0]  # noqa: E731
    assert width(by_subject) > width(by_walk)


def test_more_subjects_give_a_narrower_interval() -> None:
    small = subject_problem(n_subjects=15, seed=2)
    large = subject_problem(n_subjects=120, seed=2)
    ci_small = gm.bootstrap_intervals(small[0], [small[1]], small[2], n_boot=300)["auc"]
    ci_large = gm.bootstrap_intervals(large[0], [large[1]], large[2], n_boot=300)["auc"]
    assert ci_large[1] - ci_large[0] < ci_small[1] - ci_small[0]


def test_bootstrap_rejects_bad_arguments() -> None:
    y, proba, subjects = subject_problem(n_subjects=10)
    with pytest.raises(ValueError, match="n_boot"):
        gm.bootstrap_intervals(y, [proba], subjects, n_boot=0)
    with pytest.raises(ValueError, match="level"):
        gm.bootstrap_intervals(y, [proba], subjects, n_boot=5, level=1.5)
