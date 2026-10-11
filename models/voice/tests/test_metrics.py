"""Tests for models/voice/src/metrics.py."""

from __future__ import annotations

import metrics
import numpy as np
import pytest


def three_rows_each(person_labels: list[int], row_scores: list[list[float]]):
    """Row-level (y, scores, groups) for people with 3 rows each."""
    y = np.repeat(person_labels, 3)
    scores = np.concatenate(row_scores)
    groups = np.repeat(np.arange(len(person_labels)), 3)
    return y, scores, groups


def test_ppv_matches_hand_calculation() -> None:
    # 0.8*0.01 / (0.8*0.01 + 0.1*0.99) = 0.008 / 0.107 = 0.0748
    assert metrics.ppv_at_prevalence(0.8, 0.9, 0.01) == pytest.approx(0.0748, abs=1e-4)


def test_ppv_rejects_research_prevalence_mistakes() -> None:
    with pytest.raises(ValueError, match="prevalence"):
        metrics.ppv_at_prevalence(0.8, 0.9, 1.0)
    with pytest.raises(ValueError, match="specificity"):
        metrics.ppv_at_prevalence(0.8, 90, 0.01)  # percent instead of fraction


def test_perfect_classifier_gives_ci_1_1() -> None:
    rng = np.random.default_rng(0)
    labels = [1] * 30 + [0] * 10
    rows = [[0.9, 0.8, 0.95] if c else [0.1, 0.2, 0.05] for c in labels]
    y, s, g = three_rows_each(labels, rows)
    for metric in (metrics.auc, metrics.balanced_accuracy):
        assert metrics.bootstrap_ci(y, s, g, metric, n_boot=200, seed=int(rng.integers(99))) == (
            1.0,
            1.0,
            1.0,
        )


def test_bootstrap_resamples_people_not_rows() -> None:
    """Rows of one person overlap the other class, but each person's mean separates
    perfectly. A person-level bootstrap gives AUC CI (1, 1); a row-level one would not."""
    labels = [1] * 20 + [0] * 20
    rows = [[0.9, 0.9, 0.3] if c else [0.1, 0.1, 0.7] for c in labels]
    y, s, g = three_rows_each(labels, rows)
    assert metrics.auc(y, s) < 1.0  # row level is imperfect
    assert metrics.bootstrap_ci(y, s, g, n_boot=200) == (1.0, 1.0, 1.0)


def test_person_score_is_mean_of_rows() -> None:
    y, s, g = three_rows_each([1, 0], [[0.6, 0.9, 0.3], [0.1, 0.2, 0.3]])
    y_p, s_p = metrics.person_scores(y, s, g)
    assert y_p.tolist() == [1, 0]
    assert s_p == pytest.approx([0.6, 0.2])


def test_label_changing_within_person_fails() -> None:
    y, s, g = three_rows_each([1, 0], [[0.6, 0.9, 0.3], [0.1, 0.2, 0.3]])
    y[0] = 0
    with pytest.raises(ValueError, match="label differs"):
        metrics.person_scores(y, s, g)


def test_bootstrap_is_reproducible_and_contains_estimate() -> None:
    rng = np.random.default_rng(1)
    labels = list(rng.integers(0, 2, 60))
    rows = [list(rng.random(3) * 0.6 + 0.3 * c) for c in labels]
    y, s, g = three_rows_each(labels, rows)
    a = metrics.bootstrap_ci(y, s, g, n_boot=300)
    assert a == metrics.bootstrap_ci(y, s, g, n_boot=300)
    assert a.lower <= a.value <= a.upper


def test_paired_difference_of_identical_scores_is_zero() -> None:
    rng = np.random.default_rng(2)
    labels = list(rng.integers(0, 2, 60))
    rows = [list(rng.random(3)) for _ in labels]
    y, s, g = three_rows_each(labels, rows)
    assert metrics.paired_bootstrap_difference(y, s, s, g, n_boot=200) == (0.0, 0.0, 0.0)


def test_paired_difference_detects_a_better_pipeline() -> None:
    labels = [1] * 30 + [0] * 30
    good = [[0.8] * 3 if c else [0.2] * 3 for c in labels]
    rng = np.random.default_rng(3)
    noise = [list(rng.random(3)) for _ in labels]
    y, s_good, g = three_rows_each(labels, good)
    _, s_noise, _ = three_rows_each(labels, noise)
    diff = metrics.paired_bootstrap_difference(y, s_good, s_noise, g, n_boot=300)
    assert diff.lower > 0


def test_sensitivity_specificity() -> None:
    sens, spec = metrics.sensitivity_specificity(
        np.array([1, 1, 0, 0]), np.array([0.9, 0.1, 0.2, 0.8])
    )
    assert (sens, spec) == (0.5, 0.5)


def test_ppv_metric_ignores_the_research_set_prevalence() -> None:
    # sens 0.75, spec 0.75 on a 4 PD / 4 healthy sample. PPV must use the 1% screening
    # prevalence: 0.75*0.01 / (0.75*0.01 + 0.25*0.99) = 0.0294, not the sample's 0.75.
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3, 0.4, 0.9])
    ppv = metrics.ppv_metric(0.01)(y, scores)
    assert ppv == pytest.approx(0.0294, abs=1e-4)
    assert ppv != pytest.approx(0.75, abs=0.1)


def test_ppv_metric_bootstraps_at_person_level() -> None:
    labels = [1] * 20 + [0] * 20
    rows = [[0.9, 0.8, 0.7]] * 18 + [[0.1, 0.2, 0.3]] * 2 + [[0.1, 0.2, 0.2]] * 18 + [[0.9] * 3] * 2
    y, s, g = three_rows_each(labels, rows)
    est = metrics.bootstrap_ci(y, s, g, metrics.ppv_metric(0.05), n_boot=200)
    # sens 0.9, spec 0.9 at 5% prevalence: 0.045 / (0.045 + 0.095) = 0.3214
    assert est.value == pytest.approx(0.3214, abs=1e-4)
    # A resample with no false positives has PPV 1, so the upper bound may reach 1.
    assert est.lower < est.value < est.upper <= 1
