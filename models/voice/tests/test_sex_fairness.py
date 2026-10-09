"""Tests for models/voice/src/sex_fairness.py. Synthetic data only.

Red tests: the breakdown must DETECT a model that only works for one sex, and
must flag a score that is really just sex. A fairness check that cannot find
unfairness is not a check.
"""

from __future__ import annotations

import copy
import json

import metrics
import numpy as np
import pandas as pd
import pytest
import sex_fairness as sf

FAST = dict(score="anova_f", n_boot=100, k_grid=(5,), c_grid=(0.1,), leaves_grid=(4,))
N_PEOPLE = 200


def people(seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """200 people x 3 rows with balanced-ish classes in both sex codes."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(4000, 4000 + N_PEOPLE), 3)
    y = np.repeat((rng.random(N_PEOPLE) < 0.6).astype(int), 3)
    gender = np.repeat(rng.integers(0, 2, N_PEOPLE), 3)
    return y, groups, gender


def test_counts_cover_everyone_once() -> None:
    y, groups, gender = people()
    counts = sf.group_counts(y, groups, gender)
    assert sum(c["n_people"] for c in counts.values()) == N_PEOPLE
    assert sum(c["n_pd"] for c in counts.values()) == y[::3].sum()


def test_too_small_group_is_refused() -> None:
    y, groups, gender = people()
    gender = np.where(y == 0, 0, gender)  # sex code 1 now has no healthy people
    with pytest.raises(ValueError, match="at least"):
        sf.group_counts(y, groups, gender)


def test_unfair_model_is_detected() -> None:
    """Red test: scores that separate classes for sex code 0 only must show an AUC gap."""
    y, groups, gender = people()
    rng = np.random.default_rng(0)
    proba = np.where(gender == 0, 0.2 + 0.6 * y, 0.5) + rng.normal(0, 0.05, len(y))
    result = sf.breakdown(y, proba, groups, gender, n_boot=200)
    gap = result["gap_sex_code_1_minus_sex_code_0"]["auc"]
    assert gap["upper"] < 0 and gap["ci_excludes_zero"]
    assert "distinguishable from zero" in sf.fairness_note(result)


def test_fair_model_shows_no_gap() -> None:
    y, groups, gender = people()
    rng = np.random.default_rng(1)
    proba = 0.3 + 0.4 * y + rng.normal(0, 0.15, len(y))
    result = sf.breakdown(y, proba, groups, gender, n_boot=200)
    assert not result["gap_sex_code_1_minus_sex_code_0"]["auc"]["ci_excludes_zero"]
    assert not result["driven_by_sex"]


def test_note_names_a_threshold_gap_hidden_by_equal_auc() -> None:
    """Red test: same ranking in both groups, but sex code 1 scores shifted up.

    AUC is identical (a shift does not change ranks), yet at 0.5 far more healthy
    people of sex code 1 are flagged. The note must say so, not just report AUC.
    """
    y, groups, gender = people()
    rng = np.random.default_rng(4)
    proba = 0.25 + 0.3 * y + rng.normal(0, 0.05, len(y)) + 0.25 * gender
    result = sf.breakdown(y, proba, groups, gender, n_boot=200)
    gaps = result["gap_sex_code_1_minus_sex_code_0"]
    assert not gaps["auc"]["ci_excludes_zero"]
    assert gaps["specificity"]["upper"] < 0
    note = sf.fairness_note(result)
    assert "not distinguishable" in note and "specificity" in note


def test_score_that_is_just_sex_is_flagged() -> None:
    """Red test: if sex predicts class and the score is sex, within-sex AUC is chance."""
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(N_PEOPLE), 3)
    gender = np.repeat(rng.integers(0, 2, N_PEOPLE), 3)
    p_pd = np.where(gender == 1, 0.85, 0.25)  # strong sex-class link
    y = np.repeat((rng.random(N_PEOPLE) < p_pd[::3]).astype(int), 3)
    proba = gender * 0.6 + 0.2 + rng.normal(0, 0.01, len(y))  # score knows only sex
    result = sf.breakdown(y, proba, groups, gender, n_boot=200)
    assert result["overall_auc"]["lower"] > 0.5
    assert result["driven_by_sex"]
    assert "mostly reflects sex" in sf.fairness_note(result)


def test_gender_as_model_input_is_refused() -> None:
    y, groups, gender = people()
    X = pd.DataFrame({"f0": np.zeros(len(y)), "gender": gender})
    with pytest.raises(ValueError, match="D7"):
        sf.run_fairness(X, y, groups, gender, **FAST)


def test_group_difference_matches_point_values() -> None:
    y, groups, gender = people()
    proba = np.random.default_rng(3).random(len(y))
    diff = metrics.group_bootstrap_difference(y, proba, groups, gender == 1, n_boot=100)
    a = metrics.auc(*metrics.person_scores(y[gender == 0], proba[gender == 0], groups[gender == 0]))
    b = metrics.auc(*metrics.person_scores(y[gender == 1], proba[gender == 1], groups[gender == 1]))
    assert diff.value == pytest.approx(b - a)
    assert diff.lower <= diff.value <= diff.upper


@pytest.fixture(scope="module")
def fairness() -> dict:
    y, groups, gender = people()
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(len(y), 20)), columns=[f"f{i}" for i in range(20)])
    X["f0"] += y * 1.5
    return sf.run_fairness(X, y, groups, gender, **FAST)


def test_full_run_is_complete_and_labelled_exploratory(fairness: dict) -> None:
    sf.validate_fairness(fairness)
    assert fairness["status"] == "exploratory"
    assert fairness["gender_is_model_input"] is False
    assert "no age column" in fairness["age_band_analysis"]
    assert set(fairness["models"]) == {"l1_logistic", "lightgbm"}


def test_note_never_uses_diagnostic_wording(fairness: dict) -> None:
    for r in fairness["models"].values():
        note = r["sex_fairness_note"].lower()
        assert note.startswith("exploratory")
        for word in ("diagnos", "positive", "negative", "cleared", "healthy result"):
            assert word not in note


@pytest.mark.parametrize(
    ("breakage", "match"),
    [
        (lambda r: r.update(status="final"), "exploratory"),
        (lambda r: r["models"]["lightgbm"]["per_sex_code"].popitem(), "both sex codes"),
        (lambda r: r["models"]["l1_logistic"].pop("sex_fairness_note"), "note"),
        (
            lambda r: r["models"]["l1_logistic"]["per_sex_code"]["sex_code_0"]["auc"].pop("lower"),
            "confidence interval",
        ),
    ],
)
def test_incomplete_result_fails_validation(fairness: dict, breakage, match: str) -> None:
    broken = copy.deepcopy(fairness)
    breakage(broken)
    with pytest.raises(ValueError, match=match):
        sf.validate_fairness(broken)


def test_output_holds_no_person_ids(fairness: dict) -> None:
    text = json.dumps(fairness)
    assert not any(str(pid) in text for pid in range(4000, 4000 + N_PEOPLE))


def test_chart_and_table_are_written(fairness: dict, tmp_path) -> None:
    sf.plot_fairness(fairness, tmp_path / "sex_fairness.png")
    assert (tmp_path / "sex_fairness.png").stat().st_size > 0
    assert "Driven by sex" in sf.results_markdown(fairness)
