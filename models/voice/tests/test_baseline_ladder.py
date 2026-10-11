"""Tests for models/voice/src/baseline_ladder.py. Synthetic data only."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import baseline_ladder
import metrics
import numpy as np
import pandas as pd
import pipeline
import pytest
import splitter

FAST = dict(score="anova_f", n_boot=50, k_grid=(5, 10), c_grid=(0.1,), leaves_grid=(4,))
N_PEOPLE = 80


def ladder_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """80 people x 3 rows x 30 features; f0 carries a real signal, the rest is noise."""
    rng = np.random.default_rng(42)
    groups = np.repeat(np.arange(7000, 7000 + N_PEOPLE), 3)
    y = np.repeat((rng.random(N_PEOPLE) < 0.75).astype(int), 3)
    gender = np.repeat(rng.integers(0, 2, N_PEOPLE), 3)
    X = pd.DataFrame(rng.normal(size=(len(groups), 30)), columns=[f"f{i}" for i in range(30)])
    X["f0"] += y * 1.5
    return X, y, groups, gender


@pytest.fixture(scope="module")
def ladder() -> dict:
    """One small ladder run, shared by the tests below."""
    return baseline_ladder.run_ladder(*ladder_data(), **FAST)


def test_all_rungs_with_cis_and_screening_ppv(ladder: dict) -> None:
    assert tuple(ladder["rungs"]) == baseline_ladder.RUNGS
    for rung in ladder["rungs"].values():
        assert set(rung["ppv"]) == {"1%", "2%", "5%"}
        for est in (rung["auc"], rung["balanced_accuracy"], *rung["ppv"].values()):
            assert est["lower"] <= est["value"] <= est["upper"]
    baseline_ladder.validate_ladder(ladder)  # passes on a complete result


def test_ppv_is_computed_at_screening_prevalence(ladder: dict) -> None:
    """Each PPV must follow from the rung's own sens and spec at 1/2/5%, never at 75%."""
    for rung in ladder["rungs"].values():
        sens, spec = rung["sensitivity"]["value"], rung["specificity"]["value"]
        for key, prevalence in (("1%", 0.01), ("2%", 0.02), ("5%", 0.05)):
            expected = metrics.ppv_at_prevalence(sens, spec, prevalence)
            assert rung["ppv"][key]["value"] == pytest.approx(expected, abs=1e-3)
        ppv = [rung["ppv"][k]["value"] for k in ("1%", "2%", "5%")]
        assert ppv == sorted(ppv)


def test_every_pair_of_rungs_is_compared(ladder: dict) -> None:
    assert len(ladder["auc_differences"]) == 6
    diff = ladder["auc_differences"]["l1_logistic_all_minus_sex_only"]
    rungs = ladder["rungs"]
    expected = rungs["l1_logistic_all"]["auc"]["value"] - rungs["sex_only"]["auc"]["value"]
    assert diff["value"] == pytest.approx(expected, abs=2e-4)
    assert diff["beats"] == (diff["lower"] > 0)


def test_signal_feature_is_found_and_beats_sex(ladder: dict) -> None:
    """Gender is random noise here, f0 is real: the ladder must say so."""
    assert ladder["rungs"]["best_single_feature"]["chosen_feature_per_fold"] == ["f0"] * 5
    assert ladder["auc_differences"]["best_single_feature_minus_sex_only"]["beats"]


def test_best_feature_is_chosen_on_training_rows_only() -> None:
    """Red test: a feature that only separates the TEST people must never be picked.

    f1 is pure noise on training rows and a perfect separator on test rows. If
    selection saw the test fold it would pick f1; in-fold selection picks f0.
    """
    X, y, groups, _ = ladder_data()
    folds = splitter.person_folds(groups, y)
    train, test = folds[0]
    rigged = X.copy()
    rigged.loc[test, "f1"] = y[test] * 100.0
    _, chosen = baseline_ladder.best_single_feature_cv(rigged, y, [(train, test)], seed=42)
    assert chosen == ["f0"]
    # Control: scored on the test people, f1 wins. This is the leak the rung avoids.
    assert X.columns[baseline_ladder.best_feature(rigged.to_numpy()[test], y[test])] == "f1"


def test_sex_rung_uses_gender_only() -> None:
    X, y, groups, gender = ladder_data()
    folds = splitter.person_folds(groups, y)
    proba = baseline_ladder.single_column_cv(gender, y, folds, seed=42)
    for _, test in folds:
        assert len(np.unique(proba[test])) <= 2  # one score per gender code


def test_rungs_share_the_audit_folds() -> None:
    """Paired differences are only valid if every rung saw the same test people."""
    X, y, groups, _ = ladder_data()
    ours = splitter.person_folds(groups, y, pipeline.N_OUTER, pipeline.SEED)
    theirs = pipeline._folds(y, groups, "subject", pipeline.N_OUTER, pipeline.SEED)
    for (_, a), (_, b) in zip(ours, theirs, strict=True):
        assert np.array_equal(a, b)


def test_banned_column_is_rejected_by_single_feature_rung() -> None:
    X, y, groups, _ = ladder_data()
    X["UPDRS_III"] = y  # a perfect, forbidden feature
    folds = splitter.person_folds(groups, y)
    with pytest.raises(pipeline.contract.FeatureContractError, match="UPDRS"):
        baseline_ladder.best_single_feature_cv(X, y, folds, seed=42)


def test_lightgbm_rejects_banned_column() -> None:
    X, y, groups, _ = ladder_data()
    X["hoehn_yahr"] = y
    with pytest.raises(pipeline.contract.FeatureContractError, match="hoehn"):
        pipeline.nested_cv_predict(X, y, groups, model="lightgbm")


def test_unknown_model_fails() -> None:
    X, y, groups, _ = ladder_data()
    with pytest.raises(ValueError, match="model"):
        pipeline.nested_cv_predict(X, y, groups, model="svm")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("breakage", "match"),
    [
        (lambda r: r["rungs"].pop("sex_only"), "missing rungs"),
        (lambda r: r["rungs"]["lightgbm_all"]["ppv"].pop("1%"), "PPV"),
        (lambda r: r["rungs"]["l1_logistic_all"]["auc"].pop("lower"), "confidence interval"),
        (lambda r: r["auc_differences"].pop("lightgbm_all_minus_sex_only"), "paired"),
    ],
)
def test_incomplete_ladder_fails_validation(ladder: dict, breakage, match: str) -> None:
    broken = copy.deepcopy(ladder)
    breakage(broken)
    with pytest.raises(ValueError, match=match):
        baseline_ladder.validate_ladder(broken)


def test_output_holds_no_person_ids(ladder: dict) -> None:
    text = json.dumps(ladder)
    assert not any(str(pid) in text for pid in range(7000, 7000 + N_PEOPLE))


def test_chart_and_results_table_are_written(ladder: dict, tmp_path: Path) -> None:
    baseline_ladder.plot_ladder(ladder, tmp_path / "baseline_ladder.png")
    assert (tmp_path / "baseline_ladder.png").stat().st_size > 0
    table = baseline_ladder.results_markdown(ladder)
    assert "PPV at 1%" in table and "lightgbm_all_minus_sex_only" in table
