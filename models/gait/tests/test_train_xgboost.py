"""Tests for the training script (GAIT-19)."""

from __future__ import annotations

import json
from pathlib import Path

import gait_metrics as gm
import make_folds as mf
import numpy as np
import pandas as pd
import pytest
import train_xgboost as tx
from feature_contract import FeatureContractError
from gait_extract import FEATURE_NAMES

FAST = {**tx.XGB_PARAMS, "n_estimators": 25}


def synthetic_table(n_subjects: int = 60, signal: float = 1.5, seed: int = 0) -> pd.DataFrame:
    """Subjects in three cohorts. Class shifts three of the eight inputs by ``signal``."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_subjects):
        cohort = ("A", "B", "C")[i % 3]
        base = i % 3
        for w in range(int(rng.integers(4, 10))):
            severity = base if rng.random() < 0.7 else int(rng.integers(0, 3))
            x = rng.normal(size=len(FEATURE_NAMES))
            x[:3] += signal * severity
            rows.append((f"walk-{i:03d}-{w}", cohort, f"s{i:03d}", severity, *x))
    cols = ["walk", "cohort", "subject_key", "severity", *FEATURE_NAMES]
    return pd.DataFrame(rows, columns=cols)


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return synthetic_table()


@pytest.fixture(scope="module")
def assignments(table: pd.DataFrame) -> pd.DataFrame:
    return mf.make_assignments(table, 5, (0, 1))


# --------------------------------------------------------------------------- inputs
def test_the_synthetic_folds_pass_the_validity_gate(
    table: pd.DataFrame, assignments: pd.DataFrame
) -> None:
    assert mf.fold_problems(table, assignments) == []


def test_read_training_table_keeps_row_order_and_checks_values(tmp_path: Path) -> None:
    t = synthetic_table(n_subjects=12)
    path = tmp_path / "t.csv"
    t.assign(extra=1.0).to_csv(path, index=False)
    back = tx.read_training_table(path)
    assert list(back.columns) == list(tx.TABLE_COLUMNS)
    assert back["walk"].tolist() == t["walk"].tolist()
    bad = t.copy()
    bad.loc[3, FEATURE_NAMES[0]] = np.nan
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing or infinite"):
        tx.read_training_table(path)


# --------------------------------------------------------------------------- the ladder
def test_majority_and_cohort_only_are_priors(table: pd.DataFrame) -> None:
    y = table["severity"].to_numpy(int)
    w = np.ones(len(y))
    train, test = table.iloc[:200], table.iloc[200:230]
    proba, _ = tx.fit_majority(train, test, y[:200], w[:200], 0, FAST)
    assert proba.shape == (30, 3)
    assert proba[0] == pytest.approx(np.bincount(y[:200], minlength=3) / 200)
    proba, _ = tx.fit_cohort_only(train, test, y[:200], w[:200], 0, FAST)
    a = train["cohort"].to_numpy() == "A"
    expected = np.bincount(y[:200][a], minlength=3) / a.sum()
    first_a = np.flatnonzero(test["cohort"].to_numpy() == "A")[0]
    assert proba[first_a] == pytest.approx(expected)


def test_cohort_only_uses_the_overall_prior_for_an_unseen_cohort(table: pd.DataFrame) -> None:
    train = table[table["cohort"] != "C"]
    test = table[table["cohort"] == "C"].iloc[:5]
    y = train["severity"].to_numpy(int)
    proba, _ = tx.fit_cohort_only(train, test, y, np.ones(len(y)), 0, FAST)
    assert proba[0] == pytest.approx(np.bincount(y, minlength=3) / len(y))


def test_best_feature_is_the_one_that_follows_the_class(table: pd.DataFrame) -> None:
    x = table[list(FEATURE_NAMES)].to_numpy(float)
    y = table["severity"].to_numpy(int)
    assert FEATURE_NAMES[tx.best_feature_index(x, y)] in FEATURE_NAMES[:3]


def test_full_places_columns_in_class_order() -> None:
    out = tx._full(np.array([[0.2, 0.8]]), np.array([0, 2]))
    assert out.tolist() == [[0.2, 0.0, 0.8]]


# --------------------------------------------------------------------------- cross-validation
@pytest.fixture(scope="module")
def result(table: pd.DataFrame, assignments: pd.DataFrame) -> tx.CVResult:
    return tx.cross_validate(table, assignments, params=FAST)


def test_every_walk_is_predicted_once_per_seed_by_every_model(
    table: pd.DataFrame, result: tx.CVResult
) -> None:
    assert set(result.proba) == set(tx.MODELS)
    for probas in result.proba.values():
        assert len(probas) == 2
        for p in probas:
            assert p.shape == (len(table), 3)
            assert not np.isnan(p).any()
            assert p.sum(axis=1) == pytest.approx(1.0)


def test_xgboost_learns_the_signal_and_the_control_does_not(
    table: pd.DataFrame, result: tx.CVResult
) -> None:
    y = table["severity"].to_numpy(int)
    real = np.mean([gm.balanced_accuracy(y, p.argmax(1)) for p in result.proba["xgboost"]])
    shuffled = np.mean([gm.balanced_accuracy(y, p.argmax(1)) for p in result.proba[tx.CONTROL]])
    assert real > 0.6
    assert abs(shuffled - 1 / 3) < 0.08
    assert real > gm.balanced_accuracy(y, result.proba["majority"][0].argmax(1)) + 0.15


def test_the_ladder_orders_as_it_should(table: pd.DataFrame, result: tx.CVResult) -> None:
    y = table["severity"].to_numpy(int)
    auc = {m: gm.macro_auc(y, result.proba[m][0]) for m in tx.LADDER}
    # The prior differs a little from fold to fold, so the pooled AUC is near 0.5, not exactly.
    assert auc["majority"] == pytest.approx(0.5, abs=0.04)
    assert auc["logistic"] > 0.7
    assert auc["best_feature"] > 0.6
    assert result.best_features and set(result.best_features) <= set(FEATURE_NAMES)


def test_fold_scores_are_recorded_for_every_test_fold(result: tx.CVResult) -> None:
    for scores in result.fold_bal_acc.values():
        assert len(scores) == 2 * 5
        assert all(0.0 <= s <= 1.0 for s in scores)


def test_results_are_reproducible(table: pd.DataFrame, assignments: pd.DataFrame) -> None:
    a = tx.cross_validate(table, assignments, ["xgboost"], seeds=[0], params=FAST)
    b = tx.cross_validate(table, assignments, ["xgboost"], seeds=[0], params=FAST)
    assert np.array_equal(a.proba["xgboost"][0], b.proba["xgboost"][0])


def test_unknown_model_is_an_error(table: pd.DataFrame, assignments: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="unknown models"):
        tx.cross_validate(table, assignments, ["forest"])


def test_weighted_training_runs(table: pd.DataFrame, assignments: pd.DataFrame) -> None:
    out = tx.cross_validate(table, assignments, ["xgboost", "logistic"], [0], True, FAST)
    assert not np.isnan(out.proba["xgboost"][0]).any()


# --------------------------------------------------------------------------- the safeguards
def test_a_subject_in_both_train_and_test_is_caught() -> None:
    t = synthetic_table(n_subjects=9)
    with pytest.raises(tx.LeakageError, match="both"):
        tx.assert_no_leakage(t.iloc[:30], t.iloc[20:40])
    tx.assert_no_leakage(t[t["subject_key"] < "s004"], t[t["subject_key"] >= "s004"])


def test_a_subject_with_two_folds_is_refused_before_any_fit(
    table: pd.DataFrame, assignments: pd.DataFrame
) -> None:
    bad = pd.concat([assignments, assignments.iloc[[0]].assign(fold=4)], ignore_index=True)
    with pytest.raises(ValueError, match="more than one fold"):
        tx.cross_validate(table, bad, ["majority"], params=FAST)


def test_the_feature_contract_runs_before_a_fit(
    table: pd.DataFrame, assignments: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def refuse(names, label_name=None):  # type: ignore[no-untyped-def]
        calls.append(list(names))
        raise FeatureContractError([])

    monkeypatch.setattr(tx, "check_feature_contract", refuse)
    with pytest.raises(FeatureContractError):
        tx.cross_validate(table, assignments, ["xgboost"], seeds=[0], params=FAST)
    assert calls and calls[0] == list(FEATURE_NAMES)


def test_the_label_cannot_become_an_input(table: pd.DataFrame, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(tx, "FEATURE_NAMES", (*FEATURE_NAMES[:-1], "severity"))
    with pytest.raises(FeatureContractError):
        tx._inputs(table)


def test_permutation_control_passes_at_chance_and_fails_above() -> None:
    def scores(bal: float, auc: float) -> pd.DataFrame:
        return pd.DataFrame([{"model": tx.CONTROL, "bal_acc": bal, "auc": auc}])

    assert tx.permutation_problems(scores(0.34, 0.51)) == []
    assert len(tx.permutation_problems(scores(0.45, 0.50))) == 1
    assert len(tx.permutation_problems(scores(0.45, 0.62))) == 2


def test_a_table_with_no_signal_scores_at_chance() -> None:
    """With no link between inputs and class, the model and the control both sit at chance."""
    t = synthetic_table(n_subjects=60, signal=0.0, seed=4)
    a = mf.make_assignments(t, 5, (0,))
    out = tx.cross_validate(t, a, ["xgboost", tx.CONTROL], params=FAST)
    scores = tx.summarize(t, out, n_boot=20).set_index("model")
    assert tx.permutation_problems(scores.reset_index()) == []
    assert abs(scores.loc["xgboost", "auc"] - 0.5) < 0.12


# --------------------------------------------------------------------------- scores and files
def test_summarize_reports_cis_and_fold_spread(table: pd.DataFrame, result: tx.CVResult) -> None:
    scores = tx.summarize(table, result, n_boot=40)
    assert list(scores["model"]) == list(tx.MODELS)
    row = scores.set_index("model").loc["xgboost"]
    for m in gm.METRICS:
        assert row[f"{m}_lo"] <= row[m] <= row[f"{m}_hi"]
    assert row["fold_min"] <= row["fold_mean"] <= row["fold_max"]
    assert row["fold_sd"] > 0


def test_oof_frame_has_one_row_per_walk_seed_and_model(
    table: pd.DataFrame, result: tx.CVResult
) -> None:
    frame = tx.oof_frame(table, result)
    assert len(frame) == len(table) * 2 * len(tx.MODELS)
    assert {
        "model",
        "seed",
        "walk",
        "cohort",
        "subject_key",
        "severity",
        "p0",
        "p1",
        "p2",
        "pred",
    } <= set(frame.columns)
    one = frame[(frame["model"] == "xgboost") & (frame["seed"] == 0)]
    assert one["walk"].is_unique


# --------------------------------------------------------------------------- SHAP and the final model
def test_shap_values_add_up_and_follow_the_signal(
    table: pd.DataFrame, assignments: pd.DataFrame
) -> None:
    ref = tx.cross_validate(table, assignments, ["xgboost"], seeds=[0], params=FAST)
    values, additivity, drift = tx.explain_folds(
        table, assignments, 0, params=FAST, reference=ref.proba["xgboost"][0]
    )
    assert values.shape == (len(table), len(FEATURE_NAMES), 3)
    assert additivity < 1e-3
    assert drift == 0.0
    ranked = tx.shap_mean_abs(values)
    assert set(ranked["feature"][:3]) == set(FEATURE_NAMES[:3])
    assert ranked["overall"].is_monotonic_decreasing


def test_shap_array_accepts_both_layouts() -> None:
    arr = np.random.default_rng(0).normal(size=(5, 4, 3))
    assert tx.shap_array(arr).shape == (5, 4, 3)
    as_list = [arr[:, :, c] for c in range(3)]
    assert np.array_equal(tx.shap_array(as_list), arr)


# --------------------------------------------------------------------------- the command line
def test_cli_missing_inputs_exit_2(tmp_path: Path) -> None:
    assert tx.main(["--table", str(tmp_path / "no.csv"), "--folds", str(tmp_path / "no2.csv")]) == 2
    t = tmp_path / "t.csv"
    synthetic_table(12).to_csv(t, index=False)
    assert tx.main(["--table", str(t), "--folds", str(tmp_path / "no2.csv")]) == 2


def test_cli_runs_end_to_end_and_writes_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tx, "XGB_PARAMS", FAST)
    t = synthetic_table(60)
    table_path, folds_path, out = tmp_path / "t.csv", tmp_path / "f.csv", tmp_path / "out"
    t.to_csv(table_path, index=False)
    mf.make_assignments(t, 5, (0, 1)).to_csv(folds_path, index=False)
    code = tx.main(
        [
            "--table",
            str(table_path),
            "--folds",
            str(folds_path),
            "--out-dir",
            str(out),
            "--bootstrap",
            "30",
        ]
    )
    text = capsys.readouterr().out
    assert code == 0
    for name in (
        "oof_predictions.csv",
        "metrics.csv",
        "metrics.json",
        "shap_mean_abs.csv",
        "bundle/model.json",
        "bundle/bundle.json",
    ):
        assert (out / name).is_file()
    assert "Label-permutation control" in text and "at chance" in text
    assert "confusion matrix" in text
    info = json.loads((out / "bundle" / "bundle.json").read_text())
    assert info["features"] == list(FEATURE_NAMES)
    assert "never a result" in info["note"]
    assert f"Model bundle {info['model_version']}" in text and "reload identical" in text
    reference = info["explanation_reference"]
    assert set(reference["percentiles"]) == set(FEATURE_NAMES) == set(reference["steady"])
    assert len(reference["strength_cuts"]) == 2


def test_cli_rejects_folds_that_do_not_match_the_table(tmp_path: Path) -> None:
    t = synthetic_table(60)
    other = synthetic_table(45, seed=3)
    table_path, folds_path = tmp_path / "t.csv", tmp_path / "f.csv"
    t.to_csv(table_path, index=False)
    mf.make_assignments(other, 5, (0,)).to_csv(folds_path, index=False)
    with pytest.raises(ValueError, match="different subjects"):
        tx.main(["--table", str(table_path), "--folds", str(folds_path)])
