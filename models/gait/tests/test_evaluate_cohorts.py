"""Tests for the cohort scores, leave-one-cohort-out and ablations (GAIT-20)."""

from __future__ import annotations

import json
from pathlib import Path

import evaluate_cohorts as ec
import gait_metrics as gm
import make_folds as mf
import numpy as np
import pandas as pd
import pytest
import train_xgboost as tx
from feature_contract import FeatureContractError
from gait_extract import FEATURE_NAMES

FAST = {**tx.XGB_PARAMS, "n_estimators": 25}
COHORTS = ("A", "B", "C", "D")


def cohort_table(
    n_subjects: int = 64, signal: float = 2.0, seed: int = 0, shift: float = 0.0
) -> pd.DataFrame:
    """Four cohorts. The class does not depend on the cohort. The class shifts the first three
    inputs by ``signal``; ``shift`` adds a cohort signature to a fifth input."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_subjects):
        cohort = COHORTS[i % 4]
        base = int(rng.integers(0, 3))
        for w in range(int(rng.integers(4, 10))):
            severity = base if rng.random() < 0.7 else int(rng.integers(0, 3))
            x = rng.normal(size=len(FEATURE_NAMES))
            x[:3] += signal * severity
            x[4] += shift * (i % 4)
            rows.append((f"walk-{i:03d}-{w}", cohort, f"s{i:03d}", severity, *x))
    cols = ["walk", "cohort", "subject_key", "severity", *FEATURE_NAMES]
    return pd.DataFrame(rows, columns=cols)


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return cohort_table()


@pytest.fixture(scope="module")
def assignments(table: pd.DataFrame) -> pd.DataFrame:
    return mf.make_assignments(table, 5, (0, 1))


@pytest.fixture(scope="module")
def cv(table: pd.DataFrame, assignments: pd.DataFrame) -> tx.CVResult:
    return tx.cross_validate(table, assignments, params=FAST)


@pytest.fixture(scope="module")
def loco(table: pd.DataFrame) -> ec.LocoResult:
    return ec.leave_one_cohort_out(table, params=FAST, n_shuffles=12)


def test_the_synthetic_folds_pass_the_validity_gate(
    table: pd.DataFrame, assignments: pd.DataFrame
) -> None:
    assert mf.fold_problems(table, assignments) == []


# --------------------------------------------------------------------------- the predictions file
def test_oof_probabilities_come_back_in_table_order_per_seed(
    table: pd.DataFrame, cv: tx.CVResult
) -> None:
    frame = tx.oof_frame(table, cv).sample(frac=1, random_state=0)  # any row order
    for m in ("xgboost", "logistic"):
        got = ec.oof_probabilities(frame, table, m)
        assert len(got) == 2
        for mine, theirs in zip(got, cv.proba[m], strict=True):
            assert np.array_equal(mine, theirs)


def test_a_predictions_file_that_does_not_match_the_table_is_refused(
    table: pd.DataFrame, cv: tx.CVResult
) -> None:
    frame = tx.oof_frame(table, cv)
    with pytest.raises(ValueError, match="no rows for model"):
        ec.oof_probabilities(frame, table, "forest")
    with pytest.raises(ValueError, match="exactly once"):
        ec.oof_probabilities(frame.drop(index=frame.index[0]), table, "majority")
    with pytest.raises(ValueError, match="exactly once"):
        ec.oof_probabilities(pd.concat([frame, frame.iloc[[0]]]), table, "majority")
    for column, value in (("severity", 9), ("cohort", "Z"), ("subject_key", "nobody")):
        bad = frame.copy()
        bad.loc[bad.index[5], column] = value
        with pytest.raises(ValueError, match=f"{column} differs"):
            ec.oof_probabilities(bad, table, bad.loc[bad.index[5], "model"])
    bad = frame.copy()
    bad.loc[(bad["model"] == "xgboost") & (bad["seed"] == 0), "p0"] += 0.5
    with pytest.raises(ValueError, match="not valid"):
        ec.oof_probabilities(bad, table, "xgboost")


def test_read_oof_checks_the_columns(tmp_path: Path) -> None:
    path = tmp_path / "o.csv"
    pd.DataFrame({"model": ["x"], "seed": [0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="lacks the columns"):
        ec.read_oof(path)


# --------------------------------------------------------------------------- leave one cohort out
def test_a_cohort_is_never_in_the_training_part_of_its_own_test(
    table: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = []
    real = tx.LEARNERS["xgboost"]

    def spy(train, test, y, w, seed, params):  # type: ignore[no-untyped-def]
        seen.append((set(train["cohort"]), set(test["cohort"]), len(train), len(test)))
        return real(train, test, y, w, seed, params)

    monkeypatch.setitem(tx.LEARNERS, "xgboost", spy)
    ec.leave_one_cohort_out(table, params=FAST, models=("xgboost",), n_shuffles=2)
    assert len(seen) == 4 * (1 + 2)  # four cohorts, the model and two shuffles each
    held_out = set()
    for train_c, test_c, n_train, n_test in seen:
        assert len(test_c) == 1 and len(train_c) == 3 and not (train_c & test_c)
        assert n_train + n_test == len(table)
        held_out |= test_c
    assert held_out == set(COHORTS)


def test_every_walk_is_predicted_once_by_a_model_that_never_saw_its_cohort(
    table: pd.DataFrame, loco: ec.LocoResult
) -> None:
    assert set(loco.proba) == set(ec.LOCO_MODELS)
    for arrays in loco.proba.values():
        assert len(arrays) == 1 and arrays[0].shape == (len(table), 3)
        assert not np.isnan(arrays[0]).any()
        assert arrays[0].sum(axis=1) == pytest.approx(1.0)
    assert len(loco.null) == 12
    assert loco.train_cohorts["A"] == ("B", "C", "D")


def test_leave_one_cohort_out_is_reproducible(table: pd.DataFrame, loco: ec.LocoResult) -> None:
    again = ec.leave_one_cohort_out(table, params=FAST, n_shuffles=12)
    assert np.array_equal(again.proba["xgboost"][0], loco.proba["xgboost"][0])
    assert np.array_equal(again.null[3], loco.null[3])


def test_a_shared_signal_carries_over_to_an_unseen_cohort(
    table: pd.DataFrame, loco: ec.LocoResult
) -> None:
    y = table["severity"].to_numpy(int)
    w = gm.subject_weights(table["subject_key"].to_numpy())
    real = gm.metric_values(y, loco.proba["xgboost"], w)["bal_acc"]
    null = [gm.metric_values(y, [p], w)["bal_acc"] for p in loco.null]
    assert real > 0.6
    assert abs(np.mean(null) - 1 / 3) < 0.08
    assert real > max(null)
    maj = gm.metric_values(y, loco.proba["majority"], w)["bal_acc"]
    assert real > maj + 0.2


def test_a_cohort_signature_scores_in_cross_validation_but_not_on_an_unseen_cohort() -> None:
    """Each cohort has its own dominant class and its own offset in one input. Subject-wise
    cross-validation learns "which cohort is this", so it scores well; an unseen cohort cannot be
    recognised, so the same model scores at or below chance."""
    rng = np.random.default_rng(3)
    dominant = {"A": 0, "B": 1, "C": 2, "D": 1}
    rows = []
    for i in range(64):
        cohort = COHORTS[i % 4]
        for w in range(6):
            severity = dominant[cohort] if rng.random() < 0.85 else int(rng.integers(0, 3))
            x = rng.normal(size=len(FEATURE_NAMES))
            x[0] += 3.0 * (i % 4)
            rows.append((f"w{i}-{w}", cohort, f"s{i:03d}", severity, *x))
    cols = ["walk", "cohort", "subject_key", "severity", *FEATURE_NAMES]
    t = pd.DataFrame(rows, columns=cols)
    y = t["severity"].to_numpy(int)
    folds = mf.make_assignments(t, 5, (0,))
    cv_out = tx.cross_validate(t, folds, ["xgboost"], params=FAST)
    out = ec.leave_one_cohort_out(t, params=FAST, models=("xgboost",), n_shuffles=2)
    assert gm.balanced_accuracy(y, cv_out.proba["xgboost"][0].argmax(axis=1)) > 0.6
    assert gm.balanced_accuracy(y, out.proba["xgboost"][0].argmax(axis=1)) < 0.4


def test_a_training_part_without_a_class_is_refused(table: pd.DataFrame) -> None:
    t = table.copy()
    t.loc[t["cohort"] == "D", "severity"] = 2
    t.loc[(t["cohort"] != "D") & (t["severity"] == 2), "severity"] = 1
    with pytest.raises(ValueError, match="lacks a class"):
        ec.leave_one_cohort_out(t, params=FAST, n_shuffles=1)


def test_leave_one_cohort_out_needs_two_cohorts_and_known_models(table: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="two cohorts"):
        ec.leave_one_cohort_out(table[table["cohort"] == "A"], params=FAST)
    with pytest.raises(ValueError, match="unknown models"):
        ec.leave_one_cohort_out(table, models=("forest",))
    with pytest.raises(ValueError, match="unknown models"):
        ec.leave_one_cohort_out(table, models=(tx.CONTROL,))


def test_loco_frame_has_one_row_per_walk_and_model(
    table: pd.DataFrame, loco: ec.LocoResult
) -> None:
    frame = ec.loco_frame(table, loco)
    assert len(frame) == len(table) * len(ec.LOCO_MODELS)
    assert set(frame["model"]) == {f"loco_{m}" for m in ec.LOCO_MODELS}
    assert {"walk", "cohort", "subject_key", "severity", "p0", "p1", "p2", "pred"} <= set(frame)


# --------------------------------------------------------------------------- the shuffled labels
def fabricated_loco(table: pd.DataFrame, n_null: int = 20, perfect: bool = True) -> ec.LocoResult:
    y = table["severity"].to_numpy(int)
    rng = np.random.default_rng(0)
    real = np.eye(3)[y] if perfect else rng.dirichlet(np.ones(3), len(y))
    null = [rng.dirichlet(np.ones(3), len(y)) for _ in range(n_null)]
    return ec.LocoResult(proba={"xgboost": [real]}, null=null)


def test_a_perfect_model_beats_every_shuffle_and_the_share_is_never_zero(
    table: pd.DataFrame,
) -> None:
    null = ec.loco_null(table, fabricated_loco(table, n_null=20))
    assert len(null) == 5 * len(gm.METRICS)  # four cohorts and the pooled group
    pooled = null[(null["cohort"] == ec.POOLED) & (null["metric"] == "bal_acc")].iloc[0]
    assert pooled["real"] == 1.0
    assert pooled["p_value"] == pytest.approx(1 / 21)
    assert pooled["n_shuffles"] == 20
    assert abs(pooled["null_mean"] - 1 / 3) < 0.05
    assert pooled["null_p95"] > pooled["null_mean"]
    assert (null["p_value"] > 0).all()


def test_the_share_counts_shuffles_that_do_as_well(table: pd.DataFrame) -> None:
    result = fabricated_loco(table, n_null=20, perfect=False)
    result.null[0] = result.proba["xgboost"][0].copy()  # one shuffle scores exactly the same
    null = ec.loco_null(table, result)
    row = null[(null["cohort"] == ec.POOLED) & (null["metric"] == "auc")].iloc[0]
    y = table["severity"].to_numpy(int)
    w = gm.subject_weights(table["subject_key"].to_numpy())
    scores = np.array([gm.metric_values(y, [p], w)["auc"] for p in result.null])
    expected = (1 + int((scores >= row["real"]).sum())) / 21
    assert row["p_value"] == pytest.approx(expected)
    assert row["p_value"] >= 2 / 21


def test_loco_null_on_a_real_run_reports_each_cohort(
    table: pd.DataFrame, loco: ec.LocoResult
) -> None:
    null = ec.loco_null(table, loco)
    assert set(null["cohort"]) == {*COHORTS, ec.POOLED}
    assert (null["p_value"] >= 1 / 13).all() and (null["p_value"] <= 1).all()
    pooled = null[(null["cohort"] == ec.POOLED) & (null["metric"] == "bal_acc")].iloc[0]
    assert pooled["real"] > pooled["null_p95"]


# --------------------------------------------------------------------------- the variants
def test_the_variants_are_well_formed() -> None:
    names = [v.name for v in ec.VARIANTS]
    assert names[0] == ec.REFERENCE and len(set(names)) == len(names)
    by_name = {v.name: v for v in ec.VARIANTS}
    assert by_name[ec.REFERENCE].features == FEATURE_NAMES
    for v in ec.VARIANTS:
        assert v.features and set(v.features) <= set(FEATURE_NAMES)
        assert len(set(v.features)) == len(v.features)
    assert len(by_name["no_joint_angles"].features) == 5
    assert set(by_name["no_joint_angles"].features).isdisjoint(ec.JOINT_ANGLES)
    assert by_name["timing_only"].features == ec.TIMING
    assert "stride_time_cv" not in by_name["no_stride_time_cv"].features
    assert "hip_flexion_range_deg" not in by_name["no_hip_and_knee_flexion"].features
    assert "knee_flexion_range_deg" not in by_name["no_hip_and_knee_flexion"].features
    assert {v.drop_cohort for v in ec.VARIANTS} == {None, "BMCLab", "PD-GaM"}


def test_without_refuses_a_name_that_is_not_an_input() -> None:
    with pytest.raises(ValueError, match="not model inputs"):
        ec._without("step_width_norm")


def test_the_all_input_variant_equals_the_gait19_cross_validation(
    table: pd.DataFrame, assignments: pd.DataFrame, cv: tx.CVResult
) -> None:
    out = ec.run_variant(table, assignments, ec.VARIANTS[0], params=FAST)
    assert tuple(out.proba) == ec.ABLATION_MODELS
    for m in ec.ABLATION_MODELS:
        assert len(out.proba[m]) == 2
        for mine, theirs in zip(out.proba[m], cv.proba[m], strict=True):
            assert np.array_equal(mine, theirs)  # not close: identical


def test_a_variant_gives_the_model_only_its_own_inputs(
    table: pd.DataFrame, assignments: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract: list[list[str]] = []
    widths: list[int] = []
    real_contract, real_fit = ec.check_feature_contract, tx.fit_xgboost_model

    def contract_spy(names, label_name=None):  # type: ignore[no-untyped-def]
        contract.append(list(names))
        real_contract(names, label_name)

    def spy(x, y, w, seed, params):  # type: ignore[no-untyped-def]
        widths.append(x.shape[1])
        return real_fit(x, y, w, seed, params)

    monkeypatch.setattr(ec, "check_feature_contract", contract_spy)

    monkeypatch.setattr(tx, "fit_xgboost_model", spy)
    variant = next(v for v in ec.VARIANTS if v.name == "no_joint_angles")
    ec.run_variant(table, assignments, variant, params=FAST, seeds=[0], models=("xgboost",))
    assert contract and all(c == list(variant.features) for c in contract)
    assert widths and set(widths) == {5}


def test_removing_the_informative_inputs_removes_the_score(
    table: pd.DataFrame, assignments: pd.DataFrame
) -> None:
    """The class lives in the first three inputs: dropping them must cost, dropping others not."""
    y = table["severity"].to_numpy(int)

    def score(features: tuple[str, ...]) -> float:
        v = ec.Variant("x", "x", features)
        out = ec.run_variant(table, assignments, v, params=FAST, seeds=[0], models=("xgboost",))
        return gm.balanced_accuracy(y, out.proba["xgboost"][0].argmax(axis=1))

    full = score(FEATURE_NAMES)
    assert score(FEATURE_NAMES[3:]) < full - 0.2
    assert score(FEATURE_NAMES[:3]) > full - 0.08


def test_a_banned_input_is_refused_in_a_variant(
    table: pd.DataFrame, assignments: pd.DataFrame
) -> None:
    bad = ec.Variant("bad", "uses the label", (*FEATURE_NAMES[:3], "severity"))
    with pytest.raises(FeatureContractError):
        ec.run_variant(table, assignments, bad, params=FAST, seeds=[0], models=("xgboost",))
    missing = ec.Variant("gone", "no such column", ("nope",))
    with pytest.raises(ValueError, match="lacks the columns"):
        ec.run_variant(table, assignments, missing, params=FAST)


def test_a_removed_cohort_is_in_neither_training_nor_testing(
    table: pd.DataFrame, assignments: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = []
    real = ec.fit_variant_model

    def spy(kind, train, test, y, features, seed, params):  # type: ignore[no-untyped-def]
        seen.append((set(train["cohort"]), set(test["cohort"])))
        return real(kind, train, test, y, features, seed, params)

    monkeypatch.setattr(ec, "fit_variant_model", spy)
    variant = ec.Variant("no_A", "A removed", FEATURE_NAMES, "A")
    out = ec.run_variant(
        table, assignments, variant, params=FAST, seeds=[0], models=("xgboost", "logistic")
    )
    assert seen and all("A" not in train and "A" not in test for train, test in seen)
    assert set(out.table["cohort"]) == {"B", "C", "D"}
    assert out.proba["xgboost"][0].shape == (len(out.table), 3)
    assert out.table["walk"].is_unique
    sub_assign = assignments[assignments["subject_key"].isin(set(out.table["subject_key"]))]
    assert out.fold_problems == mf.fold_problems(out.table, sub_assign)


def test_removing_an_unknown_cohort_is_an_error(
    table: pd.DataFrame, assignments: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match="no cohort"):
        ec.run_variant(table, assignments, ec.Variant("x", "x", FEATURE_NAMES, "Z"), params=FAST)


def test_an_unknown_model_kind_is_an_error(table: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="unknown model"):
        ec.fit_variant_model(
            "forest", table, table, np.zeros(len(table), int), FEATURE_NAMES, 0, FAST
        )


def test_the_reproduction_check_counts_differing_arrays(
    table: pd.DataFrame, assignments: pd.DataFrame
) -> None:
    ref = ec.run_variant(table, assignments, ec.VARIANTS[0], params=FAST, seeds=[0])
    same = {m: [p.copy() for p in ref.proba[m]] for m in ("xgboost", "logistic")}
    assert ec.reproduction_gap(ref, same) == (0.0, 0)
    off = {m: [p.copy() for p in same[m]] for m in same}
    off["xgboost"][0][0, 0] += 0.05
    gap, n = ec.reproduction_gap(ref, off)
    assert gap == pytest.approx(0.05) and n == 1
    rounded = {m: [np.round(p, 6) for p in same[m]] for m in same}
    assert ec.reproduction_gap(ref, rounded)[1] == 0  # the 6 decimals of the file are tolerated
    with pytest.raises(ValueError, match="seeds"):
        ec.reproduction_gap(ref, {"xgboost": same["xgboost"] * 2, "logistic": same["logistic"]})


# --------------------------------------------------------------------------- scores
def test_cohort_scores_are_the_scores_of_that_cohorts_walks(
    table: pd.DataFrame, cv: tx.CVResult
) -> None:
    scores, _ = ec.cohort_analysis(table, {"xgboost": cv.proba["xgboost"]}, n_boot=20)
    for cohort in COHORTS:
        mask = (table["cohort"] == cohort).to_numpy()
        y = table["severity"].to_numpy(int)[mask]
        w = gm.subject_weights(table["subject_key"].to_numpy()[mask])
        expected = gm.metric_values(y, [p[mask] for p in cv.proba["xgboost"]], w)
        row = scores[(scores["cohort"] == cohort) & (scores["model"] == "xgboost")].iloc[0]
        for m in gm.METRICS:
            assert row[m] == pytest.approx(expected[m])
            assert row[f"{m}_lo"] <= row[m] <= row[f"{m}_hi"]
        assert row["n_walks"] == mask.sum()
        assert row["n_subjects"] == table.loc[mask, "subject_key"].nunique()
        assert row["n_class0"] + row["n_class1"] + row["n_class2"] == mask.sum()
        assert row["chance_bal_acc"] == pytest.approx(1 / 3)


def test_the_pooled_group_equals_the_gait19_summary_including_the_intervals(
    table: pd.DataFrame, cv: tx.CVResult
) -> None:
    models = {"xgboost": cv.proba["xgboost"], "logistic": cv.proba["logistic"]}
    scores, _ = ec.cohort_analysis(table, models, n_boot=30)
    summary = tx.summarize(table, cv, n_boot=30).set_index("model")
    for m in models:
        row = scores[(scores["cohort"] == ec.POOLED) & (scores["model"] == m)].iloc[0]
        for name in gm.METRICS:
            assert row[name] == pytest.approx(summary.loc[m, name])
            assert row[f"{name}_lo"] == pytest.approx(summary.loc[m, f"{name}_lo"])
            assert row[f"{name}_hi"] == pytest.approx(summary.loc[m, f"{name}_hi"])


def test_setting_marks_leave_one_cohort_out_models(
    table: pd.DataFrame, cv: tx.CVResult, loco: ec.LocoResult
) -> None:
    models = {"xgboost": cv.proba["xgboost"], "loco_xgboost": loco.proba["xgboost"]}
    scores, _ = ec.cohort_analysis(table, models, n_boot=10)
    setting = scores.drop_duplicates("model").set_index("model")["setting"]
    assert setting["xgboost"] == "cv" and setting["loco_xgboost"] == "leave_one_cohort_out"


def test_a_difference_is_model_minus_versus_and_is_paired(
    table: pd.DataFrame, cv: tx.CVResult
) -> None:
    models = {"xgboost": cv.proba["xgboost"], "majority": cv.proba["majority"]}
    y = table["severity"].to_numpy(int)
    scores, draws = ec.score_models(y, table["subject_key"].to_numpy(), models, n_boot=60)
    diffs = ec.difference_table(
        scores, draws, [("xgboost", "majority"), ("xgboost", "xgboost"), ("xgboost", "absent")]
    )
    assert set(zip(diffs["model"], diffs["versus"], strict=True)) == {
        ("xgboost", "majority"),
        ("xgboost", "xgboost"),
    }
    better = diffs[(diffs["versus"] == "majority") & (diffs["metric"] == "bal_acc")].iloc[0]
    by_model = scores.set_index("model")
    assert better["diff"] == pytest.approx(
        by_model.loc["xgboost", "bal_acc"] - by_model.loc["majority", "bal_acc"]
    )
    assert better["lo"] > 0  # a model with a real signal clearly beats the majority rung
    same = diffs[(diffs["versus"] == "xgboost")]
    assert (same["diff"] == 0).all() and (same["lo"] == 0).all() and (same["hi"] == 0).all()


# --------------------------------------------------------------------------- ablation scores
@pytest.fixture(scope="module")
def ablation_results(table: pd.DataFrame, assignments: pd.DataFrame) -> dict[str, ec.VariantResult]:
    variants = (
        ec.VARIANTS[0],
        ec.Variant("no_signal", "the three informative inputs removed", FEATURE_NAMES[3:]),
        ec.Variant("no_A", "A removed", FEATURE_NAMES, "A"),
    )
    return {v.name: ec.run_variant(table, assignments, v, params=FAST) for v in variants}


def test_ablation_scores_compare_with_the_reference_on_the_same_walks(
    table: pd.DataFrame, ablation_results: dict[str, ec.VariantResult]
) -> None:
    scores, diffs = ec.ablation_analysis(ablation_results, n_boot=30)
    assert set(scores["variant"]) == {ec.REFERENCE, "no_signal", "no_A", f"{ec.REFERENCE}_on_no_A"}
    ref = scores[(scores["variant"] == ec.REFERENCE) & (scores["model"] == "xgboost")].iloc[0]
    assert ref["n_walks"] == len(table)
    cut = scores[(scores["variant"] == "no_A") & (scores["model"] == "xgboost")].iloc[0]
    assert cut["n_walks"] == (table["cohort"] != "A").sum()
    # the reference restricted to the same walks is scored as the plain reference on those walks
    restricted = scores[
        (scores["variant"] == f"{ec.REFERENCE}_on_no_A") & (scores["model"] == "xgboost")
    ].iloc[0]
    keep = (table["cohort"] != "A").to_numpy()
    y = table["severity"].to_numpy(int)[keep]
    w = gm.subject_weights(table["subject_key"].to_numpy()[keep])
    base = ablation_results[ec.REFERENCE].proba["xgboost"]
    expected = gm.metric_values(y, [p[keep] for p in base], w)
    assert restricted["bal_acc"] == pytest.approx(expected["bal_acc"])
    assert restricted["n_walks"] == cut["n_walks"]
    # differences: zero for the reference, clearly negative once the signal is removed
    assert set(diffs["versus"]) == {ec.REFERENCE}
    zero = diffs[(diffs["variant"] == ec.REFERENCE) & (diffs["metric"] == "bal_acc")]
    assert (zero["diff"] == 0).all() and (zero["hi"] == 0).all()
    lost = diffs[
        (diffs["variant"] == "no_signal")
        & (diffs["model"] == "xgboost")
        & (diffs["metric"] == "bal_acc")
    ].iloc[0]
    assert lost["hi"] < 0
    same_rows = diffs[(diffs["variant"] == "no_A") & (diffs["model"] == "xgboost")]
    assert len(same_rows) == len(gm.METRICS)


def test_ablation_scores_carry_their_description_and_counts(
    ablation_results: dict[str, ec.VariantResult],
) -> None:
    scores, _ = ec.ablation_analysis(ablation_results, n_boot=10)
    row = scores[(scores["variant"] == "no_signal") & (scores["model"] == "xgboost")].iloc[0]
    assert row["n_features"] == 5 and "informative" in row["description"]
    assert list(scores.columns[:7]) == [
        "variant",
        "description",
        "model",
        "n_features",
        "n_walks",
        "n_subjects",
        "fold_warnings",
    ]
    assert (scores["model"].isin(ec.ABLATION_MODELS)).all()


# --------------------------------------------------------------------------- shares and gates
def test_class_shares_add_up_and_show_the_predicted_mix(
    table: pd.DataFrame, cv: tx.CVResult
) -> None:
    shares = ec.class_shares(table, {"majority": cv.proba["majority"]})
    for _, part in shares.groupby("cohort"):
        assert part["true_share"].sum() == pytest.approx(1.0)
        assert part["predicted_share"].sum() == pytest.approx(1.0)
    a = shares[shares["cohort"] == "A"].set_index("class")
    mask = (table["cohort"] == "A").to_numpy()
    expected = np.bincount(table["severity"].to_numpy(int)[mask], minlength=3) / mask.sum()
    assert a["true_share"].tolist() == pytest.approx(expected.tolist())


def test_recall_is_the_share_of_each_class_found_and_nan_for_an_absent_class(
    table: pd.DataFrame,
) -> None:
    t = table.copy()
    t.loc[(t["cohort"] == "A") & (t["severity"] == 2), "severity"] = 1  # cohort A has no class 2
    y = t["severity"].to_numpy(int)
    perfect = [np.eye(3)[y]]
    always_zero = [np.tile([1.0, 0.0, 0.0], (len(t), 1))]
    shares = ec.class_shares(t, {"perfect": perfect, "zero": always_zero}).set_index(
        ["cohort", "model", "class"]
    )
    assert shares.loc[("B", "perfect", "class 1"), "recall"] == 1.0
    assert shares.loc[("A", "perfect", "class 2-3"), "true_share"] == 0.0
    assert np.isnan(shares.loc[("A", "perfect", "class 2-3"), "recall"])
    assert shares.loc[("A", "zero", "class 0"), "recall"] == 1.0
    assert shares.loc[("A", "zero", "class 1"), "recall"] == 0.0
    assert shares.loc[("A", "zero", "class 0"), "predicted_share"] == 1.0
    two = ec.class_shares(t, {"mixed": [np.eye(3)[y], np.tile([1.0, 0, 0], (len(t), 1))]})
    row = two[(two["cohort"] == "B") & (two["class"] == "class 1")].iloc[0]
    assert row["recall"] == pytest.approx(0.5)  # the mean over the two seeds


def test_control_problems_name_the_variant_that_failed() -> None:
    assert ec.control_problems("x", 0.34, 0.51) == []
    problems = ec.control_problems("no_BMCLab", 0.45, 0.62)
    assert len(problems) == 2 and all(p.startswith("no_BMCLab:") for p in problems)
    scores = pd.DataFrame(
        [
            {"variant": "ok", "model": tx.CONTROL, "bal_acc": 0.33, "auc": 0.5},
            {"variant": "leaky", "model": tx.CONTROL, "bal_acc": 0.5, "auc": 0.5},
            {"variant": "leaky", "model": "xgboost", "bal_acc": 0.9, "auc": 0.9},
        ]
    )
    out = ec.all_control_problems(scores)
    assert len(out) == 1 and out[0].startswith("leaky:")


# --------------------------------------------------------------------------- the command line
def write_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    monkeypatch.setattr(tx, "XGB_PARAMS", FAST)
    t = cohort_table()
    table_path, folds_path = tmp_path / "t.csv", tmp_path / "f.csv"
    t.to_csv(table_path, index=False)
    mf.make_assignments(t, 5, (0, 1)).to_csv(folds_path, index=False)
    out = tmp_path / "xgb"
    code = tx.main(
        [
            "--table",
            str(table_path),
            "--folds",
            str(folds_path),
            "--out-dir",
            str(out),
            "--bootstrap",
            "20",
            "--no-shap",
        ]
    )
    assert code == 0
    return {"table": table_path, "folds": folds_path, "oof": out / "oof_predictions.csv"}


SMALL_VARIANTS = (
    ec.VARIANTS[0],
    ec.Variant("no_first_three", "the informative inputs removed", FEATURE_NAMES[3:]),
    ec.Variant("no_A", "A removed", FEATURE_NAMES, "A"),
)


def run_cli(paths: dict[str, Path], out: Path, *extra: str) -> int:
    return ec.main(
        [
            "--table",
            str(paths["table"]),
            "--folds",
            str(paths["folds"]),
            "--oof",
            str(paths["oof"]),
            "--out-dir",
            str(out),
            "--bootstrap",
            "20",
            "--shuffles",
            "6",
            *extra,
        ]
    )


def test_cli_missing_inputs_exit_2(tmp_path: Path) -> None:
    nothing = tmp_path / "no.csv"
    assert ec.main(["--table", str(nothing), "--out-dir", str(tmp_path)]) == 2
    t = tmp_path / "t.csv"
    cohort_table(12).to_csv(t, index=False)
    assert ec.main(["--table", str(t), "--folds", str(nothing), "--out-dir", str(tmp_path)]) == 2
    f = tmp_path / "f.csv"
    mf.make_assignments(cohort_table(12), 5, (0,)).to_csv(f, index=False)
    code = ec.main(
        ["--table", str(t), "--folds", str(f), "--oof", str(nothing), "--out-dir", str(tmp_path)]
    )
    assert code == 2


def test_cli_runs_end_to_end_and_writes_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = write_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(ec, "VARIANTS", SMALL_VARIANTS)
    out = tmp_path / "cohorts"
    assert run_cli(paths, out) == 0
    text = capsys.readouterr().out
    for name in (
        "cohort_scores.csv",
        "cohort_differences.csv",
        "cohort_class_shares.csv",
        "loco_null.csv",
        "loco_predictions.csv",
        "ablations.csv",
        "ablation_differences.csv",
        "results.json",
    ):
        assert (out / name).is_file()
    for heading in (
        "SCORES BY COHORT",
        "LEAVE ONE COHORT OUT AGAINST SHUFFLED LABELS",
        "CLASS SHARES",
        "ABLATIONS",
    ):
        assert heading in text
    assert "all at chance" in text
    assert "secret" not in text
    results = json.loads((out / "results.json").read_text())
    assert results["loco_shuffles"] == 6 and results["bootstrap_draws"] == 20
    assert results["reproduction_arrays_differing"] == 0  # the 6 decimals of the file are enough
    assert results["loco_train_cohorts"]["A"] == ["B", "C", "D"]
    assert results["control_problems"] == []
    ablations = pd.read_csv(out / "ablations.csv")
    assert set(ablations["variant"]) == {
        ec.REFERENCE,
        "no_first_three",
        "no_A",
        "all_inputs_on_no_A",
    }


def test_cli_without_ablations_skips_them(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = write_inputs(tmp_path, monkeypatch)
    out = tmp_path / "cohorts"
    assert run_cli(paths, out, "--no-ablations") == 0
    assert (out / "loco_null.csv").is_file() and not (out / "ablations.csv").exists()


def test_cli_exits_1_when_a_cross_validation_control_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = write_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(ec, "VARIANTS", SMALL_VARIANTS[:1])
    monkeypatch.setattr(ec, "all_control_problems", lambda scores: ["all_inputs: leak"])
    assert run_cli(paths, tmp_path / "cohorts") == 1
    assert "FAILED" in capsys.readouterr().out


def test_cli_refuses_predictions_that_do_not_belong_to_the_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = write_inputs(tmp_path, monkeypatch)
    oof = pd.read_csv(paths["oof"])
    oof.loc[oof.index[7], "severity"] = (oof.loc[oof.index[7], "severity"] + 1) % 3
    oof.to_csv(paths["oof"], index=False)
    with pytest.raises(ValueError, match="severity differs"):
        run_cli(paths, tmp_path / "cohorts", "--no-ablations")


def test_cli_rejects_folds_that_do_not_match_the_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = write_inputs(tmp_path, monkeypatch)
    mf.make_assignments(cohort_table(45, seed=3), 5, (0,)).to_csv(paths["folds"], index=False)
    with pytest.raises(ValueError, match="different subjects"):
        run_cli(paths, tmp_path / "cohorts")
