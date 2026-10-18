"""GAIT-23: the SHAP explanations of the saved model."""

from __future__ import annotations

from pathlib import Path

import explain_shap as es
import model_bundle as mb
import numpy as np
import pandas as pd
import pytest
import train_xgboost as tx
from gait_extract import FEATURE_NAMES
from make_folds import CLASS_LABELS, make_assignments
from test_evaluate_cohorts import FAST, cohort_table
from test_train_xgboost import synthetic_table

LABELS = [CLASS_LABELS[c] for c in range(3)]


@pytest.fixture(scope="module")
def setup(tmp_path_factory: pytest.TempPathFactory) -> tuple[es.Explainer, np.ndarray]:
    table = cohort_table()
    model = tx.fit_final_model(table, params=FAST)
    folder = tmp_path_factory.mktemp("bundle")
    bundle = mb.save_bundle(model, folder, {})
    return es.Explainer(bundle), table[list(FEATURE_NAMES)].to_numpy(float)


def test_shap_adds_up_to_the_model_output(setup: tuple[es.Explainer, np.ndarray]) -> None:
    explainer, x = setup
    assert explainer.values(x).shape == (len(x), len(FEATURE_NAMES), 3)
    assert explainer.additivity_error(x) < es.ADDITIVITY_TOL


def test_values_match_xgboost_own_tree_shap(setup: tuple[es.Explainer, np.ndarray]) -> None:
    import xgboost

    explainer, x = setup
    booster = explainer.bundle.model.get_booster()
    native = booster.predict(
        xgboost.DMatrix(x[:50]), pred_contribs=True
    )  # walks, classes, inputs+1
    ours = explainer.values(x[:50])
    assert np.allclose(np.transpose(native[:, :, :-1], (0, 2, 1)), ours, atol=1e-4)


def test_explain_gives_the_top_three_of_the_predicted_class(
    setup: tuple[es.Explainer, np.ndarray],
) -> None:
    explainer, x = setup
    for i in (0, 7, 40, 90):
        exp = explainer.explain(x[i])
        proba = explainer.bundle.predict_proba(x[i])[0]
        assert exp.predicted_class == int(proba.argmax())
        assert np.allclose(exp.probabilities, proba)
        own = explainer.values(x[i])[0, :, exp.predicted_class]
        assert len(exp.top) == es.TOP
        assert [c.feature for c in exp.top] == [
            FEATURE_NAMES[j] for j in np.argsort(-np.abs(own), kind="stable")[:3]
        ]
        sizes = [abs(c.shap) for c in exp.top]
        assert sizes == sorted(sizes, reverse=True)
        for c in exp.top:
            j = list(FEATURE_NAMES).index(c.feature)
            assert c.value == pytest.approx(x[i, j]) and c.shap == pytest.approx(own[j])
            assert c.direction == ("towards" if own[j] > 0 else "away")


def test_explain_is_the_same_after_a_reload(
    setup: tuple[es.Explainer, np.ndarray], tmp_path: Path
) -> None:
    explainer, x = setup
    mb.save_bundle(explainer.bundle.model, tmp_path / "b", {})
    again = es.Explainer(mb.load_bundle(tmp_path / "b"))
    assert again.explain(x[3]) == explainer.explain(x[3])


def test_the_inputs_that_carry_the_signal_lead(setup: tuple[es.Explainer, np.ndarray]) -> None:
    explainer, x = setup
    table = es.global_importance(explainer.values(x))
    assert set(table["feature"][:3]) == set(FEATURE_NAMES[:3])
    assert table["overall"].is_monotonic_decreasing
    assert np.allclose(table[LABELS].mean(axis=1), table["overall"])


def test_global_importance_is_mean_absolute_value() -> None:
    values = np.zeros((2, len(FEATURE_NAMES), 3))
    values[0, 2, 1], values[1, 2, 1] = 1.0, -3.0
    table = es.global_importance(values)
    assert table.loc[0, "feature"] == FEATURE_NAMES[2]
    assert table.loc[0, CLASS_LABELS[1]] == pytest.approx(2.0)
    assert table.loc[0, "overall"] == pytest.approx(2.0 / 3)


def test_top_features_follow_the_predicted_class_not_the_others() -> None:
    values = np.zeros((2, len(FEATURE_NAMES), 3))
    values[:, 0, 0] = 5.0  # input 0 matters a lot for class 0
    values[:, 1, 1] = 2.0
    values[:, 2, 1] = -1.0
    values[:, 3, 1] = 0.5
    predicted = np.array([1, 0])
    top = es.top_features(values, predicted)
    assert list(top[0]) == [1, 2, 3] and top[1][0] == 0
    freq = es.top_frequency(top, predicted).set_index("feature")
    assert freq.loc[FEATURE_NAMES[1], CLASS_LABELS[1]] == 1.0
    assert freq.loc[FEATURE_NAMES[0], CLASS_LABELS[0]] == 1.0
    assert freq.loc[FEATURE_NAMES[0], CLASS_LABELS[1]] == 0.0
    assert np.isnan(freq.loc[FEATURE_NAMES[0], CLASS_LABELS[2]])


def test_rank_agreement() -> None:
    a = pd.DataFrame({"feature": list("abcd"), "overall": [4.0, 3.0, 2.0, 1.0]})
    assert es.rank_agreement(a, a)["overall"] == pytest.approx(1.0)
    b = a.assign(overall=[1.0, 2.0, 3.0, 4.0])
    assert es.rank_agreement(a, b)["overall"] == pytest.approx(-1.0)


def example_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    table = pd.DataFrame(
        {"walk": [f"w{i}" for i in range(9)], "severity": [0, 0, 0, 0, 0, 1, 1, 2, 2]}
    )
    conf = [0.40, 0.50, 0.60, 0.70, 0.99, 0.80, 0.90, 0.60, 0.95]
    rows = []
    for w, c, k in zip(table["walk"], conf, [0, 0, 0, 0, 0, 1, 1, 2, 2], strict=True):
        p = [0.0, 0.0, 0.0]
        p[k] = c
        rows.append({"walk": w, "model": "xgboost", "seed": 0, "p0": p[0], "p1": p[1], "p2": p[2]})
    return table, pd.DataFrame(rows)


def test_example_walk_is_right_out_of_fold_and_in_the_saved_model_at_median_confidence() -> None:
    table, oof = example_inputs()
    saved = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2])  # w3 and w4 are wrong in the saved model
    picked = es.pick_examples(table, oof, saved)
    assert picked[0] == "w1"  # candidates w0, w1, w2: median confidence 0.50
    assert picked[1] in {"w5", "w6"}
    assert picked[2] in {"w7", "w8"}


def test_example_walk_skips_walks_the_out_of_fold_model_got_wrong() -> None:
    table, oof = example_inputs()
    oof.loc[oof["walk"].isin(["w0", "w1", "w2", "w3"]), ["p0", "p1"]] = [0.1, 0.45]
    saved = np.array([0, 0, 0, 0, 0, 1, 1, 2, 2])
    assert es.pick_examples(table, oof, saved)[0] == "w4"  # the only one right out of fold


def test_only_the_xgboost_rows_of_the_out_of_fold_file_count() -> None:
    table, oof = example_inputs()
    other = oof.assign(model="logistic", p0=0.0, p1=0.0, p2=1.0)  # wrong everywhere
    saved = np.array([0, 0, 0, 0, 0, 1, 1, 2, 2])
    assert es.pick_examples(table, pd.concat([oof, other]), saved) == es.pick_examples(
        table, oof, saved
    )


def test_class_without_a_candidate_is_left_out() -> None:
    table, oof = example_inputs()
    saved = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert set(es.pick_examples(table, oof, saved)) == {0}


# --------------------------------------------------------------------------- command line
@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("trained")
    t = synthetic_table(60)
    table_path, folds_path, out = root / "t.csv", root / "f.csv", root / "out"
    t.to_csv(table_path, index=False)
    make_assignments(t, 5, (0, 1)).to_csv(folds_path, index=False)
    old = tx.XGB_PARAMS
    tx.XGB_PARAMS = FAST
    try:
        code = tx.main(
            ["--table", str(table_path), "--folds", str(folds_path), "--out-dir", str(out),
             "--bootstrap", "20"]
        )  # fmt: skip
    finally:
        tx.XGB_PARAMS = old
    assert code == 0
    return {"table": table_path, "out": out}


def cli(trained: dict[str, Path], out: Path, **override: Path) -> list[str]:
    paths = {
        "table": trained["table"],
        "bundle": trained["out"] / "bundle",
        "oof": trained["out"] / "oof_predictions.csv",
        "shap-table": trained["out"] / "shap_mean_abs.csv",
        "out-dir": out,
        **{k.replace("_", "-"): v for k, v in override.items()},
    }
    return [a for k, v in paths.items() for a in (f"--{k}", str(v))]


def test_cli_writes_charts_and_tables(
    trained: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = es.main(cli(trained, tmp_path / "shap"))
    text = capsys.readouterr().out
    assert code == 0 and "additivity error" in text and "Spearman" in text
    for name in (
        "global_importance.csv",
        "global_importance.png",
        "top3_frequency.csv",
        "example_walks.csv",
        "example_walks.png",
    ):
        assert (tmp_path / "shap" / name).stat().st_size > 0
    walks = pd.read_csv(tmp_path / "shap" / "example_walks.csv")
    assert set(walks["direction"]) <= {"towards", "away"}
    assert walks.groupby("walk")["rank"].apply(list).map(lambda r: r == [1, 2, 3]).all()


def test_cli_missing_inputs_exit_2(trained: dict[str, Path], tmp_path: Path) -> None:
    assert es.main(cli(trained, tmp_path, bundle=tmp_path / "no")) == 2
    assert es.main(cli(trained, tmp_path, oof=tmp_path / "no.csv")) == 2


def test_cli_refuses_a_table_the_model_was_not_trained_on(
    trained: dict[str, Path], tmp_path: Path
) -> None:
    other = tmp_path / "other.csv"
    synthetic_table(60, seed=5).to_csv(other, index=False)
    assert es.main(cli(trained, tmp_path / "o", table=other)) == 2


def test_cli_fails_when_shap_does_not_add_up(
    trained: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(es.Explainer, "additivity_error", lambda self, x: 1.0)
    assert es.main(cli(trained, tmp_path / "bad")) == 1


def test_cli_refuses_a_changed_model_file(trained: dict[str, Path], tmp_path: Path) -> None:
    import shutil

    shutil.copytree(trained["out"] / "bundle", tmp_path / "b")
    with (tmp_path / "b" / mb.MODEL_FILE).open("a") as f:
        f.write(" ")
    assert es.main(cli(trained, tmp_path / "o", bundle=tmp_path / "b")) == 2
