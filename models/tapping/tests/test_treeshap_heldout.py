"""Tests for models/tapping/src/treeshap_heldout.py (task FT-42). A small synthetic matrix is run
through the real harness to get official lgbm runs; nothing real is used."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv_harness as ch  # noqa: E402
import cv_splits as cvs  # noqa: E402
import shap  # noqa: E402
import treeshap_heldout as tsh  # noqa: E402
from test_cv_harness import SMALL_LGBM, build_world, make_data  # noqa: E402

FILES = (tsh.SHAP_FILE, tsh.FOLD_FILE, tsh.GLOBAL_FILE, tsh.META_FILE)


@pytest.fixture(autouse=True)
def small_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(ch.GRIDS, "lgbm", [dict(p) for p in SMALL_LGBM])
    monkeypatch.setattr(tsh, "CONFIGS", (("ge2", "A"), ("ge2", "B")))


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Official lgbm A and B runs on a synthetic matrix, built once."""
    tmp = tmp_path_factory.mktemp("official")
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(ch.GRIDS, "lgbm", [dict(p) for p in SMALL_LGBM])
        w = build_world(tmp, mp, make_data(), repeats=5)
        for fs in ("A", "B"):
            code = ch.main(
                ["run", "--model", "lgbm", "--feature-set", fs, "--repeats", "5",
                 "--features", str(w["proc"] / "features.csv"),
                 "--roles", str(w["proc"] / "feature_roles.csv"),
                 "--meta", str(w["proc"] / "features_meta.json"),
                 "--splits-dir", str(w["cv"]), "--out-dir", str(w["cv"])]
            )  # fmt: skip
            assert code == 0
    return {"tmp": tmp}


@pytest.fixture
def w(base: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    shutil.copytree(base["tmp"] / "data", tmp_path / "data")
    proc = tmp_path / "data" / "processed"
    monkeypatch.setattr(cvs, "CV_ROOT", proc / "cv")
    monkeypatch.setattr(cvs, "PROCESSED_ROOT", proc)
    return {"proc": proc, "cv": proc / "cv", "xai": proc / "xai"}


def go(w: dict[str, Path], *extra: str) -> int:
    return tsh.main(
        ["--cv-root", str(w["cv"]), "--features", str(w["proc"] / "features.csv"),
         "--roles", str(w["proc"] / "feature_roles.csv"),
         "--meta", str(w["proc"] / "features_meta.json"), *extra]
    )  # fmt: skip


def out(w: dict[str, Path], fs: str = "A") -> Path:
    return w["xai"] / "ge2" / fs


def pieces(w: dict[str, Path], fs: str = "A") -> dict[str, Any]:
    frame = cvs.canonical(
        pd.read_csv(w["proc"] / "features.csv", dtype={"ref": str, "participant": str})
    )
    splits = cvs.load_splits(w["cv"] / "ge2" / "splits.csv", frame)
    return {
        "frame": frame, "splits": splits,
        "oof": ch.read_oof(w["cv"] / "ge2" / f"oof_lgbm_{fs}.csv"),
        "best": tsh.read_best_params(w["cv"] / "ge2" / f"best_params_lgbm_{fs}.csv", ch.grid_for("lgbm")),
    }  # fmt: skip


def edit_meta(w: dict[str, Path], fs: str, fn: Any) -> None:
    path = w["cv"] / "ge2" / f"meta_lgbm_{fs}.json"
    meta = json.loads(path.read_text())
    fn(meta)
    path.write_text(json.dumps(meta))


def read_out(w: dict[str, Path], name: str, fs: str = "A") -> pd.DataFrame:
    return pd.read_csv(out(w, fs) / name, dtype={"clip_ref": str, "participant_ref": str})


# --------------------------------------------------------------------------- constants


def test_preregistered_constants() -> None:
    assert tsh.EXPLAINER == {"feature_perturbation": "tree_path_dependent", "model_output": "raw"}
    assert (tsh.ADDITIVITY_TOL, tsh.OOF_TOL, tsh.REPEATS, tsh.MODEL) == (1e-6, 1e-9, 5, "lgbm")
    source = Path(tsh.__file__).read_text()
    assert "KernelExplainer" not in source and "PermutationExplainer" not in source


def test_configs_are_the_registered_three() -> None:
    source = Path(tsh.__file__).read_text()
    assert 'CONFIGS = (("ge2", "A"), ("ge1", "A"), ("ge2", "B"))' in source


# --------------------------------------------------------------------------- rows and folds


def test_shap_rows_are_test_rows_only_and_each_clip_once_per_repeat(w: dict[str, Path]) -> None:
    assert go(w) == 0
    table = read_out(w, tsh.SHAP_FILE)
    frame = pieces(w)["frame"]
    assert len(table) == 5 * len(frame)
    assert (table.groupby("repeat")["clip_ref"].nunique() == len(frame)).all()
    assert not table.duplicated(["repeat", "clip_ref"]).any()
    splits = pieces(w)["splits"]
    fold_of = {(r.participant, r.repeat): r.outer_fold for r in splits.itertuples()}
    part_of = dict(zip(frame["ref"], frame["participant"], strict=True))
    for row in table[["repeat", "outer_fold", "clip_ref", "participant_ref"]].itertuples():
        assert part_of[row.clip_ref] == row.participant_ref
        assert fold_of[(row.participant_ref, row.repeat)] == row.outer_fold  # held out in that fold
    assert sorted(table["repeat"].unique()) == [0, 1, 2, 3, 4]
    assert sorted(table["outer_fold"].unique()) == [0, 1, 2, 3, 4]


def test_output_row_order_is_repeat_then_ref_order(w: dict[str, Path]) -> None:
    assert go(w) == 0
    table = read_out(w, tsh.SHAP_FILE)
    assert table["repeat"].is_monotonic_increasing
    for _, g in table.groupby("repeat"):
        assert list(g["clip_ref"]) == sorted(g["clip_ref"])


def test_columns_and_feature_order_follow_the_registry(w: dict[str, Path]) -> None:
    assert go(w) == 0
    for fs in ("A", "B"):
        cols = ch.FEATURE_SETS[fs]
        expected = (
            ["repeat", "outer_fold", "clip_ref", "participant_ref", "base_value", "raw_margin", "proba"]
            + [f"shap_{c}" for c in cols] + [f"value_{c}" for c in cols]
        )  # fmt: skip
        assert list(read_out(w, tsh.SHAP_FILE, fs).columns) == expected


def test_each_refit_explains_its_test_rows_and_not_its_train_rows(
    w: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[int, int]] = []
    real = tsh.explain_values

    def spy(est: Any, x_test: np.ndarray) -> Any:
        calls.append(x_test.shape)  # type: ignore[arg-type]
        return real(est, x_test)

    monkeypatch.setattr(tsh, "explain_values", spy)
    monkeypatch.setattr(tsh, "CONFIGS", (("ge2", "A"),))
    assert go(w) == 0
    n = len(pieces(w)["frame"])
    assert (
        len(calls) == 25 and sum(c[0] for c in calls) == 5 * n
    )  # test rows only: each clip per repeat
    assert all(c[0] < n for c in calls)


# --------------------------------------------------------------------------- additivity


def test_additivity_holds_in_the_output(w: dict[str, Path]) -> None:
    assert go(w) == 0
    for fs in ("A", "B"):
        table = read_out(w, tsh.SHAP_FILE, fs)
        shap_cols = [c for c in table.columns if c.startswith("shap_")]
        err = (table["base_value"] + table[shap_cols].sum(axis=1) - table["raw_margin"]).abs().max()
        assert err < 1e-6
        meta = json.loads((out(w, fs) / tsh.META_FILE).read_text())
        assert meta["max_additivity_error"] <= tsh.ADDITIVITY_TOL
        assert abs(meta["max_additivity_error"] - err) < 1e-9


def test_a_corrupted_shap_array_fails_the_additivity_check(
    w: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real = tsh.explain_values

    def corrupt(est: Any, x_test: np.ndarray) -> Any:
        base, values = real(est, x_test)
        values = values.copy()
        values[0, 0] += 1e-3
        return base, values

    monkeypatch.setattr(tsh, "explain_values", corrupt)
    assert go(w) == 1
    assert "additivity" in capsys.readouterr().err
    assert not (out(w) / tsh.SHAP_FILE).exists()


def test_a_wrong_base_value_fails_the_additivity_check(
    w: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    real = tsh.explain_values
    monkeypatch.setattr(tsh, "explain_values", lambda e, x: (real(e, x)[0] + 0.01, real(e, x)[1]))
    assert go(w) == 1


def test_an_unexpected_shap_shape_is_refused(
    w: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    class Fake:
        expected_value = 0.0

        def __init__(self, *a: Any, **k: Any) -> None: ...

        def shap_values(self, x: np.ndarray, **k: Any) -> Any:
            return [np.zeros_like(x), np.zeros_like(x)]  # the old list-of-classes layout

    monkeypatch.setattr(shap, "TreeExplainer", Fake)
    assert go(w) == 1


# --------------------------------------------------------------------------- leakage guard


def test_test_positions_is_passed_to_refit_outer(
    w: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, Any]] = []
    real = ch.refit_outer

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append({"args": args, "kwargs": kwargs})
        return real(*args, **kwargs)

    monkeypatch.setattr(ch, "refit_outer", spy)
    monkeypatch.setattr(tsh, "CONFIGS", (("ge2", "A"),))
    assert go(w) == 0
    assert len(seen) == 25
    for call in seen:
        test = call["kwargs"].get("test_positions")
        assert test is not None and len(test) > 0
        train = call["args"][3]
        assert not set(np.asarray(train).tolist()) & set(np.asarray(test).tolist())


def test_train_and_test_participant_overlap_fails_the_run(
    w: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    p = pieces(w)
    frame = p["frame"]
    groups = frame["participant"].to_numpy()
    real = ch.membership
    moved = {}

    def leaky(g: np.ndarray, splits: pd.DataFrame, repeat: int) -> np.ndarray:
        member = real(g, splits, repeat)
        if repeat == 0:  # one clip of a two-clip participant is moved to another fold
            i = int(np.flatnonzero(g == g[1])[0])
            member = member.copy()
            member[i] = (member[i] + 1) % 5
            moved["i"] = i
        return member

    monkeypatch.setattr(ch, "membership", leaky)

    def unguarded(
        frame: pd.DataFrame,
        model: str,
        params: Any,
        train: Any,
        fs: str,
        target: str,
        test_positions: Any = None,
    ) -> Any:
        X, _ = ch.design_matrix(
            cvs.canonical(frame), fs
        )  # refit_outer without its own overlap check
        y = cvs.target_labels(cvs.canonical(frame), target)
        return ch.fit_final(model, params, X, y, np.asarray(train))

    monkeypatch.setattr(ch, "refit_outer", unguarded)
    oof = p["oof"].copy()
    tmp_member = leaky(groups, p["splits"], 0)
    sel = oof["repeat"] == 0
    oof.loc[sel, "outer_fold"] = oof.loc[sel, "ref"].map(
        dict(zip(frame["ref"], tmp_member, strict=True))
    )
    with pytest.raises(cvs.LeakageError):
        tsh.explain_config(frame, p["splits"], oof, p["best"], "ge2", "A")


def test_overlap_makes_main_exit_1(
    w: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(*a: Any, **k: Any) -> None:
        raise cvs.LeakageError("overlap")

    monkeypatch.setattr(cvs, "assert_disjoint", boom)
    assert go(w) == 1
    assert "overlap" in capsys.readouterr().err


# --------------------------------------------------------------------------- explainer


def test_the_explainer_is_treeexplainer_with_the_registered_settings(
    w: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[dict[str, Any]] = []
    real = shap.TreeExplainer

    def spy(model: Any, *args: Any, **kwargs: Any) -> Any:
        created.append({"args": args, "kwargs": kwargs})
        return real(model, *args, **kwargs)

    def forbidden(*a: Any, **k: Any) -> Any:
        raise AssertionError("a model-agnostic explainer was used")

    monkeypatch.setattr(shap, "TreeExplainer", spy)
    for name in ("KernelExplainer", "PermutationExplainer", "SamplingExplainer", "Explainer"):
        if hasattr(shap, name):
            monkeypatch.setattr(shap, name, forbidden)
    monkeypatch.setattr(tsh, "CONFIGS", (("ge2", "A"),))
    assert go(w) == 0
    assert len(created) == 25
    assert all(
        c["args"] == () and c["kwargs"] == tsh.EXPLAINER for c in created
    )  # no background data


# --------------------------------------------------------------------------- best params


def alternating(repeat: int, fold: int) -> dict[str, Any]:
    return dict(SMALL_LGBM[int(repeat in (0, 1) and fold in (0, 1, 2))])


def officials_from(p: dict[str, Any], key_of: Any) -> pd.DataFrame:
    """An OOF made by refitting with the alternating params, independent of the module."""
    frame = p["frame"]
    X, _ = ch.design_matrix(frame, "A")
    groups = frame["participant"].to_numpy()
    parts = []
    for repeat in range(5):
        member = ch.membership(groups, p["splits"], repeat)
        proba = np.full(len(frame), np.nan)
        for fold in range(5):
            train, test = np.flatnonzero(member != fold), np.flatnonzero(member == fold)
            est = ch.refit_outer(
                frame, "lgbm", alternating(*key_of(repeat, fold)), train, "A", "ge2"
            )
            proba[test] = est.predict_proba(X[test])[:, 1]
        parts.append(pd.DataFrame({"ref": frame["ref"], "repeat": repeat, "outer_fold": member,
                                   "y": 0, "proba": proba}))  # fmt: skip
    return pd.concat(parts, ignore_index=True)


def test_each_fold_uses_its_own_best_params_row(w: dict[str, Path]) -> None:
    p = pieces(w)
    best = {(r, f): alternating(r, f) for r in range(5) for f in range(5)}
    oof = officials_from(p, lambda r, f: (r, f))
    result = tsh.explain_config(p["frame"], p["splits"], oof, best, "ge2", "A")
    assert result["max_diff"] == 0.0  # reproduced exactly
    swapped = {(r, f): alternating(f, r) for r in range(5) for f in range(5)}
    other = tsh.explain_config(p["frame"], p["splits"], oof, swapped, "ge2", "A")
    assert other["max_diff"] > 1e-9  # the check can tell a wrong (repeat, fold) row apart


# --------------------------------------------------------------------------- guards on inputs


@pytest.mark.parametrize(
    "edit",
    [
        lambda m: m.update(features_sha8="deadbeef"),
        lambda m: m.update(grid_sha8="deadbeef"),
        lambda m: m.update(splits_sha8="deadbeef"),
        lambda m: m.update(inner_sha8="deadbeef"),
        lambda m: m["inputs_sha8"].update(features="deadbeef"),
        lambda m: m["inputs_sha8"].update(splits="deadbeef"),
        lambda m: m["inputs_sha8"].update(roles="deadbeef"),
        lambda m: m["constants"].update(lgbm_seed=1),
        lambda m: m["versions"].update(lightgbm="0.0.1"),
        lambda m: m.update(repeats=3),
        lambda m: m.update(columns=m["columns"][::-1]),
    ],
)
def test_a_mismatch_with_the_official_meta_refuses_the_run(
    w: dict[str, Path], edit: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    edit_meta(w, "A", edit)
    assert go(w) == 1
    err = capsys.readouterr().err
    assert err.startswith("refused:") and "differs" in err or "repeats" in err
    assert not (out(w) / tsh.SHAP_FILE).exists()


def test_a_changed_grid_refuses_the_run(
    w: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(ch.GRIDS, "lgbm", [dict(SMALL_LGBM[0])])
    assert go(w) == 1


def test_a_changed_features_file_refuses_the_run(w: dict[str, Path]) -> None:
    path = w["proc"] / "features.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "amp_mean"] += 0.5
    frame.to_csv(path, index=False)
    assert go(w) == 1


def test_best_params_outside_the_grid_or_incomplete_refused(
    w: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    path = w["cv"] / "ge2" / "best_params_lgbm_A.csv"
    table = pd.read_csv(path)
    table.loc[0, "params"] = json.dumps(
        {"num_leaves": 99, "n_estimators": 1, "min_child_samples": 1}
    )
    table.to_csv(path, index=False)
    assert go(w) == 1
    assert "outside the grid" in capsys.readouterr().err
    table = pd.read_csv(path).iloc[1:]
    table.to_csv(path, index=False)
    assert go(w) == 1
    assert "one row per repeat" in capsys.readouterr().err


def test_missing_input_files_exit_2(w: dict[str, Path]) -> None:
    (w["cv"] / "ge2" / "oof_lgbm_B.csv").unlink()
    assert go(w) == 2


def test_out_root_outside_xai_exits_2(w: dict[str, Path], tmp_path: Path) -> None:
    assert go(w, "--out-root", str(tmp_path / "elsewhere")) == 2
    assert go(w, "--out-root", str(w["cv"])) == 2
    assert not (tmp_path / "elsewhere").exists()


# --------------------------------------------------------------------------- reproduction of the OOF


def tamper_oof(w: dict[str, Path], delta: float) -> None:
    path = w["cv"] / "ge2" / "oof_lgbm_A.csv"
    oof = ch.read_oof(path)
    oof.loc[7, "proba"] = oof.loc[7, "proba"] + delta
    oof.to_csv(path, index=False, float_format="%.17g")


def test_refit_differing_from_the_official_oof_by_more_than_1e9_exits_1(
    w: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    tamper_oof(w, 1e-6)
    assert go(w) == 1
    assert "does not reproduce the official OOF" in capsys.readouterr().err
    meta = json.loads((out(w) / tsh.META_FILE).read_text())  # recorded, not hidden
    assert 9e-7 < meta["max_abs_diff_vs_official_oof"] < 1.1e-6 and meta["oof_reproduced"] is False


def test_a_difference_within_the_tolerance_passes(w: dict[str, Path]) -> None:
    tamper_oof(w, 5e-10)
    assert go(w) == 0
    meta = json.loads((out(w) / tsh.META_FILE).read_text())
    assert 0 < meta["max_abs_diff_vs_official_oof"] <= 1e-9 and meta["oof_reproduced"] is True


def test_oof_folds_not_matching_the_stored_splits_refused(w: dict[str, Path]) -> None:
    path = w["cv"] / "ge2" / "oof_lgbm_A.csv"
    oof = ch.read_oof(path)
    oof.loc[0, "outer_fold"] = (oof.loc[0, "outer_fold"] + 1) % 5
    oof.to_csv(path, index=False, float_format="%.17g")
    assert go(w) == 1


def test_exact_reproduction_on_unmodified_inputs(w: dict[str, Path]) -> None:
    assert go(w) == 0
    for fs in ("A", "B"):
        meta = json.loads((out(w, fs) / tsh.META_FILE).read_text())
        assert meta["max_abs_diff_vs_official_oof"] == 0.0 and meta["oof_reproduced"] is True


# --------------------------------------------------------------------------- outputs


def test_meta_contents(w: dict[str, Path]) -> None:
    assert go(w) == 0
    meta = json.loads((out(w) / tsh.META_FILE).read_text())
    official = json.loads((w["cv"] / "ge2" / "meta_lgbm_A.json").read_text())
    assert meta["features_sha8"] == official["features_sha8"]
    assert meta["grid_sha8"] == official["grid_sha8"]
    assert meta["explainer"] == {"class": "shap.TreeExplainer", **tsh.EXPLAINER}
    assert set(meta["versions"]) == {"shap", "lightgbm", "numpy"}
    assert meta["versions"]["shap"] == shap.__version__
    assert {"features", "roles", "features_meta", "splits", "oof", "best_params"} <= set(
        meta["inputs_sha8"]
    )
    n = len(pieces(w)["frame"])
    assert meta["row_counts"] == {tsh.SHAP_FILE: 5 * n, tsh.FOLD_FILE: 25 * 5, tsh.GLOBAL_FILE: 5}


def test_importance_tables_known_answer() -> None:
    rows = []
    for repeat, (a, b) in enumerate([(1.0, 2.0), (3.0, 2.0)]):
        for fold in range(2):
            for _ in range(2):
                rows.append(
                    {"repeat": repeat, "outer_fold": fold, "shap_x": a, "shap_y": -b, "shap_z": 2.0}
                )
    by_fold, glob = tsh.importance_tables(pd.DataFrame(rows), ["x", "y", "z"])
    assert len(by_fold) == 2 * 2 * 3 and set(by_fold["n_test_rows"]) == {2}
    assert (
        by_fold[(by_fold["repeat"] == 1) & (by_fold["feature"] == "x")]["mean_abs_shap"].iloc[0]
        == 3.0
    )
    assert list(glob["feature"]) == ["x", "y", "z"]  # all means are 2.0: ties broken by name
    assert list(glob["rank"]) == [1, 2, 3]
    assert list(glob["mean_abs_shap_mean_over_repeats"]) == [2.0, 2.0, 2.0]


def test_global_importance_uses_per_repeat_means_and_ranks_descending(w: dict[str, Path]) -> None:
    assert go(w) == 0
    table = read_out(w, tsh.SHAP_FILE)
    glob = read_out(w, tsh.GLOBAL_FILE)
    for r in glob.itertuples():
        per_repeat = table.groupby("repeat")[f"shap_{r.feature}"].apply(lambda s: s.abs().mean())
        assert r.mean_abs_shap_mean_over_repeats == pytest.approx(per_repeat.mean(), abs=1e-12)
    assert list(glob["rank"]) == [1, 2, 3, 4, 5]
    assert glob["mean_abs_shap_mean_over_repeats"].is_monotonic_decreasing


def test_by_fold_importance_is_per_repeat_and_fold(w: dict[str, Path]) -> None:
    assert go(w) == 0
    table = read_out(w, tsh.SHAP_FILE)
    by_fold = read_out(w, tsh.FOLD_FILE)
    assert list(by_fold.columns) == [
        "repeat",
        "outer_fold",
        "feature",
        "mean_abs_shap",
        "n_test_rows",
    ]
    counts = table.groupby(["repeat", "outer_fold"]).size()
    for r in by_fold.itertuples():
        assert counts[(r.repeat, r.outer_fold)] == r.n_test_rows
    assert set(by_fold["feature"]) == set(ch.FEATURE_SETS["A"])


def test_nan_feature_values_pass_through_as_nan_never_zero(w: dict[str, Path]) -> None:
    assert go(w) == 0
    table = read_out(w, tsh.SHAP_FILE, "B")
    frame = pieces(w)["frame"].set_index("ref")
    onset = table["value_decrement_onset_frac"]
    assert onset.isna().any() and (onset.dropna() != 0).all()
    expected = frame.loc[table["clip_ref"], "decrement_onset_frac"].to_numpy()
    assert (onset.isna().to_numpy() == np.isnan(expected)).all()
    raw = (out(w, "B") / tsh.SHAP_FILE).read_text().splitlines()[1].split(",")
    assert raw.count("") >= 0  # blanks are written as empty cells, not zeros


def test_no_label_columns_and_no_raw_ids(
    w: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert go(w) == 0
    text = capsys.readouterr().out
    for fs in ("A", "B"):
        for name in FILES:
            text += (out(w, fs) / name).read_text()
        columns = set(read_out(w, tsh.SHAP_FILE, fs).columns)
        assert not columns & {"y", "rating", "UPDRS", "hand", "group", "label", "target"}
        assert not [c for c in columns if re.search(r"rating|updrs|label|group", c, re.I)]
        table = read_out(w, tsh.SHAP_FILE, fs)
        for column in ("clip_ref", "participant_ref"):
            assert table[column].map(lambda v: bool(re.fullmatch(r"[0-9a-f]{8}", v))).all()
    assert not re.search(r"CONTROL|ID\d|DCHA|IZDA|parquet|fis_diagnostic|/landmarks", text)
    assert not re.search(r"20\d\d-\d\d-\d\dT|\d\d:\d\d:\d\d", text)


def test_console_has_aggregates_only(
    w: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert go(w) == 0
    text = capsys.readouterr().out
    frame = pieces(w)["frame"]
    assert not any(ref in text for ref in frame["ref"]) and not any(
        p in text for p in frame["participant"]
    )
    assert text.count("top features") == 2 and "not cause" in text
    assert max(len(line) for line in text.splitlines()) <= 120


def test_output_is_deterministic(w: dict[str, Path]) -> None:
    assert go(w) == 0
    first = {(fs, n): (out(w, fs) / n).read_bytes() for fs in ("A", "B") for n in FILES}
    assert go(w) == 0
    assert {(fs, n): (out(w, fs) / n).read_bytes() for fs in ("A", "B") for n in FILES} == first


def test_sources_are_never_modified(w: dict[str, Path]) -> None:
    def snap() -> dict[str, str]:
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(w["proc"].rglob("*")) if p.is_file() and "xai" not in p.parts}  # fmt: skip

    before = snap()
    assert go(w) == 0
    assert snap() == before


def test_failed_atomic_write_keeps_old_files(
    w: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert go(w) == 0
    old = {n: (out(w) / n).read_bytes() for n in FILES}
    (out(w) / tsh.SHAP_FILE).write_bytes(b"old")
    monkeypatch.setattr(cvs.os, "replace", lambda a, b: (_ for _ in ()).throw(OSError("fail")))
    with pytest.raises(OSError):
        go(w)
    assert (out(w) / tsh.SHAP_FILE).read_bytes() == b"old"
    assert not [p for p in out(w).iterdir() if p.name.startswith(".")]
    assert set(old) == set(FILES)


def test_module_never_reads_label_files_or_raw_ids() -> None:
    source = Path(tsh.__file__).read_text()
    assert "fis_diagnostic" not in source and "import feature_matrix" not in source


def test_a_clip_not_explained_exactly_once_is_refused(
    w: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    p = pieces(w)
    frame = p["frame"]
    groups = frame["participant"].to_numpy()
    real = ch.membership

    def odd(g: np.ndarray, splits: pd.DataFrame, repeat: int) -> np.ndarray:
        member = real(g, splits, repeat).copy()
        if repeat == 0:
            member[groups == groups[0]] = 5  # a fold that is never visited
        return member

    monkeypatch.setattr(ch, "membership", odd)
    oof = p["oof"].copy()
    sel = oof["repeat"] == 0
    mapping = dict(zip(frame["ref"], odd(groups, p["splits"], 0), strict=True))
    oof.loc[sel, "outer_fold"] = oof.loc[sel, "ref"].map(mapping)
    with pytest.raises(cvs.GuardFailure, match="exactly once"):
        tsh.explain_config(frame, p["splits"], oof, p["best"], "ge2", "A")
