"""Tests for models/tapping/src/results_table.py (task FT-41). A hand-built synthetic world of
source tables; nothing is trained and no real data is used."""

from __future__ import annotations

import builtins
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv_splits as cvs  # noqa: E402
import results_table as rt  # noqa: E402

DRAWS = rt.EXPECTED_DRAWS_SHA8
HASH = re.compile(r"\b(?=[0-9]*[a-f])[0-9a-f]{8}\b")  # a letter rules out digits of a float
GRID = {"b0": "7e846e65", "b1": "b7c060fb", "lgbm": "a2670eaf"}
FEATURES_SHA = {"A": "fc420d98", "B": "8219222e", "R": "17fb7bbf"}
SHARED = {
    "ge2": {"splits_sha8": "bcad7fb3", "inner_sha8": "9b9f183c"},
    "ge1": {"splits_sha8": "84dd55f6", "inner_sha8": "9a5de766"},
}
SUBTRACT = {
    "lgbm B minus lgbm A": (("lgbm", "B"), ("lgbm", "A")),
    "lgbm A minus b1 A": (("lgbm", "A"), ("b1", "A")),
    "lgbm A minus lgbm R": (("lgbm", "A"), ("lgbm", "R")),
    "lgbm R minus b1 R": (("lgbm", "R"), ("b1", "R")),
}


def fmt(x: float) -> str:
    return "" if x != x else format(x, ".17g")


# --------------------------------------------------------------------------- the world


def craft(seed: int = 0) -> dict[str, Any]:
    """Source tables as dicts of DataFrames/JSON, consistent with each other."""
    rng = np.random.default_rng(seed)
    w: dict[str, Any] = {"runs": {}, "meta": {}, "means": {}}
    for target in rt.TARGET_ORDER:
        for m, fs in rt.MODELS:
            rows = []
            for r in range(5):
                row = {k: fmt(float(rng.uniform(0.3, 0.9))) for k in rt.METRICS}
                for k in rt.PPV_METRICS:
                    row[k] = fmt(float(rng.uniform(0.01, 0.3)))
                if m == "b0" and target == "ge2":
                    row.update(
                        ppv_at_1pct="", ppv_at_2pct="", ppv_at_5pct=""
                    )  # no predicted positives
                row.update(
                    target=target, model=m, feature_set=fs, repeat=str(r), seed="0", threshold="0.5",
                    auroc_fold_mean="0.8", auroc_fold_sd="0.05", n_pred_pos="70", n_clips="233",
                    n_pos="63", n_neg="170", n_features="5", features_sha8=FEATURES_SHA[fs],
                    grid_sha8=GRID[m], **SHARED[target],
                )  # fmt: skip
                rows.append(row)
            frame = pd.DataFrame(rows)[rt.RUNS_COLUMNS]
            w["runs"][(target, m, fs)] = frame
            w["meta"][(target, m, fs)] = {
                "repeats": 5, "target": target, "model": m, "feature_set": fs,
                "features_sha8": FEATURES_SHA[fs], "grid_sha8": GRID[m], "roles_sha8": "4f92d429",
                "target_status": "pending_default_ge2", "participants": 117, **SHARED[target],
            }  # fmt: skip
            w["means"][(target, m, fs)] = {
                k: float(np.mean([float(v) if v != "" else np.nan for v in frame[k]]))
                for k in rt.METRICS
            }
    ci, ref, diff = [], [], []
    base = {"n_draws_finite": "2000", "n_excluded": "0", "n_boot": "2000", "seed": "2026",
            "draws_sha8": DRAWS, "roles_sha8": "4f92d429", "target_status": "pending_default_ge2"}  # fmt: skip

    def interval(e: float) -> dict[str, str]:
        if e != e:
            return {"ci_low": "", "ci_high": "", "n_draws_used": "2000"}
        return {"ci_low": format(e - 0.05, ".10g"), "ci_high": format(e + 0.05, ".10g"),
                "n_draws_used": "2000"}  # fmt: skip

    for target in rt.TARGET_ORDER:
        for m, fs in rt.CI_MODELS:
            for k in rt.METRICS:
                e = w["means"][(target, m, fs)][k]
                ci.append({"target": target, "model": m, "feature_set": fs, "metric": k,
                           "estimate": format(e, ".10g") if e == e else "nan", **interval(e),
                           "features_sha8": FEATURES_SHA[fs], **SHARED[target], **base})  # fmt: skip
        for m, fs in rt.REF_MODELS:
            for k in rt.CORE_METRICS:
                e = w["means"][(target, m, fs)][k]
                ref.append({"target": target, "section": "performance", "subject": f"{m} {fs}",
                            "metric": k, "estimate": format(e, ".10g"), **interval(e),
                            "features_sha8": FEATURES_SHA[fs], "reference_roles_sha8": "3373dae7",
                            **SHARED[target], **base})  # fmt: skip
        for name in rt.REF_DIFFS:
            a, b = SUBTRACT[name]
            e = w["means"][(target, *a)]["auroc"] - w["means"][(target, *b)]["auroc"]
            ref.append({"target": target, "section": "difference", "subject": name, "metric": "auroc",
                        "estimate": format(e, ".10g"), **interval(e), "features_sha8": "17fb7bbf",
                        "reference_roles_sha8": "3373dae7", **SHARED[target], **base})  # fmt: skip
        for name in rt.DIFF_COMPARISONS:
            a, b = SUBTRACT[name]
            e = w["means"][(target, *a)]["auroc"] - w["means"][(target, *b)]["auroc"]
            role = (
                "confirmatory"
                if (target, name) == ("ge2", "lgbm B minus lgbm A")
                else "descriptive"
            )
            for method in rt.DIFF_METHODS:
                boot = method == "participant_bootstrap"
                diff.append({
                    "target": target, "comparison": name, "role": role if boot else "descriptive",
                    "method": method, "estimate": format(e, ".10g"),
                    "ci_low": format(e - 0.03, ".10g") if boot else "",
                    "ci_high": format(e + 0.03, ".10g") if boot else "",
                    "p": "0.17" if boot else "0.4", "p_min": "" if boot else "0.03",
                    "p_max": "" if boot else "0.8", "n_draws_used": "2000" if boot else "",
                    "n_draws_finite": "2000" if boot else "", "n_excluded": "0" if boot else "",
                    "n_boot": "2000", "seed": "2026", "draws_sha8": DRAWS,
                    "roles_sha8": "4f92d429", "target_status": "pending_default_ge2",
                    **SHARED[target],
                })  # fmt: skip
    w["ci"], w["ref"], w["diff"] = map(pd.DataFrame, (ci, ref, diff))
    n = 12
    w["features"] = pd.DataFrame({
        "ref": [f"{i:08x}" for i in range(n)],
        "fps": [30.0] * 9 + [60.0] * 3,
        "qc_status": ["retained"] * 7 + ["retained_flag"] * 5,
        "n_one_sided_excluded": [0, 1, 2] * 4, "n_iti_excluded": [0] * 10 + [1, 5],
        "n_short_iti": [0] * 8 + [3, 0, 0, 20],
        "nan_reasons_b": [np.nan] * 9 + ["x:n_valid<9;y:n_intervals<8", "z:n_intervals<8", "a:no_decrement"],
        "amp_slope": [0.1] * n, "open_vel_slope": [0.1] * n, "close_vel_slope": [0.1] * n,
        "amp_min_initial": [0.2] * n, "decrement_present": [True, False] * 6,
        "decrement_onset_frac": [0.5 if i % 2 == 0 else np.nan for i in range(n)],
        "n_hesitations": [0.0] * 11 + [np.nan], "longest_pause_ratio": [0.1] * 11 + [np.nan],
        "rating": [0, 1, 2, 3] * 3,
    })  # fmt: skip
    return w


def write(w: dict[str, Any], tmp: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    cv_root = tmp / "data" / "processed" / "cv"
    monkeypatch.setattr(cvs, "CV_ROOT", cv_root)
    for (target, m, fs), frame in w["runs"].items():
        (cv_root / target).mkdir(parents=True, exist_ok=True)
        frame.to_csv(cv_root / target / f"runs_{m}_{fs}.csv", index=False)
        meta = w["meta"][(target, m, fs)]
        (cv_root / target / f"meta_{m}_{fs}.json").write_text(json.dumps(meta))
    (cv_root / "reports_ci").mkdir(parents=True, exist_ok=True)
    w["ci"].to_csv(cv_root / "reports_ci" / "ci_table.csv", index=False)
    w["ref"].to_csv(cv_root / "reports_ci" / "reference_table.csv", index=False)
    w["diff"].to_csv(cv_root / "reports_ci" / "auroc_difference.csv", index=False)
    features = tmp / "data" / "processed" / "features.csv"
    features.parent.mkdir(parents=True, exist_ok=True)
    w["features"].to_csv(features, index=False)
    return {"cv": cv_root, "features": features, "out": cv_root / "results"}


def go(paths: dict[str, Path], *extra: str) -> int:
    return rt.main(["--cv-root", str(paths["cv"]), "--features", str(paths["features"]),
                    "--out-dir", str(paths["out"]), *extra])  # fmt: skip


@pytest.fixture
def world() -> dict[str, Any]:
    return craft()


@pytest.fixture
def ok(world: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    paths = write(world, tmp_path, monkeypatch)
    assert go(paths) == 0
    return paths


def read(paths: dict[str, Path], name: str) -> pd.DataFrame:
    return pd.read_csv(paths["out"] / name, dtype=str, keep_default_na=False, comment="#")


def refused(world: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
            capsys: pytest.CaptureFixture[str], code: int = 1) -> str:  # fmt: skip
    paths = write(world, tmp_path, monkeypatch)
    assert go(paths) == code
    err = capsys.readouterr().err
    assert not (paths["out"] / rt.OUT_TABLE).exists()  # nothing written on a refusal
    return err


# --------------------------------------------------------------------------- assembly


def test_runs_csv_has_sixty_rows_in_fixed_order(ok: dict[str, Path]) -> None:
    runs = read(ok, rt.OUT_RUNS)
    assert list(runs.columns) == rt.RUNS_OUT_COLUMNS
    assert len(runs) == 60
    keys = list(
        zip(runs["target"], runs["model"], runs["feature_set"], runs["repeat"], strict=True)
    )
    expected = [(t, m, fs, str(r)) for t in ("ge2", "ge1") for m, fs in rt.MODELS for r in range(5)]
    assert keys == expected


def test_runs_values_are_copied_not_recomputed(world: dict[str, Any], ok: dict[str, Path]) -> None:
    runs = read(ok, rt.OUT_RUNS)
    src = world["runs"][("ge2", "lgbm", "B")]
    got = runs[(runs["target"] == "ge2") & (runs["model"] == "lgbm") & (runs["feature_set"] == "B")]
    assert list(got["auroc"]) == list(src["auroc"])  # same text, digit for digit
    assert set(got["roles_sha8"]) == {"4f92d429"}
    assert set(got["target_status"]) == {"pending_default_ge2"}


def test_results_table_known_answer_estimates(world: dict[str, Any], ok: dict[str, Path]) -> None:
    table = read(ok, rt.OUT_TABLE)
    assert list(table.columns) == rt.TABLE_COLUMNS
    perf = table[table["section"] == "performance"]
    assert len(perf) == 2 * 6 * 8
    row = perf[
        (perf["target"] == "ge1") & (perf["subject"] == "lgbm B") & (perf["metric"] == "mcc")
    ]
    assert float(row["estimate"].iloc[0]) == pytest.approx(
        world["means"][("ge1", "lgbm", "B")]["mcc"]
    )
    runs = world["runs"][("ge1", "lgbm", "B")]
    assert float(row["estimate"].iloc[0]) == pytest.approx(np.mean([float(v) for v in runs["mcc"]]))
    assert row["ci_source"].iloc[0] == "ci_table"
    assert row["n_draws_used"].iloc[0] == "2000"


def test_ge2_rows_come_before_ge1_and_models_in_fixed_order(ok: dict[str, Path]) -> None:
    perf = read(ok, rt.OUT_TABLE).query("section == 'performance'")
    assert list(perf["target"].unique()) == ["ge2", "ge1"]
    first = perf[perf["target"] == "ge2"]
    assert list(first["subject"].unique()) == [
        "b0 A", "b1 A", "lgbm A", "lgbm B", "b1 R", "lgbm R"
    ]  # fmt: skip
    assert rt.TARGET_ORDER == ("ge2", "ge1")
    assert list(first["metric"].iloc[:8]) == list(rt.METRICS)


def test_difference_block_follows_all_performance_rows(ok: dict[str, Path]) -> None:
    table = read(ok, rt.OUT_TABLE)
    sections = list(table["section"])
    assert sections == ["performance"] * 96 + ["difference"] * 12
    diff = table[table["section"] == "difference"]
    assert list(diff["target"].iloc[:6]) == ["ge2"] * 6


def test_reference_models_have_core_cis_and_blank_ppv_cis(ok: dict[str, Path]) -> None:
    table = read(ok, rt.OUT_TABLE)
    r = table[(table["feature_set"] == "R") & (table["section"] == "performance")]
    core = r[r["metric"].isin(rt.CORE_METRICS)]
    assert (core["ci_source"] == "reference_table").all() and (core["ci_low"] != "").all()
    ppv = r[r["metric"].isin(rt.PPV_METRICS)]
    assert (ppv["ci_low"] == "").all() and (ppv["ci_high"] == "").all()
    assert (ppv["ci_source"] == "").all() and (ppv["n_draws_used"] == "").all()
    assert (ppv["estimate"] != "").all()  # the point estimate is still reported


def test_ppv_cis_present_for_official_models(ok: dict[str, Path]) -> None:
    table = read(ok, rt.OUT_TABLE)
    a = table[(table["subject"] == "lgbm A") & (table["metric"] == "ppv_at_2pct")]
    assert (a["ci_low"] != "").all()


def test_nan_stays_blank_never_zero(ok: dict[str, Path]) -> None:
    table = read(ok, rt.OUT_TABLE)
    b0 = table[(table["target"] == "ge2") & (table["subject"] == "b0 A")]
    ppv = b0[b0["metric"].isin(rt.PPV_METRICS)]
    assert (ppv["estimate"] == "").all() and (ppv["ci_low"] == "").all()
    runs = read(ok, rt.OUT_RUNS)
    cell = runs[(runs["target"] == "ge2") & (runs["model"] == "b0")]["ppv_at_1pct"]
    assert (cell == "").all()


def test_differences_copied_unchanged_with_role_labels(
    world: dict[str, Any], ok: dict[str, Path]
) -> None:
    diff = read(ok, rt.OUT_TABLE).query("section == 'difference'")
    src = world["diff"]
    for row in diff[diff["ci_source"] == "auroc_difference"].to_dict("records"):
        s = src[(src["target"] == row["target"]) & (src["comparison"] == row["subject"])
                & (src["method"] == row["method"])].iloc[0]  # fmt: skip
        for c in ("estimate", "ci_low", "ci_high", "p", "p_min", "p_max", "role"):
            assert row[c] == s[c]
    conf = diff[diff["role"] == "confirmatory"]
    assert len(conf) == 1
    assert conf.iloc[0][["target", "subject", "method"]].tolist() == [
        "ge2",
        "lgbm B minus lgbm A",
        "participant_bootstrap",
    ]
    ref = diff[diff["ci_source"] == "reference_table"]
    assert sorted(set(ref["subject"])) == sorted(rt.REF_DIFFS)
    assert (ref["role"] == "descriptive").all()


def test_notes_text_is_in_the_table_header_and_stdout(
    ok: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    first = (ok["out"] / rt.OUT_TABLE).read_text().splitlines()[0]
    assert first == f"# {rt.NOTES}"
    assert "ge2 (rating >= 2) is the primary target" in rt.NOTES
    assert "Only ge2 lgbm B minus lgbm A is confirmatory" in rt.NOTES
    assert go(ok) == 0
    out = capsys.readouterr().out
    assert "primary target, pending the supervisor" in out.replace("\n", " ")


def test_notes_constant_is_pinned() -> None:
    assert cvs.sha8_of(rt.NOTES) == "946b3900"  # the fixed notes text; edit it only on purpose
    assert cvs.sha8_of(rt.QUALITY_HEADER) == "978a8a17"
    assert rt.NOTES.endswith("No row implies a ranking.")
    assert "B 2000, seed 2026" in rt.NOTES and "0.48" in rt.NOTES


def test_meta_json_records_inputs_hashes_and_counts(ok: dict[str, Path]) -> None:
    meta = json.loads((ok["out"] / rt.OUT_META).read_text())
    assert meta["repeats"] == 5 and meta["draws_sha8"] == DRAWS and meta["n_draws_used"] == "2000"
    assert meta["row_counts"] == {rt.OUT_RUNS: 60, rt.OUT_TABLE: 108, rt.OUT_QUALITY: 16}
    assert len(meta["inputs_sha8"]) == 2 * 6 * 2 + 3 + 1
    assert all(re.fullmatch(r"[0-9a-f]{8}", v) for v in meta["inputs_sha8"].values())
    assert meta["shared"]["ge2"]["splits_sha8"] == "bcad7fb3"
    assert meta["notes_sha8"] == cvs.sha8_of(rt.NOTES)


# --------------------------------------------------------------------------- detector quality


def test_detector_quality_counts_against_hand_built_features(ok: dict[str, Path]) -> None:
    q = read(ok, rt.OUT_QUALITY)
    assert list(q.columns) == rt.QUALITY_COLUMNS
    got = dict(zip(q["item"], q["n_clips"], strict=True))
    assert got["clips_total"] == "12"
    assert got["clips_above_45_fps"] == "3"
    assert got["qc_retained_flag"] == "5"
    assert got["one_sided_taps_excluded"] == "8"
    assert got["intervals_excluded"] == "2"
    assert got["short_intervals_under_0.2s"] == "2"
    assert got["set_b_unavailable_n_valid_lt_9"] == "1"
    assert got["set_b_unavailable_n_intervals_lt_8"] == "2"
    assert got["decrement_onset_frac"] == "6"
    assert got["n_hesitations"] == "1" and got["longest_pause_ratio"] == "1"
    assert got["amp_slope"] == "0" and got["decrement_present"] == "0"
    assert set(q["n_total"]) == {"12"}


def test_detector_quality_header_states_it_is_context_only(ok: dict[str, Path]) -> None:
    first = (ok["out"] / rt.OUT_QUALITY).read_text().splitlines()[0]
    assert first == f"# {rt.QUALITY_HEADER}"
    assert "NOT used to select, drop or reweight" in first


def test_features_ratings_are_ignored_and_labels_never_opened(
    world: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = write(world, tmp_path, monkeypatch)
    assert go(paths) == 0
    before = {
        n: (paths["out"] / n).read_bytes() for n in (rt.OUT_RUNS, rt.OUT_TABLE, rt.OUT_QUALITY)
    }
    shuffled = world["features"].copy()
    shuffled["rating"] = [3, 2, 1, 0] * 3
    shuffled.to_csv(paths["features"], index=False)
    opened: list[str] = []
    real = builtins.open

    def spy(file: Any, *a: Any, **k: Any) -> Any:
        opened.append(str(file))
        return real(file, *a, **k)

    monkeypatch.setattr(builtins, "open", spy)
    assert go(paths) == 0
    for n, data in before.items():
        assert (paths["out"] / n).read_bytes() == data
    assert not any("fis_diagnostic" in p or "UPDRS" in p for p in opened)
    assert "rating" not in rt.FEATURE_COLUMNS and "UPDRS" not in rt.FEATURE_COLUMNS


def test_missing_feature_column_is_refused_with_exit_2(
    world: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world["features"] = world["features"].drop(columns=["n_short_iti"])
    paths = write(world, tmp_path, monkeypatch)
    assert go(paths) == 1
    assert not (paths["out"] / rt.OUT_TABLE).exists()


# --------------------------------------------------------------------------- refusals


def test_missing_model_file_is_exit_2(
    world: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = write(world, tmp_path, monkeypatch)
    (paths["cv"] / "ge1" / "runs_lgbm_R.csv").unlink()
    assert go(paths) == 2
    assert "missing input" in capsys.readouterr().err


def test_missing_source_table_is_exit_2(
    world: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = write(world, tmp_path, monkeypatch)
    (paths["cv"] / "reports_ci" / "reference_table.csv").unlink()
    assert go(paths) == 2


def test_wrong_repeat_count_refused(world, tmp_path, monkeypatch, capsys):  # type: ignore[no-untyped-def]
    key = ("ge2", "b1", "A")
    world["runs"][key] = world["runs"][key].iloc[:4]
    assert "expected repeats 0-4" in refused(world, tmp_path, monkeypatch, capsys)


def test_duplicated_repeat_refused(world, tmp_path, monkeypatch, capsys):  # type: ignore[no-untyped-def]
    key = ("ge1", "lgbm", "A")
    frame = world["runs"][key].copy()
    frame.loc[4, "repeat"] = "3"
    world["runs"][key] = frame
    assert "expected repeats" in refused(world, tmp_path, monkeypatch, capsys)


def test_meta_repeats_not_five_refused(world, tmp_path, monkeypatch, capsys):  # type: ignore[no-untyped-def]
    world["meta"][("ge2", "b0", "A")]["repeats"] = 3
    assert "repeats" in refused(world, tmp_path, monkeypatch, capsys)


@pytest.mark.parametrize("key", ["splits_sha8", "inner_sha8", "roles_sha8", "target_status"])
def test_shared_hash_differing_across_models_refused(world, tmp_path, monkeypatch, capsys, key):  # type: ignore[no-untyped-def]
    world["meta"][("ge2", "lgbm", "R")][key] = "deadbeef"
    if key in ("splits_sha8", "inner_sha8"):
        world["runs"][("ge2", "lgbm", "R")][key] = "deadbeef"
    assert f"{key} differs across models" in refused(world, tmp_path, monkeypatch, capsys)


def test_runs_hash_differs_from_meta_refused(world, tmp_path, monkeypatch, capsys):  # type: ignore[no-untyped-def]
    world["runs"][("ge2", "b1", "R")]["features_sha8"] = "deadbeef"
    assert "features_sha8 in runs differs from meta" in refused(
        world, tmp_path, monkeypatch, capsys
    )


def test_grid_sha8_differs_from_official_refused(world, tmp_path, monkeypatch, capsys):  # type: ignore[no-untyped-def]
    key = ("ge1", "lgbm", "R")
    world["meta"][key]["grid_sha8"] = "deadbeef"
    world["runs"][key]["grid_sha8"] = (
        "deadbeef"  # consistent with its own meta, not the official run
    )
    assert "grid_sha8 differs from the official lgbm run" in refused(
        world, tmp_path, monkeypatch, capsys
    )


def test_grid_sha8_runs_vs_own_meta_refused(world, tmp_path, monkeypatch, capsys):  # type: ignore[no-untyped-def]
    world["runs"][("ge2", "lgbm", "B")]["grid_sha8"] = "deadbeef"
    assert "grid_sha8 in runs differs from meta" in refused(world, tmp_path, monkeypatch, capsys)


@pytest.mark.parametrize("table", ["ci", "ref"])
def test_stale_source_estimate_refused(world, tmp_path, monkeypatch, capsys, table):  # type: ignore[no-untyped-def]
    frame = world[table].copy()
    frame.loc[3, "estimate"] = format(float(frame.loc[3, "estimate"]) + 0.01, ".10g")
    world[table] = frame
    assert "differs from the mean of the runs" in refused(world, tmp_path, monkeypatch, capsys)


def test_stale_difference_estimate_refused(world, tmp_path, monkeypatch, capsys):  # type: ignore[no-untyped-def]
    frame = world["diff"].copy()
    frame.loc[0, "estimate"] = "0.9"
    world["diff"] = frame
    assert "auroc_difference" in refused(world, tmp_path, monkeypatch, capsys)


def test_stale_reference_difference_refused(world, tmp_path, monkeypatch, capsys):  # type: ignore[no-untyped-def]
    frame = world["ref"].copy()
    i = frame.index[frame["section"] == "difference"][0]
    frame.loc[i, "estimate"] = "0.5"
    world["ref"] = frame
    assert "reference_table" in refused(world, tmp_path, monkeypatch, capsys)


@pytest.mark.parametrize("table", ["ci", "ref", "diff"])
def test_draws_sha8_disagreement_refused(world, tmp_path, monkeypatch, capsys, table):  # type: ignore[no-untyped-def]
    frame = world[table].copy()
    frame.loc[1, "draws_sha8"] = "0badf00d"
    world[table] = frame
    assert "draws_sha8" in refused(world, tmp_path, monkeypatch, capsys)


@pytest.mark.parametrize("table", ["ci", "ref"])
def test_n_draws_used_disagreement_refused(world, tmp_path, monkeypatch, capsys, table):  # type: ignore[no-untyped-def]
    frame = world[table].copy()
    frame.loc[0, "n_draws_used"] = "1990"
    world[table] = frame
    assert "n_draws_used" in refused(world, tmp_path, monkeypatch, capsys)


def test_source_table_shared_hash_stale_refused(world, tmp_path, monkeypatch, capsys):  # type: ignore[no-untyped-def]
    frame = world["ci"].copy()
    frame.loc[5, "splits_sha8"] = "deadbeef"
    world["ci"] = frame
    assert "splits_sha8 differs from the runs" in refused(world, tmp_path, monkeypatch, capsys)


def test_missing_source_row_refused(world, tmp_path, monkeypatch, capsys):  # type: ignore[no-untyped-def]
    world["ci"] = world["ci"].iloc[1:].reset_index(drop=True)
    assert "row missing" in refused(world, tmp_path, monkeypatch, capsys)


def test_duplicate_source_row_refused(world, tmp_path, monkeypatch, capsys):  # type: ignore[no-untyped-def]
    world["ci"] = pd.concat([world["ci"], world["ci"].iloc[[0]]], ignore_index=True)
    assert "appears twice" in refused(world, tmp_path, monkeypatch, capsys)


def test_nan_in_runs_against_zero_in_table_refused(world, tmp_path, monkeypatch, capsys):  # type: ignore[no-untyped-def]
    frame = world["ci"].copy()
    mask = (
        (frame["target"] == "ge2") & (frame["model"] == "b0") & (frame["metric"] == "ppv_at_1pct")
    )
    frame.loc[mask, "estimate"] = "0"  # a NaN turned into zero upstream
    world["ci"] = frame
    assert "differs from the mean of the runs" in refused(world, tmp_path, monkeypatch, capsys)


def test_zero_in_runs_against_blank_table_refused(world, tmp_path, monkeypatch, capsys):  # type: ignore[no-untyped-def]
    frame = world["ci"].copy()
    mask = (frame["target"] == "ge1") & (frame["model"] == "b1") & (frame["metric"] == "mcc")
    frame.loc[mask, "estimate"] = "nan"
    world["ci"] = frame
    assert "differs from the mean of the runs" in refused(world, tmp_path, monkeypatch, capsys)


def test_non_numeric_metric_refused(world, tmp_path, monkeypatch, capsys):  # type: ignore[no-untyped-def]
    frame = world["runs"][("ge2", "b1", "A")].copy()
    frame.loc[0, "auroc"] = "abc"
    world["runs"][("ge2", "b1", "A")] = frame
    assert "not numeric" in refused(world, tmp_path, monkeypatch, capsys)


def test_confirmatory_label_dropped_refused(world, tmp_path, monkeypatch, capsys):  # type: ignore[no-untyped-def]
    world["diff"]["role"] = "descriptive"
    assert "confirmatory label" in refused(world, tmp_path, monkeypatch, capsys)


def test_confirmatory_label_on_another_row_refused(world, tmp_path, monkeypatch, capsys):  # type: ignore[no-untyped-def]
    frame = world["diff"].copy()
    frame.loc[frame["target"] == "ge1", "role"] = "confirmatory"
    world["diff"] = frame
    assert "confirmatory label" in refused(world, tmp_path, monkeypatch, capsys)


def test_refusal_messages_hold_no_ids_or_values(world, tmp_path, monkeypatch, capsys):  # type: ignore[no-untyped-def]
    frame = world["ci"].copy()
    frame.loc[3, "estimate"] = "0.123456"
    world["ci"] = frame
    err = refused(world, tmp_path, monkeypatch, capsys)
    assert "0.123456" not in err


# --------------------------------------------------------------------------- outputs and guards


def test_rerun_is_byte_identical(
    world: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = write(world, tmp_path, monkeypatch)
    names = (rt.OUT_RUNS, rt.OUT_TABLE, rt.OUT_QUALITY, rt.OUT_META)
    assert go(paths) == 0
    first = {n: (paths["out"] / n).read_bytes() for n in names}
    assert go(paths) == 0
    assert {n: (paths["out"] / n).read_bytes() for n in names} == first


def test_sources_are_not_modified(
    world: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = write(world, tmp_path, monkeypatch)
    snap = {p: p.read_bytes() for p in paths["cv"].rglob("*") if p.is_file()}
    snap[paths["features"]] = paths["features"].read_bytes()
    assert go(paths) == 0
    assert {p: p.read_bytes() for p in snap} == snap


def test_atomic_write_failed_replace_keeps_old_files(
    world: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = write(world, tmp_path, monkeypatch)
    paths["out"].mkdir(parents=True)
    old = {
        n: b"old " + n.encode() for n in (rt.OUT_RUNS, rt.OUT_TABLE, rt.OUT_QUALITY, rt.OUT_META)
    }
    for n, data in old.items():
        (paths["out"] / n).write_bytes(data)

    def boom(src: Any, dst: Any) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(rt.os, "replace", boom)
    with pytest.raises(OSError):
        go(paths)
    assert {n: (paths["out"] / n).read_bytes() for n in old} == old
    assert not [p for p in paths["out"].iterdir() if p.name.startswith(".")]  # no temp files left


def test_out_dir_default_is_results_inside_cv(
    world: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = write(world, tmp_path, monkeypatch)
    assert rt.main(["--cv-root", str(paths["cv"]), "--features", str(paths["features"])]) == 0
    assert (paths["cv"] / "results" / rt.OUT_TABLE).is_file()


def test_out_dir_outside_cv_is_exit_2(
    world: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = write(world, tmp_path, monkeypatch)
    paths["out"] = tmp_path / "elsewhere"
    assert go(paths) == 2
    assert "inside data/processed/cv" in capsys.readouterr().err
    assert not paths["out"].exists()


def test_out_dir_traversal_is_exit_2(
    world: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = write(world, tmp_path, monkeypatch)
    paths["out"] = paths["cv"] / ".." / ".." / "escaped"
    assert go(paths) == 2


def test_cv_root_outside_is_exit_2(
    world: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = write(world, tmp_path, monkeypatch)
    other = tmp_path / "other"
    other.mkdir()
    paths["cv"] = other
    assert go(paths) == 2


def test_out_dir_inside_cv_subfolder_is_allowed(ok: dict[str, Path]) -> None:
    ok["out"] = ok["cv"] / "copy_here"
    assert go(ok) == 0 and (ok["out"] / rt.OUT_META).is_file()


# --------------------------------------------------------------------------- wording and privacy


FORBIDDEN = re.compile(r"\b(best|rank|winner|outperform\w*|detects?|better|worse|delta)\b", re.I)


def test_no_ranking_or_claim_wording_or_columns(
    ok: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert go(ok) == 0
    texts = [capsys.readouterr().out.replace(rt.NOTES, "")]
    for name in (rt.OUT_RUNS, rt.OUT_TABLE, rt.OUT_QUALITY, rt.OUT_META):
        texts.append((ok["out"] / name).read_text().replace(rt.NOTES, ""))
    for text in texts:
        assert not FORBIDDEN.search(text)
    for name in (rt.OUT_RUNS, rt.OUT_TABLE, rt.OUT_QUALITY):
        cols = read(ok, name).columns
        assert not [c for c in cols if re.search(r"best|rank|winner|delta", c)]


def test_stdout_lines_are_at_most_100_characters(
    ok: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert go(ok) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines and max(len(line) for line in lines) <= 100


def test_stdout_prints_each_block_once(
    ok: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert go(ok) == 0
    out = capsys.readouterr().out
    assert out.count("AUROC [95% CI]") == 2  # once per target
    assert out.count("AUROC differences") == 1
    assert out.index("== ge2") < out.index("== ge1")
    assert out.count("Notes:") == 1


def test_console_ppv_blank_for_nan(ok: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    assert go(ok) == 0
    out = capsys.readouterr().out.split("== ge1")[0]
    assert re.search(r"b0 A\s+-\s+-\s+-", out)


def test_outputs_hold_no_ids_no_timestamps_and_only_known_hashes(
    ok: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert go(ok) == 0
    text = capsys.readouterr().out
    for name in (rt.OUT_RUNS, rt.OUT_TABLE, rt.OUT_QUALITY, rt.OUT_META):
        text += (ok["out"] / name).read_text()
    known = set(FEATURES_SHA.values()) | set(GRID.values()) | {DRAWS, "4f92d429"}
    known |= {v for s in SHARED.values() for v in s.values()}
    found = set(HASH.findall(text))
    # any other 8-hex token would be an id or an input digest; input digests are in the meta only
    meta = json.loads((ok["out"] / rt.OUT_META).read_text())
    known |= set(meta["inputs_sha8"].values()) | {meta["notes_sha8"]}
    assert found <= known
    assert not re.search(r"20\d\d-\d\d-\d\d|T\d\d:\d\d", text)
    assert not re.search(r"CONTROL|ID\d|DCHA|IZDA|parquet", text)


def test_module_never_imports_label_loaders() -> None:
    source = (Path(rt.__file__)).read_text()
    assert "import feature_matrix" not in source and "fis_diagnostic" not in source
    assert "signal" not in Path(rt.__file__).name


@pytest.mark.parametrize("column", ["target", "model", "feature_set"])
def test_runs_identity_column_differing_from_file_refused(
    world, tmp_path, monkeypatch, capsys, column
):  # type: ignore[no-untyped-def]
    frame = world["runs"][("ge2", "b1", "A")].copy()
    frame[column] = "zz"
    world["runs"][("ge2", "b1", "A")] = frame
    assert f"{column} differs between runs and meta" in refused(
        world, tmp_path, monkeypatch, capsys
    )


@pytest.mark.parametrize("table", ["ci", "ref"])
def test_source_features_sha8_differing_from_runs_refused(
    world, tmp_path, monkeypatch, capsys, table
):  # type: ignore[no-untyped-def]
    frame = world[table].copy()
    frame.loc[frame.index[0], "features_sha8"] = "deadbeef"
    world[table] = frame
    assert "features_sha8 differs from the runs" in refused(world, tmp_path, monkeypatch, capsys)


def test_ci_bounds_are_copied_from_the_source_tables(
    world: dict[str, Any], ok: dict[str, Path]
) -> None:
    perf = read(ok, rt.OUT_TABLE).query("section == 'performance' and ci_low != ''")
    for row in perf.to_dict("records"):
        if row["feature_set"] == "R":
            s = world["ref"].query("target == @row['target'] and subject == @row['subject']")
        else:
            s = world["ci"].query(
                "target == @row['target'] and model == @row['model']"
                " and feature_set == @row['feature_set']"
            )
        s = s[s["metric"] == row["metric"]].iloc[0]
        assert (row["ci_low"], row["ci_high"]) == (s["ci_low"], s["ci_high"])
        assert float(row["ci_low"]) < float(row["estimate"]) < float(row["ci_high"])
