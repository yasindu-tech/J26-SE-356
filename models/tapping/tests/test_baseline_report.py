"""Tests for models/tapping/src/baseline_report.py (task FT-34). The inputs are produced by running
the real cv_harness on a synthetic matrix (5 repeats, targets ge2 and ge1, b0, b1 and the b1
permutation control), then copied and damaged per test. No real data is used."""

from __future__ import annotations

import builtins
import io
import json
import re
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import baseline_report as bp  # noqa: E402
import cv_splits as cvs  # noqa: E402
from test_cv_harness import build_world, harness, make_data  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src"
ORDER = [("ge1", "b0"), ("ge1", "b1"), ("ge2", "b0"), ("ge2", "b1")]
HASH = re.compile(r"\b[0-9a-f]{8}\b")


@pytest.fixture(scope="module")
def official(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    patch = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("official")
    w = build_world(root, patch, make_data("noise", seed=4), repeats=5)
    proc = w["proc"]
    assert cvs.main(["--features", str(proc / "features.csv"), "--roles", str(proc / "feature_roles.csv"),
                     "--meta", str(proc / "features_meta.json"), "--target", "ge1", "--repeats", "5",
                     "--out-dir", str(w["cv"])]) == 0  # fmt: skip
    for target in ("ge2", "ge1"):
        for model in ("b0", "b1"):
            assert harness(w, "run", "--model", model, "--target", target, "--repeats", "5") == 0
        assert harness(w, "permute", "--model", "b1", "--target", target, "--draws", "4") == 0
    yield w
    patch.undo()


@pytest.fixture
def root(official: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    copy = tmp_path / "data" / "processed" / "cv"
    shutil.copytree(official["cv"], copy)
    monkeypatch.setattr(cvs, "CV_ROOT", copy)
    return copy


def report(root: Path, *extra: str) -> int:
    return bp.main(["--cv-root", str(root), *extra])


def table(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "reports" / "baseline_table.csv")


def edit_json(path: Path, **updates: Any) -> None:
    data = json.loads(path.read_text())
    data.update(updates)
    path.write_text(json.dumps(data))


def edit_runs(path: Path, **columns: Any) -> None:
    runs = pd.read_csv(path, float_precision="round_trip")
    for name, value in columns.items():
        runs[name] = value
    runs.to_csv(path, index=False, float_format="%.17g")


# --------------------------------------------------------------------------- the table


def test_one_row_per_target_and_model_in_fixed_order(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert report(root) == 0
    t = table(root)
    assert list(zip(t["target"], t["model"], strict=True)) == ORDER
    assert list(t.columns) == bp.TABLE_COLUMNS and (t["feature_set"] == "A").all()
    assert not [c for c in t.columns if "best" in c or "rank" in c]
    first = (root / "reports" / "baseline_table.csv").read_bytes()
    assert report(root, "--targets", "ge1", "ge2", "--models", "b1", "b0") == 0
    assert (
        root / "reports" / "baseline_table.csv"
    ).read_bytes() == first  # argument order is irrelevant
    capsys.readouterr()


def test_the_note_line_is_exact(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert report(root) == 0
    assert (
        bp.NOTE
        == "5 repeats reuse the same participants and are not independent evidence; CIs come in FT-38."
    )
    lines = capsys.readouterr().out.splitlines()
    assert bp.NOTE in lines and lines.index(bp.NOTE) > lines.index(
        next(x for x in lines if "n_repeats" in x)
    )


def test_header_reports_the_shared_hashes(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert report(root) == 0
    out = capsys.readouterr().out
    for target in ("ge2", "ge1"):
        meta = json.loads((root / target / "meta_b1_A.json").read_text())
        line = next(x for x in out.splitlines() if x.startswith(f"{target}: "))
        for key in bp.SHARED_KEYS:
            assert f"{key} {meta[key]}" in line
    t = table(root)
    assert (t["target_status"] == "pending_default_ge2").all() and t["roles_sha8"].nunique() == 1


def test_values_match_an_independent_recomputation(root: Path) -> None:
    assert report(root) == 0
    t = table(root).set_index(["target", "model"])
    for target, model in ORDER:
        runs = pd.read_csv(root / target / f"runs_{model}_A.csv", float_precision="round_trip")
        row = t.loc[(target, model)]
        assert row["n_repeats"] == 5
        for metric in bp.METRICS:
            assert row[f"{metric}_mean"] == pytest.approx(np.mean(runs[metric]), nan_ok=True)
            assert row[f"{metric}_sd"] == pytest.approx(np.std(runs[metric], ddof=1), nan_ok=True)
        assert row["auroc_fold_mean"] == pytest.approx(runs["auroc_fold_mean"].mean())
        for c in bp.PPV_COLUMNS:
            assert row[f"{c}_mean"] == pytest.approx(runs[c].mean(), nan_ok=True)
        assert (row["n_clips"], row["n_pos"], row["n_neg"]) == tuple(
            runs[["n_clips", "n_pos", "n_neg"]].iloc[0]
        )
        if model == "b1":
            perm = pd.read_csv(root / target / "permutation_b1_A.csv")["null_auroc"]
            assert row["perm_n_draws"] == 4 and row["perm_decision"] == "pass"
            assert row["perm_null_mean"] == pytest.approx(perm.mean())
            assert row["perm_null_sd"] == pytest.approx(perm.std(ddof=1))
        else:
            assert pd.isna(row["perm_n_draws"]) and pd.isna(row["perm_decision"])


def test_sd_uses_ddof_one(root: Path) -> None:
    values = [0.5, 0.6, 0.7, 0.8, 0.9]
    for model in ("b0", "b1"):
        edit_runs(root / "ge2" / f"runs_{model}_A.csv", auroc=values)
    assert report(root) == 0
    row = table(root).set_index(["target", "model"]).loc[("ge2", "b1")]
    assert row["auroc_mean"] == pytest.approx(0.7)
    assert row["auroc_sd"] == pytest.approx(np.std(values, ddof=1)) and row[
        "auroc_sd"
    ] == pytest.approx(0.1581138830)
    assert abs(row["auroc_sd"] - np.std(values, ddof=0)) > 0.01


def test_nan_is_propagated_never_zeroed(root: Path) -> None:
    path = root / "ge2" / "runs_b0_A.csv"
    runs = pd.read_csv(path, float_precision="round_trip")
    runs.loc[2, "sensitivity"] = np.nan
    runs.to_csv(path, index=False, float_format="%.17g")
    assert report(root) == 0
    cell = table(root).set_index(["target", "model"]).loc[("ge2", "b0"), "sensitivity_mean"]
    assert pd.isna(cell)
    raw = pd.read_csv(root / "reports" / "baseline_table.csv", dtype=str, keep_default_na=False)
    assert (
        raw.set_index(["target", "model"]).loc[("ge2", "b0"), "sensitivity_mean"] == ""
    )  # empty, not 0


def test_row_order_does_not_depend_on_the_results(root: Path) -> None:
    edit_runs(root / "ge1" / "runs_b0_A.csv", auroc=0.99)
    edit_runs(root / "ge1" / "runs_b1_A.csv", auroc=0.01)
    assert report(root) == 0
    t = table(root)
    assert list(zip(t["target"], t["model"], strict=True)) == ORDER


# --------------------------------------------------------------------------- refusals


def refuse_cases() -> list[Any]:
    def meta_key(key: str, value: Any) -> Any:
        def damage(r: Path) -> None:
            edit_json(r / "ge2" / "meta_b1_A.json", **{key: value})
            if key in {
                "splits_sha8",
                "inner_sha8",
                "features_sha8",
            }:  # keep runs and meta consistent,
                edit_runs(
                    r / "ge2" / "runs_b1_A.csv", **{key: value}
                )  # so only the cross-model check fires

        return damage

    def perm_csv(r: Path) -> None:
        p = r / "ge2" / "permutation_b1_A.csv"
        draws = pd.read_csv(p)
        draws["null_auroc"] = 0.9
        draws.to_csv(p, index=False)
        edit_json(r / "ge2" / "permutation_b1_A.json", mean=0.9)  # the file still says "pass"

    def drop_row(r: Path) -> None:
        p = r / "ge1" / "runs_b1_A.csv"
        pd.read_csv(p).iloc[1:].to_csv(p, index=False)

    def dup_repeat(r: Path) -> None:
        edit_runs(r / "ge1" / "runs_b0_A.csv", repeat=[0, 0, 1, 2, 3])

    def counts_between_repeats(r: Path) -> None:
        p = r / "ge2" / "runs_b0_A.csv"
        runs = pd.read_csv(p)
        runs.loc[0, "n_pos"] += 1
        runs.to_csv(p, index=False)

    return [
        ("splits_sha8", meta_key("splits_sha8", "deadbeef"), "differ in splits_sha8"),
        ("inner_sha8", meta_key("inner_sha8", "deadbeef"), "differ in inner_sha8"),
        ("roles_sha8", meta_key("roles_sha8", "deadbeef"), "differ in roles_sha8"),
        ("features_sha8", meta_key("features_sha8", "deadbeef"), "differ in features_sha8"),
        ("target_status", meta_key("target_status", "settled_ge1"), "differ in target_status"),
        ("repeats_in_meta", meta_key("repeats", 4), "meta has 4 repeats"),
        ("meta_other_target", meta_key("target", "ge1"), "another target, model or feature set"),
        ("runs_row_missing", drop_row, "runs file has 4 rows"),
        ("runs_repeat_duplicated", dup_repeat, "exactly one row per repeat"),
        (
            "runs_vs_meta",
            lambda r: edit_runs(r / "ge2" / "runs_b1_A.csv", features_sha8="deadbeef"),
            "disagrees with its meta file",
        ),
        (
            "counts_between_models",
            lambda r: edit_runs(r / "ge2" / "runs_b0_A.csv", n_pos=1),
            "differ in n_pos",
        ),
        ("counts_between_repeats", counts_between_repeats, "n_pos differs between repeats"),
        (
            "permutation_csv_missing",
            lambda r: (r / "ge2" / "permutation_b1_A.csv").unlink(),
            "permutation control files are missing",
        ),
        (
            "permutation_json_missing",
            lambda r: (r / "ge1" / "permutation_b1_A.json").unlink(),
            "permutation control files are missing",
        ),
        (
            "permutation_failed",
            lambda r: edit_json(r / "ge2" / "permutation_b1_A.json", decision="fail"),
            "did not pass",
        ),
        ("permutation_mean_above_rule", perm_csv, "did not pass"),
        (
            "permutation_draws",
            lambda r: edit_json(r / "ge2" / "permutation_b1_A.json", draws=5),
            "disagree with its summary",
        ),
        (
            "permutation_mean",
            lambda r: edit_json(r / "ge2" / "permutation_b1_A.json", mean=0.123),
            "summary mean disagrees",
        ),
    ]


@pytest.mark.parametrize("name,damage,needle", refuse_cases(), ids=[c[0] for c in refuse_cases()])
def test_refusals_exit_1_with_counts_only_and_write_nothing(
    root: Path, capsys: pytest.CaptureFixture[str], name: str, damage: Any, needle: str
) -> None:
    damage(root)
    assert report(root) == 1
    captured = capsys.readouterr()
    assert needle in captured.err and captured.err.startswith("refused:")
    assert not HASH.search(captured.err) and not captured.out  # no values, no table
    assert not (root / "reports").exists()


def test_a_refusal_leaves_an_earlier_table_untouched(root: Path) -> None:
    assert report(root) == 0
    before = (root / "reports" / "baseline_table.csv").read_bytes()
    edit_json(root / "ge2" / "meta_b1_A.json", inner_sha8="deadbeef")
    assert report(root) == 1
    assert (root / "reports" / "baseline_table.csv").read_bytes() == before


def test_missing_runs_or_meta_exit_2(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (root / "ge2" / "runs_b0_A.csv").unlink()
    assert report(root) == 2 and "missing input" in capsys.readouterr().err
    assert not (root / "reports").exists()
    assert report(root, "--targets", "ge1") == 0  # other targets are unaffected
    (root / "ge1" / "meta_b1_A.json").unlink()
    assert report(root, "--targets", "ge1") == 2


def test_paths_outside_data_processed_cv_exit_2_and_write_nothing(
    root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    outside = tmp_path / "elsewhere"
    assert report(root, "--out-dir", str(outside)) == 2
    assert (
        "--out-dir must be inside data/processed/cv" in capsys.readouterr().err
        and not outside.exists()
    )
    shutil.copytree(root, outside)
    assert report(outside) == 2
    assert "--cv-root must be inside data/processed/cv" in capsys.readouterr().err
    assert not (outside / "reports").exists()


def test_unknown_target_model_and_feature_set_are_rejected_by_the_parser(root: Path) -> None:
    for extra in (["--targets", "ge9"], ["--models", "lgbm"], ["--feature-set", "B"]):
        with pytest.raises(SystemExit) as err:
            report(root, *extra)
        assert err.value.code == 2


# --------------------------------------------------------------------------- privacy and files


def test_no_ids_in_the_outputs_and_the_oof_files_are_never_read(
    official: dict[str, Any],
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    opened: list[str] = []
    real_open = builtins.open

    def spy(file: Any, *args: Any, **kwargs: Any) -> Any:
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(io, "open", spy)
    assert report(root) == 0
    features = pd.read_csv(
        official["proc"] / "features.csv", dtype=str, usecols=["ref", "participant"]
    )
    ids = set(features["ref"]) | set(features["participant"])
    blob = capsys.readouterr().out + (root / "reports" / "baseline_table.csv").read_text()
    assert not [i for i in ids if i in blob]
    assert not [
        p
        for p in opened
        if "oof_" in p or "best_params" in p or "fis_diagnostic" in p or "splits.csv" in p
    ]
    assert any("runs_b1_A.csv" in p for p in opened)  # the spy sees reads
    assert not re.search(r"20\d\d-\d\d-\d\d|T\d\d:\d\d", blob)


def test_source_has_no_label_file_name_and_imports_neither_harness_nor_gait() -> None:
    source = (SRC / "baseline_report.py").read_text()
    assert "fis_diagnostic" not in source and "UPDRS" not in source
    imports = re.findall(r"^\s*(?:from|import)\s+([\w.]+)", source, flags=re.M)
    assert not {"cv_harness", "feature_matrix", "hubu_fis", "gait"} & {
        i.split(".")[0] for i in imports
    }


def test_rerun_is_byte_identical_and_leaves_no_temp_file(root: Path) -> None:
    assert report(root) == 0
    first = (root / "reports" / "baseline_table.csv").read_bytes()
    assert report(root) == 0
    assert (root / "reports" / "baseline_table.csv").read_bytes() == first
    assert [p.name for p in (root / "reports").iterdir()] == ["baseline_table.csv"]


def test_the_write_is_atomic(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(src: Any, dst: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(bp.os, "replace", broken)
    with pytest.raises(OSError):
        report(root)
    assert not (root / "reports" / "baseline_table.csv").exists()
    assert not list((root / "reports").glob(".*")) and not list((root / "reports").glob("*tmp*"))
    monkeypatch.undo()
    monkeypatch.setattr(cvs, "CV_ROOT", root)
    assert report(root) == 0
    before = (root / "reports" / "baseline_table.csv").read_bytes()
    monkeypatch.setattr(bp.os, "replace", broken)
    with pytest.raises(OSError):
        report(root)
    assert (root / "reports" / "baseline_table.csv").read_bytes() == before  # the old file survives
