"""Tests for models/tapping/src/model_report.py (task FT-35). The inputs come from running the real
cv_harness on a synthetic matrix (b0, b1, lgbm with a small grid, permutation controls), copied and
damaged per test. No real data is used."""

from __future__ import annotations

import builtins
import hashlib
import io
import itertools
import json
import re
import shutil
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import baseline_report as br  # noqa: E402
import cv_harness as ch  # noqa: E402
import cv_splits as cvs  # noqa: E402
import model_report as mr  # noqa: E402
from test_baseline_report import edit_json, edit_runs  # noqa: E402
from test_cv_harness import SMALL_LGBM, build_world, harness, make_data  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src"
MODELS = ("b0", "b1", "lgbm")
ORDER = [(t, m) for t in ("ge1", "ge2") for m in MODELS]
HASH = re.compile(r"\b[0-9a-f]{8}\b")
OUTPUTS = ["model_a_table.csv", "tuning_configs.csv", "tuning_edges.csv"]
REAL_LGBM_GRID = [
    {"num_leaves": nl, "n_estimators": ne, "min_child_samples": mc}
    for nl, ne, mc in itertools.product((3, 5, 7), (100, 200, 300), (20, 15, 10))
]


@pytest.fixture(scope="module")
def official(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    patch = pytest.MonkeyPatch()
    patch.setitem(ch.GRIDS, "lgbm", [dict(p) for p in SMALL_LGBM])
    w = build_world(
        tmp_path_factory.mktemp("official"), patch, make_data("noise", seed=4), repeats=5
    )
    proc = w["proc"]
    assert cvs.main(["--features", str(proc / "features.csv"), "--roles", str(proc / "feature_roles.csv"),
                     "--meta", str(proc / "features_meta.json"), "--target", "ge1", "--repeats", "5",
                     "--out-dir", str(w["cv"])]) == 0  # fmt: skip
    for target in ("ge2", "ge1"):
        for model in MODELS:
            assert harness(w, "run", "--model", model, "--target", target, "--repeats", "5") == 0
        for model in ("b1", "lgbm"):
            assert harness(w, "permute", "--model", model, "--target", target, "--draws", "4") == 0
    yield w
    patch.undo()


@pytest.fixture
def root(official: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    copy = tmp_path / "data" / "processed" / "cv"
    shutil.copytree(official["cv"], copy)
    monkeypatch.setattr(cvs, "CV_ROOT", copy)
    return copy


def report(root: Path, *extra: str) -> int:
    return mr.main(["--cv-root", str(root), *extra])


def out(root: Path) -> Path:
    return root / "reports_model_a"


def snap(folder: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}


def write_best(root: Path, target: str, model: str, chosen: list[dict[str, Any]]) -> None:
    rows = [
        {
            "repeat": i // 5,
            "outer_fold": i % 5,
            "params": json.dumps(c, sort_keys=True),
            "inner_auroc": 0.5,
        }
        for i, c in enumerate(chosen)
    ]
    pd.DataFrame(rows).to_csv(root / target / f"best_params_{model}_A.csv", index=False)


def set_grid(root: Path, target: str, model: str, grid: list[dict[str, Any]]) -> None:
    text = json.dumps({"model": model, "grid": grid}, sort_keys=True)
    edit_json(
        root / target / f"meta_{model}_A.json",
        grid=grid,
        grid_sha8=hashlib.sha256(text.encode()).hexdigest()[:8],
    )


# --------------------------------------------------------------------------- the table with lgbm


def test_the_report_accepts_lgbm_and_keeps_the_fixed_order(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert report(root) == 0
    t = pd.read_csv(out(root) / "model_a_table.csv")
    assert list(zip(t["target"], t["model"], strict=True)) == ORDER
    assert not [c for c in t.columns if "best" in c or "rank" in c]
    assert t.loc[t["model"] != "b0", "perm_decision"].eq("pass").all()
    assert t.loc[t["model"] == "b0", "perm_decision"].isna().all()
    printed = capsys.readouterr().out
    assert br.NOTE in printed.splitlines() and "Model A report" in printed
    table, _ = br.build_table(root, ["ge2", "ge1"], ["b0", "b1", "lgbm"], "A")
    assert (out(root) / "model_a_table.csv").read_bytes() == br.table_bytes(table)
    first = (out(root) / "model_a_table.csv").read_bytes()
    assert report(root, "--targets", "ge1", "ge2", "--models", "lgbm", "b1", "b0") == 0
    assert (out(root) / "model_a_table.csv").read_bytes() == first  # argument order is irrelevant


def test_the_baseline_cli_is_unchanged_and_still_rejects_lgbm(root: Path) -> None:
    with pytest.raises(SystemExit) as err:
        br.main(["--cv-root", str(root), "--models", "b0", "b1", "lgbm"])
    assert err.value.code == 2
    assert br.main(["--cv-root", str(root)]) == 0
    base = pd.read_csv(root / "reports" / "baseline_table.csv")
    assert list(zip(base["target"], base["model"], strict=True)) == [
        (t, m) for t in ("ge1", "ge2") for m in ("b0", "b1")
    ]
    assert br.TITLE.startswith("Baselines B0 and B1")


def test_values_for_lgbm_match_an_independent_recomputation(root: Path) -> None:
    assert report(root) == 0
    t = pd.read_csv(out(root) / "model_a_table.csv").set_index(["target", "model"])
    for target in ("ge2", "ge1"):
        runs = pd.read_csv(root / target / "runs_lgbm_A.csv", float_precision="round_trip")
        row = t.loc[(target, "lgbm")]
        assert row["auroc_mean"] == pytest.approx(runs["auroc"].mean())
        assert row["auroc_sd"] == pytest.approx(runs["auroc"].std(ddof=1))
        assert row["n_repeats"] == 5 and row["perm_n_draws"] == 4 and row["perm_decision"] == "pass"
        perm = pd.read_csv(root / target / "permutation_lgbm_A.csv")["null_auroc"]
        assert row["perm_null_mean"] == pytest.approx(perm.mean())


# --------------------------------------------------------------------------- refusals with three models

KEYS = ["splits_sha8", "inner_sha8", "roles_sha8", "features_sha8", "target_status"]


@pytest.mark.parametrize("model", ["b1", "lgbm"])
@pytest.mark.parametrize("key", KEYS)
def test_every_shared_value_must_agree_across_all_three_models(
    root: Path, capsys: pytest.CaptureFixture[str], model: str, key: str
) -> None:
    value = "deadbeef" if key != "target_status" else "settled_ge1"
    edit_json(root / "ge2" / f"meta_{model}_A.json", **{key: value})
    if key in {"splits_sha8", "inner_sha8", "features_sha8"}:
        edit_runs(
            root / "ge2" / f"runs_{model}_A.csv", **{key: value}
        )  # keep runs and meta consistent
    assert report(root) == 1
    captured = capsys.readouterr()
    assert f"differ in {key}" in captured.err and f"b0 and {model} " in captured.err
    assert not HASH.search(captured.err) and not captured.out and not out(root).exists()


def test_label_counts_must_agree_for_lgbm_too(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    edit_runs(root / "ge1" / "runs_lgbm_A.csv", n_pos=1)
    assert (
        report(root) == 1
        and "differ in n_pos" in capsys.readouterr().err
        and not out(root).exists()
    )


def lgbm_cases() -> list[Any]:
    def drop_row(r: Path) -> None:
        p = r / "ge2" / "runs_lgbm_A.csv"
        pd.read_csv(p).iloc[1:].to_csv(p, index=False)

    def mean_above(r: Path) -> None:
        p = r / "ge1" / "permutation_lgbm_A.csv"
        d = pd.read_csv(p)
        d["null_auroc"] = 0.9
        d.to_csv(p, index=False)
        edit_json(r / "ge1" / "permutation_lgbm_A.json", mean=0.9)

    return [
        (
            "meta_repeats",
            lambda r: edit_json(r / "ge2" / "meta_lgbm_A.json", repeats=4),
            "meta has 4 repeats",
        ),
        ("runs_row_missing", drop_row, "runs file has 4 rows"),
        (
            "runs_repeat_duplicated",
            lambda r: edit_runs(r / "ge2" / "runs_lgbm_A.csv", repeat=[0, 0, 1, 2, 3]),
            "exactly one row per repeat",
        ),
        (
            "permutation_missing",
            lambda r: (r / "ge2" / "permutation_lgbm_A.json").unlink(),
            "permutation control files are missing",
        ),
        (
            "permutation_failed",
            lambda r: edit_json(r / "ge2" / "permutation_lgbm_A.json", decision="fail"),
            "did not pass",
        ),
        ("permutation_mean_above_rule", mean_above, "did not pass"),
        (
            "permutation_draws",
            lambda r: edit_json(r / "ge1" / "permutation_lgbm_A.json", draws=9),
            "disagree with its summary",
        ),
        (
            "permutation_mean",
            lambda r: edit_json(r / "ge2" / "permutation_lgbm_A.json", mean=0.2),
            "summary mean disagrees",
        ),
    ]


@pytest.mark.parametrize("name,damage,needle", lgbm_cases(), ids=[c[0] for c in lgbm_cases()])
def test_lgbm_gets_the_same_refusals_as_b1(
    root: Path, capsys: pytest.CaptureFixture[str], name: str, damage: Any, needle: str
) -> None:
    damage(root)
    assert report(root) == 1
    captured = capsys.readouterr()
    assert needle in captured.err and "lgbm" in captured.err and not HASH.search(captured.err)
    assert not out(root).exists()


def test_a_missing_lgbm_file_exits_2(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (root / "ge2" / "runs_lgbm_A.csv").unlink()
    assert (
        report(root) == 2 and "missing input" in capsys.readouterr().err and not out(root).exists()
    )
    assert report(root, "--models", "b0", "b1") == 0  # the other models are unaffected


# --------------------------------------------------------------------------- the tuning summary


def test_tuning_counts_match_an_independent_count(root: Path) -> None:
    assert report(root) == 0
    configs = pd.read_csv(out(root) / "tuning_configs.csv")
    assert set(zip(configs["target"], configs["model"], strict=True)) == {
        (t, m) for t in ("ge1", "ge2") for m in ("b1", "lgbm")
    }
    for (target, model), part in configs.groupby(["target", "model"]):
        best = pd.read_csv(root / target / f"best_params_{model}_A.csv")
        meta = json.loads((root / target / f"meta_{model}_A.json").read_text())
        counts = Counter(json.dumps(json.loads(p), sort_keys=True) for p in best["params"])
        assert part["config"].tolist() == [
            json.dumps(g, sort_keys=True) for g in meta["grid"]
        ]  # grid order
        assert part["n_chosen"].tolist() == [counts[c] for c in part["config"]]
        assert part["n_chosen"].sum() == 25 and (part["n_configs"] == len(meta["grid"])).all()
        assert part["share_of_fits"].tolist() == pytest.approx([n / 25 for n in part["n_chosen"]])


def test_configurations_never_chosen_are_listed_with_zero(root: Path) -> None:
    write_best(root, "ge2", "b1", [{"C": 0.1}] * 25)
    assert report(root) == 0
    part = pd.read_csv(out(root) / "tuning_configs.csv").query("target == 'ge2' and model == 'b1'")
    assert part["C"].tolist() == [0.01, 0.1, 1.0, 10.0] and part["n_chosen"].tolist() == [
        0,
        25,
        0,
        0,
    ]


def test_edge_shares_for_b1(root: Path) -> None:
    write_best(
        root,
        "ge2",
        "b1",
        [{"C": 0.01}] * 5 + [{"C": 10.0}] * 5 + [{"C": 0.1}] * 10 + [{"C": 1.0}] * 5,
    )
    assert report(root) == 0
    edges = (
        pd.read_csv(out(root) / "tuning_edges.csv")
        .query("target == 'ge2' and model == 'b1'")
        .set_index("parameter")
    )
    assert edges.loc[
        "C", ["n_values", "grid_min", "grid_max", "n_at_min", "n_at_max", "n_at_edge"]
    ].tolist() == [4, 0.01, 10.0, 5, 5, 10]
    assert (
        edges.loc["C", "share_at_edge"] == pytest.approx(0.4)
        and edges.loc["any", "n_at_edge"] == 10
    )


def test_edge_shares_for_lgbm_with_the_real_grid_shape(root: Path) -> None:
    set_grid(root, "ge2", "lgbm", REAL_LGBM_GRID)
    chosen = (
        [{"num_leaves": 3, "n_estimators": 100, "min_child_samples": 20}]
        * 10  # every parameter at an edge
        + [{"num_leaves": 5, "n_estimators": 200, "min_child_samples": 15}] * 5  # none
        + [{"num_leaves": 7, "n_estimators": 200, "min_child_samples": 15}] * 5  # num_leaves only
        + [{"num_leaves": 5, "n_estimators": 300, "min_child_samples": 15}] * 5  # n_estimators only
    )
    write_best(root, "ge2", "lgbm", chosen)
    assert report(root) == 0
    configs = pd.read_csv(out(root) / "tuning_configs.csv").query(
        "target == 'ge2' and model == 'lgbm'"
    )
    assert (
        len(configs) == 27
        and configs["n_chosen"].sum() == 25
        and (configs["n_configs"] == 27).all()
    )
    assert sorted(configs.loc[configs["n_chosen"] > 0, "n_chosen"]) == [5, 5, 5, 10]
    edges = (
        pd.read_csv(out(root) / "tuning_edges.csv")
        .query("target == 'ge2' and model == 'lgbm'")
        .set_index("parameter")
    )
    assert edges.loc["num_leaves", ["n_at_min", "n_at_max", "n_at_edge"]].tolist() == [10, 5, 15]
    assert edges.loc["n_estimators", ["n_at_min", "n_at_max", "n_at_edge"]].tolist() == [10, 5, 15]
    assert edges.loc[
        "min_child_samples", ["grid_min", "grid_max", "n_at_min", "n_at_max", "n_at_edge"]
    ].tolist() == [10, 20, 0, 10, 10]
    assert edges.loc["num_leaves", "share_at_edge"] == pytest.approx(0.6) and edges.loc[
        "min_child_samples", "share_at_edge"
    ] == pytest.approx(0.4)
    assert edges.loc["any", "n_at_edge"] == 20 and edges.loc[
        "any", "share_at_edge"
    ] == pytest.approx(0.8)


def test_a_single_valued_parameter_is_never_at_an_edge(root: Path) -> None:
    assert (
        report(root) == 0
    )  # the synthetic lgbm grid has one value for n_estimators and min_child_samples
    edges = (
        pd.read_csv(out(root) / "tuning_edges.csv")
        .query("target == 'ge2' and model == 'lgbm'")
        .set_index("parameter")
    )
    for key in ("n_estimators", "min_child_samples"):
        assert (
            edges.loc[key, "n_values"] == 1
            and edges.loc[key, "n_at_edge"] == 0
            and edges.loc[key, "share_at_edge"] == 0
        )
    assert edges.loc["num_leaves", "n_values"] == 2 and edges.loc["num_leaves", "n_at_edge"] == 25
    assert edges.loc["any", "n_at_edge"] == 25  # only num_leaves can be at an edge


def test_the_tuning_summary_does_not_influence_the_table(root: Path) -> None:
    assert report(root) == 0
    before = (out(root) / "model_a_table.csv").read_bytes()
    write_best(root, "ge2", "lgbm", [SMALL_LGBM[1]] * 25)
    assert report(root) == 0
    assert (out(root) / "model_a_table.csv").read_bytes() == before
    configs = pd.read_csv(out(root) / "tuning_configs.csv").query(
        "target == 'ge2' and model == 'lgbm'"
    )
    assert configs["n_chosen"].tolist() == [0, 25]


def test_b0_has_no_tuning_rows_and_b0_only_writes_empty_tuning_files(root: Path) -> None:
    assert report(root, "--models", "b0") == 0
    assert (
        pd.read_csv(out(root) / "tuning_configs.csv").empty
        and pd.read_csv(out(root) / "tuning_edges.csv").empty
    )
    assert report(root, "--models", "b0", "b1") == 0
    assert set(pd.read_csv(out(root) / "tuning_configs.csv")["model"]) == {"b1"}


@pytest.mark.parametrize(
    "damage,needle",
    [
        ("rows", "best_params has 24 rows"),
        ("duplicate_fit", "not one row per repeat and outer fold"),
        ("not_in_grid", "3 chosen configuration(s) are not in the grid"),
        ("grid_hash", "does not match its grid_sha8"),
        ("columns", "unexpected columns"),
    ],
)
def test_tuning_inputs_are_checked(
    root: Path, capsys: pytest.CaptureFixture[str], damage: str, needle: str
) -> None:
    path = root / "ge2" / "best_params_lgbm_A.csv"
    best = pd.read_csv(path)
    if damage == "rows":
        best.iloc[1:].to_csv(path, index=False)
    elif damage == "duplicate_fit":
        best.loc[1, ["repeat", "outer_fold"]] = best.loc[0, ["repeat", "outer_fold"]].tolist()
        best.to_csv(path, index=False)
    elif damage == "not_in_grid":
        best.loc[:2, "params"] = json.dumps(
            {"num_leaves": 99, "n_estimators": 20, "min_child_samples": 5}
        )
        best.to_csv(path, index=False)
    elif damage == "grid_hash":
        edit_json(root / "ge2" / "meta_lgbm_A.json", grid=SMALL_LGBM[:1])
    else:
        best.rename(columns={"inner_auroc": "score"}).to_csv(path, index=False)
    assert report(root) == 1
    captured = capsys.readouterr()
    assert needle in captured.err and not out(root).exists()


def test_missing_best_params_exit_2(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (root / "ge1" / "best_params_b1_A.csv").unlink()
    assert (
        report(root) == 2
        and "best_params file" in capsys.readouterr().err
        and not out(root).exists()
    )


# --------------------------------------------------------------------------- files and privacy


def test_outputs_are_exactly_three_files_and_byte_identical_on_rerun(root: Path) -> None:
    assert report(root) == 0
    first = snap(out(root))
    assert sorted(first) == sorted(OUTPUTS)
    assert report(root) == 0 and snap(out(root)) == first
    assert not [p for p in out(root).iterdir() if p.name.startswith(".")]
    assert not (root / "reports").exists()  # the baseline report's folder is untouched


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
    assert (
        report(outside) == 2
        and "--cv-root must be inside data/processed/cv" in capsys.readouterr().err
    )
    assert not (outside / "reports_model_a").exists()


def test_the_write_is_atomic(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(src: Any, dst: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(br.os, "replace", broken)
    with pytest.raises(OSError):
        report(root)
    assert not list(out(root).glob("*.csv")) and not [p for p in out(root).iterdir()]


def test_no_ids_in_outputs_and_no_oof_or_label_file_is_opened(
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
    ids = set(
        pd.read_csv(
            official["proc"] / "features.csv", dtype=str, usecols=["ref", "participant"]
        ).stack()
    )
    blob = capsys.readouterr().out + "".join(p.read_text() for p in out(root).iterdir())
    assert not [i for i in ids if i in blob]
    assert not [p for p in opened if "oof_" in p or "fis_diagnostic" in p or "splits.csv" in p]
    assert any("best_params_lgbm_A.csv" in p for p in opened)
    assert not re.search(r"20\d\d-\d\d-\d\d|T\d\d:\d\d", blob)


def test_source_names_no_label_file_and_imports_neither_harness_nor_gait() -> None:
    source = (SRC / "model_report.py").read_text()
    assert "fis_diagnostic" not in source and "UPDRS" not in source
    imports = re.findall(r"^\s*(?:from|import)\s+([\w.]+)", source, flags=re.M)
    assert not {"cv_harness", "feature_matrix", "hubu_fis", "gait"} & {
        i.split(".")[0] for i in imports
    }
