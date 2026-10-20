"""Tests for models/tapping/src/ab_report.py (task FT-37). The inputs come from running the real
cv_harness on a synthetic matrix (lgbm on Set A and Set B with a small grid, permutation controls),
copied and damaged per test. No real data is used."""

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

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ab_report as ab  # noqa: E402
import baseline_report as br  # noqa: E402
import cv_harness as ch  # noqa: E402
import cv_splits as cvs  # noqa: E402
from test_baseline_report import edit_json, edit_runs  # noqa: E402
from test_cv_harness import SMALL_LGBM, build_world, harness, make_data  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src"
ORDER = [(t, fs) for t in ("ge1", "ge2") for fs in ("A", "B")]
HASH = re.compile(r"\b[0-9a-f]{8}\b")
OUTPUTS = ["model_b_table.csv", "tuning_configs.csv", "tuning_edges.csv"]
REAL_GRID = [
    {"num_leaves": nl, "n_estimators": ne, "min_child_samples": mc}
    for nl, ne, mc in itertools.product((3, 5, 7), (100, 200, 300), (20, 15, 10))
]
A_COLS, B_COLS = list(cvs.SET_A_PRIMARY), list(cvs.SET_A_PRIMARY) + list(cvs.SET_B_ADDED)


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
        for fs in ("A", "B"):
            assert (
                harness(
                    w,
                    "run",
                    "--model",
                    "lgbm",
                    "--feature-set",
                    fs,
                    "--target",
                    target,
                    "--repeats",
                    "5",
                )
                == 0
            )
            assert (
                harness(
                    w,
                    "permute",
                    "--model",
                    "lgbm",
                    "--feature-set",
                    fs,
                    "--target",
                    target,
                    "--draws",
                    "4",
                )
                == 0
            )
    yield w
    patch.undo()


@pytest.fixture
def root(official: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    copy = tmp_path / "data" / "processed" / "cv"
    shutil.copytree(official["cv"], copy)
    monkeypatch.setattr(cvs, "CV_ROOT", copy)
    monkeypatch.setattr(
        ab, "EXPECTED_GRID_SIZE", len(SMALL_LGBM)
    )  # the synthetic grid has 2 configurations
    return copy


def report(root: Path, *extra: str) -> int:
    return ab.main(["--cv-root", str(root), *extra])


def out(root: Path) -> Path:
    return root / "reports_model_b"


def snap(folder: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}


def table(root: Path) -> pd.DataFrame:
    return pd.read_csv(out(root) / "model_b_table.csv")


def meta_path(root: Path, target: str, fs: str) -> Path:
    return root / target / f"meta_lgbm_{fs}.json"


def runs_path(root: Path, target: str, fs: str) -> Path:
    return root / target / f"runs_lgbm_{fs}.csv"


def read_meta(root: Path, target: str, fs: str) -> dict[str, Any]:
    return json.loads(meta_path(root, target, fs).read_text())


def set_columns(root: Path, target: str, fs: str, columns: list[str]) -> None:
    """Give a feature set another column list, keeping its hashes and n_features consistent."""
    sha = hashlib.sha256(json.dumps(columns).encode()).hexdigest()[:8]
    edit_json(meta_path(root, target, fs), columns=columns, features_sha8=sha)
    edit_runs(runs_path(root, target, fs), features_sha8=sha, n_features=len(columns))


def set_grid(
    root: Path, target: str, grid: list[dict[str, Any]], sets: tuple[str, ...] = ("A", "B")
) -> None:
    sha = hashlib.sha256(
        json.dumps({"model": "lgbm", "grid": grid}, sort_keys=True).encode()
    ).hexdigest()[:8]
    for fs in sets:
        edit_json(meta_path(root, target, fs), grid=grid, grid_sha8=sha)
        edit_runs(runs_path(root, target, fs), grid_sha8=sha)


def write_best(root: Path, target: str, chosen: list[dict[str, Any]]) -> None:
    rows = [
        {
            "repeat": i // 5,
            "outer_fold": i % 5,
            "params": json.dumps(c, sort_keys=True),
            "inner_auroc": 0.5,
        }
        for i, c in enumerate(chosen)
    ]
    pd.DataFrame(rows).to_csv(root / target / "best_params_lgbm_B.csv", index=False)


# --------------------------------------------------------------------------- the table


def test_one_row_per_target_and_feature_set_in_fixed_order(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert report(root) == 0
    t = table(root)
    assert list(zip(t["target"], t["feature_set"], strict=True)) == ORDER
    assert list(t.columns) == ab.TABLE_COLUMNS and (t["model"] == "lgbm").all()
    assert t["n_features"].tolist() == [5, 13, 5, 13] and (t["n_repeats"] == 5).all()
    assert not [c for c in t.columns if re.search(r"delta|diff|best|rank|gain", c)]
    first = (out(root) / "model_b_table.csv").read_bytes()
    assert (
        report(root, "--targets", "ge1", "ge2") == 0
        and (out(root) / "model_b_table.csv").read_bytes() == first
    )
    capsys.readouterr()


def test_the_note_and_the_header(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert report(root) == 0
    out_text = capsys.readouterr().out
    lines = out_text.splitlines()
    assert (
        ab.NOTE
        == "5 repeats reuse the same participants and are not independent evidence; the A-vs-B difference and its CI come in FT-39."
    )
    assert ab.NOTE in lines and lines.index(ab.NOTE) > next(
        i for i, x in enumerate(lines) if "auroc_mean" in x
    )
    for target in ("ge2", "ge1"):
        a, b = read_meta(root, target, "A"), read_meta(root, target, "B")
        line = next(x for x in lines if x.startswith(f"{target}: "))
        for key in ab.PAIR_KEYS:
            assert f"{key} {a[key]}" in line
        assert f"features_sha8 A {a['features_sha8']}, B {b['features_sha8']}" in line
    assert not re.search(r"delta|difference of|A minus B|B minus A", out_text, re.I)


def test_the_printed_table_has_separated_columns_and_integer_counts(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert report(root) == 0
    lines = capsys.readouterr().out.splitlines()
    start = next(i for i, x in enumerate(lines) if "auroc_mean" in x)
    for line in lines[start : start + 5]:
        assert len(re.split(r"\s{2,}", line.strip())) == len(ab.SHOWN)  # every column separated
        assert not re.search(r"\d\.\d{4}\d", line)  # no run-together numbers like 0.00000.5000
    assert not re.search(r"\b4\.0000\b", lines[start + 1])  # the draw count prints as 4, not 4.0000


def test_values_match_an_independent_recomputation(root: Path) -> None:
    assert report(root) == 0
    t = table(root).set_index(["target", "feature_set"])
    for target, fs in ORDER:
        runs = pd.read_csv(runs_path(root, target, fs), float_precision="round_trip")
        row = t.loc[(target, fs)]
        for metric in ab.METRICS:
            assert row[f"{metric}_mean"] == pytest.approx(np.mean(runs[metric]), nan_ok=True)
            assert row[f"{metric}_sd"] == pytest.approx(np.std(runs[metric], ddof=1), nan_ok=True)
        assert row["n_pred_pos_mean"] == pytest.approx(runs["n_pred_pos"].mean())
        assert row["n_pred_pos_sd"] == pytest.approx(runs["n_pred_pos"].std(ddof=1))
        perm = pd.read_csv(root / target / f"permutation_lgbm_{fs}.csv")["null_auroc"]
        assert row["perm_n_draws"] == 4 and row["perm_decision"] == "pass"
        assert row["perm_null_mean"] == pytest.approx(perm.mean()) and row[
            "perm_null_sd"
        ] == pytest.approx(perm.std(ddof=1))
        assert row["features_sha8"] == read_meta(root, target, fs)["features_sha8"]
        assert (row["n_clips"], row["n_pos"], row["n_neg"]) == tuple(
            runs[["n_clips", "n_pos", "n_neg"]].iloc[0]
        )


def test_nan_stays_nan_and_is_written_empty(root: Path) -> None:
    runs = pd.read_csv(runs_path(root, "ge2", "B"), float_precision="round_trip")
    runs.loc[1, "sensitivity"] = np.nan
    runs.to_csv(runs_path(root, "ge2", "B"), index=False, float_format="%.17g")
    assert report(root) == 0
    raw = pd.read_csv(out(root) / "model_b_table.csv", dtype=str, keep_default_na=False).set_index(
        ["target", "feature_set"]
    )
    assert (
        raw.loc[("ge2", "B"), "sensitivity_mean"] == ""
        and raw.loc[("ge2", "A"), "sensitivity_mean"] != ""
    )


# --------------------------------------------------------------------------- pair refusals


def pair_cases() -> list[Any]:
    def meta_b(**updates: Any) -> Any:
        return lambda r: edit_json(meta_path(r, "ge2", "B"), **updates)

    def meta_and_runs(key: str, value: Any) -> Any:
        def damage(r: Path) -> None:
            edit_json(meta_path(r, "ge2", "B"), **{key: value})
            edit_runs(runs_path(r, "ge2", "B"), **{key: value})

        return damage

    def constants(path: tuple[str, ...], value: Any) -> Any:
        def damage(r: Path) -> None:
            meta = read_meta(r, "ge2", "B")
            node = meta["constants"]
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = value
            meta_path(r, "ge2", "B").write_text(json.dumps(meta))

        return damage

    def other_grid(r: Path) -> None:
        set_grid(r, "ge2", [dict(SMALL_LGBM[0], num_leaves=9), dict(SMALL_LGBM[1])], sets=("B",))

    def copy_a_as_b(r: Path) -> None:
        for name in (
            "runs_lgbm_{}.csv",
            "meta_lgbm_{}.json",
            "best_params_lgbm_{}.csv",
            "permutation_lgbm_{}.csv",
            "permutation_lgbm_{}.json",
        ):
            shutil.copy(r / "ge2" / name.format("A"), r / "ge2" / name.format("B"))
        edit_json(meta_path(r, "ge2", "B"), feature_set="B")
        edit_runs(runs_path(r, "ge2", "B"), feature_set="B")

    def doctored_hash(r: Path) -> None:
        edit_json(meta_path(r, "ge2", "B"), features_sha8="deadbeef")
        edit_runs(runs_path(r, "ge2", "B"), features_sha8="deadbeef")

    return [
        ("splits_sha8", meta_and_runs("splits_sha8", "deadbeef"), "differ in splits_sha8"),
        ("inner_sha8", meta_and_runs("inner_sha8", "deadbeef"), "differ in inner_sha8"),
        ("roles_sha8", meta_b(roles_sha8="deadbeef"), "differ in roles_sha8"),
        ("target_status", meta_b(target_status="settled_ge1"), "differ in target_status"),
        ("seed", meta_b(seed=1), "differ in seed"),
        ("grid_sha8", other_grid, "differ in grid_sha8"),
        ("lgbm_fixed", constants(("lgbm_fixed", "learning_rate"), 0.1), "fixed LightGBM settings"),
        ("lgbm_seed", constants(("lgbm_seed",), 1), "fixed LightGBM settings"),
        ("threshold", constants(("threshold",), 0.4), "fixed LightGBM settings"),
        ("n_inner", constants(("n_inner",), 5), "fixed LightGBM settings"),
        ("label_counts", lambda r: edit_runs(runs_path(r, "ge2", "B"), n_pos=1), "differ in n_pos"),
        ("features_sha8_equal", copy_a_as_b, "must differ in features_sha8"),
        (
            "n_features_not_greater",
            lambda r: set_columns(r, "ge2", "B", A_COLS[:4]),
            "B has 4 features, not more than A's 5",
        ),
        (
            "n_features_equal",
            lambda r: set_columns(r, "ge2", "B", A_COLS[::-1]),
            "B has 5 features, not more than A's 5",
        ),
        (
            "a_not_subset_of_b",
            lambda r: set_columns(r, "ge2", "B", [*A_COLS[1:], *cvs.SET_B_ADDED, "n_taps"]),
            "not a subset",
        ),
        (
            "not_the_registered_set_b",
            lambda r: set_columns(r, "ge2", "B", [*B_COLS[:-1], "n_taps"]),
            "differ from the registered Set A and Set B",
        ),
        (
            "not_the_registered_set_a",
            lambda r: set_columns(r, "ge2", "A", A_COLS[:4] + ["amp_slope"]),
            "differ from the registered Set A and Set B",
        ),
        ("features_hash_doctored", doctored_hash, "does not match its column list"),
        (
            "n_features_vs_columns",
            lambda r: edit_runs(runs_path(r, "ge2", "B"), n_features=99),
            "n_features disagrees",
        ),
        (
            "runs_grid_hash",
            lambda r: edit_runs(runs_path(r, "ge2", "A"), grid_sha8="deadbeef"),
            "disagrees with its meta file on grid_sha8",
        ),
        (
            "grid_vs_hash",
            lambda r: edit_json(meta_path(r, "ge1", "A"), grid=SMALL_LGBM[:1]),
            "does not match its grid_sha8",
        ),
    ]


@pytest.mark.parametrize("name,damage,needle", pair_cases(), ids=[c[0] for c in pair_cases()])
def test_pair_refusals_exit_1_with_counts_only_and_write_nothing(
    root: Path, capsys: pytest.CaptureFixture[str], name: str, damage: Any, needle: str
) -> None:
    damage(root)
    assert report(root) == 1
    captured = capsys.readouterr()
    assert needle in captured.err and captured.err.startswith("refused:")
    assert not HASH.search(captured.err) and not captured.out and not out(root).exists()


def test_the_grid_must_have_the_registered_size(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(ab, "EXPECTED_GRID_SIZE", 27)  # the synthetic grid has 2
    assert report(root) == 1 and "grid has 2 configurations, expected 27" in capsys.readouterr().err
    assert not out(root).exists()
    for target in ("ge2", "ge1"):
        set_grid(root, target, REAL_GRID)
        write_best(root, target, [REAL_GRID[0]] * 25)
    assert report(root) == 0  # the real 27-configuration shape passes


@pytest.mark.parametrize("fs", ["A", "B"])
def test_meta_repeats_must_be_five_for_both_sets(
    root: Path, capsys: pytest.CaptureFixture[str], fs: str
) -> None:
    edit_json(meta_path(root, "ge2", fs), repeats=4)
    assert report(root) == 1 and "meta has 4 repeats" in capsys.readouterr().err


@pytest.mark.parametrize("fs", ["A", "B"])
def test_runs_must_hold_one_row_per_repeat_for_both_sets(
    root: Path, capsys: pytest.CaptureFixture[str], fs: str
) -> None:
    pd.read_csv(runs_path(root, "ge1", fs)).iloc[1:].to_csv(runs_path(root, "ge1", fs), index=False)
    assert report(root) == 1 and "runs file has 4 rows" in capsys.readouterr().err
    edit_runs(runs_path(root, "ge2", fs), repeat=[0, 0, 1, 2, 3])
    assert (
        report(root, "--targets", "ge2") == 1
        and "exactly one row per repeat" in capsys.readouterr().err
    )


@pytest.mark.parametrize("fs", ["A", "B"])
@pytest.mark.parametrize(
    "name", ["csv_missing", "json_missing", "failed", "mean_above_rule", "draws", "mean"]
)
def test_both_permutation_controls_are_required_and_recomputed(
    root: Path, capsys: pytest.CaptureFixture[str], fs: str, name: str
) -> None:
    csv_path, json_path = (
        root / "ge2" / f"permutation_lgbm_{fs}.csv",
        root / "ge2" / f"permutation_lgbm_{fs}.json",
    )
    needle = {
        "csv_missing": "files are missing",
        "json_missing": "files are missing",
        "failed": "did not pass",
        "mean_above_rule": "did not pass",
        "draws": "disagree with its summary",
        "mean": "summary mean disagrees",
    }[name]
    if name == "csv_missing":
        csv_path.unlink()
    elif name == "json_missing":
        json_path.unlink()
    elif name == "failed":
        edit_json(json_path, decision="fail")
    elif name == "mean_above_rule":
        draws = pd.read_csv(csv_path)
        draws["null_auroc"] = 0.9
        draws.to_csv(csv_path, index=False)
        edit_json(json_path, mean=0.9)
    elif name == "draws":
        edit_json(json_path, draws=9)
    else:
        edit_json(json_path, mean=0.2)
    assert report(root) == 1
    captured = capsys.readouterr()
    assert needle in captured.err and f"lgbm {fs}" in captured.err and not HASH.search(captured.err)
    assert not out(root).exists()


@pytest.mark.parametrize("which", ["runs_B", "meta_A", "best_params_B"])
def test_missing_input_exits_2(root: Path, capsys: pytest.CaptureFixture[str], which: str) -> None:
    kind, fs = which.split("_", 1)[0], which[-1]
    name = {
        "runs": f"runs_lgbm_{fs}.csv",
        "meta": f"meta_lgbm_{fs}.json",
        "best": f"best_params_lgbm_{fs}.csv",
    }[kind]
    (root / "ge1" / name).unlink()
    assert (
        report(root) == 2 and "missing input" in capsys.readouterr().err and not out(root).exists()
    )
    assert report(root, "--targets", "ge2") == 0  # other targets are unaffected


# --------------------------------------------------------------------------- tuning summary of B


def test_tuning_counts_match_an_independent_count(root: Path) -> None:
    assert report(root) == 0
    configs = pd.read_csv(out(root) / "tuning_configs.csv")
    assert set(zip(configs["target"], configs["model"], strict=True)) == {
        ("ge1", "lgbm"),
        ("ge2", "lgbm"),
    }
    assert (
        list(configs.columns[:4]) == ["target", "model", "n_configs", "config"]
        and "n_chosen" in configs
        and "share_of_fits" in configs
    )
    for target, part in configs.groupby("target"):
        best = pd.read_csv(root / target / "best_params_lgbm_B.csv")
        counts = Counter(json.dumps(json.loads(p), sort_keys=True) for p in best["params"])
        assert part["config"].tolist() == [
            json.dumps(g, sort_keys=True) for g in read_meta(root, target, "B")["grid"]
        ]
        assert (
            part["n_chosen"].tolist() == [counts[c] for c in part["config"]]
            and part["n_chosen"].sum() == 25
        )
        assert part["share_of_fits"].tolist() == pytest.approx([n / 25 for n in part["n_chosen"]])


def test_unchosen_configurations_are_listed_with_zero(root: Path) -> None:
    write_best(root, "ge2", [SMALL_LGBM[1]] * 25)
    assert report(root) == 0
    part = pd.read_csv(out(root) / "tuning_configs.csv").query("target == 'ge2'")
    assert part["n_chosen"].tolist() == [0, 25] and part["n_configs"].tolist() == [2, 2]


def test_edge_shares_with_the_real_grid_shape(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    set_grid(root, "ge2", REAL_GRID)
    monkeypatch.setattr(ab, "EXPECTED_GRID_SIZE", 27)
    chosen = (
        [{"num_leaves": 3, "n_estimators": 100, "min_child_samples": 20}] * 10
        + [{"num_leaves": 5, "n_estimators": 200, "min_child_samples": 15}] * 5
        + [{"num_leaves": 7, "n_estimators": 200, "min_child_samples": 15}] * 5
        + [{"num_leaves": 5, "n_estimators": 300, "min_child_samples": 15}] * 5
    )
    write_best(root, "ge2", chosen)
    assert report(root, "--targets", "ge2") == 0
    configs = pd.read_csv(out(root) / "tuning_configs.csv").query("target == 'ge2'")
    assert len(configs) == 27 and sorted(configs.loc[configs["n_chosen"] > 0, "n_chosen"]) == [
        5,
        5,
        5,
        10,
    ]
    edges = (
        pd.read_csv(out(root) / "tuning_edges.csv").query("target == 'ge2'").set_index("parameter")
    )
    assert edges.loc["num_leaves", ["n_at_min", "n_at_max", "n_at_edge"]].tolist() == [10, 5, 15]
    assert edges.loc["n_estimators", ["n_at_min", "n_at_max", "n_at_edge"]].tolist() == [10, 5, 15]
    assert edges.loc[
        "min_child_samples", ["grid_min", "grid_max", "n_at_min", "n_at_max", "n_at_edge"]
    ].tolist() == [10, 20, 0, 10, 10]
    assert edges.loc["any", "n_at_edge"] == 20 and edges.loc[
        "any", "share_at_edge"
    ] == pytest.approx(0.8)
    assert edges.loc["num_leaves", "share_at_edge"] == pytest.approx(0.6)


def test_a_single_valued_parameter_is_never_at_an_edge(root: Path) -> None:
    assert report(root) == 0
    edges = (
        pd.read_csv(out(root) / "tuning_edges.csv").query("target == 'ge2'").set_index("parameter")
    )
    for key in ("n_estimators", "min_child_samples"):
        assert edges.loc[key, "n_values"] == 1 and edges.loc[key, "n_at_edge"] == 0
    assert edges.loc["num_leaves", "n_at_edge"] == 25 and edges.loc["any", "n_at_edge"] == 25


@pytest.mark.parametrize(
    "damage,needle",
    [("rows", "best_params has 24 rows"), ("duplicate_fit", "not one row per repeat and outer fold"),
     ("not_in_grid", "3 chosen configuration(s) are not in the grid"), ("columns", "unexpected columns")],
)  # fmt: skip
def test_tuning_inputs_of_b_are_checked(
    root: Path, capsys: pytest.CaptureFixture[str], damage: str, needle: str
) -> None:
    path = root / "ge2" / "best_params_lgbm_B.csv"
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
    else:
        best.rename(columns={"inner_auroc": "score"}).to_csv(path, index=False)
    assert report(root) == 1 and needle in capsys.readouterr().err and not out(root).exists()


def test_the_tuning_summary_does_not_influence_the_table_and_a_files_are_not_used(
    root: Path,
) -> None:
    assert report(root) == 0
    before = (out(root) / "model_b_table.csv").read_bytes()
    write_best(root, "ge2", [SMALL_LGBM[0]] * 25)
    (root / "ge2" / "best_params_lgbm_A.csv").write_text(
        "garbage"
    )  # A's tuning file is not read at all
    assert report(root) == 0
    assert (out(root) / "model_b_table.csv").read_bytes() == before
    assert pd.read_csv(out(root) / "tuning_configs.csv").query("target == 'ge2'")[
        "n_chosen"
    ].tolist() == [25, 0]


# --------------------------------------------------------------------------- files and privacy


def test_outputs_are_exactly_three_files_and_byte_identical_on_rerun(root: Path) -> None:
    assert report(root) == 0
    first = snap(out(root))
    assert sorted(first) == sorted(OUTPUTS)
    assert report(root) == 0 and snap(out(root)) == first
    assert not [p for p in out(root).iterdir() if p.name.startswith(".")]


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
    assert not (outside / "reports_model_b").exists()


def test_the_write_is_atomic(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(src: Any, dst: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(br.os, "replace", broken)
    with pytest.raises(OSError):
        report(root)
    assert not list(out(root).iterdir())


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
    assert any("best_params_lgbm_B.csv" in p for p in opened) and not [
        p for p in opened if "best_params_lgbm_A" in p
    ]
    assert not re.search(r"20\d\d-\d\d-\d\d|T\d\d:\d\d", blob)


def test_source_names_no_label_file_and_imports_neither_harness_nor_gait() -> None:
    source = (SRC / "ab_report.py").read_text()
    assert "fis_diagnostic" not in source and "UPDRS" not in source
    imports = re.findall(r"^\s*(?:from|import)\s+([\w.]+)", source, flags=re.M)
    assert not {"cv_harness", "feature_matrix", "hubu_fis", "gait", "model_report"} & {
        i.split(".")[0] for i in imports
    }
