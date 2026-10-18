"""A-vs-B report (task FT-37): LightGBM on Set A and on Set B from the official runs, side by side.
No difference is computed here (that is FT-39) and nothing is re-fitted.

    python models/tapping/src/ab_report.py [--targets ge2 ge1]

Refuses (exit 1, counts only) unless, per target, A and B share splits, inner folds, roles, target
status, grid, seed, repeats, the fixed LightGBM settings, the 27-configuration grid and the label
counts, differ in features, and B's columns are a larger superset of A's that equals the registered
Set B. Both permutation controls must exist, pass and agree with their draws. Writes model_b_table.csv
plus the tuning summary of B. Exit codes: 0 ok, 1 refused, 2 missing input or a path outside
data/processed/cv.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import baseline_report as br
import cv_splits as cvs
import pandas as pd
from cv_splits import SET_A_PRIMARY, SET_B_ADDED, GuardFailure, InputProblem

MODEL = "lgbm"
FEATURE_SETS = ("A", "B")
EXPECTED_GRID_SIZE = 27
N_OUTER = 5
N_FITS = br.EXPECTED_REPEATS * N_OUTER
PAIR_KEYS = ("splits_sha8", "inner_sha8", "roles_sha8", "target_status", "grid_sha8", "seed")
OUT_TABLE = "model_b_table.csv"
OUT_CONFIGS = "tuning_configs.csv"
OUT_EDGES = "tuning_edges.csv"
TITLE = "Model B report: LightGBM on Set A and Set B (mean and sd over repeats; pooled out-of-fold)"
NOTE = (
    "5 repeats reuse the same participants and are not independent evidence; "
    "the A-vs-B difference and its CI come in FT-39."
)
BEST_COLUMNS = ["repeat", "outer_fold", "params", "inner_auroc"]
METRICS = br.METRICS
TABLE_COLUMNS = [
    "target", "feature_set", "model", "n_repeats", "n_features",
    *[f"{m}_{s}" for m in METRICS for s in ("mean", "sd")],
    "auroc_fold_mean", "n_pred_pos_mean", "n_pred_pos_sd", *br.COUNT_COLUMNS,
    "perm_n_draws", "perm_null_mean", "perm_null_sd", "perm_decision",
    *PAIR_KEYS, "features_sha8",
]  # fmt: skip
CONFIG_COLUMNS = ["target", "model", "n_configs", "config", "n_chosen", "share_of_fits"]
EDGE_COLUMNS = [
    "target", "model", "parameter", "n_values", "grid_min", "grid_max", "n_at_min", "n_at_max",
    "n_at_edge", "share_at_edge",
]  # fmt: skip
SHOWN = [
    "target", "feature_set", "n_features", "auroc_mean", "auroc_sd", "balanced_accuracy_mean",
    "balanced_accuracy_sd", "sensitivity_mean", "specificity_mean", "mcc_mean", "mcc_sd",
    "n_pred_pos_mean", "n_pred_pos_sd", "perm_n_draws", "perm_null_mean", "perm_decision",
]  # fmt: skip


def canon(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True)


def grid_hash(grid: list[dict[str, Any]]) -> str:
    """The harness's grid_sha8, recomputed so a doctored grid or column list is caught."""
    return cvs.sha8_of(json.dumps({"model": MODEL, "grid": grid}, sort_keys=True))


# --------------------------------------------------------------------------- loading and checks


def load_set(cv_root: Path, target: str, feature_set: str) -> dict[str, Any]:
    folder, tag = cv_root / target, f"{MODEL}_{feature_set}"
    where = f"{target} {MODEL} {feature_set}"
    for kind, name in (
        ("runs", f"runs_{tag}.csv"),
        ("meta", f"meta_{tag}.json"),
        ("best_params", f"best_params_{tag}.csv"),
    ):
        if not (folder / name).is_file():
            raise InputProblem(f"missing input: {kind} file for {where}")
    entry = br.load_entry(cv_root, target, MODEL, feature_set)
    perm = entry["perm"] or br.load_permutation(folder, tag, where)
    meta, runs = entry["meta"], entry["runs"]
    columns, grid = meta.get("columns"), meta.get("grid")
    if not isinstance(columns, list) or not columns or not isinstance(grid, list) or not grid:
        raise GuardFailure(f"{where}: meta lacks the column list or the grid")
    if cvs.sha8_of(json.dumps(columns)) != meta.get("features_sha8"):
        raise GuardFailure(f"{where}: features_sha8 does not match its column list")
    if grid_hash(grid) != meta.get("grid_sha8"):
        raise GuardFailure(f"{where}: the grid does not match its grid_sha8")
    if runs["grid_sha8"].nunique() != 1 or runs["grid_sha8"].iloc[0] != meta["grid_sha8"]:
        raise GuardFailure(f"{where}: runs file disagrees with its meta file on grid_sha8")
    if runs["n_features"].nunique() != 1 or runs["n_features"].iloc[0] != len(columns):
        raise GuardFailure(f"{where}: n_features disagrees with the column list")
    return {"meta": meta, "runs": runs, "perm": perm, "columns": columns, "grid": grid}


def check_pair(target: str, a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Same folds, seed and tuning budget; different features. Returns the shared values."""
    where = f"{target}: {MODEL} A and B"
    shared = {k: a["meta"].get(k) for k in PAIR_KEYS}
    for key in PAIR_KEYS:
        if b["meta"].get(key) != shared[key]:
            raise GuardFailure(f"{where} differ in {key}")
    if a["meta"].get("constants") != b["meta"].get("constants"):
        raise GuardFailure(f"{where} differ in the fixed LightGBM settings or constants")
    for label, entry in (("A", a), ("B", b)):
        if len(entry["grid"]) != EXPECTED_GRID_SIZE:
            raise GuardFailure(
                f"{target}: {MODEL} {label} grid has {len(entry['grid'])} configurations, expected {EXPECTED_GRID_SIZE}"
            )
    for column in br.COUNT_COLUMNS:
        if a["runs"][column].iloc[0] != b["runs"][column].iloc[0]:
            raise GuardFailure(f"{where} differ in {column}")
    if a["meta"]["features_sha8"] == b["meta"]["features_sha8"]:
        raise GuardFailure(f"{where} must differ in features_sha8")
    if len(b["columns"]) <= len(a["columns"]):
        raise GuardFailure(
            f"{where}: B has {len(b['columns'])} features, not more than A's {len(a['columns'])}"
        )
    if not set(a["columns"]) <= set(b["columns"]):
        raise GuardFailure(f"{where}: Set A columns are not a subset of Set B columns")
    if a["columns"] != list(SET_A_PRIMARY) or b["columns"] != list(SET_A_PRIMARY) + list(
        SET_B_ADDED
    ):
        raise GuardFailure(f"{where}: feature columns differ from the registered Set A and Set B")
    return shared


# --------------------------------------------------------------------------- the table


def summarise(
    target: str, feature_set: str, entry: dict[str, Any], shared: dict[str, Any]
) -> dict[str, Any]:
    runs, perm = entry["runs"], entry["perm"]
    row: dict[str, Any] = {"target": target, "feature_set": feature_set, "model": MODEL,
                           "n_repeats": len(runs), "n_features": len(entry["columns"])}  # fmt: skip
    for metric in METRICS:
        row[f"{metric}_mean"], row[f"{metric}_sd"] = br.mean_sd(runs[metric])
    row["auroc_fold_mean"] = br.mean_sd(runs["auroc_fold_mean"])[0]
    row["n_pred_pos_mean"], row["n_pred_pos_sd"] = br.mean_sd(runs["n_pred_pos"])
    for column in br.COUNT_COLUMNS:
        row[column] = int(runs[column].iloc[0])
    row.update({"perm_n_draws": perm["n_draws"], "perm_null_mean": perm["null_mean"],
                "perm_null_sd": perm["null_sd"], "perm_decision": perm["decision"]})  # fmt: skip
    row.update(shared)
    row["features_sha8"] = entry["meta"]["features_sha8"]
    return row


def build_table(
    cv_root: Path, targets: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    shared_by_target: dict[str, dict[str, Any]] = {}
    sets: dict[str, dict[str, Any]] = {}
    for target in sorted(set(targets)):
        a, b = (load_set(cv_root, target, fs) for fs in FEATURE_SETS)
        shared = check_pair(target, a, b)
        shared_by_target[target] = shared
        sets[target] = b
        rows += [
            summarise(target, fs, e, shared) for fs, e in zip(FEATURE_SETS, (a, b), strict=True)
        ]
    table = pd.DataFrame(rows, columns=TABLE_COLUMNS)
    return (
        table.sort_values(["target", "feature_set"], kind="stable").reset_index(drop=True),
        shared_by_target,
        sets,
    )


def cell(value: Any) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "-"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def render(table: pd.DataFrame, shared_by_target: dict[str, dict[str, Any]]) -> str:
    lines = [TITLE]
    for target, shared in shared_by_target.items():
        sets = table[table["target"] == target].set_index("feature_set")["features_sha8"]
        text = ", ".join(f"{k} {v}" for k, v in shared.items())
        lines.append(f"{target}: {text}; features_sha8 A {sets['A']}, B {sets['B']}")
    cells = [[cell(v) for v in row] for row in table[SHOWN].itertuples(index=False, name=None)]
    widths = [max(len(h), *(len(r[i]) for r in cells)) for i, h in enumerate(SHOWN)]
    lines.append("  ".join(h.rjust(w) for h, w in zip(SHOWN, widths, strict=True)))
    lines += ["  ".join(c.rjust(w) for c, w in zip(r, widths, strict=True)) for r in cells]
    lines.append(NOTE)
    return "\n".join(lines)


# --------------------------------------------------------------------------- tuning summary of B


def load_choices(cv_root: Path, target: str, entry: dict[str, Any]) -> list[dict[str, Any]]:
    where = f"{target} {MODEL} B"
    best = pd.read_csv(cv_root / target / f"best_params_{MODEL}_B.csv", dtype={"params": str})
    if list(best.columns) != BEST_COLUMNS:
        raise GuardFailure(f"{where}: best_params file has unexpected columns")
    if len(best) != N_FITS:
        raise GuardFailure(f"{where}: best_params has {len(best)} rows, expected {N_FITS}")
    if set(zip(best["repeat"], best["outer_fold"], strict=True)) != {
        (r, f) for r in range(br.EXPECTED_REPEATS) for f in range(N_OUTER)
    }:
        raise GuardFailure(f"{where}: best_params is not one row per repeat and outer fold")
    chosen = [json.loads(p) for p in best["params"]]
    allowed = {canon(g) for g in entry["grid"]}
    outside = sum(canon(c) not in allowed for c in chosen)
    if outside:
        raise GuardFailure(f"{where}: {outside} chosen configuration(s) are not in the grid")
    return chosen


def tuning_tables(
    target: str, grid: list[dict[str, Any]], chosen: list[dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = Counter(canon(c) for c in chosen)
    config_rows = []
    for g in grid:  # grid order (simplest first), never sorted by how often a config won
        n = counts[canon(g)]
        config_rows.append(
            {
                "target": target,
                "model": MODEL,
                "n_configs": len(grid),
                "config": canon(g),
                **g,
                "n_chosen": n,
                "share_of_fits": n / len(chosen),
            }
        )
    edge_rows, flags = [], []
    for key in grid[0]:
        values = sorted({g[key] for g in grid})
        lo, hi = values[0], values[-1]
        at_edge = [len(values) > 1 and c[key] in (lo, hi) for c in chosen]
        flags.append(at_edge)
        edge_rows.append({"target": target, "model": MODEL, "parameter": key, "n_values": len(values), "grid_min": lo, "grid_max": hi, "n_at_min": sum(c[key] == lo for c in chosen), "n_at_max": sum(c[key] == hi for c in chosen), "n_at_edge": sum(at_edge), "share_at_edge": sum(at_edge) / len(chosen)})  # fmt: skip
    any_edge = sum(any(f[i] for f in flags) for i in range(len(chosen)))
    edge_rows.append({"target": target, "model": MODEL, "parameter": "any", "n_values": None, "grid_min": None, "grid_max": None, "n_at_min": None, "n_at_max": None, "n_at_edge": any_edge, "share_at_edge": any_edge / len(chosen)})  # fmt: skip
    return pd.DataFrame(config_rows), pd.DataFrame(edge_rows, columns=EDGE_COLUMNS)


def build_tuning(
    cv_root: Path, sets: dict[str, dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    configs, edges = [], []
    for target in sorted(sets):
        c, e = tuning_tables(
            target, sets[target]["grid"], load_choices(cv_root, target, sets[target])
        )
        configs.append(c)
        edges.append(e)
    return pd.concat(configs, ignore_index=True), pd.concat(edges, ignore_index=True)


def render_tuning(configs: pd.DataFrame, edges: pd.DataFrame) -> str:
    lines = ["Tuning summary of Set B (descriptive; configuration chosen in the 25 outer fits)"]
    for target, part in configs.groupby("target", sort=True):
        shown = ", ".join(f"{r.config} x{r.n_chosen}" for r in part.itertuples() if r.n_chosen)
        any_edge = edges[(edges["target"] == target) & (edges["parameter"] == "any")].iloc[0]
        lines.append(
            f"{target}: {int(part['n_configs'].iloc[0])} configs; chosen: {shown}; fits with a parameter at a grid edge {int(any_edge['n_at_edge'])} of {N_FITS}"
        )
    return "\n".join(lines)


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.10g").encode()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cv-root", type=Path, default=None)
    ap.add_argument("--targets", nargs="+", choices=sorted(cvs.TARGETS), default=["ge2", "ge1"])
    ap.add_argument("--out-dir", type=Path, default=None)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cv_root = Path(args.cv_root or cvs.CV_ROOT)
    out_dir = Path(args.out_dir or cv_root / "reports_model_b")
    for name, path in (("--cv-root", cv_root), ("--out-dir", out_dir)):
        if not cvs.is_inside(path, cvs.CV_ROOT):
            print(f"error: {name} must be inside data/processed/cv", file=sys.stderr)
            return 2
    try:
        table, shared, sets = build_table(cv_root, args.targets)
        configs, edges = build_tuning(cv_root, sets)
        files = {
            OUT_TABLE: csv_bytes(table),
            OUT_CONFIGS: csv_bytes(configs),
            OUT_EDGES: csv_bytes(edges),
        }
        text = render(table, shared) + "\n" + render_tuning(configs, edges)
        for name, data in files.items():
            br._atomic(out_dir / name, data)
    except InputProblem as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except GuardFailure as err:
        print(f"refused: {err}", file=sys.stderr)
        return 1
    print(text)
    print(f"wrote {', '.join(files)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
