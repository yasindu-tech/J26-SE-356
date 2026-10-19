"""Model A report (task FT-35): B0, B1 and LightGBM on Set A from the official runs, plus counts of
which tuning configuration each outer fit chose. Nothing is re-fitted and the tuning summary is
descriptive only: it is computed after the table and feeds nothing back.

    python models/tapping/src/model_report.py [--targets ge2 ge1] [--models b0 b1 lgbm]

Reuses baseline_report's loading and refusals (shared splits, inner folds, roles, features and
target status across all listed models; 5 repeats, one row each; B1 and LightGBM need a passing
permutation control). Writes model_a_table.csv, tuning_configs.csv and tuning_edges.csv.
Exit codes: 0 ok, 1 refused, 2 missing input or a path outside data/processed/cv.
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
from cv_splits import GuardFailure, InputProblem

TUNED_MODELS = ("b1", "lgbm")
N_OUTER = 5
N_FITS = br.EXPECTED_REPEATS * N_OUTER  # 25 outer fits per model and target
OUT_TABLE = "model_a_table.csv"
OUT_CONFIGS = "tuning_configs.csv"
OUT_EDGES = "tuning_edges.csv"
TITLE = (
    "Model A report on Set A: B0, B1 and LightGBM (mean and sd over repeats; pooled out-of-fold)"
)
BEST_COLUMNS = ["repeat", "outer_fold", "params", "inner_auroc"]
CONFIG_COLUMNS = ["target", "model", "n_configs", "config", "n_chosen", "share_of_fits"]
EDGE_COLUMNS = [
    "target", "model", "parameter", "n_values", "grid_min", "grid_max", "n_at_min", "n_at_max",
    "n_at_edge", "share_at_edge",
]  # fmt: skip


def canon(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True)


def grid_hash(model: str, grid: list[dict[str, Any]]) -> str:
    """The harness's grid_sha8, recomputed here so a doctored grid in the meta file is caught."""
    return cvs.sha8_of(json.dumps({"model": model, "grid": grid}, sort_keys=True))


def load_choices(
    cv_root: Path, target: str, model: str, feature_set: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The tuning grid and the configuration chosen in each of the 25 outer fits."""
    folder, tag, where = cv_root / target, f"{model}_{feature_set}", f"{target} {model}"
    path = folder / f"best_params_{tag}.csv"
    if not path.is_file():
        raise InputProblem(f"missing input: best_params file for {target} {model}")
    meta = br.read_json(folder / f"meta_{tag}.json")
    grid = meta.get("grid")
    if not isinstance(grid, list) or not grid or grid_hash(model, grid) != meta.get("grid_sha8"):
        raise GuardFailure(f"{where}: the grid in the meta file does not match its grid_sha8")
    best = pd.read_csv(path, dtype={"params": str})
    if list(best.columns) != BEST_COLUMNS:
        raise GuardFailure(f"{where}: best_params file has unexpected columns")
    if len(best) != N_FITS:
        raise GuardFailure(f"{where}: best_params has {len(best)} rows, expected {N_FITS}")
    pairs = set(zip(best["repeat"], best["outer_fold"], strict=True))
    if pairs != {(r, f) for r in range(br.EXPECTED_REPEATS) for f in range(N_OUTER)}:
        raise GuardFailure(f"{where}: best_params is not one row per repeat and outer fold")
    chosen = [json.loads(p) for p in best["params"]]
    allowed = {canon(g) for g in grid}
    outside = sum(canon(c) not in allowed for c in chosen)
    if outside:
        raise GuardFailure(f"{where}: {outside} chosen configuration(s) are not in the grid")
    return grid, chosen


def tuning_tables(
    target: str, model: str, grid: list[dict[str, Any]], chosen: list[dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = Counter(canon(c) for c in chosen)
    keys = list(grid[0])
    config_rows = []
    for g in grid:  # grid order (simplest first), never sorted by how often a config won
        n = counts[canon(g)]
        config_rows.append(
            {
                "target": target,
                "model": model,
                "n_configs": len(grid),
                "config": canon(g),
                **g,
                "n_chosen": n,
                "share_of_fits": n / len(chosen),
            }
        )
    edge_rows = []
    edge_flags = []
    for key in keys:
        values = sorted({g[key] for g in grid})
        lo, hi = values[0], values[-1]
        n_lo = sum(c[key] == lo for c in chosen)
        n_hi = sum(c[key] == hi for c in chosen)
        at_edge = [len(values) > 1 and c[key] in (lo, hi) for c in chosen]
        edge_flags.append(at_edge)
        n_edge = sum(at_edge)
        edge_rows.append(
            {
                "target": target,
                "model": model,
                "parameter": key,
                "n_values": len(values),
                "grid_min": lo,
                "grid_max": hi,
                "n_at_min": n_lo,
                "n_at_max": n_hi,
                "n_at_edge": n_edge,
                "share_at_edge": n_edge / len(chosen),
            }
        )
    any_edge = sum(any(flags[i] for flags in edge_flags) for i in range(len(chosen)))
    edge_rows.append({"target": target, "model": model, "parameter": "any", "n_values": None, "grid_min": None, "grid_max": None, "n_at_min": None, "n_at_max": None, "n_at_edge": any_edge, "share_at_edge": any_edge / len(chosen)})  # fmt: skip
    return pd.DataFrame(config_rows), pd.DataFrame(edge_rows, columns=EDGE_COLUMNS)


def build_tuning(
    cv_root: Path, targets: Sequence[str], models: Sequence[str], feature_set: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    configs, edges = [], []
    for target in sorted(set(targets)):
        for model in sorted(set(models) & set(TUNED_MODELS)):
            grid, chosen = load_choices(cv_root, target, model, feature_set)
            c, e = tuning_tables(target, model, grid, chosen)
            configs.append(c)
            edges.append(e)
    if not configs:
        return pd.DataFrame(columns=CONFIG_COLUMNS), pd.DataFrame(columns=EDGE_COLUMNS)
    return pd.concat(configs, ignore_index=True), pd.concat(edges, ignore_index=True)


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.10g").encode()


def render_tuning(configs: pd.DataFrame, edges: pd.DataFrame) -> str:
    if configs.empty:
        return "Tuning summary: no tuned model listed."
    lines = [
        "Tuning summary (descriptive; counts of the configuration chosen in the 25 outer fits)"
    ]
    for (target, model), part in configs.groupby(["target", "model"], sort=True):
        shown = ", ".join(f"{r.config} x{r.n_chosen}" for r in part.itertuples() if r.n_chosen)
        edge = edges[(edges["target"] == target) & (edges["model"] == model)]
        any_edge = edge[edge["parameter"] == "any"].iloc[0]
        lines.append(
            f"{target} {model}: {int(part['n_configs'].iloc[0])} configs; chosen: {shown}; fits with a parameter at a grid edge {int(any_edge['n_at_edge'])} of {N_FITS}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cv-root", type=Path, default=None)
    ap.add_argument("--targets", nargs="+", choices=sorted(cvs.TARGETS), default=["ge2", "ge1"])
    ap.add_argument(
        "--models", nargs="+", choices=["b0", "b1", "lgbm"], default=["b0", "b1", "lgbm"]
    )
    ap.add_argument("--feature-set", choices=["A"], default="A")
    ap.add_argument("--out-dir", type=Path, default=None)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cv_root = Path(args.cv_root or cvs.CV_ROOT)
    out_dir = Path(args.out_dir or cv_root / "reports_model_a")
    for name, path in (("--cv-root", cv_root), ("--out-dir", out_dir)):
        if not cvs.is_inside(path, cvs.CV_ROOT):
            print(f"error: {name} must be inside data/processed/cv", file=sys.stderr)
            return 2
    try:
        table, shared = br.build_table(cv_root, args.targets, args.models, args.feature_set)
        configs, edges = build_tuning(cv_root, args.targets, args.models, args.feature_set)
        files = {
            OUT_TABLE: br.table_bytes(table),
            OUT_CONFIGS: csv_bytes(configs),
            OUT_EDGES: csv_bytes(edges),
        }
        text = br.render(table, shared, TITLE) + "\n" + render_tuning(configs, edges)
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
