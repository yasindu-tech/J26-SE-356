"""Baseline table (task FT-34): B0 and B1 on Set A from the official runs, nothing re-fitted.

    python models/tapping/src/baseline_report.py [--targets ge2 ge1] [--models b0 b1]

Reads runs_/meta_/permutation_ files written by cv_harness.py, never the oof files. Refuses (exit 1,
counts only) unless both models of a target share the splits, inner folds, roles, features and
target status, the repeats are 5 with one run row each, and the B1 permutation control exists and
passed. Rows are in fixed (target, model) order with no ranking. Writes baseline_table.csv.
Exit codes: 0 ok, 1 refused, 2 missing input or a path outside data/processed/cv.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv_splits as cvs
import numpy as np
import pandas as pd
from cv_splits import GuardFailure, InputProblem

EXPECTED_REPEATS = 5
PERMUTATION_MAX_MEAN_AUROC = 0.55  # the pre-registered rule, repeated here to re-check the file
SHARED_KEYS = ("splits_sha8", "inner_sha8", "roles_sha8", "features_sha8", "target_status")
METRICS = ("auroc", "balanced_accuracy", "sensitivity", "specificity", "mcc")
PPV_COLUMNS = ("ppv_at_1pct", "ppv_at_2pct", "ppv_at_5pct")
COUNT_COLUMNS = ("n_clips", "n_pos", "n_neg")
OUT_TABLE = "baseline_table.csv"
NOTE = "5 repeats reuse the same participants and are not independent evidence; CIs come in FT-38."
TABLE_COLUMNS = [
    "target", "model", "feature_set", "n_repeats",
    *[f"{m}_{s}" for m in METRICS for s in ("mean", "sd")],
    "auroc_fold_mean", "n_pred_pos_mean", *COUNT_COLUMNS,
    *[f"{c}_mean" for c in PPV_COLUMNS],
    "perm_n_draws", "perm_null_mean", "perm_null_sd", "perm_decision",
    *SHARED_KEYS, "grid_sha8",
]  # fmt: skip


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise GuardFailure(f"{path.name}: expected a JSON object")
    return data


def mean_sd(values: pd.Series) -> tuple[float, float]:
    """Mean and sample sd (ddof=1) across repeats; NaN propagates, it is never skipped or zeroed."""
    arr = values.to_numpy(dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")


def load_entry(cv_root: Path, target: str, model: str, feature_set: str) -> dict[str, Any]:
    folder = cv_root / target
    tag = f"{model}_{feature_set}"
    paths = {"runs": folder / f"runs_{tag}.csv", "meta": folder / f"meta_{tag}.json"}
    for kind, path in paths.items():
        if not path.is_file():
            raise InputProblem(f"missing input: {kind} file for {target} {model}")
    meta = read_json(paths["meta"])
    runs = pd.read_csv(paths["runs"], float_precision="round_trip")
    where = f"{target} {model}"
    if (
        meta.get("target") != target
        or meta.get("model") != model
        or meta.get("feature_set") != feature_set
    ):
        raise GuardFailure(f"{where}: meta file is for another target, model or feature set")
    if meta.get("repeats") != EXPECTED_REPEATS:
        raise GuardFailure(
            f"{where}: meta has {meta.get('repeats')} repeats, expected {EXPECTED_REPEATS}"
        )
    if len(runs) != EXPECTED_REPEATS:
        raise GuardFailure(f"{where}: runs file has {len(runs)} rows, expected {EXPECTED_REPEATS}")
    if sorted(runs["repeat"].tolist()) != list(range(EXPECTED_REPEATS)):
        raise GuardFailure(f"{where}: runs file does not hold exactly one row per repeat")
    for key in ("splits_sha8", "inner_sha8", "features_sha8"):
        if runs[key].nunique() != 1 or runs[key].iloc[0] != meta[key]:
            raise GuardFailure(f"{where}: runs file disagrees with its meta file on {key}")
    for column in COUNT_COLUMNS:
        if runs[column].nunique() != 1:
            raise GuardFailure(f"{where}: {column} differs between repeats")
    entry: dict[str, Any] = {"meta": meta, "runs": runs.sort_values("repeat"), "perm": None}
    if model == "b1":
        entry["perm"] = load_permutation(folder, tag, where)
    return entry


def load_permutation(folder: Path, tag: str, where: str) -> dict[str, Any]:
    csv_path, json_path = folder / f"permutation_{tag}.csv", folder / f"permutation_{tag}.json"
    if not csv_path.is_file() or not json_path.is_file():
        raise GuardFailure(f"{where}: the permutation control files are missing")
    summary = read_json(json_path)
    draws = pd.read_csv(csv_path, float_precision="round_trip")
    values = draws["null_auroc"].to_numpy(dtype=float)
    if len(values) < 1 or summary.get("draws") != len(values) or np.isnan(values).any():
        raise GuardFailure(f"{where}: permutation draws ({len(values)}) disagree with its summary")
    mean = float(values.mean())
    if abs(mean - float(summary["mean"])) > 1e-9:
        raise GuardFailure(f"{where}: permutation summary mean disagrees with its draws")
    if summary.get("decision") != "pass" or mean > PERMUTATION_MAX_MEAN_AUROC:
        raise GuardFailure(
            f"{where}: permutation control did not pass (decision {summary.get('decision')}, "
            f"{len(values)} draws)"
        )
    sd = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    return {"n_draws": len(values), "null_mean": mean, "null_sd": sd, "decision": "pass"}


def shared_values(target: str, entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The values both models of a target must agree on; refuse on the first difference."""
    first_model, *others = sorted(entries)
    shared = {k: entries[first_model]["meta"].get(k) for k in SHARED_KEYS}
    for model in others:
        for key in SHARED_KEYS:
            if entries[model]["meta"].get(key) != shared[key]:
                raise GuardFailure(f"{target}: {first_model} and {model} differ in {key}")
        for column in COUNT_COLUMNS:
            if (
                entries[model]["runs"][column].iloc[0]
                != entries[first_model]["runs"][column].iloc[0]
            ):
                raise GuardFailure(f"{target}: {first_model} and {model} differ in {column}")
    return shared


def summarise(
    target: str, model: str, feature_set: str, entry: dict[str, Any], shared: dict[str, Any]
) -> dict[str, Any]:
    runs = entry["runs"]
    row: dict[str, Any] = {"target": target, "model": model, "feature_set": feature_set,
                           "n_repeats": len(runs)}  # fmt: skip
    for metric in METRICS:
        row[f"{metric}_mean"], row[f"{metric}_sd"] = mean_sd(runs[metric])
    row["auroc_fold_mean"] = mean_sd(runs["auroc_fold_mean"])[0]
    row["n_pred_pos_mean"] = mean_sd(runs["n_pred_pos"])[0]
    for column in COUNT_COLUMNS:
        row[column] = int(runs[column].iloc[0])
    for column in PPV_COLUMNS:
        row[f"{column}_mean"] = mean_sd(runs[column])[0]
    perm = entry["perm"]
    row["perm_n_draws"] = perm["n_draws"] if perm else None
    row["perm_null_mean"] = perm["null_mean"] if perm else None
    row["perm_null_sd"] = perm["null_sd"] if perm else None
    row["perm_decision"] = perm["decision"] if perm else None
    row.update(shared)
    row["grid_sha8"] = entry["meta"]["grid_sha8"]
    return row


def build_table(
    cv_root: Path, targets: Sequence[str], models: Sequence[str], feature_set: str
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    shared_by_target: dict[str, dict[str, Any]] = {}
    for target in sorted(set(targets)):
        entries = {m: load_entry(cv_root, target, m, feature_set) for m in sorted(set(models))}
        shared = shared_values(target, entries)
        shared_by_target[target] = shared
        rows += [summarise(target, m, feature_set, entries[m], shared) for m in sorted(entries)]
    table = pd.DataFrame(rows, columns=TABLE_COLUMNS)
    return table.sort_values(["target", "model"], kind="stable").reset_index(
        drop=True
    ), shared_by_target


def render(table: pd.DataFrame, shared_by_target: dict[str, dict[str, Any]]) -> str:
    lines = [
        "Baselines B0 and B1 on Set A (mean and sd over repeats; pooled out-of-fold predictions)"
    ]
    for target, shared in shared_by_target.items():
        lines.append(f"{target}: " + ", ".join(f"{k} {v}" for k, v in shared.items()))
    shown = ["target", "model", "n_repeats", "auroc_mean", "auroc_sd", "balanced_accuracy_mean",
             "balanced_accuracy_sd", "sensitivity_mean", "specificity_mean", "mcc_mean", "mcc_sd",
             "n_pred_pos_mean", "n_clips", "n_pos", "n_neg", "perm_n_draws", "perm_null_mean",
             "perm_decision"]  # fmt: skip
    lines.append(table[shown].to_string(index=False, float_format=lambda v: f"{v:.4f}", na_rep="-"))
    lines.append(NOTE)
    return "\n".join(lines)


def table_bytes(table: pd.DataFrame) -> bytes:
    return table.to_csv(index=False, lineterminator="\n", float_format="%.10g").encode()


def _atomic(final: Path, data: bytes) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f".{final.name}.tmp{os.getpid()}")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, final)
    finally:
        if tmp.exists():
            tmp.unlink()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cv-root", type=Path, default=None)
    ap.add_argument("--targets", nargs="+", choices=sorted(cvs.TARGETS), default=["ge2", "ge1"])
    ap.add_argument("--models", nargs="+", choices=["b0", "b1"], default=["b0", "b1"])
    ap.add_argument("--feature-set", choices=["A"], default="A")
    ap.add_argument("--out-dir", type=Path, default=None)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cv_root = Path(args.cv_root or cvs.CV_ROOT)
    out_dir = Path(args.out_dir or cv_root / "reports")
    for name, path in (("--cv-root", cv_root), ("--out-dir", out_dir)):
        if not cvs.is_inside(path, cvs.CV_ROOT):
            print(f"error: {name} must be inside data/processed/cv", file=sys.stderr)
            return 2
    try:
        table, shared = build_table(cv_root, args.targets, args.models, args.feature_set)
        data = table_bytes(table)
        text = render(table, shared)
        _atomic(out_dir / OUT_TABLE, data)
    except InputProblem as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except GuardFailure as err:
        print(f"refused: {err}", file=sys.stderr)
        return 1
    print(text)
    print(f"wrote {OUT_TABLE}: {len(table)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
