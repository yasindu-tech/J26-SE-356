"""Results tables (task FT-41). Trains nothing: collects existing outputs and checks they agree.

    python models/tapping/src/results_table.py [--targets ge2 ge1] [--out-dir DIR]

Reads runs_/meta_ files of b0 A, b1 A, lgbm A, lgbm B, b1 R, lgbm R plus ci_table.csv,
auroc_difference.csv, reference_table.csv and features.csv (label columns never parsed), checks
hashes, repeats and estimates agree, and writes runs.csv, results_table.csv, detector_quality.csv
and results_meta.json (default data/processed/cv/results, always inside data/processed/cv).
Exit codes: 0 ok, 1 refused, 2 missing input or a path outside data/processed/cv.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv_splits as cvs
import numpy as np
import pandas as pd
from cv_splits import GuardFailure, InputProblem

TOOL = "results-table-1"
TARGET_ORDER = ("ge2", "ge1")  # ge2 is the primary target, pending the supervisor
MODELS = (("b0", "A"), ("b1", "A"), ("lgbm", "A"), ("lgbm", "B"), ("b1", "R"), ("lgbm", "R"))
CORE_METRICS = ("auroc", "balanced_accuracy", "sensitivity", "specificity", "mcc")
PPV_METRICS = ("ppv_at_1pct", "ppv_at_2pct", "ppv_at_5pct")
METRICS = CORE_METRICS + PPV_METRICS
REPEATS = 5
EXPECTED_DRAWS_SHA8 = "fa9c8702"
TOL = 1e-9
CI_MODELS = (("b0", "A"), ("b1", "A"), ("lgbm", "A"), ("lgbm", "B"))
REF_MODELS = (("b1", "R"), ("lgbm", "R"))
OFFICIAL_LABEL = {("ge2", "lgbm B minus lgbm A", "participant_bootstrap")}
DIFF_COMPARISONS = ("lgbm B minus lgbm A", "lgbm A minus b1 A")
DIFF_METHODS = ("participant_bootstrap", "delong")
REF_DIFFS = ("lgbm A minus lgbm R", "lgbm R minus b1 R")
SHARED_KEYS = ("splits_sha8", "inner_sha8", "roles_sha8", "target_status")
OUT_RUNS, OUT_TABLE, OUT_QUALITY, OUT_META = (
    "runs.csv", "results_table.csv", "detector_quality.csv", "results_meta.json"
)  # fmt: skip

NOTES = (
    "ge2 (rating >= 2) is the primary target, pending the supervisor; ge1 is reported too. "
    "CIs are participant-cluster bootstrap (B 2000, seed 2026) over 117 participants; the 5 "
    "repeats are not independent evidence. Chance AUROC in this harness is about 0.48. Threshold "
    "fixed at 0.5. PPV at 1/2/5% prevalence is derived from sensitivity and specificity at that "
    "threshold. Only ge2 lgbm B minus lgbm A is confirmatory (not significant); every other "
    "difference is descriptive, with no multiplicity adjustment. R = our reimplementation of the "
    "reference pipeline's classical 27 features on our landmarks; parity with the reference code "
    "is not verified. LightGBM chose a parameter at a grid edge in 25 of 25 outer fits "
    "(limitation; grid not widened). No row implies a ranking."
)
QUALITY_HEADER = (
    "Context for reading Set B against Set A: detector limits (extended-index pose, shoulder or "
    "double peaks, false taps on noise-sized clips, fps-dependent velocity) may suppress Set B "
    "signals. These counts are descriptive only and are NOT used to select, drop or reweight "
    "anything. No flag for tracker failure or noise-sized clips exists, so those are not counted."
)

RUNS_COLUMNS = [
    "target", "model", "feature_set", "repeat", "seed", "threshold", "auroc", "auroc_fold_mean",
    "auroc_fold_sd", "balanced_accuracy", "sensitivity", "specificity", "mcc", "n_pred_pos",
    "ppv_at_1pct", "ppv_at_2pct", "ppv_at_5pct", "n_clips", "n_pos", "n_neg", "n_features",
    "features_sha8", "splits_sha8", "inner_sha8", "grid_sha8",
]  # fmt: skip
RUNS_OUT_COLUMNS = RUNS_COLUMNS + ["roles_sha8", "target_status"]
TABLE_COLUMNS = [
    "section", "target", "subject", "model", "feature_set", "metric", "estimate", "ci_low",
    "ci_high", "n_draws_used", "ci_source", "role", "method", "p", "p_min", "p_max", "draws_sha8",
    "splits_sha8", "inner_sha8", "roles_sha8", "target_status",
]  # fmt: skip
QUALITY_COLUMNS = ["group", "item", "n_clips", "n_total", "definition"]
FEATURE_COLUMNS = [
    "fps",
    "qc_status",
    "n_one_sided_excluded",
    "n_iti_excluded",
    "n_short_iti",
    "nan_reasons_b",
    *cvs.SET_B_ADDED,
]  # label columns (rating, UPDRS) are deliberately absent: never parsed


def label(model: str, fs: str) -> str:
    return f"{model} {fs}"


# --------------------------------------------------------------------------- reading


def read_raw(path: Path) -> pd.DataFrame:
    """Every cell as text, blanks kept as '' (a blank is never turned into a number)."""
    if not path.is_file():
        raise InputProblem(f"missing input file {path.name}")
    return pd.read_csv(path, dtype=str, keep_default_na=False, comment=None)


def to_float(values: pd.Series, where: str) -> np.ndarray:
    out = []
    for v in values:
        if v == "" or v.lower() == "nan":
            out.append(float("nan"))
            continue
        try:
            out.append(float(v))
        except ValueError:
            raise GuardFailure(f"{where}: a value is not numeric") from None
    return np.array(out, dtype=float)


def need_columns(frame: pd.DataFrame, columns: Sequence[str], where: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise GuardFailure(f"{where}: {len(missing)} required column(s) missing")


def agree(a: float, b: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)  # NaN only matches NaN, never 0
    return abs(a - b) <= TOL


def load_model(cv_root: Path, target: str, model: str, fs: str) -> dict[str, Any]:
    where = f"{target} {label(model, fs)}"
    runs = read_raw(cv_root / target / f"runs_{model}_{fs}.csv")
    meta_path = cv_root / target / f"meta_{model}_{fs}.json"
    if not meta_path.is_file():
        raise InputProblem(f"missing input file {meta_path.name}")
    meta = json.loads(meta_path.read_text())
    need_columns(runs, RUNS_COLUMNS, where)
    repeats = sorted(int(float(v)) for v in runs["repeat"]) if len(runs) else []
    if len(runs) != REPEATS or repeats != list(range(REPEATS)):
        raise GuardFailure(
            f"{where}: expected repeats 0-{REPEATS - 1} once each, got {len(runs)} rows"
        )
    if meta.get("repeats") != REPEATS:
        raise GuardFailure(
            f"{where}: meta records {meta.get('repeats')} repeats, expected {REPEATS}"
        )
    for column, expected in (("target", target), ("model", model), ("feature_set", fs)):
        if set(runs[column]) != {expected} or meta.get(column) != expected:
            raise GuardFailure(f"{where}: {column} differs between runs and meta")
    for key in ("splits_sha8", "inner_sha8", "features_sha8", "grid_sha8"):
        if set(runs[key]) != {meta.get(key)}:
            raise GuardFailure(f"{where}: {key} in runs differs from meta")
    runs = runs.assign(_r=runs["repeat"].astype(float).astype(int)).sort_values("_r", kind="stable")
    runs = runs.drop(columns="_r").reset_index(drop=True)
    means = {m: float(np.mean(to_float(runs[m], where))) for m in METRICS}
    return {"runs": runs, "meta": meta, "means": means}


def load_target(cv_root: Path, target: str) -> dict[tuple[str, str], dict[str, Any]]:
    models = {(m, fs): load_model(cv_root, target, m, fs) for m, fs in MODELS}
    for key in SHARED_KEYS:
        values = {entry["meta"].get(key) for entry in models.values()}
        if len(values) != 1 or None in values:
            raise GuardFailure(f"{target}: {key} differs across models ({len(values)} distinct)")
    for (m, fs), entry in models.items():
        official = models[(m, "A")]["meta"]["grid_sha8"]
        if entry["meta"]["grid_sha8"] != official:
            raise GuardFailure(
                f"{target} {label(m, fs)}: grid_sha8 differs from the official {m} run"
            )
    return models


def shared_of(models: dict[tuple[str, str], dict[str, Any]]) -> dict[str, str]:
    meta = models[MODELS[0]]["meta"]
    return {k: str(meta[k]) for k in SHARED_KEYS}


# --------------------------------------------------------------------------- source tables


def unique_index(
    frame: pd.DataFrame, keys: Sequence[str], where: str
) -> dict[tuple[str, ...], Any]:
    need_columns(frame, keys, where)
    out: dict[tuple[str, ...], Any] = {}
    for row in frame.to_dict("records"):
        k = tuple(row[c] for c in keys)
        if k in out:
            raise GuardFailure(f"{where}: a row appears twice")
        out[k] = row
    return out


def check_shared(row: dict[str, str], shared: dict[str, str], where: str) -> None:
    for key in SHARED_KEYS:
        if row[key] != shared[key]:
            raise GuardFailure(f"{where}: {key} differs from the runs")
    if row["draws_sha8"] != EXPECTED_DRAWS_SHA8:
        raise GuardFailure(f"{where}: draws_sha8 is not {EXPECTED_DRAWS_SHA8}")


def check_estimate(row: dict[str, str], mean: float, where: str) -> None:
    if not agree(to_float(pd.Series([row["estimate"]]), where)[0], mean):
        raise GuardFailure(f"{where}: estimate differs from the mean of the runs (stale table?)")


def source_tables(
    cv_root: Path, targets: Sequence[str], models: dict[str, dict[tuple[str, str], dict[str, Any]]]
) -> dict[str, Any]:
    ci = read_raw(cv_root / "reports_ci" / "ci_table.csv")
    ref = read_raw(cv_root / "reports_ci" / "reference_table.csv")
    diff = read_raw(cv_root / "reports_ci" / "auroc_difference.csv")
    ci_rows = unique_index(ci, ("target", "model", "feature_set", "metric"), "ci_table")
    ref_rows = unique_index(ref, ("target", "section", "subject", "metric"), "reference_table")
    diff_rows = unique_index(diff, ("target", "comparison", "method"), "auroc_difference")
    need_columns(ci, ["estimate", "ci_low", "ci_high", "n_draws_used", "features_sha8", *SHARED_KEYS,
                      "draws_sha8"], "ci_table")  # fmt: skip
    need_columns(ref, ["estimate", "ci_low", "ci_high", "n_draws_used", "features_sha8", *SHARED_KEYS,
                       "draws_sha8"], "reference_table")  # fmt: skip
    need_columns(diff, ["role", "estimate", "ci_low", "ci_high", "p", "p_min", "p_max",
                        "n_draws_used", *SHARED_KEYS, "draws_sha8"], "auroc_difference")  # fmt: skip
    draws_used: set[str] = set()
    for target in targets:
        mods = models[target]
        shared = shared_of(mods)
        for m, fs in CI_MODELS:
            for metric in METRICS:
                where = f"ci_table {target} {label(m, fs)} {metric}"
                row = ci_rows.get((target, m, fs, metric))
                if row is None:
                    raise GuardFailure(f"{where}: row missing")
                check_shared(row, shared, where)
                if row["features_sha8"] != mods[(m, fs)]["meta"]["features_sha8"]:
                    raise GuardFailure(f"{where}: features_sha8 differs from the runs")
                check_estimate(row, mods[(m, fs)]["means"][metric], where)
                draws_used.add(row["n_draws_used"])
        for m, fs in REF_MODELS:
            for metric in CORE_METRICS:
                where = f"reference_table {target} {label(m, fs)} {metric}"
                row = ref_rows.get((target, "performance", label(m, fs), metric))
                if row is None:
                    raise GuardFailure(f"{where}: row missing")
                check_shared(row, shared, where)
                if row["features_sha8"] != mods[(m, fs)]["meta"]["features_sha8"]:
                    raise GuardFailure(f"{where}: features_sha8 differs from the runs")
                check_estimate(row, mods[(m, fs)]["means"][metric], where)
                draws_used.add(row["n_draws_used"])
        means = {k: v["means"]["auroc"] for k, v in mods.items()}
        for name, (a, b) in zip(
            REF_DIFFS, ((("lgbm", "A"), ("lgbm", "R")), (("lgbm", "R"), ("b1", "R"))), strict=True
        ):
            where = f"reference_table {target} {name}"
            row = ref_rows.get((target, "difference", name, "auroc"))
            if row is None:
                raise GuardFailure(f"{where}: row missing")
            check_shared(row, shared, where)
            check_estimate(row, means[a] - means[b], where)
            draws_used.add(row["n_draws_used"])
        subtract = {
            DIFF_COMPARISONS[0]: (("lgbm", "B"), ("lgbm", "A")),
            DIFF_COMPARISONS[1]: (("lgbm", "A"), ("b1", "A")),
        }
        for name in DIFF_COMPARISONS:
            for method in DIFF_METHODS:
                where = f"auroc_difference {target} {name} {method}"
                row = diff_rows.get((target, name, method))
                if row is None:
                    raise GuardFailure(f"{where}: row missing")
                check_shared(row, shared, where)
                minuend, subtrahend = subtract[name]
                check_estimate(row, means[minuend] - means[subtrahend], where)
                if method == DIFF_METHODS[0]:
                    draws_used.add(row["n_draws_used"])
    if len(draws_used) != 1 or "" in draws_used:
        raise GuardFailure(f"tables disagree on n_draws_used ({len(draws_used)} distinct values)")
    confirmatory = {(r["target"], r["comparison"], r["method"]) for r in diff.to_dict("records")
                    if r["role"] == "confirmatory"}  # fmt: skip
    expected = {k for k in OFFICIAL_LABEL if k[0] in targets}
    if confirmatory != expected:
        raise GuardFailure(
            f"confirmatory label on {len(confirmatory)} row(s), expected {len(expected)}"
        )
    return {"ci": ci_rows, "ref": ref_rows, "diff": diff_rows, "n_draws_used": draws_used.pop()}


# --------------------------------------------------------------------------- assembly


def f10(value: float) -> str:
    return "" if math.isnan(value) else format(value, ".10g")


def build_runs(
    models: dict[str, dict[tuple[str, str], dict[str, Any]]], targets: Sequence[str]
) -> pd.DataFrame:
    parts = []
    for target in targets:
        shared = shared_of(models[target])
        for key in MODELS:
            runs = models[target][key]["runs"][
                RUNS_COLUMNS
            ].copy()  # copied as text, not recomputed
            for column in ("roles_sha8", "target_status"):
                runs[column] = shared[column]
            parts.append(runs)
    return pd.concat(parts, ignore_index=True)[RUNS_OUT_COLUMNS]


def blank_row() -> dict[str, str]:
    return {c: "" for c in TABLE_COLUMNS}


def build_table(
    models: dict[str, dict[tuple[str, str], dict[str, Any]]],
    src: dict[str, Any],
    targets: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for target in targets:
        shared = shared_of(models[target])
        for m, fs in MODELS:
            for metric in METRICS:
                row = blank_row()
                row.update(section="performance", target=target, subject=label(m, fs), model=m,
                           feature_set=fs, metric=metric, **shared)  # fmt: skip
                row["estimate"] = f10(models[target][(m, fs)]["means"][metric])
                if fs == "R":
                    source = src["ref"].get((target, "performance", label(m, fs), metric))
                    source_name = "reference_table"
                else:
                    source = src["ci"][(target, m, fs, metric)]
                    source_name = "ci_table"
                if source is not None and source["ci_low"] != "":
                    row.update(ci_low=source["ci_low"], ci_high=source["ci_high"],
                               n_draws_used=source["n_draws_used"], ci_source=source_name,
                               draws_sha8=source["draws_sha8"])  # fmt: skip
                rows.append(row)
    for target in targets:
        shared = shared_of(models[target])
        for name in DIFF_COMPARISONS:
            for method in DIFF_METHODS:
                s = src["diff"][(target, name, method)]
                row = blank_row()
                row.update(section="difference", target=target, subject=name, metric="auroc",
                           ci_source="auroc_difference", role=s["role"], method=method,
                           draws_sha8=s["draws_sha8"], **shared)  # fmt: skip
                for c in ("estimate", "ci_low", "ci_high", "p", "p_min", "p_max", "n_draws_used"):
                    row[c] = s[c]
                rows.append(row)
        for name in REF_DIFFS:
            s = src["ref"][(target, "difference", name, "auroc")]
            row = blank_row()
            row.update(section="difference", target=target, subject=name, metric="auroc",
                       ci_source="reference_table", role="descriptive", method="participant_bootstrap",
                       draws_sha8=s["draws_sha8"], **shared)  # fmt: skip
            for c in ("estimate", "ci_low", "ci_high", "n_draws_used"):
                row[c] = s[c]
            rows.append(row)
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def build_quality(features_path: Path) -> pd.DataFrame:
    if not features_path.is_file():
        raise InputProblem("missing input file features.csv")
    header = pd.read_csv(features_path, nrows=0).columns
    need_columns(pd.DataFrame(columns=header), FEATURE_COLUMNS, "features")
    f = pd.read_csv(features_path, usecols=FEATURE_COLUMNS)
    n = len(f)
    reasons = f["nan_reasons_b"].astype("string")
    rows: list[tuple[str, str, int, str]] = [
        ("context", "clips_total", n, "clips in features.csv"),
        ("context", "clips_above_45_fps", int((f["fps"] > 45).sum()), "recorded at 60 fps"),
        ("context", "qc_retained_flag", int((f["qc_status"] == "retained_flag").sum()),
         "retained by QC with a flag"),
        ("context", "one_sided_taps_excluded", int((f["n_one_sided_excluded"] > 0).sum()),
         "n_one_sided_excluded > 0"),
        ("context", "intervals_excluded", int((f["n_iti_excluded"] > 0).sum()), "n_iti_excluded > 0"),
        ("context", "short_intervals_under_0.2s", int((f["n_short_iti"] > 0).sum()), "n_short_iti > 0"),
        ("context", "set_b_unavailable_n_valid_lt_9",
         int(reasons.str.contains("n_valid<9", regex=False).fillna(False).sum()),
         "nan_reasons_b names n_valid<9"),
        ("context", "set_b_unavailable_n_intervals_lt_8",
         int(reasons.str.contains("n_intervals<8", regex=False).fillna(False).sum()),
         "nan_reasons_b names n_intervals<8"),
    ]  # fmt: skip
    for column in cvs.SET_B_ADDED:
        note = "NaN count (a missing value, never zero)"
        if column == "decrement_onset_frac":
            note += "; missing by design when no decrement is detected"
        rows.append(("set_b_nan", column, int(f[column].isna().sum()), note))
    return pd.DataFrame([(g, i, c, n, d) for g, i, c, d in rows], columns=QUALITY_COLUMNS)


# --------------------------------------------------------------------------- output


def csv_bytes(frame: pd.DataFrame, comment: str | None = None) -> bytes:
    text = frame.to_csv(index=False, lineterminator="\n", na_rep="")
    return ((f"# {comment}\n" if comment else "") + text).encode()


def _atomic_many(items: Sequence[tuple[Path, bytes]]) -> None:
    staged: list[tuple[Path, Path]] = []
    try:
        for final, data in items:
            final.parent.mkdir(parents=True, exist_ok=True)
            tmp = final.with_name(f".{final.name}.tmp{os.getpid()}")
            staged.append((tmp, final))
            tmp.write_bytes(data)
        for tmp, final in staged:
            os.replace(tmp, final)
    finally:
        for tmp, _ in staged:
            if tmp.exists():
                tmp.unlink()


def flt(value: str) -> float:
    return float("nan") if value == "" else float(value)


def num(value: str) -> str:
    return "-" if value == "" or flt(value) != flt(value) else f"{flt(value):.4f}"


def render(table: pd.DataFrame, shared: dict[str, dict[str, str]], n_used: str) -> str:
    perf = table[table["section"] == "performance"]
    diff = table[table["section"] == "difference"]
    lines: list[str] = []
    for target in [t for t in TARGET_ORDER if t in set(perf["target"])]:
        p = perf[perf["target"] == target]
        s = shared[target]
        lines.append(f"== {target}  splits {s['splits_sha8']}  inner {s['inner_sha8']}  roles "
                     f"{s['roles_sha8']}  status {s['target_status']}  draws used {n_used}")  # fmt: skip
        lines.append("AUROC [95% CI]")
        for m, fs in MODELS:
            r = p[(p["subject"] == label(m, fs)) & (p["metric"] == "auroc")].iloc[0]
            ci = f"[{num(r['ci_low'])}, {num(r['ci_high'])}]"
            lines.append(f"  {label(m, fs):<8} {num(r['estimate'])} {ci}")
        lines.append("  balanced_acc  sensitivity  specificity       mcc")
        for m, fs in MODELS:
            vals = [p[(p["subject"] == label(m, fs)) & (p["metric"] == k)].iloc[0]["estimate"]
                    for k in CORE_METRICS[1:]]  # fmt: skip
            lines.append(f"  {label(m, fs):<8} " + " ".join(f"{num(v):>11}" for v in vals))
        lines.append("PPV at 1% / 2% / 5% prevalence (estimate; CI in the csv where one exists)")
        lines.append("           ppv_1pct    ppv_2pct    ppv_5pct")
        for m, fs in MODELS:
            vals = [p[(p["subject"] == label(m, fs)) & (p["metric"] == k)].iloc[0]["estimate"]
                    for k in PPV_METRICS]  # fmt: skip
            lines.append(f"  {label(m, fs):<8} " + " ".join(f"{num(v):>11}" for v in vals))
        lines.append("")
    lines.append(
        "AUROC differences (bootstrap or DeLong; DeLong p = median over 5 repeats, min-max)"
    )
    lines.append(f"{'tgt':<4}{'comparison':<20}{'role':<13}{'method':<10}{'diff':>8}{'ci_low':>8}"
                 f"{'ci_high':>8}{'p':>8}{'p_min':>8}{'p_max':>8}")  # fmt: skip
    for r in diff.to_dict("records"):
        method = "bootstrap" if r["method"] == "participant_bootstrap" else r["method"]
        lines.append(f"{r['target']:<4}{r['subject']:<20}{r['role']:<13}{method:<10}"
                     f"{num(r['estimate']):>8}{num(r['ci_low']):>8}{num(r['ci_high']):>8}"
                     f"{num(r['p']):>8}{num(r['p_min']):>8}{num(r['p_max']):>8}")  # fmt: skip
    lines.append("")
    lines += textwrap.wrap("Notes: " + NOTES, width=98)
    return "\n".join(lines)


def sha8_file(path: Path) -> str:
    return cvs.sha8_of(path.read_bytes())


def build_meta(cv_root: Path, features: Path, targets: Sequence[str], src: dict[str, Any],
               models: dict[str, dict[tuple[str, str], dict[str, Any]]], counts: dict[str, int]) -> dict[str, Any]:  # fmt: skip
    inputs = {f"{t}/{kind}_{m}_{fs}.{ext}": sha8_file(cv_root / t / f"{kind}_{m}_{fs}.{ext}")
              for t in targets for m, fs in MODELS for kind, ext in (("runs", "csv"), ("meta", "json"))}  # fmt: skip
    for name in ("ci_table.csv", "auroc_difference.csv", "reference_table.csv"):
        inputs[f"reports_ci/{name}"] = sha8_file(cv_root / "reports_ci" / name)
    inputs["features.csv"] = sha8_file(features)
    return {
        "tool": TOOL, "repeats": REPEATS, "draws_sha8": EXPECTED_DRAWS_SHA8,
        "n_draws_used": src["n_draws_used"], "targets": list(targets),
        "models": [label(m, fs) for m, fs in MODELS], "row_counts": counts,
        "participants": {t: models[t][MODELS[0]]["meta"].get("participants") for t in targets},
        "shared": {t: shared_of(models[t]) for t in targets},
        "inputs_sha8": dict(sorted(inputs.items())), "notes_sha8": cvs.sha8_of(NOTES),
    }  # fmt: skip


def run(cv_root: Path, features: Path, out_dir: Path, targets: Sequence[str]) -> str:
    order = [t for t in TARGET_ORDER if t in set(targets)]
    models = {t: load_target(cv_root, t) for t in order}
    src = source_tables(cv_root, order, models)
    runs, table, quality = (
        build_runs(models, order),
        build_table(models, src, order),
        build_quality(features),
    )
    counts = {OUT_RUNS: len(runs), OUT_TABLE: len(table), OUT_QUALITY: len(quality)}
    meta = build_meta(cv_root, features, order, src, models, counts)
    text = render(table, {t: shared_of(models[t]) for t in order}, str(src["n_draws_used"]))
    _atomic_many([
        (out_dir / OUT_RUNS, csv_bytes(runs)),
        (out_dir / OUT_TABLE, csv_bytes(table, NOTES)),
        (out_dir / OUT_QUALITY, csv_bytes(quality, QUALITY_HEADER)),
        (out_dir / OUT_META, (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode()),
    ])  # fmt: skip
    return text + "\n" + "\n".join(f"wrote {k}: {v} rows" for k, v in counts.items())


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cv-root", type=Path, default=None)
    ap.add_argument("--features", type=Path, default=None)
    ap.add_argument("--targets", nargs="+", choices=["ge2", "ge1"], default=list(TARGET_ORDER))
    ap.add_argument("--out-dir", type=Path, default=None)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cv_root = Path(args.cv_root or cvs.CV_ROOT)
    out_dir = Path(args.out_dir or cv_root / "results")
    features = Path(args.features or cvs.PROCESSED_ROOT / "features.csv")
    for name, path in (("--cv-root", cv_root), ("--out-dir", out_dir)):
        if not cvs.is_inside(path, cvs.CV_ROOT):
            print(f"error: {name} must be inside data/processed/cv", file=sys.stderr)
            return 2
    try:
        text = run(cv_root, features, out_dir, args.targets)
    except InputProblem as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except GuardFailure as err:
        print(f"refused: {err}", file=sys.stderr)
        return 1
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
