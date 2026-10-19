"""Participant-level bootstrap CIs for the official runs (task FT-38). Nothing is re-fitted.

    python models/tapping/src/bootstrap_ci.py [--targets ge2 ge1]

Pre-registered: resample the participants with replacement, 2000 draws, seed 2026, 95% percentile
interval. One draw set is shared by all 5 repeats and by every model and feature set of a target
(paired, reusable by FT-39). A draw's statistic is the mean over repeats of the metric on that
repeat's pooled out-of-fold predictions, with each clip weighted by its participant's multiplicity.
Draws where a repeat lacks a class are excluded for every model; more than 1% excluded is refused.
Writes ci_table.csv. Exit codes: 0 ok, 1 refused, 2 missing input or a path outside the folder.
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

SEED = 2026
N_DRAWS = 2000
CI_PERCENTILES = (2.5, 97.5)
MAX_EXCLUDED_SHARE = 0.01
EXPECTED_REPEATS = 5
THRESHOLD = 0.5  # the harness's fixed decision threshold
PPV_PREVALENCES = (0.01, 0.02, 0.05)
MODELS = (("b0", "A"), ("b1", "A"), ("lgbm", "A"), ("lgbm", "B"))
METRICS = (
    "auroc", "balanced_accuracy", "sensitivity", "specificity", "mcc",
    "ppv_at_1pct", "ppv_at_2pct", "ppv_at_5pct",
)  # fmt: skip
AGREE_KEYS = ("splits_sha8", "inner_sha8", "roles_sha8", "target_status")
OUT_TABLE = "ci_table.csv"
TABLE_COLUMNS = [
    "target", "model", "feature_set", "metric", "estimate", "ci_low", "ci_high", "n_draws_used",
    "n_draws_finite", "n_excluded", "n_boot", "seed", "draws_sha8", "splits_sha8", "inner_sha8",
    "roles_sha8", "features_sha8", "target_status",
]  # fmt: skip
OOF_COLUMNS = ["ref", "repeat", "outer_fold", "y", "proba"]


# --------------------------------------------------------------------------- the draws


def sorted_participants(participants: Sequence[str]) -> list[str]:
    """The participant order every draw refers to: sorted, unique."""
    unique = sorted(set(participants))
    if not unique:
        raise GuardFailure("no participants")
    return unique


def bootstrap_draws(
    participants: Sequence[str], n_draws: int = N_DRAWS, seed: int = SEED
) -> np.ndarray:
    """(n_draws, n_participants) multiplicities: how often each participant (sorted order) is drawn.

    Deterministic from (seed, n_draws, the set of participants); this is the function FT-39 imports."""
    order = sorted_participants(participants)
    if n_draws < 1:
        raise GuardFailure("n_draws must be at least 1")
    n = len(order)
    picks = np.random.default_rng(seed).integers(0, n, size=(n_draws, n))
    counts = np.zeros((n_draws, n), dtype=np.int64)
    np.add.at(counts, (np.arange(n_draws)[:, None], picks), 1)
    return counts


def draws_sha8(counts: np.ndarray) -> str:
    return cvs.sha8_of(np.ascontiguousarray(counts, dtype=np.int64).tobytes())


# --------------------------------------------------------------------------- weighted metrics


def weighted_auroc(y: np.ndarray, proba: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """AUROC (ties 0.5) of each weighted copy of the data; weights is (draws, clips)."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    wp, wn = weights[:, pos], weights[:, neg]
    sp, sn = proba[pos][:, None], proba[neg][None, :]
    c = (sp > sn).astype(float) + 0.5 * (sp == sn)
    den = wp.sum(1) * wn.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, ((wp @ c) * wn).sum(1) / den, np.nan)


def weighted_threshold_metrics(
    y: np.ndarray, proba: np.ndarray, weights: np.ndarray
) -> dict[str, np.ndarray]:
    pred, pos = proba >= THRESHOLD, y == 1
    tp, fn = weights @ (pred & pos), weights @ (~pred & pos)
    tn, fp = weights @ (~pred & ~pos), weights @ (pred & ~pos)
    with np.errstate(invalid="ignore", divide="ignore"):
        sens = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        spec = np.where(tn + fp > 0, tn / (tn + fp), np.nan)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = np.where(denominator > 0, (tp * tn - fp * fn) / denominator, 0.0)
        out = {
            "balanced_accuracy": (sens + spec) / 2,
            "sensitivity": sens,
            "specificity": spec,
            "mcc": mcc,
        }
        for p in PPV_PREVALENCES:
            den = sens * p + (1 - spec) * (1 - p)
            out[f"ppv_at_{round(p * 100)}pct"] = np.where(den == 0, np.nan, sens * p / den)
    return out


def statistic(y: np.ndarray, probas: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
    """Mean over repeats of every metric; probas is (repeats, clips). NaN propagates."""
    per_repeat = []
    for proba in probas:
        metrics = weighted_threshold_metrics(y, proba, weights)
        metrics["auroc"] = weighted_auroc(y, proba, weights)
        per_repeat.append(metrics)
    return {m: np.mean([r[m] for r in per_repeat], axis=0) for m in METRICS}


def interval(values: np.ndarray) -> tuple[float, float, int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan"), 0
    low, high = np.percentile(finite, CI_PERCENTILES)
    return float(low), float(high), int(finite.size)


# --------------------------------------------------------------------------- inputs


def read_participants(features_path: Path) -> pd.DataFrame:
    if not features_path.is_file():
        raise InputProblem("missing input: features file")
    frame = pd.read_csv(
        features_path, dtype=str, usecols=["ref", "participant"], keep_default_na=False
    )
    if frame["ref"].duplicated().any() or (frame["ref"] == "").any():
        raise GuardFailure("features: ref must be present and unique")
    if (frame["participant"] == "").any():
        raise GuardFailure(
            f"features: {int((frame['participant'] == '').sum())} clip(s) have no participant"
        )
    return frame.sort_values("ref").reset_index(drop=True)


def load_model(
    cv_root: Path, target: str, model: str, feature_set: str, refs: pd.Series
) -> dict[str, Any]:
    folder, tag, where = (
        cv_root / target,
        f"{model}_{feature_set}",
        f"{target} {model} {feature_set}",
    )
    paths = {
        "oof": folder / f"oof_{tag}.csv",
        "runs": folder / f"runs_{tag}.csv",
        "meta": folder / f"meta_{tag}.json",
    }
    for kind, path in paths.items():
        if not path.is_file():
            raise InputProblem(f"missing input: {kind} file for {where}")
    meta = json.loads(paths["meta"].read_text())
    if (meta.get("target"), meta.get("model"), meta.get("feature_set")) != (
        target,
        model,
        feature_set,
    ):
        raise GuardFailure(f"{where}: meta file is for another target, model or feature set")
    if meta.get("repeats") != EXPECTED_REPEATS:
        raise GuardFailure(
            f"{where}: meta has {meta.get('repeats')} repeats, expected {EXPECTED_REPEATS}"
        )
    oof = pd.read_csv(paths["oof"], dtype={"ref": str}, float_precision="round_trip")
    if list(oof.columns) != OOF_COLUMNS:
        raise GuardFailure(f"{where}: oof file has unexpected columns")
    if sorted(oof["repeat"].unique().tolist()) != list(range(EXPECTED_REPEATS)):
        raise GuardFailure(f"{where}: oof file does not hold repeats 0..{EXPECTED_REPEATS - 1}")
    if not oof["y"].isin([0, 1]).all() or not np.isfinite(oof["proba"]).all():
        raise GuardFailure(f"{where}: oof has a missing label or probability")
    known = set(refs)
    unknown = int((~oof["ref"].isin(known)).sum())
    if unknown:
        raise GuardFailure(
            f"{where}: {unknown} oof row(s) have a clip that is not in the features file"
        )
    probas, labels = [], []
    for repeat in range(EXPECTED_REPEATS):
        part = oof[oof["repeat"] == repeat]
        if len(part) != len(refs) or part["ref"].duplicated().any() or set(part["ref"]) != known:
            raise GuardFailure(
                f"{where}: repeat {repeat} does not predict every clip exactly once ({len(part)} rows, {len(refs)} clips)"
            )
        part = part.set_index("ref").loc[refs]
        probas.append(part["proba"].to_numpy(dtype=float))
        labels.append(part["y"].to_numpy(dtype=int))
    if any(not np.array_equal(labels[0], other) for other in labels[1:]):
        raise GuardFailure(f"{where}: the labels differ between repeats")
    runs = pd.read_csv(paths["runs"], float_precision="round_trip")
    if sorted(runs["repeat"].tolist()) != list(range(EXPECTED_REPEATS)):
        raise GuardFailure(f"{where}: runs file does not hold exactly one row per repeat")
    y = labels[0]
    if (runs["n_pos"] != y.sum()).any() or (runs["n_clips"] != len(y)).any():
        raise GuardFailure(f"{where}: oof and runs disagree on the label counts")
    return {"meta": meta, "y": y, "probas": np.array(probas), "runs": runs}


def check_agreement(target: str, models: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    first = next(iter(models.values()))
    shared = {k: first["meta"].get(k) for k in AGREE_KEYS}
    for (model, fs), entry in models.items():
        for key in AGREE_KEYS:
            if entry["meta"].get(key) != shared[key]:
                raise GuardFailure(f"{target}: {model} {fs} differs from the other models in {key}")
        if not np.array_equal(entry["y"], first["y"]):
            raise GuardFailure(f"{target}: {model} {fs} has different labels from the other models")
    return shared


def target_rows(
    cv_root: Path, target: str, ref_table: pd.DataFrame, counts: np.ndarray, order: list[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    refs = ref_table["ref"]
    models = {(m, fs): load_model(cv_root, target, m, fs, refs) for m, fs in MODELS}
    shared = check_agreement(target, models)
    index = {p: i for i, p in enumerate(order)}
    clip_participant = np.array([index[p] for p in ref_table["participant"]])
    y = next(iter(models.values()))["y"]
    weights = counts[:, clip_participant].astype(float)  # (draws, clips); the same for every model
    valid = (weights @ (y == 1) > 0) & (weights @ (y == 0) > 0)
    n_excluded = int((~valid).sum())
    if n_excluded > MAX_EXCLUDED_SHARE * len(counts):
        raise GuardFailure(
            f"{target}: {n_excluded} of {len(counts)} draws lack a class in a repeat, more than {MAX_EXCLUDED_SHARE:.0%}"
        )
    weights = weights[valid]
    hashes = draws_sha8(counts)
    rows: list[dict[str, Any]] = []
    for (model, fs), entry in models.items():
        point = statistic(entry["y"], entry["probas"], np.ones((1, len(y))))
        boot = statistic(entry["y"], entry["probas"], weights)
        for metric in METRICS:
            reported = entry["runs"][metric].to_numpy(dtype=float).mean()
            estimate = float(point[metric][0])
            if not (np.isnan(estimate) and np.isnan(reported)) and abs(estimate - reported) > 1e-9:
                raise GuardFailure(f"{target}: {model} {fs}: oof and runs disagree on {metric}")
            low, high, n_finite = interval(boot[metric])
            rows.append({
                "target": target, "model": model, "feature_set": fs, "metric": metric,
                "estimate": estimate, "ci_low": low, "ci_high": high, "n_draws_used": int(valid.sum()),
                "n_draws_finite": n_finite, "n_excluded": n_excluded, "n_boot": len(counts),
                "seed": SEED, "draws_sha8": hashes, **shared, "features_sha8": entry["meta"]["features_sha8"],
            })  # fmt: skip
    return rows, {
        **shared,
        "n_excluded": n_excluded,
        "draws_sha8": hashes,
        "n_draws_used": int(valid.sum()),
    }


def build_table(
    cv_root: Path, features_path: Path, targets: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], int]:
    ref_table = read_participants(features_path)
    order = sorted_participants(ref_table["participant"].tolist())
    counts = bootstrap_draws(order, N_DRAWS, SEED)
    rows: list[dict[str, Any]] = []
    info: dict[str, dict[str, Any]] = {}
    for target in sorted(set(targets)):
        r, i = target_rows(cv_root, target, ref_table, counts, order)
        rows += r
        info[target] = i
    table = pd.DataFrame(rows, columns=TABLE_COLUMNS)
    model_rank = {m: i for i, m in enumerate(MODELS)}
    table["_m"] = [
        model_rank[(m, f)] for m, f in zip(table["model"], table["feature_set"], strict=True)
    ]
    table["_k"] = [METRICS.index(m) for m in table["metric"]]
    table = (
        table.sort_values(["target", "_m", "_k"], kind="stable")
        .drop(columns=["_m", "_k"])
        .reset_index(drop=True)
    )
    return table, info, len(order)


# --------------------------------------------------------------------------- output


def note(n_participants: int) -> str:
    return f"CIs are over {n_participants} participants; 5 repeats are not independent evidence; coverage is approximate with this sample size."


def num(value: float) -> str:
    return "-" if value != value else f"{value:.4f}"


def render(table: pd.DataFrame, info: dict[str, dict[str, Any]], n_participants: int) -> str:
    lines = [
        f"Participant-bootstrap 95% CIs ({N_DRAWS} draws, seed {SEED}; point estimate = mean over the 5 repeats)"
    ]
    for target, i in info.items():
        lines.append(
            f"{target}: "
            + ", ".join(f"{k} {i[k]}" for k in (*AGREE_KEYS, "draws_sha8"))
            + f"; draws used {i['n_draws_used']}, excluded {i['n_excluded']}"
        )
        part = table[table["target"] == target]
        grid = [[f"{m} {fs}"] for m, fs in MODELS]
        header = ["model"] + list(METRICS)
        for k, (m, fs) in enumerate(MODELS):
            for metric in METRICS:
                r = part[
                    (part["model"] == m) & (part["feature_set"] == fs) & (part["metric"] == metric)
                ].iloc[0]
                grid[k].append(f"{num(r['estimate'])} [{num(r['ci_low'])}, {num(r['ci_high'])}]")
        widths = [max(len(header[c]), *(len(row[c]) for row in grid)) for c in range(len(header))]
        lines.append("  ".join(h.ljust(w) for h, w in zip(header, widths, strict=True)))
        lines += ["  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) for row in grid]
    lines.append(note(n_participants))
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
    ap.add_argument("--features", type=Path, default=None)
    ap.add_argument("--targets", nargs="+", choices=sorted(cvs.TARGETS), default=["ge2", "ge1"])
    ap.add_argument("--out-dir", type=Path, default=None)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cv_root = Path(args.cv_root or cvs.CV_ROOT)
    out_dir = Path(args.out_dir or cv_root / "reports_ci")
    features = Path(args.features or cvs.PROCESSED_ROOT / "features.csv")
    for name, path in (("--cv-root", cv_root), ("--out-dir", out_dir)):
        if not cvs.is_inside(path, cvs.CV_ROOT):
            print(f"error: {name} must be inside data/processed/cv", file=sys.stderr)
            return 2
    try:
        table, info, n_participants = build_table(cv_root, features, args.targets)
        data = table_bytes(table)
        text = render(table, info, n_participants)
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
