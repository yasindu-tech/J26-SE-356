"""Paired AUROC differences between models (task FT-39). Nothing is re-fitted.

    python models/tapping/src/auroc_difference.py [--targets ge2 ge1]

Comparisons per target: lgbm B minus lgbm A (confirmatory on ge2 only) and lgbm A minus b1 A
(descriptive, no multiplicity adjustment). Primary method: the participant-cluster bootstrap of
bootstrap_ci (same 2000 draws, seed 2026, shared by both models), difference of AUROC per repeat on
the weighted clips, mean over the 5 repeats, 95% percentile interval, a bootstrap two-sided p.
Secondary method: DeLong's paired test per repeat on the pooled out-of-fold predictions; it is
anti-conservative here (two clips per participant, repeats not independent). Refusals as in FT-38.
Writes auroc_difference.csv. Exit codes: 0 ok, 1 refused, 2 missing input or a path outside the folder.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import bootstrap_ci as bc
import cv_splits as cvs
import numpy as np
import pandas as pd
from cv_splits import GuardFailure, InputProblem
from scipy.stats import norm

MODELS_USED = (("b1", "A"), ("lgbm", "A"), ("lgbm", "B"))
COMPARISONS = ((("lgbm", "B"), ("lgbm", "A")), (("lgbm", "A"), ("b1", "A")))  # minuend, subtrahend
METHODS = ("participant_bootstrap", "delong")
CONFIRMATORY = ("ge2", 0, "participant_bootstrap")  # target, comparison index, method
OUT_TABLE = "auroc_difference.csv"
TABLE_COLUMNS = [
    "target", "comparison", "role", "method", "estimate", "ci_low", "ci_high", "p", "p_min",
    "p_max", "n_draws_used", "n_draws_finite", "n_excluded", "n_boot", "seed", "draws_sha8",
    "splits_sha8", "inner_sha8", "roles_sha8", "target_status", "features_sha8_minuend",
    "features_sha8_subtrahend",
]  # fmt: skip


def label(key: tuple[str, str]) -> str:
    return f"{key[0]} {key[1]}"


def comparison_name(index: int) -> str:
    minuend, subtrahend = COMPARISONS[index]
    return f"{label(minuend)} minus {label(subtrahend)}"


# --------------------------------------------------------------------------- DeLong


def delong_components(y: np.ndarray, proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Structural components: V10 per positive (share of negatives it beats, ties half) and V01 per
    negative (share of positives that beat it)."""
    pos, neg = proba[y == 1], proba[y == 0]
    psi = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    return psi.mean(axis=1), psi.mean(axis=0)


def delong_test(y: np.ndarray, proba_1: np.ndarray, proba_2: np.ndarray) -> dict[str, float]:
    """DeLong, DeLong and Clarke-Pearson paired test of AUC(1) - AUC(2) on the same cases."""
    y = np.asarray(y)
    m, n = int((y == 1).sum()), int((y == 0).sum())
    if m < 2 or n < 2:
        raise GuardFailure("DeLong needs at least two cases of each class")
    c1, c2 = delong_components(y, proba_1), delong_components(y, proba_2)
    v10, v01 = np.vstack([c1[0], c2[0]]), np.vstack([c1[1], c2[1]])
    auc = v10.mean(axis=1)
    s10, s01 = np.cov(v10), np.cov(v01)  # sample covariances of the components (ddof=1)
    cov = s10 / m + s01 / n  # covariance matrix of (AUC1, AUC2)
    diff = float(auc[0] - auc[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if var <= 1e-15:
        z, p = float("nan"), (1.0 if diff == 0 else float("nan"))  # identical models: no difference
    else:
        z = diff / np.sqrt(var)
        p = float(2 * norm.sf(abs(z)))
    return {
        "auc_1": float(auc[0]),
        "auc_2": float(auc[1]),
        "diff": diff,
        "var": var,
        "z": float(z),
        "p": p,
    }


# --------------------------------------------------------------------------- bootstrap


def bootstrap_p(draws: np.ndarray) -> float:
    """min(1, 2 * min((1 + #{d <= 0}) / (B + 1), (1 + #{d >= 0}) / (B + 1))); approximate."""
    b = len(draws)
    low = (1 + int((draws <= 0).sum())) / (b + 1)
    high = (1 + int((draws >= 0).sum())) / (b + 1)
    return float(min(1.0, 2 * min(low, high)))


def target_rows(
    cv_root: Path, target: str, ref_table: pd.DataFrame, counts: np.ndarray, order: list[str]
) -> list[dict[str, Any]]:
    models = {k: bc.load_model(cv_root, target, k[0], k[1], ref_table["ref"]) for k in MODELS_USED}
    shared = bc.check_agreement(target, models)
    y = models[MODELS_USED[0]]["y"]
    index = {p: i for i, p in enumerate(order)}
    clip_participant = np.array([index[p] for p in ref_table["participant"]])
    weights = counts[:, clip_participant].astype(float)  # the same clip weights for every model
    valid = (weights @ (y == 1) > 0) & (weights @ (y == 0) > 0)
    n_excluded = int((~valid).sum())
    if n_excluded > bc.MAX_EXCLUDED_SHARE * len(counts):
        raise GuardFailure(
            f"{target}: {n_excluded} of {len(counts)} draws lack a class in a repeat, more than {bc.MAX_EXCLUDED_SHARE:.0%}"
        )
    weights = weights[valid]
    ones = np.ones((1, len(y)))
    boot: dict[tuple[str, str], np.ndarray] = {}
    point: dict[tuple[str, str], np.ndarray] = {}
    for key, entry in models.items():
        boot[key] = np.array(
            [bc.weighted_auroc(y, p, weights) for p in entry["probas"]]
        )  # (repeats, draws)
        point[key] = np.array(
            [bc.weighted_auroc(y, p, ones)[0] for p in entry["probas"]]
        )  # (repeats,)
        reported = entry["runs"].sort_values("repeat")["auroc"].to_numpy(dtype=float)
        if abs(point[key].mean() - reported.mean()) > 1e-9:
            raise GuardFailure(f"{target}: {label(key)}: oof and runs disagree on auroc")
    common = {
        "target": target, "n_boot": len(counts), "seed": bc.SEED, "draws_sha8": bc.draws_sha8(counts),
        "splits_sha8": shared["splits_sha8"], "inner_sha8": shared["inner_sha8"],
        "roles_sha8": shared["roles_sha8"], "target_status": shared["target_status"],
    }  # fmt: skip
    rows: list[dict[str, Any]] = []
    for i, (minuend, subtrahend) in enumerate(COMPARISONS):
        features = {
            "features_sha8_minuend": models[minuend]["meta"]["features_sha8"],
            "features_sha8_subtrahend": models[subtrahend]["meta"]["features_sha8"],
        }
        estimate = float((point[minuend] - point[subtrahend]).mean())
        draws = (boot[minuend] - boot[subtrahend]).mean(axis=0)
        low, high, n_finite = bc.interval(draws)
        p_boot = bootstrap_p(draws[np.isfinite(draws)]) if n_finite else float("nan")
        per_repeat = [
            delong_test(y, models[minuend]["probas"][r], models[subtrahend]["probas"][r])
            for r in range(bc.EXPECTED_REPEATS)
        ]
        diffs = np.array([r["diff"] for r in per_repeat])
        ps = np.array([r["p"] for r in per_repeat])
        if abs(diffs.mean() - estimate) > 1e-9:
            raise GuardFailure(
                f"{target}: DeLong and bootstrap point estimates disagree for {comparison_name(i)}"
            )
        base = {**common, **features, "comparison": comparison_name(i)}
        rows.append({**base, "role": "confirmatory" if (target, i, METHODS[0]) == CONFIRMATORY else "descriptive",
                     "method": METHODS[0], "estimate": estimate, "ci_low": low, "ci_high": high, "p": p_boot,
                     "n_draws_used": int(valid.sum()), "n_draws_finite": n_finite, "n_excluded": n_excluded})  # fmt: skip
        rows.append({**base, "role": "descriptive", "method": METHODS[1], "estimate": float(diffs.mean()),
                     "p": float(np.median(ps)), "p_min": float(ps.min()), "p_max": float(ps.max())})  # fmt: skip
    return rows


def build_table(
    cv_root: Path, features_path: Path, targets: Sequence[str]
) -> tuple[pd.DataFrame, int, int]:
    ref_table = bc.read_participants(features_path)
    order = bc.sorted_participants(ref_table["participant"].tolist())
    counts = bc.bootstrap_draws(order, bc.N_DRAWS, bc.SEED)
    rows: list[dict[str, Any]] = []
    for target in sorted(set(targets)):
        rows += target_rows(cv_root, target, ref_table, counts, order)
    table = pd.DataFrame(rows, columns=TABLE_COLUMNS)
    names = [comparison_name(i) for i in range(len(COMPARISONS))]
    table["_c"] = [names.index(c) for c in table["comparison"]]
    table["_m"] = [METHODS.index(m) for m in table["method"]]
    table = (
        table.sort_values(["target", "_c", "_m"], kind="stable")
        .drop(columns=["_c", "_m"])
        .reset_index(drop=True)
    )
    two_clips = int((ref_table.groupby("participant").size() >= 2).sum())
    return table, len(order), two_clips


# --------------------------------------------------------------------------- output


def note(n_participants: int) -> str:
    return (
        f"Differences are over {n_participants} participants; 5 repeats are not independent evidence; "
        "only lgbm B minus lgbm A on ge2 is confirmatory; the rest are descriptive."
    )


def render(table: pd.DataFrame, n_participants: int, two_clips: int) -> str:
    first = table.iloc[0]
    lines = [
        f"Paired AUROC differences (participant bootstrap: {int(first['n_boot'])} draws, seed {int(first['seed'])}; DeLong per repeat)"
    ]
    for target, part in table.groupby("target", sort=True):
        r = part.iloc[0]
        lines.append(f"{target}: splits_sha8 {r['splits_sha8']}, inner_sha8 {r['inner_sha8']}, roles_sha8 {r['roles_sha8']}, target_status {r['target_status']}, draws_sha8 {r['draws_sha8']}; draws used {int(part['n_draws_used'].dropna().iloc[0])}, excluded {int(part['n_excluded'].dropna().iloc[0])}")  # fmt: skip
    header = ["target", "comparison", "role", "method", "difference", "95% CI", "p"]
    grid = []
    for r in table.itertuples(index=False):
        is_boot = r.method == METHODS[0]
        ci = f"[{bc.num(r.ci_low)}, {bc.num(r.ci_high)}]" if is_boot else "-"
        p = bc.num(r.p) if is_boot else f"{bc.num(r.p)} (range {bc.num(r.p_min)}-{bc.num(r.p_max)})"
        grid.append([r.target, r.comparison, r.role, r.method, bc.num(r.estimate), ci, p])
    widths = [max(len(header[c]), *(len(row[c]) for row in grid)) for c in range(len(header))]
    lines.append("  ".join(h.ljust(w) for h, w in zip(header, widths, strict=True)))
    lines += ["  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) for row in grid]
    lines.append(
        "The bootstrap p is approximate and descriptive; no multiplicity adjustment across comparisons."
    )
    lines.append(
        f"DeLong is anti-conservative here: {two_clips} of {n_participants} participants contribute two clips (independence violated) and the 5 repeats are not independent."
    )
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
        table, n_participants, two_clips = build_table(cv_root, features, args.targets)
        data = table_bytes(table)
        text = render(table, n_participants, two_clips)
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
