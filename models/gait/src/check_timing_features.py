"""GAIT-14: compute the timing features on every CARE-PD walk that passes the cycle check.

Run from the repository root::

    python models/gait/src/check_timing_features.py

Prints, per cohort and severity class, the median stride time, cadence and stride-time
variability, then three checks:

- stride time and cadence are plausible in every cohort (median stride time 0.8-2.0 s);
- the features pass the feature contract (no label or banned input);
- stride-time variability is not mostly a stride-count signal: its rank correlation with the
  number of strides must stay below ``MAX_CV_STRIDE_RHO`` (0.50 with all strides, 0.32 with the
  middle-6 window; the stride count differs by severity, see ``docs/timing_features.md``).

Writes ``artifacts/timing/timing_features.csv`` (git-ignored): one row per walk with features.
``n_strides`` is in the table for checking only; it is not a model input.
Exit code 0 = all checks pass, 1 = a check fails, 2 = data files are missing.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from carepd_format import FormatConfig, load_config
from carepd_loader import DEFAULT_CONFIG, DEFAULT_DATA, load_all
from check_cycles import CLASS_NAMES, check_all
from cycles import CAREPD_MIN_STRIDES
from feature_contract import FeatureContractError, check_feature_contract
from scipy.stats import spearmanr
from timing_features import FEATURE_NAMES, WINDOW_STRIDES, TimingFeatures, timing_features

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "artifacts" / "timing"
PLAUSIBLE_STRIDE_S = (0.8, 2.0)  # median stride time of a cohort
MAX_CV_STRIDE_RHO = 0.4  # variability may not track the stride count more than this
LABEL_NAMES = ["UPDRS_GAIT", "severity", "score"]


def _median(values: list[float]) -> str:
    return f"{np.median(values):.2f}" if values else "-"


def _rho(a: list[float], b: list[int]) -> float:
    if len(a) < 3 or len(set(b)) < 2 or len(set(a)) < 2:
        return float("nan")
    return float(spearmanr(a, b)[0])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Timing features on CARE-PD (GAIT-14)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-strides", type=int, default=CAREPD_MIN_STRIDES)
    parser.add_argument(
        "--no-rank-gate", action="store_true", help="skip the stride-count check (tiny test data)"
    )
    args = parser.parse_args(argv)

    cfg: FormatConfig = load_config(args.config)
    try:
        walks, _ = load_all(args.data_dir, cfg)
    except FileNotFoundError as err:
        print(f"MISSING: {err}")
        return 2
    checks = check_all(walks, args.min_strides)
    rows = [(w, timing_features(c)) for w, c in zip(walks, checks, strict=True) if c.passed]
    skipped = len(walks) - len(rows)

    print(
        f"Timing features on {len(rows)} walks that pass the cycle check "
        f"({skipped} fail it and get none)."
    )
    print(
        f"Variability: corrected CV over the middle {WINDOW_STRIDES} strides. "
        "Cells: median (walks).\n"
    )
    print(
        f"{'cohort':<10}{'class':<11}{'walks':>6}{'stride time (s)':>17}"
        f"{'cadence (steps/min)':>21}{'stride time CV (%)':>20}"
    )
    status = 0
    cohorts = list(dict.fromkeys(w.cohort for w, _ in rows))
    for cohort in cohorts:
        for k, name in CLASS_NAMES.items():
            sel = [f for w, f in rows if w.cohort == cohort and w.severity == k]
            if not sel:
                continue
            print(
                f"{cohort:<10}{name:<11}{len(sel):>6}"
                f"{_median([f.stride_time_mean for f in sel]):>17}"
                f"{_median([f.cadence for f in sel]):>21}"
                f"{_median([f.stride_time_cv for f in sel]):>20}"
            )
    print()
    for cohort in cohorts:
        med = float(np.median([f.stride_time_mean for w, f in rows if w.cohort == cohort]))
        lo, hi = PLAUSIBLE_STRIDE_S
        if not lo <= med <= hi:
            print(f"  -> {cohort}: median stride time {med:.2f} s is outside {lo}-{hi} s")
            status = 1

    try:
        check_feature_contract(FEATURE_NAMES, label_name=LABEL_NAMES)
        print(f"Feature contract: passed for {', '.join(FEATURE_NAMES)}.")
    except FeatureContractError as err:
        print(f"Feature contract FAILED: {err}")
        status = 1

    feats: list[TimingFeatures] = [f for _, f in rows]
    cv = [f.stride_time_cv for f in feats]
    n = [f.n_strides for f in feats]
    rho_all = _rho(cv, n)
    per_class = []
    for k, name in CLASS_NAMES.items():
        idx = [i for i, (w, _) in enumerate(rows) if w.severity == k]
        per_class.append(f"{name} {_rho([cv[i] for i in idx], [n[i] for i in idx]):.2f}")
    print(
        f"Variability vs stride count (rank correlation): all {rho_all:.2f}; "
        + "; ".join(per_class)
        + f" (limit {MAX_CV_STRIDE_RHO})."
    )
    if not args.no_rank_gate and rho_all > MAX_CV_STRIDE_RHO:
        print("  -> variability follows the stride count too closely")
        status = 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "timing_features.csv"
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["walk", "cohort", "score", "severity", "n_strides", *FEATURE_NAMES])
        for w, f in rows:
            writer.writerow(
                [w.tag, w.cohort, w.score, w.severity, f.n_strides]
                + [f"{v:.4f}" for v in f.as_dict().values()]
            )
    print(f"\nPer-walk table (n_strides is for checking, not a feature): {out}")
    return status


if __name__ == "__main__":
    sys.exit(main())
