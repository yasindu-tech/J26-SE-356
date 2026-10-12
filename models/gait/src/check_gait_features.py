"""GAIT-15: compute the step, joint-range and arm-swing features on every CARE-PD walk that passes.

Run from the repository root::

    python models/gait/src/check_gait_features.py

Prints, per cohort and severity class, the median of the five model features (and of the two
experimental ones, which are not model inputs), then these checks:

- the features pass the feature contract (no label or banned input);
- no model feature is mostly a stride-count signal: its rank correlation with the number of
  strides stays below ``MAX_STRIDE_RHO`` (0.4, as for GAIT-14);
- direction: in every cohort with enough walks of both classes, class 2-3 has shorter steps,
  smaller knee, hip and arm-swing ranges and a larger step asymmetry than class 0 (the clinical
  expectation; a cohort that reverses it points to a bug or a data problem).

Writes ``artifacts/gait/gait_features.csv`` (git-ignored): one row per walk. ``n_strides`` and
``leg_length_m`` are in the table for checking only; they are not model inputs.
Exit code 0 = all checks pass, 1 = a check fails, 2 = data files are missing.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from carepd_format import FormatConfig, load_config
from carepd_loader import DEFAULT_CONFIG, DEFAULT_DATA, Walk, load_all
from check_cycles import CLASS_NAMES, check_all
from cycles import CAREPD_MIN_STRIDES
from feature_contract import FeatureContractError, check_feature_contract
from gait_features import EXPERIMENTAL_NAMES, FEATURE_NAMES, GaitFeatures, gait_features
from scipy.stats import spearmanr

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "artifacts" / "gait"
MAX_STRIDE_RHO = 0.4  # a feature may not track the stride count more than this
MIN_WALKS_PER_CLASS = 5  # a cohort needs this many walks in class 0 and in class 2-3 to be judged
LABEL_NAMES = ["UPDRS_GAIT", "severity", "score"]
# +1: class 2-3 should have a LARGER median than class 0; -1: a smaller one.
EXPECTED_DIRECTION = {
    "step_length_norm": -1,
    "step_length_asymmetry": +1,
    "knee_flexion_range_deg": -1,
    "hip_flexion_range_deg": -1,
    "arm_swing_range_norm": -1,
}
SHORT = {
    "step_length_norm": "step len",
    "step_length_asymmetry": "step asym",
    "knee_flexion_range_deg": "knee deg",
    "hip_flexion_range_deg": "hip deg",
    "arm_swing_range_norm": "arm swing",
    "step_width_norm": "step width",
    "arm_swing_asymmetry": "arm asym",
}

Row = tuple[Walk, GaitFeatures]


def _med(rows: list[Row], name: str) -> float:
    return float(np.median([getattr(f, name) for _, f in rows]))


def _rho(a: list[float], b: list[int]) -> float:
    if len(a) < 3 or len(set(b)) < 2 or len(set(a)) < 2:
        return float("nan")
    return float(spearmanr(a, b)[0])


def _table(rows: list[Row], names: tuple[str, ...], cohorts: list[str]) -> None:
    print(f"{'cohort':<10}{'class':<11}{'walks':>6}" + "".join(f"{SHORT[n]:>12}" for n in names))
    for cohort in cohorts:
        for k, label in CLASS_NAMES.items():
            sel = [r for r in rows if r[0].cohort == cohort and r[0].severity == k]
            if sel:
                cells = "".join(f"{_med(sel, n):>12.3f}" for n in names)
                print(f"{cohort:<10}{label:<11}{len(sel):>6}{cells}")
    print()


def _direction_problems(rows: list[Row], cohorts: list[str]) -> list[str]:
    problems = []
    top = max(CLASS_NAMES)
    for cohort in cohorts:
        low = [r for r in rows if r[0].cohort == cohort and r[0].severity == 0]
        high = [r for r in rows if r[0].cohort == cohort and r[0].severity == top]
        if min(len(low), len(high)) < MIN_WALKS_PER_CLASS:
            continue
        for name, sign in EXPECTED_DIRECTION.items():
            if sign * (_med(high, name) - _med(low, name)) <= 0:
                problems.append(f"{cohort}: {name} goes the wrong way in class 2-3")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gait features on CARE-PD (GAIT-15)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-strides", type=int, default=CAREPD_MIN_STRIDES)
    parser.add_argument(
        "--no-rank-gate", action="store_true", help="skip the stride-count check (tiny test data)"
    )
    parser.add_argument(
        "--no-direction-gate", action="store_true", help="skip the direction check (tiny test data)"
    )
    args = parser.parse_args(argv)

    cfg: FormatConfig = load_config(args.config)
    try:
        walks, _ = load_all(args.data_dir, cfg)
    except FileNotFoundError as err:
        print(f"MISSING: {err}")
        return 2
    checks = check_all(walks, args.min_strides)
    rows: list[Row] = []
    unusable = 0
    for w, c in zip(walks, checks, strict=True):
        if not c.passed:
            continue
        try:
            rows.append((w, gait_features(w.joints, w.fps, c)))
        except ValueError:
            unusable += 1  # e.g. real strides on one foot only
    failed = len(walks) - len(rows) - unusable
    print(
        f"Gait features on {len(rows)} walks that pass the cycle check "
        f"({failed} fail it, {unusable} more cannot give features)."
    )
    print("Cells: median over the walks of the cohort and class.\n")
    cohorts = list(dict.fromkeys(w.cohort for w, _ in rows))
    print("Model features (step length and arm swing in leg lengths):")
    _table(rows, FEATURE_NAMES, cohorts)
    print("Experimental, NOT model inputs (see docs/gait_features.md):")
    _table(rows, EXPERIMENTAL_NAMES, cohorts)
    print(
        "Leg length (check only, m): "
        + "; ".join(
            f"{c} {_med([r for r in rows if r[0].cohort == c], 'leg_length_m'):.2f}"
            for c in cohorts
        )
    )

    status = 0
    try:
        check_feature_contract(FEATURE_NAMES, label_name=LABEL_NAMES)
        print(f"Feature contract: passed for {', '.join(FEATURE_NAMES)}.")
    except FeatureContractError as err:
        print(f"Feature contract FAILED: {err}")
        status = 1

    n = [f.n_strides for _, f in rows]
    cells = []
    for name in FEATURE_NAMES:
        rho = _rho([getattr(f, name) for _, f in rows], n)
        cells.append(f"{SHORT[name]} {rho:.2f}")
        if not args.no_rank_gate and abs(rho) > MAX_STRIDE_RHO:
            print(f"  -> {name} follows the stride count too closely ({rho:.2f})")
            status = 1
    print(
        f"Features vs stride count (rank correlation, limit {MAX_STRIDE_RHO}): " + "; ".join(cells)
    )

    if not args.no_direction_gate:
        problems = _direction_problems(rows, cohorts)
        for p in problems:
            print(f"  -> {p}")
        if problems:
            status = 1
        else:
            print("Direction: class 2-3 vs class 0 goes the expected way in every cohort checked.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "gait_features.csv"
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["walk", "cohort", "score", "severity", "n_strides", "leg_length_m"]
            + list(FEATURE_NAMES)
            + list(EXPERIMENTAL_NAMES)
        )
        for w, f in rows:
            writer.writerow(
                [w.tag, w.cohort, w.score, w.severity, f.n_strides, f"{f.leg_length_m:.4f}"]
                + [f"{v:.4f}" for v in f.as_dict().values()]
                + [f"{v:.4f}" for v in f.experimental_dict().values()]
            )
    print(f"\nPer-walk table (n_strides, leg_length_m are for checking, not features): {out}")
    return status


if __name__ == "__main__":
    sys.exit(main())
