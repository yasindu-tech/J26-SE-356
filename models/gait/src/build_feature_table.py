"""GAIT-17: run ``extract_features`` on every CARE-PD walk and write the training table.

Run from the repository root (venv active)::

    python models/gait/src/build_feature_table.py

Every walk goes through the same function the demo uses (``gait_extract.extract_features``).
Walks that give the eight features go to ``artifacts/features/feature_table.csv``; walks that
cannot go to ``artifacts/features/rejected_walks.csv`` with the reason. Nothing is imputed.
Both files are git-ignored. Subject IDs are never printed or written: the table carries a
``subject_key`` (a short hash of cohort and subject) for the subject-wise folds of GAIT-18.

Columns of the feature table, in groups:

- meta: ``walk`` (short hash), ``cohort``, ``subject_key``, ``severity`` (label), ``score`` (label);
- checks (never model inputs): ``n_strides``, ``n_left``, ``n_right``, ``leg_length_m``,
  ``lr_difference``, ``lr_flag``;
- the eight model inputs, in the order of ``gait_extract.FEATURE_NAMES``;
- the two experimental features (not model inputs).

The report counts walks, rejections and flags per cohort and class, because rejection depends on
severity and GAIT-18 and GAIT-20 need to know. Checks (exit 1 if one fails): the eight names pass
the feature contract, and every feature value is finite.
Exit code 0 = ok, 1 = a check failed, 2 = data files are missing.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

from carepd_format import FormatConfig, load_config, short_id
from carepd_loader import DEFAULT_CONFIG, DEFAULT_DATA, Walk, load_all
from check_cycles import CLASS_NAMES
from cycles import CAREPD_MIN_STRIDES
from feature_contract import FeatureContractError, check_feature_contract
from gait_extract import FEATURE_NAMES, LR_FLAG, WalkFeatures, extract_features
from gait_features import EXPERIMENTAL_NAMES

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "artifacts" / "features"
LABEL_NAMES = ["UPDRS_GAIT", "severity", "score"]
META = ["walk", "cohort", "subject_key", "severity", "score"]
CHECKS = ["n_strides", "n_left", "n_right", "leg_length_m", "lr_difference", "lr_flag"]

Result = tuple[Walk, WalkFeatures]


def subject_key(w: Walk) -> str:
    return short_id(w.group)


def table_row(w: Walk, r: WalkFeatures) -> list[object]:
    assert r.checks is not None  # ok walks always carry their checks
    c = r.checks
    lr = "" if math.isnan(c.lr_difference) else f"{c.lr_difference:.4f}"
    return [
        w.tag,
        w.cohort,
        subject_key(w),
        w.severity,
        w.score,
        c.n_strides,
        c.n_left,
        c.n_right,
        f"{c.leg_length_m:.4f}",
        lr,
        int(bool(r.flags)),
        *[f"{r.features[n]:.6f}" for n in FEATURE_NAMES],
        *[f"{r.experimental[n]:.6f}" for n in EXPERIMENTAL_NAMES],
    ]


def report(results: list[Result]) -> None:
    print(
        f"{'cohort':<10}{'class':<11}{'walks':>6}{'ok':>6}{'rejected':>9}{'rej %':>7}{'flagged':>8}"
    )
    cohorts = list(dict.fromkeys(w.cohort for w, _ in results))
    for cohort in cohorts:
        for k, label in CLASS_NAMES.items():
            sel = [r for w, r in results if w.cohort == cohort and w.severity == k]
            if not sel:
                continue
            ok = [r for r in sel if r.ok]
            rej = len(sel) - len(ok)
            flagged = sum(bool(r.flags) for r in ok)
            print(
                f"{cohort:<10}{label:<11}{len(sel):>6}{len(ok):>6}{rej:>9}"
                f"{rej / len(sel):>7.0%}{flagged:>8}"
            )
    print()
    print("Subjects with at least one usable walk (each is one unit in the GAIT-18 folds):")
    for cohort in cohorts:
        cells = []
        for k, label in CLASS_NAMES.items():
            subs = {
                subject_key(w)
                for w, r in results
                if r.ok and w.cohort == cohort and w.severity == k
            }
            cells.append(f"{label} {len(subs)}")
        print(f"  {cohort:<10}" + ", ".join(cells))
    all_subs = {subject_key(w) for w, r in results if r.ok}
    print(f"  total {len(all_subs)} subjects")
    reasons = Counter(r.reason for _, r in results if not r.ok)
    print("\nWhy walks were rejected:")
    for reason, n in reasons.most_common():
        print(f"  {n:>5}  {reason}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CARE-PD feature table (GAIT-17)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-strides", type=int, default=CAREPD_MIN_STRIDES)
    args = parser.parse_args(argv)

    cfg: FormatConfig = load_config(args.config)
    try:
        walks, _ = load_all(args.data_dir, cfg)
    except FileNotFoundError as err:
        print(f"MISSING: {err}")
        return 2
    results: list[Result] = [
        (w, extract_features(w.joints, float(w.fps), min_strides=args.min_strides)) for w in walks
    ]
    ok = [(w, r) for w, r in results if r.ok]
    print(
        f"{len(walks)} walks: {len(ok)} give the eight features, {len(walks) - len(ok)} rejected."
    )
    print(f"Flag = left/right stride times differ by more than {LR_FLAG:.0%}.\n")
    report(results)

    status = 0
    try:
        check_feature_contract(FEATURE_NAMES, label_name=LABEL_NAMES)
        print(f"\nFeature contract: passed for {len(FEATURE_NAMES)} features.")
    except FeatureContractError as err:
        print(f"\nFeature contract FAILED: {err}")
        status = 1
    bad = [w.tag for w, r in ok if not all(math.isfinite(r.features[n]) for n in FEATURE_NAMES)]
    if bad:
        print(f"FAILED: {len(bad)} walks have a non-finite feature (first: {bad[0]})")
        status = 1
    else:
        print("All feature values are finite.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = args.out_dir / "feature_table.csv"
    with table.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([*META, *CHECKS, *FEATURE_NAMES, *EXPERIMENTAL_NAMES])
        writer.writerows(table_row(w, r) for w, r in ok)
    rejected = args.out_dir / "rejected_walks.csv"
    with rejected.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["walk", "cohort", "subject_key", "severity", "score", "reason"])
        writer.writerows(
            [w.tag, w.cohort, subject_key(w), w.severity, w.score, r.reason]
            for w, r in results
            if not r.ok
        )
    print(f"\nFeature table ({len(ok)} rows): {table}")
    print(f"Rejected walks with reasons: {rejected}")
    return status


if __name__ == "__main__":
    sys.exit(main())
