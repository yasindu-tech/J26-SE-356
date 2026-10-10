"""GAIT-07 / Check A: confirm the CARE-PD joint format on the real files.

Run from the repo root (venv active):

    python models/gait/src/check_carepd_format.py

For each cohort it checks 5 evenly spaced walks against the rules in
``models/gait/configs/carepd_format.toml``, summarises every walk in the file,
and (by default) joins the walks to the severity labels in the cohort ``.pkl``.
Exit code 0 = Check A passed, 1 = a rule failed, 2 = a file is missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from carepd_format import (
    FormatConfig,
    cohort_violations,
    label_coverage,
    load_config,
    load_label_pickle,
    sample_keys,
    short_id,
    summarise_cohort,
    walk_metrics,
    walk_violations,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "carepd_format.toml"
DEFAULT_DATA = REPO_ROOT / "data" / "raw" / "CARE-PD"


def check_cohort(
    name: str, data_dir: Path, cfg: FormatConfig, use_labels: bool
) -> tuple[bool, bool]:
    """Check one cohort. Returns ``(passed, files_found)``."""
    cohort = cfg.cohorts[name]
    world = data_dir / "h36m" / name / cohort.world_file
    print(f"\n=== {name} (fps {cohort.fps}) ===")
    if not world.is_file():
        print(f"  MISSING: {world}")
        return False, False

    with np.load(world, allow_pickle=False) as npz:
        walks = {k: npz[k] for k in npz.files}
    passed = True

    # 1. The 5 sample walks.
    print(f"  sample walks (evenly spaced, {cfg.checks.n_sample_walks}):")
    print("    id          frames  height  floor  knee-hip  hip L-R  forward   result")
    for key in sample_keys(list(walks), cfg.checks.n_sample_walks):
        w = walks[key]
        problems = walk_violations(w, cfg)
        if w.ndim == 3 and w.shape[1:] == (17, 3) and np.isfinite(w).all():
            m = walk_metrics(w)
            nums = (
                f"{m.n_frames:6d}  {m.height_m:5.2f}m  {m.floor_offset_m:+5.2f}  "
                f"{m.knee_minus_hip_m:+7.2f}  {m.hip_separation_m:+7.2f}  {m.net_forward_m:+7.2f}"
            )
        else:
            nums = "(unreadable walk)"
        verdict = "PASS" if not problems else "FAIL: " + "; ".join(problems)
        print(f"    {short_id(key)}  {nums}   {verdict}")
        passed &= not problems

    # 2. The whole file.
    s = summarise_cohort(walks, cfg)
    print(f"  file: {s.n_entries} entries = {s.n_walks} distinct walks, {s.n_subjects} subjects")
    print(
        f"  all walks: median height {s.median_height_m:.2f} m | floor ok {s.frac_floor_ok:.1%} | "
        f"Y up {s.frac_knee_ok:.1%} | +x is left {s.frac_hip_ok:.1%} | moves +z {s.frac_forward_ok:.1%}"
    )
    copies = s.n_entries / s.n_walks
    if cohort.has_down_copies != (copies > 1):
        print(
            f"  FAIL: config has_down_copies={cohort.has_down_copies} but file has {copies:.1f} entries/walk"
        )
        passed = False
    for problem in cohort_violations(s, cfg):
        print(f"  FAIL: {problem}")
        passed = False

    # 3. Labels.
    if use_labels:
        label_path = data_dir / cohort.label_file
        if not label_path.is_file():
            print(f"  MISSING labels: {label_path}")
            return False, False
        cov = label_coverage(list(walks), load_label_pickle(label_path), cfg)
        print(
            f"  labels: {cov.n_label_walks} walks in {cohort.label_file}; "
            f"{cov.h36m_without_label} h36m walks have no label; "
            f"{cov.labels_without_h36m} labelled walks have no h36m walk (CARE-PD drops walks < 30 frames)"
        )
        print(f"  severity classes (walk-level, 2-3 merged): {cov.class_counts}")
        if cov.h36m_without_label:
            print(
                "  FAIL: h36m walks without labels. The h36m file and the .pkl are from different releases."
            )
            passed = False

    print(f"  -> {'PASS' if passed else 'FAIL'}")
    return passed, True


def main() -> int:
    parser = argparse.ArgumentParser(description="CARE-PD h36m format check (GAIT-07)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cohorts", nargs="+", help="default: every cohort in the config")
    parser.add_argument("--labels", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    names = args.cohorts or list(cfg.cohorts)
    unknown = [n for n in names if n not in cfg.cohorts]
    if unknown:
        print(f"Unknown cohort(s): {unknown}. Known: {list(cfg.cohorts)}")
        return 2

    results = [check_cohort(n, args.data_dir, cfg, args.labels) for n in names]
    passed = [p for p, _ in results]
    if not all(found for _, found in results):
        print("\nCheck A: INCOMPLETE (some files are missing)")
        return 2
    print(f"\nCheck A: {'PASS' if all(passed) else 'FAIL'} ({sum(passed)}/{len(passed)} cohorts)")
    return 0 if all(passed) else 1


if __name__ == "__main__":
    sys.exit(main())
