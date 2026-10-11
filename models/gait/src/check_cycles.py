"""GAIT-13: segment gait cycles on every CARE-PD walk and apply the cycle check (Rule B).

Run from the repo root (venv active):

    python models/gait/src/check_cycles.py

Prints, per cohort and severity class, how many walks pass the cycle check, so any imbalance
the check introduces is visible. Writes a per-walk table (short hashes only) and five overlay
plots for the check by eye to ``models/gait/artifacts/cycles`` (git-ignored): one walk per
cohort and one UPDRS 3 walk, picked by hash order among passing walks (not the nicest ones).

Exit code 0 = done and class 2-3 keeps at least 10 subjects (the GAIT-08 rule), 1 = it does
not, 2 = a file is missing.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from carepd_format import FormatConfig, load_config
from carepd_loader import DEFAULT_CONFIG, DEFAULT_DATA, Walk, load_all
from check_heel_strikes import plot_overlay
from count_labels import MIN_SUBJECTS_TOP_CLASS
from cycles import CAREPD_MIN_STRIDES, MIN_STEP_EXCURSION_M, CycleCheck, segment_cycles
from preprocess import detect_heel_strikes

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "artifacts" / "cycles"
CLASS_NAMES = {0: "class 0", 1: "class 1", 2: "class 2-3"}


def check_all(walks: list[Walk], min_strides: int = CAREPD_MIN_STRIDES) -> list[CycleCheck]:
    return [segment_cycles(detect_heel_strikes(w.joints, w.fps), min_strides) for w in walks]


def eye_check_walks(walks: list[Walk], checks: list[CycleCheck]) -> list[Walk]:
    """One passing walk per cohort plus one passing UPDRS 3 walk, first in hash order."""
    passing = [w for w, c in zip(walks, checks, strict=True) if c.passed]
    picks: list[Walk] = []
    for cohort in dict.fromkeys(w.cohort for w in walks):
        pool = sorted((w for w in passing if w.cohort == cohort), key=lambda w: w.tag)
        if pool:
            picks.append(pool[0])
    severe = sorted((w for w in passing if w.score == 3 and w not in picks), key=lambda w: w.tag)
    if severe:
        picks.append(severe[0])
    return picks


def _kept(walks: list[Walk], checks: list[CycleCheck], severity: int) -> str:
    pairs = [(w, c) for w, c in zip(walks, checks, strict=True) if w.severity == severity]
    if not pairs:
        return "-"
    kept = sum(c.passed for _, c in pairs)
    return f"{kept}/{len(pairs)} ({kept / len(pairs):.0%})"


def write_table(path: Path, walks: list[Walk], checks: list[CycleCheck]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        out = csv.writer(f)
        out.writerow(
            ["walk", "cohort", "score", "severity", "fps", "frames", "strides_left",
             "strides_right", "cycles_bad_duration", "cycles_not_real_step", "passed"]
        )  # fmt: skip
        for w, c in zip(walks, checks, strict=True):
            out.writerow(
                [w.tag, w.cohort, w.score, w.severity, w.fps, w.n_frames, c.strides("left"),
                 c.strides("right"), c.bad_duration, c.not_real_step, int(c.passed)]
            )  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gait cycles and the cycle check (GAIT-13)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-strides", type=int, default=CAREPD_MIN_STRIDES)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    cfg: FormatConfig = load_config(args.config)
    try:
        walks, _ = load_all(args.data_dir, cfg)
    except FileNotFoundError as err:
        print(f"MISSING: {err}")
        return 2
    checks = check_all(walks, args.min_strides)

    print(
        f"Cycle check: pass = at least {args.min_strides} real strides counting both feet (Rule B)"
    )
    print(
        "A real stride lasts 0.5-3.0 s and swings the ankle at least "
        f"{MIN_STEP_EXCURSION_M:.2f} m past the body (not stepping on the spot)."
    )
    print("Cells: walks that pass / walks (share kept)\n")
    print(
        f"{'cohort':<10}{'walks':>7}{'pass':>7}" + "".join(f"{n:>18}" for n in CLASS_NAMES.values())
    )
    for cohort in [*dict.fromkeys(w.cohort for w in walks), "ALL"]:
        idx = [i for i, w in enumerate(walks) if cohort == "ALL" or w.cohort == cohort]
        ws = [walks[i] for i in idx]
        cs = [checks[i] for i in idx]
        print(
            f"{cohort:<10}{len(ws):>7}{sum(c.passed for c in cs):>7}"
            + "".join(f"{_kept(ws, cs, k):>18}" for k in CLASS_NAMES)
        )

    print("\nSubjects with at least one passing walk:")
    status = 0
    for k, name in CLASS_NAMES.items():
        subj = {w.group for w, c in zip(walks, checks, strict=True) if c.passed and w.severity == k}
        print(f"  {name}: {len(subj)}")
        if k == 2 and len(subj) < MIN_SUBJECTS_TOP_CLASS:
            print(f"  -> class 2-3 has fewer than {MIN_SUBJECTS_TOP_CLASS} subjects left")
            status = 1

    print("\nCycles not counted, share of all cycles:")
    print(f"  {'':<11}{'outside 0.5-3.0 s':>20}{'not a real step':>18}")
    for k, name in CLASS_NAMES.items():
        cs = [c for w, c in zip(walks, checks, strict=True) if w.severity == k]
        total = sum(len(c.cycles) for c in cs) or 1
        dur = sum(c.bad_duration for c in cs)
        step = sum(c.not_real_step for c in cs)
        print(f"  {name:<11}{f'{dur} ({dur / total:.1%})':>20}{f'{step} ({step / total:.1%})':>18}")

    table = args.out_dir / "cycle_check.csv"
    write_table(table, walks, checks)
    print(f"\nPer-walk table: {table}")
    if not args.no_plot:
        for i, w in enumerate(eye_check_walks(walks, checks), start=1):
            out = args.out_dir / f"eye_check_{i}_{w.cohort}.png"
            plot_overlay([w], out)
            print(f"Eye-check plot {i}: {out} (walk {w.tag}, {w.cohort}, UPDRS gait {w.score})")
    return status


if __name__ == "__main__":
    sys.exit(main())
