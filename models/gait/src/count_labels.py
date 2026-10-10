"""GAIT-08 / Check B: count subjects and walks per severity class, per cohort.

Run from the repo root (venv active):

    python models/gait/src/count_labels.py
    python models/gait/src/count_labels.py --csv models/gait/docs/label_counts.csv

The labels come from the cohort ``.pkl`` files (field ``UPDRS_GAIT``). A walk is
*usable* when it also has a skeleton in the h36m world file. Walk copies named
``..._down0`` to ``..._down4`` count once. Only counts are printed, never subject
IDs (CLAUDE.md section 4).

Class rule from the plan: three classes (0, 1, 2-3). If the 2-3 class has fewer
than ``MIN_SUBJECTS_TOP_CLASS`` subjects in total, fall back to 0 versus 1-and-above.
Exit code 0 = counted, 2 = a file is missing.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from carepd_format import FormatConfig, load_config, load_label_pickle, split_key

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "carepd_format.toml"
DEFAULT_DATA = REPO_ROOT / "data" / "raw" / "CARE-PD"

# Plan, Check B: below about 10 subjects the top class is too small to learn or test.
MIN_SUBJECTS_TOP_CLASS = 10
TOP_CLASS = 2  # the merged "2-3" class


@dataclass(frozen=True)
class ClassCount:
    subjects: int
    walks: int


@dataclass(frozen=True)
class CohortCounts:
    name: str
    n_subjects: int
    n_labelled_walks: int  # walks in the .pkl, copies counted once
    n_usable_walks: int  # labelled walks that also have an h36m skeleton
    mixed_subjects: int  # subjects whose usable walks fall in more than one raw score
    raw: dict[int, ClassCount]  # UPDRS gait score 0, 1, 2, 3
    merged: dict[int, ClassCount]  # model classes 0, 1, 2 (2 means "2-3")


def usable_walks(h36m_keys: list[str]) -> set[tuple[str, str]]:
    """``(subject, walk)`` pairs present in the h36m file; ``_downK`` copies collapse to one."""
    return {split_key(k) for k in h36m_keys}


def count_cohort(
    name: str,
    labels: dict[Any, dict[str, dict[str, Any]]],
    usable: set[tuple[str, str]],
    cfg: FormatConfig,
) -> CohortCounts:
    """Count subjects and usable walks per raw score and per merged class."""
    subj_by_raw: dict[int, set[str]] = {}
    walks_by_raw: dict[int, int] = {}
    scores_by_subject: dict[str, set[int]] = {}
    n_labelled = 0
    for subject, walks in labels.items():
        sid = str(subject)
        for walk, fields in walks.items():
            value = fields.get(cfg.labels.field)
            if value is None:
                continue
            score = int(value)
            if score not in cfg.labels.class_map:
                raise ValueError(f"{name}: score {score} is not in the class map")
            n_labelled += 1
            if (sid, walk) not in usable:
                continue
            subj_by_raw.setdefault(score, set()).add(sid)
            walks_by_raw[score] = walks_by_raw.get(score, 0) + 1
            scores_by_subject.setdefault(sid, set()).add(score)

    raw = {s: ClassCount(len(subj_by_raw[s]), walks_by_raw[s]) for s in sorted(subj_by_raw)}

    merged_subj: dict[int, set[str]] = {}
    merged_walks: dict[int, int] = {}
    for score, subjects in subj_by_raw.items():
        cls = cfg.labels.class_map[score]
        merged_subj.setdefault(cls, set()).update(subjects)
        merged_walks[cls] = merged_walks.get(cls, 0) + walks_by_raw[score]
    merged = {c: ClassCount(len(merged_subj[c]), merged_walks[c]) for c in sorted(merged_subj)}

    return CohortCounts(
        name=name,
        n_subjects=len(labels),
        n_labelled_walks=n_labelled,
        n_usable_walks=sum(walks_by_raw.values()),
        mixed_subjects=sum(1 for s in scores_by_subject.values() if len(s) > 1),
        raw=raw,
        merged=merged,
    )


def pool(counts: list[CohortCounts], scheme: str) -> dict[int, ClassCount]:
    """Add up per-cohort counts. Subject IDs belong to one cohort, so subjects add up too."""
    total: dict[int, ClassCount] = {}
    for c in counts:
        for cls, cc in (c.raw if scheme == "raw" else c.merged).items():
            old = total.get(cls, ClassCount(0, 0))
            total[cls] = ClassCount(old.subjects + cc.subjects, old.walks + cc.walks)
    return dict(sorted(total.items()))


def choose_classes(pooled_merged: dict[int, ClassCount]) -> str:
    """Apply the plan's rule: keep 0 / 1 / 2-3 unless 2-3 has too few subjects."""
    top = pooled_merged.get(TOP_CLASS, ClassCount(0, 0))
    return "three-class" if top.subjects >= MIN_SUBJECTS_TOP_CLASS else "binary-0-vs-1plus"


# --------------------------------------------------------------------------- printing
RAW_NAMES = {0: "0", 1: "1", 2: "2", 3: "3"}
MERGED_NAMES = {0: "0", 1: "1", 2: "2-3"}


def _print_block(
    title: str, rows: list[tuple[str, dict[int, ClassCount]]], names: dict[int, str]
) -> None:
    classes = sorted(names)
    print(f"\n{title}")
    header = f"  {'cohort':<10}" + "".join(f"{'class ' + names[c]:>16}" for c in classes)
    print(header)
    print("  " + " " * 10 + "".join(f"{'subj / walks':>16}" for _ in classes))
    for label, per_class in rows:
        cells = []
        for c in classes:
            cc = per_class.get(c)
            cells.append(f"{cc.subjects} / {cc.walks}" if cc else "-")
        print(f"  {label:<10}" + "".join(f"{x:>16}" for x in cells))


def write_csv(path: Path, counts: list[CohortCounts]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cohort", "scheme", "class", "subjects", "walks"])
        for c in counts:
            for scheme, table, names in (
                ("raw", c.raw, RAW_NAMES),
                ("merged", c.merged, MERGED_NAMES),
            ):
                for cls, cc in table.items():
                    w.writerow([c.name, scheme, names[cls], cc.subjects, cc.walks])
        for scheme, names in (("raw", RAW_NAMES), ("merged", MERGED_NAMES)):
            for cls, cc in pool(counts, scheme).items():
                w.writerow(["ALL", scheme, names[cls], cc.subjects, cc.walks])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CARE-PD label counts (GAIT-08, Check B)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--csv", type=Path, default=None, help="also write the counts to this CSV")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    results: list[CohortCounts] = []
    for name, cohort in cfg.cohorts.items():
        world = args.data_dir / "h36m" / name / cohort.world_file
        labels_path = args.data_dir / cohort.label_file
        missing = [p for p in (world, labels_path) if not p.is_file()]
        if missing:
            for p in missing:
                print(f"MISSING: {p}")
            return 2
        with np.load(world, allow_pickle=False) as npz:
            usable = usable_walks(list(npz.files))
        labels = load_label_pickle(labels_path)
        results.append(count_cohort(name, labels, usable, cfg))

    print("Subjects / walks per class. Walk copies (_down0..4) count once; walks shorter than")
    print("30 frames have no skeleton and are left out.")
    _print_block(
        "UPDRS gait score as recorded (0, 1, 2, 3)",
        [(c.name, c.raw) for c in results] + [("ALL", pool(results, "raw"))],
        RAW_NAMES,
    )
    _print_block(
        "Model classes (2 and 3 merged)",
        [(c.name, c.merged) for c in results] + [("ALL", pool(results, "merged"))],
        MERGED_NAMES,
    )

    print("\nPer cohort:")
    for c in results:
        print(
            f"  {c.name:<10} {c.n_subjects} subjects | {c.n_labelled_walks} labelled walks | "
            f"{c.n_usable_walks} usable | {c.mixed_subjects} subjects have walks in more than one score"
        )
    merged_total = pool(results, "merged")
    total_walks = sum(cc.walks for cc in merged_total.values())
    print(
        f"\nPooled usable walks: {total_walks} from {sum(c.n_subjects for c in results)} subjects"
    )
    top = merged_total.get(TOP_CLASS, ClassCount(0, 0))
    choice = choose_classes(merged_total)
    print(
        f"Class 2-3 has {top.subjects} subjects (rule: keep three classes if >= {MIN_SUBJECTS_TOP_CLASS})."
        f"\nDecision: {choice}"
    )
    if args.csv:
        write_csv(args.csv, results)
        print(f"Wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
