"""GAIT-18: subject-wise cross-validation folds, shared by training and reporting (GAIT-19/20).

Run from the repository root (venv active, after ``build_feature_table.py``)::

    python models/gait/src/make_folds.py

Every subject has many walks (median 21), so a walk-level split lets the model recognise people
it has already seen. All walks of a subject therefore go to the same fold. The folds are made
once, saved, and reused, so every model and ablation is compared on the same splits.

Decisions (checked on the real feature table, 12 Oct 2026, see ``models/gait/docs/cv_folds.md``):

1. *Split by ``subject_key``* (a hash of cohort and subject). A random walk-level split scored
   macro-F1 0.71 against 0.555 with the subject-wise split on the same baseline model.
2. *``StratifiedGroupKFold`` on the walk-level class.* 59 of 109 subjects have walks in more than
   one class, so there is no single class per subject to stratify on; the walk-level class keeps
   the class shares of every test fold within about 3 points of the overall shares.
3. *5 folds, repeated with 5 seeds.* 5 folds gives test folds of 12-33 subjects (10 folds gives
   about 11). Report the spread over the 25 test folds (sd about 0.05 on the baseline): the
   spread over seeds (about 0.01) is smaller, because every seed reshuffles the same 109 people,
   so it understates the uncertainty about new people.
4. *Validity gate.* Every test fold needs at least ``MIN_TEST_SUBJECTS_PER_CLASS`` subjects with
   walks of each class. None of 50 test folds on the real data failed; if one ever does, the
   script exits 1.
5. *Cohort mix is reported, not forced.* A cohort's share of a test fold varies a lot (PD-GaM
   32-81%, 3DGait 0-8%), so per-cohort scores must be pooled over all folds (out-of-fold
   predictions), never taken per fold.
6. *Walk weights are saved, off by default.* ``walk_weight`` = 1 / (walks of the subject),
   scaled to a mean of 1. On the baseline it moved the subject-weighted macro-F1 by about 0.01,
   well inside the fold-to-fold spread, so it is an option for GAIT-19/20, not a default.

Writes ``artifacts/folds/fold_assignments.csv`` (git-ignored): one row per subject and seed.
The folds are deterministic for a given feature table and scikit-learn version; the printed
fingerprint lets team members check that they have the same ones.
Exit code 0 = ok, 1 = a fold failed the validity gate, 2 = the feature table is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DEFAULT_TABLE = Path(__file__).resolve().parents[1] / "artifacts" / "features" / "feature_table.csv"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "artifacts" / "folds"
N_FOLDS = 5
N_SEEDS = 5
MIN_TEST_SUBJECTS_PER_CLASS = 3
CLASSES = (0, 1, 2)
CLASS_LABELS = {0: "class 0", 1: "class 1", 2: "class 2-3"}
REQUIRED = ("subject_key", "cohort", "severity")


def read_table(path: Path) -> pd.DataFrame:
    """The columns of the feature table the folds need, in the table's row order."""
    return pd.read_csv(path, usecols=list(REQUIRED), dtype={"subject_key": str, "cohort": str})


def check_table(table: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED if c not in table.columns]
    if missing:
        raise ValueError(f"table lacks the columns {missing}")
    if table.empty:
        raise ValueError("the table has no walks")
    if not set(table["severity"]) <= set(CLASSES):
        raise ValueError(f"severity must be in {CLASSES}")
    if (table.groupby("subject_key")["cohort"].nunique() > 1).any():
        raise ValueError("a subject_key belongs to more than one cohort")


def subject_summary(table: pd.DataFrame) -> pd.DataFrame:
    """One row per subject: cohort, number of walks, and the per-walk weight (mean 1 over walks)."""
    grouped = table.groupby("subject_key", sort=True)
    out = pd.DataFrame({"cohort": grouped["cohort"].first(), "n_walks": grouped.size()})
    out["walk_weight"] = len(table) / len(out) / out["n_walks"]
    return out.reset_index()


def make_assignments(
    table: pd.DataFrame, n_folds: int = N_FOLDS, seeds: tuple[int, ...] | None = None
) -> pd.DataFrame:
    """Fold of every subject for every seed: columns seed, subject_key, cohort, n_walks,
    walk_weight, fold."""
    check_table(table)
    seeds = tuple(range(N_SEEDS)) if seeds is None else seeds
    subjects = subject_summary(table)
    groups = table["subject_key"].to_numpy()
    y = table["severity"].to_numpy()
    x = np.zeros((len(table), 1))
    parts = []
    for seed in seeds:
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_of: dict[str, int] = {}
        for fold, (_, test) in enumerate(splitter.split(x, y, groups)):
            for key in np.unique(groups[test]):
                fold_of[str(key)] = fold
        part = subjects.copy()
        part["fold"] = part["subject_key"].map(fold_of)
        part.insert(0, "seed", seed)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def walk_folds(table: pd.DataFrame, assignments: pd.DataFrame, seed: int) -> np.ndarray:
    """The test fold of every walk of ``table`` for this seed."""
    mine = assignments[assignments["seed"] == seed]
    if mine.empty:
        raise ValueError(f"no folds for seed {seed}")
    mapping = mine.set_index("subject_key")["fold"]
    if mapping.index.has_duplicates:
        raise ValueError(f"a subject has more than one fold for seed {seed}")
    folds = table["subject_key"].map(mapping)
    if folds.isna().any() or set(mapping.index) != set(table["subject_key"]):
        raise ValueError("the table and the fold assignments have different subjects: rebuild them")
    return folds.to_numpy(dtype=int)


def load_fold(
    table: pd.DataFrame, assignments: pd.DataFrame, seed: int, fold: int
) -> tuple[np.ndarray, np.ndarray]:
    """Row positions ``(train, test)`` in ``table`` for one fold of one seed."""
    folds = walk_folds(table, assignments, seed)
    if fold not in set(folds):
        raise ValueError(f"seed {seed} has no fold {fold}")
    return np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)


def iter_folds(
    table: pd.DataFrame, assignments: pd.DataFrame, seed: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """``(train, test)`` row positions for every fold of one seed, in fold order."""
    folds = walk_folds(table, assignments, seed)
    for fold in sorted(set(folds)):
        yield np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)


def walk_weights(table: pd.DataFrame) -> np.ndarray:
    """Optional training weight per walk: 1 / (walks of its subject), scaled to a mean of 1."""
    n = table.groupby("subject_key")["subject_key"].transform("size").to_numpy(dtype=float)
    w = 1.0 / n
    return w * len(w) / w.sum()


def fold_problems(table: pd.DataFrame, assignments: pd.DataFrame) -> list[str]:
    """Anything wrong with the folds: leakage, empty folds, or classes too thin to test."""
    problems = []
    for seed in sorted(set(assignments["seed"])):
        folds = walk_folds(table, assignments, seed)  # also checks the subjects match
        for k in sorted(set(folds)):
            test = table[folds == k]
            train = table[folds != k]
            if set(test["subject_key"]) & set(train["subject_key"]):
                problems.append(f"seed {seed} fold {k}: a subject is in train and test")
            for c in CLASSES:
                n = test.loc[test["severity"] == c, "subject_key"].nunique()
                if n < MIN_TEST_SUBJECTS_PER_CLASS:
                    problems.append(
                        f"seed {seed} fold {k}: {n} test subjects with {CLASS_LABELS[c]} walks "
                        f"(need {MIN_TEST_SUBJECTS_PER_CLASS})"
                    )
    return problems


def fingerprint(assignments: pd.DataFrame) -> str:
    """Short hash of who is in which fold, to compare folds between machines."""
    rows = assignments.sort_values(["seed", "subject_key"])[["seed", "subject_key", "fold"]]
    return hashlib.sha1(rows.to_csv(index=False).encode()).hexdigest()[:8]


def report(table: pd.DataFrame, assignments: pd.DataFrame) -> None:
    cohorts = list(dict.fromkeys(table["cohort"]))
    overall_c = table["severity"].value_counts(normalize=True)
    overall_k = table["cohort"].value_counts(normalize=True)
    first = int(assignments["seed"].min())
    print(f"Test folds of seed {first} (walks: share per class, share per cohort):")
    print(
        f"{'fold':<5}{'subjects':>9}{'walks':>7}  "
        + "".join(f"{CLASS_LABELS[c]:>11}" for c in CLASSES)
        + "".join(f"{k:>10}" for k in cohorts)
    )
    cls_shares: list[list[float]] = []
    coh_shares: list[list[float]] = []
    n_subjects: list[int] = []
    for seed in sorted(set(assignments["seed"])):
        folds = walk_folds(table, assignments, seed)
        for k in sorted(set(folds)):
            test = table[folds == k]
            cs = [float((test["severity"] == c).mean()) for c in CLASSES]
            ks = [float((test["cohort"] == c).mean()) for c in cohorts]
            cls_shares.append(cs)
            coh_shares.append(ks)
            n_subjects.append(test["subject_key"].nunique())
            if seed == first:
                print(
                    f"{k:<5}{n_subjects[-1]:>9}{len(test):>7}  "
                    + "".join(f"{v:>11.0%}" for v in cs)
                    + "".join(f"{v:>10.0%}" for v in ks)
                )
    a, b = np.array(cls_shares), np.array(coh_shares)
    print(f"\nOver all {len(n_subjects)} test folds (min-max, overall in brackets):")
    print(f"  test subjects per fold: {min(n_subjects)}-{max(n_subjects)}")
    for i, c in enumerate(CLASSES):
        print(
            f"  {CLASS_LABELS[c]} share of walks: {a[:, i].min():.0%}-{a[:, i].max():.0%} "
            f"({overall_c[c]:.0%})"
        )
    for i, name in enumerate(cohorts):
        absent = int((b[:, i] == 0).sum())
        print(
            f"  {name} share of walks: {b[:, i].min():.0%}-{b[:, i].max():.0%} "
            f"({overall_k[name]:.0%}); absent from {absent} folds"
        )
    print("  -> per-cohort scores must be pooled over all folds, never taken per fold.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Subject-wise CV folds (GAIT-18)")
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--seeds", type=int, default=N_SEEDS, help="number of repeats")
    args = parser.parse_args(argv)

    if not args.table.is_file():
        print(f"MISSING: {args.table}\nRun build_feature_table.py first.")
        return 2
    table = read_table(args.table)
    assignments = make_assignments(table, args.folds, tuple(range(args.seeds)))
    n_sub = len(subject_summary(table))
    print(
        f"{len(table)} walks, {n_sub} subjects: {args.folds} folds x {args.seeds} seeds, "
        f"split by subject.\n"
    )
    report(table, assignments)

    problems = fold_problems(table, assignments)
    print()
    for p in problems:
        print(f"  -> {p}")
    status = 1 if problems else 0
    if not problems:
        print(
            f"Validity: every test fold has >= {MIN_TEST_SUBJECTS_PER_CLASS} subjects with walks "
            "of each class, and no subject is in both train and test."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "fold_assignments.csv"
    assignments.to_csv(out, index=False, float_format="%.6f")
    print(f"\nFold fingerprint: {fingerprint(assignments)} (team members should get the same one)")
    print(f"Fold assignments ({len(assignments)} rows): {out}")
    return status


if __name__ == "__main__":
    sys.exit(main())
