"""Tests for models/mri/src/folds.py. Each guard is shown to fail as well as pass."""

from __future__ import annotations

import numpy as np
import pytest
from folds import (
    SplitError,
    SubjectLeakError,
    assert_subject_disjoint,
    check_labels_per_subject,
    subject_folds,
    unseen_test_groups,
)


def cohort(n: int = 60, rows_per_subject: int = 1):
    """n people, one third with the condition, optionally several rows (scans) each."""
    subjects = np.repeat([f"s{i:03d}" for i in range(n)], rows_per_subject)
    y = np.repeat([1 if i % 3 == 0 else 0 for i in range(n)], rows_per_subject)
    scanner = np.repeat([f"m{i % 4}" for i in range(n)], rows_per_subject)
    return subjects, y, scanner


def test_no_subject_on_both_sides_and_each_tested_once() -> None:
    subjects, y, scanner = cohort()
    folds = subject_folds(subjects, y, n_splits=5, n_repeats=3, strata=scanner)
    assert len(folds) == 15
    for r in range(3):
        tested = np.concatenate([f.test_idx for f in folds if f.repeat == r])
        assert sorted(tested) == list(range(len(y)))
    for f in folds:
        assert not set(subjects[f.train_idx]) & set(subjects[f.test_idx])
        assert set(y[f.test_idx]) == {0, 1}


def test_scans_of_one_person_stay_together() -> None:
    subjects, y, _ = cohort(n=40, rows_per_subject=3)
    for f in subject_folds(subjects, y, n_splits=4, n_repeats=2):
        assert not set(subjects[f.train_idx]) & set(subjects[f.test_idx])


def test_repeats_shuffle_differently() -> None:
    subjects, y, _ = cohort()
    folds = subject_folds(subjects, y, n_splits=5, n_repeats=2, seed=1)
    first = {tuple(sorted(f.test_idx)) for f in folds if f.repeat == 0}
    second = {tuple(sorted(f.test_idx)) for f in folds if f.repeat == 1}
    assert first != second


def test_same_seed_same_folds() -> None:
    subjects, y, _ = cohort()
    a = subject_folds(subjects, y, n_repeats=1, seed=7)
    b = subject_folds(subjects, y, n_repeats=1, seed=7)
    assert all(np.array_equal(x.test_idx, z.test_idx) for x, z in zip(a, b, strict=True))


# --- the guards fail when the rule is broken -----------------------------------------


def test_overlap_is_caught() -> None:
    subjects = np.array(["a", "b", "c", "a"])
    with pytest.raises(SubjectLeakError):
        assert_subject_disjoint(np.array([0, 1]), np.array([2, 3]), subjects)


def test_row_level_split_would_leak() -> None:
    # What a plain row shuffle does with two scans of one person: the guard must refuse it.
    subjects = np.array(["a", "a", "b", "b"])
    with pytest.raises(SubjectLeakError):
        assert_subject_disjoint(np.array([0, 2]), np.array([1, 3]), subjects)


def test_conflicting_labels_for_one_person_fail() -> None:
    with pytest.raises(SplitError, match="different labels"):
        check_labels_per_subject(["a", "a"], [0, 1])


def test_single_class_fails() -> None:
    with pytest.raises(SplitError, match="both classes"):
        subject_folds(["a", "b", "c"], [0, 0, 0], n_splits=2)


def test_too_few_positives_for_the_folds_fails() -> None:
    subjects = [f"s{i}" for i in range(10)]
    y = [1, 1] + [0] * 8  # 2 positives cannot appear in all 5 test parts
    with pytest.raises(SplitError, match="one class"):
        subject_folds(subjects, y, n_splits=5, n_repeats=1)


def test_length_mismatch_fails() -> None:
    with pytest.raises(SplitError, match="lengths"):
        subject_folds(["a", "b"], [0], n_splits=2)


def test_unseen_scanner_is_reported() -> None:
    subjects, y, scanner = cohort(n=30)
    scanner = scanner.astype(object)  # fixed-width numpy strings would truncate the new name
    folds = subject_folds(subjects, y, n_splits=3, n_repeats=1)
    f = folds[0]
    scanner[f.test_idx[0]] = "only_in_test"
    assert unseen_test_groups(f, scanner) == {"only_in_test"}


def test_one_row_per_person_always_gets_both_classes_in_every_test_fold() -> None:
    # StratifiedGroupKFold with shuffling put a single class into a test fold for some
    # seeds with 30 people (15 + 15) and 5 folds; every seed must work.
    subjects = np.array([f"s{i:02d}" for i in range(30)])
    y = np.tile([0, 1], 15)
    for seed in range(50):
        for f in subject_folds(subjects, y, n_splits=5, n_repeats=1, seed=seed):
            assert sorted(np.bincount(y[f.test_idx])) == [3, 3]


# --- spread: every group dealt evenly over the folds ----------------------------------


def test_spread_caps_each_group_per_test_fold() -> None:
    subjects = np.array([f"s{i:02d}" for i in range(37)])
    y = np.array([i % 2 for i in range(37)])
    group = np.array(["A"] * 12 + ["B"] * 9 + ["C"] * 7 + ["D"] * 6 + ["E"] * 3)
    for seed in range(25):
        folds = subject_folds(subjects, y, n_splits=5, n_repeats=2, seed=seed, spread=group)
        for f in folds:
            in_test = {g: int((group[f.test_idx] == g).sum()) for g in set(group)}
            cap = {"A": 3, "B": 2, "C": 2, "D": 2, "E": 1}  # ceil(size / 5)
            assert all(in_test[g] <= cap[g] for g in cap)
            assert set(y[f.test_idx]) == {0, 1}
            assert not unseen_test_groups(f, group)


def test_spread_keeps_rows_of_one_person_together() -> None:
    subjects = np.repeat([f"s{i}" for i in range(20)], 2)
    y = np.repeat([i % 2 for i in range(20)], 2)
    group = np.repeat(["A", "B"] * 10, 2)
    for f in subject_folds(subjects, y, n_splits=4, n_repeats=2, spread=group):
        assert not set(subjects[f.train_idx]) & set(subjects[f.test_idx])


def test_spread_differs_between_repeats_and_repeats_with_seed() -> None:
    subjects = [f"s{i}" for i in range(30)]
    y = [i % 2 for i in range(30)]
    group = ["A", "B", "C"] * 10
    a = subject_folds(subjects, y, n_splits=5, n_repeats=2, seed=3, spread=group)
    b = subject_folds(subjects, y, n_splits=5, n_repeats=2, seed=3, spread=group)
    assert all(np.array_equal(x.test_idx, z.test_idx) for x, z in zip(a, b, strict=True))
    first = {tuple(f.test_idx) for f in a if f.repeat == 0}
    second = {tuple(f.test_idx) for f in a if f.repeat == 1}
    assert first != second


def test_spread_and_strata_together_fail() -> None:
    with pytest.raises(SplitError, match="not both"):
        subject_folds(
            ["a", "b", "c", "d"], [0, 1, 0, 1], 2, 1, strata=list("xyxy"), spread=list("xyxy")
        )


def test_spread_group_must_be_constant_per_person() -> None:
    with pytest.raises(SplitError, match="different spread groups"):
        subject_folds(["a", "a", "b", "c"], [0, 0, 1, 1], 2, 1, spread=["x", "y", "x", "y"])
