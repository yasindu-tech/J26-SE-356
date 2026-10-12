"""Subject-wise cross-validation folds for the MRI models (CLAUDE.md section 3.3).

Every person lands in the test part exactly once per repeat, and never in train and test of
the same fold. That is enforced by an assertion inside the fold generator itself, not only in
the tests, so a bug elsewhere cannot quietly produce overlapping folds.

Folds are stratified on the diagnosis label and, optionally, on a second variable such as the
scanner model (the ML pipeline spec asks for "stratified by diagnosis and site"). With small
groups the stratification is approximate; the scanner check below says when a test fold holds
a scanner that is absent from its training part, which ComBat cannot correct.

Usage:

    folds = subject_folds(subjects, y, n_splits=5, n_repeats=10, strata=scanner_model)
    for fold in folds:
        train, test = fold.train_idx, fold.test_idx   # row indices into the table
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


class SplitError(ValueError):
    """The folds would break a rule; the message says which."""


class SubjectLeakError(SplitError):
    """A person appears in both the training and the test part of one fold."""


@dataclass(frozen=True)
class Fold:
    repeat: int
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def assert_subject_disjoint(
    train_idx: np.ndarray, test_idx: np.ndarray, subjects: Sequence[str]
) -> None:
    """Raise if any person is on both sides. Call it in every CV loop."""
    s = np.asarray(subjects)
    shared = set(s[train_idx]) & set(s[test_idx])
    if shared:
        raise SubjectLeakError(f"{len(shared)} subject(s) in both train and test")


def check_labels_per_subject(subjects: Sequence[str], y: Sequence[int]) -> None:
    """Rows of one person must carry one label; two labels means a data error."""
    seen: dict[str, int] = {}
    for s, label in zip(subjects, y, strict=True):
        if seen.setdefault(s, label) != label:
            raise SplitError("one subject has rows with different labels")


def subject_folds(
    subjects: Sequence[str],
    y: Sequence[int],
    n_splits: int = 5,
    n_repeats: int = 10,
    seed: int = 0,
    strata: Sequence[str] | None = None,
) -> list[Fold]:
    """Repeated, stratified, subject-wise K-fold. Every test part holds both classes."""
    from sklearn.model_selection import StratifiedGroupKFold

    subjects_arr = np.asarray(subjects)
    y_arr = np.asarray(y)
    if len(subjects_arr) != len(y_arr):
        raise SplitError("subjects and labels have different lengths")
    check_labels_per_subject(subjects_arr, y_arr)
    if len(set(y_arr)) < 2:
        raise SplitError("need both classes to build folds")
    strat = y_arr.astype(str)
    if strata is not None:
        strat = np.char.add(np.char.add(strat, "|"), np.asarray(strata).astype(str))

    folds = []
    x_dummy = np.zeros(len(y_arr))
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        times_tested = np.zeros(len(y_arr), dtype=int)
        for k, (train, test) in enumerate(cv.split(x_dummy, strat, groups=subjects_arr)):
            assert_subject_disjoint(train, test, subjects_arr)
            if len(set(y_arr[test])) < 2:
                raise SplitError(f"repeat {r} fold {k}: test part has one class; use fewer folds")
            times_tested[test] += 1
            folds.append(Fold(r, k, train, test))
        if not np.all(times_tested == 1):
            raise SplitError(f"repeat {r}: not every row was tested exactly once")
    return folds


def unseen_test_groups(fold: Fold, groups: Sequence[str]) -> set[str]:
    """Groups (e.g. scanner models) in the test part that the training part never saw."""
    g = np.asarray(groups)
    return set(g[fold.test_idx]) - set(g[fold.train_idx])
