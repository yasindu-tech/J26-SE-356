"""Subject-wise cross-validation folds for the MRI models.

Every person lands in the test part exactly once per repeat, and never in train and test of
the same fold. That is enforced by an assertion inside the fold generator itself, not only in
the tests, so a bug elsewhere cannot quietly produce overlapping folds.

Folds are stratified on the diagnosis label and, optionally, on a second variable such as the
scanner model (the ML pipeline spec asks for "stratified by diagnosis and site"). With small
groups the stratification is approximate; the scanner check below says when a test fold holds
a scanner that is absent from its training part, which ComBat cannot correct.

``spread`` is the stronger option for a variable such as the ComBat batch (scanner group):
every group is dealt evenly over the folds, so a test fold never holds more than
ceil(group size / n_splits) people of one group. A group of 3 or more people then keeps at
least 2 in every training part with 5 folds, which ComBat needs, and no test person comes
from a group the training part has not seen. Labels are sorted inside each group as it is
dealt, so they spread evenly too. Only the group and the label are used.

Usage:

    folds = subject_folds(subjects, y, n_splits=5, n_repeats=10, strata=scanner_model)
    folds = subject_folds(subjects, y, n_splits=5, n_repeats=10, spread=scanner_group)
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
    spread: Sequence[str] | None = None,
) -> list[Fold]:
    """Repeated, stratified, subject-wise K-fold. Every test part holds both classes.

    ``strata`` stratifies on the label and that variable (approximate with small groups).
    ``spread`` deals every group of that variable evenly over the folds instead (see the
    module docstring). Give at most one of them.

    When every person has exactly one row, plain StratifiedKFold is used: it is then
    subject-wise by construction and stratifies exactly. StratifiedGroupKFold with shuffling
    is greedy and can put a single class into a test fold even when an even split exists.
    Several rows per person always use StratifiedGroupKFold. The disjointness assertion runs
    either way.
    """
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    subjects_arr = np.asarray(subjects)
    y_arr = np.asarray(y)
    if len(subjects_arr) != len(y_arr):
        raise SplitError("subjects and labels have different lengths")
    check_labels_per_subject(subjects_arr, y_arr)
    if len(set(y_arr)) < 2:
        raise SplitError("need both classes to build folds")
    if strata is not None and spread is not None:
        raise SplitError("give strata or spread, not both")
    strat = y_arr.astype(str)
    if strata is not None:
        strat = np.char.add(np.char.add(strat, "|"), np.asarray(strata).astype(str))

    folds = []
    x_dummy = np.zeros(len(y_arr))
    one_row_each = len(set(subjects_arr)) == len(subjects_arr)
    for r in range(n_repeats):
        if spread is not None:
            splits = _dealt_splits(subjects_arr, y_arr, np.asarray(spread), n_splits, seed + r)
        elif one_row_each:
            splits = StratifiedKFold(n_splits, shuffle=True, random_state=seed + r).split(
                x_dummy, strat
            )
        else:
            splits = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed + r).split(
                x_dummy, strat, groups=subjects_arr
            )
        times_tested = np.zeros(len(y_arr), dtype=int)
        for k, (train, test) in enumerate(splits):
            assert_subject_disjoint(train, test, subjects_arr)
            if len(set(y_arr[test])) < 2:
                raise SplitError(f"repeat {r} fold {k}: test part has one class; use fewer folds")
            times_tested[test] += 1
            folds.append(Fold(r, k, train, test))
        if not np.all(times_tested == 1):
            raise SplitError(f"repeat {r}: not every row was tested exactly once")
    return folds


def _dealt_splits(
    subjects: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deal people over the folds group by group, labels sorted inside each group.

    Groups come in random order and people in random order within (group, label). One fold
    counter runs on across groups, so each group lands on consecutive folds (at most
    ceil(size / n_splits) per fold) and the labels spread evenly overall. Several rows of one
    person always share a fold.
    """
    if len(groups) != len(subjects):
        raise SplitError("spread and subjects have different lengths")
    people, first = np.unique(subjects, return_index=True)
    group_of = {p: str(groups[i]) for p, i in zip(people, first, strict=True)}
    for s, g in zip(subjects, groups, strict=True):
        if group_of[s] != str(g):
            raise SplitError("one subject has rows in different spread groups")
    label_of = {p: int(y[i]) for p, i in zip(people, first, strict=True)}
    rng = np.random.default_rng(seed)
    names = sorted(set(group_of.values()))
    rng.shuffle(names)
    order: list[str] = []
    for g in names:
        members = [p for p in people if group_of[p] == g]
        rng.shuffle(members)
        order += sorted(members, key=lambda p: label_of[p])  # stable: still random within label
    start = int(rng.integers(n_splits))
    fold_of = {p: (start + i) % n_splits for i, p in enumerate(order)}
    row_fold = np.array([fold_of[s] for s in subjects])
    idx = np.arange(len(subjects))
    return [(idx[row_fold != k], idx[row_fold == k]) for k in range(n_splits)]


def unseen_test_groups(fold: Fold, groups: Sequence[str]) -> set[str]:
    """Groups (e.g. scanner models) in the test part that the training part never saw."""
    g = np.asarray(groups)
    return set(g[fold.test_idx]) - set(g[fold.train_idx])
