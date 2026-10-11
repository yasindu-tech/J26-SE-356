"""Tests for models/voice/src/splitter.py. Each check is proved to fail on bad input."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import splitter
from sklearn.model_selection import KFold


@pytest.fixture
def arrays(synthetic_table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = synthetic_table
    return t["id"].to_numpy(), t["class"].to_numpy(), t["gender"].to_numpy()


def test_row_level_split_is_rejected(arrays: tuple[np.ndarray, ...]) -> None:
    """The red test: a plain row-level KFold puts one person on both sides."""
    groups, _, _ = arrays
    train_idx, test_idx = next(KFold(5, shuffle=True, random_state=0).split(groups))
    with pytest.raises(ValueError, match="subject leak"):
        splitter.assert_subject_disjoint(train_idx, test_idx, groups)


def test_disjoint_split_passes(arrays: tuple[np.ndarray, ...]) -> None:
    groups, _, _ = arrays
    is_test = np.isin(groups, groups[:30])  # first 10 people, all their rows
    splitter.assert_subject_disjoint(np.flatnonzero(~is_test), np.flatnonzero(is_test), groups)


def test_person_folds_are_disjoint_and_cover_everyone_once(
    arrays: tuple[np.ndarray, ...],
) -> None:
    groups, y, _ = arrays
    folds = splitter.person_folds(groups, y)
    assert len(folds) == 5
    test_rows = np.concatenate([test for _, test in folds])
    assert sorted(test_rows) == list(range(len(groups)))  # every row tested exactly once
    for train, test in folds:
        assert not set(groups[train]) & set(groups[test])


def test_person_folds_keep_class_balance(arrays: tuple[np.ndarray, ...]) -> None:
    groups, y, _ = arrays
    overall = y.mean()
    for _, test in splitter.person_folds(groups, y):
        assert abs(y[test].mean() - overall) < 0.15


def test_person_folds_are_reproducible(arrays: tuple[np.ndarray, ...]) -> None:
    groups, y, _ = arrays
    a = splitter.person_folds(groups, y, seed=42)
    b = splitter.person_folds(groups, y, seed=42)
    assert all(np.array_equal(ta, tb) for (_, ta), (_, tb) in zip(a, b, strict=True))


def test_holdout_is_20_percent_of_people_and_disjoint(arrays: tuple[np.ndarray, ...]) -> None:
    groups, y, gender = arrays
    train, test = splitter.holdout_split(groups, y, gender)
    assert len(np.unique(groups[test])) == 12  # 20% of 60 people
    assert not set(groups[train]) & set(groups[test])
    assert len(train) + len(test) == len(groups)


def test_holdout_is_reproducible(arrays: tuple[np.ndarray, ...]) -> None:
    groups, y, gender = arrays
    a = splitter.holdout_split(groups, y, gender)
    b = splitter.holdout_split(groups, y, gender)
    assert np.array_equal(a[1], b[1])
