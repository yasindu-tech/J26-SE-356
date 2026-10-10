"""Person-level splitting for the voice module (task VOICE-10).

Every split here works on people, not rows: all recordings of one person land
on the same side (CLAUDE.md section 3.3). Each split is checked with
``assert_subject_disjoint`` as it is produced, so a leaking split cannot be
returned at all.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

SEED = 42

Split = tuple[np.ndarray, np.ndarray]


def assert_subject_disjoint(
    train_idx: np.ndarray, test_idx: np.ndarray, groups: np.ndarray
) -> None:
    """Raise if any person has rows on both sides of the split.

    Raises ValueError (not ``assert``) so the check survives ``python -O``.
    """
    groups = np.asarray(groups)
    shared = set(groups[train_idx]) & set(groups[test_idx])
    if shared:
        raise ValueError(f"subject leak: {len(shared)} people appear in both train and test")


def person_folds(
    groups: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = SEED
) -> list[Split]:
    """Stratified, person-level K-fold. Returns (train_idx, test_idx) row indices per fold."""
    groups, y = np.asarray(groups), np.asarray(y)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(cv.split(np.zeros(len(y)), y, groups))
    for train_idx, test_idx in folds:
        assert_subject_disjoint(train_idx, test_idx, groups)
    return folds


def holdout_split(
    groups: np.ndarray,
    y: np.ndarray,
    gender: np.ndarray,
    test_size: float = 0.2,
    seed: int = SEED,
) -> Split:
    """One-off person-level hold-out (decision D3), stratified by class and gender.

    Returns (train_idx, test_idx) row indices. Class and gender must be constant
    per person (``data.load_uci470`` checks this).
    """
    groups, y, gender = np.asarray(groups), np.asarray(y), np.asarray(gender)
    people, first_row = np.unique(groups, return_index=True)
    strata = [f"{c}_{g}" for c, g in zip(y[first_row], gender[first_row], strict=True)]
    _, test_people = train_test_split(
        people, test_size=test_size, stratify=strata, random_state=seed
    )
    is_test = np.isin(groups, test_people)
    train_idx, test_idx = np.flatnonzero(~is_test), np.flatnonzero(is_test)
    assert_subject_disjoint(train_idx, test_idx, groups)
    return train_idx, test_idx
