"""Tests for the subject-wise cross-validation folds (GAIT-18)."""

from __future__ import annotations

from pathlib import Path

import make_folds as mf
import numpy as np
import pandas as pd
import pytest


def synthetic_table(n_subjects: int = 40, seed: int = 0) -> pd.DataFrame:
    """Subjects in two cohorts, with different numbers of walks and mixed classes."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_subjects):
        cohort = "A" if i % 4 else "B"
        base = i % 3
        for _ in range(int(rng.integers(1, 9))):
            severity = base if rng.random() < 0.7 else int(rng.integers(0, 3))
            rows.append((f"w-{i:04d}", cohort, severity))
    return pd.DataFrame(rows, columns=list(mf.REQUIRED))


@pytest.fixture()
def table() -> pd.DataFrame:
    return synthetic_table()


# --------------------------------------------------------------------------- the folds
def test_every_subject_has_exactly_one_fold_per_seed(table: pd.DataFrame) -> None:
    a = mf.make_assignments(table, 5, (0, 1))
    n = table["subject_key"].nunique()
    assert len(a) == 2 * n
    assert not a["fold"].isna().any()
    assert (a.groupby(["seed", "subject_key"]).size() == 1).all()
    assert set(a["fold"]) == set(range(5))


def test_no_subject_is_in_train_and_test(table: pd.DataFrame) -> None:
    a = mf.make_assignments(table, 5, (0, 1, 2))
    for seed in (0, 1, 2):
        seen = np.zeros(len(table), dtype=int)
        for train, test in mf.iter_folds(table, a, seed):
            assert not set(table["subject_key"].iloc[train]) & set(table["subject_key"].iloc[test])
            seen[test] += 1
        assert (seen == 1).all()  # every walk is tested exactly once


def test_folds_are_reproducible_and_differ_between_seeds(table: pd.DataFrame) -> None:
    a = mf.make_assignments(table, 5, (0, 1))
    b = mf.make_assignments(table, 5, (0, 1))
    pd.testing.assert_frame_equal(a, b)
    s0 = a[a["seed"] == 0]["fold"].to_numpy()
    s1 = a[a["seed"] == 1]["fold"].to_numpy()
    assert (s0 != s1).any()
    assert mf.fingerprint(a) == mf.fingerprint(b)


def test_class_shares_are_kept_close_to_the_overall_shares(table: pd.DataFrame) -> None:
    a = mf.make_assignments(table, 5, (0,))
    overall = table["severity"].value_counts(normalize=True)
    for _, test in mf.iter_folds(table, a, 0):
        share = table.iloc[test]["severity"].value_counts(normalize=True)
        for c in mf.CLASSES:
            assert abs(share.get(c, 0.0) - overall[c]) < 0.2


def test_the_default_gives_five_folds_by_five_seeds(table: pd.DataFrame) -> None:
    a = mf.make_assignments(table)
    assert set(a["seed"]) == set(range(5)) and set(a["fold"]) == set(range(5))


# --------------------------------------------------------------------------- the gate
def test_a_valid_split_has_no_problems(table: pd.DataFrame) -> None:
    assert mf.fold_problems(table, mf.make_assignments(table, 5, (0,))) == []


def test_a_class_with_too_few_subjects_fails_the_gate() -> None:
    # class 2 exists in two subjects only: no test fold can hold three of them
    t = synthetic_table()
    keys = sorted(set(t["subject_key"]))[:2]
    t["severity"] = np.where(t["subject_key"].isin(keys), 2, t["severity"].clip(upper=1))
    problems = mf.fold_problems(t, mf.make_assignments(t, 5, (0,)))
    assert problems and any("class 2-3" in p for p in problems)


def test_a_subject_with_two_folds_is_rejected(table: pd.DataFrame) -> None:
    a = mf.make_assignments(table, 5, (0,))
    twice = pd.concat([a, a.iloc[[0]].assign(fold=(a["fold"].iloc[0] + 1) % 5)], ignore_index=True)
    with pytest.raises(ValueError, match="more than one fold"):
        mf.fold_problems(table, twice)


# --------------------------------------------------------------------------- loading
def test_load_fold_gives_disjoint_train_and_test(table: pd.DataFrame) -> None:
    a = mf.make_assignments(table, 5, (0,))
    train, test = mf.load_fold(table, a, 0, 2)
    assert len(train) + len(test) == len(table) and not set(train) & set(test)
    assert len(test) > 0


def test_load_fold_rejects_a_table_with_different_subjects(table: pd.DataFrame) -> None:
    a = mf.make_assignments(table, 5, (0,))
    with pytest.raises(ValueError, match="different subjects"):
        mf.load_fold(table.iloc[: len(table) // 2], a, 0, 0)
    extra = table.copy()
    extra.loc[extra.index[0], "subject_key"] = "w-new"
    with pytest.raises(ValueError, match="different subjects"):
        mf.load_fold(extra, a, 0, 0)


def test_load_fold_rejects_an_unknown_seed_or_fold(table: pd.DataFrame) -> None:
    a = mf.make_assignments(table, 5, (0,))
    with pytest.raises(ValueError, match="seed 3"):
        mf.load_fold(table, a, 3, 0)
    with pytest.raises(ValueError, match="no fold 7"):
        mf.load_fold(table, a, 0, 7)


def test_bad_tables_raise() -> None:
    with pytest.raises(ValueError, match="lacks"):
        mf.make_assignments(pd.DataFrame({"subject_key": ["a"]}))
    with pytest.raises(ValueError, match="no walks"):
        mf.make_assignments(pd.DataFrame(columns=list(mf.REQUIRED)))
    with pytest.raises(ValueError, match="severity"):
        mf.make_assignments(pd.DataFrame([("a", "A", 5)], columns=list(mf.REQUIRED)))
    with pytest.raises(ValueError, match="more than one cohort"):
        mf.make_assignments(pd.DataFrame([("a", "A", 0), ("a", "B", 1)], columns=list(mf.REQUIRED)))


# --------------------------------------------------------------------------- weights
def test_walk_weights_give_each_subject_the_same_total(table: pd.DataFrame) -> None:
    w = mf.walk_weights(table)
    assert w.mean() == pytest.approx(1.0)
    totals = pd.Series(w).groupby(table["subject_key"].to_numpy()).sum()
    assert totals.nunique() == 1 or np.allclose(totals, totals.iloc[0])


def test_saved_weights_match_walk_weights(table: pd.DataFrame) -> None:
    a = mf.make_assignments(table, 5, (0,)).set_index("subject_key")
    per_walk = table["subject_key"].map(a["walk_weight"]).to_numpy()
    np.testing.assert_allclose(per_walk, mf.walk_weights(table))


# --------------------------------------------------------------------------- command line
def test_cli_writes_the_assignments_and_reports(
    tmp_path: Path, table: pd.DataFrame, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "feature_table.csv"
    table.assign(extra=1.0).to_csv(path, index=False)  # other columns are ignored
    assert mf.main(["--table", str(path), "--out-dir", str(tmp_path), "--seeds", "2"]) == 0
    out = capsys.readouterr().out
    assert "pooled over all folds" in out and "fingerprint" in out
    saved = pd.read_csv(tmp_path / "fold_assignments.csv")
    assert list(saved.columns) == [
        "seed",
        "subject_key",
        "cohort",
        "n_walks",
        "walk_weight",
        "fold",
    ]
    assert len(saved) == 2 * table["subject_key"].nunique()


def test_cli_exits_1_when_the_gate_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    t = synthetic_table()
    keys = sorted(set(t["subject_key"]))[:2]
    t["severity"] = np.where(t["subject_key"].isin(keys), 2, t["severity"].clip(upper=1))
    path = tmp_path / "feature_table.csv"
    t.to_csv(path, index=False)
    assert mf.main(["--table", str(path), "--out-dir", str(tmp_path)]) == 1
    assert "need 3" in capsys.readouterr().out


def test_cli_missing_table_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert mf.main(["--table", str(tmp_path / "nope.csv"), "--out-dir", str(tmp_path)]) == 2
    assert "build_feature_table" in capsys.readouterr().out
