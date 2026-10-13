"""Tests for models/mri/src/harmonise.py on synthetic data. Every guard is shown to fail."""

from __future__ import annotations

import json

import numpy as np
import pytest
from harmonise import (
    ComBat,
    HarmonisationError,
    PermutationControlError,
    SiteGateError,
    UnseenBatchError,
    label_permutation_control,
    out_of_fold_harmonise,
    site_predictability,
    site_prediction_gate,
)


def scanner_data(n_per: int = 20, p: int = 6, seed: int = 0):
    """Three scanners that shift and stretch the same biology; a real group difference."""
    rng = np.random.default_rng(seed)
    batch = np.repeat(["A", "B", "C"], n_per)
    y = np.tile([0, 1], len(batch) // 2)
    age = rng.normal(65, 8, len(batch))
    bio = rng.normal(0, 1, (len(batch), p)) - 0.8 * y[:, None] + 0.03 * (age - 65)[:, None]
    shift = {"A": 0.0, "B": 3.0, "C": -2.0}
    scale = {"A": 1.0, "B": 2.0, "C": 0.6}
    x = np.vstack([bio[i] * scale[b] + shift[b] for i, b in enumerate(batch)])
    return x, batch, y, age


def kfold(n: int, k: int = 5, seed: int = 0):
    idx = np.random.default_rng(seed).permutation(n)
    parts = np.array_split(idx, k)
    return [(np.concatenate(parts[:i] + parts[i + 1 :]), parts[i]) for i in range(k)]


# --- ComBat itself --------------------------------------------------------------------


def test_combat_removes_scanner_shift_and_keeps_group_difference() -> None:
    x, batch, y, _ = scanner_data()
    h = ComBat().fit_transform(x, batch)
    means = [h[batch == b].mean() for b in "ABC"]
    assert np.ptp(means) < 0.5 < np.ptp([x[batch == b].mean() for b in "ABC"])
    assert h[y == 1].mean() < h[y == 0].mean()  # the group gap survives


def test_matches_neurocombat() -> None:
    nc = pytest.importorskip("neuroCombat")
    pd = pytest.importorskip("pandas")
    x, batch, _, age = scanner_data()
    covars = pd.DataFrame({"batch": batch, "age": age})
    ref = nc.neuroCombat(x.T, covars, "batch", continuous_cols=["age"])["data"].T
    ours = ComBat().fit_transform(x, batch, covariates=age)
    assert np.allclose(ours, ref, atol=1e-6)


def test_saved_parameters_give_the_same_result() -> None:
    x, batch, _, _ = scanner_data()
    train, test = np.arange(0, len(x), 2), np.arange(1, len(x), 2)  # every scanner on both sides
    m = ComBat().fit(x[train], batch[train])
    again = ComBat.from_dict(json.loads(json.dumps(m.to_dict())))
    assert np.allclose(again.transform(x[test], batch[test]), m.transform(x[test], batch[test]))


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda x, b: (x, np.r_[b[:-1], ["Z"]]), "fewer than 2"),
        (lambda x, b: (x, np.full(len(b), "A")), "two batches"),
        (lambda x, b: (np.where(np.arange(x.size).reshape(x.shape) == 0, np.nan, x), b), "missing"),
        (lambda x, b: (np.c_[x, np.ones(len(x))], b), "no variance"),
    ],
)
def test_bad_training_data_fails(change, match) -> None:
    x, batch, _, _ = scanner_data()
    x2, b2 = change(x, batch)
    with pytest.raises(HarmonisationError, match=match):
        ComBat().fit(x2, b2)


def test_unseen_scanner_fails() -> None:
    x, batch, _, _ = scanner_data()
    m = ComBat().fit(x, batch)
    with pytest.raises(UnseenBatchError):
        m.transform(x[:2], ["A", "new_scanner"])


def test_covariates_must_be_consistent() -> None:
    x, batch, _, age = scanner_data()
    m = ComBat().fit(x, batch, covariates=age)
    with pytest.raises(HarmonisationError, match="covariates"):
        m.transform(x, batch)


# --- inside the fold ------------------------------------------------------------------


def test_out_of_fold_models_never_see_test_rows() -> None:
    x, batch, _, _ = scanner_data()
    folds = kfold(len(x))
    _, models = out_of_fold_harmonise(x, batch, folds)
    for (train, test), model in zip(folds, models, strict=True):
        poisoned = x.copy()
        poisoned[test] += 1000.0  # if the fit used test rows, its parameters would move
        refit = ComBat().fit(poisoned[train], batch[train])
        assert np.allclose(refit.grand_mean_, model.grand_mean_)
        assert np.allclose(refit.var_pooled_, model.var_pooled_)


def test_fit_on_everyone_is_not_what_out_of_fold_does() -> None:
    # The leaky version (ComBat on all rows, then split) gives different test values.
    x, batch, _, _ = scanner_data()
    oof, _ = out_of_fold_harmonise(x, batch, kfold(len(x)))
    leaky = ComBat().fit_transform(x, batch)
    assert not np.allclose(oof, leaky)


def test_folds_must_cover_every_row_once() -> None:
    x, batch, _, _ = scanner_data()
    folds = kfold(len(x))[:-1]  # one fold missing
    with pytest.raises(HarmonisationError, match="exactly once"):
        out_of_fold_harmonise(x, batch, folds)


def test_overlapping_fold_fails() -> None:
    x, batch, _, _ = scanner_data()
    train, test = kfold(len(x))[0]
    with pytest.raises(HarmonisationError, match="both"):
        out_of_fold_harmonise(x, batch, [(np.r_[train, test[:1]], test)])


# --- site-prediction gate ----------------------------------------------------------------


def test_site_gate_fails_before_and_passes_after_harmonisation() -> None:
    x, batch, _, _ = scanner_data()
    assert site_predictability(x, batch) > 0.9  # raw: scanner is obvious
    with pytest.raises(SiteGateError):
        site_prediction_gate(x, batch, n_permutations=30)
    oof, _ = out_of_fold_harmonise(x, batch, kfold(len(x)))
    result = site_prediction_gate(oof, batch, n_permutations=30)
    assert result.passed and result.score <= result.threshold


# --- label-permutation control ------------------------------------------------------------


def cv_balanced_accuracy(x: np.ndarray, y: np.ndarray) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    pred = cross_val_predict(LogisticRegression(max_iter=1000), x, y, cv=cv)
    return float(balanced_accuracy_score(y, pred))


def test_permutation_control_passes_an_honest_pipeline() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(80, 5))
    y = np.tile([0, 1], 40)
    result = label_permutation_control(lambda yy: cv_balanced_accuracy(x, yy), y)
    assert result.passed


def test_permutation_control_catches_a_leak() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(80, 5))
    y = np.tile([0, 1], 40)

    def leaky(yy: np.ndarray) -> float:
        # Bug on purpose: the label is copied into the features.
        return cv_balanced_accuracy(np.c_[x, yy], yy)

    with pytest.raises(PermutationControlError):
        label_permutation_control(leaky, y)


def test_permutation_keeps_one_label_per_person() -> None:
    subjects = np.repeat(["a", "b", "c", "d"], 3)
    y = np.repeat([0, 1, 0, 1], 3)
    seen = []

    def record(yy: np.ndarray) -> float:
        seen.append(yy)
        return 0.5

    label_permutation_control(record, y, subjects=subjects, n_permutations=5)
    for yy in seen:
        for s in "abcd":
            assert len(set(yy[subjects == s])) == 1
