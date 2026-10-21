"""Tests for the voice XAI faithfulness and stability checks. Synthetic data only."""

from __future__ import annotations

import audio_features
import numpy as np
import pandas as pd
import pytest
import xai_faithfulness as xf

NAMES = list(audio_features.FEATURE_NAMES)


def test_shap_order_ranks_used_features_by_size_and_breaks_ties_by_name() -> None:
    abs_shap = np.array([[0.1, 0.5, 0.5, 0.9]])
    names = ["d", "c", "b", "a"]
    used = np.array([0, 1, 2])  # feature 3 is not used, however large
    assert xf.shap_order(abs_shap, used, names).tolist() == [[2, 1, 0]]  # b before c on a tie


def test_delete_top_sets_the_first_k_features_to_the_training_median() -> None:
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    medians = np.array([0.0, -1.0, -2.0])
    order = np.array([[2, 0], [1, 2]])
    assert (xf.delete_top(X, medians, order, 0) == X).all()
    assert xf.delete_top(X, medians, order, 1).tolist() == [[1, 2, -2], [4, -1, 6]]
    assert xf.delete_top(X, medians, order, 9).tolist() == [[0, 2, -2], [4, -1, -2]]
    assert X[0, 2] == 3.0  # the input is never changed


def test_empirical_p_counts_random_orders_at_least_as_good() -> None:
    random_values = np.array([0.1, 0.2, 0.3, 0.4])
    assert xf.empirical_p(0.5, random_values, larger_is_better=True) == pytest.approx(1 / 5)
    assert xf.empirical_p(0.25, random_values, larger_is_better=True) == pytest.approx(3 / 5)
    assert xf.empirical_p(0.05, random_values, larger_is_better=False) == pytest.approx(1 / 5)


def test_card_groups_are_built_like_the_desktop_card() -> None:
    row = np.zeros(len(NAMES))
    row[NAMES.index("jitter_local")] = 0.4
    row[NAMES.index("jitter_rap")] = 0.3  # jitter total 0.7
    row[NAMES.index("shimmer_local")] = -0.6
    row[NAMES.index("hnr_db")] = 0.5
    row[NAMES.index("f0_mean_hz")] = 0.2
    row[NAMES.index("mfcc1_mean")] = 0.1  # 6th largest: not in the top 5, so not shown
    assert xf.card_groups(row, NAMES) == ["jitter", "shimmer", "noise"]


def _table(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = np.array([0, 1] * (n // 2))
    X = rng.normal(size=(n, len(NAMES)))
    X[:, :3] += labels[:, None] * np.array([1.5, 1.0, 0.7])  # signal in 3 features
    table = pd.DataFrame(X, columns=NAMES)
    table["person"] = [f"synthetic:{i}" for i in range(n)]
    table["label"] = labels
    return table


@pytest.fixture(scope="module")
def setup() -> tuple[np.ndarray, np.ndarray, list[xf.FoldModel]]:
    table = _table()
    y = table.label.to_numpy()
    folds = xf.fit_folds(table[NAMES], y, table.person.to_numpy(), "l1_logistic", seed=1)
    return table[NAMES].to_numpy(float), y, folds


def test_a_faithful_explanation_passes(
    setup: tuple[np.ndarray, np.ndarray, list[xf.FoldModel]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(xf, "N_RANDOM", 30)
    X, y, folds = setup
    result = xf.deletion_curve(X, y, folds, NAMES, seed=1)
    assert result["margin_area"]["score_changes_faster"] is True
    assert result["margin_area"]["shap"] > result["margin_area"]["reverse"]
    assert result["auc"]["shap"][0] == result["auc"]["reverse"][0]  # k = 0: nothing deleted


def test_an_unfaithful_explanation_fails(
    setup: tuple[np.ndarray, np.ndarray, list[xf.FoldModel]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An 'explanation' that ranks the least important features first must not pass."""
    monkeypatch.setattr(xf, "N_RANDOM", 30)
    real = xf.full_shap

    def backwards(model, X, names):  # type: ignore[no-untyped-def]
        values = real(model, X, names)
        return np.where(np.abs(values) > xf.EPS, 1 / (np.abs(values) + xf.EPS), 0.0)

    monkeypatch.setattr(xf, "full_shap", backwards)
    X, y, folds = setup
    result = xf.deletion_curve(X, y, folds, NAMES, seed=1)
    assert result["margin_area"]["score_changes_faster"] is False
    assert result["auc_area"]["auc_falls_faster"] is False


def test_stability_measures_are_shares_between_0_and_1(
    setup: tuple[np.ndarray, np.ndarray, list[xf.FoldModel]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(xf, "N_REFITS", 3)
    X, y, folds = setup
    groups = np.array([f"synthetic:{i}" for i in range(len(y))])
    person = xf.person_stability(X, y, groups, folds, NAMES, seed=1)
    for key in ("mean_top5_jaccard", "top1_same_share", "card_top3_groups_jaccard"):
        assert 0.0 <= person[key] <= 1.0
    between = xf.fold_stability(X, folds, NAMES)
    assert -1.0 <= between["mean_spearman"] <= 1.0
    assert len(between["top3_groups_overall"]) == xf.CARD_GROUPS


def test_more_than_one_recording_per_person_is_refused() -> None:
    table = _table()
    with pytest.raises(ValueError, match="one recording per person"):
        xf.run(pd.concat([table, table.iloc[:1]]))
