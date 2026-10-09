"""Tests for models/voice/src/pipeline.py.

The pure-noise test is the key red test: on data with no signal, honest in-fold
selection must score at chance, and leaky selection must score clearly above it.
"""

from __future__ import annotations

import metrics
import numpy as np
import pandas as pd
import pipeline
import pytest

N_NOISE_PEOPLE = 100
N_NOISE_FEATURES = 752


def noise_data(seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """100 people x 3 rows x 752 pure-noise features, random person-level labels."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(N_NOISE_PEOPLE), 3)
    y = np.repeat(rng.integers(0, 2, N_NOISE_PEOPLE), 3)
    X = pd.DataFrame(
        rng.normal(size=(len(groups), N_NOISE_FEATURES)),
        columns=[f"noise_{i}" for i in range(N_NOISE_FEATURES)],
    )
    return X, y, groups


def person_auc(y: np.ndarray, proba: np.ndarray, groups: np.ndarray) -> float:
    return metrics.auc(*metrics.person_scores(y, proba, groups))


@pytest.mark.parametrize("seed", [0, 1])
def test_noise_infold_is_chance_and_leaky_is_not(seed: int) -> None:
    X, y, groups = noise_data(seed)
    honest = pipeline.nested_cv_predict(X, y, groups, selection="infold", score="anova_f")
    leaky = pipeline.nested_cv_predict(X, y, groups, selection="leaky", score="anova_f")
    honest_auc = person_auc(y, honest.proba, groups)
    leaky_auc = person_auc(y, leaky.proba, groups)
    assert 0.35 < honest_auc < 0.65, f"in-fold AUC {honest_auc:.3f} on pure noise"
    assert leaky_auc > 0.75, f"leaky AUC {leaky_auc:.3f} should be inflated on pure noise"
    assert leaky_auc - honest_auc > 0.2


def test_infold_selector_never_sees_the_outer_test_fold(
    synthetic_table: pd.DataFrame,
) -> None:
    """Spy on the feature scorer: in-fold selection may only ever score training rows."""
    seen: list[int] = []

    def spy(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        seen.append(len(y))
        return pipeline.f_classif(X, y)

    t = synthetic_table
    X = t.drop(columns=["id", "gender", "class"])
    y, groups = t["class"].to_numpy(), t["id"].to_numpy()
    pipeline.nested_cv_predict(
        X, y, groups, selection="infold", score=spy, k_grid=(5,), c_grid=(0.1,), n_jobs=1
    )
    largest_training_fold = max(len(tr) for tr, _ in pipeline.splitter.person_folds(groups, y))
    assert max(seen) <= largest_training_fold < len(y)


def test_leaky_selector_does_see_every_row(synthetic_table: pd.DataFrame) -> None:
    """The control for the test above: the leaky variant scores all rows at once."""
    seen: list[int] = []

    def spy(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        seen.append(len(y))
        return pipeline.f_classif(X, y)

    t = synthetic_table
    X = t.drop(columns=["id", "gender", "class"])
    pipeline.nested_cv_predict(
        X,
        t["class"].to_numpy(),
        t["id"].to_numpy(),
        selection="leaky",
        score=spy,
        k_grid=(5,),
        c_grid=(0.1,),
        n_jobs=1,
    )
    assert max(seen) == len(t)


def test_every_row_predicted_once_and_reproducible(synthetic_table: pd.DataFrame) -> None:
    t = synthetic_table
    X = t.drop(columns=["id", "gender", "class"])
    y, groups = t["class"].to_numpy(), t["id"].to_numpy()
    kw = dict(score="anova_f", k_grid=(5, 10), c_grid=(0.1, 0.5))
    a = pipeline.nested_cv_predict(X, y, groups, **kw)
    b = pipeline.nested_cv_predict(X, y, groups, **kw)
    assert a.proba.shape == (len(t),) and not np.isnan(a.proba).any()
    assert np.array_equal(a.proba, b.proba)
    assert len(a.best_params) == 5


def test_banned_column_is_rejected_before_fitting(synthetic_table: pd.DataFrame) -> None:
    t = synthetic_table
    X = t.drop(columns=["gender", "class"])  # 'id' left in as a feature
    with pytest.raises(pipeline.contract.FeatureContractError, match="id"):
        pipeline.nested_cv_predict(X, t["class"].to_numpy(), t["id"].to_numpy())


@pytest.mark.parametrize(
    ("kwargs", "match"), [({"selection": "both"}, "selection"), ({"split": "x"}, "split")]
)
def test_unknown_variant_fails(
    synthetic_table: pd.DataFrame, kwargs: dict[str, str], match: str
) -> None:
    t = synthetic_table
    X = t.drop(columns=["id", "gender", "class"])
    with pytest.raises(ValueError, match=match):
        pipeline.nested_cv_predict(X, t["class"].to_numpy(), t["id"].to_numpy(), **kwargs)
