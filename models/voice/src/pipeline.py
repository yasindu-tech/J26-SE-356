"""In-fold model pipeline and nested cross-validation (task VOICE-13).

The honest pipeline fits the scaler, the feature selector and the model inside
each training fold only (CLAUDE.md section 3.2). With 752 features and 252
people, selecting features on the full table first decides the result.

``nested_cv_predict`` can also run the deliberately leaky variants that the
leakage audit (VOICE-14) compares against:

- ``selection="leaky"`` ranks features on ALL rows (test folds included) before
  splitting. The scaler, model and hyperparameter search are otherwise identical,
  so the only difference is where the feature scores were computed.
- ``split="row"`` splits rows instead of people, so one person's recordings can
  sit on both sides of a split.

The honest setting is ``selection="infold", split="subject"`` (variant B).

``model="lightgbm"`` swaps the L1 logistic regression for LightGBM (decision D4)
behind the same scaler and in-fold selector; the baseline ladder (VOICE-15)
uses it as its top rung.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Literal, NamedTuple

import contract
import numpy as np
import pandas as pd
import splitter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
N_OUTER = 5
N_INNER = 3
K_GRID = (10, 25, 50, 100)
C_GRID = (0.05, 0.1, 0.5)
LEAVES_GRID = (4, 8)  # LightGBM num_leaves; small trees for 252 people

Selection = Literal["infold", "leaky"]
Split = Literal["subject", "row"]
Model = Literal["l1_logistic", "lightgbm"]
ScoreFunc = Callable[[np.ndarray, np.ndarray], object]

SCORE_FUNCS: dict[str, ScoreFunc] = {
    "mutual_info": partial(mutual_info_classif, random_state=SEED),
    "anova_f": f_classif,  # much faster; use if mutual information is too slow
}


class CVResult(NamedTuple):
    proba: np.ndarray  # out-of-fold PD probability, one per row
    best_params: list[dict[str, float]]  # inner-CV choice of k and C (or leaves), per fold


class TopKByRanking(BaseEstimator, TransformerMixin):
    """Keep the k columns ranked highest by a ranking computed before the split.

    Used ONLY by the leaky variant. ``fit`` learns nothing, which is the point:
    the ranking already saw every row.
    """

    def __init__(self, ranking: np.ndarray | None = None, k: int = 10) -> None:
        self.ranking = ranking
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> TopKByRanking:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.ranking is None:
            raise ValueError("TopKByRanking needs a ranking")
        return np.asarray(X)[:, self.ranking[: self.k]]


def _feature_scores(score_func: ScoreFunc, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    result = score_func(X, y)
    scores = result[0] if isinstance(result, tuple) else result  # f_classif returns (F, p)
    return np.nan_to_num(np.asarray(scores, dtype=float), nan=-np.inf)


def _model(model: Model, seed: int) -> BaseEstimator:
    if model == "l1_logistic":
        return LogisticRegression(
            l1_ratio=1.0,  # pure L1 (scikit-learn >= 1.8 deprecates penalty="l1")
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        )
    from lightgbm import LGBMClassifier  # imported here so the L1 path does not need it

    return LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        min_child_samples=10,
        class_weight="balanced",
        random_state=seed,
        deterministic=True,
        n_jobs=1,  # GridSearchCV already runs fits in parallel
        verbose=-1,
    )


def build_pipeline(
    score_func: ScoreFunc,
    ranking: np.ndarray | None = None,
    seed: int = SEED,
    model: Model = "l1_logistic",
) -> Pipeline:
    """Scaler -> selector -> model (balanced class weights).

    With ``ranking`` the selector uses that precomputed (leaky) ranking; without
    it, SelectKBest scores features on whatever data the pipeline is fitted on.
    """
    selector = SelectKBest(score_func) if ranking is None else TopKByRanking(ranking=ranking)
    return Pipeline(
        [("scale", StandardScaler()), ("select", selector), ("model", _model(model, seed))]
    )


def _folds(
    y: np.ndarray, groups: np.ndarray, split: Split, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    if split == "subject":
        return splitter.person_folds(groups, y, n_splits=n_splits, seed=seed)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y))


def nested_cv_predict(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    selection: Selection = "infold",
    split: Split = "subject",
    score: str | ScoreFunc = "mutual_info",
    n_splits: int = N_OUTER,
    k_grid: Sequence[int] = K_GRID,
    c_grid: Sequence[float] = C_GRID,
    seed: int = SEED,
    n_jobs: int = -1,
    model: Model = "l1_logistic",
    leaves_grid: Sequence[int] = LEAVES_GRID,
) -> CVResult:
    """Out-of-fold probabilities for every row, with k and C tuned in an inner CV.

    For ``model="lightgbm"`` the inner CV tunes k and ``num_leaves`` instead of C.

    The inner CV uses the same split type as the outer one (by person for
    ``split="subject"``), so tuning never sees the outer test fold either.
    """
    if selection not in ("infold", "leaky"):
        raise ValueError(f"selection must be 'infold' or 'leaky', got {selection!r}")
    if split not in ("subject", "row"):
        raise ValueError(f"split must be 'subject' or 'row', got {split!r}")
    if model not in ("l1_logistic", "lightgbm"):
        raise ValueError(f"model must be 'l1_logistic' or 'lightgbm', got {model!r}")
    contract.validate_features(X.columns)

    X_arr = X.to_numpy(dtype=float)
    y, groups = np.asarray(y), np.asarray(groups)
    score_func = SCORE_FUNCS[score] if isinstance(score, str) else score

    ranking = None
    if selection == "leaky":
        # Deliberately leaky: features are scored on every row, test folds included.
        ranking = np.argsort(-_feature_scores(score_func, X_arr, y), kind="stable")

    grid: dict[str, list[float]] = {"select__k": list(k_grid)}
    if model == "l1_logistic":
        grid["model__C"] = list(c_grid)
    else:
        grid["model__num_leaves"] = list(leaves_grid)

    proba = np.full(len(y), np.nan)
    best_params: list[dict[str, float]] = []
    for train, test in _folds(y, groups, split, n_splits, seed):
        inner = _folds(y[train], groups[train], split, N_INNER, seed)
        search = GridSearchCV(
            build_pipeline(score_func, ranking, seed, model),
            grid,
            scoring="roc_auc",
            cv=inner,
            n_jobs=n_jobs,
        )
        search.fit(X_arr[train], y[train])
        proba[test] = search.predict_proba(X_arr[test])[:, 1]
        best_params.append(search.best_params_)

    if np.isnan(proba).any():  # every row must be predicted exactly once
        raise RuntimeError("some rows received no out-of-fold prediction")
    return CVResult(proba, best_params)
