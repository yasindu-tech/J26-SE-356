"""Scores for the severity classifier, with subject-level bootstrap intervals (GAIT-19 to 21).

Balanced accuracy and AUC are the headline scores (CLAUDE.md 3.4); macro-F1 gets the same
interval. GAIT-19 (training) and GAIT-20 (cohorts, ablations, noise) share this code.

Decisions:

1. The bootstrap resamples subjects, not walks: the walks of one person are not independent.
2. A score is computed per seed on the pooled out-of-fold predictions, then averaged over seeds.
   Probabilities are never averaged across seeds.
3. Walk-level and subject-weighted scores: PD-GaM is 61% of the walks.
4. AUC (one-vs-rest, macro) and macro-F1 are written with sorts and counts, not scikit-learn,
   to keep 1,000 bootstrap draws fast; the tests check them against scikit-learn.
5. Several models can share one bootstrap (``bootstrap_draws``), so differences are paired.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import numpy as np

CLASSES = (0, 1, 2)
METRICS = ("bal_acc", "auc", "bal_acc_subject", "auc_subject", "macro_f1")


def weighted_auc(positive: np.ndarray, score: np.ndarray, weight: np.ndarray) -> float:
    """Probability that a positive scores above a negative (ties count half), weighted.

    NaN if there are no positives or no negatives.
    """
    pos = positive.astype(bool)
    w_pos = float(weight[pos].sum())
    w_neg = float(weight[~pos].sum())
    if w_pos <= 0 or w_neg <= 0:
        return float("nan")
    values, inverse = np.unique(score, return_inverse=True)
    n = len(values)
    p = np.bincount(inverse, weights=weight * pos, minlength=n)
    q = np.bincount(inverse, weights=weight * ~pos, minlength=n)
    negatives_below = np.cumsum(q) - q
    return float((p * (negatives_below + 0.5 * q)).sum() / (w_pos * w_neg))


def macro_auc(y: np.ndarray, proba: np.ndarray, weight: np.ndarray | None = None) -> float:
    """One-vs-rest AUC averaged over the classes that have both positives and negatives."""
    w = np.ones(len(y)) if weight is None else weight
    values = [weighted_auc(y == c, proba[:, i], w) for i, c in enumerate(CLASSES)]
    values = [v for v in values if not np.isnan(v)]
    return float(np.mean(values)) if values else float("nan")


def balanced_accuracy(y: np.ndarray, pred: np.ndarray, weight: np.ndarray | None = None) -> float:
    """Mean recall over the classes present in ``y`` (weighted)."""
    w = np.ones(len(y)) if weight is None else weight
    recalls = []
    for c in CLASSES:
        in_class = y == c
        total = float(w[in_class].sum())
        if total > 0:
            recalls.append(float(w[in_class & (pred == c)].sum()) / total)
    return float(np.mean(recalls)) if recalls else float("nan")


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    """Unweighted mean of the per-class F1 over the three classes (0 for a class never met).

    Written with counts, not scikit-learn, because the bootstrap calls it for every draw, seed
    and model; ``tests/test_gait_metrics.py`` checks it against ``f1_score``.
    """
    n = len(CLASSES)
    tp = np.bincount(y[y == pred], minlength=n).astype(float)
    n_true = np.bincount(y, minlength=n).astype(float)
    n_pred = np.bincount(pred, minlength=n).astype(float)
    denominator = n_true + n_pred
    f1 = np.divide(2 * tp, denominator, out=np.zeros(n), where=denominator > 0)
    return float(f1.mean())


def confusion(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Counts, rows = true class, columns = predicted class."""
    out = np.zeros((len(CLASSES), len(CLASSES)), dtype=float)
    np.add.at(out, (y, pred), 1.0)
    return out


def metric_values(
    y: np.ndarray, probas: Sequence[np.ndarray], subject_weight: np.ndarray
) -> dict[str, float]:
    """The five scores: each is computed per seed, then averaged over the seeds.

    ``probas`` holds one (walks x 3) out-of-fold probability array per seed.
    ``subject_weight`` is 1 / (walks of the walk's subject).
    """
    ones = np.ones(len(y))
    rows: dict[str, list[float]] = {m: [] for m in METRICS}
    for proba in probas:
        pred = proba.argmax(axis=1)
        rows["bal_acc"].append(balanced_accuracy(y, pred, ones))
        rows["auc"].append(macro_auc(y, proba, ones))
        rows["bal_acc_subject"].append(balanced_accuracy(y, pred, subject_weight))
        rows["auc_subject"].append(macro_auc(y, proba, subject_weight))
        rows["macro_f1"].append(macro_f1(y, pred))
    return {
        m: float(np.nanmean(v)) if not np.all(np.isnan(v)) else float("nan")
        for m, v in rows.items()
    }


def subject_weights(subjects: np.ndarray) -> np.ndarray:
    """1 / (walks of the subject) for every walk, scaled to a mean of 1."""
    _, inverse, counts = np.unique(subjects, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inverse]
    return w * len(w) / w.sum()


def subject_groups(subjects: np.ndarray) -> list[np.ndarray]:
    """Row positions of every subject (one array per subject, in sorted subject order)."""
    _, inverse = np.unique(subjects, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse)
    return np.split(order, np.cumsum(counts)[:-1])


def bootstrap_draws(
    y: np.ndarray,
    models: Mapping[str, Sequence[np.ndarray]],
    subjects: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """The scores of every model in every bootstrap draw (``n_boot`` x ``len(METRICS)``).

    Each draw resamples whole subjects and applies the same resample to every model, so draws
    of different models are paired. ``models`` maps a name to its out-of-fold probabilities, one
    (walks x 3) array per seed.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    weight = subject_weights(subjects)
    groups = subject_groups(subjects)
    rng = np.random.default_rng(seed)
    draws = {name: np.empty((n_boot, len(METRICS))) for name in models}
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        for name, probas in models.items():
            values = metric_values(y[idx], [p[idx] for p in probas], weight[idx])
            draws[name][b] = [values[m] for m in METRICS]
    return draws


def percentile_interval(draws: np.ndarray, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper percentile bounds of each column of ``draws`` (NaN if a column is all NaN)."""
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # an all-NaN column gives NaN, not a crash
        lo, hi = np.nanpercentile(draws, [50 * (1 - level), 50 * (1 + level)], axis=0)
    return lo, hi


def bootstrap_intervals(
    y: np.ndarray,
    probas: Sequence[np.ndarray],
    subjects: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Percentile intervals for every score in ``METRICS``, resampling whole subjects."""
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    draws = bootstrap_draws(y, {"model": probas}, subjects, n_boot=n_boot, seed=seed)["model"]
    lo, hi = percentile_interval(draws, level)
    return {m: (float(lo[i]), float(hi[i])) for i, m in enumerate(METRICS)}
