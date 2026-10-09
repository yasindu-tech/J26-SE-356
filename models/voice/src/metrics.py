"""Person-level metrics for the voice module (task VOICE-12).

The unit of evaluation is the person (decision D2): a person's row
probabilities are averaged into one score, and bootstraps resample people,
never rows. Headline metrics are balanced accuracy and AUC with bootstrap 95%
CIs, plus PPV at screening prevalence (CLAUDE.md section 3.4).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

SEED = 42
N_BOOT = 1000
SCREENING_PREVALENCES = (0.01, 0.02, 0.05)

Metric = Callable[[np.ndarray, np.ndarray], float]


class Estimate(NamedTuple):
    value: float  # on the full set of people
    lower: float  # 2.5th percentile of the bootstrap
    upper: float  # 97.5th percentile of the bootstrap


def person_scores(
    y: np.ndarray, scores: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse rows to one (label, mean score) per person, sorted by person id.

    Raises if a person's label differs across rows.
    """
    frame = pd.DataFrame({"g": np.asarray(groups), "y": np.asarray(y), "s": np.asarray(scores)})
    per_person = frame.groupby("g").agg(y=("y", "first"), y_n=("y", "nunique"), s=("s", "mean"))
    if (per_person.y_n > 1).any():
        raise ValueError("a person's label differs across rows")
    return per_person.y.to_numpy(), per_person.s.to_numpy()


def auc(y: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(y, scores))


def balanced_accuracy(y: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> float:
    return float(balanced_accuracy_score(y, np.asarray(scores) >= threshold))


def sensitivity_specificity(
    y: np.ndarray, scores: np.ndarray, threshold: float = 0.5
) -> tuple[float, float]:
    y, pred = np.asarray(y), np.asarray(scores) >= threshold
    return float(pred[y == 1].mean()), float((~pred[y == 0]).mean())


def ppv_at_prevalence(sensitivity: float, specificity: float, prevalence: float) -> float:
    """PPV at a given prevalence: sens*prev / (sens*prev + (1-spec)*(1-prev))."""
    for name, v in (("sensitivity", sensitivity), ("specificity", specificity)):
        if not 0 <= v <= 1:
            raise ValueError(f"{name} must be in [0, 1], got {v}")
    if not 0 < prevalence < 1:
        raise ValueError(f"prevalence must be in (0, 1), got {prevalence}")
    true_pos = sensitivity * prevalence
    false_pos = (1 - specificity) * (1 - prevalence)
    return float(true_pos / (true_pos + false_pos)) if true_pos + false_pos else 0.0


def sensitivity(y: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> float:
    return sensitivity_specificity(y, scores, threshold)[0]


def specificity(y: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> float:
    return sensitivity_specificity(y, scores, threshold)[1]


def ppv_metric(prevalence: float, threshold: float = 0.5) -> Metric:
    """A metric for ``bootstrap_ci``: PPV at ``prevalence`` from the sample's sens and spec.

    The research set's own PD fraction (75% in UCI-470) is never used: sens and
    spec are measured on the sample, then re-weighted to the screening prevalence.
    """

    def ppv(y: np.ndarray, scores: np.ndarray) -> float:
        return ppv_at_prevalence(*sensitivity_specificity(y, scores, threshold), prevalence)

    return ppv


def _bootstrap_indices(y_person: np.ndarray, n_boot: int, seed: int) -> list[np.ndarray]:
    """Resampled person indices. Draws containing only one class are skipped (AUC undefined)."""
    rng = np.random.default_rng(seed)
    n = len(y_person)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_person[idx])) == 2:
            draws.append(idx)
    if len(draws) < 0.9 * n_boot:
        raise ValueError(f"only {len(draws)} of {n_boot} bootstrap draws had both classes")
    return draws


def bootstrap_ci(
    y: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    metric: Metric = auc,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> Estimate:
    """Person-level metric with a 95% bootstrap CI over people.

    Takes row-level inputs; rows are first collapsed to one score per person,
    so all of a person's rows always move together in a resample.
    """
    y_p, s_p = person_scores(y, scores, groups)
    values = [metric(y_p[i], s_p[i]) for i in _bootstrap_indices(y_p, n_boot, seed)]
    lower, upper = np.percentile(values, [2.5, 97.5])
    return Estimate(metric(y_p, s_p), float(lower), float(upper))


def paired_bootstrap_difference(
    y: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    groups: np.ndarray,
    metric: Metric = auc,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> Estimate:
    """metric(A) - metric(B) with a 95% CI, both scored on the same resampled people."""
    y_p, a_p = person_scores(y, scores_a, groups)
    _, b_p = person_scores(y, scores_b, groups)
    diffs = [
        metric(y_p[i], a_p[i]) - metric(y_p[i], b_p[i])
        for i in _bootstrap_indices(y_p, n_boot, seed)
    ]
    lower, upper = np.percentile(diffs, [2.5, 97.5])
    return Estimate(metric(y_p, a_p) - metric(y_p, b_p), float(lower), float(upper))
