"""Metrics for the MRI models: what we report, and with what uncertainty.

* Balanced accuracy and AUC, never raw accuracy: with unequal groups, always guessing the
  bigger group looks good on raw accuracy and means nothing.
* A bootstrap 95% confidence interval on every headline number, resampling people.
* A paired bootstrap for "model beats baseline": both models are scored on the same
  resampled people, and the interval is for the difference. The project's primary success
  criterion is that this interval excludes zero.
* PPV at screening prevalence (1%, 2%, 5%), worked out from sensitivity and specificity,
  because the research set (about half PD) says nothing about how many flagged people in a
  clinic would actually have PD.

All functions take one score per person (a probability) and one label per person.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

SCREENING_PREVALENCE = (0.01, 0.02, 0.05)
Metric = Callable[[np.ndarray, np.ndarray], float]


class MetricError(ValueError):
    """The metric is undefined for these inputs."""


def _check(y: Sequence[int], p: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    y_arr, p_arr = np.asarray(y), np.asarray(p, dtype=float)
    if y_arr.shape != p_arr.shape:
        raise MetricError("labels and scores have different lengths")
    if not set(np.unique(y_arr)) <= {0, 1}:
        raise MetricError("labels must be 0 or 1")
    if len(np.unique(y_arr)) < 2:
        raise MetricError("need both classes")
    if not np.all(np.isfinite(p_arr)):
        raise MetricError("scores have missing values")
    return y_arr.astype(int), p_arr


def sensitivity_specificity(y, p, threshold: float = 0.5) -> tuple[float, float]:
    y, p = _check(y, p)
    pred = p >= threshold
    return float(pred[y == 1].mean()), float((~pred[y == 0]).mean())


def balanced_accuracy(y, p, threshold: float = 0.5) -> float:
    sens, spec = sensitivity_specificity(y, p, threshold)
    return (sens + spec) / 2


def auc(y, p) -> float:
    """Area under the ROC curve (the chance a PD person scores above a control; ties count half)."""
    y, p = _check(y, p)
    pos, neg = p[y == 1], p[y == 0]
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (pos.size * neg.size))


def ppv_at_prevalence(sensitivity: float, specificity: float, prevalence: float) -> float:
    """Share of flagged people who have the condition, at a given prevalence."""
    if not 0 < prevalence < 1:
        raise MetricError("prevalence must be between 0 and 1")
    tp = sensitivity * prevalence
    fp = (1 - specificity) * (1 - prevalence)
    return float(tp / (tp + fp)) if tp + fp > 0 else float("nan")


def _resamples(y: np.ndarray, n_boot: int, seed: int):
    """Indices of bootstrap samples that still contain both classes."""
    rng = np.random.default_rng(seed)
    n, made, tries = len(y), 0, 0
    while made < n_boot:
        tries += 1
        if tries > n_boot * 20:
            raise MetricError("could not draw bootstrap samples with both classes")
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) == 2:
            made += 1
            yield idx


def bootstrap_ci(
    y, p, metric: Metric, n_boot: int = 2000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Percentile interval of ``metric`` over people resampled with replacement."""
    y, p = _check(y, p)
    vals = [metric(y[i], p[i]) for i in _resamples(y, n_boot, seed)]
    lo, hi = np.percentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def paired_difference_ci(
    y, p_model, p_baseline, metric: Metric, n_boot: int = 2000, seed: int = 0, level: float = 0.95
) -> tuple[float, float, float]:
    """(difference, low, high) of metric(model) - metric(baseline) on the same resampled people."""
    y, p_model = _check(y, p_model)
    _, p_baseline = _check(y, p_baseline)
    diff = metric(y, p_model) - metric(y, p_baseline)
    vals = [
        metric(y[i], p_model[i]) - metric(y[i], p_baseline[i]) for i in _resamples(y, n_boot, seed)
    ]
    lo, hi = np.percentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(diff), float(lo), float(hi)


@dataclass(frozen=True)
class Summary:
    n: int
    n_pd: int
    auc: float
    auc_ci: tuple[float, float]
    balanced_accuracy: float
    balanced_accuracy_ci: tuple[float, float]
    sensitivity: float
    specificity: float
    ppv: dict[float, float]  # prevalence -> PPV


def summarise(y, p, threshold: float = 0.5, n_boot: int = 2000, seed: int = 0) -> Summary:
    """Every number we report for one model, with its interval."""
    y, p = _check(y, p)
    sens, spec = sensitivity_specificity(y, p, threshold)
    return Summary(
        n=len(y),
        n_pd=int(y.sum()),
        auc=auc(y, p),
        auc_ci=bootstrap_ci(y, p, auc, n_boot, seed),
        balanced_accuracy=balanced_accuracy(y, p, threshold),
        balanced_accuracy_ci=bootstrap_ci(
            y, p, lambda a, b: balanced_accuracy(a, b, threshold), n_boot, seed
        ),
        sensitivity=sens,
        specificity=spec,
        ppv={prev: ppv_at_prevalence(sens, spec, prev) for prev in SCREENING_PREVALENCE},
    )
