"""ComBat harmonisation fitted inside the training fold, and the two mandatory checks.

CLAUDE.md section 3.2: harmonisation is learned from data, so it is fitted on the training
part of each fold only and then applied, unchanged, to the test part. Two controls must fail
the run loudly, not warn:

* ``site_prediction_gate``: after harmonisation, a classifier should not be able to tell
  which scanner a sample came from any better than with shuffled scanner labels.
* ``label_permutation_control``: with the diagnosis labels shuffled, the whole pipeline must
  score at chance. Above chance means something leaks.

ComBat is implemented here (parametric empirical Bayes, Johnson, Li and Rabinovic 2007, the
same steps as Fortin's ``neuroCombat``) rather than imported, because:

* ``neuroCombat.neuroCombatFromTraining``, the function that applies saved estimates to new
  data, is marked "under development" in neuroCombat 0.2.12 and handles covariates by
  averaging them; and
* we need the fitted parameters saved, to apply to one new scan in ``predict_scan``.

A test checks this implementation against ``neuroCombat`` on synthetic data when that package
is installed. It is not a runtime dependency.

**Diagnosis is not a ComBat covariate.** At prediction time a new person's diagnosis is
unknown, so ComBat cannot use it on the test side; using it only on the training side makes
the two sides differ and is known to inflate results. Only covariates that are known for
every scan (for example age and sex) may be passed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np


class HarmonisationError(ValueError):
    """ComBat cannot be fitted or applied to these inputs."""


class UnseenBatchError(HarmonisationError):
    """A scan comes from a scanner that the training data did not contain."""


class SiteGateError(RuntimeError):
    """The scanner can still be predicted after harmonisation."""


class PermutationControlError(RuntimeError):
    """With shuffled labels the pipeline still scores above chance: something leaks."""


# ================================================================= ComBat


@dataclass
class ComBat:
    """ComBat with fit / transform. Rows are samples, columns are features."""

    eb: bool = True
    tol: float = 1e-4
    max_iter: int = 1000
    min_per_batch: int = 2
    batches_: list[str] = field(default_factory=list)
    grand_mean_: np.ndarray | None = None  # (p,)
    var_pooled_: np.ndarray | None = None  # (p,)
    cov_coef_: np.ndarray | None = None  # (k, p) or None
    gamma_star_: np.ndarray | None = None  # (b, p)
    delta_star_: np.ndarray | None = None  # (b, p)

    # ------------------------------------------------------------- fit
    def fit(
        self, x: np.ndarray, batch: Sequence[str], covariates: np.ndarray | None = None
    ) -> ComBat:
        x = _as_matrix(x)
        batch_arr = np.asarray(batch).astype(str)
        cov = _as_covariates(covariates, len(x))
        if len(batch_arr) != len(x):
            raise HarmonisationError("batch labels and data have different lengths")
        levels, counts = np.unique(batch_arr, return_counts=True)
        small = levels[counts < self.min_per_batch]
        if small.size:
            raise HarmonisationError(
                f"{small.size} batch(es) have fewer than {self.min_per_batch} training samples"
            )
        if levels.size < 2:
            raise HarmonisationError("ComBat needs at least two batches")

        n, _ = x.shape
        onehot = (batch_arr[:, None] == levels[None, :]).astype(float)
        design = onehot if cov is None else np.hstack([onehot, cov])
        coef, *_ = np.linalg.lstsq(design, x, rcond=None)  # (b + k, p)
        b = levels.size
        grand_mean = (counts / n) @ coef[:b]
        var_pooled = ((x - design @ coef) ** 2).mean(axis=0)
        if np.any(var_pooled <= 0):
            raise HarmonisationError("a feature has no variance left; drop constant features")
        cov_coef = None if cov is None else coef[b:]
        stand_mean = grand_mean + (0 if cov is None else cov @ cov_coef)
        s = (x - stand_mean) / np.sqrt(var_pooled)

        gamma_hat = np.vstack([s[batch_arr == lv].mean(axis=0) for lv in levels])
        delta_hat = np.vstack([s[batch_arr == lv].var(axis=0, ddof=1) for lv in levels])
        if np.any(delta_hat <= 0):
            raise HarmonisationError("a batch has zero variance for some feature")

        if self.eb:
            gamma_star, delta_star = self._empirical_bayes(
                s, batch_arr, levels, gamma_hat, delta_hat
            )
        else:
            gamma_star, delta_star = gamma_hat, delta_hat

        self.batches_ = [str(v) for v in levels]
        self.grand_mean_, self.var_pooled_, self.cov_coef_ = grand_mean, var_pooled, cov_coef
        self.gamma_star_, self.delta_star_ = gamma_star, delta_star
        return self

    def _empirical_bayes(self, s, batch_arr, levels, gamma_hat, delta_hat):
        gamma_bar = gamma_hat.mean(axis=1)
        t2 = gamma_hat.var(axis=1, ddof=1)
        m = delta_hat.mean(axis=1)
        s2 = delta_hat.var(axis=1, ddof=1)
        a_prior = (2 * s2 + m**2) / s2
        b_prior = (m * s2 + m**3) / s2
        g_out, d_out = [], []
        for i, lv in enumerate(levels):
            rows = s[batch_arr == lv]
            n = rows.shape[0]
            g_old, d_old = gamma_hat[i], delta_hat[i]
            for _ in range(self.max_iter):
                g_new = (t2[i] * n * gamma_hat[i] + d_old * gamma_bar[i]) / (t2[i] * n + d_old)
                sum2 = ((rows - g_new) ** 2).sum(axis=0)
                d_new = (0.5 * sum2 + b_prior[i]) / (n / 2.0 + a_prior[i] - 1.0)
                change = max(
                    np.max(np.abs(g_new - g_old) / np.abs(g_old)),
                    np.max(np.abs(d_new - d_old) / d_old),
                )
                g_old, d_old = g_new, d_new
                if change <= self.tol:
                    break
            else:
                raise HarmonisationError("empirical Bayes step did not converge")
            g_out.append(g_old)
            d_out.append(d_old)
        return np.vstack(g_out), np.vstack(d_out)

    # ------------------------------------------------------------- apply
    def transform(
        self, x: np.ndarray, batch: Sequence[str], covariates: np.ndarray | None = None
    ) -> np.ndarray:
        if self.grand_mean_ is None:
            raise HarmonisationError("ComBat is not fitted")
        x = _as_matrix(x)
        if x.shape[1] != self.grand_mean_.size:
            raise HarmonisationError("number of features differs from the fitted data")
        batch_arr = np.asarray(batch).astype(str)
        unseen = sorted(set(batch_arr) - set(self.batches_))
        if unseen:
            raise UnseenBatchError(f"{len(unseen)} batch(es) not in the training data")
        cov = _as_covariates(covariates, len(x))
        if (cov is None) != (self.cov_coef_ is None):
            raise HarmonisationError("covariates must be given at fit and transform, or neither")
        stand_mean = self.grand_mean_ + (0 if cov is None else cov @ self.cov_coef_)
        idx = np.array([self.batches_.index(v) for v in batch_arr])
        s = (x - stand_mean) / np.sqrt(self.var_pooled_)
        adj = (s - self.gamma_star_[idx]) / np.sqrt(self.delta_star_[idx])
        return adj * np.sqrt(self.var_pooled_) + stand_mean

    def fit_transform(self, x, batch, covariates=None) -> np.ndarray:
        return self.fit(x, batch, covariates).transform(x, batch, covariates)

    # ------------------------------------------------------------- save
    def to_dict(self) -> dict:
        """Plain lists, for a JSON model bundle (the frozen ComBat parameters)."""
        if self.grand_mean_ is None:
            raise HarmonisationError("ComBat is not fitted")
        return {
            "batches": self.batches_,
            "grand_mean": self.grand_mean_.tolist(),
            "var_pooled": self.var_pooled_.tolist(),
            "cov_coef": None if self.cov_coef_ is None else self.cov_coef_.tolist(),
            "gamma_star": self.gamma_star_.tolist(),
            "delta_star": self.delta_star_.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ComBat:
        m = cls()
        m.batches_ = list(d["batches"])
        m.grand_mean_ = np.asarray(d["grand_mean"], dtype=float)
        m.var_pooled_ = np.asarray(d["var_pooled"], dtype=float)
        m.cov_coef_ = None if d["cov_coef"] is None else np.asarray(d["cov_coef"], dtype=float)
        m.gamma_star_ = np.asarray(d["gamma_star"], dtype=float)
        m.delta_star_ = np.asarray(d["delta_star"], dtype=float)
        return m


def _as_matrix(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise HarmonisationError("data must be a 2D table: samples x features")
    if not np.all(np.isfinite(x)):
        raise HarmonisationError("data has missing or infinite values")
    return x


def _as_covariates(cov, n: int) -> np.ndarray | None:
    if cov is None:
        return None
    cov = np.asarray(cov, dtype=float)
    cov = cov[:, None] if cov.ndim == 1 else cov
    if cov.shape[0] != n or not np.all(np.isfinite(cov)):
        raise HarmonisationError("covariates must be finite, one row per sample")
    return cov


# ================================================================= inside the fold


def out_of_fold_harmonise(
    x: np.ndarray,
    batch: Sequence[str],
    folds: Iterable[tuple[np.ndarray, np.ndarray]],
    covariates: np.ndarray | None = None,
) -> tuple[np.ndarray, list[ComBat]]:
    """Harmonise every row with a ComBat fitted on the other folds only.

    ``folds`` is one repeat of (train_idx, test_idx) pairs in which every row is tested once.
    Returns the harmonised table and the fitted model of each fold.
    """
    x = _as_matrix(x)
    batch_arr = np.asarray(batch).astype(str)
    cov = _as_covariates(covariates, len(x))
    out = np.full_like(x, np.nan)
    times = np.zeros(len(x), dtype=int)
    models = []
    for train, test in folds:
        if set(train) & set(test):
            raise HarmonisationError("a row is in both the training and the test part")
        model = ComBat().fit(x[train], batch_arr[train], None if cov is None else cov[train])
        out[test] = model.transform(x[test], batch_arr[test], None if cov is None else cov[test])
        times[test] += 1
        models.append(model)
    if not np.all(times == 1):
        raise HarmonisationError("folds must test every row exactly once")
    return out, models


# ================================================================= the two controls


@dataclass(frozen=True)
class ControlResult:
    score: float  # observed score (site gate) or mean shuffled-label score (permutation)
    threshold: float
    null_scores: tuple[float, ...]
    passed: bool


def site_predictability(x: np.ndarray, batch: Sequence[str], seed: int = 0) -> float:
    """Cross-validated balanced accuracy of guessing the scanner from the features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    batch_arr = np.asarray(batch).astype(str)
    _, counts = np.unique(batch_arr, return_counts=True)
    n_splits = int(min(5, counts.min()))
    if n_splits < 2:
        raise HarmonisationError("every scanner needs at least two samples for the site gate")
    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = cross_val_predict(model, _as_matrix(x), batch_arr, cv=cv)
    return float(balanced_accuracy_score(batch_arr, pred))


def site_prediction_gate(
    x_harmonised: np.ndarray,
    batch: Sequence[str],
    n_permutations: int = 100,
    alpha: float = 0.05,
    seed: int = 0,
    raise_on_fail: bool = True,
) -> ControlResult:
    """Pass when the scanner is no easier to guess than with shuffled scanner labels.

    The threshold is the (1 - alpha) quantile of the shuffled-label scores, fixed before the
    real score is looked at. Use the out-of-fold harmonised table.
    """
    rng = np.random.default_rng(seed)
    batch_arr = np.asarray(batch).astype(str)
    observed = site_predictability(x_harmonised, batch_arr, seed)
    null = tuple(
        site_predictability(x_harmonised, rng.permutation(batch_arr), seed)
        for _ in range(n_permutations)
    )
    threshold = float(np.quantile(null, 1 - alpha))
    result = ControlResult(observed, threshold, null, observed <= threshold)
    if raise_on_fail and not result.passed:
        raise SiteGateError(
            f"scanner still predictable after harmonisation: balanced accuracy {observed:.3f} "
            f"> shuffled-label threshold {threshold:.3f}"
        )
    return result


def label_permutation_control(
    evaluate: Callable[[np.ndarray], float],
    y: Sequence[int],
    subjects: Sequence[str] | None = None,
    n_permutations: int = 20,
    chance: float = 0.5,
    tolerance: float = 0.05,
    seed: int = 0,
    raise_on_fail: bool = True,
) -> ControlResult:
    """Run the whole pipeline with shuffled labels; it must score at chance.

    ``evaluate(y)`` runs everything learned from data (folds, ComBat, selection, model) with
    the labels it is given and returns the mean balanced accuracy. Labels are shuffled per
    person, so rows of one person keep one label. Fails if the mean shuffled score is above
    ``chance + tolerance``.
    """
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y)
    subj = np.arange(len(y_arr)).astype(str) if subjects is None else np.asarray(subjects)
    people, first = np.unique(subj, return_index=True)
    scores = []
    for _ in range(n_permutations):
        shuffled = dict(zip(people, rng.permutation(y_arr[first]), strict=True))
        scores.append(float(evaluate(np.array([shuffled[s] for s in subj]))))
    mean = float(np.mean(scores))
    threshold = chance + tolerance
    result = ControlResult(mean, threshold, tuple(scores), mean <= threshold)
    if raise_on_fail and not result.passed:
        raise PermutationControlError(
            f"shuffled labels score {mean:.3f} on average, above chance + tolerance "
            f"({threshold:.3f}); something leaks"
        )
    return result
