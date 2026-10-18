"""GAIT-20: how far the GAIT-19 scores can be trusted: by cohort, on an unseen cohort, by input.

Run from the repository root (venv active, after ``train_xgboost.py``)::

    python models/gait/src/evaluate_cohorts.py

The GAIT-19 headline (balanced accuracy 0.55 against a chance of 0.33) is one number pooled over
four clinical cohorts of very different size (PD-GaM is 61% of the walks). The cohorts also
differ in how they were recorded, and the eight inputs identify the cohort with 0.72 balanced
accuracy, so a model can look good by recognising the cohort instead of the gait. Three analyses
look at that:

1. *Scores by cohort.* The out-of-fold predictions of GAIT-19 are pooled over all folds and scored
   inside each cohort, with the baseline ladder next to the model. A cohort score is never taken
   per fold (a fold holds 12-33 subjects and some cohorts are absent from a fold).
2. *Leave one cohort out.* Train on three cohorts, test on the fourth, which the model has never
   seen in any form. This is the strictest test of "grades the gait, not the cohort". It is
   closer to the demo than cross-validation is, because a new video is a new setup.
3. *Ablations.* The same folds and the same XGBoost settings, with one input group removed
   (``stride_time_cv``, ``arm_swing_range_norm``, ``hip_flexion_range_deg``, hip and knee, all
   three joint-angle inputs, or everything except the three timing inputs), or one cohort removed
   (BMCLab, PD-GaM). They show what the score rests on. The model without an input group is also
   fitted with logistic regression, because logistic regression scored as well as XGBoost in
   GAIT-19.

Decisions:

1. *Differences are paired.* Every bootstrap draw (whole subjects, ``gait_metrics.py``) is applied
   to all models, and the interval is taken on the difference. Two overlapping intervals do not
   mean that a difference is zero; the interval of the difference says so.
2. *The reference of an ablation is recomputed here*, by the same code, in the same environment.
   The ``all_inputs`` variant is compared with the GAIT-19 predictions file and the largest
   difference is printed. On the machine that made the file it is 0 (to the 6 decimals of the
   file). On another machine a fit can differ: the day-6 review found one fit in 25 that differs
   between the Mac and the sandbox (same XGBoost version, other scikit-learn and numpy). The
   ablations never mix the two: they are compared with the recomputed reference.
3. *Removing a cohort removes it from training and from testing.* The reference is the full
   model's out-of-fold predictions on the walks that are left, so the comparison is on the same
   walks. The folds are the GAIT-18 folds with the removed subjects dropped (the gate of at least
   3 test subjects per class and fold is re-checked; a miss is reported, not hidden).
4. *Leave-one-cohort-out has no "cohort only" rung* (an unseen cohort gets the overall prior, which
   is the majority rung). The models are fitted once per held-out cohort with the fixed GAIT-19
   settings. The training part never contains the held-out cohort (asserted).
5. *Label-permutation controls (CLAUDE.md 3.2).* In cross-validation the shuffled-label model must
   land at chance (same margin as GAIT-19), for the GAIT-19 model and for every ablation, or the
   run exits with code 1. On an unseen cohort chance is not 1/3 and 0.5: the class mix differs
   between cohorts (the majority rung scores 0.305 pooled) and a model fitted to noise lands
   anywhere within a wide band (sd 0.04-0.11 on one cohort). A fixed margin around 1/3 would
   fail for a reason that is not leakage (leakage cannot happen: the cohorts are disjoint). So
   leave-one-cohort-out is tested against ``N_SHUFFLES`` shuffled-label models, and a score is
   reported with the share of shuffles that reach it (a one-sided permutation p-value, smallest
   possible 1 / (N_SHUFFLES + 1)).
6. *No tuning, no new model.* Nothing here changes the GAIT-19 model. The final model in
   ``artifacts/models/xgboost`` stays as it is.

The model grades gait impairment (UPDRS gait 0 / 1 / 2-3) inside four labelled clinical cohorts.
It does not separate people with PD from people without. Class 0 reads "no gait impairment
pattern detected in this walk", never "healthy" and never "no PD".

Writes to ``artifacts/models/cohorts/`` (git-ignored): ``cohort_scores.csv``,
``cohort_differences.csv``, ``cohort_class_shares.csv`` (true and predicted shares, recall),
``loco_null.csv``, ``ablations.csv``,
``ablation_differences.csv``, ``loco_predictions.csv`` and ``results.json``.
Exit code 0 = ok, 1 = a fold problem or a cross-validation permutation control failed, 2 = an input
is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gait_metrics as gm
import numpy as np
import pandas as pd
import train_xgboost as tx
import xgboost
from feature_contract import check_feature_contract
from gait_extract import FEATURE_NAMES
from make_folds import CLASS_LABELS, fingerprint, fold_problems, walk_folds
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_TABLE = tx.DEFAULT_TABLE
DEFAULT_FOLDS = tx.DEFAULT_FOLDS
DEFAULT_OOF = tx.DEFAULT_OUT / "oof_predictions.csv"
DEFAULT_OUT = tx.GAIT_DIR / "artifacts" / "models" / "cohorts"

LOCO_PREFIX = "loco_"
LOCO_MODELS = ("majority", "best_feature", "logistic", "xgboost")
N_SHUFFLES = 100  # shuffled-label models per held-out cohort
ABLATION_MODELS = ("xgboost", "logistic", tx.CONTROL)
REFERENCE = "all_inputs"
REPRODUCTION_TOLERANCE = 2e-6  # the predictions file has 6 decimals
POOLED = "all"

# (model, versus): the difference "model minus versus" is estimated for every cohort.
COHORT_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("xgboost", "majority"),
    ("xgboost", "cohort_only"),
    ("xgboost", "best_feature"),
    ("xgboost", "logistic"),
    (f"{LOCO_PREFIX}xgboost", "xgboost"),
    (f"{LOCO_PREFIX}xgboost", f"{LOCO_PREFIX}majority"),
    (f"{LOCO_PREFIX}xgboost", f"{LOCO_PREFIX}logistic"),
)


# --------------------------------------------------------------------------- the variants
@dataclass(frozen=True)
class Variant:
    """One ablation: a set of inputs, and optionally a cohort removed from the data."""

    name: str
    description: str
    features: tuple[str, ...] = FEATURE_NAMES
    drop_cohort: str | None = None


def _without(*dropped: str) -> tuple[str, ...]:
    unknown = [n for n in dropped if n not in FEATURE_NAMES]
    if unknown:
        raise ValueError(f"not model inputs: {unknown}")
    return tuple(n for n in FEATURE_NAMES if n not in dropped)


JOINT_ANGLES = ("knee_flexion_range_deg", "hip_flexion_range_deg", "arm_swing_range_norm")
TIMING = ("stride_time_mean", "cadence", "stride_time_cv")

VARIANTS: tuple[Variant, ...] = (
    Variant(REFERENCE, "all eight inputs, as in GAIT-19"),
    Variant("no_stride_time_cv", "without stride_time_cv", _without("stride_time_cv")),
    Variant(
        "no_arm_swing_range_norm",
        "without arm_swing_range_norm",
        _without("arm_swing_range_norm"),
    ),
    Variant(
        "no_hip_flexion_range_deg",
        "without hip_flexion_range_deg (the input SHAP ranks first)",
        _without("hip_flexion_range_deg"),
    ),
    Variant(
        "no_hip_and_knee_flexion",
        "without hip and knee flexion range",
        _without("hip_flexion_range_deg", "knee_flexion_range_deg"),
    ),
    Variant(
        "no_joint_angles",
        "without the three joint-angle inputs (knee, hip, arm swing)",
        _without(*JOINT_ANGLES),
    ),
    Variant(
        "timing_only",
        "only the three timing inputs (stride_time_mean, cadence, stride_time_cv)",
        TIMING,
    ),
    Variant(
        "no_BMCLab",
        "BMCLab removed from training and testing",
        FEATURE_NAMES,
        "BMCLab",
    ),
    Variant(
        "no_PD-GaM",
        "PD-GaM (61% of the walks) removed from training and testing",
        FEATURE_NAMES,
        "PD-GaM",
    ),
)


# --------------------------------------------------------------------------- inputs
def read_oof(path: Path) -> pd.DataFrame:
    """The GAIT-19 out-of-fold predictions file."""
    oof = pd.read_csv(path, dtype={"walk": str, "cohort": str, "subject_key": str})
    needed = {"model", "seed", "walk", "cohort", "subject_key", "severity", *tx.PROBA_COLUMNS}
    missing = needed - set(oof.columns)
    if missing:
        raise ValueError(f"the predictions file lacks the columns {sorted(missing)}")
    return oof


def cohort_names(table: pd.DataFrame) -> list[str]:
    return sorted(str(c) for c in table["cohort"].unique())


def oof_probabilities(oof: pd.DataFrame, table: pd.DataFrame, model: str) -> list[np.ndarray]:
    """The out-of-fold probabilities of one model: one (walks x 3) array per seed, in table order.

    Refuses a file that does not match the table: every walk must be there once per seed, with the
    same cohort, subject and label.
    """
    part = oof[oof["model"] == model]
    if part.empty:
        raise ValueError(f"the predictions file has no rows for model {model!r}")
    walks = table["walk"].to_numpy()
    out = []
    for seed in sorted(part["seed"].unique()):
        rows = part[part["seed"] == seed]
        if rows["walk"].duplicated().any() or set(rows["walk"]) != set(walks):
            raise ValueError(
                f"model {model!r}, seed {seed}: the predictions do not cover each walk of the "
                "feature table exactly once; rerun train_xgboost.py"
            )
        rows = rows.set_index("walk").loc[walks]
        for column in ("severity", "cohort", "subject_key"):
            if not np.array_equal(rows[column].to_numpy(), table[column].to_numpy()):
                raise ValueError(
                    f"model {model!r}, seed {seed}: {column} differs from the feature table; "
                    "rerun train_xgboost.py"
                )
        proba = rows[list(tx.PROBA_COLUMNS)].to_numpy(dtype=float)
        if not np.isfinite(proba).all() or np.abs(proba.sum(axis=1) - 1).max() > 1e-3:
            raise ValueError(f"model {model!r}, seed {seed}: probabilities are not valid")
        out.append(proba)
    return out


# --------------------------------------------------------------------------- leave one cohort out
@dataclass
class LocoResult:
    """Predictions for every walk by a model that never saw the walk's cohort.

    ``proba`` holds the models (one array each); ``null`` the shuffled-label XGBoost models
    (one (walks x 3) array per shuffle).
    """

    proba: dict[str, list[np.ndarray]] = field(default_factory=dict)
    null: list[np.ndarray] = field(default_factory=list)
    train_cohorts: dict[str, tuple[str, ...]] = field(default_factory=dict)


def leave_one_cohort_out(
    table: pd.DataFrame,
    params: dict[str, Any] | None = None,
    models: Sequence[str] = LOCO_MODELS,
    n_shuffles: int = N_SHUFFLES,
) -> LocoResult:
    """Fit on all cohorts but one and predict the one left out, for every cohort in turn."""
    unknown = [m for m in models if m not in tx.LEARNERS or m == tx.CONTROL]
    if unknown:
        raise ValueError(f"unknown models {unknown}")
    params = tx.XGB_PARAMS if params is None else params
    cohorts = cohort_names(table)
    if len(cohorts) < 2:
        raise ValueError("leave-one-cohort-out needs at least two cohorts")
    y_all = table["severity"].to_numpy(dtype=int)
    n = len(table)
    result = LocoResult()
    for m in models:
        result.proba[m] = [np.full((n, len(gm.CLASSES)), np.nan)]
    result.null = [np.full((n, len(gm.CLASSES)), np.nan) for _ in range(n_shuffles)]
    for i, held in enumerate(cohorts):
        test_idx = np.flatnonzero(table["cohort"].to_numpy() == held)
        train_idx = np.flatnonzero(table["cohort"].to_numpy() != held)
        train, test = table.iloc[train_idx], table.iloc[test_idx]
        tx.assert_no_leakage(train, test)
        if held in set(train["cohort"]):
            raise tx.LeakageError(f"cohort {held} is in the training part of its own test")
        y_train = y_all[train_idx]
        if set(y_train) != set(gm.CLASSES):
            raise ValueError(f"without {held} the training part lacks a class")
        result.train_cohorts[held] = tuple(c for c in cohorts if c != held)
        ones = np.ones(len(train))
        for m in models:
            proba, _ = tx.LEARNERS[m](train, test, y_train, ones, 0, params)
            result.proba[m][0][test_idx] = proba
        for j in range(n_shuffles):
            labels = np.random.default_rng([j, i]).permutation(y_train)
            proba, _ = tx.LEARNERS["xgboost"](train, test, labels, ones, j, params)
            result.null[j][test_idx] = proba
    for m, arrays in result.proba.items():
        if any(np.isnan(a).any() for a in arrays):
            raise RuntimeError(f"{m}: a walk was not predicted")
    if any(np.isnan(a).any() for a in result.null):
        raise RuntimeError("a walk was not predicted by a shuffled-label model")
    return result


def loco_frame(table: pd.DataFrame, result: LocoResult) -> pd.DataFrame:
    """Long table of the leave-one-cohort-out predictions: one row per walk and model.

    The shuffled-label models are not written (100 x 2,738 rows); ``loco_null`` summarises them.
    """
    base = table[["walk", "cohort", "subject_key", "severity"]]
    parts = []
    for m, arrays in result.proba.items():
        for seed, proba in enumerate(arrays):
            part = base.copy()
            part.insert(0, "model", f"{LOCO_PREFIX}{m}")
            part.insert(1, "seed", seed)
            for i, name in enumerate(tx.PROBA_COLUMNS):
                part[name] = proba[:, i]
            part["pred"] = proba.argmax(axis=1)
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def cohort_groups(table: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    """Every cohort as a row mask, then all walks pooled."""
    groups = [(c, (table["cohort"] == c).to_numpy()) for c in cohort_names(table)]
    groups.append((POOLED, np.ones(len(table), dtype=bool)))
    return groups


def loco_null(table: pd.DataFrame, result: LocoResult) -> pd.DataFrame:
    """The leave-one-cohort-out XGBoost against the shuffled-label models, per cohort and pooled.

    One row per group and score: the real score, the mean, sd and 95th percentile of the shuffled
    models, and the share of shuffles that reach the real score (one-sided, counting the real
    model as one more draw so that it is never 0).
    """
    y = table["severity"].to_numpy(dtype=int)
    subjects = table["subject_key"].to_numpy()
    real = result.proba["xgboost"]
    rows = []
    for name, mask in cohort_groups(table):
        w = gm.subject_weights(subjects[mask])
        real_scores = gm.metric_values(y[mask], [p[mask] for p in real], w)
        null = np.array(
            [[gm.metric_values(y[mask], [p[mask]], w)[m] for m in gm.METRICS] for p in result.null]
        )
        for i, metric in enumerate(gm.METRICS):
            r, draws = real_scores[metric], null[:, i]
            reached = int(np.sum(draws >= r)) if not np.isnan(r) else 0
            rows.append(
                {
                    "cohort": name,
                    "metric": metric,
                    "real": r,
                    "null_mean": float(np.nanmean(draws)),
                    "null_sd": float(np.nanstd(draws, ddof=1)),
                    "null_p95": float(np.nanpercentile(draws, 95)),
                    "n_shuffles": len(draws),
                    "p_value": (1 + reached) / (1 + len(draws)),
                }
            )
    return pd.DataFrame(rows)


def class_shares(table: pd.DataFrame, models: Mapping[str, Sequence[np.ndarray]]) -> pd.DataFrame:
    """Per cohort and class: the true share of walks, the predicted share and the recall.

    Predicted share and recall are means over the seeds. They show whether a model on an unseen
    cohort ranks the classes but puts them at the wrong thresholds, which a balanced accuracy
    near chance does not tell. Recall is NaN for a class the cohort has no walks of.
    """
    y = table["severity"].to_numpy(dtype=int)
    rows = []
    for name, mask in cohort_groups(table):
        n_true = np.bincount(y[mask], minlength=len(gm.CLASSES))
        for m, probas in models.items():
            pred = [p[mask].argmax(axis=1) for p in probas]
            predicted = np.mean([np.bincount(q, minlength=len(gm.CLASSES)) for q in pred], axis=0)
            for c in gm.CLASSES:
                hit = np.mean([int(np.sum((q == c) & (y[mask] == c))) for q in pred])
                rows.append(
                    {
                        "cohort": name,
                        "model": m,
                        "class": CLASS_LABELS[c],
                        "true_share": float(n_true[c] / mask.sum()),
                        "predicted_share": float(predicted[c] / mask.sum()),
                        "recall": float(hit / n_true[c]) if n_true[c] else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- ablations
@dataclass
class VariantResult:
    """Out-of-fold probabilities of one ablation, on the rows of ``table``."""

    variant: Variant
    table: pd.DataFrame
    seeds: tuple[int, ...]
    proba: dict[str, list[np.ndarray]]
    fold_problems: list[str]


def fit_variant_model(
    kind: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    features: Sequence[str],
    seed: int,
    params: dict[str, Any],
) -> np.ndarray:
    """Test probabilities (rows x 3) of ``kind`` fitted on ``features`` only.

    The feature contract (CLAUDE.md 3.1) is called on exactly the columns that are used.
    """
    columns = list(features)
    check_feature_contract(columns, label_name=tx.LABEL_NAME)
    x_train = train[columns].to_numpy(dtype=float)
    x_test = test[columns].to_numpy(dtype=float)
    ones = np.ones(len(y))
    if kind == "logistic":
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        model.fit(x_train, y, logisticregression__sample_weight=ones)
        return tx._full(model.predict_proba(x_test), model.classes_)
    if kind in ("xgboost", tx.CONTROL):
        booster = tx.fit_xgboost_model(x_train, y, ones, seed, params)
        return tx._full(booster.predict_proba(x_test), booster.classes_)
    raise ValueError(f"unknown model {kind!r}")


def run_variant(
    table: pd.DataFrame,
    assignments: pd.DataFrame,
    variant: Variant,
    params: dict[str, Any] | None = None,
    seeds: Sequence[int] | None = None,
    models: Sequence[str] = ABLATION_MODELS,
) -> VariantResult:
    """Cross-validate ``models`` on the GAIT-18 folds with the inputs and cohorts of ``variant``."""
    params = tx.XGB_PARAMS if params is None else params
    missing = [f for f in variant.features if f not in table.columns]
    if missing:
        raise ValueError(f"{variant.name}: the table lacks the columns {missing}")
    if variant.drop_cohort is None:
        sub = table
        problems: list[str] = []
    else:
        if variant.drop_cohort not in set(table["cohort"]):
            raise ValueError(f"{variant.name}: no cohort {variant.drop_cohort!r} in the table")
        sub = table[table["cohort"] != variant.drop_cohort].reset_index(drop=True)
    assign = assignments[assignments["subject_key"].isin(set(sub["subject_key"]))]
    if variant.drop_cohort is not None:
        problems = fold_problems(sub, assign)
    all_seeds = tuple(sorted(int(s) for s in set(assign["seed"])))
    use = all_seeds if seeds is None else tuple(seeds)
    y_all = sub["severity"].to_numpy(dtype=int)
    proba: dict[str, list[np.ndarray]] = {m: [] for m in models}
    for seed in use:
        folds = walk_folds(sub, assign, seed)
        out = {m: np.full((len(sub), len(gm.CLASSES)), np.nan) for m in models}
        for k in sorted(set(folds)):
            train_idx, test_idx = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
            train, test = sub.iloc[train_idx], sub.iloc[test_idx]
            tx.assert_no_leakage(train, test)
            y_train = y_all[train_idx]
            if set(y_train) != set(gm.CLASSES):
                raise ValueError(f"{variant.name}, seed {seed} fold {k}: training lacks a class")
            for m in models:
                labels = y_train
                if m == tx.CONTROL:
                    labels = np.random.default_rng([seed, k]).permutation(y_train)
                out[m][test_idx] = fit_variant_model(
                    m, train, test, labels, variant.features, seed, params
                )
        for m in models:
            if np.isnan(out[m]).any():
                raise RuntimeError(f"{variant.name}/{m}: a walk was not predicted in seed {seed}")
            proba[m].append(out[m])
    return VariantResult(variant, sub, use, proba, problems)


def reproduction_gap(
    reference: VariantResult, oof_proba: Mapping[str, Sequence[np.ndarray]]
) -> tuple[float, int]:
    """Compare the recomputed all-input predictions with the GAIT-19 file.

    Returns the largest probability difference and the number of (model, seed) arrays that differ
    by more than ``REPRODUCTION_TOLERANCE``.
    """
    gap, n_differ = 0.0, 0
    for m in ("xgboost", "logistic"):
        if m not in reference.proba:
            continue
        mine, theirs = reference.proba[m], oof_proba[m]
        if len(mine) != len(theirs):
            raise ValueError(f"{m}: {len(mine)} seeds recomputed, {len(theirs)} in the file")
        for a, b in zip(mine, theirs, strict=True):
            d = float(np.abs(a - b).max())
            gap = max(gap, d)
            n_differ += int(d > REPRODUCTION_TOLERANCE)
    return gap, n_differ


# --------------------------------------------------------------------------- scoring
def score_models(
    y: np.ndarray,
    subjects: np.ndarray,
    models: Mapping[str, Sequence[np.ndarray]],
    n_boot: int,
    boot_seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Headline scores with subject-bootstrap intervals, one row per model, and the draws.

    All models share the same bootstrap draws, so differences can be taken between them.
    """
    weight = gm.subject_weights(subjects)
    draws = gm.bootstrap_draws(y, models, subjects, n_boot=n_boot, seed=boot_seed)
    rows = []
    for m, probas in models.items():
        point = gm.metric_values(y, probas, weight)
        lo, hi = gm.percentile_interval(draws[m])
        row: dict[str, Any] = {"model": m}
        for i, name in enumerate(gm.METRICS):
            row[name] = point[name]
            row[f"{name}_lo"], row[f"{name}_hi"] = float(lo[i]), float(hi[i])
        rows.append(row)
    return pd.DataFrame(rows), draws


def difference_table(
    scores: pd.DataFrame,
    draws: Mapping[str, np.ndarray],
    pairs: Sequence[tuple[str, str]],
    level: float = 0.95,
) -> pd.DataFrame:
    """Paired differences ``model - versus`` in the five scores, with intervals."""
    by_model = scores.set_index("model")
    rows = []
    for model, versus in pairs:
        if model not in draws or versus not in draws:
            continue
        lo, hi = gm.percentile_interval(draws[model] - draws[versus], level)
        for i, name in enumerate(gm.METRICS):
            rows.append(
                {
                    "model": model,
                    "versus": versus,
                    "metric": name,
                    "diff": float(by_model.loc[model, name] - by_model.loc[versus, name]),
                    "lo": float(lo[i]),
                    "hi": float(hi[i]),
                }
            )
    return pd.DataFrame(rows, columns=["model", "versus", "metric", "diff", "lo", "hi"])


def class_counts(part: pd.DataFrame) -> dict[str, int]:
    counts = {"n_walks": len(part), "n_subjects": int(part["subject_key"].nunique())}
    for c in gm.CLASSES:
        counts[f"n_class{c}"] = int((part["severity"] == c).sum())
    return counts


def cohort_analysis(
    table: pd.DataFrame,
    models: Mapping[str, Sequence[np.ndarray]],
    n_boot: int,
    boot_seed: int = 0,
    pairs: Sequence[tuple[str, str]] = COHORT_COMPARISONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scores inside every cohort (and pooled) from predictions made for the whole table."""
    all_scores, all_diffs = [], []
    for name, mask in cohort_groups(table):
        part = table[mask]
        y = part["severity"].to_numpy(dtype=int)
        subjects = part["subject_key"].to_numpy()
        sub_models = {m: [p[mask] for p in probas] for m, probas in models.items()}
        scores, draws = score_models(y, subjects, sub_models, n_boot, boot_seed)
        present = int((np.bincount(y, minlength=len(gm.CLASSES)) > 0).sum())
        scores.insert(0, "cohort", name)
        scores.insert(
            2,
            "setting",
            [
                "leave_one_cohort_out" if m.startswith(LOCO_PREFIX) else "cv"
                for m in scores["model"]
            ],
        )
        for key, value in class_counts(part).items():
            scores[key] = value
        scores["chance_bal_acc"] = 1 / present
        all_scores.append(scores)
        diffs = difference_table(scores, draws, pairs)
        diffs.insert(0, "cohort", name)
        all_diffs.append(diffs)
    return pd.concat(all_scores, ignore_index=True), pd.concat(all_diffs, ignore_index=True)


def ablation_analysis(
    results: Mapping[str, VariantResult], n_boot: int, boot_seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scores of every variant and its paired difference from the all-input reference.

    Variants on the same walks share one bootstrap. A variant that removes a cohort is compared
    with the all-input predictions restricted to the walks that are left.
    """
    reference = results[REFERENCE]
    by_rows: dict[str | None, list[VariantResult]] = {}
    for r in results.values():
        by_rows.setdefault(r.variant.drop_cohort, []).append(r)
    all_scores, all_diffs = [], []
    for drop, group in by_rows.items():
        sub = group[0].table
        if drop is None:
            ref_models = reference.proba
        else:
            mask = reference.table["walk"].isin(set(sub["walk"])).to_numpy()
            if not np.array_equal(reference.table["walk"].to_numpy()[mask], sub["walk"].to_numpy()):
                raise ValueError("the reference and the reduced table differ in walk order")
            ref_models = {m: [p[mask] for p in ps] for m, ps in reference.proba.items()}
        models: dict[str, Sequence[np.ndarray]] = {
            f"{REFERENCE}/{m}": ps for m, ps in ref_models.items() if m != tx.CONTROL
        }
        for r in group:
            for m, ps in r.proba.items():
                models[f"{r.variant.name}/{m}"] = ps
        y = sub["severity"].to_numpy(dtype=int)
        scores, draws = score_models(y, sub["subject_key"].to_numpy(), models, n_boot, boot_seed)
        pairs = [
            (f"{r.variant.name}/{m}", f"{REFERENCE}/{m}")
            for r in group
            for m in ("xgboost", "logistic")
            if m in r.proba
        ]
        diffs = difference_table(scores, draws, pairs)
        scores[["variant", "model"]] = scores["model"].str.split("/", n=1, expand=True)
        diffs[["variant", "model"]] = diffs["model"].str.split("/", n=1, expand=True)
        diffs["versus"] = REFERENCE
        meta = {
            r.variant.name: {
                "description": r.variant.description,
                "n_features": len(r.variant.features),
                "fold_warnings": len(r.fold_problems),
            }
            for r in group
        }
        counts = class_counts(sub)
        scores["description"] = [
            meta[v]["description"] if v in meta else f"{REFERENCE}, scored on the same walks"
            for v in scores["variant"]
        ]
        scores["n_features"] = [
            meta[v]["n_features"] if v in meta else len(FEATURE_NAMES) for v in scores["variant"]
        ]
        scores["fold_warnings"] = [
            meta[v]["fold_warnings"] if v in meta else 0 for v in scores["variant"]
        ]
        for key, value in counts.items():
            scores[key] = value
        if drop is not None:
            same = scores["variant"] == REFERENCE
            scores.loc[same, "variant"] = f"{REFERENCE}_on_{group[0].variant.name}"
        all_scores.append(scores)
        all_diffs.append(diffs)
    scores = pd.concat(all_scores, ignore_index=True)
    first = [
        "variant",
        "description",
        "model",
        "n_features",
        "n_walks",
        "n_subjects",
        "fold_warnings",
    ]
    scores = scores[[*first, *[c for c in scores.columns if c not in first]]]
    diffs = pd.concat(all_diffs, ignore_index=True)
    diffs = diffs[["variant", "model", "versus", "metric", "diff", "lo", "hi"]]
    return scores, diffs


# --------------------------------------------------------------------------- gates
def control_problems(label: str, bal_acc: float, auc: float) -> list[str]:
    """Reasons the shuffled-label control of ``label`` did not land at chance."""
    frame = pd.DataFrame([{"model": tx.CONTROL, "bal_acc": bal_acc, "auc": auc}])
    return [f"{label}: {p}" for p in tx.permutation_problems(frame)]


def all_control_problems(ablation_scores: pd.DataFrame) -> list[str]:
    """Cross-validation controls that did not land at chance (every ablation, with all inputs too)."""
    problems: list[str] = []
    controls = ablation_scores[ablation_scores["model"] == tx.CONTROL]
    for _, r in controls.iterrows():
        problems += control_problems(str(r["variant"]), r["bal_acc"], r["auc"])
    return problems


# --------------------------------------------------------------------------- report
def _ci(r: Mapping[str, Any], name: str) -> str:
    return f"{r[name]:.3f} [{r[f'{name}_lo']:.3f}, {r[f'{name}_hi']:.3f}]"


def _delta(diff: float, lo: float, hi: float) -> str:
    mark = " *" if (lo > 0 or hi < 0) else ""
    return f"{diff:+.3f} [{lo:+.3f}, {hi:+.3f}]{mark}"


def report_cohorts(scores: pd.DataFrame, diffs: pd.DataFrame) -> None:
    print("SCORES BY COHORT (out-of-fold predictions pooled over all folds and seeds)")
    print("cv = GAIT-19 cross-validation; loco_* = trained on the other three cohorts.\n")
    for cohort in [*[c for c in scores["cohort"].unique() if c != POOLED], POOLED]:
        part = scores[scores["cohort"] == cohort]
        first = part.iloc[0]
        title = "All four cohorts pooled" if cohort == POOLED else cohort
        print(
            f"{title}: {first['n_walks']:,} walks, {first['n_subjects']} subjects; walks per class "
            f"{first['n_class0']} / {first['n_class1']} / {first['n_class2']}; "
            f"chance balanced accuracy {first['chance_bal_acc']:.3f}"
        )
        print(f"  {'model':<26}{'balanced accuracy':>24}{'AUC':>24}{'bal.acc subject-wtd':>24}")
        for _, r in part.iterrows():
            print(
                f"  {r['model']:<26}{_ci(r, 'bal_acc'):>24}{_ci(r, 'auc'):>24}"
                f"{_ci(r, 'bal_acc_subject'):>24}"
            )
        d = diffs[(diffs["cohort"] == cohort) & (diffs["metric"] == "bal_acc")]
        da = diffs[(diffs["cohort"] == cohort) & (diffs["metric"] == "auc")].set_index(
            ["model", "versus"]
        )
        for _, r in d.iterrows():
            a = da.loc[(r["model"], r["versus"])]
            print(
                f"    {r['model']} - {r['versus']}: balanced accuracy "
                f"{_delta(r['diff'], r['lo'], r['hi'])}; AUC {_delta(a['diff'], a['lo'], a['hi'])}"
            )
        print()
    print("* = the 95% interval of the paired difference excludes 0.")
    print(
        "cohort_only can fall below chance (BMCLab): inside a cohort whose two commonest classes are\n"
        "nearly tied, the class it picks flips with the training subjects. 'majority' is the chance\n"
        "line.\n"
    )


def report_loco_null(null: pd.DataFrame) -> None:
    print("LEAVE ONE COHORT OUT AGAINST SHUFFLED LABELS")
    n = int(null["n_shuffles"].iloc[0])
    print(
        f"Each shuffled-label model is XGBoost trained on the other three cohorts with its labels\n"
        f"shuffled; {n} of them. 'share' = share of shuffled models that score at least as well as\n"
        f"the real one (smallest possible {1 / (n + 1):.3f}). On an unseen cohort chance is not 1/3\n"
        "and 0.5, so the shuffled models, not those numbers, are the yardstick.\n"
    )
    print(
        f"{'cohort':<10}{'score':<20}{'real':>7}{'shuffled mean':>15}{'sd':>7}"
        f"{'95th pct':>10}{'share':>8}"
    )
    for _, r in null[null["metric"].isin(["bal_acc", "auc"])].iterrows():
        name = "balanced accuracy" if r["metric"] == "bal_acc" else "AUC"
        print(
            f"{r['cohort']:<10}{name:<20}{r['real']:>7.3f}{r['null_mean']:>15.3f}"
            f"{r['null_sd']:>7.3f}{r['null_p95']:>10.3f}{r['p_value']:>8.3f}"
        )
    print()


def report_shares(shares: pd.DataFrame) -> None:
    print("CLASS SHARES AND RECALL (% of the cohort's walks, in the order class 0 / 1 / 2-3)")
    print("'true' = share of the cohort's walks in each class; 'predicts' = share a model puts in")
    print("each class; 'finds' = share of each class it gets right (recall).\n")
    for cohort in shares["cohort"].unique():
        part = shares[shares["cohort"] == cohort]
        first = part[part["model"] == part["model"].iloc[0]]
        print(f"{cohort:<10} true {_triple(first['true_share'])}")
        for m in part["model"].unique():
            row = part[part["model"] == m]
            print(
                f"{'':<10} {m:<14} predicts {_triple(row['predicted_share'])}"
                f"   finds {_triple(row['recall'])}"
            )
    print()


def _triple(values: pd.Series) -> str:
    return " / ".join("  -" if np.isnan(v) else f"{v:>3.0%}" for v in values)


def report_ablations(
    scores: pd.DataFrame, diffs: pd.DataFrame, gap: float, n_differ: int, n_total: int
) -> None:
    print("ABLATIONS (same GAIT-18 folds, same XGBoost settings; difference from all inputs)")
    if n_differ == 0:
        verdict = "reproduces the GAIT-19 file"
    else:
        verdict = (
            "differs from the file: another machine or other software versions; the ablations "
            "are compared with the recomputed reference, so their differences are not affected"
        )
    print(
        "All-input predictions recomputed here against the GAIT-19 file: largest difference "
        f"{gap:.1e}, {n_differ} of {n_total} (model, seed) arrays differ ({verdict}).\n"
    )
    d = diffs.set_index(["variant", "model", "metric"])
    for variant in scores["variant"].unique():
        part = scores[scores["variant"] == variant].set_index("model")
        first = part.iloc[0]
        warn = f"; {first['fold_warnings']} fold warning(s)" if first["fold_warnings"] else ""
        print(
            f"{variant}: {first['description']} ({first['n_features']} inputs, "
            f"{first['n_walks']:,} walks, {first['n_subjects']} subjects{warn})"
        )
        for m in ("xgboost", "logistic", tx.CONTROL):
            if m not in part.index:
                continue
            r = part.loc[m]
            line = f"  {m:<18} balanced accuracy {_ci(r, 'bal_acc')}  AUC {_ci(r, 'auc')}"
            if (variant, m, "bal_acc") in d.index:
                b, a = d.loc[(variant, m, "bal_acc")], d.loc[(variant, m, "auc")]
                line += (
                    f"\n  {'':<18} vs all inputs: balanced accuracy "
                    f"{_delta(b['diff'], b['lo'], b['hi'])}; AUC {_delta(a['diff'], a['lo'], a['hi'])}"
                )
            print(line)
        print()
    print("* = the 95% interval of the paired difference excludes 0.\n")


def file_sha(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:8]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scores by cohort, LOCO and ablations (GAIT-20)")
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap", type=int, default=tx.N_BOOT, help="bootstrap draws")
    parser.add_argument("--shuffles", type=int, default=N_SHUFFLES, help="shuffled-label models")
    parser.add_argument("--no-ablations", action="store_true", help="skip the ablations")
    args = parser.parse_args(argv)

    for path, hint in (
        (args.table, "build_feature_table.py"),
        (args.folds, "make_folds.py"),
        (args.oof, "train_xgboost.py"),
    ):
        if not path.is_file():
            print(f"MISSING: {path}\nRun {hint} first.")
            return 2
    started = time.time()
    table = tx.read_training_table(args.table)
    assignments = tx.read_assignments(args.folds)
    problems = fold_problems(table, assignments)
    if problems:
        for p in problems:
            print(f"  -> {p}")
        print("The folds are not valid for this table. Rebuild them with make_folds.py.")
        return 1
    oof = read_oof(args.oof)
    cv_proba = {m: oof_probabilities(oof, table, m) for m in tx.MODELS}
    print(
        f"{len(table)} walks, {table['subject_key'].nunique()} subjects; folds "
        f"{fingerprint(assignments)}; table {tx.table_fingerprint(args.table)}; "
        f"predictions {file_sha(args.oof)}; xgboost {xgboost.__version__}.\n"
    )

    loco = leave_one_cohort_out(table, n_shuffles=args.shuffles)
    print(f"Leave-one-cohort-out fitted ({time.time() - started:.0f} s).\n")
    models: dict[str, Sequence[np.ndarray]] = dict(cv_proba)
    for m, arrays in loco.proba.items():
        models[f"{LOCO_PREFIX}{m}"] = arrays
    cohort_scores, cohort_diffs = cohort_analysis(table, models, args.bootstrap)
    null = loco_null(table, loco)
    shares = class_shares(
        table,
        {m: models[m] for m in ("xgboost", f"{LOCO_PREFIX}xgboost", f"{LOCO_PREFIX}logistic")},
    )
    report_cohorts(cohort_scores, cohort_diffs)
    report_loco_null(null)
    report_shares(shares)

    ablation_scores = pd.DataFrame()
    ablation_diffs = pd.DataFrame()
    gap, n_differ, n_total = float("nan"), 0, 0
    failed: list[str] = []
    if not args.no_ablations:
        results = {}
        for v in VARIANTS:
            results[v.name] = run_variant(table, assignments, v)
            print(f"  ablation {v.name} done ({time.time() - started:.0f} s)")
        print()
        gap, n_differ = reproduction_gap(results[REFERENCE], cv_proba)
        n_total = sum(len(cv_proba[m]) for m in ("xgboost", "logistic"))
        ablation_scores, ablation_diffs = ablation_analysis(results, args.bootstrap)
        report_ablations(ablation_scores, ablation_diffs, gap, n_differ, n_total)
        failed = all_control_problems(ablation_scores)
        print(
            "Label-permutation controls of the cross-validated models (all inputs and every "
            "ablation): " + ("FAILED" if failed else "all at chance, as they should be.")
        )
        for p in failed:
            print(f"  -> {p}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cohort_scores.to_csv(args.out_dir / "cohort_scores.csv", index=False, float_format="%.6f")
    cohort_diffs.to_csv(args.out_dir / "cohort_differences.csv", index=False, float_format="%.6f")
    shares.to_csv(args.out_dir / "cohort_class_shares.csv", index=False, float_format="%.6f")
    null.to_csv(args.out_dir / "loco_null.csv", index=False, float_format="%.6f")
    loco_frame(table, loco).to_csv(
        args.out_dir / "loco_predictions.csv", index=False, float_format="%.6f"
    )
    if not args.no_ablations:
        ablation_scores.to_csv(args.out_dir / "ablations.csv", index=False, float_format="%.6f")
        ablation_diffs.to_csv(
            args.out_dir / "ablation_differences.csv", index=False, float_format="%.6f"
        )
    config = {
        "xgboost_params": tx.XGB_PARAMS,
        "xgboost_version": xgboost.__version__,
        "bootstrap_draws": args.bootstrap,
        "loco_shuffles": args.shuffles,
        "folds_fingerprint": fingerprint(assignments),
        "feature_table": tx.table_fingerprint(args.table),
        "predictions_file": file_sha(args.oof),
        "variants": {v.name: v.description for v in VARIANTS},
        "loco_train_cohorts": {k: list(v) for k, v in loco.train_cohorts.items()},
        "reproduction_gap": None if np.isnan(gap) else gap,
        "reproduction_arrays_differing": n_differ,
        "control_problems": failed,
        "classes": {str(c): CLASS_LABELS[c] for c in gm.CLASSES},
    }
    (args.out_dir / "results.json").write_text(json.dumps(config, indent=2))
    print(f"\nWrote {args.out_dir} in {time.time() - started:.0f} s.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
