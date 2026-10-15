"""GAIT-19: train and score the gait severity classifier on the GAIT-18 folds.

Run from the repository root (venv active, after ``build_feature_table.py`` and
``make_folds.py``)::

    python models/gait/src/train_xgboost.py

What it does, for every one of the 25 test folds (5 folds x 5 seeds):

1. checks that no subject is in both train and test (``fold_problems`` once, and again for the
   split being fitted),
2. calls the feature contract (CLAUDE.md 3.1) on the model inputs before every ``fit()``,
3. fits the baseline ladder and XGBoost on the training subjects and predicts the test subjects.

It then scores the pooled out-of-fold predictions, runs a label-permutation control, explains the
model with SHAP and saves a final model for the demo.

The model grades gait impairment (UPDRS gait 0 / 1 / 2-3) inside four labelled clinical cohorts.
It does not separate people with PD from people without. Class 0 reads "no gait impairment
pattern detected in this walk", never "healthy" and never "no PD".

Decisions:

1. *Fixed settings, no tuning.* 150 trees, depth 3, learning rate 0.1, softmax over 3 classes.
   These are the settings of the GAIT-18 split test. With 109 subjects there is no test set left
   over for tuning; any later tuning has to happen inside the training part of each fold.
2. *Baseline ladder, on the same folds.* Majority class, cohort only, best single feature (a depth-2
   tree on the feature with the highest training rank correlation with the class), logistic
   regression, XGBoost. XGBoost is only worth reporting to the extent that it beats the rungs
   below it. Cohort only matters most: the eight features identify the cohort with 0.72 balanced
   accuracy, so the model could be recognising the cohort instead of the gait.
3. *Label-permutation control.* The same XGBoost is trained on shuffled training labels and scored
   against the real test labels. It has to land at chance (balanced accuracy 1/3, AUC 0.5, within
   ``CHANCE_MARGIN``); otherwise the run exits with code 1 because something leaks.
4. *Headline metrics (CLAUDE.md 3.4).* Balanced accuracy and one-vs-rest AUC with a bootstrap 95% CI
   resampled by subject (``gait_metrics.py``). Macro-F1 and the confusion matrix are secondary.
   Walk-level and subject-weighted scores are both reported. The spread over the 25 test folds is
   printed as well, because the folds hold only 12-33 subjects.
5. *Per-cohort scores are not made here.* The out-of-fold predictions are saved with the cohort
   column so GAIT-20 can pool them per cohort and run leave-one-cohort-out and the ablations.
6. *Training weights are off by default* (``--weighted`` turns on 1 / walks-per-subject for every
   rung that fits). Scores always report both walk-level and subject-weighted results.
7. *SHAP* is computed on the out-of-fold models of seed 0 (``TreeExplainer``, exact for trees),
   so every walk is explained by a model that never saw its subject. Values are in log-odds.
8. *The final model* is fitted on all subjects, for the demo. Its own score is never a result:
   it has seen every subject.

Writes to ``artifacts/models/xgboost/`` (git-ignored): ``oof_predictions.csv``, ``metrics.csv``,
``metrics.json``, ``shap_mean_abs.csv``, ``final_model.json``, ``final_model_meta.json``.
Exit code 0 = ok, 1 = a fold problem or the permutation control failed, 2 = an input is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gait_metrics as gm
import numpy as np
import pandas as pd
import xgboost
from feature_contract import check_feature_contract
from gait_extract import FEATURE_NAMES
from make_folds import (
    CLASS_LABELS,
    check_table,
    fingerprint,
    fold_problems,
    walk_folds,
    walk_weights,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

GAIT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = GAIT_DIR / "artifacts" / "features" / "feature_table.csv"
DEFAULT_FOLDS = GAIT_DIR / "artifacts" / "folds" / "fold_assignments.csv"
DEFAULT_OUT = GAIT_DIR / "artifacts" / "models" / "xgboost"

LABEL_NAME = "UPDRS_GAIT"
TABLE_COLUMNS = ("walk", "cohort", "subject_key", "severity", *FEATURE_NAMES)
XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 150,
    "max_depth": 3,
    "learning_rate": 0.1,
    "objective": "multi:softprob",
    "tree_method": "hist",
    "n_jobs": 1,  # one thread: the same numbers every run
    "verbosity": 0,
}
N_BOOT = 1000
CHANCE_BAL_ACC = 1 / 3
CHANCE_AUC = 0.5
CHANCE_MARGIN = 0.03  # GAIT-18 saw 0.32-0.34 with shuffled labels
LADDER = ("majority", "cohort_only", "best_feature", "logistic", "xgboost")
CONTROL = "xgboost_permuted"
MODELS = (*LADDER, CONTROL)
PROBA_COLUMNS = ("p0", "p1", "p2")


class LeakageError(RuntimeError):
    """A subject is in both the training and the test part of a split."""


# --------------------------------------------------------------------------- inputs
def read_training_table(path: Path) -> pd.DataFrame:
    """The feature table with the columns training needs, in the table's row order."""
    table = pd.read_csv(
        path, usecols=list(TABLE_COLUMNS), dtype={"walk": str, "cohort": str, "subject_key": str}
    )
    table = table[list(TABLE_COLUMNS)]
    check_table(table)
    if table["walk"].duplicated().any():
        raise ValueError("walk ids are not unique")
    if not np.isfinite(table[list(FEATURE_NAMES)].to_numpy(dtype=float)).all():
        raise ValueError("a model input is missing or infinite; rejected walks must not be here")
    return table


def read_assignments(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"subject_key": str})


def _inputs(frame: pd.DataFrame) -> np.ndarray:
    """The model inputs of ``frame``, after the feature contract (call before every fit)."""
    columns = list(FEATURE_NAMES)
    check_feature_contract(columns, label_name=LABEL_NAME)
    return frame[columns].to_numpy(dtype=float)


def _full(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Place the columns of a fitted model into a (rows x 3) array in class order."""
    out = np.zeros((proba.shape[0], len(gm.CLASSES)))
    for column, c in enumerate(classes):
        out[:, int(c)] = proba[:, column]
    return out


# --------------------------------------------------------------------------- the ladder
# A learner gets the training and test rows, the training labels and weights, the seed and the
# XGBoost settings. It returns the test probabilities (rows x 3) and a short note ("" or, for the
# best-single-feature rung, the name of the feature it chose).
Learner = Callable[
    [pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, int, dict[str, Any]],
    tuple[np.ndarray, str],
]


def _prior(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, weights=w, minlength=len(gm.CLASSES))
    return counts / counts.sum()


def fit_majority(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    seed: int,
    params: dict[str, Any],
) -> tuple[np.ndarray, str]:
    return np.tile(_prior(y, w), (len(test), 1)), ""


def fit_cohort_only(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    seed: int,
    params: dict[str, Any],
) -> tuple[np.ndarray, str]:
    check_feature_contract(["cohort"], label_name=LABEL_NAME)
    overall = _prior(y, w)
    cohorts = train["cohort"].to_numpy()
    priors = {c: _prior(y[cohorts == c], w[cohorts == c]) for c in np.unique(cohorts)}
    return np.vstack([priors.get(c, overall) for c in test["cohort"].to_numpy()]), ""


def best_feature_index(x: np.ndarray, y: np.ndarray) -> int:
    """The input with the highest absolute Spearman correlation with the class (training only)."""
    rho = pd.DataFrame(x).corrwith(pd.Series(y), method="spearman").abs().fillna(0.0)
    return int(rho.to_numpy().argmax())


def fit_best_feature(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    seed: int,
    params: dict[str, Any],
) -> tuple[np.ndarray, str]:
    x_train, x_test = _inputs(train), _inputs(test)
    j = best_feature_index(x_train, y)
    tree = DecisionTreeClassifier(max_depth=2, min_samples_leaf=25, random_state=seed)
    tree.fit(x_train[:, [j]], y, sample_weight=w)
    return _full(tree.predict_proba(x_test[:, [j]]), tree.classes_), FEATURE_NAMES[j]


def fit_logistic(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    seed: int,
    params: dict[str, Any],
) -> tuple[np.ndarray, str]:
    x_train, x_test = _inputs(train), _inputs(test)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(x_train, y, logisticregression__sample_weight=w)
    return _full(model.predict_proba(x_test), model.classes_), ""


def fit_xgboost_model(
    x: np.ndarray, y: np.ndarray, w: np.ndarray | None, seed: int, params: dict[str, Any]
) -> XGBClassifier:
    model = XGBClassifier(**params, random_state=seed)
    model.fit(x, y, sample_weight=w)
    return model


def fit_xgboost(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    seed: int,
    params: dict[str, Any],
) -> tuple[np.ndarray, str]:
    x_train, x_test = _inputs(train), _inputs(test)
    model = fit_xgboost_model(x_train, y, w, seed, params)
    return _full(model.predict_proba(x_test), model.classes_), ""


LEARNERS: dict[str, Learner] = {
    "majority": fit_majority,
    "cohort_only": fit_cohort_only,
    "best_feature": fit_best_feature,
    "logistic": fit_logistic,
    "xgboost": fit_xgboost,
    CONTROL: fit_xgboost,  # same learner; the training labels are shuffled
}


# --------------------------------------------------------------------------- cross-validation
@dataclass
class CVResult:
    """Out-of-fold probabilities (one walks x 3 array per seed) and fold scores per model."""

    seeds: tuple[int, ...]
    proba: dict[str, list[np.ndarray]] = field(default_factory=dict)
    fold_bal_acc: dict[str, list[float]] = field(default_factory=dict)
    best_features: Counter[str] = field(default_factory=Counter)


def assert_no_leakage(train: pd.DataFrame, test: pd.DataFrame) -> None:
    shared = set(train["subject_key"]) & set(test["subject_key"])
    if shared:
        raise LeakageError(f"{len(shared)} subject(s) are in both the training and the test part")


def cross_validate(
    table: pd.DataFrame,
    assignments: pd.DataFrame,
    models: Sequence[str] = MODELS,
    seeds: Sequence[int] | None = None,
    weighted: bool = False,
    params: dict[str, Any] | None = None,
    test_table: pd.DataFrame | None = None,
) -> CVResult:
    """Fit every model in ``models`` on each training split and predict its test split.

    ``test_table`` (GAIT-20, noise check) replaces the inputs of the *test* rows only: same rows in
    the same order as ``table``, so every model is still fitted on the clean training rows.
    """
    unknown = [m for m in models if m not in LEARNERS]
    if unknown:
        raise ValueError(f"unknown models {unknown}")
    if test_table is not None and not (
        len(test_table) == len(table)
        and (test_table["walk"].to_numpy() == table["walk"].to_numpy()).all()
        and (test_table["subject_key"].to_numpy() == table["subject_key"].to_numpy()).all()
    ):
        raise ValueError("test_table must hold the same walks in the same order as table")
    params = XGB_PARAMS if params is None else params
    all_seeds = tuple(sorted(int(s) for s in set(assignments["seed"])))
    seeds = all_seeds if seeds is None else tuple(seeds)
    y_all = table["severity"].to_numpy(dtype=int)
    w_all = walk_weights(table) if weighted else np.ones(len(table))
    result = CVResult(seeds=seeds)
    for m in models:
        result.proba[m] = []
        result.fold_bal_acc[m] = []
    chosen: Counter[str] = Counter()
    for seed in seeds:
        folds = walk_folds(table, assignments, seed)
        out = {m: np.full((len(table), len(gm.CLASSES)), np.nan) for m in models}
        for k in sorted(set(folds)):
            train_idx = np.flatnonzero(folds != k)
            test_idx = np.flatnonzero(folds == k)
            train = table.iloc[train_idx]
            test = (table if test_table is None else test_table).iloc[test_idx]
            assert_no_leakage(train, test)
            y_train = y_all[train_idx]
            if set(y_train) != set(gm.CLASSES):
                raise ValueError(f"seed {seed} fold {k}: the training part lacks a class")
            for m in models:
                labels = y_train
                if m == CONTROL:
                    labels = np.random.default_rng([seed, k]).permutation(y_train)
                proba, note = LEARNERS[m](train, test, labels, w_all[train_idx], seed, params)
                if note:
                    chosen[note] += 1
                out[m][test_idx] = proba
                result.fold_bal_acc[m].append(
                    gm.balanced_accuracy(y_all[test_idx], proba.argmax(axis=1))
                )
        for m in models:
            if np.isnan(out[m]).any():
                raise RuntimeError(f"{m}: a walk was not predicted in seed {seed}")
            result.proba[m].append(out[m])
    result.best_features = chosen
    return result


# --------------------------------------------------------------------------- scores
def summarize(
    table: pd.DataFrame, result: CVResult, n_boot: int = N_BOOT, boot_seed: int = 0
) -> pd.DataFrame:
    """One row per model: headline scores with subject-bootstrap CIs, and the fold spread."""
    y = table["severity"].to_numpy(dtype=int)
    subjects = table["subject_key"].to_numpy()
    weight = gm.subject_weights(subjects)
    rows = []
    for m, probas in result.proba.items():
        point = gm.metric_values(y, probas, weight)
        ci = gm.bootstrap_intervals(y, probas, subjects, n_boot=n_boot, seed=boot_seed)
        per_seed = [gm.balanced_accuracy(y, p.argmax(axis=1)) for p in probas]
        folds = np.array(result.fold_bal_acc[m])
        row: dict[str, Any] = {"model": m}
        for name in gm.METRICS:
            row[name] = point[name]
            row[f"{name}_lo"], row[f"{name}_hi"] = ci[name]
        row["bal_acc_seed_sd"] = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0
        row["fold_mean"] = float(folds.mean())
        row["fold_sd"] = float(folds.std(ddof=1)) if len(folds) > 1 else 0.0
        row["fold_min"], row["fold_max"] = float(folds.min()), float(folds.max())
        rows.append(row)
    return pd.DataFrame(rows)


def permutation_problems(scores: pd.DataFrame) -> list[str]:
    """Reasons the label-permutation control failed (empty list = it landed at chance)."""
    row = scores.set_index("model").loc[CONTROL]
    problems = []
    if row["bal_acc"] > CHANCE_BAL_ACC + CHANCE_MARGIN:
        problems.append(
            f"shuffled labels give balanced accuracy {row['bal_acc']:.3f} "
            f"(chance {CHANCE_BAL_ACC:.3f} + {CHANCE_MARGIN})"
        )
    if row["auc"] > CHANCE_AUC + CHANCE_MARGIN:
        problems.append(
            f"shuffled labels give AUC {row['auc']:.3f} (chance {CHANCE_AUC} + {CHANCE_MARGIN})"
        )
    return problems


def oof_frame(table: pd.DataFrame, result: CVResult) -> pd.DataFrame:
    """Long table of out-of-fold probabilities: one row per walk, seed and model."""
    base = table[["walk", "cohort", "subject_key", "severity"]]
    parts = []
    for m, probas in result.proba.items():
        for seed, proba in zip(result.seeds, probas, strict=True):
            part = base.copy()
            part.insert(0, "model", m)
            part.insert(1, "seed", seed)
            for i, name in enumerate(PROBA_COLUMNS):
                part[name] = proba[:, i]
            part["pred"] = proba.argmax(axis=1)
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- SHAP and final model
def shap_array(values: Any) -> np.ndarray:
    """SHAP values as (walks, features, classes), whichever layout shap returns."""
    if isinstance(values, list):
        return np.stack([np.asarray(v) for v in values], axis=-1)
    return np.asarray(values)


def explain_folds(
    table: pd.DataFrame,
    assignments: pd.DataFrame,
    seed: int,
    params: dict[str, Any] | None = None,
    weighted: bool = False,
    reference: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    """Out-of-fold SHAP values (walks, features, classes) of one seed, in log-odds.

    Returns the values, the largest additivity error (SHAP plus base value against the model's
    raw output) and the largest difference from ``reference`` (the scored probabilities of the
    same seed), which should be 0 because the model is refitted with the same settings.
    """
    import shap

    params = XGB_PARAMS if params is None else params
    folds = walk_folds(table, assignments, seed)
    y_all = table["severity"].to_numpy(dtype=int)
    w_all = walk_weights(table) if weighted else np.ones(len(table))
    values = np.zeros((len(table), len(FEATURE_NAMES), len(gm.CLASSES)))
    additivity = 0.0
    drift = 0.0
    for k in sorted(set(folds)):
        train_idx, test_idx = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        assert_no_leakage(table.iloc[train_idx], table.iloc[test_idx])
        x_train, x_test = _inputs(table.iloc[train_idx]), _inputs(table.iloc[test_idx])
        model = fit_xgboost_model(x_train, y_all[train_idx], w_all[train_idx], seed, params)
        explainer = shap.TreeExplainer(model)
        sv = shap_array(explainer.shap_values(x_test))
        values[test_idx] = sv
        margin = model.predict(x_test, output_margin=True)
        base = np.asarray(explainer.expected_value).reshape(-1)
        additivity = max(additivity, float(np.abs(sv.sum(axis=1) + base - margin).max()))
        if reference is not None:
            drift = max(
                drift, float(np.abs(model.predict_proba(x_test) - reference[test_idx]).max())
            )
    return values, additivity, drift


def shap_mean_abs(values: np.ndarray) -> pd.DataFrame:
    """Mean |SHAP| per feature and class, plus the mean over classes, largest first."""
    mean_abs = np.abs(values).mean(axis=0)  # features x classes
    out = pd.DataFrame(
        mean_abs, index=list(FEATURE_NAMES), columns=[CLASS_LABELS[c] for c in gm.CLASSES]
    )
    out["overall"] = out.mean(axis=1)
    return out.sort_values("overall", ascending=False).rename_axis("feature").reset_index()


def fit_final_model(
    table: pd.DataFrame, seed: int = 0, params: dict[str, Any] | None = None, weighted: bool = False
) -> XGBClassifier:
    """XGBoost fitted on every subject, for the demo. It has seen all of them: never score it."""
    params = XGB_PARAMS if params is None else params
    w = walk_weights(table) if weighted else np.ones(len(table))
    return fit_xgboost_model(_inputs(table), table["severity"].to_numpy(dtype=int), w, seed, params)


def table_fingerprint(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:8]


# --------------------------------------------------------------------------- report
def _ci(value: float, lo: float, hi: float) -> str:
    return f"{value:.3f} [{lo:.3f}, {hi:.3f}]"


def report(table: pd.DataFrame, result: CVResult, scores: pd.DataFrame, n_boot: int) -> None:
    n_sub = table["subject_key"].nunique()
    n_fold = len(result.fold_bal_acc["xgboost"])
    print(
        f"{len(table)} walks, {n_sub} subjects, {len(result.seeds)} seeds, {n_fold} test folds "
        "per model. Scores: pooled out-of-fold predictions per seed, averaged over seeds;\n"
        f"95% CI from {n_boot:,}-draw bootstrap of subjects.\n"
    )
    print(
        f"{'model':<18}{'balanced accuracy':>26}{'AUC (one-vs-rest)':>26}"
        f"{'bal.acc subject-wtd':>26}{'macro-F1':>26}"
    )
    for _, r in scores.iterrows():
        print(
            f"{r['model']:<18}{_ci(r['bal_acc'], r['bal_acc_lo'], r['bal_acc_hi']):>26}"
            f"{_ci(r['auc'], r['auc_lo'], r['auc_hi']):>26}"
            f"{_ci(r['bal_acc_subject'], r['bal_acc_subject_lo'], r['bal_acc_subject_hi']):>26}"
            f"{_ci(r['macro_f1'], r['macro_f1_lo'], r['macro_f1_hi']):>26}"
        )
    x = scores.set_index("model").loc["xgboost"]
    print(
        f"\nXGBoost spread over its {n_fold} test folds: balanced accuracy "
        f"{x['fold_mean']:.3f} +/- {x['fold_sd']:.3f} (range {x['fold_min']:.2f}-{x['fold_max']:.2f}); "
        f"sd over seeds {x['bal_acc_seed_sd']:.3f}."
    )
    y = table["severity"].to_numpy(dtype=int)
    cm = np.mean([gm.confusion(y, p.argmax(axis=1)) for p in result.proba["xgboost"]], axis=0)
    print("\nXGBoost confusion matrix (walks per seed; rows = true class, columns = predicted):")
    print(f"{'':<12}" + "".join(f"{CLASS_LABELS[c]:>12}" for c in gm.CLASSES) + f"{'recall':>10}")
    for i, c in enumerate(gm.CLASSES):
        print(
            f"{CLASS_LABELS[c]:<12}"
            + "".join(f"{v:>12.0f}" for v in cm[i])
            + f"{cm[i, i] / cm[i].sum():>10.2f}"
        )
    if result.best_features:
        top = ", ".join(f"{n} ({c})" for n, c in result.best_features.most_common())
        print(f"\nBest single feature chosen in training (folds): {top}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and score the gait classifier (GAIT-19)")
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap", type=int, default=N_BOOT, help="bootstrap draws for the CIs")
    parser.add_argument("--weighted", action="store_true", help="train with 1 / walks-per-subject")
    parser.add_argument("--no-shap", action="store_true", help="skip SHAP and the final model")
    args = parser.parse_args(argv)

    for path, hint in ((args.table, "build_feature_table.py"), (args.folds, "make_folds.py")):
        if not path.is_file():
            print(f"MISSING: {path}\nRun {hint} first.")
            return 2
    started = time.time()
    table = read_training_table(args.table)
    assignments = read_assignments(args.folds)

    problems = fold_problems(table, assignments)
    if problems:
        for p in problems:
            print(f"  -> {p}")
        print("The folds are not valid for this table. Rebuild them with make_folds.py.")
        return 1
    print(f"Folds: fingerprint {fingerprint(assignments)}; table {table_fingerprint(args.table)}.")

    result = cross_validate(table, assignments, weighted=args.weighted)
    scores = summarize(table, result, n_boot=args.bootstrap)
    report(table, result, scores, args.bootstrap)

    failed = permutation_problems(scores)
    perm = scores.set_index("model").loc[CONTROL]
    print(
        f"\nLabel-permutation control: balanced accuracy {perm['bal_acc']:.3f} "
        f"(chance {CHANCE_BAL_ACC:.3f}), AUC {perm['auc']:.3f} (chance {CHANCE_AUC}): "
        + ("FAILED" if failed else "at chance, as it should be.")
    )
    for p in failed:
        print(f"  -> {p}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    oof_frame(table, result).to_csv(
        args.out_dir / "oof_predictions.csv", index=False, float_format="%.6f"
    )
    scores.to_csv(args.out_dir / "metrics.csv", index=False, float_format="%.6f")
    config = {
        "xgboost_params": XGB_PARAMS,
        "weighted_training": args.weighted,
        "bootstrap_draws": args.bootstrap,
        "folds_fingerprint": fingerprint(assignments),
        "feature_table": table_fingerprint(args.table),
        "xgboost_version": xgboost.__version__,
    }
    (args.out_dir / "metrics.json").write_text(
        json.dumps({"config": config, "scores": scores.to_dict(orient="records")}, indent=2)
    )

    if not args.no_shap:
        reference = result.proba["xgboost"][0] if result.seeds[0] == min(result.seeds) else None
        values, additivity, drift = explain_folds(
            table, assignments, result.seeds[0], weighted=args.weighted, reference=reference
        )
        mean_abs = shap_mean_abs(values)
        mean_abs.to_csv(args.out_dir / "shap_mean_abs.csv", index=False, float_format="%.6f")
        print(
            f"\nSHAP (out-of-fold, seed {result.seeds[0]}, log-odds; mean |value| per class): "
            f"additivity error {additivity:.1e}, refit difference {drift:.1e}."
        )
        print(
            f"{'feature':<26}"
            + "".join(f"{CLASS_LABELS[c]:>12}" for c in gm.CLASSES)
            + f"{'overall':>10}"
        )
        for _, r in mean_abs.iterrows():
            print(
                f"{r['feature']:<26}"
                + "".join(f"{r[CLASS_LABELS[c]]:>12.3f}" for c in gm.CLASSES)
                + f"{r['overall']:>10.3f}"
            )
        final = fit_final_model(table, weighted=args.weighted)
        final.save_model(args.out_dir / "final_model.json")
        (args.out_dir / "final_model_meta.json").write_text(
            json.dumps(
                {
                    **config,
                    "features": list(FEATURE_NAMES),
                    "classes": {str(c): CLASS_LABELS[c] for c in gm.CLASSES},
                    "trained_on": f"all {table['subject_key'].nunique()} subjects, {len(table)} walks",
                    "note": "fitted on every subject for the demo; its score is never a result",
                },
                indent=2,
            )
        )
        print(f"Final model (all subjects, for the demo): {args.out_dir / 'final_model.json'}")

    print(f"\nWrote {args.out_dir} in {time.time() - started:.0f} s.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
