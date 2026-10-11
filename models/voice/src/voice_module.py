"""Voice module: train the demo model and score one person (task VOICE-18).

``train_voice_model`` builds everything ``predict_voice`` needs, on the D3 split:

1. **Hold-out (D3).** 20% of people, stratified by class and sex, seed 42, are
   set aside once and never touched in training. They are used only for the
   model card's hold-out check and for the demo.
2. **Model.** On the other 80%: scaler + in-fold selector + model, with k and C
   (or leaves) tuned by person-level CV, then refitted on all 80%.
3. **Calibration (Platt).** Out-of-fold scores from a nested CV on the 80% are
   mapped to probabilities by a 1-feature logistic regression on their logit,
   with balanced class weights, so a calibrated 0.5 means the same operating
   point used in every evidence table (RESULTS.md). The scores are calibrated
   to a 50/50 prior, NOT to screening prevalence; PPV at 1/2/5% prevalence is in
   the model card.
4. **CI (bootstrap ensemble).** ``n_ensemble`` copies of the tuned model, each
   fitted on a bootstrap resample of training PEOPLE. A person's CI is the 2.5th
   and 97.5th percentile of the ensemble's calibrated scores. It shows model
   uncertainty from the small training set; it is not a CI on the person.
5. **Explanation (SHAP).** Exact SHAP values in the model's log-odds: for L1
   logistic regression, coefficient x standardised value (the training mean is
   the baseline); for LightGBM, its built-in TreeSHAP (``pred_contrib``). No
   ``shap`` package needed. A person's values are averaged over their recordings.

``predict_voice`` returns the result vector (decision D5). ``triage_priority``
is only ever ``prioritised_referral`` or ``not_prioritised`` (D6). If the input
fails the quality gate the module abstains (Modality-Contract rule 3): no score,
no triage, and ``availability_mask`` marks voice as absent. It never guesses.

Gender is not an input (D7). PP1 has no audio, so the quality gate checks the
feature row (schema, banned columns, finite values), not recording quality.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import contract
import metrics
import numpy as np
import pandas as pd
import pipeline
import sex_fairness
import splitter
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import data

PIPELINE_VERSION = "voice-pp1-0.1.0"
DEFAULT_MODEL: pipeline.Model = "l1_logistic"
N_ENSEMBLE = 50
TOP_K = 5
TRIAGE_THRESHOLD = 0.5
TRIAGE_VALUES = ("prioritised_referral", "not_prioritised")
MAX_RECORDINGS = 3
RESULT_FIELDS = (
    "modality",
    "risk_score",
    "ci_lower",
    "ci_upper",
    "triage_priority",
    "availability_mask",
    "shap_top5",
    "sex_fairness_note",
    "pipeline_version",
    "quality_gate_passed",
    "abstained",
    "problems",
    "n_training",
    "note",
)
NOTE = (
    "Voice screening score for triage only. It can prioritise a referral; "
    "it cannot rule Parkinson's disease in or out."
)
_EPS = 1e-6


@dataclass
class VoiceBundle:
    """Everything needed to score a person. Holds no person ids or data rows."""

    model_name: str
    feature_names: list[str]
    model: Pipeline
    calibrator: LogisticRegression
    ensemble: list[Pipeline]
    best_params: dict[str, float]
    n_training: int
    model_card: dict[str, Any] = field(default_factory=dict)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def fit_platt(
    raw_scores: np.ndarray, y: np.ndarray, seed: int = pipeline.SEED
) -> LogisticRegression:
    """Platt scaling with balanced class weights, on per-person out-of-fold scores."""
    return LogisticRegression(class_weight="balanced", random_state=seed).fit(
        _logit(raw_scores).reshape(-1, 1), y
    )


def calibrate(calibrator: LogisticRegression, raw_scores: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(_logit(raw_scores).reshape(-1, 1))[:, 1]


def _person_raw(model: Pipeline, X: np.ndarray) -> float:
    """D2: a person's score is the mean of their recordings' probabilities."""
    return float(model.predict_proba(X)[:, 1].mean())


def shap_values(model: Pipeline, X: np.ndarray, feature_names: Sequence[str]) -> pd.DataFrame:
    """Per-recording SHAP values (log-odds) for the selected features, plus the baseline."""
    scaled = model["scale"].transform(X)
    selected = model["select"].transform(scaled)
    names = [feature_names[i] for i in model["select"].get_support(indices=True)]
    estimator = model["model"]
    if isinstance(estimator, LogisticRegression):
        # Training mean of a standardised feature is 0, so coef * value is exact SHAP.
        values = selected * estimator.coef_[0]
        base = np.full(len(X), estimator.intercept_[0])
    else:
        contrib = estimator.predict(selected, pred_contrib=True)
        values, base = contrib[:, :-1], contrib[:, -1]
    frame = pd.DataFrame(values, columns=names)
    frame["_baseline"] = base
    return frame


def top_drivers(model: Pipeline, X: np.ndarray, feature_names: Sequence[str]) -> list[dict]:
    """The TOP_K features with the largest mean |SHAP| for this person."""
    mean = shap_values(model, X, feature_names).drop(columns="_baseline").mean()
    top = mean.reindex(mean.abs().sort_values(ascending=False).index)[:TOP_K]
    return [
        {
            "feature": name,
            "shap_log_odds": round(float(v), 4),
            "direction": "towards higher risk" if v > 0 else "towards lower risk",
        }
        for name, v in top.items()
    ]


def quality_gate(rows: pd.DataFrame, feature_names: Sequence[str]) -> list[str]:
    """Problems with the input; an empty list means it is usable."""
    problems = []
    if len(rows) == 0:
        return ["no recordings given"]
    if len(rows) > MAX_RECORDINGS:
        problems.append(f"{len(rows)} recordings given; at most {MAX_RECORDINGS} per person")
    if data.GENDER_COL in rows.columns:
        problems.append("gender must not be a model input (D7)")
    try:
        contract.validate_features(c for c in rows.columns if c != data.GENDER_COL)
    except contract.FeatureContractError as err:
        problems.append(str(err))
    missing = [c for c in feature_names if c not in rows.columns]
    if missing:
        problems.append(f"{len(missing)} expected features missing, e.g. {missing[:3]}")
    extra = [c for c in rows.columns if c not in feature_names and c != data.GENDER_COL]
    if extra and not problems:
        problems.append(f"{len(extra)} unexpected columns, e.g. {extra[:3]}")
    if not missing:
        values = rows[list(feature_names)].apply(pd.to_numeric, errors="coerce").to_numpy()
        if not np.isfinite(values).all():
            problems.append(f"{int((~np.isfinite(values)).sum())} blank or non-numeric values")
    return problems


def _abstained(bundle: VoiceBundle, problems: list[str]) -> dict[str, Any]:
    return {
        "modality": "voice",
        "risk_score": None,
        "ci_lower": None,
        "ci_upper": None,
        "triage_priority": None,
        "availability_mask": {"voice": False},
        "shap_top5": [],
        "sex_fairness_note": bundle.model_card.get("sex_fairness_note", ""),
        "pipeline_version": f"{PIPELINE_VERSION}+{bundle.model_name}",
        "quality_gate_passed": False,
        "abstained": True,
        "problems": problems,
        "n_training": bundle.n_training,
        "note": "Voice input could not be scored, so voice is treated as absent. " + NOTE,
    }


def predict_voice(rows: pd.DataFrame, bundle: VoiceBundle) -> dict[str, Any]:
    """Score one person from their 1 to 3 recordings' feature rows (decision D5)."""
    problems = quality_gate(rows, bundle.feature_names)
    if problems:
        return _abstained(bundle, problems)

    X = rows[bundle.feature_names].to_numpy(dtype=float)
    risk = float(calibrate(bundle.calibrator, np.array([_person_raw(bundle.model, X)]))[0])
    members = calibrate(bundle.calibrator, np.array([_person_raw(m, X) for m in bundle.ensemble]))
    lower, upper = np.percentile(members, [2.5, 97.5])
    return {
        "modality": "voice",
        "risk_score": round(risk, 4),
        "ci_lower": round(float(lower), 4),
        "ci_upper": round(float(upper), 4),
        "triage_priority": TRIAGE_VALUES[0] if risk >= TRIAGE_THRESHOLD else TRIAGE_VALUES[1],
        "availability_mask": {"voice": True},
        "shap_top5": top_drivers(bundle.model, X, bundle.feature_names),
        "sex_fairness_note": bundle.model_card.get("sex_fairness_note", ""),
        "pipeline_version": f"{PIPELINE_VERSION}+{bundle.model_name}",
        "quality_gate_passed": True,
        "abstained": False,
        "problems": [],
        "n_training": bundle.n_training,
        "note": NOTE,
    }


def _estimate(e: metrics.Estimate) -> dict[str, float]:
    return {"value": round(e.value, 4), "lower": round(e.lower, 4), "upper": round(e.upper, 4)}


def _holdout_check(
    bundle: VoiceBundle,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Hold-out performance of the calibrated model, at person level with bootstrap CIs."""
    X_arr = X[bundle.feature_names].to_numpy(dtype=float)
    raw = pd.Series(bundle.model.predict_proba(X_arr)[:, 1]).groupby(groups).transform("mean")
    # Same order as predict_voice: average a person's recordings, then calibrate.
    risk = calibrate(bundle.calibrator, raw.to_numpy())
    y_p, _ = metrics.person_scores(y, risk, groups)

    def ci(metric: metrics.Metric) -> dict[str, float]:
        return _estimate(metrics.bootstrap_ci(y, risk, groups, metric, n_boot, seed))

    return {
        "n_people": int(len(y_p)),
        "n_pd": int(y_p.sum()),
        "n_healthy": int(len(y_p) - y_p.sum()),
        "auc": ci(metrics.auc),
        "balanced_accuracy": ci(metrics.balanced_accuracy),
        "sensitivity": ci(metrics.sensitivity),
        "specificity": ci(metrics.specificity),
        "ppv": {
            f"{p:.0%}": ci(metrics.ppv_metric(p, TRIAGE_THRESHOLD))
            for p in metrics.SCREENING_PREVALENCES
        },
    }


def _bootstrap_rows(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row indices for one bootstrap resample of PEOPLE (all of a person's rows together)."""
    people = np.unique(groups)
    rows_of = {p: np.flatnonzero(groups == p) for p in people}
    drawn = rng.choice(people, size=len(people), replace=True)
    return np.concatenate([rows_of[p] for p in drawn])


def train_voice_model(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    gender: np.ndarray,
    model: pipeline.Model = DEFAULT_MODEL,
    score: str = "mutual_info",
    n_ensemble: int = N_ENSEMBLE,
    n_boot: int = metrics.N_BOOT,
    seed: int = pipeline.SEED,
    k_grid: Sequence[int] = pipeline.K_GRID,
    c_grid: Sequence[float] = pipeline.C_GRID,
    leaves_grid: Sequence[int] = pipeline.LEAVES_GRID,
) -> tuple[VoiceBundle, np.ndarray]:
    """Train on the D3 training people. Returns the bundle and the hold-out row indices."""
    if data.GENDER_COL in X.columns:
        raise ValueError("gender must not be a model input (decision D7)")
    contract.validate_features(X.columns)
    y, groups, gender = map(np.asarray, (y, groups, gender))

    train, test = splitter.holdout_split(groups, y, gender, seed=seed)
    X_tr, y_tr, g_tr = X.iloc[train], y[train], groups[train]

    print("  out-of-fold scores for calibration ...", flush=True)
    oof = pipeline.nested_cv_predict(
        X_tr,
        y_tr,
        g_tr,
        score=score,
        seed=seed,
        model=model,
        k_grid=k_grid,
        c_grid=c_grid,
        leaves_grid=leaves_grid,
    ).proba
    y_person, oof_person = metrics.person_scores(y_tr, oof, g_tr)
    calibrator = fit_platt(oof_person, y_person, seed)

    print("  final model ...", flush=True)
    final, best_params = pipeline.fit_tuned(
        X_tr,
        y_tr,
        g_tr,
        model,
        score,
        seed=seed,
        k_grid=k_grid,
        c_grid=c_grid,
        leaves_grid=leaves_grid,
    )

    print(f"  bootstrap ensemble of {n_ensemble} ...", flush=True)
    rng = np.random.default_rng(seed)
    X_tr_arr = X_tr.to_numpy(dtype=float)
    ensemble = []
    for _ in range(n_ensemble):
        rows = _bootstrap_rows(g_tr, rng)
        ensemble.append(clone(final).fit(X_tr_arr[rows], y_tr[rows]))

    fairness = sex_fairness.breakdown(y_tr, oof, g_tr, gender[train], n_boot, seed)
    bundle = VoiceBundle(
        model_name=model,
        feature_names=[str(c) for c in X.columns],
        model=final,
        calibrator=calibrator,
        ensemble=ensemble,
        best_params=dict(best_params),
        n_training=int(len(np.unique(g_tr))),
    )
    bundle.model_card = {
        "modality": "voice",
        "pipeline_version": f"{PIPELINE_VERSION}+{model}",
        "intended_use": "Screening and triage only. Not a diagnostic device.",
        "training_data": "UCI-470 (Sakar et al.), precomputed acoustic features, 80% of people",
        "n_training_people": bundle.n_training,
        "n_holdout_people": int(len(np.unique(groups[test]))),
        "chosen_params": bundle.best_params,
        "calibration": "Platt, balanced class weights, on out-of-fold training scores "
        "(probabilities assume a 50/50 prior, not screening prevalence)",
        "ci_method": f"bootstrap ensemble of {n_ensemble} models over training people",
        "triage_threshold": TRIAGE_THRESHOLD,
        "holdout_check": _holdout_check(bundle, X.iloc[test], y[test], groups[test], n_boot, seed),
        "sex_fairness_note": sex_fairness.fairness_note(fairness),
        "evidence": "Headline numbers (5-fold person-level CV over all people) are in "
        "models/voice/RESULTS.md; the hold-out check above is a small sanity check.",
        "known_limits": [
            "Trained on people already diagnosed versus healthy controls: no evidence "
            "for early or prodromal detection.",
            "One dataset and one recording protocol; no external validation yet.",
            "Does not distinguish Parkinson's disease from other parkinsonian syndromes.",
            "Sex-code meaning is undocumented; fairness by age band was not possible.",
            "PPV at screening prevalence is low (see holdout_check.ppv and RESULTS.md).",
        ],
    }
    return bundle, test
