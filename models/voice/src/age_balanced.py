"""Does the raw-audio voice model detect PD or age? Two age-balanced tests (exploratory).

In Figshare the healthy group is about 19 years younger than the PD group, and
age and sex alone separate them better than voice does (``audio_evidence.py``).
The ladder's age-matched check only RE-SCORES the full model on people of
shared ages; the model was still trained with the young healthy people. Here
age is taken out of training too, in two ways:

1. **Same ages only.** Keep the people whose age lies in the range BOTH classes
   cover (``audio_evidence.shared_age_range``), then train AND test inside it
   (person-level nested CV). The age + sex baseline is scored on the same people,
   so the remaining age gap is visible, not hidden.
2. **Age-adjusted voice.** Voice changes with normal ageing. Inside each training
   fold, every voice measure is regressed on age using the HEALTHY training
   people only (normal ageing), and that age effect is subtracted from every
   person, train and test (``AgeAdjuster``). The model then sees voice "beyond
   what age predicts" and cannot use age as a linear shortcut. Age is used only
   for the adjustment, never as a model input: a clinic knows it (year of birth
   is on the assessment form).

Both are compared with the current model on the same people, with paired
bootstrap differences, and the age-adjusted pipeline gets its own label-
permutation control (CLAUDE.md section 3.2): shuffled labels must score at
chance, or the run fails before anything is written.

Reading the result: if voice keeps its AUC once age is taken out, the model
hears more than age. If it falls to chance, the current 0.71 was mostly age.

Usage (from the repo root):
    python models/voice/src/age_balanced.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import audio_evidence
import audio_features
import contract
import metrics
import numpy as np
import pandas as pd
import permutation_control
import pipeline
import splitter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
N_PERMUTATIONS = 10
MIN_HEALTHY_FOR_FIT = 5  # fewer healthy training people than this: fit the age effect on all


class AgeAdjuster(BaseEstimator, TransformerMixin):
    """Removes the normal-ageing effect from every voice measure.

    Input: the voice measures with age as the LAST column. ``fit`` learns, per
    measure, a straight line ``measure = a + b * age`` on the healthy rows
    (``y == 0``) it is given; ``transform`` subtracts that line and drops the age
    column. Inside a Pipeline it is refitted on each training fold only.
    """

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> AgeAdjuster:
        X = np.asarray(X, dtype=float)
        voice, age = X[:, :-1], X[:, -1]
        rows = np.ones(len(X), bool) if y is None else np.asarray(y) == 0
        if rows.sum() < MIN_HEALTHY_FOR_FIT:
            rows = np.ones(len(X), bool)
        design = np.column_stack([np.ones(rows.sum()), age[rows]])
        coef, *_ = np.linalg.lstsq(design, voice[rows], rcond=None)
        self.intercept_, self.slope_ = coef[0], coef[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        voice, age = X[:, :-1], X[:, -1]
        return voice - (self.intercept_ + np.outer(age, self.slope_))


def adjusted_pipeline(seed: int, model: pipeline.Model = "l1_logistic") -> Pipeline:
    """AgeAdjuster -> scaler -> in-fold selection -> model (same settings as the ladder)."""
    return Pipeline(
        [
            ("adjust", AgeAdjuster()),
            ("scale", StandardScaler()),
            ("select", SelectKBest(pipeline.SCORE_FUNCS["mutual_info"])),
            ("model", pipeline._model(model, seed)),
        ]
    )


def nested_cv_adjusted(
    voice: pd.DataFrame,
    age: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int = pipeline.SEED,
    k_grid: Sequence[int] = audio_evidence.AUDIO_K_GRID,
) -> np.ndarray:
    """Out-of-fold probabilities of the age-adjusted model (mirrors pipeline.nested_cv_predict)."""
    contract.validate_features(voice.columns)  # age is NOT a feature: it is only the adjustment
    if np.isnan(age).any():
        raise ValueError("age is blank for some people; the age adjustment needs it")
    X = np.column_stack([voice.to_numpy(float), age])
    y, groups = np.asarray(y), np.asarray(groups)
    grid = pipeline.param_grid("l1_logistic", k_grid)
    proba = np.full(len(y), np.nan)
    for train, test in splitter.person_folds(groups, y, seed=seed):
        inner = splitter.person_folds(groups[train], y[train], n_splits=pipeline.N_INNER, seed=seed)
        search = GridSearchCV(adjusted_pipeline(seed), grid, scoring="roc_auc", cv=inner, n_jobs=-1)
        search.fit(X[train], y[train])
        proba[test] = search.predict_proba(X[test])[:, 1]
    if np.isnan(proba).any():
        raise RuntimeError("some people received no out-of-fold prediction")
    return proba


def _voice_cv(table: pd.DataFrame, seed: int) -> np.ndarray:
    X = table[list(audio_features.FEATURE_NAMES)]
    return pipeline.nested_cv_predict(
        X, table.label.to_numpy(int), table.person.to_numpy(), model="l1_logistic", seed=seed,
        k_grid=audio_evidence.AUDIO_K_GRID,
    ).proba  # fmt: skip


def _adjusted_cv(table: pd.DataFrame, seed: int, labels: np.ndarray | None = None) -> np.ndarray:
    y = table.label.to_numpy(int) if labels is None else labels
    voice = table[list(audio_features.FEATURE_NAMES)]
    return nested_cv_adjusted(voice, table.age.to_numpy(float), y, table.person.to_numpy(), seed)


def _auc(y: np.ndarray, p: np.ndarray, g: np.ndarray, n_boot: int, seed: int) -> dict[str, float]:
    return audio_evidence._estimate(metrics.bootstrap_ci(y, p, g, metrics.auc, n_boot, seed))


def _difference(
    y: np.ndarray, a: np.ndarray, b: np.ndarray, g: np.ndarray, n_boot: int, seed: int
) -> dict[str, float]:
    return audio_evidence._estimate(
        metrics.paired_bootstrap_difference(y, a, b, g, metrics.auc, n_boot, seed)
    )


def run(table: pd.DataFrame, n_boot: int, seed: int = pipeline.SEED) -> dict[str, Any]:
    if table.person.duplicated().any():
        raise ValueError("expected one recording per person; average per person first")
    y, g = table.label.to_numpy(int), table.person.to_numpy()
    folds = splitter.person_folds(g, y, seed=seed)

    print("  current model and age + sex, all people ...", flush=True)
    current = _voice_cv(table, seed)
    age_sex = audio_evidence.age_sex_cv(table, y, folds, seed)

    print(f"  age-adjusted voice: label-permutation control ({N_PERMUTATIONS}) ...", flush=True)
    rng = np.random.default_rng(seed)
    shuffled = [
        metrics.auc(y_s, _adjusted_cv(table, seed, y_s))
        for y_s in (rng.permutation(y) for _ in range(N_PERMUTATIONS))
    ]
    permutation_control.check_chance(shuffled)

    print("  age-adjusted voice, all people ...", flush=True)
    adjusted = _adjusted_cv(table, seed)

    low, high = audio_evidence.shared_age_range(table)
    same_age = table[table.age.between(low, high)].reset_index(drop=True)
    ys, gs = same_age.label.to_numpy(int), same_age.person.to_numpy()
    print(f"  same ages only ({low:.1f}-{high:.1f}, {len(same_age)} people) ...", flush=True)
    folds_s = splitter.person_folds(gs, ys, seed=seed)
    same_voice = _voice_cv(same_age, seed)
    same_age_sex = audio_evidence.age_sex_cv(same_age, ys, folds_s, seed)

    return {
        "status": "exploratory",
        "metric_unit": "person",
        "age_is_model_input": False,
        "all_people": {
            "n_pd": int(y.sum()),
            "n_healthy": int(len(y) - y.sum()),
            "age_gap_years": round(float(table[y == 1].age.mean() - table[y == 0].age.mean()), 1),
            "auc": {
                "current_voice_model": _auc(y, current, g, n_boot, seed),
                "age_adjusted_voice": _auc(y, adjusted, g, n_boot, seed),
                "age_and_sex_only": _auc(y, age_sex, g, n_boot, seed),
            },
            "age_adjusted_minus_current": _difference(y, adjusted, current, g, n_boot, seed),
        },
        "same_ages_only": {
            "age_range": [round(low, 1), round(high, 1)],
            "n_pd": int(ys.sum()),
            "n_healthy": int(len(ys) - ys.sum()),
            "age_gap_years": round(
                float(same_age[ys == 1].age.mean() - same_age[ys == 0].age.mean()), 1
            ),
            "auc": {
                "voice_trained_and_tested_here": _auc(ys, same_voice, gs, n_boot, seed),
                "age_and_sex_only": _auc(ys, same_age_sex, gs, n_boot, seed),
            },
            "voice_minus_age_and_sex": _difference(ys, same_voice, same_age_sex, gs, n_boot, seed),
        },
        "permutation_control_adjusted": {
            "n": N_PERMUTATIONS,
            "mean_auc": round(float(np.mean(shuffled)), 4),
            "max_auc": round(float(np.max(shuffled)), 4),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-boot", type=int, default=metrics.N_BOOT)
    args = ap.parse_args(argv)
    table = audio_evidence.load_passed()
    print(f"age-balanced tests on {len(table)} people ...", flush=True)
    result = run(table, args.n_boot)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS / "age_balanced.json"
    out.write_text(json.dumps(result, indent=2) + "\n")

    def show(label: str, e: dict[str, float]) -> None:
        print(f"  {label:42s} {e['value']:+.3f} [{e['lower']:+.3f}, {e['upper']:+.3f}]")

    a, s = result["all_people"], result["same_ages_only"]
    print(f"all people (age gap {a['age_gap_years']} years):")
    for name, e in a["auc"].items():
        show(f"AUC {name}", e)
    show("age-adjusted minus current", a["age_adjusted_minus_current"])
    print(f"same ages only {s['age_range']} (age gap {s['age_gap_years']} years):")
    for name, e in s["auc"].items():
        show(f"AUC {name}", e)
    show("voice minus age + sex", s["voice_minus_age_and_sex"])
    pc = result["permutation_control_adjusted"]
    print(f"permutation control (adjusted): mean {pc['mean_auc']:.3f}, max {pc['max_auc']:.3f}")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
