"""Honest evidence for the raw-audio voice model (task VOICE-22).

Reads the feature table from ``build_audio_table.py`` (quality-passed recordings
only) and reports, all at person level with bootstrap 95% CIs:

1. **Baseline ladder.** age + sex only -> best single voice feature (chosen per
   training fold) -> L1 logistic on all voice features -> LightGBM. Age and sex
   are inputs ONLY in the first rung; the voice models never see them.
2. **Age-matched check.** In Figshare, healthy people are much younger than PD
   people, and voice changes with age. So the voice model is also scored on the
   people whose age lies in the range both classes share. If it only works on
   the full set, it learned age, not PD.
3. **Dataset gate.** With more than one dataset, AUC is also reported within
   each dataset, and the run fails if the pooled model is above chance while
   no dataset is on its own (it learned the recording setup).
4. **Label-permutation control.** Shuffled labels must score at chance, or the
   run fails before anything is written (CLAUDE.md section 3.2).
5. PPV at 1/2/5% screening prevalence for each rung.

Everything uses the honest pipeline: person-level nested CV, scaling and
feature selection inside each training fold, seed 42.

Usage (from the repo root):
    python models/voice/src/audio_evidence.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import audio_features
import baseline_ladder
import metrics
import numpy as np
import pandas as pd
import permutation_control
import pipeline
import splitter
from build_audio_table import DEFAULT_OUT as FEATURE_TABLE

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
AUDIO_K_GRID = (5, 10, 20, 40)  # 56 features, so the UCI grid (up to 100) does not fit
THRESHOLD = 0.5
MIN_PER_CLASS = 10  # smallest per-class count for a subgroup AUC worth reporting


class DatasetGateError(RuntimeError):
    """The pooled model works but no single dataset does: it learned the recording setup."""


def load_passed(path: Path = FEATURE_TABLE) -> pd.DataFrame:
    """Quality-passed rows of the feature table. Refused rows are counted, never used."""
    table = pd.read_csv(path)
    passed = table[table.quality_passed].reset_index(drop=True)
    features = passed[list(audio_features.FEATURE_NAMES)]
    if features.isna().any().any():
        raise ValueError("a quality-passed recording has a blank feature")
    return passed


def _estimate(e: metrics.Estimate) -> dict[str, float]:
    return {"value": round(e.value, 4), "lower": round(e.lower, 4), "upper": round(e.upper, 4)}


def rung_summary(
    y: np.ndarray, proba: np.ndarray, groups: np.ndarray, n_boot: int, seed: int
) -> dict[str, Any]:
    def ci(metric: metrics.Metric) -> dict[str, float]:
        return _estimate(metrics.bootstrap_ci(y, proba, groups, metric, n_boot, seed))

    return {
        "auc": ci(metrics.auc),
        "balanced_accuracy": ci(metrics.balanced_accuracy),
        "sensitivity": ci(metrics.sensitivity),
        "specificity": ci(metrics.specificity),
        "ppv": {
            f"{p:.0%}": ci(metrics.ppv_metric(p, THRESHOLD)) for p in metrics.SCREENING_PREVALENCES
        },
    }


def age_sex_cv(table: pd.DataFrame, y: np.ndarray, folds: list, seed: int) -> np.ndarray:
    """Out-of-fold probabilities from age and sex alone (the bottom rung)."""
    if table.age.isna().any() or table.sex.isna().any():
        raise ValueError("age or sex is blank for some people; the age + sex rung needs both")
    X = np.column_stack([table.age.to_numpy(float), (table.sex == "M").to_numpy(float)])
    proba = np.full(len(y), np.nan)
    for train, test in folds:
        model = baseline_ladder._logistic(seed).fit(X[train], y[train])
        proba[test] = model.predict_proba(X[test])[:, 1]
    return proba


def shared_age_range(table: pd.DataFrame) -> tuple[float, float]:
    """Ages covered by BOTH classes: from the older of the two minimums to the younger maximum."""
    by_class = table.groupby("label").age
    return float(by_class.min().max()), float(by_class.max().min())


def subgroup_auc(
    y: np.ndarray, proba: np.ndarray, groups: np.ndarray, mask: np.ndarray, n_boot: int, seed: int
) -> dict[str, Any]:
    n_pd, n_hc = int(y[mask].sum()), int((1 - y[mask]).sum())
    result: dict[str, Any] = {"n_pd": n_pd, "n_healthy": n_hc}
    if min(n_pd, n_hc) < MIN_PER_CLASS:
        result["auc"] = None
        result["note"] = f"fewer than {MIN_PER_CLASS} people in a class"
        return result
    est = metrics.bootstrap_ci(y[mask], proba[mask], groups[mask], metrics.auc, n_boot, seed)
    result["auc"] = _estimate(est)
    return result


def dataset_gate(
    y: np.ndarray,
    proba: np.ndarray,
    groups: np.ndarray,
    datasets: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Within-dataset AUCs. Raises if pooled AUC is above chance but no dataset is."""
    names = sorted(set(datasets))
    within = {
        name: subgroup_auc(y, proba, groups, datasets == name, n_boot, seed) for name in names
    }
    result: dict[str, Any] = {"datasets": names, "within": within}
    if len(names) < 2:
        result["gate"] = "not applicable: only one dataset"
        return result
    pooled = metrics.bootstrap_ci(y, proba, groups, metrics.auc, n_boot, seed)
    any_within = any(w["auc"] and w["auc"]["lower"] > 0.5 for w in within.values())
    if pooled.lower > 0.5 and not any_within:
        raise DatasetGateError(
            "the pooled model is above chance but no single dataset is: it learned the "
            "recording setup, not PD. Do not report this model."
        )
    result["gate"] = "passed"
    return result


def run_evidence(
    table: pd.DataFrame, n_boot: int = metrics.N_BOOT, seed: int = pipeline.SEED, n_perm: int = 10
) -> dict[str, Any]:
    X = table[list(audio_features.FEATURE_NAMES)]
    y = table.label.to_numpy(int)
    groups = table.person.to_numpy()
    datasets = table.dataset.to_numpy()
    folds = splitter.person_folds(groups, y, seed=seed)
    grids = {"k_grid": AUDIO_K_GRID}

    print("  rung: age + sex only ...", flush=True)
    oof = {"age_sex_only": age_sex_cv(table, y, folds, seed)}
    print("  rung: best single voice feature ...", flush=True)
    oof["best_single_feature"], chosen = baseline_ladder.best_single_feature_cv(X, y, folds, seed)
    params: dict[str, Any] = {"best_single_feature": chosen}
    for model in ("l1_logistic", "lightgbm"):
        print(f"  rung: {model} on all voice features ...", flush=True)
        result = pipeline.nested_cv_predict(X, y, groups, model=model, seed=seed, **grids)
        oof[f"{model}_voice"] = result.proba
        params[f"{model}_voice"] = result.best_params

    ladder = {name: rung_summary(y, p, groups, n_boot, seed) for name, p in oof.items()}
    differences = {
        f"{a} - age_sex_only": _estimate(
            metrics.paired_bootstrap_difference(
                y, oof[a], oof["age_sex_only"], groups, metrics.auc, n_boot, seed
            )
        )
        for a in ("best_single_feature", "l1_logistic_voice", "lightgbm_voice")
    }

    low, high = shared_age_range(table)
    in_range = table.age.between(low, high).to_numpy()
    age_matched = {
        "age_range": [round(low, 1), round(high, 1)],
        **{
            name: subgroup_auc(y, oof[name], groups, in_range, n_boot, seed)
            for name in ("age_sex_only", "l1_logistic_voice", "lightgbm_voice")
        },
    }

    gate = {
        name: dataset_gate(y, oof[name], groups, datasets, n_boot, seed)
        for name in ("l1_logistic_voice", "lightgbm_voice")
    }

    print(f"  label-permutation control ({n_perm} shuffles per model) ...", flush=True)
    permutation = {}
    for model in ("l1_logistic", "lightgbm"):
        aucs = permutation_control.run_permutations(
            X, y, groups, model, n_permutations=n_perm, seed=seed, **grids
        )
        permutation_control.check_chance(aucs)  # raises before anything is written
        permutation[model] = {
            "mean": round(float(np.mean(aucs)), 4),
            "max": round(float(np.max(aucs)), 4),
        }

    y_p, _ = metrics.person_scores(y, oof["age_sex_only"], groups)
    return {
        "metric_unit": "person",
        "n_people": int(len(y_p)),
        "n_pd": int(y_p.sum()),
        "n_healthy": int(len(y_p) - y_p.sum()),
        "datasets": sorted(set(datasets)),
        "n_features": len(audio_features.FEATURE_NAMES),
        "seed": seed,
        "ladder": ladder,
        "auc_difference_vs_age_sex": differences,
        "chosen_params": params,
        "age_matched": age_matched,
        "dataset_gate": gate,
        "permutation_control": permutation,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", type=Path, default=FEATURE_TABLE)
    ap.add_argument("--n-boot", type=int, default=metrics.N_BOOT)
    args = ap.parse_args(argv)

    table = load_passed(args.table)
    print(f"audio evidence on {table.person.nunique()} people ...", flush=True)
    result = run_evidence(table, n_boot=args.n_boot)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS / "audio_evidence.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("n_people", "n_pd", "n_healthy", "datasets")}))
    for name, rung in result["ladder"].items():
        a = rung["auc"]
        print(f"  {name:22s} AUC {a['value']:.3f} [{a['lower']:.3f}, {a['upper']:.3f}]")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
