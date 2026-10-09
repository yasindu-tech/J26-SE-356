"""Baseline ladder and screening PPV for the voice module (task VOICE-15).

Every rung is scored on the SAME person-level outer folds (seed 42), so each
rung can be compared with every other on the same resampled people:

| Rung                | Input                       | Fitted inside each training fold        |
|---------------------|-----------------------------|-----------------------------------------|
| sex_only            | gender code only            | logistic regression                     |
| best_single_feature | one acoustic feature        | choice of feature (training-fold AUC) + |
|                     |                             | scaler + logistic regression            |
| l1_logistic_all     | all 752 features            | variant B of the leakage audit          |
| lightgbm_all        | all 752 features            | same scaler + selector, LightGBM model  |

Gender is a model input ONLY in the sex_only rung (decision D7). It shows how
much of the signal sex alone carries, since sex correlates with class here.

For each rung: person-level AUC, balanced accuracy, sensitivity, specificity and
PPV at 1%, 2% and 5% screening prevalence, each with a person-level bootstrap
95% CI. The threshold is a fixed 0.5 on balanced-class-weight probabilities; it
is never tuned on test people. The ladder rule (CLAUDE.md section 3.4): a rung
"beats" a lower rung only if the paired AUC difference CI lies above zero.

Outputs (gitignored, aggregates only, no ids):
    models/voice/artifacts/baseline_ladder.json
    models/voice/artifacts/baseline_ladder.png

Usage (from the repo root):
    python models/voice/src/baseline_ladder.py
    python models/voice/src/baseline_ladder.py --score anova_f --n-boot 200   # quick look
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import contract
import lightgbm
import metrics
import numpy as np
import pandas as pd
import pipeline
import sklearn
import splitter
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import data

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

RUNGS = ("sex_only", "best_single_feature", "l1_logistic_all", "lightgbm_all")
THRESHOLD = 0.5
PREVALENCE_KEYS = {p: f"{p:.0%}" for p in metrics.SCREENING_PREVALENCES}  # 0.01 -> "1%"


def _logistic(seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed),
    )


def best_feature(X: np.ndarray, y: np.ndarray) -> int:
    """Index of the column that best separates the classes, in either direction.

    Call on training rows only. Ties go to the lowest index.
    """
    strength = [abs(metrics.auc(y, X[:, j]) - 0.5) for j in range(X.shape[1])]
    return int(np.argmax(strength))


def single_column_cv(
    values: np.ndarray, y: np.ndarray, folds: Sequence[tuple[np.ndarray, np.ndarray]], seed: int
) -> np.ndarray:
    """Out-of-fold probabilities from a logistic regression on one column."""
    column = np.asarray(values, dtype=float).reshape(-1, 1)
    proba = np.full(len(y), np.nan)
    for train, test in folds:
        model = _logistic(seed).fit(column[train], y[train])
        proba[test] = model.predict_proba(column[test])[:, 1]
    return proba


def best_single_feature_cv(
    X: pd.DataFrame, y: np.ndarray, folds: Sequence[tuple[np.ndarray, np.ndarray]], seed: int
) -> tuple[np.ndarray, list[str]]:
    """Out-of-fold probabilities, with the feature chosen on each training fold alone."""
    contract.validate_features(X.columns)
    X_arr = X.to_numpy(dtype=float)
    proba = np.full(len(y), np.nan)
    chosen: list[str] = []
    for train, test in folds:
        j = best_feature(X_arr[train], y[train])
        model = _logistic(seed).fit(X_arr[train][:, [j]], y[train])
        proba[test] = model.predict_proba(X_arr[test][:, [j]])[:, 1]
        chosen.append(str(X.columns[j]))
    return proba, chosen


def _estimate(e: metrics.Estimate) -> dict[str, float]:
    return {"value": round(e.value, 4), "lower": round(e.lower, 4), "upper": round(e.upper, 4)}


def score_rung(
    y: np.ndarray, proba: np.ndarray, groups: np.ndarray, n_boot: int, seed: int
) -> dict[str, Any]:
    """Every headline metric for one rung, at person level with bootstrap CIs."""

    def ci(metric: metrics.Metric) -> dict[str, float]:
        return _estimate(metrics.bootstrap_ci(y, proba, groups, metric, n_boot, seed))

    return {
        "auc": ci(metrics.auc),
        "balanced_accuracy": ci(metrics.balanced_accuracy),
        "sensitivity": ci(metrics.sensitivity),
        "specificity": ci(metrics.specificity),
        "ppv": {key: ci(metrics.ppv_metric(p, THRESHOLD)) for p, key in PREVALENCE_KEYS.items()},
    }


def run_ladder(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    gender: np.ndarray,
    score: str = "mutual_info",
    n_boot: int = metrics.N_BOOT,
    seed: int = pipeline.SEED,
    k_grid: Sequence[int] = pipeline.K_GRID,
    c_grid: Sequence[float] = pipeline.C_GRID,
    leaves_grid: Sequence[int] = pipeline.LEAVES_GRID,
) -> dict[str, Any]:
    """Run all four rungs on the same person-level folds, plus every paired AUC difference."""
    y, groups = np.asarray(y), np.asarray(groups)
    # nested_cv_predict builds its outer folds with this same call, so all rungs share folds.
    folds = splitter.person_folds(groups, y, pipeline.N_OUTER, seed)

    proba: dict[str, np.ndarray] = {}
    details: dict[str, Any] = {}

    print("  running sex_only ...", flush=True)
    contract.validate_features([data.GENDER_COL])
    proba["sex_only"] = single_column_cv(gender, y, folds, seed)
    details["sex_only"] = {"input": data.GENDER_COL}

    print("  running best_single_feature ...", flush=True)
    proba["best_single_feature"], chosen = best_single_feature_cv(X, y, folds, seed)
    details["best_single_feature"] = {"chosen_feature_per_fold": chosen}

    full_models: tuple[tuple[str, pipeline.Model], ...] = (
        ("l1_logistic_all", "l1_logistic"),
        ("lightgbm_all", "lightgbm"),
    )
    for rung, model in full_models:
        print(f"  running {rung} ...", flush=True)
        result = pipeline.nested_cv_predict(
            X,
            y,
            groups,
            score=score,
            k_grid=k_grid,
            c_grid=c_grid,
            leaves_grid=leaves_grid,
            seed=seed,
            model=model,
        )
        proba[rung] = result.proba
        details[rung] = {"chosen_params_per_fold": result.best_params}

    rungs = {
        name: {**score_rung(y, proba[name], groups, n_boot, seed), **details[name]}
        for name in RUNGS
    }

    differences = {}
    for lower, upper in combinations(RUNGS, 2):
        diff = _estimate(
            metrics.paired_bootstrap_difference(
                y, proba[upper], proba[lower], groups, metrics.auc, n_boot, seed
            )
        )
        differences[f"{upper}_minus_{lower}"] = {**diff, "beats": diff["lower"] > 0}

    y_person, _ = metrics.person_scores(y, proba["sex_only"], groups)
    return {
        "metric_unit": "person (mean of a person's row probabilities)",
        "n_people": int(len(y_person)),
        "n_pd": int(y_person.sum()),
        "n_healthy": int(len(y_person) - y_person.sum()),
        "n_rows": int(len(y)),
        "n_features": int(X.shape[1]),
        "score_func": score,
        "threshold": THRESHOLD,
        "screening_prevalences": list(PREVALENCE_KEYS.values()),
        "n_boot": n_boot,
        "seed": seed,
        "k_grid": list(k_grid),
        "c_grid": list(c_grid),
        "leaves_grid": list(leaves_grid),
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
        },
        "rungs": rungs,
        "auc_differences": differences,
    }


def validate_ladder(result: dict[str, Any]) -> None:
    """Raise if the ladder output is incomplete. Run before anything is written."""
    missing = [r for r in RUNGS if r not in result.get("rungs", {})]
    if missing:
        raise ValueError(f"ladder is missing rungs: {missing}")
    for name, rung in result["rungs"].items():
        ppv = rung.get("ppv", {})
        absent = [k for k in PREVALENCE_KEYS.values() if k not in ppv]
        if absent:
            raise ValueError(f"{name} has no PPV at screening prevalence {absent}")
        estimates = {m: rung.get(m, {}) for m in ("auc", "balanced_accuracy")}
        estimates |= {f"ppv {k}": v for k, v in ppv.items()}
        for metric, est in estimates.items():
            if not {"value", "lower", "upper"} <= est.keys():
                raise ValueError(f"{name}.{metric} has no confidence interval")
            if not est["lower"] <= est["value"] <= est["upper"]:
                raise ValueError(f"{name}.{metric} value lies outside its own CI")
    expected = {f"{u}_minus_{lo}" for lo, u in combinations(RUNGS, 2)}
    absent = expected - result.get("auc_differences", {}).keys()
    if absent:
        raise ValueError(f"paired differences missing: {sorted(absent)}")


def _fmt(e: dict[str, float], signed: bool = False) -> str:
    f = "+.3f" if signed else ".3f"
    return f"{e['value']:{f}} [{e['lower']:{f}}, {e['upper']:{f}}]"


def results_markdown(result: dict[str, Any]) -> str:
    """Aggregate tables for models/voice/RESULTS.md."""
    keys = result["screening_prevalences"]
    rows = [
        "| Rung | AUC [95% CI] | Balanced accuracy [95% CI] | Sensitivity | Specificity |",
        "|---|---|---|---|---|",
    ]
    for name, r in result["rungs"].items():
        rows.append(
            f"| {name} | {_fmt(r['auc'])} | {_fmt(r['balanced_accuracy'])} | "
            f"{_fmt(r['sensitivity'])} | {_fmt(r['specificity'])} |"
        )
    rows += [
        "",
        "| Rung | " + " | ".join(f"PPV at {k} [95% CI]" for k in keys) + " |",
        "|---|" + "---|" * len(keys),
    ]
    for name, r in result["rungs"].items():
        rows.append(f"| {name} | " + " | ".join(_fmt(r["ppv"][k]) for k in keys) + " |")
    rows += ["", "| Paired AUC difference | Value [95% CI] | Beats? |", "|---|---|---|"]
    for name, d in result["auc_differences"].items():
        rows.append(f"| {name} | {_fmt(d, signed=True)} | {'yes' if d['beats'] else 'no'} |")
    return "\n".join(rows)


def plot_ladder(result: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(RUNGS)
    rungs = result["rungs"]
    colors = ["#6C7572", "#6C7572", "#0F4C5C", "#0F4C5C"]  # packages/shared theme tokens
    fig, (ax_auc, ax_ppv) = plt.subplots(1, 2, figsize=(11, 4.5))

    auc = [rungs[n]["auc"] for n in names]
    values = [a["value"] for a in auc]
    errors = [[a["value"] - a["lower"] for a in auc], [a["upper"] - a["value"] for a in auc]]
    labels = [n.replace("_", "\n", 1).replace("_", " ") for n in names]
    ax_auc.bar(labels, values, yerr=errors, capsize=6, color=colors)
    for i, v in enumerate(values):
        ax_auc.text(i, 0.42, f"{v:.3f}", ha="center", color="white", fontweight="bold")
    ax_auc.axhline(0.5, color="#B45309", linestyle="--", label="chance")
    ax_auc.set_ylim(0.4, 1.0)
    ax_auc.set_ylabel("AUC")
    ax_auc.set_title("Baseline ladder (person-level AUC, 95% CI)")
    ax_auc.legend(loc="upper left", fontsize=8)

    keys = result["screening_prevalences"]
    x = np.arange(len(keys))
    width = 0.8 / len(names)
    for i, (name, color) in enumerate(zip(names, colors, strict=True)):
        ppv = [rungs[name]["ppv"][k] for k in keys]
        ax_ppv.bar(
            x + (i - (len(names) - 1) / 2) * width,
            [p["value"] for p in ppv],
            width,
            yerr=[[p["value"] - p["lower"] for p in ppv], [p["upper"] - p["value"] for p in ppv]],
            capsize=3,
            color=color,
            alpha=0.45 + 0.15 * i,
            label=name,
        )
    ax_ppv.set_xticks(x, [f"{k} prevalence" for k in keys])
    ax_ppv.set_ylabel("PPV")
    ax_ppv.set_title(f"PPV at screening prevalence (threshold {result['threshold']})")
    ax_ppv.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--csv", type=Path, default=data.DEFAULT_CSV)
    ap.add_argument("--out-dir", type=Path, default=ARTIFACTS)
    ap.add_argument("--score", choices=sorted(pipeline.SCORE_FUNCS), default="mutual_info")
    ap.add_argument("--n-boot", type=int, default=metrics.N_BOOT)
    args = ap.parse_args(argv)

    X, y, groups, gender = data.load_uci470(args.csv)
    print(
        f"baseline ladder: {len(np.unique(groups))} people, {X.shape[1]} features, "
        f"score={args.score}, n_boot={args.n_boot}"
    )
    result = run_ladder(X, y, groups, gender, score=args.score, n_boot=args.n_boot)
    validate_ladder(result)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "baseline_ladder.json").write_text(json.dumps(result, indent=2) + "\n")
    plot_ladder(result, args.out_dir / "baseline_ladder.png")

    print()
    print(results_markdown(result))
    print(f"\nwritten to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
