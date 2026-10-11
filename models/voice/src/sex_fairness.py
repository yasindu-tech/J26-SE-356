"""Sex-fairness breakdown for the voice module (task VOICE-17).

Sex correlates with class in UCI-470 (sex code 0: 41 healthy / 81 PD; sex code
1: 23 healthy / 107 PD). This script answers two questions about the honest
models (in-fold selection, person-level split, the same runs as the ladder):

1. **Does the model work equally well for both sexes?** Person-level AUC,
   balanced accuracy, sensitivity and specificity per sex code, each with a
   bootstrap 95% CI, and the gap between the groups (code 1 minus code 0) with a
   CI from a stratified bootstrap (each group resampled on its own).
   The sensitivity gap is the equal-opportunity gap; the specificity gap is the
   false-referral gap.
2. **Is the model just detecting sex?** If it were, its AUC *within* each sex
   would fall to chance. ``driven_by_sex`` is True when the overall AUC CI is
   above 0.5 but neither within-sex AUC CI is.

Gender is NOT a model input (decision D7): it is used only to split the
results. Every finding here is exploratory (two groups of 122 and 130 people,
one with only 23 healthy people), and is labelled so in the output.

UCI-470 has no age column, so the age-band breakdown promised in the proposal
cannot be done on this dataset.

The male/female meaning of the codes is not documented in the vault, so
results are reported by code only.

Outputs (gitignored, aggregates only, no ids):
    models/voice/artifacts/sex_fairness.json
    models/voice/artifacts/sex_fairness.png

Usage (from the repo root):
    python models/voice/src/sex_fairness.py
    python models/voice/src/sex_fairness.py --score anova_f --n-boot 200   # quick look
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import metrics
import numpy as np
import pandas as pd
import pipeline
import sklearn

import data

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

MODELS: tuple[pipeline.Model, ...] = ("l1_logistic", "lightgbm")
THRESHOLD = 0.5
MIN_PER_CLASS = 10  # below this a per-group CI is not worth reporting
GROUP_METRICS: dict[str, metrics.Metric] = {
    "auc": metrics.auc,
    "balanced_accuracy": metrics.balanced_accuracy,
    "sensitivity": metrics.sensitivity,
    "specificity": metrics.specificity,
}
GAP_METRICS = ("auc", "sensitivity", "specificity")


def _estimate(e: metrics.Estimate) -> dict[str, float]:
    return {"value": round(e.value, 4), "lower": round(e.lower, 4), "upper": round(e.upper, 4)}


def _codes(gender: np.ndarray) -> list[Any]:
    codes = sorted(np.unique(gender).tolist())
    if len(codes) != 2:
        raise ValueError(f"expected exactly 2 sex codes, found {len(codes)}")
    return codes


def group_counts(y: np.ndarray, groups: np.ndarray, gender: np.ndarray) -> dict[str, Any]:
    """People per sex code and class. Raises if a group is too small to report."""
    people = pd.DataFrame({"g": groups, "y": y, "sex": gender}).groupby("g").first()
    counts: dict[str, Any] = {}
    for code in _codes(gender):
        in_group = people[people.sex == code]
        n_pd, n_hc = int(in_group.y.sum()), int((in_group.y == 0).sum())
        if min(n_pd, n_hc) < MIN_PER_CLASS:
            raise ValueError(
                f"sex code {code} has {n_pd} PD and {n_hc} healthy people; "
                f"need at least {MIN_PER_CLASS} of each"
            )
        counts[f"sex_code_{code}"] = {"n_people": n_pd + n_hc, "n_pd": n_pd, "n_healthy": n_hc}
    return counts


def breakdown(
    y: np.ndarray,
    proba: np.ndarray,
    groups: np.ndarray,
    gender: np.ndarray,
    n_boot: int = metrics.N_BOOT,
    seed: int = pipeline.SEED,
) -> dict[str, Any]:
    """Per-sex metrics, between-sex gaps and the sex-confound check for one model."""
    y, proba, groups, gender = map(np.asarray, (y, proba, groups, gender))
    code_a, code_b = _codes(gender)

    per_group = {}
    for code in (code_a, code_b):
        mask = gender == code
        per_group[f"sex_code_{code}"] = {
            name: _estimate(
                metrics.bootstrap_ci(y[mask], proba[mask], groups[mask], fn, n_boot, seed)
            )
            for name, fn in GROUP_METRICS.items()
        }

    gaps = {
        name: _estimate(
            metrics.group_bootstrap_difference(
                y, proba, groups, gender == code_b, GROUP_METRICS[name], n_boot, seed
            )
        )
        for name in GAP_METRICS
    }
    for gap in gaps.values():
        gap["ci_excludes_zero"] = gap["lower"] > 0 or gap["upper"] < 0

    overall = _estimate(metrics.bootstrap_ci(y, proba, groups, metrics.auc, n_boot, seed))
    within_above_chance = [g["auc"]["lower"] > 0.5 for g in per_group.values()]
    return {
        "overall_auc": overall,
        "per_sex_code": per_group,
        f"gap_sex_code_{code_b}_minus_sex_code_{code_a}": gaps,
        "driven_by_sex": overall["lower"] > 0.5 and not any(within_above_chance),
    }


def fairness_note(model_result: dict[str, Any]) -> str:
    """One plain sentence for the result vector's ``sex_fairness_note`` field (decision D5)."""
    gap_key = next(k for k in model_result if k.startswith("gap_"))
    groups = model_result["per_sex_code"]
    (name_a, a), (name_b, b) = groups.items()
    auc_gap = model_result[gap_key]["auc"]
    verdict = (
        "the gap is not distinguishable from zero"
        if not auc_gap["ci_excludes_zero"]
        else "the gap is distinguishable from zero"
    )
    note = (
        f"Exploratory: AUC {a['auc']['value']:.2f} for {name_a.replace('_', ' ')} and "
        f"{b['auc']['value']:.2f} for {name_b.replace('_', ' ')}; {verdict} "
        f"(gap {auc_gap['value']:+.2f}, 95% CI {auc_gap['lower']:+.2f} to {auc_gap['upper']:+.2f})."
    )
    # Equal AUC can hide an unequal operating point, so threshold gaps are named too.
    for name in ("sensitivity", "specificity"):
        gap = model_result[gap_key][name]
        if gap["ci_excludes_zero"]:
            note += (
                f" At the {THRESHOLD} threshold {name} is {a[name]['value']:.2f} for "
                f"{name_a.replace('_', ' ')} and {b[name]['value']:.2f} for "
                f"{name_b.replace('_', ' ')} (gap {gap['value']:+.2f}, 95% CI "
                f"{gap['lower']:+.2f} to {gap['upper']:+.2f})."
            )
    if model_result["driven_by_sex"]:
        note += " Within each sex the score is at chance: it mostly reflects sex, not voice."
    return note


def run_fairness(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    gender: np.ndarray,
    models: Sequence[pipeline.Model] = MODELS,
    score: str = "mutual_info",
    n_boot: int = metrics.N_BOOT,
    seed: int = pipeline.SEED,
    **grids: Any,
) -> dict[str, Any]:
    """Fit each honest model once (out-of-fold) and break its scores down by sex."""
    if data.GENDER_COL in X.columns:
        raise ValueError("gender must not be a model input (decision D7)")
    y, groups, gender = map(np.asarray, (y, groups, gender))
    counts = group_counts(y, groups, gender)

    results = {}
    for model in models:
        print(f"  {model} ...", flush=True)
        proba = pipeline.nested_cv_predict(
            X, y, groups, score=score, seed=seed, model=model, **grids
        ).proba
        result = breakdown(y, proba, groups, gender, n_boot, seed)
        result["sex_fairness_note"] = fairness_note(result)
        results[model] = result

    return {
        "status": "exploratory",
        "metric_unit": "person (mean of a person's row probabilities)",
        "pipeline": "honest: in-fold selection, person-level split (variant B)",
        "gender_is_model_input": False,
        "sex_code_meaning": "not documented in the vault; reported by code only",
        "age_band_analysis": "not possible: UCI-470 has no age column",
        "threshold": THRESHOLD,
        "n_boot": n_boot,
        "seed": seed,
        "score_func": score,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
        "counts": counts,
        "models": results,
    }


def validate_fairness(result: dict[str, Any]) -> None:
    """Raise if the fairness output is incomplete. Run before anything is written."""
    if result.get("status") != "exploratory":
        raise ValueError("fairness results must be labelled exploratory")
    for model, r in result.get("models", {}).items():
        if len(r.get("per_sex_code", {})) != 2:
            raise ValueError(f"{model} needs results for both sex codes")
        for code, group in r["per_sex_code"].items():
            for name in GROUP_METRICS:
                est = group.get(name, {})
                if not {"value", "lower", "upper"} <= est.keys():
                    raise ValueError(f"{model}.{code}.{name} has no confidence interval")
        gap_key = next((k for k in r if k.startswith("gap_")), None)
        if gap_key is None or set(r[gap_key]) != set(GAP_METRICS):
            raise ValueError(f"{model} is missing between-sex gaps")
        if not r.get("sex_fairness_note"):
            raise ValueError(f"{model} has no sex_fairness_note")
    if not result.get("models"):
        raise ValueError("no models in the fairness result")


def _fmt(e: dict[str, float], signed: bool = False) -> str:
    f = "+.3f" if signed else ".3f"
    return f"{e['value']:{f}} [{e['lower']:{f}}, {e['upper']:{f}}]"


def results_markdown(result: dict[str, Any]) -> str:
    rows = ["| Group | People | PD | Healthy |", "|---|---|---|---|"]
    for code, c in result["counts"].items():
        rows.append(f"| {code} | {c['n_people']} | {c['n_pd']} | {c['n_healthy']} |")
    for model, r in result["models"].items():
        gap_key = next(k for k in r if k.startswith("gap_"))
        rows += [
            "",
            f"**{model}** (overall AUC {_fmt(r['overall_auc'])})",
            "",
            "| Group | AUC | Balanced accuracy | Sensitivity | Specificity |",
            "|---|---|---|---|---|",
        ]
        for code, g in r["per_sex_code"].items():
            rows.append(
                f"| {code} | {_fmt(g['auc'])} | {_fmt(g['balanced_accuracy'])} | "
                f"{_fmt(g['sensitivity'])} | {_fmt(g['specificity'])} |"
            )
        gaps = r[gap_key]
        rows.append(
            f"| {gap_key.removeprefix('gap_')} | {_fmt(gaps['auc'], True)} | - | "
            f"{_fmt(gaps['sensitivity'], True)} | {_fmt(gaps['specificity'], True)} |"
        )
        rows += [
            "",
            f"Driven by sex: {'yes' if r['driven_by_sex'] else 'no'}. "
            f"Note: {r['sex_fairness_note']}",
        ]
    return "\n".join(rows)


def plot_fairness(result: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(GROUP_METRICS)
    models = list(result["models"])
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4.2), squeeze=False)
    colors = ("#0F4C5C", "#B45309")  # packages/shared theme tokens
    width = 0.38
    x = np.arange(len(names))
    for ax, model in zip(axes[0], models, strict=True):
        groups = result["models"][model]["per_sex_code"]
        for offset, (code, color) in zip(
            (-width / 2, width / 2), zip(groups, colors, strict=True), strict=True
        ):
            est = [groups[code][n] for n in names]
            ax.bar(
                x + offset,
                [e["value"] for e in est],
                width,
                yerr=[
                    [e["value"] - e["lower"] for e in est],
                    [e["upper"] - e["value"] for e in est],
                ],
                capsize=4,
                color=color,
                label=code.replace("_", " "),
            )
        ax.axhline(0.5, color="#6C7572", linestyle="--", linewidth=1)
        ax.set_xticks(x, [n.replace("_", "\n") for n in names])
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{model} by sex code (95% CI), exploratory")
        ax.legend(fontsize=8, loc="lower right")
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
        f"sex fairness: {len(np.unique(groups))} people, score={args.score}, n_boot={args.n_boot}"
    )
    result = run_fairness(X, y, groups, gender, score=args.score, n_boot=args.n_boot)
    validate_fairness(result)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sex_fairness.json").write_text(json.dumps(result, indent=2) + "\n")
    plot_fairness(result, args.out_dir / "sex_fairness.png")
    print()
    print(results_markdown(result))
    print(f"\nwritten to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
