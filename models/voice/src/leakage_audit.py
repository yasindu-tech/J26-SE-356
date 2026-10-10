"""Leakage audit: the voice module's headline result (task VOICE-14).

Runs the same models and seeds under four evaluation designs, all scored at
person level with person-level bootstrap 95% CIs:

| Variant | Feature selection        | Split     | What it shows                          |
|---------|--------------------------|-----------|----------------------------------------|
| A_full  | on all rows first (leaky) | by row    | the way many papers do it              |
| A_sel   | on all rows first (leaky) | by person | cost of selection leakage alone        |
| A_split | inside each fold          | by row    | cost of one person on both sides alone |
| B       | inside each fold          | by person | our honest result                      |

The headline is the paired AUC difference A_full - B with its CI. A pure-noise
check (no signal at all) is run alongside as a sanity chart.

Outputs (gitignored, aggregates only, no ids):
    models/voice/artifacts/leakage_audit.json
    models/voice/artifacts/leakage_audit.png
    models/voice/artifacts/noise_check.png

Usage (from the repo root):
    python models/voice/src/leakage_audit.py
    python models/voice/src/leakage_audit.py --score anova_f --n-boot 200   # quick look
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

import metrics
import numpy as np
import pandas as pd
import pipeline
import sklearn

import data

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

VARIANTS: dict[str, tuple[pipeline.Selection, pipeline.Split]] = {
    "A_full": ("leaky", "row"),
    "A_sel": ("leaky", "subject"),
    "A_split": ("infold", "row"),
    "B": ("infold", "subject"),
}
HONEST = "B"
HEADLINE = "A_full"
NOISE_PEOPLE = 100
NOISE_REPEATS = 10


def _estimate(e: metrics.Estimate) -> dict[str, float]:
    return {"value": round(e.value, 4), "lower": round(e.lower, 4), "upper": round(e.upper, 4)}


def run_audit(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    score: str = "mutual_info",
    n_boot: int = metrics.N_BOOT,
    seed: int = pipeline.SEED,
    k_grid: Sequence[int] = pipeline.K_GRID,
    c_grid: Sequence[float] = pipeline.C_GRID,
) -> dict[str, Any]:
    """Run all four variants and the paired differences against B."""
    proba: dict[str, np.ndarray] = {}
    variants: dict[str, Any] = {}
    for name, (selection, split) in VARIANTS.items():
        print(f"  running {name} (selection={selection}, split={split}) ...", flush=True)
        result = pipeline.nested_cv_predict(
            X, y, groups, selection, split, score, k_grid=k_grid, c_grid=c_grid, seed=seed
        )
        proba[name] = result.proba
        variants[name] = {
            "selection": selection,
            "split": split,
            "auc": _estimate(
                metrics.bootstrap_ci(y, result.proba, groups, metrics.auc, n_boot, seed)
            ),
            "balanced_accuracy": _estimate(
                metrics.bootstrap_ci(
                    y, result.proba, groups, metrics.balanced_accuracy, n_boot, seed
                )
            ),
            "chosen_k_and_C_per_fold": [
                [p["select__k"], p["model__C"]] for p in result.best_params
            ],
        }

    differences = {
        f"{name}_minus_{HONEST}": _estimate(
            metrics.paired_bootstrap_difference(
                y, proba[name], proba[HONEST], groups, metrics.auc, n_boot, seed
            )
        )
        for name in VARIANTS
        if name != HONEST
    }
    y_person, _ = metrics.person_scores(y, proba[HONEST], groups)
    return {
        "metric_unit": "person (mean of a person's row probabilities)",
        "n_people": int(len(y_person)),
        "n_pd": int(y_person.sum()),
        "n_healthy": int(len(y_person) - y_person.sum()),
        "n_rows": int(len(y)),
        "n_features": int(X.shape[1]),
        "score_func": score if isinstance(score, str) else "custom",
        "n_boot": n_boot,
        "seed": seed,
        "k_grid": list(k_grid),
        "c_grid": list(c_grid),
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
        "variants": variants,
        "auc_differences": differences,
        "headline": f"{HEADLINE}_minus_{HONEST}",
    }


def validate_audit(result: dict[str, Any]) -> None:
    """Raise if the audit output is incomplete. Run before anything is written."""
    missing = [v for v in VARIANTS if v not in result.get("variants", {})]
    if missing:
        raise ValueError(f"audit is missing variants: {missing}")
    for name, v in result["variants"].items():
        for metric in ("auc", "balanced_accuracy"):
            est = v.get(metric, {})
            if not {"value", "lower", "upper"} <= est.keys():
                raise ValueError(f"{name}.{metric} has no confidence interval")
            if not est["lower"] <= est["value"] <= est["upper"]:
                raise ValueError(f"{name}.{metric} value lies outside its own CI")
    if result.get("headline") not in result.get("auc_differences", {}):
        raise ValueError("headline paired difference is missing")


def noise_check(
    scores: Sequence[str] = ("anova_f", "mutual_info"),
    n_repeats: int = NOISE_REPEATS,
    seed: int = pipeline.SEED,
    n_features: int = 752,
) -> dict[str, dict[str, float]]:
    """In-fold vs leaky AUC on pure noise, averaged over ``n_repeats`` noise datasets.

    One noise dataset of 100 people swings by about +/-0.08 AUC by chance alone,
    so a single draw can look like signal; the mean over repeats is reported.

    Honest (in-fold) should average near 0.5 for every scorer. How far leaky
    rises depends on the scorer: ANOVA F picks features with a linear (spurious)
    link to the label, which logistic regression can exploit; mutual information
    on noise scores most features as exactly 0, so its leaky picks transfer less.
    Both are reported so neither number is shown without the other.
    """
    aucs: dict[str, dict[str, list[float]]] = {s: {"infold": [], "leaky": []} for s in scores}
    for repeat in range(n_repeats):
        rng = np.random.default_rng(seed + repeat)
        groups = np.repeat(np.arange(NOISE_PEOPLE), 3)
        y = np.repeat(rng.integers(0, 2, NOISE_PEOPLE), 3)
        X = pd.DataFrame(
            rng.normal(size=(len(groups), n_features)),
            columns=[f"noise_{i}" for i in range(n_features)],
        )
        for score in scores:
            for selection in ("infold", "leaky"):
                proba = pipeline.nested_cv_predict(X, y, groups, selection, "subject", score).proba
                aucs[score][selection].append(metrics.auc(*metrics.person_scores(y, proba, groups)))
    return {
        score: {
            "n_repeats": n_repeats,
            **{
                f"{sel}_{stat}": round(float(fn(v)), 4)
                for sel, v in per.items()
                for stat, fn in (("mean", np.mean), ("sd", partial(np.std, ddof=1)))
            },
        }
        for score, per in aucs.items()
    }


def plot_audit(result: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(VARIANTS)
    auc = [result["variants"][n]["auc"] for n in names]
    values = [a["value"] for a in auc]
    errors = [[a["value"] - a["lower"] for a in auc], [a["upper"] - a["value"] for a in auc]]
    labels = [
        "A_full\nleaky selection\nrow split",
        "A_sel\nleaky selection\nperson split",
        "A_split\nin-fold selection\nrow split",
        "B (honest)\nin-fold selection\nperson split",
    ]
    colors = ["#B45309", "#B45309", "#B45309", "#0F4C5C"]  # packages/shared theme tokens

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, yerr=errors, capsize=6, color=colors)
    for i, v in enumerate(values):
        ax.text(i, 0.52, f"{v:.3f}", ha="center", color="white", fontweight="bold")
    head = result["auc_differences"][result["headline"]]
    ax.set_title(
        f"Voice leakage audit (person-level AUC, 95% bootstrap CI)\n"
        f"A_full - B = {head['value']:+.3f} [{head['lower']:+.3f}, {head['upper']:+.3f}]"
    )
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUC")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_noise(noise: dict[str, dict[str, float]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = list(noise)
    x = np.arange(len(scores))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6, 4))
    for offset, selection, color in (
        (-width / 2, "infold", "#0F4C5C"),
        (width / 2, "leaky", "#B45309"),
    ):
        values = [noise[s][f"{selection}_mean"] for s in scores]
        sds = [noise[s][f"{selection}_sd"] for s in scores]
        ax.bar(
            x + offset,
            values,
            width,
            yerr=sds,
            capsize=5,
            color=color,
            label=f"{selection} selection",
        )
        for xi, v in zip(x + offset, values, strict=True):
            ax.text(xi, 0.04, f"{v:.3f}", ha="center", color="white", fontweight="bold")
    ax.axhline(0.5, color="#6C7572", linestyle="--", label="chance")
    ax.set_xticks(x, [s.replace("_", " ") + " scoring" for s in scores])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUC")
    n_rep = next(iter(noise.values()))["n_repeats"]
    ax.set_title(f"Pure-noise check: mean AUC (+/- SD) over {n_rep} noise datasets")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def results_markdown(result: dict[str, Any]) -> str:
    """Aggregate table for models/voice/RESULTS.md."""
    rows = [
        "| Variant | Selection | Split | AUC [95% CI] | Balanced accuracy [95% CI] |",
        "|---|---|---|---|---|",
    ]
    for name, v in result["variants"].items():
        a, b = v["auc"], v["balanced_accuracy"]
        rows.append(
            f"| {name} | {v['selection']} | {v['split']} | "
            f"{a['value']:.3f} [{a['lower']:.3f}, {a['upper']:.3f}] | "
            f"{b['value']:.3f} [{b['lower']:.3f}, {b['upper']:.3f}] |"
        )
    rows += ["", "| Paired AUC difference | Value [95% CI] |", "|---|---|"]
    for name, d in result["auc_differences"].items():
        rows.append(f"| {name} | {d['value']:+.3f} [{d['lower']:+.3f}, {d['upper']:+.3f}] |")
    if "noise_check" in result:
        rows += [
            "",
            "| Pure-noise check (mean +/- SD over noise datasets) | In-fold AUC | Leaky AUC |",
            "|---|---|---|",
        ]
        for score, n in result["noise_check"].items():
            rows.append(
                f"| {score} scoring, {n['n_repeats']} datasets | "
                f"{n['infold_mean']:.3f} +/- {n['infold_sd']:.3f} | "
                f"{n['leaky_mean']:.3f} +/- {n['leaky_sd']:.3f} |"
            )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--csv", type=Path, default=data.DEFAULT_CSV)
    ap.add_argument("--out-dir", type=Path, default=ARTIFACTS)
    ap.add_argument("--score", choices=sorted(pipeline.SCORE_FUNCS), default="mutual_info")
    ap.add_argument("--n-boot", type=int, default=metrics.N_BOOT)
    ap.add_argument("--skip-noise", action="store_true")
    args = ap.parse_args(argv)

    X, y, groups, _ = data.load_uci470(args.csv)
    print(
        f"leakage audit: {len(np.unique(groups))} people, {X.shape[1]} features, "
        f"score={args.score}, n_boot={args.n_boot}"
    )
    result = run_audit(X, y, groups, score=args.score, n_boot=args.n_boot)
    if not args.skip_noise:
        print("  running pure-noise check ...", flush=True)
        result["noise_check"] = noise_check()
    validate_audit(result)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "leakage_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    plot_audit(result, args.out_dir / "leakage_audit.png")
    if "noise_check" in result:
        plot_noise(result["noise_check"], args.out_dir / "noise_check.png")

    print()
    print(results_markdown(result))
    print(f"\nwritten to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
