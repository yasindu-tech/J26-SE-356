"""Label-permutation control for the voice module (task VOICE-16).

Mandatory control (CLAUDE.md section 3.2): shuffle the labels, re-run the honest
pipeline, and require chance-level AUC. With shuffled labels there is nothing
real to learn, so anything above chance means information is leaking from the
test people into training.

How the labels are shuffled: per PERSON, not per row. Each person keeps one
label for all 3 recordings and the class balance (188 PD / 64 healthy) is
unchanged; only who has which label is random.

One permutation at n=252 swings by several AUC points by chance, so the gate is
on the mean over ``n_permutations`` shuffles: it fails if the mean person-level
AUC exceeds 0.5 + ``TOLERANCE``. A failed gate raises ``PermutationControlError``
and the script exits non-zero; it never just warns.

Outputs (gitignored, aggregates only, no ids):
    models/voice/artifacts/permutation_control.json

Usage (from the repo root):
    python models/voice/src/permutation_control.py
    python models/voice/src/permutation_control.py --score anova_f --n-permutations 5   # quick
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

N_PERMUTATIONS = 10
TOLERANCE = 0.05  # about 3 standard errors of a 10-permutation mean at n=252
MODELS: tuple[pipeline.Model, ...] = ("l1_logistic", "lightgbm")


class PermutationControlError(RuntimeError):
    """The pipeline scored above chance on shuffled labels: something leaks."""


def permute_labels(y: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle labels between people. Every row of one person gets the same new label."""
    y, groups = np.asarray(y), np.asarray(groups)
    _, first_row, row_person = np.unique(groups, return_index=True, return_inverse=True)
    shuffled = rng.permutation(y[first_row])
    return shuffled[row_person]


def check_chance(aucs: Sequence[float], tolerance: float = TOLERANCE) -> None:
    """Raise if the mean AUC on shuffled labels is above chance."""
    if not len(aucs):
        raise ValueError("no permutation results to check")
    mean = float(np.mean(aucs))
    if mean > 0.5 + tolerance:
        raise PermutationControlError(
            f"mean AUC on shuffled labels is {mean:.3f} over {len(aucs)} permutations "
            f"(limit {0.5 + tolerance:.3f}); the pipeline is learning from something "
            "other than the labels. Do not report any result until this is fixed."
        )


def run_permutations(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    model: pipeline.Model = "l1_logistic",
    selection: pipeline.Selection = "infold",
    split: pipeline.Split = "subject",
    score: str = "mutual_info",
    n_permutations: int = N_PERMUTATIONS,
    seed: int = pipeline.SEED,
    k_grid: Sequence[int] = pipeline.K_GRID,
    c_grid: Sequence[float] = pipeline.C_GRID,
    leaves_grid: Sequence[int] = pipeline.LEAVES_GRID,
) -> list[float]:
    """Person-level AUC of the pipeline on ``n_permutations`` label shuffles."""
    aucs = []
    for i in range(n_permutations):
        y_perm = permute_labels(y, groups, np.random.default_rng(seed + i))
        proba = pipeline.nested_cv_predict(
            X,
            y_perm,
            groups,
            selection,
            split,
            score,
            k_grid=k_grid,
            c_grid=c_grid,
            leaves_grid=leaves_grid,
            seed=seed,
            model=model,
        ).proba
        aucs.append(metrics.auc(*metrics.person_scores(y_perm, proba, groups)))
    return aucs


def run_control(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    models: Sequence[pipeline.Model] = MODELS,
    score: str = "mutual_info",
    n_permutations: int = N_PERMUTATIONS,
    tolerance: float = TOLERANCE,
    seed: int = pipeline.SEED,
    **grids: Any,
) -> dict[str, Any]:
    """Run the control for each honest model. Raises on the first model that fails."""
    results: dict[str, Any] = {}
    for model in models:
        print(f"  {model}: {n_permutations} label permutations ...", flush=True)
        aucs = run_permutations(
            X, y, groups, model, score=score, n_permutations=n_permutations, seed=seed, **grids
        )
        check_chance(aucs, tolerance)  # raises before anything is written
        results[model] = {
            "auc_per_permutation": [round(a, 4) for a in aucs],
            "mean": round(float(np.mean(aucs)), 4),
            "sd": round(float(np.std(aucs, ddof=1)), 4) if len(aucs) > 1 else None,
            "max": round(float(np.max(aucs)), 4),
            "passed": True,
        }
    return {
        "metric_unit": "person (mean of a person's row probabilities)",
        "pipeline": "honest: in-fold selection, person-level split (variant B)",
        "shuffle_unit": "person",
        "n_people": int(len(np.unique(groups))),
        "n_permutations": n_permutations,
        "chance_limit": round(0.5 + tolerance, 4),
        "score_func": score,
        "seed": seed,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
        "models": results,
    }


def results_markdown(result: dict[str, Any]) -> str:
    rows = [
        f"| Model | Mean AUC on shuffled labels (n={result['n_permutations']}) | SD | Max | "
        f"Gate (mean <= {result['chance_limit']:.2f}) |",
        "|---|---|---|---|---|",
    ]
    for model, r in result["models"].items():
        rows.append(
            f"| {model} | {r['mean']:.3f} | {r['sd']:.3f} | {r['max']:.3f} | "
            f"{'pass' if r['passed'] else 'FAIL'} |"
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--csv", type=Path, default=data.DEFAULT_CSV)
    ap.add_argument("--out-dir", type=Path, default=ARTIFACTS)
    ap.add_argument("--score", choices=sorted(pipeline.SCORE_FUNCS), default="mutual_info")
    ap.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    args = ap.parse_args(argv)

    X, y, groups, _ = data.load_uci470(args.csv)
    print(
        f"permutation control: {len(np.unique(groups))} people, {X.shape[1]} features, "
        f"score={args.score}, permutations={args.n_permutations}"
    )
    try:
        result = run_control(X, y, groups, score=args.score, n_permutations=args.n_permutations)
    except PermutationControlError as err:
        print(f"\nPERMUTATION CONTROL FAILED: {err}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "permutation_control.json").write_text(json.dumps(result, indent=2) + "\n")
    print()
    print(results_markdown(result))
    print(f"\nwritten to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
