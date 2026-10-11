"""Demo: train the voice model and score five held-out people (task VOICE-18, PP1 item 7).

Trains on the D3 training people only, then scores five people from the
hold-out (stratified pick: PD and healthy both shown), plus one deliberately
broken input to show the quality gate abstaining.

People are shown by pseudonym only. The reference label is printed so the demo
can be checked; it is never an input.

Outputs (gitignored):
    models/voice/artifacts/demo_results.json   result vectors, pseudonymised
    models/voice/artifacts/model_card.json     the model card
    models/voice/artifacts/voice_bundle.joblib  the trained bundle (no data rows)

Usage (from the repo root):
    python models/voice/src/demo_voice.py
    python models/voice/src/demo_voice.py --model lightgbm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pipeline
import voice_module

import data

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
N_DEMO = 5


def pick_demo_people(
    y: np.ndarray, groups: np.ndarray, test: np.ndarray, n: int = N_DEMO, seed: int = 42
) -> list[Any]:
    """``n`` hold-out people: about 3 PD and 2 healthy, chosen reproducibly."""
    people = pd.DataFrame({"g": groups[test], "y": y[test]}).groupby("g").y.first()
    rng = np.random.default_rng(seed)
    n_hc = min(n // 2, int((people == 0).sum()))
    pd_people = rng.choice(people.index[people == 1], n - n_hc, replace=False)
    hc_people = rng.choice(people.index[people == 0], n_hc, replace=False)
    return [*pd_people, *hc_people]


def run_demo(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    gender: np.ndarray,
    model: pipeline.Model = voice_module.DEFAULT_MODEL,
    **train_kwargs: Any,
) -> tuple[voice_module.VoiceBundle, list[dict[str, Any]]]:
    bundle, test = voice_module.train_voice_model(X, y, groups, gender, model, **train_kwargs)
    results = []
    for person in pick_demo_people(y, groups, test):
        rows = X[groups == person]
        result = voice_module.predict_voice(rows, bundle)
        label = "PD" if y[groups == person][0] == 1 else "healthy"
        results.append({"person": data.pseudonym(person), "reference_label": label, **result})

    # One broken input: a recording with a blank feature value. The module must abstain.
    broken = X[groups == pick_demo_people(y, groups, test)[0]].iloc[:1].copy()
    broken.iloc[0, 0] = np.nan
    result = voice_module.predict_voice(broken, bundle)
    results.append({"person": "broken-input-example", "reference_label": "-", **result})
    return bundle, results


def summary_table(results: list[dict[str, Any]]) -> str:
    rows = [
        "| Person | Reference | Risk [95% CI] | Triage | Top driver |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        if r["abstained"]:
            rows.append(
                f"| {r['person']} | {r['reference_label']} | abstained | - | {r['problems'][0]} |"
            )
            continue
        top = r["shap_top5"][0]
        rows.append(
            f"| {r['person']} | {r['reference_label']} | "
            f"{r['risk_score']:.2f} [{r['ci_lower']:.2f}, {r['ci_upper']:.2f}] | "
            f"{r['triage_priority']} | {top['feature']} ({top['direction']}) |"
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--csv", type=Path, default=data.DEFAULT_CSV)
    ap.add_argument("--out-dir", type=Path, default=ARTIFACTS)
    ap.add_argument(
        "--model", choices=("l1_logistic", "lightgbm"), default=voice_module.DEFAULT_MODEL
    )
    ap.add_argument("--score", choices=sorted(pipeline.SCORE_FUNCS), default="mutual_info")
    ap.add_argument("--n-ensemble", type=int, default=voice_module.N_ENSEMBLE)
    args = ap.parse_args(argv)

    X, y, groups, gender = data.load_uci470(args.csv)
    print(f"voice demo: training {args.model} on the D3 training people ...")
    bundle, results = run_demo(
        X, y, groups, gender, args.model, score=args.score, n_ensemble=args.n_ensemble
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "demo_results.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.out_dir / "model_card.json").write_text(json.dumps(bundle.model_card, indent=2) + "\n")
    joblib.dump(bundle, args.out_dir / "voice_bundle.joblib")

    print()
    print(summary_table(results))
    check = bundle.model_card["holdout_check"]
    print(
        f"\nhold-out check ({check['n_people']} people): AUC {check['auc']['value']:.3f} "
        f"[{check['auc']['lower']:.3f}, {check['auc']['upper']:.3f}]"
    )
    print(f"sex fairness note: {bundle.model_card['sex_fairness_note']}")
    print(f"\nwritten to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
