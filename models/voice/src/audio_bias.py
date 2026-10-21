"""Does the raw-audio model's score depend on age or sex? (exploratory)

The model the desktop upload uses (``audio_model.py``) never sees age or sex,
but voice changes with both, and in Figshare healthy people are much younger
than PD people. This script asks, on the honest out-of-fold scores (person-level
nested CV, the same run as the ladder in ``audio_evidence.py``):

1. **Sex.** Per-sex AUC, sensitivity and specificity, and the gap between the
   sexes with a CI (``sex_fairness.breakdown``).
2. **Age, within each class.** If the model had learned age instead of PD,
   older healthy people would get higher scores than younger healthy people.
   The Spearman correlation of score with age is reported separately for healthy
   and PD people, with a bootstrap CI over people. ``age_driven`` is True when the
   healthy-group CI lies wholly above zero.
3. **False referrals by age band.** How many healthy people in each age band
   score at or above the triage threshold.

Other conditions that change the voice (a cold, asthma, a vocal-cord problem)
cannot be checked: no dataset we hold labels them. That limit is written into
the output, not left out.

Every result is exploratory: 77 people, some bands with very few.

Usage (from the repo root):
    python models/voice/src/audio_bias.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import audio_evidence
import audio_features
import metrics
import numpy as np
import pandas as pd
import pipeline
import sex_fairness
from scipy.stats import spearmanr

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
AGE_BANDS = (0, 50, 65, 120)  # (0, 50], (50, 65], (65, 120]
THRESHOLD = audio_evidence.THRESHOLD
UNTESTED_CONDITIONS = (
    "Not checked: no dataset available to us labels other conditions that change the voice "
    "(colds, sore throat, asthma or other breathing problems, vocal-cord, throat or neck "
    "conditions). The model may score such voices higher; the result card says so."
)


def age_correlation(
    age: np.ndarray, proba: np.ndarray, n_boot: int, seed: int
) -> dict[str, float | int]:
    """Spearman correlation of score with age, with a bootstrap 95% CI over people."""
    rng = np.random.default_rng(seed)
    value = float(spearmanr(age, proba).statistic)
    draws = []
    for _ in range(n_boot):
        rows = rng.integers(0, len(age), len(age))
        if np.unique(age[rows]).size > 1:
            draws.append(float(spearmanr(age[rows], proba[rows]).statistic))
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return {
        "n": int(len(age)),
        "value": round(value, 4),
        "lower": round(float(low), 4),
        "upper": round(float(high), 4),
    }


def false_referrals_by_age(age: np.ndarray, proba: np.ndarray) -> dict[str, dict[str, int]]:
    """Healthy people per age band, and how many of them score at or above the threshold."""
    bands = pd.cut(age, AGE_BANDS)
    result = {}
    for band in bands.categories:
        mask = np.asarray(bands == band)
        result[f"{int(band.left)}-{int(band.right)}"] = {
            "n_healthy": int(mask.sum()),
            "referred": int((proba[mask] >= THRESHOLD).sum()),
        }
    return result


def bias_report(table: pd.DataFrame, proba: np.ndarray, n_boot: int, seed: int) -> dict[str, Any]:
    """Person-level bias report from one score per person (one recording each in Figshare)."""
    y = table.label.to_numpy(int)
    groups = table.person.to_numpy()
    age = table.age.to_numpy(float)
    healthy, pd_ = y == 0, y == 1
    by_sex = sex_fairness.breakdown(y, proba, groups, table.sex.to_numpy(), n_boot, seed)
    by_sex["note"] = sex_fairness.fairness_note(by_sex)
    within = {
        "healthy": age_correlation(age[healthy], proba[healthy], n_boot, seed),
        "pd": age_correlation(age[pd_], proba[pd_], n_boot, seed),
    }
    return {
        "status": "exploratory",
        "metric_unit": "person",
        "n_people": int(len(np.unique(groups))),
        "age_and_sex_are_model_inputs": False,
        "sex": by_sex,
        "age_vs_score_within_class": within,
        "age_driven": within["healthy"]["lower"] > 0,
        "healthy_referred_by_age_band": false_referrals_by_age(age[healthy], proba[healthy]),
        "other_voice_conditions": UNTESTED_CONDITIONS,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-boot", type=int, default=metrics.N_BOOT)
    args = ap.parse_args(argv)

    table = audio_evidence.load_passed()
    if table.person.duplicated().any():
        raise ValueError("expected one recording per person; average per person first")
    X = table[list(audio_features.FEATURE_NAMES)]
    y = table.label.to_numpy(int)
    print(f"bias check on {len(table)} people (out-of-fold L1 logistic) ...", flush=True)
    proba = pipeline.nested_cv_predict(
        X,
        y,
        table.person.to_numpy(),
        model="l1_logistic",
        seed=pipeline.SEED,
        k_grid=audio_evidence.AUDIO_K_GRID,
    ).proba
    result = bias_report(table, proba, args.n_boot, pipeline.SEED)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS / "audio_bias.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"sex: {result['sex']['note']}")
    for cls, r in result["age_vs_score_within_class"].items():
        print(f"age vs score, {cls}: r {r['value']:+.2f} [{r['lower']:+.2f}, {r['upper']:+.2f}]")
    print(f"healthy referred by age band: {result['healthy_referred_by_age_band']}")
    print(f"age_driven: {result['age_driven']}")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
