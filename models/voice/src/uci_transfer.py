"""Use the UCI-470 model on uploaded audio, through the features Praat can recompute (VOICE-25).

The full UCI-470 model cannot score new audio: 22 of the 24 features it uses
(TQWT, MFCC, wavelet) were computed by the dataset authors' own tools and
cannot be reproduced. But UCI-470's "Baseline" group was computed with Praat,
which ``parselmouth`` runs too. This module:

1. Recomputes those measures from an audio file with Praat's own "Voice report"
   (``uci_praat.voice_report``), on the same steady vowel ``audio_features`` finds.
   ``UCI_PRAAT_FEATURES`` lists the 16 kept; excluded on purpose:
   - numPulses, numPeriodsPulses: depend on how long the recording is;
   - PPE, DFA, RPDE: not Praat measures (custom algorithms, settings unknown);
   - intensity: depends on microphone gain; formants: Praat settings unknown
     and F4 sits above the 4 kHz limit of telephone audio.
2. Trains the UCI model on only these 16 columns, with the same honest pipeline
   (person-level nested CV, in-fold selection), and its own label-permutation
   control.
3. Tests the transfer: the UCI-trained model scores Figshare recordings it has
   never seen (another country, telephone audio). An external AUC above chance
   means the transfer works; a drop is the cross-cohort drop CLAUDE.md asks us to
   measure.
4. Checks the units: the median of each recomputed measure in Figshare healthy
   voices is compared with UCI-470 healthy voices. A large ratio means our
   numbers do not mean the same thing as theirs, and is reported, not hidden.

Usage (from the repo root):
    python models/voice/src/uci_transfer.py           # evidence + final model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import audio_evidence
import audio_model
import joblib
import metrics
import numpy as np
import pandas as pd
import permutation_control
import pipeline
import voice_module
from uci_praat import UCI_FEATURE_INFO, UCI_PRAAT_FEATURES

import data

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
BUNDLE_PATH = ARTIFACTS / "voice_uci_praat_bundle.joblib"
UCI_K_GRID = (4, 8, 16)
MISMATCH_RATIO = 2.0  # medians more than 2x apart are flagged as a unit or method mismatch

# --- evidence ------------------------------------------------------------------


def load_uci_subset() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """UCI-470 restricted to the 16 Praat-recomputable columns."""
    X, y, groups, gender = data.load_uci470()
    return X[list(UCI_PRAAT_FEATURES)], np.asarray(y), np.asarray(groups), np.asarray(gender)


def unit_check(uci_X: pd.DataFrame, uci_y: np.ndarray, ext: pd.DataFrame) -> dict[str, Any]:
    """Median of each measure in healthy voices: UCI-470 vs the external recordings."""
    uci_med = uci_X[uci_y == 0].median()
    ext_med = ext.loc[ext.label == 0, list(UCI_PRAAT_FEATURES)].median()
    rows = {}
    for name in UCI_PRAAT_FEATURES:
        ratio = float(ext_med[name] / uci_med[name]) if uci_med[name] else float("nan")
        rows[name] = {
            "uci_healthy_median": round(float(uci_med[name]), 6),
            "external_healthy_median": round(float(ext_med[name]), 6),
            "ratio": round(ratio, 2),
            "mismatch": not (1 / MISMATCH_RATIO <= ratio <= MISMATCH_RATIO),
        }
    return rows


def external_scores(bundle: voice_module.VoiceBundle, ext: pd.DataFrame) -> np.ndarray:
    """Calibrated scores of the UCI-trained model on external recordings."""
    X = ext[bundle.feature_names].to_numpy(float)
    raw = bundle.model.predict_proba(X)[:, 1]
    return voice_module.calibrate(bundle.calibrator, raw)


def run_transfer(
    external: pd.DataFrame,
    n_boot: int = metrics.N_BOOT,
    seed: int = pipeline.SEED,
    n_perm: int = 10,
) -> tuple[dict[str, Any], audio_model.AudioBundle]:
    X, y, groups, gender = load_uci_subset()
    grids = {"k_grid": UCI_K_GRID}

    print("  UCI-470, 16 Praat features: nested CV ...", flush=True)
    uci_cv = {}
    for model in ("l1_logistic", "lightgbm"):
        proba = pipeline.nested_cv_predict(X, y, groups, model=model, seed=seed, **grids).proba
        uci_cv[model] = audio_evidence.rung_summary(y, proba, groups, n_boot, seed)

    print(f"  label-permutation control ({n_perm} shuffles) ...", flush=True)
    aucs = permutation_control.run_permutations(
        X, y, groups, "l1_logistic", n_permutations=n_perm, seed=seed, **grids
    )
    permutation_control.check_chance(aucs)

    print("  final UCI model (D3 hold-out) ...", flush=True)
    voice, test = voice_module.train_voice_model(X, y, groups, gender, seed=seed, **grids)
    train = np.setdiff1d(np.arange(len(y)), test)
    bundle = audio_model.AudioBundle(
        voice=voice,
        reference=audio_model.feature_percentiles(X.iloc[train], y[train]),
        global_importance=audio_model.global_importance(voice, X.iloc[train], UCI_FEATURE_INFO),
        datasets=["UCI-470"],
        feature_set="uci_praat",
        feature_info=UCI_FEATURE_INFO,
    )

    print("  external test on the raw-audio recordings ...", flush=True)
    ext_y = external.label.to_numpy(int)
    ext_groups = external.person.to_numpy()
    scores = external_scores(voice, external)
    low, high = audio_evidence.shared_age_range(external)
    in_range = external.age.between(low, high).to_numpy()
    transfer = {
        "datasets": sorted(external.dataset.unique()),
        "n_pd": int(ext_y.sum()),
        "n_healthy": int(len(ext_y) - ext_y.sum()),
        **audio_evidence.rung_summary(ext_y, scores, ext_groups, n_boot, seed),
        "age_matched": {
            "age_range": [round(low, 1), round(high, 1)],
            **audio_evidence.subgroup_auc(ext_y, scores, ext_groups, in_range, n_boot, seed),
        },
    }

    bundle.model_card = dict(voice.model_card)
    bundle.model_card.update(
        {
            "pipeline_version": bundle.pipeline_version,
            "input": "one sustained 'aaah' recording, any common audio format",
            "training_data": "UCI-470 (Sakar et al.), 80% of people, ONLY the 16 Baseline "
            "measures Praat can recompute from audio (UCI_PRAAT_FEATURES)",
            "global_importance": bundle.global_importance,
            "external_test": {
                k: transfer[k] for k in ("datasets", "n_pd", "n_healthy", "auc", "age_matched")
            },
            "known_limits": [
                "Trained on people already diagnosed versus healthy controls.",
                "Uses 16 of UCI-470's 752 features; the full model's TQWT features cannot be "
                "recomputed from audio.",
                "UCI-470 audio was a studio-quality microphone; uploads and Figshare are "
                "resampled to 8 kHz, so measures may differ (see the unit check).",
                "Does not distinguish Parkinson's disease from other voice conditions.",
            ],
        }
    )
    bundle.voice.model_card["pipeline_version"] = bundle.pipeline_version

    units = unit_check(X, y, external)
    bundle.incomparable = [name for name, row in units.items() if row["mismatch"]]
    bundle.model_card["incomparable_features"] = bundle.incomparable

    result = {
        "metric_unit": "person",
        "features": list(UCI_PRAAT_FEATURES),
        "uci_cv": uci_cv,
        "uci_permutation": {
            "mean": round(float(np.mean(aucs)), 4),
            "max": round(float(np.max(aucs)), 4),
        },
        "final_model_params": voice.best_params,
        "uci_holdout_check": voice.model_card["holdout_check"]["auc"],
        "external_test": transfer,
        "unit_check": units,
    }
    return result, bundle


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-boot", type=int, default=metrics.N_BOOT)
    args = ap.parse_args(argv)

    external = audio_evidence.load_passed()
    if any(c not in external.columns for c in UCI_PRAAT_FEATURES):
        print(
            "the feature table lacks UCI measures; run build_audio_table.py first", file=sys.stderr
        )
        return 1
    # Recordings where Praat could not measure a value are counted, not imputed.
    unmeasured = external[list(UCI_PRAAT_FEATURES)].isna().any(axis=1)
    external = external[~unmeasured].reset_index(drop=True)
    result, bundle = run_transfer(external, n_boot=args.n_boot)
    result["external_test"]["n_unmeasured_excluded"] = int(unmeasured.sum())

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "uci_transfer.json").write_text(json.dumps(result, indent=2) + "\n")
    joblib.dump(bundle, BUNDLE_PATH)
    (ARTIFACTS / "uci_praat_model_card.json").write_text(
        json.dumps(bundle.model_card, indent=2) + "\n"
    )

    for model, rung in result["uci_cv"].items():
        a = rung["auc"]
        print(f"UCI-470 CV, {model}: AUC {a['value']:.3f} [{a['lower']:.3f}, {a['upper']:.3f}]")
    a = result["external_test"]["auc"]
    print(
        f"external test ({', '.join(result['external_test']['datasets'])}): AUC "
        f"{a['value']:.3f} [{a['lower']:.3f}, {a['upper']:.3f}]"
    )
    flagged = [k for k, v in result["unit_check"].items() if v["mismatch"]]
    print(
        f"unit check: {len(flagged)} of 16 measures differ more than {MISMATCH_RATIO}x: {flagged}"
    )
    print(f"written to {ARTIFACTS / 'uci_transfer.json'} and {BUNDLE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
