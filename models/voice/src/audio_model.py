"""Train the raw-audio voice model used by ``analyze_audio`` (task VOICE-23).

Reuses ``voice_module.train_voice_model`` unchanged (D3 hold-out by class and
sex, in-fold tuning, Platt calibration, bootstrap-ensemble CI), on the feature
table from ``build_audio_table.py``. On top of it, the bundle stores what the
explanations need, all computed from TRAINING people only:

- ``reference``: percentiles (0..100) of every feature for healthy and for PD
  training people, so a new voice can be placed among them ("higher than 90% of
  healthy voices"). Only percentiles are kept, never a person's row.
- ``global_importance``: mean |SHAP| per feature and per feature group over the
  training recordings: what the model relies on in general.

``AudioBundle`` is shared with ``uci_transfer.py``, whose bundle holds the
UCI-470 model on the 16 Praat measures (``feature_set = "uci_praat"``).

The evidence for this model (ladder, age-matched check, permutation control) is
in ``audio_evidence.py``; the hold-out check in the model card is a small
sanity check only.

Usage (from the repo root):
    python models/voice/src/audio_model.py
    python models/voice/src/audio_model.py --model lightgbm
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import audio_evidence
import audio_features
import joblib
import numpy as np
import pandas as pd
import pipeline
import voice_module
from build_audio_table import DEFAULT_OUT as FEATURE_TABLE

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
BUNDLE_PATH = ARTIFACTS / "voice_audio_bundle.joblib"
PIPELINE_VERSIONS = {"audio": "voice-audio-0.1.0", "uci_praat": "voice-uci-praat-0.1.0"}
PERCENTILES = np.arange(101)

FeatureInfo = dict[str, tuple[str, str]]  # feature -> (group, plain meaning)


@dataclass
class AudioBundle:
    """A trained voice model plus training-population references. Holds no person rows."""

    voice: voice_module.VoiceBundle
    reference: dict[str, dict[str, list[float]]]  # class -> feature -> 101 percentiles
    global_importance: dict[str, Any]
    datasets: list[str]
    feature_set: str = "audio"  # "audio" (56 features) or "uci_praat" (16 UCI measures)
    feature_info: FeatureInfo = field(default_factory=lambda: dict(audio_features.FEATURE_INFO))
    model_card: dict[str, Any] = field(default_factory=dict)
    # Local paths of hold-out recordings (never trained on): the only files to demo with.
    holdout_paths: list[str] = field(default_factory=list)
    # Features whose values from new audio are known not to match the training data's
    # (``uci_transfer.unit_check``); explanations must say so instead of comparing them.
    incomparable: list[str] = field(default_factory=list)

    @property
    def pipeline_version(self) -> str:
        return f"{PIPELINE_VERSIONS[self.feature_set]}+{self.voice.model_name}"


def feature_percentiles(X: pd.DataFrame, y: np.ndarray) -> dict[str, dict[str, list[float]]]:
    """101 percentiles of each feature, separately for healthy (0) and PD (1) rows."""
    y = np.asarray(y)
    return {
        cls: {
            name: [round(float(v), 9) for v in np.percentile(X.loc[y == code, name], PERCENTILES)]
            for name in X.columns
        }
        for cls, code in (("healthy", 0), ("pd", 1))
    }


def global_importance(
    bundle: voice_module.VoiceBundle,
    X: pd.DataFrame,
    feature_info: FeatureInfo = audio_features.FEATURE_INFO,
) -> dict[str, Any]:
    """Mean |SHAP| (log-odds) per selected feature and per feature group, largest first."""
    shap = voice_module.shap_values(
        bundle.model, X[bundle.feature_names].to_numpy(float), bundle.feature_names
    ).drop(columns="_baseline")
    per_feature = shap.abs().mean().sort_values(ascending=False)
    groups = per_feature.groupby(lambda name: feature_info[name][0]).sum()
    return {
        "unit": "mean |SHAP| in log-odds over training recordings",
        "per_feature": {k: round(float(v), 4) for k, v in per_feature.items()},
        "per_group": {
            k: round(float(v), 4) for k, v in groups.sort_values(ascending=False).items()
        },
    }


def train_audio_model(
    table: pd.DataFrame,
    model: pipeline.Model = voice_module.DEFAULT_MODEL,
    n_ensemble: int = voice_module.N_ENSEMBLE,
    seed: int = pipeline.SEED,
    **train_kwargs: Any,
) -> AudioBundle:
    """Train on the quality-passed feature table. Age and sex are never model inputs."""
    X = table[list(audio_features.FEATURE_NAMES)]
    y = table.label.to_numpy(int)
    groups = table.person.to_numpy()
    sex = table.sex.to_numpy()
    voice, test = voice_module.train_voice_model(
        X,
        y,
        groups,
        sex,
        model=model,
        n_ensemble=n_ensemble,
        seed=seed,
        k_grid=audio_evidence.AUDIO_K_GRID,
        **train_kwargs,
    )
    train = np.setdiff1d(np.arange(len(table)), test)
    X_tr, y_tr = X.iloc[train], y[train]

    datasets = sorted(table.dataset.unique())
    bundle = AudioBundle(
        voice=voice,
        reference=feature_percentiles(X_tr, y_tr),
        global_importance=global_importance(voice, X_tr),
        datasets=datasets,
    )
    card = dict(voice.model_card)  # keeps hold-out check, calibration, CI method, fairness
    card.update(
        {
            "pipeline_version": bundle.pipeline_version,
            "input": "one sustained 'aaah' recording, any common audio format",
            "training_data": f"raw audio from {', '.join(datasets)}; 56 features from "
            "audio_features.extract_voice_features (8 kHz, quality-gated); 80% of people",
            "global_importance": bundle.global_importance,
            "evidence": "Ladder, age-matched check, dataset gate and permutation control "
            "are in models/voice/RESULTS.md (audio_evidence.py).",
            "known_limits": [
                "Trained on people already diagnosed versus healthy controls: no evidence "
                "for early or prodromal detection.",
                "Small training set; CIs are wide.",
                "Figshare healthy controls are much younger than its PD group; age and sex "
                "alone separate them better than voice does (see RESULTS.md).",
                "Telephone-band (8 kHz) features only.",
                "Does not distinguish Parkinson's disease from other conditions that change "
                "the voice.",
            ],
        }
    )
    bundle.model_card = card
    voice.model_card["pipeline_version"] = bundle.pipeline_version
    bundle.holdout_paths = sorted(table.path.iloc[test]) if "path" in table else []
    return bundle


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", type=Path, default=FEATURE_TABLE)
    ap.add_argument(
        "--model", choices=("l1_logistic", "lightgbm"), default=voice_module.DEFAULT_MODEL
    )
    ap.add_argument("--out", type=Path, default=BUNDLE_PATH)
    args = ap.parse_args(argv)

    table = audio_evidence.load_passed(args.table)
    print(f"training {args.model} on {table.person.nunique()} people ...", flush=True)
    bundle = train_audio_model(table, model=args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out)
    card_path = args.out.with_name("audio_model_card.json")
    card_path.write_text(json.dumps(bundle.model_card, indent=2) + "\n")
    holdout_path = args.out.with_name("holdout_audio_files.txt")
    holdout_path.write_text("\n".join(bundle.holdout_paths) + "\n")

    check = bundle.model_card["holdout_check"]
    print(
        f"hold-out check ({check['n_people']} people): AUC {check['auc']['value']:.3f} "
        f"[{check['auc']['lower']:.3f}, {check['auc']['upper']:.3f}]"
    )
    print("top feature groups:", list(bundle.global_importance["per_group"])[:4])
    print(f"demo files (hold-out people only): {holdout_path}")
    print(f"written to {args.out} and {card_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
