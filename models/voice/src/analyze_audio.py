"""Score uploaded audio files end to end: extract, predict, explain (task VOICE-24).

``analyze_audio(path, bundle)`` runs one file through:

1. ``audio_features.extract_voice_features``: any audio format, quality gate,
   steady vowel, 56 named features;
2. the model's own feature set: the 56 audio features, or (``--model uci``) the
   16 UCI-470 Praat measures recomputed from the same vowel
   (``uci_praat.voice_report``);
3. ``voice_module.predict_voice``: calibrated score, bootstrap-ensemble 95% CI,
   triage priority, SHAP top 5;
4. ``audio_explain.explain``: plain-language reasons, comparison with healthy and
   PD training voices, uncertainty reasons, quality report, model overview.

A file that fails the quality gate, is not audio at all, or gives a voice far
outside every training voice (``audio_explain.far_outside_training``: a forced,
sung or synthetic sound) gets an abstained result with the reasons: no score
and no triage, never a guess.

The output keeps the PP1 result-vector fields (decision D5), so the desktop
app's "Open result file" can show it. The file name is never written out (it
may contain a person's name); a result is identified by a hash of the file's
bytes.

Usage (from the repo root):
    python models/voice/src/analyze_audio.py recording.wav             # UCI-470 model
    python models/voice/src/analyze_audio.py a.mp3 b.m4a --model audio  # Figshare model
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import audio_explain
import audio_features
import audio_io
import joblib
import numpy as np
import pandas as pd
import uci_praat
import uci_transfer
import voice_module
from audio_model import ARTIFACTS, AudioBundle
from audio_model import BUNDLE_PATH as AUDIO_BUNDLE_PATH

DEFAULT_OUT = ARTIFACTS / "analysis_results.json"
BUNDLES = {"uci": uci_transfer.BUNDLE_PATH, "audio": AUDIO_BUNDLE_PATH}


def file_id(path: Path) -> str:
    """Short hash of the file's contents: identifies a result without its file name."""
    return "upload-" + hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def _model_features(
    extraction: audio_features.Extraction, bundle: AudioBundle
) -> tuple[dict[str, float] | None, list[str]]:
    """The features this bundle's model needs, or None and the reason they are missing."""
    if extraction.features is None:
        return None, extraction.quality.problems
    if bundle.feature_set == "audio":
        return extraction.features, []
    values = uci_praat.uci_features_from(extraction)
    missing = [k for k, v in values.items() if not np.isfinite(v)]
    if missing:
        return None, [f"Praat could not measure {', '.join(missing[:3])} from this recording."]
    return values, []


def analyze_audio(path: str | Path, bundle: AudioBundle) -> dict[str, Any]:
    """Result vector plus explanations for one audio file."""
    path = Path(path)
    header = {
        "person": file_id(path) if path.is_file() else "upload-missing",
        "reference_label": "-",
    }
    try:
        extraction = audio_features.extract_voice_features(path)
    except audio_io.AudioLoadError as error:
        extraction = audio_features.Extraction(
            None, audio_features.QualityReport(False, [f"The file could not be read: {error}."])
        )

    features, problems = _model_features(extraction, bundle)
    if features is not None:
        # A voice unlike every training voice gets no score, never an extrapolated guess.
        far = audio_explain.far_outside_training(features, bundle.reference, bundle.incomparable)
        problems = audio_explain.unlike_training_problem(far)
        if problems:
            features = None
    if features is None:
        quality = audio_features.QualityReport(False, problems, extraction.quality.info)
        result = voice_module._abstained(bundle.voice, problems)
    else:
        quality = extraction.quality
        result = voice_module.predict_voice(pd.DataFrame([features]), bundle.voice)
    widened = _widen_to_include_score(result)
    result["pipeline_version"] = bundle.pipeline_version
    result["explanations"] = audio_explain.explain(
        result,
        features,
        quality,
        bundle.reference,
        bundle.global_importance,
        bundle.voice.n_training,
        bundle.datasets,
        bundle.feature_info,
        bundle.incomparable,
    )
    if widened:
        result["explanations"]["uncertainty"].insert(
            0,
            "The main model and the 50 bootstrap models disagree about this voice: the main "
            "score fell outside their range, so the range was widened to include it.",
        )
    return {**header, **result}


def _widen_to_include_score(result: dict[str, Any]) -> bool:
    """Make sure ci_lower <= risk_score <= ci_upper; True if the range had to grow.

    The CI comes from a bootstrap ensemble, the score from the main model. For a
    voice unlike the training data they can disagree, and a score outside its own
    range cannot be shown honestly (the desktop app refuses it). Widening is the
    cautious fix: it never narrows the uncertainty.
    """
    score = result.get("risk_score")
    if score is None:
        return False
    lower, upper = result["ci_lower"], result["ci_upper"]
    result["ci_lower"], result["ci_upper"] = min(lower, score), max(upper, score)
    return (lower, upper) != (result["ci_lower"], result["ci_upper"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--model", choices=sorted(BUNDLES), default="uci")
    ap.add_argument("--bundle", type=Path, help="a bundle file (overrides --model)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    bundle_path = args.bundle or BUNDLES[args.model]
    if not bundle_path.is_file():
        print(f"no trained model at {bundle_path}; train it first", file=sys.stderr)
        return 1
    bundle: AudioBundle = joblib.load(bundle_path)
    results = [analyze_audio(path, bundle) for path in args.files]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")

    print(f"model: {bundle.pipeline_version}")
    for result in results:
        print(f"\n{result['person']}: {result['explanations']['summary']}")
        for reason in result["explanations"]["top_reasons"]:
            print(f"  - {reason['sentence']}")
        for reason in result["explanations"]["uncertainty"]:
            print(f"  ! {reason}")
    print(f"\nwritten to {args.out} (open it in the desktop app with 'Open result file')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
