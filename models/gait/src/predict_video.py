"""GAIT-27: a walking video in, one result out: a score with its explanation, or a rejection.

Chains ``video_to_joints``, the quality gate, ``extract_features``, the saved model bundle and the
plain-words SHAP sentences. A video that cannot be scored gets ``available: false`` and its reasons;
nothing is raised for a bad video. Probabilities are uncalibrated model scores. One person in view
is assumed unless a count is passed or the person who uploaded it confirmed it.

    python models/gait/src/predict_video.py VIDEO [--people N] [--json] [--out result.json]

Exit code 0 = scored, 1 = not scored, 2 = the model bundle or pose model cannot be used.
"""

from __future__ import annotations

import argparse
import json
from functools import cache
from pathlib import Path
from typing import Any

from carepd_format import short_id
from explain_shap import DEFAULT_BUNDLE, Explainer
from feature_contract import FeatureContractError, check_feature_contract
from gait_extract import FEATURE_NAMES, MAX_FPS, MIN_FPS, extract_features
from make_folds import CLASS_LABELS
from model_bundle import BundleError, load_bundle
from quality_gate import ONE_PERSON_ASSUMED, GateResult, check_walk, reject
from video_to_joints import DEFAULT_MODEL, MAX_SECONDS, VideoError, VideoWalk, video_to_joints

SCHEMA_VERSION = "gait-result/1"
LABEL_NAMES = ["UPDRS_GAIT", "severity", "score"]  # never model inputs (CLAUDE.md 3.1)
ONE_PERSON_CONFIRMED = (
    "You confirmed one person is walking. The video itself was not checked for other people."
)
SEVERITY_NOTE = "A summary of the three model scores on a 0 to 2 scale; not a UPDRS score."


def plain_reason(reason: str) -> str:
    """The feature step's reason as a sentence for the person who uploaded the video."""
    if reason.startswith("real strides on one foot only"):
        return "Real strides were found on only one foot, so both feet could not be timed."
    if reason.startswith("leg length"):
        return f"The leg length measured from the video is not plausible ({reason})."
    return f"The walk could not be turned into the model's inputs ({reason})."


class GaitPredictor:
    """The loaded model, explainer and settings. Build it once, then call ``predict`` per video."""

    def __init__(
        self,
        bundle_dir: Path = DEFAULT_BUNDLE,
        pose_model: Path = DEFAULT_MODEL,
        max_seconds: float | None = MAX_SECONDS,
    ) -> None:
        if not Path(pose_model).is_file():
            raise FileNotFoundError(f"pose model not found: {pose_model}")
        self.bundle = load_bundle(Path(bundle_dir))
        self.bundle.reference  # noqa: B018 - refuses a bundle without the plain-words reference
        check_feature_contract(self.bundle.features, label_name=LABEL_NAMES)  # before any scoring
        self.explainer = Explainer(self.bundle)
        self.pose_model = Path(pose_model)
        self.max_seconds = max_seconds

    def predict(
        self,
        path: Path,
        *,
        people_in_view: int | None = None,
        one_person_confirmed: bool = False,
    ) -> dict[str, Any]:
        """The result record of one video (JSON-safe).

        ``people_in_view`` is a detector's count; ``one_person_confirmed`` says the person who
        uploaded the video confirmed it shows one person walking (it changes the warning only).
        """
        record = self._predict(Path(path), people_in_view)
        if one_person_confirmed:
            record["warnings"] = [
                ONE_PERSON_CONFIRMED if w == ONE_PERSON_ASSUMED else w for w in record["warnings"]
            ]
        return {
            "modality": "gait",
            "schema_version": SCHEMA_VERSION,
            **record,
            "one_person_confirmed": one_person_confirmed,
        }

    def _predict(self, path: Path, people_in_view: int | None) -> dict[str, Any]:
        tag = short_id(Path(path).name)  # never the file name
        try:
            walk = video_to_joints(Path(path), self.pose_model, max_seconds=self.max_seconds)
        except VideoError as err:
            return self._not_scored(tag, reject(err), None, [])
        gate = check_walk(walk, people_in_view=people_in_view)
        if not gate.passed:
            return self._not_scored(tag, gate, walk, [])
        try:
            found = extract_features(walk.joints, walk.fps)
        except ValueError:  # the gate allows frame rates the model's inputs were not made for
            reason = f"The video has {walk.fps:.0f} frames per second; {MIN_FPS:g} to {MAX_FPS:g} are supported."
            return self._not_scored(tag, gate, walk, [reason])
        if not found.ok:
            return self._not_scored(tag, gate, walk, [plain_reason(found.reason)])

        x = found.vector()
        words = self.explainer.in_words(x)
        proba = [float(p) for p in words.probabilities]
        return {
            "available": True,
            "video": tag,
            "reasons": [],
            "warnings": [*gate.warnings, *found.flags],
            "gate": gate.as_record(),
            "walk": _walk_record(walk),
            "predicted_class": words.predicted_class,
            "predicted_label": CLASS_LABELS[words.predicted_class],
            "class_labels": [CLASS_LABELS[c] for c in range(3)],
            "probabilities": proba,
            "probabilities_calibrated": False,
            "expected_severity": proba[1] + 2 * proba[2],
            "expected_severity_note": SEVERITY_NOTE,
            "top_features": [e.as_record() for e in words.entries],
            "feature_vector": {n: float(v) for n, v in zip(FEATURE_NAMES, x, strict=True)},
            "model_version": self.bundle.version,
        }

    def _not_scored(
        self, tag: str, gate: GateResult, walk: VideoWalk | None, extra: list[str]
    ) -> dict[str, Any]:
        return {
            "available": False,
            "video": tag,
            "reasons": [*gate.reasons, *extra],
            "warnings": list(gate.warnings),
            "gate": gate.as_record(),
            "walk": _walk_record(walk) if walk else None,
            "predicted_class": None,
            "predicted_label": None,
            "class_labels": [CLASS_LABELS[c] for c in range(3)],
            "probabilities": None,
            "probabilities_calibrated": False,
            "expected_severity": None,
            "expected_severity_note": SEVERITY_NOTE,
            "top_features": [],
            "feature_vector": None,
            "model_version": self.bundle.version,
        }


def _walk_record(walk: VideoWalk) -> dict[str, float]:
    return {
        "seconds": float(walk.seconds),
        "start_s": float(walk.start_s),
        "end_s": float(walk.end_s),
        "fps": float(walk.fps),
    }


@cache
def default_predictor() -> GaitPredictor:
    return GaitPredictor()


def predict_video(
    path: Path, *, people_in_view: int | None = None, one_person_confirmed: bool = False
) -> dict[str, Any]:
    """``GaitPredictor.predict`` with the default bundle and pose model, loaded once."""
    return default_predictor().predict(
        path, people_in_view=people_in_view, one_person_confirmed=one_person_confirmed
    )


def show(record: dict[str, Any]) -> str:
    if not record["available"]:
        lines = [f"{record['video']}: NOT SCORED"]
        lines += [f"  reason: {r}" for r in record["reasons"]]
    else:
        walk = record["walk"]
        scores = "  ".join(
            f"{label} {p:.2f}"
            for label, p in zip(record["class_labels"], record["probabilities"], strict=True)
        )
        lines = [
            f"{record['video']}: SCORED (walk {walk['seconds']:.1f} s, "
            f"{walk['start_s']:.1f} to {walk['end_s']:.1f} s)",
            f"  model's pick: {record['predicted_label']}",
            f"  model scores (uncalibrated): {scores}",
            f"  expected severity {record['expected_severity']:.2f} of 2. {SEVERITY_NOTE}",
        ]
        lines += [f"  - {e['sentence']}" for e in record["top_features"]]
    lines += [f"  warning: {w}" for w in record["warnings"]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score one walking video (GAIT-27)")
    parser.add_argument("video", type=Path)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    parser.add_argument("--people", type=int, help="people in view, if a detector counted them")
    parser.add_argument("--json", action="store_true", help="print the whole record as JSON")
    parser.add_argument("--out", type=Path, help="also write the record to this JSON file")
    parser.add_argument(
        "--one-person-confirmed",
        action="store_true",
        help="the person who uploaded the video confirmed it shows one person walking",
    )
    args = parser.parse_args(argv)
    try:
        predictor = GaitPredictor(args.bundle, args.model, args.max_seconds)
    except (BundleError, FileNotFoundError, FeatureContractError) as err:
        print(f"Cannot start: {err}")
        return 2
    record = predictor.predict(
        args.video, people_in_view=args.people, one_person_confirmed=args.one_person_confirmed
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2) if args.json else show(record))
    return 0 if record["available"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
