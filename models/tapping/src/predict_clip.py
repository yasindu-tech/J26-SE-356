"""One finger-tapping video to one JSON result (task FT-63), for the desktop app.
`python predict_clip.py --video <path> --target ge2` runs the SAME functions as the Streamlit
demo (upload_pipeline, scoring, tap_curve, summary_pdf) and writes one "tapping-result/1" object
to stdout. Progress goes to stderr as "STAGE n/N label". No file name, path or label is written.
The models come from the installed bundle (data/models/tapping/CURRENT, tapping_bundle.py) when
there is one, every file hash-checked; otherwise from data/processed on the training laptop.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "demo"))

import cv2  # noqa: E402
import data_access as da  # noqa: E402
import numpy as np  # noqa: E402
import resources as res  # noqa: E402
import scoring as sc  # noqa: E402
import summary_pdf as pdf  # noqa: E402
import tap_curve as tc  # noqa: E402
import tapping_bundle as tb  # noqa: E402
import texts as tx  # noqa: E402
import upload_pipeline as up  # noqa: E402

SCHEMA = "tapping-result/1"
N_STAGES = 5
MAX_POINTS = 600
PREVIEW_WIDTH = 320
JPEG_QUALITY = 70
DECIMALS = 6
STAGES = (
    "Reading the video",
    "Tracking the hand and finding taps",
    "Scoring the clip",
    "Explaining the score",
    "Building the summary",
)
GENERIC = "The analysis could not be completed for this video."
MODELS_MISSING = "The research models for the score are not available on this computer."
MODELS_CHANGED = "The research models on this computer failed their integrity check."
PATHLIKE = re.compile(r"[/\\]|\.(?:mp4|mov|m4v|task|csv|txt|json|parquet)\b", re.I)

Progress = Callable[[int, str], None]


@dataclass(frozen=True)
class Models:
    """What scoring one clip needs, from the bundle or from data/processed."""

    model: Any
    folds: list[Any]
    context: Callable[[dict[str, float]], dict[str, Any]]
    auroc: dict[str, float]
    model_sha8: str
    folds_sha8: str
    input_sha8: list[str]


def bundle_models(bundle: tb.TappingBundle, target: str) -> Models:
    return Models(
        model=bundle.model(target), folds=bundle.folds(target),
        context=lambda f: tb.context(f, bundle.reference), auroc=bundle.auroc(target),
        model_sha8=bundle.model_sha8(target), folds_sha8=bundle.folds_sha8(target),
        input_sha8=[bundle.folds_sha8(target), bundle.features_sha8(target)],
    )  # fmt: skip


def processed_models(target: str) -> Models:
    reference = sc.load_reference()
    folds_sha8 = res.fold_sha8(target)
    features = da.cvs.file_sha256(da.processed_root() / "features.csv")[: da.cvs.HASH_LEN]
    return Models(
        model=res.load_demo_model(target), folds=res.load_fold_models(target),
        context=lambda f: sc.context(f, reference), auroc=sc.cv_auroc(target),
        model_sha8=res.model_sha8(target), folds_sha8=folds_sha8,
        input_sha8=[x for x in (folds_sha8, features) if x],
    )  # fmt: skip


def num(value: Any) -> float | None:
    """A finite number rounded to a fixed length, or None (written as null, never as 0)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return round(v, DECIMALS) if np.isfinite(v) else None


def nums(values: Any) -> list[float | None]:
    return [num(v) for v in np.asarray(values, dtype=float).ravel()]


def thin(n: int, limit: int = MAX_POINTS) -> np.ndarray:
    """Evenly spaced indices, at most ``limit``, always including the first and last."""
    if n <= limit:
        return np.arange(n)
    return np.unique(np.round(np.linspace(0, n - 1, limit)).astype(int))


def safe_reason(err: Exception) -> str:
    """The message of a known error when it is plain English without a path; else a fixed text."""
    if isinstance(err, tb.BundleError):
        return MODELS_CHANGED if tb.installed(tb.bundle_root()) else MODELS_MISSING
    if isinstance(err, up.UploadError | sc.ScoreError):
        text = str(err)
    elif isinstance(err, da.DemoDataError):
        text = MODELS_MISSING
    else:
        return GENERIC
    return GENERIC if PATHLIKE.search(text) or not text else text


def failure(reason: str, target: str, ref: str | None) -> dict[str, Any]:
    return {"schema_version": SCHEMA, "status": "failed", "reason": reason, "clip_ref": ref,
            "target": target}  # fmt: skip


def jpeg_b64(rgb: np.ndarray) -> str:
    h, w = rgb.shape[:2]
    if w > PREVIEW_WIDTH:
        rgb = cv2.resize(rgb, (PREVIEW_WIDTH, max(1, round(h * PREVIEW_WIDTH / w))),
                         interpolation=cv2.INTER_AREA)  # fmt: skip
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])  # fmt: skip
    if not ok:
        raise up.UploadError("A preview frame could not be encoded.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def quality_part(r: up.UploadResult, notes: list[str]) -> dict[str, Any]:
    return {
        "flags": [{"name": n, "text": why} for n, why in r.qc_flags],
        "notes": [n for n in notes if not n.startswith("Automatic check")],
    }


def build_result(
    data: bytes, suffix: str, target: str, factory: Any, models: Models, today: str,
    progress: Progress,
) -> dict[str, Any]:  # fmt: skip
    """The result object of one video; raises the demo's own errors, which `run` turns into a
    failed result."""
    progress(2, STAGES[1])
    analysis = up.analyse_upload(data, suffix, factory)
    r = analysis.result
    progress(3, STAGES[2])
    features = sc.upload_features(r)
    result = sc.score(features, models.model, models.folds)
    progress(4, STAGES[3])
    explained = sc.explain(features, models.model)
    rows = sc.contributions(features, explained["shap"], models.context(features))
    sentences = sc.wording(rows)
    auroc = models.auroc
    progress(5, STAGES[4])
    curve = tc.curve_for_upload(r)
    title = f"Contributions to the model score (clip {r.upload_ref})"
    outside = [sc.PLAIN[x["feature"]][0] for x in rows if not x["in_range"]]
    count_note = tc.count_sentence(curve) if (curve.n_edge or curve.n_nan) else None
    notes = pdf.quality_notes(
        mode="upload", target=target, fps=r.fps, tracked_share=r.tracked_share,
        qc_flags=list(r.qc_flags), outside=outside, taps_note=count_note,
        curve_reason=None if curve.usable else curve.reason,
    )  # fmt: skip
    side = "above" if result["score"] >= sc.THRESHOLD else "below"
    band_text = f"(range {result['band_low']:.2f}-{result['band_high']:.2f}) {tx.BAND_UPLOAD}"
    summary = pdf.SummaryData(
        ref=r.upload_ref, mode="upload", target=target, date_text=today,
        recording={"duration_s": r.duration_s, "fps": r.fps, "tracked_share": r.tracked_share,
                   "n_found": curve.n_found, "n_shown": curve.n_taps},
        qc_flags=list(r.qc_flags), notes=notes, score=result["score"], band_text=band_text,
        threshold_text=f"Fixed study threshold {sc.THRESHOLD}: {side}",
        auroc_text=tx.SCORE_CAPTION.format(
            est=auroc["estimate"], lo=auroc["ci_low"], hi=auroc["ci_high"], target=target),
        source_text=tx.TRAINED_ALL, sentences=sentences,
        shap_png=up.figure_png(sc.shap_figure(rows, title, compact=True)),
        signal_png=up.figure_png(up.signal_figure(r, compact=True)),
        curve_png=up.figure_png(tc.curve_figure(curve, r.upload_ref, True)),
        curve_count=tc.count_sentence(curve), curve_slope=tc.slope_sentence(curve),
        model_sha8=models.model_sha8, input_sha8=models.input_sha8,
    )  # fmt: skip
    pdf_bytes = pdf.build_pdf(summary)
    s = thin(len(r.t))
    p = thin(len(r.peak_t))
    return {
        "schema_version": SCHEMA, "status": "scored", "clip_ref": r.upload_ref, "target": target,
        "recording": {"duration_s": num(r.duration_s), "fps": num(r.fps),
                      "hand_found_pct": num(r.tracked_share * 100), "n_taps": int(r.n_taps)},
        "quality": quality_part(r, notes),
        "score": {"value": num(result["score"]), "band_low": num(result["band_low"]),
                  "band_high": num(result["band_high"]), "band_basis": tx.BAND_UPLOAD,
                  "threshold": sc.THRESHOLD, "above_threshold": bool(result["score"] >= sc.THRESHOLD),
                  "cv_auroc": num(auroc["estimate"]), "cv_ci_low": num(auroc["ci_low"]),
                  "cv_ci_high": num(auroc["ci_high"])},
        "contributions": [
            {"feature_label": sc.PLAIN[x["feature"]][0], "value_text": sc.value_text(x),
             "percentile_text": sc.percentile_text(x), "shap": num(x["shap"]), "sentence": sent}
            for x, sent in zip(rows, sentences, strict=True)
        ],
        "curve": {"times": nums(curve.times), "amps": nums(curve.amps),
                  "line_x": nums(curve.line_x), "line_y": nums(curve.line_y),
                  "slope_sentence": tc.slope_sentence(curve),
                  "count_sentence": tc.count_sentence(curve), "note": tc.NOTE,
                  "usable": bool(curve.usable)},
        "signal": {"t": nums(r.t[s]), "smoothed": nums(r.smoothed[s]),
                   "peak_t": nums(r.peak_t[p]), "peak_v": nums(r.peak_v[p])},
        "preview_frames": [{"t": num(t), "jpeg_base64": jpeg_b64(f)}
                           for t, f in zip(r.preview_times, analysis.frames, strict=True)],
        "model": {"demo_model_sha8": models.model_sha8, "folds_sha8": models.folds_sha8},
        "pdf_base64": base64.b64encode(pdf_bytes).decode("ascii"),
    }  # fmt: skip


def run(video: Path, target: str, progress: Progress, today: str | None = None) -> dict[str, Any]:
    """Never raises for a bad video or missing models: those give a failed result."""
    ref: str | None = None
    try:
        progress(1, STAGES[0])
        data = Path(video).read_bytes()
        ref = hashlib.sha256(data).hexdigest()[:8]
        suffix = Path(video).suffix.lower()
        if suffix not in up.SUFFIXES:
            raise up.UploadError("Use an mp4, mov or m4v video.")
        if len(data) > up.MAX_BYTES:
            raise up.UploadError(f"The video is larger than {up.MAX_BYTES // (1024 * 1024)} MB.")
        root = tb.bundle_root()
        if tb.installed(root):  # a broken bundle fails here; it never falls back to data/
            bundle = tb.load(root)
            factory, models = res.make_landmarker(bundle.landmarker), bundle_models(bundle, target)
        else:
            factory, models = res.make_landmarker(), processed_models(target)
        return build_result(
            data, suffix, target, factory, models, today or datetime.date.today().isoformat(),
            progress,
        )  # fmt: skip
    except OSError:
        return failure("The video could not be read.", target, ref)
    except Exception as err:  # noqa: BLE001  (the reason is fixed text, never the error itself)
        return failure(safe_reason(err), target, ref)


def to_json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One tapping video to one JSON result.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--target", choices=sc.da.TARGETS, default="ge2")
    args = parser.parse_args(argv)

    def progress(n: int, label: str) -> None:
        print(f"STAGE {n}/{N_STAGES} {label}", file=sys.stderr, flush=True)

    real_stdout = sys.stdout
    sys.stdout = sys.stderr  # nothing a library prints may reach the JSON channel
    try:
        out = run(args.video, args.target, progress)
    finally:
        sys.stdout = real_stdout
    real_stdout.write(to_json(out) + "\n")
    real_stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
