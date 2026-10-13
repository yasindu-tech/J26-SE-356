"""Hand landmarks for one finger-tapping clip (task FT-11).

Runs the MediaPipe HandLandmarker in VIDEO running mode over every decoded frame of ONE
clip and returns one row per frame: 21 landmarks in normalised image coordinates, the same
21 in pixels of the displayed frame, and the 21 MediaPipe world landmarks.

Columns (``COLUMNS``):

- ``frame`` (0-based index of the decoded frame);
- ``timestamp_ms`` (int): ``int(round(frame * 1000 / fps))``, raised to the previous value + 1
  if rounding ever repeats it. MediaPipe needs strictly increasing integer milliseconds. The
  clip's own clock is used, never the wall clock;
- ``pts_ms`` (float): the container's presentation time of that frame (OpenCV
  ``CAP_PROP_POS_MSEC``), NaN if the container gives none. It is NOT given to MediaPipe;
- ``detected`` (bool), ``handedness_label`` and ``handedness_score`` as MediaPipe reports
  them (it assumes a mirrored image, so do not read left and right off the label);
- ``x00..x20``, ``y00..y20``, ``z00..z20``: normalised landmarks;
- ``x_px00..x_px20``, ``y_px00..y_px20``: ``x * width`` and ``y * height`` of the displayed
  frame, so distances computed in pixels are not stretched by the aspect ratio;
- ``wx00..wx20``, ``wy00..wy20``, ``wz00..wz20``: MediaPipe world landmarks.

FT-12 MUST USE ``pts_ms``, NOT ``timestamp_ms``, FOR TAP TIMING. Phone clips are not
constant frame rate: on irregular clips the real time between frames varies and the
container time can differ from ``frame / fps`` by tens of milliseconds, while ``timestamp_ms``
is only a tidy increasing counter for MediaPipe. ``summarize`` reports the largest
difference per clip.

A frame with no hand has ``detected = False`` and NaN in every landmark column. It is never
zero: zero is a real position (the top-left corner) and would read as a hand there.

Orientation: phones store many clips rotated and record the rotation as metadata. The
capture is told to auto-rotate (``CAP_PROP_ORIENTATION_AUTO = 1``), the setting is read
back, and the width and height used for pixels come from the first decoded frame, so they
are the displayed size. Frames are converted from BGR to RGB before MediaPipe sees them.

fps, width, height, library versions, the model's sha256 and every detector parameter are
stored once in the parquet file's metadata (not as columns); read them with
``read_metadata`` or ``read_clip_geometry``. Detector parameters are arguments because
task FT-18 will try lower confidence thresholds.

Clip IDs are never printed: the file stem is the raw HUBU-FIS clip ID, so messages carry
only its short hash (``hubu_fis.pseudonym``), and OpenCV's own stderr output, which prints
the full file path when a clip cannot be opened, is muted around the open call. The saved
parquet file is named after the clip ID so later steps can join it to labels; it goes under
``data/interim/landmarks/`` (gitignored) and its name is never printed.

Exit codes: 0 ok, 1 the clip could not be read or broke a rule, 2 the video or the model
file is missing.

Usage (from the repo root):
    python models/tapping/src/hand_landmarks.py --video path/to/clip.mp4
    python models/tapping/src/hand_landmarks.py --video clip.mp4 --min-hand-detection-confidence 0.3
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from clip_id import UnparseableClipId, parse_clip_id
from hubu_fis import pseudonym

N_LANDMARKS = 21
MAX_FPS = 1000.0  # integer milliseconds cannot separate frames closer than 1 ms
TOOL_VERSION = "hand-landmarks-0.1.0"
EXPECTED_MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
MODEL_HINT = "place hand_landmarker.task in models/tapping/artifacts/ (see models/tapping/README.md)"

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = REPO_ROOT / "models" / "tapping" / "artifacts" / "hand_landmarker.task"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "interim" / "landmarks"
META_KEY = b"tapping.hand_landmarks"

NORMALISED = ("x", "y", "z")
PIXEL = ("x_px", "y_px")
WORLD = ("wx", "wy", "wz")
BASE_COLUMNS = (
    "frame",
    "timestamp_ms",
    "pts_ms",
    "detected",
    "handedness_label",
    "handedness_score",
)


def landmark_column(prefix: str, index: int) -> str:
    return f"{prefix}{index:02d}"


LANDMARK_COLUMNS = tuple(
    landmark_column(prefix, i)
    for prefix in (*NORMALISED, *PIXEL, *WORLD)
    for i in range(N_LANDMARKS)
)
COLUMNS = BASE_COLUMNS + LANDMARK_COLUMNS

# Created by the caller; Any because the real objects are cv2.VideoCapture and a MediaPipe
# wrapper, and the tests pass small fakes with the same few methods.
CaptureFactory = Callable[[str], Any]
LandmarkerFactory = Callable[..., Any]


class HandLandmarksError(ValueError):
    """The clip or the detector output breaks a rule (command line exit code 1)."""


class InputMissingError(FileNotFoundError):
    """The video or the model file is missing (command line exit code 2)."""


class ClipGeometry(NamedTuple):
    fps: float
    width: int
    height: int


# --------------------------------------------------------------------------- small rules


def check_fps(fps: float) -> float:
    """The frame rate, if usable as a timestamp scale; raise otherwise."""
    if not math.isfinite(fps) or fps <= 0 or fps > MAX_FPS:
        raise HandLandmarksError(
            f"frame rate must be finite, above 0 and at most {MAX_FPS:g}, got {fps!r}"
        )
    return float(fps)


def next_timestamp_ms(index: int, fps: float, last_ms: int) -> int:
    """Integer millisecond timestamp of frame ``index``, strictly above ``last_ms``.

    ``round`` can in principle give the same value twice; ``last_ms + 1`` then keeps the
    sequence strictly increasing, which MediaPipe requires. Start with ``last_ms = -1``.
    """
    return max(int(round(index * 1000 / fps)), last_ms + 1)


def clip_ref(video_path: str | Path) -> str:
    """Short hash of the clip ID (the file stem): the only form that may be printed."""
    return pseudonym(Path(video_path).stem)


def expected_hand(video_path: str | Path) -> str | None:
    """'left' or 'right' from the clip ID (DCHA / IZDA), or None if the name is not an ID."""
    try:
        return parse_clip_id(Path(video_path).stem).hand
    except UnparseableClipId:
        return None


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


# --------------------------------------------------------------------------- default factories


@contextlib.contextmanager
def _stderr_to_devnull() -> Iterator[None]:
    """Mute file descriptor 2, which also covers messages printed by C code."""
    sys.stderr.flush()
    saved = os.dup(2)
    try:
        with open(os.devnull, "wb") as sink:
            os.dup2(sink.fileno(), 2)
            yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


def open_video(path: str) -> Any:
    """``cv2.VideoCapture`` with stderr muted: a failed open prints the full file path."""
    with _stderr_to_devnull():
        return cv2.VideoCapture(path)


class MediaPipeHandLandmarker:
    """The MediaPipe HandLandmarker in VIDEO mode, taking RGB arrays."""

    def __init__(
        self,
        model_path: Path,
        *,
        num_hands: int,
        min_hand_detection_confidence: float,
        min_hand_presence_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker,
            HandLandmarkerOptions,
            RunningMode,
        )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._mp = mp
        self._landmarker = HandLandmarker.create_from_options(options)

    def detect_for_video(self, rgb: np.ndarray, timestamp_ms: int) -> Any:
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
        )
        return self._landmarker.detect_for_video(image, timestamp_ms)

    def close(self) -> None:
        self._landmarker.close()


# --------------------------------------------------------------------------- tracking


def _parse_result(result: Any, ref: str) -> tuple[np.ndarray, np.ndarray, str | None, float] | None:
    """(normalised 21x3, world 21x3, handedness label, score) of the first hand, or None."""
    if not result.hand_landmarks:
        return None
    hand = result.hand_landmarks[0]
    if len(hand) != N_LANDMARKS:
        raise HandLandmarksError(f"clip {ref}: expected {N_LANDMARKS} landmarks, got {len(hand)}")
    normalised = np.array([[p.x, p.y, p.z] for p in hand], dtype=float)

    world = np.full((N_LANDMARKS, 3), np.nan)
    if result.hand_world_landmarks:
        world_hand = result.hand_world_landmarks[0]
        if len(world_hand) != N_LANDMARKS:
            raise HandLandmarksError(
                f"clip {ref}: expected {N_LANDMARKS} world landmarks, got {len(world_hand)}"
            )
        world = np.array([[p.x, p.y, p.z] for p in world_hand], dtype=float)

    label: str | None = None
    score = float("nan")
    if result.handedness and result.handedness[0]:
        label = result.handedness[0][0].category_name
        score = float(result.handedness[0][0].score)
    return normalised, world, label, score


def track_clip(
    video_path: str | Path,
    model_path: str | Path,
    *,
    num_hands: int = 1,
    min_hand_detection_confidence: float = 0.5,
    min_hand_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    landmarker_factory: LandmarkerFactory | None = None,
    capture_factory: CaptureFactory | None = None,
) -> pd.DataFrame:
    """Landmarks for every decoded frame of one clip: one row per frame, columns ``COLUMNS``.

    The metadata (fps, size, versions, model hash, detector parameters, frame counts) is in
    ``df.attrs["hand_landmarks"]``; ``write_landmarks`` stores it in the parquet file.
    ``capture_factory`` and ``landmarker_factory`` exist so tests can pass fakes; by default
    the real OpenCV capture and MediaPipe landmarker are used, and then the video and model
    files must exist (``InputMissingError`` otherwise).
    """
    video_path, model_path = Path(video_path), Path(model_path)
    ref = clip_ref(video_path)
    if capture_factory is None:
        if not video_path.is_file():
            raise InputMissingError(f"clip {ref}: video file not found")
        capture_factory = open_video
    if landmarker_factory is None:
        if not model_path.is_file():
            raise InputMissingError(
                f"hand landmarker model not found at {model_path}. {MODEL_HINT}"
            )
        landmarker_factory = MediaPipeHandLandmarker

    cap = capture_factory(str(video_path))
    try:
        if not cap.isOpened():
            raise HandLandmarksError(f"clip {ref}: cannot be opened as a video")
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        if cap.get(cv2.CAP_PROP_ORIENTATION_AUTO) != 1:
            raise HandLandmarksError(f"clip {ref}: auto-rotation could not be switched on")
        try:
            fps = check_fps(float(cap.get(cv2.CAP_PROP_FPS)))
        except HandLandmarksError as err:
            raise HandLandmarksError(f"clip {ref}: {err}") from err
        frames_container = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orientation_meta = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))

        detector = {
            "running_mode": "VIDEO",
            "num_hands": num_hands,
            "min_hand_detection_confidence": min_hand_detection_confidence,
            "min_hand_presence_confidence": min_hand_presence_confidence,
            "min_tracking_confidence": min_tracking_confidence,
        }
        landmarker = landmarker_factory(
            model_path,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        try:
            rows = _run_frames(cap, landmarker, fps, ref)
        finally:
            close = getattr(landmarker, "close", None)
            if close is not None:
                close()
    finally:
        cap.release()

    sha = _sha256(model_path)
    meta: dict[str, Any] = {
        "tool": TOOL_VERSION,
        "mediapipe": _version("mediapipe"),
        "opencv": cv2.__version__,
        "model_sha256": sha,
        "model_sha256_expected": EXPECTED_MODEL_SHA256,
        "model_matches_expected": None if sha is None else sha == EXPECTED_MODEL_SHA256,
        "detector": detector,
        "fps": fps,
        "width": rows.width,
        "height": rows.height,
        "orientation_auto": 1,
        "orientation_meta": orientation_meta,
        "frames_decoded": len(rows.timestamp_ms),
        "frames_container": frames_container,
        "n_landmarks": N_LANDMARKS,
        "timestamp_rule": "timestamp_ms = max(int(round(frame * 1000 / fps)), previous + 1)",
        "pts_rule": "pts_ms = container presentation time (CAP_PROP_POS_MSEC), NaN if missing",
        "pixel_rule": "x_px = x * width, y_px = y * height of the displayed (auto-rotated) frame",
    }
    df = _build_table(rows)
    df.attrs["hand_landmarks"] = meta
    return df


class _Rows:
    """What the frame loop collected."""

    def __init__(self) -> None:
        self.width = 0
        self.height = 0
        self.timestamp_ms: list[int] = []
        self.pts_ms: list[float] = []
        self.detected: list[bool] = []
        self.labels: list[str | None] = []
        self.scores: list[float] = []
        self.normalised: list[np.ndarray] = []
        self.world: list[np.ndarray] = []


def _run_frames(cap: Any, landmarker: Any, fps: float, ref: str) -> _Rows:
    rows = _Rows()
    missing = np.full((N_LANDMARKS, 3), np.nan)
    last_ms = -1
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index == 0:
            rows.height, rows.width = int(frame.shape[0]), int(frame.shape[1])
            reported = (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            if reported != (rows.width, rows.height):
                raise HandLandmarksError(
                    f"clip {ref}: first frame is {rows.width}x{rows.height} but OpenCV reports "
                    f"{reported[0]}x{reported[1]}; the rotation may not have been applied"
                )
        elif frame.shape[:2] != (rows.height, rows.width):
            raise HandLandmarksError(f"clip {ref}: the frame size changed at frame {index}")

        pts = float(cap.get(cv2.CAP_PROP_POS_MSEC))
        rows.pts_ms.append(round(pts, 3) if math.isfinite(pts) and pts >= 0 else float("nan"))
        last_ms = next_timestamp_ms(index, fps, last_ms)
        rows.timestamp_ms.append(last_ms)

        found = _parse_result(
            landmarker.detect_for_video(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), last_ms), ref
        )
        if found is None:
            rows.detected.append(False)
            rows.labels.append(None)
            rows.scores.append(float("nan"))
            rows.normalised.append(missing)
            rows.world.append(missing)
        else:
            normalised, world, label, score = found
            rows.detected.append(True)
            rows.labels.append(label)
            rows.scores.append(score)
            rows.normalised.append(normalised)
            rows.world.append(world)
        index += 1
    if index == 0:
        raise HandLandmarksError(f"clip {ref}: no frames could be decoded")
    return rows


def _build_table(rows: _Rows) -> pd.DataFrame:
    n = len(rows.timestamp_ms)
    norm = np.stack(rows.normalised)  # (n, 21, 3)
    world = np.stack(rows.world)
    data: dict[str, Any] = {
        "frame": np.arange(n, dtype=np.int64),
        "timestamp_ms": np.array(rows.timestamp_ms, dtype=np.int64),
        "pts_ms": np.array(rows.pts_ms, dtype=float),
        "detected": np.array(rows.detected, dtype=bool),
        "handedness_label": pd.Series(rows.labels, dtype=object),
        "handedness_score": np.array(rows.scores, dtype=float),
    }
    for i in range(N_LANDMARKS):
        for k, prefix in enumerate(NORMALISED):
            data[landmark_column(prefix, i)] = norm[:, i, k]
        data[landmark_column("x_px", i)] = norm[:, i, 0] * rows.width
        data[landmark_column("y_px", i)] = norm[:, i, 1] * rows.height
        for k, prefix in enumerate(WORLD):
            data[landmark_column(prefix, i)] = world[:, i, k]
    return pd.DataFrame(data, columns=list(COLUMNS))


# --------------------------------------------------------------------------- parquet


def output_path(video_path: str | Path, out_dir: str | Path) -> Path:
    """``<out_dir>/<clip ID>.parquet``. The name holds the raw ID: never print it."""
    return Path(out_dir) / f"{Path(video_path).stem}.parquet"


def write_landmarks(df: pd.DataFrame, path: str | Path) -> None:
    """Write the table, with the clip metadata stored once in the file's metadata."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[META_KEY] = json.dumps(df.attrs["hand_landmarks"], sort_keys=True).encode()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(metadata), path)


def read_metadata(path: str | Path) -> dict[str, Any]:
    """The metadata stored by ``write_landmarks``."""
    raw = (pq.read_schema(path).metadata or {}).get(META_KEY)
    if raw is None:
        raise HandLandmarksError("this parquet file has no hand_landmarks metadata")
    meta: dict[str, Any] = json.loads(raw)
    return meta


def read_clip_geometry(path: str | Path) -> ClipGeometry:
    """fps, width and height of the clip, read back from the parquet metadata."""
    meta = read_metadata(path)
    return ClipGeometry(
        fps=float(meta["fps"]), width=int(meta["width"]), height=int(meta["height"])
    )


def read_landmarks(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The table and its metadata."""
    return pd.read_parquet(path), read_metadata(path)


# --------------------------------------------------------------------------- summary


def summarize(
    df: pd.DataFrame, meta: dict[str, Any], expected: str | None = None
) -> dict[str, Any]:
    """Numbers that show whether a clip was tracked sensibly. No IDs."""
    detected = df["detected"].to_numpy(dtype=bool)
    n = len(df)
    x_cols = [landmark_column("x", i) for i in range(N_LANDMARKS)]
    y_cols = [landmark_column("y", i) for i in range(N_LANDMARKS)]
    on = df.loc[detected]
    per_frame = on[x_cols].notna().sum(axis=1).to_numpy() if len(on) else np.array([0])
    xy = on[[*x_cols, *y_cols]].to_numpy(dtype=float)
    outside = (xy < 0) | (xy > 1)
    off_values = df.loc[~detected, list(LANDMARK_COLUMNS)].to_numpy(dtype=float)

    pts = df["pts_ms"].to_numpy(dtype=float)
    nominal = df["frame"].to_numpy(dtype=float) * 1000.0 / float(meta["fps"])
    finite = np.isfinite(pts)
    labels = on["handedness_label"].tolist()
    expected_label = None if expected is None else expected.capitalize()
    return {
        "frames": n,
        "frames_container": meta["frames_container"],
        "fps": meta["fps"],
        "width": meta["width"],
        "height": meta["height"],
        "detected_frames": int(detected.sum()),
        "detection_rate": float(detected.mean()),
        "landmarks_per_detected_frame_min": int(per_frame.min()),
        "landmarks_per_detected_frame_max": int(per_frame.max()),
        "undetected_rows_with_zero_values": int((off_values == 0).sum()),
        "undetected_rows_not_nan": int((~np.isnan(off_values)).sum()),
        "timestamp_ms_strictly_increasing": bool((np.diff(df["timestamp_ms"]) > 0).all()),
        "pts_ms_strictly_increasing": bool((np.diff(pts[finite]) > 0).all()),
        "pts_ms_missing": int((~finite).sum()),
        "max_abs_pts_minus_index_over_fps_ms": float(np.abs(pts - nominal)[finite].max())
        if finite.any()
        else float("nan"),
        "landmark_values_outside_0_1": int(outside.sum()),
        "frames_with_any_value_outside_0_1": int(outside.any(axis=1).sum()),
        "handedness_counts": {k: labels.count(k) for k in sorted(set(labels))},
        "handedness_switches": sum(a != b for a, b in zip(labels, labels[1:], strict=False)),
        "expected_label": expected_label,
        "frames_label_equals_expected": None
        if expected_label is None
        else labels.count(expected_label),
    }


def format_summary(ref: str, s: dict[str, Any]) -> str:
    yes = {True: "yes", False: "NO"}
    counts = ", ".join(f"{k} {v}" for k, v in s["handedness_counts"].items()) or "none"
    expected = (
        "not known from the file name"
        if s["expected_label"] is None
        else f"{s['frames_label_equals_expected']} frames equal the clip's nominal hand "
        f"({s['expected_label']}); MediaPipe assumes a mirrored image, so this is reported "
        "as observed"
    )
    return "\n".join(
        [
            f"clip {ref}: {s['frames']} frames decoded (container says {s['frames_container']}), "
            f"{s['fps']:.4f} fps, displayed size {s['width']}x{s['height']}",
            f"  detection: {s['detected_frames']}/{s['frames']} frames "
            f"({s['detection_rate']:.1%}); landmarks per detected frame "
            f"{s['landmarks_per_detected_frame_min']}..{s['landmarks_per_detected_frame_max']}",
            f"  undetected rows: {s['undetected_rows_with_zero_values']} zero values, "
            f"{s['undetected_rows_not_nan']} non-NaN values (both must be 0)",
            f"  timestamp_ms strictly increasing: {yes[s['timestamp_ms_strictly_increasing']]}; "
            f"pts_ms strictly increasing: {yes[s['pts_ms_strictly_increasing']]} "
            f"({s['pts_ms_missing']} missing); "
            f"max |pts_ms - frame/fps| = {s['max_abs_pts_minus_index_over_fps_ms']:.1f} ms",
            f"  x/y outside [0, 1]: {s['landmark_values_outside_0_1']} values in "
            f"{s['frames_with_any_value_outside_0_1']} detected frames",
            f"  handedness as reported: {counts}; label switches {s['handedness_switches']}; "
            f"{expected}",
        ]
    )


# --------------------------------------------------------------------------- command line


def main(
    argv: Sequence[str] | None = None,
    *,
    capture_factory: CaptureFactory | None = None,
    landmarker_factory: LandmarkerFactory | None = None,
) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--video", type=Path, required=True, help="one clip")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="hand_landmarker.task")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="where the parquet goes")
    ap.add_argument("--num-hands", type=int, default=1)
    ap.add_argument("--min-hand-detection-confidence", type=float, default=0.5)
    ap.add_argument("--min-hand-presence-confidence", type=float, default=0.5)
    ap.add_argument("--min-tracking-confidence", type=float, default=0.5)
    args = ap.parse_args(argv)

    ref = clip_ref(args.video)
    if not args.video.is_file():
        print(f"error: clip {ref}: video file not found", file=sys.stderr)
        return 2
    if landmarker_factory is None and not args.model.is_file():
        print(
            f"error: hand landmarker model not found at {args.model}. "
            f"{MODEL_HINT}",
            file=sys.stderr,
        )
        return 2

    try:
        df = track_clip(
            args.video,
            args.model,
            num_hands=args.num_hands,
            min_hand_detection_confidence=args.min_hand_detection_confidence,
            min_hand_presence_confidence=args.min_hand_presence_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            landmarker_factory=landmarker_factory,
            capture_factory=capture_factory,
        )
    except InputMissingError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except HandLandmarksError as err:
        print(f"FAILED: {err}", file=sys.stderr)
        return 1

    meta = df.attrs["hand_landmarks"]
    write_landmarks(df, output_path(args.video, args.out_dir))
    print(format_summary(ref, summarize(df, meta, expected_hand(args.video))))
    print(f"wrote 1 parquet file to {args.out_dir} (named by clip ID, not shown)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
