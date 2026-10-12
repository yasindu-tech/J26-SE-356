"""Batch hand tracking with a per-clip manifest and confidence-gating FLAGS (task FT-15).

Runs ``hand_landmarks.track_clip`` over a folder of clips, one parquet per clip (same schema as
FT-11), and writes ``tracking_manifest.csv``: one row per clip with what was tracked, how well,
and which gating flags it raised. The full run over all 234 clips is FT-16; FT-17 builds the
detection log from the manifest; FT-18 re-runs low-detection clips with other parameters.

Rules:

- **Flags, never deletions.** Gating is a set of clip-level flags in the manifest. No clip is
  dropped, skipped or deleted because of a flag, and a flag never changes the exit code. The
  thresholds are parameters; the values used are written into every manifest row. A value
  exactly AT a threshold is not flagged.
- **MediaPipe already gates each frame** with its three confidence thresholds (palm detection,
  hand presence, tracking). A frame that fails becomes a gap (NaN, ``detected = False``). No
  extra per-frame gate is added, and the handedness label and score are never used (the label
  flickers on correctly tracked hands; the hand comes from the clip name).
- **No label is used anywhere.** Group, hand, resolution and orientation are recorded only so
  the retention audit (FT-17) can check for bias. Nothing here depends on UPDRS or on
  CONTROL/ID.
- **Missing is NaN, never zero; no smoothing and no gap filling** (FT-19). ``pts_ms`` is the
  timing truth.
- **Hashed references only** (``hubu_fis.pseudonym``) in every print, log line, error and
  manifest status text. The raw clip ID appears only as the file name of the parquet and in the
  manifest's ``clip_id`` column, both under gitignored ``data/``.
- The normalised thumb-index series is NOT stored: it is deterministic from the landmarks plus
  the scale metadata in the manifest, and FT-19 owns the signal module.

Resume. A parquet is up to date only if its tool version, mediapipe and opencv versions, model
sha256, every detector parameter and the video's fingerprint (size plus sha256 of the first and
last MiB) all equal the current ones. Files from before FT-15 have no fingerprint and are
recomputed once. If the output folder holds files made with a DIFFERENT parameter set, the run
is refused (exit 2) and says how many; ``--force`` recomputes everything selected. Use one
``--out-dir`` per parameter set (FT-18). One runner per folder at a time.

Writes are atomic (temporary file, then rename). A killed run leaves no partial parquet; its
temporary files are removed at the next start. A clip that fails is recorded as ``failed`` with
a reason free of the raw ID, and the batch continues.

Exit codes: 0 every selected clip done, 1 some clips failed, 2 an input is missing, the output
folder is outside ``data/interim``, or the folder mixes parameter sets (refused), 130 interrupted.

Usage (from the repo root):
    python models/tapping/src/tracking.py --dry-run
    python models/tapping/src/tracking.py --workers 4
    python models/tapping/src/tracking.py --clips 245fbb13,5f05706d --out-dir data/interim/x
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import multiprocessing
import os
import signal
import sys
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import hand_landmarks as hl
import hand_scale as hs
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from clip_id import UnparseableClipId, parse_clip_id
from hubu_fis import pseudonym
from plot_distance import count_open_close_cycles, thumb_index_distance_px, time_axis_s

RUNNER_VERSION = "tracking-0.1.0"
REPO_ROOT = Path(__file__).resolve().parents[3]
INTERIM_ROOT = REPO_ROOT / "data" / "interim"
DEFAULT_VIDEO_DIR = REPO_ROOT / "data" / "raw" / "HUBU-FIS" / "videos"
DEFAULT_OUT_DIR = hl.DEFAULT_OUT_DIR
DEFAULT_MANIFEST = INTERIM_ROOT / "tracking_manifest.csv"
MANIFEST_NAME = "tracking_manifest.csv"
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi"}
FINGERPRINT_BYTES = 1024 * 1024
TMP_MARKER = ".tmp-"

# --------------------------------------------------------------------------- parameters


@dataclass(frozen=True)
class Detector:
    """The MediaPipe parameters; they define a parameter set together with the model."""

    num_hands: int = 1
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    def as_metadata(self) -> dict[str, Any]:
        """The dict ``track_clip`` stores under ``detector``."""
        return {"running_mode": "VIDEO", **dataclasses.asdict(self)}


@dataclass(frozen=True)
class Gating:
    """Clip-level flag thresholds. A value exactly at a threshold is not flagged."""

    min_detection_rate: float = 0.90  # flag low_detection below this share of frames
    fillable_gap_frames: int = 5  # FT-19 will interpolate gaps of up to this many frames
    max_unfillable_share: float = 0.05  # flag gaps_not_fillable above this share of frames
    min_frames: int = 150  # flag short_clip below this many decoded frames
    max_frame_mismatch: float = 0.02  # flag frame_count_mismatch above this relative gap
    max_out_of_frame_share: float = 0.05  # flag landmarks_out_of_frame above this share
    max_coupling_r: float = 0.5  # flag scale_tap_coupling above this |r| (FT-13)
    irregular_pts_gap_ms: float = 50.0  # info flag irregular_timing above this frame gap


@dataclass(frozen=True)
class Expected:
    """What an up-to-date parquet must have been made with."""

    tool: str
    mediapipe: str | None
    opencv: str
    model_sha256: str | None
    detector: dict[str, Any]

    def parameter_set(self) -> tuple[str, ...]:
        return parameter_set({**dataclasses.asdict(self), "orientation_auto": 1})


def parameter_set(meta: dict[str, Any]) -> tuple[str, ...]:
    """Everything except the video that a parquet's values depend on, as a comparable key."""
    return (
        str(meta.get("tool")),
        str(meta.get("mediapipe")),
        str(meta.get("opencv")),
        str(meta.get("model_sha256")),
        json.dumps(meta.get("detector"), sort_keys=True),
        str(meta.get("orientation_auto")),
    )


def make_expected(detector: Detector, model_sha256: str | None) -> Expected:
    return Expected(
        tool=hl.TOOL_VERSION,
        mediapipe=hl._version("mediapipe"),
        opencv=cv2.__version__,
        model_sha256=model_sha256,
        detector=detector.as_metadata(),
    )


# --------------------------------------------------------------------------- files and resume


def video_fingerprint(path: Path) -> dict[str, Any]:
    """Cheap content fingerprint: size and sha256 of the first and the last MiB."""
    size = path.stat().st_size
    with path.open("rb") as f:
        head = f.read(FINGERPRINT_BYTES)
        f.seek(max(size - FINGERPRINT_BYTES, 0))
        tail = f.read(FINGERPRINT_BYTES)
    return {
        "size": size,
        "head_sha256": hashlib.sha256(head).hexdigest(),
        "tail_sha256": hashlib.sha256(tail).hexdigest(),
    }


def find_videos(video_dir: Path) -> list[Path]:
    """Video files in ``video_dir`` sorted by name; two files with one clip ID are an error."""
    videos = sorted(
        p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
    )
    counts = Counter(p.stem for p in videos)
    repeated = sorted(pseudonym(stem) for stem, n in counts.items() if n > 1)
    if repeated:
        raise hl.HandLandmarksError(f"two video files share a clip ID: {', '.join(repeated)}")
    return videos


def stale_reason(path: Path, expected: Expected, fingerprint: dict[str, Any]) -> str | None:
    """Why the parquet at ``path`` must be recomputed, or None if it is up to date."""
    if not path.is_file():
        return "missing"
    try:
        meta = hl.read_metadata(path)
        rows = pq.read_metadata(path).num_rows
    except (hl.HandLandmarksError, pa.ArrowException, OSError, ValueError):
        return "unreadable"
    if parameter_set(meta) != expected.parameter_set():
        return "parameters_differ"
    if "video_fingerprint" not in meta:
        return "legacy_no_fingerprint"
    if meta["video_fingerprint"] != fingerprint:
        return "video_changed"
    if rows != meta.get("frames_decoded"):
        return "row_count_mismatch"
    return None


def other_parameter_sets(out_dir: Path, expected: Expected) -> Counter[tuple[str, ...]]:
    """Counts of parquet files in ``out_dir`` made with a parameter set other than ``expected``."""
    found: Counter[tuple[str, ...]] = Counter()
    for path in sorted(out_dir.glob("*.parquet")):
        try:
            meta = hl.read_metadata(path)
        except (hl.HandLandmarksError, pa.ArrowException, OSError, ValueError):
            continue  # an unreadable file is recomputed, it is not a different parameter set
        key = parameter_set(meta)
        if key != expected.parameter_set():
            found[key] += 1
    return found


def remove_temporary_files(out_dir: Path) -> int:
    """Delete leftovers of a killed run (``.<name>.parquet.tmp-<pid>``)."""
    removed = 0
    for path in out_dir.glob(f".*{TMP_MARKER}*"):
        with contextlib.suppress(OSError):
            path.unlink()
            removed += 1
    return removed


def write_parquet_atomic(df: pd.DataFrame, final: Path) -> None:
    """``write_landmarks`` to a temporary file, then rename: no partial file is ever visible."""
    tmp = final.with_name(f".{final.name}{TMP_MARKER}{os.getpid()}")
    try:
        hl.write_landmarks(df, tmp)
        os.replace(tmp, final)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def scrub(text: str, video: Path) -> str:
    """``text`` with the clip's path, file name and ID replaced by its short hash."""
    ref = pseudonym(video.stem)
    for secret in (str(video), video.name, video.stem):
        text = text.replace(secret, ref)
    return text


# --------------------------------------------------------------------------- numbers per clip


def missing_runs(detected: np.ndarray) -> list[int]:
    """Lengths of the runs of consecutive frames without a hand, in order."""
    runs: list[int] = []
    count = 0
    for value in np.asarray(detected, dtype=bool):
        if value:
            if count:
                runs.append(count)
            count = 0
        else:
            count += 1
    if count:
        runs.append(count)
    return runs


def edge_missing(detected: np.ndarray) -> int:
    """Missing frames at the start plus at the end (a gap there cannot be interpolated)."""
    d = np.asarray(detected, dtype=bool)
    if not d.any():
        return int(len(d))
    return int(np.argmax(d)) + int(np.argmax(d[::-1]))


def tap_rate_exploratory(df: pd.DataFrame) -> float:
    """Cycles per second from the simple 25/50 crossing counter. NOT the tap detector."""
    t, _ = time_axis_s(df)
    finite = t[np.isfinite(t)]
    duration = float(finite.max() - finite.min()) if finite.size else float("nan")
    cycles = count_open_close_cycles(thumb_index_distance_px(df))
    return cycles / duration if duration > 0 else float("nan")


def clip_metrics(df: pd.DataFrame, meta: dict[str, Any], gating: Gating) -> dict[str, Any]:
    """Every per-clip number of the manifest, from the table and its metadata."""
    summary = hl.summarize(df, meta)
    n = int(summary["frames"])
    detected = df["detected"].to_numpy(dtype=bool)
    runs = missing_runs(detected)
    unfillable = sum(r for r in runs if r > gating.fillable_gap_frames)
    pts = df["pts_ms"].to_numpy(dtype=float)
    finite_pts = pts[np.isfinite(pts)]
    steps = np.diff(finite_pts) if len(finite_pts) > 1 else np.array([])
    width, height = int(meta["width"]), int(meta["height"])
    try:
        normalised = hs.normalised_distance(df)
        scale_meta = normalised.metadata()
        coupling = hs.scale_tap_correlation(df)
    except (ValueError, KeyError, hl.HandLandmarksError):
        scale_meta, coupling = {}, float("nan")
    return {
        "fps": float(meta["fps"]),
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "orientation": "portrait"
        if height > width
        else "landscape"
        if width > height
        else "square",
        "orientation_meta": meta.get("orientation_meta"),
        "duration_s": float(finite_pts.max() - finite_pts.min()) / 1000.0
        if finite_pts.size
        else float("nan"),
        "frames_container": int(summary["frames_container"]),
        "frames_decoded": n,
        "frames_detected": int(summary["detected_frames"]),
        "detection_rate": float(summary["detection_rate"]),
        "longest_missing_run": max(runs, default=0),
        "n_gaps": len(runs),
        "gap_median_frames": float(np.median(runs)) if runs else 0.0,
        "gap_share_fillable": (
            sum(r <= gating.fillable_gap_frames for r in runs) / len(runs) if runs else 1.0
        ),
        "unfillable_frames_share": unfillable / n if n else float("nan"),
        "edge_missing_frames": edge_missing(detected),
        "handedness_switches": int(summary["handedness_switches"]),
        "out_of_range_frames": int(summary["frames_with_any_value_outside_0_1"]),
        "out_of_range_share": (
            summary["frames_with_any_value_outside_0_1"] / max(summary["detected_frames"], 1)
        ),
        "pts_missing": int(summary["pts_ms_missing"]),
        "pts_increasing": bool(summary["pts_ms_strictly_increasing"]),
        "pts_max_gap_ms": float(steps.max()) if steps.size else float("nan"),
        "pts_max_dev_from_index_ms": float(summary["max_abs_pts_minus_index_over_fps_ms"]),
        "scale": scale_meta.get("scale"),
        "scale_mode": scale_meta.get("mode"),
        "scale_window_s": scale_meta.get("window_s"),
        "scale_rejected_frames": scale_meta.get("scale_rejected_frames"),
        "scale_median_px": scale_meta.get("scale_median_px", float("nan")),
        "scale_tap_abs_r": coupling,
        "tap_rate_exploratory_hz": tap_rate_exploratory(df),
    }


def evaluate_flags(m: dict[str, Any], g: Gating) -> list[tuple[str, str]]:
    """Gating flags as (name, reason). Flags only: nothing is removed. At a threshold: no flag.

    Names starting with ``info_`` are informational and not a reason to review a clip.
    """
    flags: list[tuple[str, str]] = []
    if m["detection_rate"] < g.min_detection_rate:
        flags.append(
            ("low_detection", f"detection rate {m['detection_rate']:.3f} < {g.min_detection_rate}")
        )
    if m["unfillable_frames_share"] > g.max_unfillable_share:
        flags.append(
            (
                "gaps_not_fillable",
                f"{m['unfillable_frames_share']:.3f} of frames in gaps longer than "
                f"{g.fillable_gap_frames} frames > {g.max_unfillable_share}",
            )
        )
    if m["frames_decoded"] < g.min_frames:
        flags.append(("short_clip", f"{m['frames_decoded']} frames < {g.min_frames}"))
    container = m["frames_container"]
    if container > 0 and abs(m["frames_decoded"] - container) / container > g.max_frame_mismatch:
        flags.append(
            (
                "frame_count_mismatch",
                f"decoded {m['frames_decoded']} vs container {container} "
                f"(more than {g.max_frame_mismatch:.0%})",
            )
        )
    if m["pts_missing"] > 0 or not m["pts_increasing"]:
        flags.append(
            ("pts_unreliable", f"{m['pts_missing']} frames without pts_ms or not increasing")
        )
    if m["out_of_range_share"] > g.max_out_of_frame_share:
        flags.append(
            (
                "landmarks_out_of_frame",
                f"{m['out_of_range_share']:.3f} of detected frames have a value outside [0, 1] "
                f"> {g.max_out_of_frame_share}",
            )
        )
    r = m["scale_tap_abs_r"]
    if np.isfinite(r) and r > g.max_coupling_r:
        flags.append(
            ("scale_tap_coupling", f"|r| between scale and tap {r:.2f} > {g.max_coupling_r}")
        )
    if np.isfinite(m["pts_max_gap_ms"]) and m["pts_max_gap_ms"] > g.irregular_pts_gap_ms:
        flags.append(
            (
                "info_irregular_timing",
                f"largest pts_ms gap {m['pts_max_gap_ms']:.0f} ms > {g.irregular_pts_gap_ms:g}",
            )
        )
    return flags


MANIFEST_COLUMNS = (
    "clip_id",
    "ref",
    "participant_ref",
    "group",
    "hand",
    "container",
    "resolution",
    "width",
    "height",
    "orientation",
    "orientation_meta",
    "fps",
    "duration_s",
    "frames_container",
    "frames_decoded",
    "frames_detected",
    "detection_rate",
    "longest_missing_run",
    "n_gaps",
    "gap_median_frames",
    "gap_share_fillable",
    "unfillable_frames_share",
    "edge_missing_frames",
    "handedness_switches",
    "out_of_range_frames",
    "out_of_range_share",
    "pts_missing",
    "pts_increasing",
    "pts_max_gap_ms",
    "pts_max_dev_from_index_ms",
    "scale",
    "scale_mode",
    "scale_window_s",
    "scale_rejected_frames",
    "scale_median_px",
    "scale_tap_abs_r",
    "tap_rate_exploratory_hz",
    "flags",
    "flag_details",
    "status",
    "reason",
    "action",
    "recompute_reason",
    "wall_s",
    "runner",
    "tool",
    "mediapipe",
    "opencv",
    "model_sha256",
    "num_hands",
    "min_hand_detection_confidence",
    "min_hand_presence_confidence",
    "min_tracking_confidence",
    "gating",
)  # fmt: skip


def identity(video: Path) -> dict[str, Any]:
    """The clip's name-derived columns. Group and hand are recorded for the audit only."""
    try:
        key = parse_clip_id(video.stem)
        group: str | None = key.group
        hand: str | None = key.hand
        participant: str | None = pseudonym(key.participant)
    except UnparseableClipId:
        group = hand = participant = None
    return {
        "clip_id": video.stem,
        "ref": pseudonym(video.stem),
        "participant_ref": participant,
        "group": group,
        "hand": hand,
        "container": video.suffix.lower().lstrip("."),
    }


def build_row(
    parquet: Path,
    video: Path,
    gating: Gating,
    *,
    action: str,
    recompute_reason: str | None,
) -> dict[str, Any]:
    """A manifest row for a finished clip, always built from the parquet as stored on disk."""
    df, meta = hl.read_landmarks(parquet)
    metrics = clip_metrics(df, meta, gating)
    flags = evaluate_flags(metrics, gating)
    detector = meta["detector"]
    return {
        **identity(video),
        **metrics,
        "flags": ";".join(name for name, _ in flags),
        "flag_details": "; ".join(f"{name}: {why}" for name, why in flags),
        "status": "ok",
        "reason": "",
        "action": action,
        "recompute_reason": recompute_reason or "",
        "wall_s": float(meta.get("tracking_wall_s", float("nan"))),
        "runner": meta.get("runner"),
        "tool": meta["tool"],
        "mediapipe": meta["mediapipe"],
        "opencv": meta["opencv"],
        "model_sha256": meta["model_sha256"],
        "num_hands": detector["num_hands"],
        "min_hand_detection_confidence": detector["min_hand_detection_confidence"],
        "min_hand_presence_confidence": detector["min_hand_presence_confidence"],
        "min_tracking_confidence": detector["min_tracking_confidence"],
        "gating": json.dumps(dataclasses.asdict(gating), sort_keys=True),
    }


def failed_row(
    video: Path, gating: Gating, detector: Detector, reason: str, recompute_reason: str | None
) -> dict[str, Any]:
    row: dict[str, Any] = {c: None for c in MANIFEST_COLUMNS}
    row.update(identity(video))
    row.update(
        status="failed",
        reason=reason,
        action="failed",
        recompute_reason=recompute_reason or "",
        runner=RUNNER_VERSION,
        num_hands=detector.num_hands,
        min_hand_detection_confidence=detector.min_hand_detection_confidence,
        min_hand_presence_confidence=detector.min_hand_presence_confidence,
        min_tracking_confidence=detector.min_tracking_confidence,
        gating=json.dumps(dataclasses.asdict(gating), sort_keys=True),
        flags="",
        flag_details="",
    )
    return row


# --------------------------------------------------------------------------- one clip


@dataclass(frozen=True)
class Task:
    """Everything one clip needs. Picklable, so it can go to a spawned worker."""

    video: str
    out_dir: str
    model: str
    detector: Detector
    gating: Gating
    reason: str | None  # None = up to date, otherwise why it is recomputed
    capture_factory: Any = None  # fakes for tests; must be a module-level function for N > 1
    landmarker_factory: Any = None


def run_task(task: Task) -> dict[str, Any]:
    """Track (or just read) one clip and return its manifest row. Never raises for a bad clip."""
    video = Path(task.video)
    parquet = hl.output_path(video, task.out_dir)
    try:
        if task.reason is None:
            return build_row(parquet, video, task.gating, action="skipped", recompute_reason=None)
        started = time.perf_counter()
        fingerprint = video_fingerprint(video)
        df = hl.track_clip(
            video,
            task.model,
            num_hands=task.detector.num_hands,
            min_hand_detection_confidence=task.detector.min_hand_detection_confidence,
            min_hand_presence_confidence=task.detector.min_hand_presence_confidence,
            min_tracking_confidence=task.detector.min_tracking_confidence,
            landmarker_factory=task.landmarker_factory,
            capture_factory=task.capture_factory,
        )
        df.attrs["hand_landmarks"].update(
            video_fingerprint=fingerprint,
            runner=RUNNER_VERSION,
            tracking_wall_s=round(time.perf_counter() - started, 3),
        )
        write_parquet_atomic(df, parquet)
        return build_row(
            parquet, video, task.gating, action="computed", recompute_reason=task.reason
        )
    except Exception as err:  # one bad clip must not stop the batch
        text = scrub(f"{type(err).__name__}: {err}", video)
        return failed_row(video, task.gating, task.detector, text, task.reason)


# --------------------------------------------------------------------------- the batch


def write_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    """The manifest, sorted by clip, written atomically."""
    frame = pd.DataFrame(sorted(rows, key=lambda r: r["clip_id"]), columns=list(MANIFEST_COLUMNS))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}{TMP_MARKER}{os.getpid()}")
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def plan(
    videos: Sequence[Path], out_dir: Path, expected: Expected, force: bool
) -> dict[Path, str | None]:
    """For each video: None if up to date, else the reason it is recomputed."""
    reasons: dict[Path, str | None] = {}
    for video in videos:
        reason = stale_reason(hl.output_path(video, out_dir), expected, video_fingerprint(video))
        reasons[video] = "forced" if force else reason
    return reasons


def run_batch(tasks: Sequence[Task], workers: int, progress: bool = True) -> list[dict[str, Any]]:
    """Run every task; in-process for one worker, in spawned processes otherwise."""
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    def report(row: dict[str, Any]) -> None:
        rows.append(row)
        if progress:
            flags = f" flags={row['flags']}" if row["flags"] else ""
            detail = f" ({row['reason']})" if row["status"] == "failed" else ""
            print(
                f"[{len(rows)}/{len(tasks)}] {row['ref']} {row['action']}{detail}{flags} "
                f"{time.perf_counter() - started:.0f}s",
                flush=True,
            )

    if workers <= 1 or len(tasks) <= 1:
        for task in tasks:
            report(run_task(task))
        return rows
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
    try:
        futures = [pool.submit(run_task, t) for t in tasks]
        for future in as_completed(futures):
            report(future.result())
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()
    return rows


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def summary_lines(rows: list[dict[str, Any]], elapsed: float) -> list[str]:
    ok = [r for r in rows if r["status"] == "ok"]
    failed = [r for r in rows if r["status"] == "failed"]
    flag_counts: Counter[str] = Counter()
    for r in ok:
        flag_counts.update(f for f in r["flags"].split(";") if f)
    reviewed = sum(any(f and not f.startswith("info_") for f in r["flags"].split(";")) for r in ok)
    lines = [
        f"done in {elapsed:.0f}s: {len(rows)} clips, {len(ok)} ok "
        f"({sum(r['action'] == 'computed' for r in ok)} computed, "
        f"{sum(r['action'] == 'skipped' for r in ok)} up to date), {len(failed)} failed",
        f"flags (clips are kept; flags only): {reviewed} clips with a review flag; "
        + (", ".join(f"{k} {v}" for k, v in sorted(flag_counts.items())) or "none"),
    ]
    lines += [f"  failed: {r['ref']}: {r['reason']}" for r in failed]
    return lines


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--manifest", type=Path, default=None, help="default depends on --out-dir")
    ap.add_argument("--model", type=Path, default=hl.DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--clips", default=None, help="comma-separated 8-character hashes")
    ap.add_argument("--limit", type=int, default=None, help="only the first N clips by name")
    ap.add_argument("--force", action="store_true", help="recompute everything selected")
    ap.add_argument("--dry-run", action="store_true", help="show the plan, write nothing")
    ap.add_argument("--interim-root", type=Path, default=INTERIM_ROOT, help="outputs stay inside")
    det, gate = Detector(), Gating()
    ap.add_argument("--num-hands", type=int, default=det.num_hands)
    ap.add_argument(
        "--min-hand-detection-confidence",
        type=float,
        default=det.min_hand_detection_confidence,
    )
    ap.add_argument(
        "--min-hand-presence-confidence",
        type=float,
        default=det.min_hand_presence_confidence,
    )
    ap.add_argument("--min-tracking-confidence", type=float, default=det.min_tracking_confidence)
    ap.add_argument("--min-detection-rate", type=float, default=gate.min_detection_rate)
    ap.add_argument("--fillable-gap-frames", type=int, default=gate.fillable_gap_frames)
    ap.add_argument("--max-unfillable-share", type=float, default=gate.max_unfillable_share)
    ap.add_argument("--min-frames", type=int, default=gate.min_frames)
    ap.add_argument("--max-frame-mismatch", type=float, default=gate.max_frame_mismatch)
    ap.add_argument("--max-out-of-frame-share", type=float, default=gate.max_out_of_frame_share)
    ap.add_argument("--max-coupling-r", type=float, default=gate.max_coupling_r)
    ap.add_argument("--irregular-pts-gap-ms", type=float, default=gate.irregular_pts_gap_ms)
    return ap


class _Interrupted(KeyboardInterrupt):
    """Raised from the SIGTERM handler so a terminated run stops like Ctrl-C."""


def main(
    argv: Sequence[str] | None = None,
    *,
    capture_factory: Any = None,
    landmarker_factory: Any = None,
) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.manifest or (
        DEFAULT_MANIFEST if args.out_dir == DEFAULT_OUT_DIR else args.out_dir / MANIFEST_NAME
    )
    for folder in (args.out_dir, manifest_path):
        if not _inside(folder, args.interim_root):
            print(f"error: outputs must be inside {args.interim_root}", file=sys.stderr)
            return 2
    if not args.video_dir.is_dir():
        print(f"error: video folder not found: {args.video_dir}", file=sys.stderr)
        return 2
    if landmarker_factory is None and not args.model.is_file():
        print(
            f"error: hand landmarker model not found at {args.model}. "
            f"Download it with {hl.FETCH_SCRIPT}",
            file=sys.stderr,
        )
        return 2
    try:
        videos = find_videos(args.video_dir)
    except hl.HandLandmarksError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if args.clips:
        wanted = [c.strip() for c in args.clips.split(",") if c.strip()]
        known = {pseudonym(v.stem) for v in videos}
        unknown = [c for c in wanted if c not in known]
        if unknown:
            print(f"error: no video for clip(s) {', '.join(unknown)}", file=sys.stderr)
            return 2
        videos = [v for v in videos if pseudonym(v.stem) in set(wanted)]
    if args.limit is not None:
        videos = videos[: max(args.limit, 0)]
    if not videos:
        print("error: no video files selected", file=sys.stderr)
        return 2

    detector = Detector(
        args.num_hands,
        args.min_hand_detection_confidence,
        args.min_hand_presence_confidence,
        args.min_tracking_confidence,
    )
    gating = Gating(
        args.min_detection_rate,
        args.fillable_gap_frames,
        args.max_unfillable_share,
        args.min_frames,
        args.max_frame_mismatch,
        args.max_out_of_frame_share,
        args.max_coupling_r,
        args.irregular_pts_gap_ms,
    )
    expected = make_expected(detector, hl._sha256(args.model))

    other: Counter[tuple[str, ...]] = (
        other_parameter_sets(args.out_dir, expected) if args.out_dir.is_dir() else Counter()
    )
    if other and not args.force:
        print(
            f"error: {args.out_dir.name}/ holds {sum(other.values())} file(s) made with another "
            f"parameter set ({len(other)} different set(s)); mixing them is refused. Use a "
            "separate --out-dir for each parameter set, or --force to recompute the selected "
            "clips.",
            file=sys.stderr,
        )
        return 2

    reasons = plan(videos, args.out_dir, expected, args.force)
    counts = Counter(r or "up_to_date" for r in reasons.values())
    print(
        f"{len(videos)} clips selected: {counts.get('up_to_date', 0)} up to date, "
        f"{sum(v for k, v in counts.items() if k != 'up_to_date')} to compute "
        + (
            "("
            + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()) if k != "up_to_date")
            + ")"
            if len(counts) > (1 if "up_to_date" in counts else 0)
            else ""
        )
    )
    if other:
        print(f"warning: --force; {sum(other.values())} other-parameter file(s) in the folder")
    if args.dry_run:
        print("dry run: nothing written")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    removed = remove_temporary_files(args.out_dir)
    if removed:
        print(f"removed {removed} temporary file(s) left by an interrupted run")
    tasks = [
        Task(
            str(v),
            str(args.out_dir),
            str(args.model),
            detector,
            gating,
            reasons[v],
            capture_factory,
            landmarker_factory,
        )
        for v in videos
    ]
    started = time.perf_counter()

    def on_term(signum: int, frame: Any) -> None:
        raise _Interrupted

    previous = signal.signal(signal.SIGTERM, on_term)
    try:
        rows = run_batch(tasks, args.workers)
    except KeyboardInterrupt:
        print("interrupted: finished clips are kept, the rest will be done on the next run")
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous)
    write_manifest(rows, manifest_path)
    for line in summary_lines(rows, time.perf_counter() - started):
        print(line)
    print(f"manifest: {manifest_path.name}")
    return 1 if any(r["status"] == "failed" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
