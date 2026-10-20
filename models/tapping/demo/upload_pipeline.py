"""Upload pipeline for the demo (task FT-48): one video to landmarks, the aperture signal, tap
peaks, automatic QC flags and preview frames, by calling the research pipeline (track_clip,
build_signal, smooth_signal, extract_taps with TapParams v1). No streamlit import. An upload is
identified only by sha256[:8] of its bytes; its name is never used. The video lives only in a
TemporaryDirectory that is removed in a finally block.
"""

from __future__ import annotations

import hashlib
import io
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import hand_landmarks as hl  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tracking  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from signal_processing import SignalError, build_signal  # noqa: E402
from smoothing import SmoothingError, smooth_signal  # noqa: E402
from taps import PARAMS, TapError, extract_taps  # noqa: E402

MAX_BYTES = 200 * 1024 * 1024
MAX_DURATION_S = 60.0
MIN_FPS = 15.0
MIN_TRACKED_SHARE = 0.5
N_PREVIEW = 6
SUFFIXES = (".mp4", ".mov", ".m4v")
WORK_NAME = "upload"  # the only file name that exists on disk
FOOTER = "Signal from the research pipeline (hand landmarks, scaled by hand size). Not a diagnosis."
POINTS = (0, 4, 8, 9)  # wrist and middle knuckle (hand size), thumb tip and index tip
THUMB_RGB, INDEX_RGB, SCALE_RGB, LINE_RGB = (
    (255, 200, 0),
    (255, 0, 80),
    (0, 160, 255),
    (255, 255, 255),
)
MIN_RADIUS = 6  # marker radius grows with the frame width so it stays visible when shrunk
RAW_COLOUR, SMOOTH_COLOUR, PEAK_COLOUR = "#9db4d6", "#1b4f9c", "#c4622d"
LandmarkerFactory = Callable[..., Any]


class UploadError(ValueError):
    """The upload cannot be used; the message is safe to show on screen."""


@dataclass(eq=False)
class UploadResult:
    upload_ref: str
    fps: float
    duration_s: float
    n_frames: int
    tracked_share: float
    t: np.ndarray  # s, from the first decoded frame
    raw: np.ndarray  # thumb-index distance in hand sizes, before smoothing
    smoothed: np.ndarray
    peak_t: np.ndarray
    peak_v: np.ndarray
    n_taps: int
    median_interval_s: float
    qc_flags: tuple[tuple[str, str], ...]
    preview_indices: tuple[int, ...]
    preview_times: tuple[float, ...]
    preview_points: dict[int, np.ndarray]  # frame -> (21, 2) landmark pixels
    time_offset_s: float  # signal time = tap table time + this (the first kept sample's offset)
    taps_table: pd.DataFrame  # the official tap table of the clip (input of the Set A features)
    stats: dict[str, Any]  # the official clip summary numbers (fps, duration_s, ...)


@dataclass(eq=False)
class UploadAnalysis:
    result: UploadResult
    frames: list[np.ndarray]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_video(path: Path) -> tuple[float, int]:
    """(fps, container frame count) of a video, or UploadError if it cannot be read."""
    cap = hl.open_video(str(path))
    try:
        if not cap.isOpened():
            raise UploadError("The video could not be read. Try an mp4 or mov file.")
        return float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def pick_preview(detected: np.ndarray, n: int = N_PREVIEW) -> np.ndarray:
    """n evenly spaced frame indices among the frames with a hand."""
    on = np.flatnonzero(detected)
    if len(on) <= n:
        return on
    return on[np.round(np.linspace(0, len(on) - 1, n)).astype(int)]


PLAIN_FLAGS = {  # fixed plain-language texts for the official flag names
    "low_detection": "the hand was found in fewer frames than expected",
    "gaps_not_fillable": "the hand was lost for long stretches",
    "short_clip": "the clip is short",
    "frame_count_mismatch": "the number of decoded frames differs from the file header",
    "pts_unreliable": "frame times are missing or not increasing",
    "landmarks_out_of_frame": "part of the hand left the picture in many frames",
    "scale_tap_coupling": "the hand size estimate moves with the taps",
    "info_irregular_timing": "frame times are uneven",
    "low_range": "the opening range is small, so taps may be missed",
}


def qc_flags(df: Any, stats: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    gating = tracking.Gating()
    metrics = tracking.clip_metrics(df, df.attrs["hand_landmarks"], gating)
    names = [name for name, _ in tracking.evaluate_flags(metrics, gating)]
    if stats.get("low_range"):
        names.append("low_range")
    return tuple((n, PLAIN_FLAGS.get(n, "see the research pipeline flag")) for n in names)


def process_video(path: str | Path, landmarker_factory: LandmarkerFactory) -> UploadResult:
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError as err:
        raise UploadError("The video could not be read.") from err
    if size > MAX_BYTES:
        raise UploadError(f"The video is larger than {MAX_BYTES // (1024 * 1024)} MB.")
    ref = sha256_of(path)[:8]
    fps, frames = probe_video(path)
    if fps < MIN_FPS:
        raise UploadError(f"The frame rate is below {MIN_FPS:g} fps; film at 30 fps or more.")
    if fps > 0 and frames / fps > MAX_DURATION_S:
        raise UploadError(f"The video is longer than {MAX_DURATION_S:g} s.")
    model = getattr(landmarker_factory, "model_path", hl.DEFAULT_MODEL)
    try:
        df = hl.track_clip(path, model, landmarker_factory=landmarker_factory)
    except (hl.HandLandmarksError, hl.InputMissingError, OSError) as err:
        raise UploadError("The hand could not be tracked in this video.") from err
    share = float(df["detected"].mean())
    if share < MIN_TRACKED_SHARE:
        raise UploadError("Hand not found in enough frames; keep the hand in view.")
    try:
        sig = build_signal(df, ref)
        if len(sig.table) == 0:
            raise UploadError("Hand not found in enough frames; keep the hand in view.")
        tb = smooth_signal(sig).table
        t_abs = tb["t_s"].to_numpy(dtype=float)
        result = extract_taps(
            ref, t_abs, tb["value"].to_numpy(dtype=float), tb["value_raw"].to_numpy(dtype=float),
            tb["filled"].to_numpy(dtype=bool), tb["frame"].to_numpy(dtype=np.int64), PARAMS,
        )  # fmt: skip
    except (SignalError, SmoothingError, TapError, KeyError, ValueError) as err:
        if isinstance(err, UploadError):
            raise
        raise UploadError("The signal could not be built from this video.") from err
    pts = df["pts_ms"].to_numpy(dtype=float) / 1000.0
    t0 = float(pts[np.isfinite(pts)][0]) if np.isfinite(pts).any() else float(t_abs[0])
    t = t_abs - t0
    duration = float(t[-1])
    if duration > MAX_DURATION_S:
        raise UploadError(f"The video is longer than {MAX_DURATION_S:g} s.")
    taps = result.table
    idx = pick_preview(df["detected"].to_numpy(dtype=bool))
    names = range(hl.N_LANDMARKS)
    xs = df[[hl.landmark_column("x_px", k) for k in names]].to_numpy(dtype=float)
    ys = df[[hl.landmark_column("y_px", k) for k in names]].to_numpy(dtype=float)
    xy = {int(i): np.column_stack([xs[i], ys[i]]) for i in idx}
    return UploadResult(
        upload_ref=ref, fps=float(df.attrs["hand_landmarks"]["fps"]), duration_s=duration,
        n_frames=len(df), tracked_share=share, t=t,
        raw=tb["value_raw"].to_numpy(dtype=float), smoothed=tb["value"].to_numpy(dtype=float),
        peak_t=taps["t_peak_s"].to_numpy(dtype=float) + (t_abs[0] - t0),
        peak_v=taps["value_peak"].to_numpy(dtype=float), n_taps=len(taps),
        median_interval_s=float(result.stats.get("median_interval_s", float("nan"))),
        qc_flags=qc_flags(df, result.stats), preview_indices=tuple(int(i) for i in idx),
        preview_times=tuple(float(pts[i] - t0) for i in idx), preview_points=xy,
        time_offset_s=float(t_abs[0] - t0),
        taps_table=taps.reset_index(drop=True), stats=dict(result.stats),
    )  # fmt: skip


def draw_marks(rgb: np.ndarray, points: np.ndarray) -> np.ndarray:
    """A copy of the frame with the thumb-index line and landmarks 0, 4, 8, 9 marked."""
    out = np.ascontiguousarray(rgb.copy())
    p = {k: (int(round(points[k, 0])), int(round(points[k, 1]))) for k in POINTS}
    radius = max(MIN_RADIUS, out.shape[1] // 90)
    cv2.line(out, p[4], p[8], LINE_RGB, max(2, radius // 3), cv2.LINE_AA)
    for k, colour in ((0, SCALE_RGB), (9, SCALE_RGB), (4, THUMB_RGB), (8, INDEX_RGB)):
        cv2.circle(out, p[k], radius, colour, -1, cv2.LINE_8)
    return out


def preview_frames(
    path: str | Path, indices: Sequence[int], landmarks: dict[int, np.ndarray]
) -> list[np.ndarray]:
    """RGB frames at the given decoded-frame indices with the landmarks drawn (in memory only)."""
    wanted = sorted(set(indices))
    cap = hl.open_video(str(path))
    found: dict[int, np.ndarray] = {}
    try:
        if not cap.isOpened():
            raise UploadError("The video could not be read.")
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)  # as in track_clip
        for i in range(max(wanted, default=-1) + 1):
            ok, bgr = cap.read()
            if not ok:
                break
            if i in wanted:
                found[i] = draw_marks(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), landmarks[i])
    finally:
        cap.release()
    if set(found) != set(wanted):
        raise UploadError("A preview frame could not be read.")
    return [found[i] for i in indices]


def analyse_upload(
    data: bytes, suffix: str, landmarker_factory: LandmarkerFactory
) -> UploadAnalysis:
    """Bytes to result and preview frames; the video exists only inside the temp directory."""
    if suffix not in SUFFIXES:
        raise UploadError("Use an mp4, mov or m4v video.")
    if len(data) > MAX_BYTES:
        raise UploadError(f"The video is larger than {MAX_BYTES // (1024 * 1024)} MB.")
    tmp = tempfile.TemporaryDirectory()
    try:
        path = Path(tmp.name) / f"{WORK_NAME}{suffix}"
        path.write_bytes(data)
        result = process_video(path, landmarker_factory)
        frames = preview_frames(path, result.preview_indices, result.preview_points)
        return UploadAnalysis(result, frames)
    finally:
        tmp.cleanup()


def signal_figure(r: UploadResult, compact: bool = False) -> Figure:
    """The signal figure; ``compact`` is the page-size variant (larger relative fonts, no footer:
    the PDF carries the disclaimer as text)."""
    big, mid, title = (9, 8.5, 10) if compact else (12, 11, 14)
    fig = Figure(figsize=(5.2, 2.3) if compact else (11.5, 4.8), dpi=110, layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.plot(r.t, r.raw, color=RAW_COLOUR, lw=0.8, label="raw")
    ax.plot(r.t, r.smoothed, color=SMOOTH_COLOUR, lw=1.8, label="smoothed")
    ax.plot(r.peak_t, r.peak_v, "v", color=PEAK_COLOUR, ms=5 if compact else 7, ls="none",
            label="tap peaks")  # fmt: skip
    ax.set_xlabel("time (s)", fontsize=big)
    ax.set_ylabel("distance (hand sizes)" if compact else "thumb-index distance (hand sizes)",
                  fontsize=big)  # fmt: skip
    ax.tick_params(labelsize=mid)
    ax.spines[["top", "right"]].set_visible(False)
    where: Literal["outside lower center", "outside upper right"] = (
        "outside lower center" if compact else "outside upper right"
    )  # never over the title
    fig.legend(fontsize=mid, loc=where, frameon=False, ncols=3)  # off the curve
    fig.suptitle(f"Upload {r.upload_ref}: {r.n_taps} taps found", fontsize=title)
    if not compact:
        fig.supxlabel(FOOTER, fontsize=11, color="#333333")
    return fig


def figure_png(fig: Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", metadata={"Software": None})
    return buf.getvalue()
