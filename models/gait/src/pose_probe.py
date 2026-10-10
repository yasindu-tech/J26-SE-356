"""Run MediaPipe Pose on one walking video and measure whether it is usable (GAIT-09, Check C).

The questions this answers for a clip:

* In what fraction of frames does MediaPipe find a person?
* How many landmarks (all 33, and the 10 leg landmarks) are visible?
* Does the ankle-height trace show clear, repeating minima (gait cycles)?

The pass mark is the plan's suggestion, not a standard: a person found in at least 90% of
frames and at least 2 gait cycles. The cycle count uses the *world* landmarks, because those
are what the later pipeline feeds to the gait features. The image-plane trace is only a
diagnostic.

Nothing is written about the video's content beyond numbers and trace plots: no frames are
saved and no file names are printed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import find_peaks, savgol_filter

N_LANDMARKS = 33
L_HIP_IDX, R_HIP_IDX = 23, 24
L_ANKLE_IDX, R_ANKLE_IDX = 27, 28
LEG_IDX = tuple(range(23, 33))  # hips, knees, ankles, heels, foot index

VISIBLE_THRESHOLD = 0.5  # a landmark counts as visible at or above this visibility score
DETECTION_PASS = 0.90  # plan, Check C: person found in at least 90% of frames
CYCLES_PASS = 2  # plan, Check C: at least 2 gait cycles
MIN_CYCLE_S = 0.4  # minima closer than this are the same event
MAX_GAP_S = 0.5  # interpolate over detection gaps up to this long
CLEAN_LEG_VISIBILITY = (
    0.8  # a frame is "clean" if found and at least this share of leg landmarks is visible
)
MIN_CLEAN_S = (
    3.0  # need at least this many seconds of clean frames for the cycle count to mean anything
)
MIN_SEGMENT_S = 0.8  # ignore stretches of continuous detection shorter than this

# Smallest ankle excursion counted as a step. World traces are in metres (about 2 cm);
# the image trace is in fractions of the frame height (about 0.5%).
ABS_FLOOR = {"world": 0.02, "image": 0.005}
REL_PROMINENCE = 0.3  # and at least this fraction of the trace's 5th-95th percentile range


# --------------------------------------------------------------------------- video info
@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    n_frames: int  # as reported by the container; the real count can differ slightly

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps > 0 else 0.0


def read_info(path: Path) -> VideoInfo:
    """Frame rate, size and frame count from the container (OpenCV)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("could not open the video")
        return VideoInfo(
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


# --------------------------------------------------------------------------- crop windows
Window = tuple[int, int, int, int]  # x0, y0, x1, y1 in full-frame pixels


def fixed_window(frac: tuple[float, float, float, float], width: int, height: int) -> Window:
    """Turn ``(x0, y0, x1, y1)`` given as fractions of the frame into pixels."""
    x0, y0, x1, y1 = frac
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("crop fractions must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    win = (round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height))
    if win[2] - win[0] < 32 or win[3] - win[1] < 32:
        raise ValueError("crop is smaller than 32 pixels")
    return win


def motion_windows(
    gray: np.ndarray,
    full_size: tuple[int, int],
    *,
    threshold: int = 25,
    min_pixels: int = 4,
    margin: float = 1.4,
    min_height_frac: float = 0.35,
    smooth: float = 0.3,
) -> list[Window]:
    """One crop window per frame that follows the moving person.

    ``gray`` is ``(n, h, w)`` uint8 (frames shrunk for speed) from a *static* camera. The
    background is the median over time, so a walker who crosses the frame is removed from it.
    Pixels that differ from the background mark the walker; the window is centred on them,
    ``margin`` times the walker's height, with a 3:4 (width:height) shape, and its centre is
    smoothed so the crop does not jitter. Frames with no motion reuse the previous window.
    Windows are returned in full-frame pixels (``full_size`` is ``(width, height)``).
    """
    n, h, w = gray.shape
    full_w, full_h = full_size
    bg = np.median(gray, axis=0)
    sx, sy = full_w / w, full_h / h

    cx = cy = None
    win_h = float(full_h)
    out: list[Window] = []
    for i in range(n):
        mask = np.abs(gray[i].astype(np.float32) - bg) > threshold
        rows = np.flatnonzero(mask.sum(axis=1) >= min_pixels)
        cols = np.flatnonzero(mask.sum(axis=0) >= min_pixels)
        if rows.size and cols.size:
            tcx, tcy = (cols[0] + cols[-1]) / 2 * sx, (rows[0] + rows[-1]) / 2 * sy
            target_h = (rows[-1] - rows[0] + 1) * sy * margin
            if cx is None or cy is None:
                cx, cy, win_h = tcx, tcy, target_h
            else:
                cx += smooth * (tcx - cx)
                cy += smooth * (tcy - cy)
                win_h += smooth * (target_h - win_h)
        if cx is None or cy is None:  # no motion seen yet: start with the whole frame
            out.append((0, 0, full_w, full_h))
            continue
        height = min(max(win_h, min_height_frac * full_h), full_h)
        width = min(height * 3 / 4, full_w)
        x0 = int(np.clip(cx - width / 2, 0, full_w - width))
        y0 = int(np.clip(cy - height / 2, 0, full_h - height))
        out.append((x0, y0, x0 + int(width), y0 + int(height)))
    return out


# --------------------------------------------------------------------------- pose trace
@dataclass
class PoseTrace:
    fps: float
    width: int
    height: int
    detected: np.ndarray  # (n,) bool
    visibility: np.ndarray  # (n, 33), NaN where nothing was detected
    image: np.ndarray  # (n, 33, 2) x, y as fractions of the FULL frame, NaN if not detected
    world: np.ndarray  # (n, 33, 3) metres around the hip midpoint, y down, NaN if not detected

    @property
    def n_frames(self) -> int:
        return int(self.detected.shape[0])


def landmarks_from_result(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """``(visibility[33], image_xy[33, 2], world_xyz[33, 3])`` for the first person, or None."""
    if not result.pose_landmarks or not result.pose_world_landmarks:
        return None
    img, world = result.pose_landmarks[0], result.pose_world_landmarks[0]
    if len(img) != N_LANDMARKS or len(world) != N_LANDMARKS:
        return None
    vis = np.array([p.visibility for p in img], dtype=np.float64)
    xy = np.array([[p.x, p.y] for p in img], dtype=np.float64)
    xyz = np.array([[p.x, p.y, p.z] for p in world], dtype=np.float64)
    return vis, xy, xyz


def run_pose(
    frames: Iterator[np.ndarray],
    info: VideoInfo,
    landmarker: Any,
    windows: list[Window] | None = None,
    *,
    make_image: Callable[[np.ndarray], Any],
) -> PoseTrace:
    """Run a Pose Landmarker (video mode) over RGB frames and collect the landmarks.

    ``landmarker`` needs ``detect_for_video(image, timestamp_ms)``; ``make_image`` wraps an RGB
    array for it. Both are passed in so the logic can be tested without MediaPipe. Image
    landmarks are mapped back to the full frame when a crop window was used.
    """
    fps = info.fps if info.fps > 0 else 30.0
    det: list[bool] = []
    vis_l: list[np.ndarray] = []
    xy_l: list[np.ndarray] = []
    world_l: list[np.ndarray] = []
    nan_vis = np.full(N_LANDMARKS, np.nan)
    nan_xy = np.full((N_LANDMARKS, 2), np.nan)
    nan_world = np.full((N_LANDMARKS, 3), np.nan)
    last_ts = -1
    for i, frame in enumerate(frames):
        # The container's frame count can be off by a frame or two; reuse the last window.
        x0, y0, x1, y1 = (
            windows[min(i, len(windows) - 1)] if windows else (0, 0, info.width, info.height)
        )
        crop = frame[y0:y1, x0:x1]
        ts = max(int(round(i * 1000 / fps)), last_ts + 1)  # MediaPipe needs strictly rising ms
        last_ts = ts
        found = landmarks_from_result(landmarker.detect_for_video(make_image(crop), ts))
        if found is None:
            det.append(False)
            vis_l.append(nan_vis)
            xy_l.append(nan_xy)
            world_l.append(nan_world)
            continue
        vis, xy, world = found
        full_xy = np.empty_like(xy)
        full_xy[:, 0] = (x0 + xy[:, 0] * (x1 - x0)) / info.width
        full_xy[:, 1] = (y0 + xy[:, 1] * (y1 - y0)) / info.height
        det.append(True)
        vis_l.append(vis)
        xy_l.append(full_xy)
        world_l.append(world)
    if not det:
        raise ValueError("no frames could be read from the video")
    return PoseTrace(
        fps=fps,
        width=info.width,
        height=info.height,
        detected=np.array(det),
        visibility=np.stack(vis_l),
        image=np.stack(xy_l),
        world=np.stack(world_l),
    )


# --------------------------------------------------------------------------- ankle traces
def clean_mask(trace: PoseTrace) -> np.ndarray:
    """Frames where the person is found and the legs are visible enough to trust the ankles.

    Frames where MediaPipe "finds" a person but sees few leg landmarks give ankle heights that
    jump around, and counting steps in them gives false cycles.
    """
    leg_vis = np.zeros(trace.n_frames)
    if trace.detected.any():
        leg_vis[trace.detected] = (
            trace.visibility[trace.detected][:, list(LEG_IDX)] >= VISIBLE_THRESHOLD
        ).mean(axis=1)
    return trace.detected & (leg_vis >= CLEAN_LEG_VISIBILITY)


def ankle_traces(trace: PoseTrace) -> dict[str, dict[str, np.ndarray]]:
    """Ankle height over time for both feet, in two coordinate systems.

    ``world``: metres, up positive, relative to the hip midpoint (MediaPipe's y points down).
    ``image``: fraction of the frame height, up positive (image y points down).
    """
    hip_mid_y = (trace.world[:, L_HIP_IDX, 1] + trace.world[:, R_HIP_IDX, 1]) / 2
    return {
        "world": {
            "left": -(trace.world[:, L_ANKLE_IDX, 1] - hip_mid_y),
            "right": -(trace.world[:, R_ANKLE_IDX, 1] - hip_mid_y),
        },
        "image": {
            "left": -trace.image[:, L_ANKLE_IDX, 1],
            "right": -trace.image[:, R_ANKLE_IDX, 1],
        },
    }


def _interpolate_short_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly fill NaN runs of at most ``max_gap`` samples; longer runs stay NaN."""
    y = x.copy()
    idx = np.arange(len(y))
    good = np.isfinite(y)
    if good.sum() < 2:
        return y
    filled = np.interp(idx, idx[good], y[good])
    nan_run = 0
    for i in range(len(y)):
        if good[i]:
            if nan_run and nan_run <= max_gap:
                y[i - nan_run : i] = filled[i - nan_run : i]
            nan_run = 0
        else:
            nan_run += 1
    return y


@dataclass(frozen=True)
class CycleResult:
    n_minima: int
    cycles: int  # intervals between consecutive minima, counted inside continuous stretches
    interval_cv: float | None  # spread of the intervals: std / mean; None with fewer than 2
    minima: np.ndarray  # sample indices of the minima


def count_cycles(height: np.ndarray, fps: float, *, abs_floor: float) -> CycleResult:
    """Count ankle-height minima and the gait cycles between them.

    Short detection gaps are bridged, long ones split the trace. Each stretch is smoothed,
    and a minimum counts only if it stands out by at least ``abs_floor`` and a fraction of
    the trace's overall range, so noise on a flat trace is not mistaken for steps.
    """
    x = _interpolate_short_gaps(np.asarray(height, dtype=np.float64), int(MAX_GAP_S * fps))
    finite = np.isfinite(x)
    if finite.sum() < 3:
        return CycleResult(0, 0, None, np.array([], dtype=int))
    lo, hi = np.percentile(x[finite], [5, 95])
    prominence = max(abs_floor, REL_PROMINENCE * (hi - lo))
    min_dist = max(int(MIN_CYCLE_S * fps), 1)
    win = max(int(round(0.2 * fps)) | 1, 5)  # odd window of about 0.2 s

    minima: list[int] = []
    intervals: list[int] = []
    cycles = 0
    start = None
    for i in range(len(x) + 1):
        ok = i < len(x) and finite[i]
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            seg = x[start:i]
            if len(seg) >= max(int(MIN_SEGMENT_S * fps), win):
                smooth = savgol_filter(seg, win, 2)
                peaks, _ = find_peaks(-smooth, distance=min_dist, prominence=prominence)
                minima.extend(int(p) + start for p in peaks)
                cycles += max(len(peaks) - 1, 0)
                intervals.extend(int(d) for d in np.diff(peaks))
            start = None
    cv = None
    if len(intervals) >= 2 and np.mean(intervals) > 0:
        cv = float(np.std(intervals) / np.mean(intervals))
    return CycleResult(len(minima), cycles, cv, np.array(minima, dtype=int))


# --------------------------------------------------------------------------- verdict
@dataclass(frozen=True)
class ClipMetrics:
    n_frames: int
    detection_rate: float
    visible_all: float  # mean share of the 33 landmarks visible, over detected frames
    visible_legs: float  # same for the 10 leg landmarks
    world_left: CycleResult
    world_right: CycleResult
    image_left: CycleResult
    image_right: CycleResult
    # Walkers often enter the frame part-way through a clip, so also look only at the stretch
    # from the first to the last frame where the person was found.
    first_found_s: float | None  # when the person is first found; None if never
    # Cycles above are counted on clean frames only (see clean_mask). These two say how much
    # clean tracking there was, and what the count would have been without the mask.
    clean_s: float  # seconds of clean frames
    raw_world_cycles: int  # world-trace cycles with no mask (for comparison only)
    in_view_rate: float  # share of frames found between the first and last found frame
    longest_gap_s: float  # longest run of missed frames inside that stretch
    reasons: tuple[str, ...]  # why the clip failed the plan's mark; empty means it passed

    @property
    def passed_in_view(self) -> bool:
        """The plan's pass mark, with detection counted from when the walker is first found."""
        return (
            self.in_view_rate >= DETECTION_PASS
            and self.gait_cycles >= CYCLES_PASS
            and self.clean_s >= MIN_CLEAN_S
        )

    @property
    def gait_cycles(self) -> int:
        """Cycles from the better foot in the world trace (the far foot is often hidden)."""
        return max(self.world_left.cycles, self.world_right.cycles)

    @property
    def image_cycles(self) -> int:
        return max(self.image_left.cycles, self.image_right.cycles)

    @property
    def passed(self) -> bool:
        return not self.reasons


def evaluate(trace: PoseTrace) -> ClipMetrics:
    """Measure one clip and apply the plan's pass mark."""
    det = trace.detected
    rate = float(det.mean())
    if det.any():
        vis = trace.visibility[det] >= VISIBLE_THRESHOLD
        visible_all = float(vis.mean())
        visible_legs = float(vis[:, list(LEG_IDX)].mean())
    else:
        visible_all = visible_legs = 0.0
    traces = ankle_traces(trace)
    clean = clean_mask(trace)
    clean_s = float(clean.sum() / trace.fps)
    res = {
        (space, side): count_cycles(
            np.where(clean, arr, np.nan), trace.fps, abs_floor=ABS_FLOOR[space]
        )
        for space, sides in traces.items()
        for side, arr in sides.items()
    }
    raw_world_cycles = max(
        count_cycles(arr, trace.fps, abs_floor=ABS_FLOOR["world"]).cycles
        for arr in traces["world"].values()
    )
    world_cycles = max(res[("world", "left")].cycles, res[("world", "right")].cycles)
    found = np.flatnonzero(det)
    if found.size:
        span = det[found[0] : found[-1] + 1]
        first_found_s: float | None = float(found[0] / trace.fps)
        in_view_rate = float(span.mean())
        longest = run = 0
        for hit in span:
            run = 0 if hit else run + 1
            longest = max(longest, run)
        longest_gap_s = float(longest / trace.fps)
    else:
        first_found_s, in_view_rate, longest_gap_s = None, 0.0, 0.0
    reasons = []
    if rate < DETECTION_PASS:
        reasons.append(f"person found in {rate:.0%} of frames (need {DETECTION_PASS:.0%})")
    if clean_s < MIN_CLEAN_S:
        reasons.append(
            f"only {clean_s:.1f} s of clean tracking, legs visible in >= "
            f"{CLEAN_LEG_VISIBILITY:.0%} of leg landmarks (need {MIN_CLEAN_S:.0f} s)"
        )
    if world_cycles < CYCLES_PASS:
        reasons.append(
            f"{world_cycles} gait cycles in the world ankle trace on clean frames "
            f"(need {CYCLES_PASS})"
        )
    return ClipMetrics(
        n_frames=trace.n_frames,
        detection_rate=rate,
        visible_all=visible_all,
        visible_legs=visible_legs,
        world_left=res[("world", "left")],
        world_right=res[("world", "right")],
        image_left=res[("image", "left")],
        image_right=res[("image", "right")],
        first_found_s=first_found_s,
        clean_s=clean_s,
        raw_world_cycles=raw_world_cycles,
        in_view_rate=in_view_rate,
        longest_gap_s=longest_gap_s,
        reasons=tuple(reasons),
    )


# --------------------------------------------------------------------------- real video
def read_rgb_frames(path: Path) -> Iterator[np.ndarray]:
    """Yield RGB frames one at a time (OpenCV reads BGR)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def read_small_gray(path: Path, width: int = 240) -> np.ndarray:
    """All frames shrunk to ``width`` pixels and made grey, for finding the moving person."""
    import cv2

    frames = []
    cap = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (width, max(round(h * width / w), 1)))
            frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    finally:
        cap.release()
    if not frames:
        raise ValueError("no frames could be read from the video")
    return np.stack(frames)


def probe_video(
    path: Path,
    model: Path,
    *,
    crop: tuple[float, float, float, float] | None = None,
    follow: bool = False,
) -> tuple[VideoInfo, PoseTrace]:
    """Run the heavy Pose Landmarker (video mode, one person) over a clip."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

    info = read_info(path)
    windows: list[Window] | None = None
    if crop is not None:
        windows = [fixed_window(crop, info.width, info.height)] * max(info.n_frames, 1)
    elif follow:
        windows = motion_windows(read_small_gray(path), (info.width, info.height))

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )

    def make_image(rgb: np.ndarray) -> Any:
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    with PoseLandmarker.create_from_options(options) as landmarker:
        trace = run_pose(read_rgb_frames(path), info, landmarker, windows, make_image=make_image)
    return info, trace


# --------------------------------------------------------------------------- plot
def plot_clip(trace: PoseTrace, metrics: ClipMetrics, title: str, out_png: Path) -> None:
    """Save a three-panel plot: detection, world ankle heights, image ankle heights."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(trace.n_frames) / trace.fps
    traces = ankle_traces(trace)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    legs = np.where(
        trace.detected, (trace.visibility[:, list(LEG_IDX)] >= VISIBLE_THRESHOLD).mean(1), 0
    )
    axes[0].fill_between(
        t, 0, trace.detected.astype(float), step="mid", alpha=0.25, label="person found"
    )
    axes[0].fill_between(
        t,
        0,
        clean_mask(trace).astype(float),
        step="mid",
        alpha=0.35,
        color="tab:green",
        label="clean frames (cycles counted here)",
    )
    axes[0].plot(t, legs, lw=1, label="leg landmarks visible")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("share")
    axes[0].legend(loc="lower right", fontsize=8)

    for ax, space, label, cyc in (
        (axes[1], "world", "ankle height vs hips (m)", (metrics.world_left, metrics.world_right)),
        (
            axes[2],
            "image",
            "ankle height in frame (fraction)",
            (metrics.image_left, metrics.image_right),
        ),
    ):
        for side, res, colour in zip(("left", "right"), cyc, ("tab:blue", "tab:red"), strict=True):
            y = traces[space][side]
            ax.plot(t, y, color=colour, lw=1, label=side)
            if res.minima.size:
                ax.plot(t[res.minima], y[res.minima], "v", color=colour, ms=6)
        ax.set_ylabel(label)
        ax.legend(loc="upper right", fontsize=8)
    axes[2].set_xlabel("time (s)")
    verdict = "PASS" if metrics.passed else "FAIL"
    fig.suptitle(
        f"{title}: {verdict}, found {metrics.detection_rate:.0%} of frames "
        f"({metrics.in_view_rate:.0%} once in view), {metrics.clean_s:.1f} s clean, "
        f"{metrics.gait_cycles} cycles"
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
