"""Plot the RAW thumb-tip to index-tip distance of one clip (task FT-12).

Reads a landmarks parquet written by ``hand_landmarks.py`` (FT-11) and draws the distance
between landmark 4 (thumb tip) and landmark 8 (index tip) in pixels of the displayed frame,
frame by frame, against time. The signal is RAW: no smoothing, no interpolation of gaps, no
peak detection, no normalisation. Those belong to later tasks (FT-13, FT-19, FT-20).

Reusable pieces (FT-13 should start from these):

- ``thumb_index_distance_px(df)``: one value per frame, NaN where there is no hand;
- ``hand_size_px(df)``: median wrist (0) to knuckle (9) distance, a reference scale only;
- ``time_axis_s(df)``: time in seconds from ``pts_ms``, the container's own clock.

Time axis: ``pts_ms``, never ``timestamp_ms`` (the tidy index / fps counter given to
MediaPipe). On irregular clips the two differ by tens of milliseconds. If ``pts_ms`` is
missing for every frame the plot falls back to ``timestamp_ms`` and the axis label says so;
if it is missing for some frames those frames are left off the plot and counted. The two
clocks are never mixed in one line.

Gaps: a frame without a hand has distance NaN. In the plot the line breaks there and a red
tick marks the frame along the bottom edge; nothing is ever drawn at zero.

Hand size appears only as a dashed reference line and a labelled right-hand axis that shows
the same raw curve in hand sizes. It never replaces the pixel signal.

Plots go to ``data/interim/plots/`` (gitignored). Titles, axis text and file names carry only
the short hash of the clip ID (``hubu_fis.pseudonym``), never the raw ID. Figures are drawn
with ``matplotlib.figure.Figure`` and an Agg canvas, so nothing needs a display.

Exit codes: 0 ok, 1 the landmarks file cannot be read or lacks the needed columns, 2 the
landmarks file is missing.

Usage (from the repo root):
    python models/tapping/src/plot_distance.py --clip <8-character hash>
    python models/tapping/src/plot_distance.py --parquet path/to/landmarks.parquet
    python models/tapping/src/plot_distance.py --clip <hash> --compare-clocks
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from hand_landmarks import HandLandmarksError, landmark_column, read_landmarks
from hubu_fis import pseudonym
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_KNUCKLE = 9

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LANDMARKS_DIR = REPO_ROOT / "data" / "interim" / "landmarks"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "interim" / "plots"

LOW_LEVEL = 0.25  # cycle counter: the distance must fall below this share of the range ...
HIGH_LEVEL = 0.5  # ... and then rise above this share (range = 5th to 95th percentile)
ZOOM_SECONDS = 4.0

REQUIRED_COLUMNS = (
    "detected",
    "pts_ms",
    "timestamp_ms",
    *(
        landmark_column(prefix, i)
        for i in (WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_KNUCKLE)
        for prefix in ("x_px", "y_px")
    ),
)


# --------------------------------------------------------------------------- the signal


def _xy(df: pd.DataFrame, index: int) -> tuple[np.ndarray, np.ndarray]:
    x = df[landmark_column("x_px", index)].to_numpy(dtype=float)
    y = df[landmark_column("y_px", index)].to_numpy(dtype=float)
    return x, y


def thumb_index_distance_px(
    df: pd.DataFrame, thumb: int = THUMB_TIP, index: int = INDEX_TIP
) -> np.ndarray:
    """Distance between two landmarks in pixels, one value per frame, NaN without a hand.

    Raw: no smoothing, no gap filling. A frame with ``detected = False`` is NaN even if
    coordinates were present, and a NaN coordinate gives NaN, never zero.
    """
    tx, ty = _xy(df, thumb)
    ix, iy = _xy(df, index)
    distance = np.hypot(tx - ix, ty - iy)
    distance[~df["detected"].to_numpy(dtype=bool)] = np.nan
    return distance


def hand_size_px(df: pd.DataFrame) -> float:
    """Median wrist (0) to knuckle (9) distance over detected frames; NaN if there are none.

    A reference scale only. It is never used to change the raw signal.
    """
    wx, wy = _xy(df, WRIST)
    kx, ky = _xy(df, MIDDLE_KNUCKLE)
    size = np.hypot(wx - kx, wy - ky)[df["detected"].to_numpy(dtype=bool)]
    size = size[np.isfinite(size)]
    return float(np.median(size)) if size.size else float("nan")


def time_axis_s(df: pd.DataFrame, clock: str = "pts") -> tuple[np.ndarray, str]:
    """Time in seconds per frame, and the axis label that says which clock it is.

    ``clock="pts"`` (default) uses the container time ``pts_ms``; ``clock="index"`` uses
    ``timestamp_ms``. If ``pts_ms`` is missing for every frame, ``pts`` falls back to the
    index clock and the label says so. Frames with a missing ``pts_ms`` get NaN time.
    """
    if clock not in ("pts", "index"):
        raise ValueError(f"clock must be 'pts' or 'index', got {clock!r}")
    pts = df["pts_ms"].to_numpy(dtype=float) / 1000.0
    if clock == "pts" and np.isfinite(pts).any():
        return pts, "time from the container timestamps pts_ms (s)"
    index_clock = df["timestamp_ms"].to_numpy(dtype=float) / 1000.0
    if clock == "pts":
        return index_clock, "time from the index/fps clock timestamp_ms (s); pts_ms was missing"
    return index_clock, "time from the index/fps clock timestamp_ms (s)"


# --------------------------------------------------------------------------- exploratory numbers


def count_open_close_cycles(distance: np.ndarray) -> int:
    """Rough count of open/close cycles in a raw distance signal. NOT the tap detector.

    Method: take the clip's own range from the 5th to the 95th percentile of the finite
    values; count each time the distance falls below 25% of that range and then rises above
    the midpoint (50%). Frames without a value are skipped, not filled. No smoothing, so
    jitter near the thresholds is tolerated only by the gap between the two levels.

    Why 50% and not 75%: with a 75% level the main FT-12 clip was counted as 9 cycles while 12
    are visible, because its peak heights vary by about 14% and three of the lower peaks stay
    under the 75% level. On the four clips looked at, 25/50 and 40/60 gave the same counts
    (12, 28, 21, 12); the count does depend on the thresholds.

    Limits: the thresholds depend on the clip itself, a tap that never rises above the
    midpoint or never falls below the 25% level is missed, and a clip that starts open is
    counted from its first close.
    """
    finite = distance[np.isfinite(distance)]
    if finite.size < 2:
        return 0
    low, high = np.percentile(finite, [5, 95])
    if high <= low:
        return 0
    below = low + LOW_LEVEL * (high - low)
    above = low + HIGH_LEVEL * (high - low)
    cycles, closed = 0, False
    for value in finite:
        if value <= below:
            closed = True
        elif value >= above and closed:
            cycles += 1
            closed = False
    return cycles


def exploratory_numbers(df: pd.DataFrame) -> dict[str, Any]:
    """Simple numbers for the raw signal. Hand-size values are reference only."""
    distance = thumb_index_distance_px(df)
    finite = distance[np.isfinite(distance)]
    t, _ = time_axis_s(df)
    t_finite = t[np.isfinite(t)]
    duration = float(t_finite.max() - t_finite.min()) if t_finite.size else float("nan")
    hand = hand_size_px(df)
    stats: dict[str, Any] = {
        "frames": len(df),
        "nan_frames": int((~np.isfinite(distance)).sum()),
        "pts_missing": int(df["pts_ms"].isna().sum()),
        "duration_s": duration,
        "hand_size_px": hand,
    }
    if finite.size:
        low, median, high = float(finite.min()), float(np.median(finite)), float(finite.max())
        stats.update(min_px=low, median_px=median, max_px=high)
        stats.update(min_hand=low / hand, median_hand=median / hand, max_hand=high / hand)
    cycles = count_open_close_cycles(distance)
    stats["cycles"] = cycles
    stats["cycles_per_second"] = cycles / duration if duration > 0 else float("nan")
    gaps = np.diff(df["pts_ms"].to_numpy(dtype=float))
    stats["pts_gap_ms_min"] = float(np.nanmin(gaps)) if np.isfinite(gaps).any() else float("nan")
    stats["pts_gap_ms_max"] = float(np.nanmax(gaps)) if np.isfinite(gaps).any() else float("nan")
    return stats


def format_numbers(ref: str, s: dict[str, Any]) -> str:
    def f(key: str, unit: str = "", digits: int = 1) -> str:
        return f"{s[key]:.{digits}f}{unit}" if key in s else "n/a"

    return "\n".join(
        [
            f"clip {ref}: {s['frames']} frames, {f('duration_s', ' s')} from pts_ms, "
            f"{s['nan_frames']} frames without a distance (NaN), {s['pts_missing']} without pts_ms",
            f"  raw distance, pixels: min {f('min_px')} | median {f('median_px')} | "
            f"max {f('max_px')}",
            f"  same in hand sizes (reference only; hand size = median wrist-to-knuckle "
            f"{f('hand_size_px', ' px', 0)}): min {f('min_hand', digits=2)} | "
            f"median {f('median_hand', digits=2)} | max {f('max_hand', digits=2)}",
            f"  rough open/close cycles: {s['cycles']} = {f('cycles_per_second', ' per second', 2)} "
            "(simple 25%/50% crossing count on the raw signal, not the tap detector)",
            f"  frame gaps from pts_ms: {f('pts_gap_ms_min', ' ms')} to {f('pts_gap_ms_max', ' ms')}",
        ]
    )


# --------------------------------------------------------------------------- figures


def _new_figure(height: float = 4.8) -> Figure:
    fig = Figure(figsize=(12, height), dpi=110, layout="constrained")
    FigureCanvasAgg(fig)  # Agg canvas: drawing and saving need no display
    return fig


def _draw_gaps(ax: Any, t: np.ndarray, distance: np.ndarray) -> None:
    """Red ticks along the bottom edge (axes coordinates) at frames without a distance."""
    missing = ~np.isfinite(distance) & np.isfinite(t)
    if missing.any():
        ax.vlines(
            t[missing],
            0,
            0.04,
            transform=ax.get_xaxis_transform(),
            color="tab:red",
            lw=0.8,
            label=f"no hand detected ({int(missing.sum())} frames, line breaks here)",
        )


def plot_distance(df: pd.DataFrame, ref: str, *, clock: str = "pts") -> Figure:
    """The raw distance in pixels against time. ``ref`` is the clip's short hash."""
    distance = thumb_index_distance_px(df)
    t, x_label = time_axis_s(df, clock)
    hand = hand_size_px(df)
    fig = _new_figure()
    ax = fig.add_subplot()
    ax.plot(t, distance, lw=0.8, marker=".", ms=2.5, color="tab:blue", label="raw distance")
    _draw_gaps(ax, t, distance)
    if np.isfinite(hand):
        ax.axhline(
            hand,
            ls="--",
            lw=0.8,
            color="gray",
            label=f"median hand size, wrist to knuckle 9 (reference only): {hand:.0f} px",
        )

        def to_hand_sizes(y: ArrayLike) -> np.ndarray:
            return np.asarray(y, dtype=float) / hand

        def to_pixels(y: ArrayLike) -> np.ndarray:
            return np.asarray(y, dtype=float) * hand

        secondary = ax.secondary_yaxis("right", functions=(to_hand_sizes, to_pixels))
        secondary.set_ylabel("the same raw curve in hand sizes (reference scale only)")
    if np.isfinite(distance).any():
        ax.set_ylim(0, float(np.nanmax(distance)) * 1.08)
    ax.set_xlabel(x_label)
    ax.set_ylabel("thumb tip (4) to index tip (8), pixels of the displayed frame (raw)")
    ax.set_title(f"Raw thumb-index distance, clip {ref} ({len(df)} frames, no smoothing)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    return fig


def plot_clock_comparison(df: pd.DataFrame, ref: str) -> Figure:
    """The same raw signal on both clocks: full clip, and a zoom where they differ most."""
    distance = thumb_index_distance_px(df)
    t_pts, _ = time_axis_s(df, "pts")
    t_idx, _ = time_axis_s(df, "index")
    shift_ms = np.abs(t_pts - t_idx) * 1000.0
    worst = int(np.nanargmax(shift_ms)) if np.isfinite(shift_ms).any() else 0
    centre = float(t_idx[worst]) if np.isfinite(t_idx[worst]) else 0.0

    fig = _new_figure(height=7.5)
    top, bottom = fig.subplots(2, 1)
    for ax in (top, bottom):
        ax.plot(t_pts, distance, lw=0.9, color="tab:blue", label="against pts_ms (container time)")
        ax.plot(
            t_idx, distance, lw=0.9, ls="--", color="tab:orange", label="against index/fps clock"
        )
        ax.grid(alpha=0.25)
        ax.set_ylabel("raw distance, pixels")
    # a window of ZOOM_SECONDS around that frame, moved so that it stays inside the clip
    finite_t = t_idx[np.isfinite(t_idx)]
    start, stop = (float(finite_t.min()), float(finite_t.max())) if finite_t.size else (0.0, 1.0)
    width = min(ZOOM_SECONDS, stop - start)
    left = float(np.clip(centre - width / 2, start, stop - width))
    bottom.set_xlim(left, left + width)
    top.set_title(
        f"Raw thumb-index distance on two clocks, clip {ref} "
        f"(largest difference {np.nanmax(shift_ms) if np.isfinite(shift_ms).any() else float('nan'):.1f} ms)"
    )
    bottom.set_title(f"zoom on {ZOOM_SECONDS:g} s around the frame where the clocks differ most")
    bottom.set_xlabel("time (s)")
    top.legend(loc="upper right", fontsize=8)
    return fig


def save_figure(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)


# --------------------------------------------------------------------------- command line


def find_parquet(ref: str, landmarks_dir: Path) -> Path | None:
    """The landmarks file whose clip ID has this short hash, or None."""
    for path in sorted(Path(landmarks_dir).glob("*.parquet")):
        if pseudonym(path.stem) == ref:
            return path
    return None


def load_landmarks(path: Path, ref: str) -> pd.DataFrame:
    try:
        df, _ = read_landmarks(path)
    except (HandLandmarksError, pa.ArrowException, OSError) as err:
        raise HandLandmarksError(f"clip {ref}: cannot read the landmarks file ({err})") from err
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise HandLandmarksError(f"clip {ref}: the landmarks file lacks columns {missing[:4]}")
    if len(df) == 0:
        raise HandLandmarksError(f"clip {ref}: the landmarks file has no frames")
    return df


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--clip", help="8-character hash of the clip ID")
    source.add_argument("--parquet", type=Path, help="a landmarks file from hand_landmarks.py")
    ap.add_argument("--landmarks-dir", type=Path, default=DEFAULT_LANDMARKS_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--compare-clocks", action="store_true", help="also plot pts_ms vs index/fps")
    args = ap.parse_args(argv)

    if args.clip is not None:
        path = find_parquet(args.clip, args.landmarks_dir)
        if path is None:
            print(
                f"error: no landmarks file for clip {args.clip} in {args.landmarks_dir}",
                file=sys.stderr,
            )
            return 2
    else:
        path = args.parquet
        if not path.is_file():
            print(f"error: clip {pseudonym(path.stem)}: landmarks file not found", file=sys.stderr)
            return 2
    ref = pseudonym(path.stem)

    try:
        df = load_landmarks(path, ref)
    except HandLandmarksError as err:
        print(f"FAILED: {err}", file=sys.stderr)
        return 1

    print(format_numbers(ref, exploratory_numbers(df)))
    out_png = args.out_dir / f"{ref}_distance.png"
    save_figure(plot_distance(df, ref), out_png)
    print(f"saved {out_png.name}")
    if args.compare_clocks:
        clocks_png = args.out_dir / f"{ref}_distance_clocks.png"
        save_figure(plot_clock_comparison(df, ref), clocks_png)
        print(f"saved {clocks_png.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
