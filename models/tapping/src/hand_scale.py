"""Hand-size normalisation of the thumb-index distance (task FT-13).

The raw distance of FT-12 is in pixels, so it grows when the hand moves toward the camera and
shrinks when it moves away. Dividing it by a length that belongs to the hand itself, and not to
the fingers that tap, removes that. Without it a model would learn camera distance.

Pieces:

- ``scale_length(df, scale, space)``: one length per frame (NaN without a hand) for a named
  candidate (``SCALES``), in pixels or, from the world landmarks, in metres;
- ``estimate_scale(df, scale, mode)``: the length actually used to divide, in one of three
  modes: ``per_frame``, ``rolling_median`` (centred window in seconds of ``pts_ms``) or
  ``clip_median`` (one number per clip);
- ``normalised_distance(df, scale, mode)``: thumb-index distance divided by that length, with
  the scale and mode recorded in the result so that the batch runner can store them.

Rules:

- The scale never uses the thumb tip (4) or the index tip (8), so it cannot move with the tap.
  It uses only the wrist (0) and the knuckles (5, 9, 13, 17), which belong to the palm.
- Missing is NaN, never zero or one. A frame without a hand, without a scale, or with a scale
  that fails ``MIN_SCALE_FRACTION`` gives NaN. Nothing is filled or smoothed here: the rolling
  median is a scale estimate, not a smoothing of the signal.
- Time is ``pts_ms`` (container time) as in FT-12.
- Nothing here looks at UPDRS or at the CONTROL/ID group. Everything is per clip and uses only
  the geometry of that clip, so there is nothing to fit inside a cross-validation fold.

Limits (normalisation cannot fix these): rotation of the hand out of the image plane shrinks
every 2D length (foreshortening); the closed floor of the thumb-index distance is never zero
because the landmarks sit short of the nail; the landmark positions themselves are approximate.

Plots go to ``data/interim/plots/`` (gitignored), named and titled by the clip's short hash.

Usage (from the repo root):
    python models/tapping/src/hand_scale.py --clip <8-character hash>
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from hand_landmarks import HandLandmarksError, landmark_column
from hubu_fis import pseudonym
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from plot_distance import (
    DEFAULT_LANDMARKS_DIR,
    DEFAULT_OUT_DIR,
    find_parquet,
    load_landmarks,
    save_figure,
    thumb_index_distance_px,
    time_axis_s,
)

WRIST = 0
INDEX_KNUCKLE = 5
MIDDLE_KNUCKLE = 9
RING_KNUCKLE = 13
LITTLE_KNUCKLE = 17
FORBIDDEN_LANDMARKS = (4, 8)  # thumb tip and index tip: they are the tap itself

MODES = ("per_frame", "rolling_median", "clip_median")
SPACES = ("pixel", "world")
DEFAULT_SCALE = "w9"
DEFAULT_MODE = "rolling_median"
DEFAULT_WINDOW_S = 3.0
MIN_WINDOW_SAMPLES = 5  # a rolling window with fewer scale values than this gives NaN
MIN_SCALE_FRACTION = 0.25  # a frame scale below this share of the clip median is rejected

# A scale is a function of the per-pair lengths; every pair lists landmarks 0/5/9/13/17 only.
_PAIRS: dict[str, tuple[tuple[int, int], ...]] = {
    "w9": ((WRIST, MIDDLE_KNUCKLE),),
    "w5": ((WRIST, INDEX_KNUCKLE),),
    "w17": ((WRIST, LITTLE_KNUCKLE),),
    "w5_17_mean": ((WRIST, INDEX_KNUCKLE), (WRIST, LITTLE_KNUCKLE)),
    "palm_width": ((INDEX_KNUCKLE, LITTLE_KNUCKLE),),
    "median4": (
        (WRIST, INDEX_KNUCKLE),
        (WRIST, MIDDLE_KNUCKLE),
        (WRIST, RING_KNUCKLE),
        (WRIST, LITTLE_KNUCKLE),
    ),
}
_COMBINE: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "w5_17_mean": lambda lengths: lengths.mean(axis=0),
    "median4": lambda lengths: np.median(lengths, axis=0),
}
SCALES = tuple(_PAIRS)
assert not {i for pairs in _PAIRS.values() for pair in pairs for i in pair} & set(
    FORBIDDEN_LANDMARKS
)


class NormalisedDistance(NamedTuple):
    """The normalised signal and what was done to make it (for the batch runner to store)."""

    values: np.ndarray
    scale: str
    mode: str
    window_s: float | None
    scale_rejected_frames: int
    scale_median_px: float

    def metadata(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "mode": self.mode,
            "window_s": self.window_s,
            "scale_rejected_frames": self.scale_rejected_frames,
            "scale_median_px": self.scale_median_px,
            "unit": "thumb-index distance in units of the scale length",
        }


# --------------------------------------------------------------------------- the scale


def _coords(df: pd.DataFrame, index: int, space: str) -> np.ndarray:
    prefixes = ("x_px", "y_px") if space == "pixel" else ("wx", "wy", "wz")
    return np.column_stack([df[landmark_column(p, index)].to_numpy(dtype=float) for p in prefixes])


def _pair_length(df: pd.DataFrame, a: int, b: int, space: str) -> np.ndarray:
    return np.asarray(np.linalg.norm(_coords(df, a, space) - _coords(df, b, space), axis=1))


def _check(scale: str, mode: str | None = None, space: str = "pixel") -> None:
    if scale not in _PAIRS:
        raise ValueError(f"scale must be one of {SCALES}, got {scale!r}")
    if mode is not None and mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if space not in SPACES:
        raise ValueError(f"space must be one of {SPACES}, got {space!r}")


def scale_length(df: pd.DataFrame, scale: str = DEFAULT_SCALE, space: str = "pixel") -> np.ndarray:
    """One length per frame for a candidate scale; NaN where there is no hand or a NaN input.

    ``space="pixel"``: pixels of the displayed frame. ``space="world"``: metres from the
    MediaPipe world landmarks (a model prediction with a hand-shape prior, not a measurement;
    use it as a reference, not as a pipeline input). Combined scales need every landmark.
    """
    _check(scale, space=space)
    prefixes = ("x_px", "y_px") if space == "pixel" else ("wx", "wy", "wz")
    needed = ["detected"] + [
        landmark_column(p, i) for pair in _PAIRS[scale] for i in pair for p in prefixes
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise HandLandmarksError(f"the landmarks table lacks columns {sorted(set(missing))[:4]}")
    lengths = np.vstack([_pair_length(df, a, b, space) for a, b in _PAIRS[scale]])
    out = _COMBINE[scale](lengths) if scale in _COMBINE else lengths[0]
    out = np.asarray(out, dtype=float)
    out[~df["detected"].to_numpy(dtype=bool)] = np.nan
    return out


def _guard(per_frame: np.ndarray) -> tuple[np.ndarray, int]:
    """Reject non-finite, non-positive and absurdly small scales; return the series and a count.

    "Absurdly small" is below ``MIN_SCALE_FRACTION`` of the clip's own median scale (landmarks
    collapsed on each other). The rule is relative so that it holds in pixels and in metres.
    If the clip has no valid scale at all, every frame is rejected.
    """
    scale = per_frame.copy()
    valid = np.isfinite(scale) & (scale > 0)
    if not valid.any():
        return np.full_like(scale, np.nan), int(np.isfinite(per_frame).sum())
    floor = MIN_SCALE_FRACTION * float(np.median(scale[valid]))
    valid &= scale >= floor
    rejected = int((np.isfinite(per_frame) & ~valid).sum())
    scale[~valid] = np.nan
    return scale, rejected


def _rolling_median(t: np.ndarray, values: np.ndarray, window_s: float) -> np.ndarray:
    """Median of the finite values within +/- window_s/2 seconds of each frame (centred).

    A window with fewer than ``MIN_WINDOW_SAMPLES`` finite values gives NaN. At the clip ends
    the window is cut short, not padded. Frames without a time give NaN.
    """
    half = window_s / 2.0
    out = np.full(len(values), np.nan)
    finite = np.isfinite(values) & np.isfinite(t)
    for i in np.flatnonzero(np.isfinite(t)):
        inside = finite & (np.abs(t - t[i]) <= half)
        if inside.sum() >= MIN_WINDOW_SAMPLES:
            out[i] = np.median(values[inside])
    return out


def estimate_scale(
    df: pd.DataFrame,
    scale: str = DEFAULT_SCALE,
    mode: str = DEFAULT_MODE,
    *,
    window_s: float = DEFAULT_WINDOW_S,
) -> tuple[np.ndarray, int]:
    """The length to divide by, one value per frame, plus how many frame scales were rejected.

    - ``per_frame``: this frame's own length. Follows approach and recession exactly, but a
      single bad landmark moves that frame's value, and any coupling to the tap goes straight
      into the signal.
    - ``rolling_median``: median of the valid lengths within a centred window of ``window_s``
      seconds of ``pts_ms``. Follows slow approach and recession, rejects a lone bad frame and
      averages out a tap-locked wobble. A frame without a hand can still get a window value;
      its normalised distance is NaN anyway because its distance is NaN.
    - ``clip_median``: one number for the clip. Very stable, but it cannot follow a hand that
      moves toward or away from the camera.
    """
    _check(scale, mode)
    if not (np.isfinite(window_s) and window_s > 0):
        raise ValueError(f"window_s must be a positive number of seconds, got {window_s!r}")
    guarded, rejected = _guard(scale_length(df, scale))
    if mode == "per_frame":
        return guarded, rejected
    if mode == "clip_median":
        finite = guarded[np.isfinite(guarded)]
        value = float(np.median(finite)) if finite.size else float("nan")
        return np.full(len(guarded), value), rejected
    t, _ = time_axis_s(df)
    return _rolling_median(t, guarded, window_s), rejected


def normalised_distance(
    df: pd.DataFrame,
    scale: str = DEFAULT_SCALE,
    mode: str = DEFAULT_MODE,
    *,
    window_s: float = DEFAULT_WINDOW_S,
) -> NormalisedDistance:
    """Thumb-index distance (landmarks 4 and 8) divided by the chosen hand scale.

    NaN in, NaN out: a frame without a distance or without a valid scale is NaN, never zero.
    """
    divisor, rejected = estimate_scale(df, scale, mode, window_s=window_s)
    distance = thumb_index_distance_px(df)
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.where(np.isfinite(divisor) & (divisor > 0), distance / divisor, np.nan)
    finite = divisor[np.isfinite(divisor)]
    return NormalisedDistance(
        values=values,
        scale=scale,
        mode=mode,
        window_s=window_s if mode == "rolling_median" else None,
        scale_rejected_frames=rejected,
        scale_median_px=float(np.median(finite)) if finite.size else float("nan"),
    )


def thumb_index_distance_world_cm(df: pd.DataFrame) -> np.ndarray:
    """Thumb-tip to index-tip distance from the world landmarks, in centimetres (reference)."""
    out = _pair_length(df, 4, 8, "world") * 100.0
    out[~df["detected"].to_numpy(dtype=bool)] = np.nan
    return out


# --------------------------------------------------------------------------- figure


def plot_normalisation(
    df: pd.DataFrame,
    ref: str,
    *,
    scale: str = DEFAULT_SCALE,
    mode: str = DEFAULT_MODE,
    window_s: float = DEFAULT_WINDOW_S,
) -> Figure:
    """Raw pixels, the hand-size series, and the normalised signal on one time axis."""
    t, x_label = time_axis_s(df)
    raw = thumb_index_distance_px(df)
    per_frame = scale_length(df, scale)
    used, _ = estimate_scale(df, scale, mode, window_s=window_s)
    result = normalised_distance(df, scale, mode, window_s=window_s)

    fig = Figure(figsize=(12, 8.5), dpi=110, layout="constrained")
    FigureCanvasAgg(fig)
    top, mid, bottom = fig.subplots(3, 1, sharex=True)
    top.plot(t, raw, lw=0.8, color="tab:blue")
    top.set_ylabel("raw distance (px)")
    top.set_title(f"Hand-size normalisation, clip {ref}: {scale}, {mode}")
    mid.plot(t, per_frame, lw=0.6, color="gray", label=f"{scale} per frame")
    mid.plot(t, used, lw=1.6, color="tab:red", label=f"{scale} used ({mode})")
    mid.set_ylabel("hand size (px)")
    mid.legend(loc="upper right", fontsize=8)
    bottom.plot(t, result.values, lw=0.8, color="tab:green")
    bottom.set_ylabel("normalised distance (hand sizes)")
    bottom.set_xlabel(x_label)
    for ax, data in ((top, raw), (mid, per_frame), (bottom, result.values)):
        ax.grid(alpha=0.25)
        if np.isfinite(data).any():
            ax.set_ylim(0, float(np.nanmax(data)) * 1.1)
    return fig


# --------------------------------------------------------------------------- command line


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--clip", required=True, help="8-character hash of the clip ID")
    ap.add_argument("--scale", choices=SCALES, default=DEFAULT_SCALE)
    ap.add_argument("--mode", choices=MODES, default=DEFAULT_MODE)
    ap.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S)
    ap.add_argument("--landmarks-dir", type=Path, default=DEFAULT_LANDMARKS_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args(argv)

    path = find_parquet(args.clip, args.landmarks_dir)
    if path is None:
        print(
            f"error: no landmarks file for clip {args.clip} in {args.landmarks_dir}",
            file=sys.stderr,
        )
        return 2
    ref = pseudonym(path.stem)
    try:
        df = load_landmarks(path, ref)
        result = normalised_distance(df, args.scale, args.mode, window_s=args.window_s)
    except (HandLandmarksError, KeyError, ValueError) as err:
        print(f"FAILED: clip {ref}: {err}", file=sys.stderr)
        return 1

    finite = result.values[np.isfinite(result.values)]
    print(f"clip {ref}: scale {args.scale}, mode {args.mode}, {result.metadata()}")
    if finite.size:
        low, median, high = np.percentile(finite, [5, 50, 95])
        print(
            f"  normalised distance (hand sizes): p5 {low:.2f} | median {median:.2f} | p95 {high:.2f}"
        )
    out_png = args.out_dir / f"{ref}_normalised_{args.scale}_{args.mode}.png"
    save_figure(
        plot_normalisation(df, ref, scale=args.scale, mode=args.mode, window_s=args.window_s),
        out_png,
    )
    print(f"saved {out_png.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
