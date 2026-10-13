"""Landmark-overlay GIF of a short tapping window, for a proof-of-concept slide (task FT-14).

Draws the tracked hand skeleton (from the FT-11 landmarks parquet) on the clip, cropped to the
hand, with a live trace of the normalised thumb-index distance (FT-13, ``w9`` with a 3 s
rolling median) underneath and a cursor that moves with the video. Thumb tip (4) and index
tip (8) are highlighted. It is a proof-of-concept visual, not a result.

Rules:

- **Timing follows ``pts_ms``** (container time), never index/fps. The frames of the window are
  those whose ``pts_ms`` lies in ``[start, start + duration)``; with ``--fps`` lower than the
  clip's rate every ``stride``-th frame is kept. GIF delays are stored in 10 ms units, so each
  frame's delay is the gap to the next kept frame, rounded on a running total (the sum of the
  delays stays within 5 ms of the true span, and no delay is below 20 ms). Rates with exact
  delays are 50, 25, 20 and 10 fps; at 30 fps the delays alternate 30/30/40 ms.
- **No stale or zero skeletons.** A frame without a hand shows the picture and no skeleton, and
  its trace value is NaN (the line breaks).
- **Privacy.** No label is drawn: no rating, no group, no clip ID, no wording about disease.
  The output is named by the clip's 8-character hash only. The picture is cropped to the hand
  (``--blur-outside`` additionally blurs everything outside the hand). Always look at the
  result before it is shown in public.
- **Limits** (enforced): width 160 to 640 px, at most 6 s, at most 150 frames, at most 5 MB.
- Output stays inside ``data/interim`` (gitignored); nothing is tracked.

The clip is opened with ``hand_landmarks.open_video`` and OpenCV auto-rotation is switched on
and read back, as in FT-11. Every decoded frame's container time is checked against the
parquet's ``pts_ms``, so an off-by-one frame cannot slip through.

Exit codes: 0 ok, 1 unreadable clip or landmarks, or a limit was exceeded, 2 missing input.

Usage (from the repo root):
    python models/tapping/src/overlay_gif.py --clip <8-character hash> --start-s 3 --duration-s 4
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
from hand_landmarks import (
    N_LANDMARKS,
    HandLandmarksError,
    landmark_column,
    open_video,
    read_landmarks,
)
from hand_scale import normalised_distance
from hubu_fis import pseudonym
from numpy.typing import ArrayLike
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from plot_distance import find_parquet

REPO_ROOT = Path(__file__).resolve().parents[3]
INTERIM_ROOT = REPO_ROOT / "data" / "interim"
DEFAULT_LANDMARKS_DIR = INTERIM_ROOT / "landmarks"
DEFAULT_VIDEO_DIR = REPO_ROOT / "data" / "raw" / "HUBU-FIS" / "videos"
DEFAULT_OUT_DIR = INTERIM_ROOT / "gifs"

MIN_WIDTH, MAX_WIDTH = 160, 640
MAX_DURATION_S = 6.0
MAX_FRAMES = 150
MAX_BYTES = 5 * 1024 * 1024
MIN_DELAY_MS = 20  # many viewers treat smaller GIF delays as 100 ms
PALETTE_COLOURS = 72
TRACE_HEIGHT_RATIO = 0.22
CROP_PAD_HAND_SIZES = 0.6  # empty margin around the hand's reach, in wrist-to-knuckle lengths
PTS_TOLERANCE_MS = 1.0

THUMB_TIP, INDEX_TIP = 4, 8
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)  # fmt: skip

BONE = (0, 230, 110)
JOINT = (255, 255, 255)
THUMB_COLOUR = (255, 140, 0)
INDEX_COLOUR = (255, 0, 200)
LINK = (255, 235, 0)
PANEL = (28, 28, 34)
TRACE_DIM = (110, 110, 125)
TRACE_LIVE = (120, 220, 255)
TEXT = (225, 225, 230)

CaptureFactory = Callable[[str], Any]


class OverlayError(ValueError):
    """The clip, the window or a size limit is not acceptable (command line exit code 1)."""


class InputMissingError(FileNotFoundError):
    """A landmarks file or video is missing (command line exit code 2)."""


# --------------------------------------------------------------------------- window and timing


def select_window(pts_ms: np.ndarray, start_s: float, duration_s: float) -> np.ndarray:
    """Indices of the frames whose container time is in [start, start + duration)."""
    pts = np.asarray(pts_ms, dtype=float)
    lo, hi = start_s * 1000.0, (start_s + duration_s) * 1000.0
    with np.errstate(invalid="ignore"):
        inside = (pts >= lo) & (pts < hi)
    return np.flatnonzero(inside)


def stride_for(source_fps: float, target_fps: float) -> int:
    """Keep every n-th frame so that the rate is close to ``target_fps`` (never above the clip)."""
    return max(1, int(round(source_fps / target_fps)))


def gif_delays_ms(pts_ms: ArrayLike) -> list[int]:
    """GIF delay per kept frame, in whole 10 ms units, from the frames' container times.

    The delay of a frame is the gap to the next kept frame; the last frame repeats the median
    gap. Rounding is done on the running total, so the sum stays within 5 ms of the true span
    (plus the last frame's delay) and irregular gaps are kept. No delay is below 20 ms.
    """
    pts = np.asarray(pts_ms, dtype=float)
    if pts.size == 0:
        return []
    if pts.size == 1:
        return [MIN_DELAY_MS]
    gaps = np.diff(pts)
    if not (np.isfinite(pts).all() and (gaps > 0).all()):
        raise OverlayError("the frame times of the window are missing or not increasing")
    starts = pts - pts[0]
    ends = np.append(starts, starts[-1] + float(np.median(gaps)))
    cum = [0]
    for t in ends[1:]:
        cum.append(max(int(math.floor(t / 10.0 + 0.5)) * 10, cum[-1] + MIN_DELAY_MS))
    return [b - a for a, b in zip(cum[:-1], cum[1:], strict=True)]


# --------------------------------------------------------------------------- geometry


def landmark_xy(df: pd.DataFrame, index: int) -> np.ndarray:
    """(21, 2) pixel positions of one frame; NaN for a frame without a hand."""
    row = df.iloc[index]
    xy = np.array(
        [
            [row[landmark_column("x_px", i)], row[landmark_column("y_px", i)]]
            for i in range(N_LANDMARKS)
        ],
        dtype=float,
    )
    if not bool(row["detected"]):
        xy[:] = np.nan
    return xy


def crop_square(points: np.ndarray, frame_w: int, frame_h: int, pad: float) -> tuple[int, int, int]:
    """(x0, y0, side) of a square crop around every point (NaN ignored) plus ``pad`` pixels.

    The square is as small as the hand's whole reach allows, never larger than the frame, and
    shifted to stay inside it. With no point at all it is the largest centred square.
    """
    finite = points[np.isfinite(points).all(axis=1)] if points.size else points
    limit = min(frame_w, frame_h)
    if len(finite) == 0:
        return (frame_w - limit) // 2, (frame_h - limit) // 2, limit
    lo, hi = finite.min(axis=0), finite.max(axis=0)
    side = int(min(limit, math.ceil(max(hi - lo) + 2 * pad)))
    centre = (lo + hi) / 2.0
    x0 = int(np.clip(round(centre[0] - side / 2), 0, frame_w - side))
    y0 = int(np.clip(round(centre[1] - side / 2), 0, frame_h - side))
    return x0, y0, side


def to_panel(xy: np.ndarray, box: tuple[int, int, int], width: int) -> np.ndarray:
    """Frame pixel positions to positions in the square panel of ``width`` pixels."""
    x0, y0, side = box
    return (np.asarray(xy, dtype=float) - np.array([x0, y0], dtype=float)) * (width / side)


# --------------------------------------------------------------------------- drawing (pure)


def draw_skeleton(img: Image.Image, xy: np.ndarray) -> None:
    """Draw bones and joints at ``xy`` (21, 2) on ``img`` in place; NaN points draw nothing.

    A bone is drawn only if both of its ends are finite, so a frame without a hand (all NaN)
    draws nothing at all: never a skeleton at (0, 0) and never a stale one.
    """
    draw = ImageDraw.Draw(img)
    ok = np.isfinite(xy).all(axis=1)
    for a, b in HAND_CONNECTIONS:
        if ok[a] and ok[b]:
            draw.line([tuple(xy[a]), tuple(xy[b])], fill=BONE, width=3)
    if ok[THUMB_TIP] and ok[INDEX_TIP]:
        draw.line([tuple(xy[THUMB_TIP]), tuple(xy[INDEX_TIP])], fill=LINK, width=2)
    for i in range(N_LANDMARKS):
        if not ok[i]:
            continue
        x, y = float(xy[i, 0]), float(xy[i, 1])
        if i in (THUMB_TIP, INDEX_TIP):
            colour, r = (THUMB_COLOUR if i == THUMB_TIP else INDEX_COLOUR), 7
        else:
            colour, r = JOINT, 3
        draw.ellipse([x - r, y - r, x + r, y + r], fill=colour, outline=(0, 0, 0))


def blur_outside(img: Image.Image, xy: np.ndarray, radius: int = 14) -> Image.Image:
    """``img`` with everything away from the hand blurred (the hand's outline is kept sharp)."""
    ok = np.isfinite(xy).all(axis=1)
    if not ok.any():
        return img.filter(ImageFilter.GaussianBlur(radius))
    hull = cv2.convexHull(xy[ok].astype(np.float32)).reshape(-1, 2)
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).polygon([tuple(p) for p in hull], fill=255)
    mask = mask.filter(ImageFilter.MaxFilter(2 * (img.size[0] // 12) + 1))
    mask = mask.filter(ImageFilter.GaussianBlur(radius))
    return Image.composite(img, img.filter(ImageFilter.GaussianBlur(radius)), mask)


def trace_ymax(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    top = float(finite.max()) if finite.size else 1.0
    return max(1.0, math.ceil(top * 11.0) / 10.0)


def draw_trace(
    size: tuple[int, int], values: np.ndarray, cursor: int, ymax: float, font: Any = None
) -> Image.Image:
    """The normalised distance over the window with a cursor at frame ``cursor``.

    The part up to the cursor is bright, the rest dim; NaN breaks the line. The vertical axis
    is 0 to ``ymax`` hand sizes and the same for every frame of the GIF.
    """
    width, height = size
    img = Image.new("RGB", size, PANEL)
    draw = ImageDraw.Draw(img)
    left, right, top, bottom = 8, width - 8, 22, height - 8
    n = len(values)

    def px(i: int) -> float:
        return left + (right - left) * (i / max(n - 1, 1))

    def py(v: float) -> float:
        return bottom - (bottom - top) * min(max(v / ymax, 0.0), 1.0)

    draw.line([(left, bottom), (right, bottom)], fill=TRACE_DIM, width=1)
    for i in range(n - 1):
        a, b = values[i], values[i + 1]
        if np.isfinite(a) and np.isfinite(b):
            colour = TRACE_LIVE if i + 1 <= cursor else TRACE_DIM
            draw.line([(px(i), py(float(a))), (px(i + 1), py(float(b)))], fill=colour, width=2)
    x = px(cursor)
    draw.line([(x, top - 2), (x, bottom)], fill=TEXT, width=1)
    if 0 <= cursor < n and np.isfinite(values[cursor]):
        y = py(float(values[cursor]))
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=TRACE_LIVE, outline=(0, 0, 0))
    draw.text((left, 3), "thumb-index distance, in hand sizes", fill=TEXT, font=font)
    draw.text((right - 28, top - 2), f"{ymax:.1f}", fill=TRACE_DIM, font=font)
    return img


def compose(
    frame: Image.Image,
    xy_panel: np.ndarray,
    trace: Image.Image,
    *,
    blur: bool = False,
) -> Image.Image:
    """The hand panel (with skeleton) above the trace panel."""
    panel = blur_outside(frame, xy_panel) if blur else frame.copy()
    draw_skeleton(panel, xy_panel)
    out = Image.new("RGB", (panel.width, panel.height + trace.height))
    out.paste(panel, (0, 0))
    out.paste(trace, (0, panel.height))
    return out


FIXED_COLOURS = (
    BONE,
    JOINT,
    THUMB_COLOUR,
    INDEX_COLOUR,
    LINK,
    PANEL,
    TRACE_DIM,
    TRACE_LIVE,
    TEXT,
    (0, 0, 0),
)


def to_palette(frames: Sequence[Image.Image], colours: int = PALETTE_COLOURS) -> list[Image.Image]:
    """Quantise every frame to one shared palette (no dithering).

    The palette is ``colours`` entries: the overlay's own colours exactly, so the highlighted
    tips cannot merge with skin tones, and the rest built from a few of the frames.
    """
    picks = [frames[i] for i in np.linspace(0, len(frames) - 1, num=min(6, len(frames)), dtype=int)]
    sheet = Image.new("RGB", (picks[0].width, picks[0].height * len(picks)))
    for k, p in enumerate(picks):
        sheet.paste(p, (0, k * p.height))
    adaptive = sheet.quantize(
        colors=colours - len(FIXED_COLOURS),
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE,
    )
    table = (adaptive.getpalette() or [])[: 3 * (colours - len(FIXED_COLOURS))]
    for colour in FIXED_COLOURS:
        table.extend(colour)
    palette = Image.new("P", (1, 1))
    palette.putpalette(table)
    return [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]


# --------------------------------------------------------------------------- reading the clip


def read_frames(
    cap: Any, wanted: Sequence[int], expected_pts: dict[int, float], size: tuple[int, int], ref: str
) -> dict[int, np.ndarray]:
    """Decode the clip from the start and keep the ``wanted`` frames (BGR).

    Each kept frame's container time must equal the parquet's ``pts_ms`` and the frame must
    have the parquet's displayed size, so a shifted frame or a missing rotation is an error.
    """
    if not cap.isOpened():
        raise OverlayError(f"clip {ref}: cannot be opened as a video")
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    if cap.get(cv2.CAP_PROP_ORIENTATION_AUTO) != 1:
        raise OverlayError(f"clip {ref}: auto-rotation could not be switched on")
    keep = set(wanted)
    last = max(keep)
    frames: dict[int, np.ndarray] = {}
    for index in range(last + 1):
        ok, frame = cap.read()
        if not ok:
            raise OverlayError(f"clip {ref}: the video ended at frame {index}")
        if index not in keep:
            continue
        if (frame.shape[1], frame.shape[0]) != size:
            raise OverlayError(f"clip {ref}: frame {index} is not {size[0]}x{size[1]}")
        pts = float(cap.get(cv2.CAP_PROP_POS_MSEC))
        if not abs(pts - expected_pts[index]) <= PTS_TOLERANCE_MS:
            raise OverlayError(f"clip {ref}: frame {index} time does not match the landmarks file")
        frames[index] = frame
    return frames


# --------------------------------------------------------------------------- the whole job


def build_overlay(
    df: pd.DataFrame,
    meta: dict[str, Any],
    cap: Any,
    ref: str,
    *,
    start_s: float,
    duration_s: float,
    width: int,
    fps: float,
    blur: bool = False,
) -> tuple[list[Image.Image], list[int], Image.Image]:
    """RGB frames, GIF delays in ms, and a poster still (the frame where the hand is most open)."""
    if not MIN_WIDTH <= width <= MAX_WIDTH:
        raise OverlayError(f"width must be {MIN_WIDTH} to {MAX_WIDTH} px, got {width}")
    if not 0 < duration_s <= MAX_DURATION_S:
        raise OverlayError(f"duration must be above 0 and at most {MAX_DURATION_S:g} s")
    if not fps > 0:
        raise OverlayError("fps must be positive")
    window = select_window(df["pts_ms"].to_numpy(dtype=float), start_s, duration_s)
    if window.size < 2:
        raise OverlayError(f"clip {ref}: fewer than 2 frames in the window")
    stride = stride_for(float(meta["fps"]), fps)
    kept = window[::stride]
    if len(kept) > MAX_FRAMES:
        raise OverlayError(f"{len(kept)} frames is above the limit of {MAX_FRAMES}")

    pts = df["pts_ms"].to_numpy(dtype=float)
    delays = gif_delays_ms(pts[kept])
    size = (int(meta["width"]), int(meta["height"]))
    frames_bgr = read_frames(cap, list(kept), {int(i): float(pts[i]) for i in kept}, size, ref)

    points = np.vstack([landmark_xy(df, int(i)) for i in kept])
    hand = float(np.nanmedian(np.hypot(*(points[0::N_LANDMARKS] - points[9::N_LANDMARKS]).T)))
    pad = CROP_PAD_HAND_SIZES * (hand if np.isfinite(hand) else 0.0)
    box = crop_square(points, size[0], size[1], pad)

    values = normalised_distance(df).values[kept]
    ymax = trace_ymax(values)
    trace_size = (width, int(round(width * TRACE_HEIGHT_RATIO)))
    font = ImageFont.load_default(size=12)

    composed: list[Image.Image] = []
    for n, i in enumerate(kept):
        x0, y0, side = box
        rgb = cv2.cvtColor(frames_bgr[int(i)][y0 : y0 + side, x0 : x0 + side], cv2.COLOR_BGR2RGB)
        panel = Image.fromarray(rgb).resize((width, width), Image.Resampling.LANCZOS)
        xy_panel = to_panel(landmark_xy(df, int(i)), box, width)
        trace = draw_trace(trace_size, values, n, ymax, font)
        composed.append(compose(panel, xy_panel, trace, blur=blur))
    poster_at = int(np.nanargmax(values)) if np.isfinite(values).any() else 0
    return composed, delays, composed[poster_at]


def save_gif(frames: Sequence[Image.Image], delays: Sequence[int], path: Path) -> int:
    """Write an endlessly looping GIF with a shared palette; return its size in bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    paletted = to_palette(frames)
    paletted[0].save(
        path,
        save_all=True,
        append_images=paletted[1:],
        duration=list(delays),
        loop=0,
        optimize=False,
        disposal=1,
    )
    size = path.stat().st_size
    if size > MAX_BYTES:
        path.unlink()
        raise OverlayError(
            f"the GIF would be {size / 1e6:.1f} MB, above the {MAX_BYTES / 1e6:.0f} MB limit"
        )
    return size


# --------------------------------------------------------------------------- command line


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def find_video(ref: str, video_dir: Path) -> Path | None:
    for path in sorted(Path(video_dir).iterdir()) if Path(video_dir).is_dir() else []:
        if path.is_file() and pseudonym(path.stem) == ref:
            return path
    return None


def main(
    argv: Sequence[str] | None = None, *, capture_factory: CaptureFactory | None = None
) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--clip", help="8-character hash of the clip ID")
    source.add_argument("--parquet", type=Path, help="a landmarks file from hand_landmarks.py")
    ap.add_argument("--start-s", type=float, default=0.0, help="window start, seconds of pts_ms")
    ap.add_argument("--duration-s", type=float, default=4.0)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--fps", type=float, default=15.0, help="target rate; every n-th frame is kept")
    ap.add_argument("--blur-outside", action="store_true", help="blur everything but the hand")
    ap.add_argument("--landmarks-dir", type=Path, default=DEFAULT_LANDMARKS_DIR)
    ap.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--interim-root", type=Path, default=INTERIM_ROOT, help="output must stay inside"
    )
    args = ap.parse_args(argv)

    if args.clip is not None:
        parquet = find_parquet(args.clip, args.landmarks_dir)
        if parquet is None:
            print(f"error: no landmarks file for clip {args.clip}", file=sys.stderr)
            return 2
    else:
        parquet = args.parquet
        if not parquet.is_file():
            print(
                f"error: clip {pseudonym(parquet.stem)}: landmarks file not found", file=sys.stderr
            )
            return 2
    ref = pseudonym(parquet.stem)
    if not _inside(args.out_dir, args.interim_root):
        print(f"error: the output folder must be inside {args.interim_root}", file=sys.stderr)
        return 1
    video = find_video(ref, args.video_dir) if capture_factory is None else parquet
    if video is None:
        print(f"error: clip {ref}: video file not found", file=sys.stderr)
        return 2

    try:
        df, meta = read_landmarks(parquet)
        factory = capture_factory or open_video
        cap = factory(str(video))
        try:
            frames, delays, poster = build_overlay(
                df,
                meta,
                cap,
                ref,
                start_s=args.start_s,
                duration_s=args.duration_s,
                width=args.width,
                fps=args.fps,
                blur=args.blur_outside,
            )
        finally:
            cap.release()
        gif_path = args.out_dir / f"{ref}_overlay.gif"
        size = save_gif(frames, delays, gif_path)
        poster_path = args.out_dir / f"{ref}_overlay_poster.png"
        poster.save(poster_path)
    except (OverlayError, HandLandmarksError, pa.ArrowException, OSError, KeyError) as err:
        print(f"FAILED: clip {ref}: {err}", file=sys.stderr)
        return 1

    print(
        f"clip {ref}: {len(frames)} frames, {sum(delays) / 1000:.2f} s, "
        f"{frames[0].width}x{frames[0].height}, {size / 1e6:.2f} MB, loops forever"
    )
    print(f"saved {gif_path.name} and {poster_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
