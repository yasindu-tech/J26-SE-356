"""Clean time axis, validity and gap policy for the thumb-index signal (task FT-19).

Turns one clip's landmarks table (``hand_landmarks.py``, FT-11) into the series the next stages
work on: the thumb-index distance in hand-size units, on the container clock, with every
sample marked as measured, filled or missing. It does NOT smooth or filter (FT-20) and does not
find taps. The file is called ``signal_processing`` and never ``signal`` because ``src/`` is on
``sys.path`` and ``signal.py`` would shadow the standard library.

What it owns:

- the series: ``hand_scale.normalised_distance`` (default ``w9``, centred 3 s rolling median,
  the FT-13 decision) over ``plot_distance.thumb_index_distance_px``; nothing is re-implemented;
- the time axis: ``pts_ms`` in seconds. It must be strictly increasing and complete, otherwise
  the clip is refused (an error by hashed reference), because a guessed clock would silently
  move every later timing;
- validity: a sample is valid when its value is finite. NaN stays NaN, it never becomes 0;
- gaps (explicit policy, ``FillPolicy``): a run of at most ``max_gap_frames`` consecutive
  missing frames BETWEEN two valid samples is filled by linear interpolation in ``pts_ms``
  time, and each filled sample is flagged ``filled``. Longer runs stay NaN. The limit is the
  one ``tracking.py`` uses for its ``gaps_not_fillable`` flag (``Gating.fillable_gap_frames``,
  5). It is a count of FRAMES, not a duration: about 167 ms at 30 Hz but only about 83 ms at
  60 Hz. Neither 60 Hz clip (``6b83804b``, ``78bd0691``) has a gap, so this is untested on
  real data;
- the ends: leading and trailing missing samples (the hand entering or leaving) are trimmed
  from the table, and the trimmed durations are recorded. They are never extrapolated;
- an optional uniform grid (``resample_uniform``): native timestamps stay the default, see the
  decision below.

Native timestamps or a uniform grid? The stored signal keeps the native ``pts_ms`` samples.
Reason: 92 of 234 clips have irregular frame times (21 have a gap up to 101 ms against a
33.3 ms frame), a grid would put values at times no frame was captured, and tap detection and
timing only need the true time of each sample. A uniform grid is still needed later by anything
that assumes equal spacing (FFT, fixed-length filters); ``resample_uniform`` provides it,
opt-in, at the clip's own rate (``round(1 / median frame interval)``: 30 Hz for 232 clips, 60
Hz for 2) and only by interpolating between two valid native neighbours that are close in time.
Irregular timestamps are never treated as uniform.

Labels: nothing here looks at UPDRS or at the CONTROL/ID group. Everything is per clip.

Output ``data/interim/signals/<hash>.parquet`` (gitignored; the file name is the clip's short
hash). Columns: ``frame`` (row in the landmarks table), ``t_s`` (``pts_ms`` / 1000), ``value``
(hand sizes; NaN where missing), ``measured`` (the value before filling, NaN where it was
filled or missing), ``filled`` (True for interpolated samples), ``scale_px`` (the divisor).
The processing parameters and per-clip numbers are stored in the file metadata. Writes are
atomic, a batch run skips a clip whose output was made from the same landmarks file and the
same parameters, and every message carries the hashed reference only.

The summary CSV (hashed refs only) is written by ``--all`` only: a ``--clip`` run prints its
numbers and leaves the batch summary untouched.

Exit codes: 0 every selected clip done, 1 some clips failed, 2 an input is missing or the
output folder is outside ``data/interim``.

Usage (from the repo root):
    python models/tapping/src/signal_processing.py --clip <8-character hash>
    python models/tapping/src/signal_processing.py --all
    python models/tapping/src/signal_processing.py --clip <hash> --resample-hz 30 --plot
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from hand_landmarks import HandLandmarksError
from hand_scale import (
    DEFAULT_MODE,
    DEFAULT_SCALE,
    DEFAULT_WINDOW_S,
    MODES,
    SCALES,
    estimate_scale,
    normalised_distance,
)
from hubu_fis import pseudonym
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from plot_distance import (
    DEFAULT_LANDMARKS_DIR,
    find_parquet,
    load_landmarks,
    save_figure,
    time_axis_s,
)
from tracking import Gating

TOOL_VERSION = "signal-processing-0.1.0"
META_KEY = b"tapping.signal"
TMP_MARKER = ".signal-tmp-"

REPO_ROOT = Path(__file__).resolve().parents[3]
INTERIM_ROOT = REPO_ROOT / "data" / "interim"
DEFAULT_OUT_DIR = INTERIM_ROOT / "signals"
DEFAULT_PLOT_DIR = INTERIM_ROOT / "plots"
SUMMARY_NAME = "signals_summary.csv"

DEFAULT_MAX_GAP_FRAMES = Gating().fillable_gap_frames  # the same limit as tracking.py
STATUS_OK = "ok"
STATUS_EMPTY = "no_valid_samples"
UNIFORM_SPAN_FACTOR = 1.0  # a grid point needs neighbours at most (limit + 1) frame intervals apart

TABLE_COLUMNS = ("frame", "t_s", "value", "measured", "filled", "scale_px")


class SignalError(ValueError):
    """A clip cannot be turned into a signal. The message names the clip by its short hash."""


@dataclass(frozen=True)
class FillPolicy:
    """The gap policy. Stored in the output metadata, so a file says how it was made."""

    max_gap_frames: int = DEFAULT_MAX_GAP_FRAMES
    method: str = "linear_in_pts_time"

    def __post_init__(self) -> None:
        if self.max_gap_frames < 0:
            raise ValueError(f"max_gap_frames must be 0 or more, got {self.max_gap_frames!r}")
        if self.method != "linear_in_pts_time":
            raise ValueError(f"method must be 'linear_in_pts_time', got {self.method!r}")


@dataclass(frozen=True)
class SeriesResult:
    """The outcome of ``process_series``: arrays of the kept (untrimmed) samples and the rest."""

    t: np.ndarray
    value: np.ndarray
    measured: np.ndarray
    filled: np.ndarray
    start: int  # index of the first kept input sample
    stop: int  # one past the last kept input sample
    trimmed_start_s: float
    trimmed_end_s: float


@dataclass(frozen=True)
class Signal:
    """A processed clip: the table and the numbers that describe how it was made."""

    ref: str
    table: pd.DataFrame
    meta: dict[str, Any]


# --------------------------------------------------------------------------- pure functions


def find_gaps(values: np.ndarray) -> list[tuple[int, int]]:
    """Runs of non-finite samples as ``(start index, length)``, in order."""
    missing = ~np.isfinite(np.asarray(values, dtype=float))
    if not missing.any():
        return []
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [(int(a), int(b - a)) for a, b in zip(starts, stops, strict=True)]


def valid_span(values: np.ndarray) -> tuple[int, int] | None:
    """``(first, last + 1)`` index of the finite samples, or None if there are none."""
    finite = np.flatnonzero(np.isfinite(np.asarray(values, dtype=float)))
    if finite.size == 0:
        return None
    return int(finite[0]), int(finite[-1]) + 1


def check_time(t: np.ndarray, ref: str) -> None:
    """The clock must be finite and strictly increasing, or the clip is refused."""
    if not np.isfinite(t).all():
        raise SignalError(
            f"clip {ref}: pts_ms is missing for {int((~np.isfinite(t)).sum())} frames"
        )
    if t.size > 1 and not (np.diff(t) > 0).all():
        raise SignalError(f"clip {ref}: pts_ms is not strictly increasing")


def fill_gaps(
    t: np.ndarray, values: np.ndarray, max_gap_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fill interior runs of at most ``max_gap_frames`` missing samples by linear interpolation.

    Interpolation is in ``t`` (time), not in the sample index, so irregular timestamps are
    respected. Only runs with a valid sample on both sides are filled; a run at the start or
    the end stays NaN. A run of ``max_gap_frames`` is filled, ``max_gap_frames + 1`` is not.
    Returns a NEW array and a boolean mask of the filled samples; the inputs are not changed.
    """
    t = np.asarray(t, dtype=float)
    out = np.array(values, dtype=float, copy=True)
    filled = np.zeros(out.shape, dtype=bool)
    n = len(out)
    for start, length in find_gaps(out):
        stop = start + length
        if length > max_gap_frames or start == 0 or stop == n:
            continue
        t0, t1 = t[start - 1], t[stop]
        v0, v1 = out[start - 1], out[stop]
        out[start:stop] = v0 + (v1 - v0) * (t[start:stop] - t0) / (t1 - t0)
        filled[start:stop] = True
    return out, filled


def process_series(t: np.ndarray, values: np.ndarray, policy: FillPolicy) -> SeriesResult:
    """Trim the missing ends, then fill short interior gaps. Pure: the inputs are not changed.

    With no valid sample at all the result is empty (``start == stop == 0``) and both trimmed
    durations are NaN, because there is no span to measure them against.
    """
    t = np.asarray(t, dtype=float)
    values = np.asarray(values, dtype=float)
    if t.shape != values.shape:
        raise ValueError("t and values must have the same length")
    span = valid_span(values)
    if span is None:
        empty = np.empty(0)
        return SeriesResult(empty, empty, empty, np.empty(0, dtype=bool), 0, 0, np.nan, np.nan)
    start, stop = span
    kept_t, kept = t[start:stop], values[start:stop]
    filled_values, filled = fill_gaps(kept_t, kept, policy.max_gap_frames)
    measured = np.where(filled, np.nan, kept)
    return SeriesResult(
        t=kept_t.copy(),
        value=filled_values,
        measured=measured,
        filled=filled,
        start=start,
        stop=stop,
        trimmed_start_s=float(t[start] - t[0]),
        trimmed_end_s=float(t[-1] - t[stop - 1]),
    )


def median_rate_hz(t: np.ndarray) -> float:
    """``1 / median frame interval`` in Hz; NaN with fewer than two samples."""
    if len(t) < 2:
        return float("nan")
    return float(1.0 / np.median(np.diff(t)))


def resample_uniform(sig: Signal, rate_hz: float | None = None) -> Signal:
    """The signal on an equally spaced grid, from its first to its last sample time.

    A grid point is valid only if the native sample before it and the one after it are both
    valid and at most ``max_gap_frames + 1`` frame intervals apart; it is then interpolated
    linearly and flagged ``filled`` unless it falls exactly on a measured sample. Otherwise it
    is NaN: nothing is invented across a gap that was too long to fill. ``rate_hz`` defaults to
    the clip's own rate, ``round(1 / median frame interval)``.
    """
    table = sig.table
    t = table["t_s"].to_numpy(dtype=float)
    v = table["value"].to_numpy(dtype=float)
    f = table["filled"].to_numpy(dtype=bool)
    if len(t) < 2:
        raise SignalError(f"clip {sig.ref}: too few samples to resample")
    rate = float(round(median_rate_hz(t))) if rate_hz is None else float(rate_hz)
    if not (np.isfinite(rate) and rate > 0):
        raise ValueError(f"rate_hz must be positive, got {rate_hz!r}")
    n_grid = int(np.floor((t[-1] - t[0]) * rate + 1e-9)) + 1
    grid = t[0] + np.arange(n_grid) / rate
    left = np.clip(np.searchsorted(t, grid + 1e-9, side="right") - 1, 0, len(t) - 1)
    right = np.minimum(left + 1, len(t) - 1)
    span = t[right] - t[left]
    interval = float(np.median(np.diff(t)))
    limit = (int(sig.meta["fill_max_gap_frames"]) + 1) * interval * UNIFORM_SPAN_FACTOR
    exact = np.isclose(grid, t[left], atol=1e-9)
    both_valid = np.isfinite(v[left]) & np.isfinite(v[right])
    ok = np.where(exact, np.isfinite(v[left]), both_valid & (span <= limit + 1e-9))
    weight = np.where(span > 0, (grid - t[left]) / np.where(span > 0, span, 1.0), 0.0)
    out = np.where(exact, v[left], v[left] + (v[right] - v[left]) * weight)
    out = np.where(ok, out, np.nan)
    grid_filled = ok & np.where(exact, f[left], True)
    measured = np.where(grid_filled | ~ok, np.nan, out)
    scale = table["scale_px"].to_numpy(dtype=float)
    scale_grid = np.where(ok, scale[left] + (scale[right] - scale[left]) * weight, np.nan)
    new = pd.DataFrame(
        {
            "frame": np.full(n_grid, -1, dtype=np.int64),  # no landmarks row: grid point
            "t_s": grid,
            "value": out,
            "measured": measured,
            "filled": grid_filled,
            "scale_px": scale_grid,
        }
    )
    meta = dict(sig.meta)
    meta.update(_numbers(new, meta))
    meta["uniform_hz"] = rate
    meta["frames_out"] = int(len(new))
    return Signal(sig.ref, new, meta)


# --------------------------------------------------------------------------- from a clip


def json_safe(x: Any) -> Any:
    """Make a number JSON-safe: NaN and inf become None."""
    if isinstance(x, float | np.floating):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def _numbers(table: pd.DataFrame, base: dict[str, Any]) -> dict[str, Any]:
    """Per-clip numbers of a table: fraction filled, remaining gaps, duration."""
    value = table["value"].to_numpy(dtype=float)
    t = table["t_s"].to_numpy(dtype=float)
    gaps = find_gaps(value)
    n = len(table)
    interval = float(np.median(np.diff(t))) if n > 1 else float("nan")
    longest = max((length for _, length in gaps), default=0)
    return {
        "frames_out": n,
        "duration_s": float(t[-1] - t[0]) if n > 1 else 0.0,
        "filled_frames": int(table["filled"].sum()),
        "fraction_filled": float(table["filled"].mean()) if n else float("nan"),
        "remaining_gap_runs": len(gaps),
        "remaining_gap_frames": int(sum(length for _, length in gaps)),
        "longest_remaining_gap_frames": int(longest),
        "longest_remaining_gap_s": float(longest * interval) if n > 1 else 0.0,
        "median_rate_hz": median_rate_hz(t),
    }


def build_signal(
    df: pd.DataFrame,
    ref: str,
    *,
    policy: FillPolicy | None = None,
    scale: str = DEFAULT_SCALE,
    mode: str = DEFAULT_MODE,
    window_s: float = DEFAULT_WINDOW_S,
) -> Signal:
    """One clip's landmarks table to a ``Signal``. The table is not changed.

    Raises ``SignalError`` (message: hashed reference only) if the clock cannot be trusted. A
    clip with no valid sample does not raise: it gives an empty table with status
    ``no_valid_samples``.
    """
    policy = policy or FillPolicy()
    t, _ = time_axis_s(df)
    check_time(t, ref)
    try:
        result = normalised_distance(df, scale, mode, window_s=window_s)
        divisor, _ = estimate_scale(df, scale, mode, window_s=window_s)
    except (HandLandmarksError, KeyError, ValueError) as err:
        raise SignalError(f"clip {ref}: cannot build the signal ({err})") from err
    series = process_series(t, result.values, policy)
    keep = slice(series.start, series.stop)
    table = pd.DataFrame(
        {
            "frame": df["frame"].to_numpy()[keep].astype(np.int64),
            "t_s": series.t,
            "value": series.value,
            "measured": series.measured,
            "filled": series.filled,
            "scale_px": divisor[keep],
        },
        columns=list(TABLE_COLUMNS),
    )
    status = STATUS_OK if len(table) else STATUS_EMPTY
    meta: dict[str, Any] = {
        "tool": TOOL_VERSION,
        "clip_ref": ref,
        "status": status,
        "unit": "thumb-index distance in units of the scale length (hand sizes)",
        "scale": scale,
        "scale_mode": mode,
        "scale_window_s": result.window_s,
        "scale_median_px": result.scale_median_px,
        "fill_max_gap_frames": policy.max_gap_frames,
        "fill_method": policy.method,
        "fill_rule": "interior runs of at most max_gap_frames, between two valid samples",
        "time_axis": "pts_ms / 1000",
        "uniform_hz": None,
        "frames_in": int(len(df)),
        "trimmed_start_s": series.trimmed_start_s,
        "trimmed_end_s": series.trimmed_end_s,
        "trimmed_start_frames": series.start,
        "trimmed_end_frames": int(len(df) - series.stop) if len(table) else int(len(df)),
        "scale_rejected_frames": result.scale_rejected_frames,
    }
    meta.update(_numbers(table, meta))
    return Signal(ref, table, meta)


# --------------------------------------------------------------------------- files


def write_signal(sig: Signal, path: Path, source_sha256: str | None = None) -> None:
    """Write atomically (temporary file, then rename); the metadata goes in the parquet file."""
    meta = {**sig.meta, "source_sha256": source_sha256}
    meta = {k: json_safe(v) for k, v in meta.items()}
    table = pa.Table.from_pandas(sig.table, preserve_index=False)
    schema_meta = dict(table.schema.metadata or {})
    schema_meta[META_KEY] = json.dumps(meta, sort_keys=True).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}{TMP_MARKER}{os.getpid()}")
    try:
        pq.write_table(table.replace_schema_metadata(schema_meta), tmp)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def read_signal(path: Path) -> Signal:
    """The table and metadata written by ``write_signal``."""
    raw = (pq.read_schema(path).metadata or {}).get(META_KEY)
    if raw is None:
        raise SignalError(f"clip {pseudonym(path.stem)}: the file has no signal metadata")
    meta: dict[str, Any] = json.loads(raw)
    return Signal(str(meta["clip_ref"]), pd.read_parquet(path), meta)


def remove_temporary_files(out_dir: Path) -> int:
    removed = 0
    for path in out_dir.glob(f".*{TMP_MARKER}*"):
        with contextlib.suppress(OSError):
            path.unlink()
            removed += 1
    return removed


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scrub(text: str, path: Path) -> str:
    """``text`` with the file's path, name and stem (the raw clip ID) replaced by the hash."""
    ref = pseudonym(path.stem)
    for secret in (str(path), path.name, path.stem):
        text = text.replace(secret, ref)
    return text


def up_to_date(path: Path, source_sha: str, params: dict[str, Any]) -> bool:
    """True if ``path`` was made by this tool from this landmarks file with these parameters."""
    if not path.is_file():
        return False
    try:
        meta = read_signal(path).meta
    except (SignalError, OSError, pa.ArrowException, ValueError, KeyError):
        return False
    return meta.get("source_sha256") == source_sha and all(
        meta.get(k) == json_safe(v) for k, v in params.items()
    )


def process_file(
    landmarks: Path,
    out_dir: Path,
    *,
    policy: FillPolicy,
    scale: str,
    mode: str,
    window_s: float,
    resample_hz: float | None,
    force: bool,
) -> dict[str, Any]:
    """Process one landmarks file; return its summary row. Never raises for a bad clip."""
    ref = pseudonym(landmarks.stem)
    out = out_dir / f"{ref}.parquet"
    params = {
        "tool": TOOL_VERSION,
        "scale": scale,
        "scale_mode": mode,
        "scale_window_s": window_s if mode == "rolling_median" else None,
        "fill_max_gap_frames": policy.max_gap_frames,
        "fill_method": policy.method,
        "uniform_hz": resample_hz,
    }
    row: dict[str, Any] = {"ref": ref, "action": "failed", "status": "failed", "reason": ""}
    try:
        sha = file_sha256(landmarks)
        if not force and up_to_date(out, sha, params):
            sig = read_signal(out)
            row.update(action="skipped", status=sig.meta["status"])
            row.update(_row_numbers(sig.meta))
            return row
        df = load_landmarks(landmarks, ref)
        sig = build_signal(df, ref, policy=policy, scale=scale, mode=mode, window_s=window_s)
        if resample_hz is not None and len(sig.table) >= 2:
            sig = resample_uniform(sig, resample_hz)
        write_signal(sig, out, sha)
        row.update(action="computed", status=sig.meta["status"])
        row.update(_row_numbers(sig.meta))
    except (SignalError, HandLandmarksError, OSError, pa.ArrowException, ValueError) as err:
        row["reason"] = scrub(str(err), landmarks)
    return row


ROW_KEYS = (
    "frames_in",
    "frames_out",
    "duration_s",
    "trimmed_start_s",
    "trimmed_end_s",
    "filled_frames",
    "fraction_filled",
    "remaining_gap_runs",
    "remaining_gap_frames",
    "longest_remaining_gap_frames",
    "scale_median_px",
)


def _row_numbers(meta: dict[str, Any]) -> dict[str, Any]:
    return {k: (np.nan if meta.get(k) is None else meta[k]) for k in ROW_KEYS}


# --------------------------------------------------------------------------- figure


def plot_signal(df: pd.DataFrame, sig: Signal) -> Figure:
    """Raw (unprocessed) normalised signal against the processed one, filled samples marked."""
    raw = normalised_distance(
        df,
        sig.meta["scale"],
        sig.meta["scale_mode"],
        window_s=sig.meta["scale_window_s"] or DEFAULT_WINDOW_S,
    ).values
    t_all, _ = time_axis_s(df)
    table = sig.table
    fig = Figure(figsize=(12, 6.5), dpi=110, layout="constrained")
    FigureCanvasAgg(fig)
    top, bottom = fig.subplots(2, 1, sharex=True)
    top.plot(t_all, raw, lw=0.8, marker=".", ms=2.5, color="tab:gray", label="before: NaN breaks")
    top.set_ylabel("raw, hand sizes")
    top.set_title(
        f"Signal processing, clip {sig.ref}: fill up to {sig.meta['fill_max_gap_frames']} frames, "
        f"trimmed {sig.meta['trimmed_start_s'] or 0:.2f} s / {sig.meta['trimmed_end_s'] or 0:.2f} s"
    )
    bottom.plot(table["t_s"], table["value"], lw=0.8, color="tab:blue", label="after")
    filled = table[table["filled"]]
    bottom.scatter(
        filled["t_s"],
        filled["value"],
        s=28,
        color="tab:red",
        zorder=3,
        label=f"filled ({len(filled)} samples)",
    )
    for ax in (top, bottom):
        miss = ~np.isfinite(raw)
        ax.vlines(
            t_all[miss], 0, 0.04, transform=ax.get_xaxis_transform(), color="tab:orange", lw=0.8
        )
        ax.grid(alpha=0.25)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", fontsize=8)
    bottom.set_ylabel("processed, hand sizes")
    bottom.set_xlabel("time from the container timestamps pts_ms (s)")
    return fig


# --------------------------------------------------------------------------- command line


def is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--clip", help="8-character hash of the clip ID")
    source.add_argument("--all", action="store_true", help="every clip in --landmarks-dir")
    ap.add_argument("--landmarks-dir", type=Path, default=DEFAULT_LANDMARKS_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--max-gap-frames", type=int, default=DEFAULT_MAX_GAP_FRAMES)
    ap.add_argument("--scale", choices=SCALES, default=DEFAULT_SCALE)
    ap.add_argument("--mode", choices=MODES, default=DEFAULT_MODE)
    ap.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S)
    ap.add_argument(
        "--resample-hz", type=float, default=None, help="write a uniform grid at this rate"
    )
    ap.add_argument("--force", action="store_true", help="recompute even if up to date")
    ap.add_argument("--plot", action="store_true", help="also save a before/after plot")
    ap.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    return ap


def summary_lines(rows: list[dict[str, Any]], elapsed: float) -> list[str]:
    ok = [r for r in rows if r["status"] != "failed"]
    failed = [r for r in rows if r["status"] == "failed"]
    empty = [r for r in ok if r["status"] == STATUS_EMPTY]
    lines = [
        f"done in {elapsed:.1f}s: {len(rows)} clips, {len(ok)} ok "
        f"({sum(r['action'] == 'computed' for r in ok)} computed, "
        f"{sum(r['action'] == 'skipped' for r in ok)} up to date), "
        f"{len(empty)} without a valid sample, {len(failed)} failed"
    ]
    lines += [f"  failed: {r['ref']}: {r['reason']}" for r in failed]
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not is_inside(args.out_dir, INTERIM_ROOT):
        print(f"error: --out-dir must be inside {INTERIM_ROOT}", file=sys.stderr)
        return 2
    try:
        policy = FillPolicy(max_gap_frames=args.max_gap_frames)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if args.clip is not None:
        path = find_parquet(args.clip, args.landmarks_dir)
        if path is None:
            print(
                f"error: no landmarks file for clip {args.clip} in {args.landmarks_dir}",
                file=sys.stderr,
            )
            return 2
        files = [path]
    else:
        files = sorted(Path(args.landmarks_dir).glob("*.parquet"))
        if not files:
            print(f"error: no landmarks files in {args.landmarks_dir}", file=sys.stderr)
            return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    remove_temporary_files(args.out_dir)

    started = time.time()
    rows: list[dict[str, Any]] = []
    for path in files:
        row = process_file(
            path,
            args.out_dir,
            policy=policy,
            scale=args.scale,
            mode=args.mode,
            window_s=args.window_s,
            resample_hz=args.resample_hz,
            force=args.force,
        )
        rows.append(row)
        if args.plot and row["status"] == STATUS_OK:
            df = load_landmarks(path, row["ref"])
            sig = read_signal(args.out_dir / f"{row['ref']}.parquet")
            if args.resample_hz is None:
                out_png = args.plot_dir / f"{row['ref']}_signal.png"
                save_figure(plot_signal(df, sig), out_png)
                print(f"saved {out_png.name}")
    if args.all:  # a --clip run must not replace the batch summary with its one row
        pd.DataFrame(rows).to_csv(args.out_dir / SUMMARY_NAME, index=False)
    if args.clip is not None and rows[0]["status"] != "failed":
        meta = read_signal(args.out_dir / f"{rows[0]['ref']}.parquet").meta
        print(f"clip {rows[0]['ref']}: {json.dumps({k: json_safe(meta[k]) for k in ROW_KEYS})}")
    for line in summary_lines(rows, time.time() - started):
        print(line)
    return 1 if any(r["status"] == "failed" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
