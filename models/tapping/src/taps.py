"""Tap detection and per-tap measurements from the smoothed thumb-index signal (task FT-23).

Input: ``data/interim/smoothed/<hash>.parquet`` (FT-20: ``frame, t_s, value_raw, value, filled,
smoothed``). ``value`` is the thumb-index distance in hand sizes on the container clock ``t_s``;
NaN is a gap and stays a gap. Output: one parquet per clip (one row per tap) and, for ``--all``
only, a per-clip summary, all under ``data/interim/taps/`` (gitignored). Hashed refs only.

FROZEN PARAMETERS (``TapParams``, fixed from the FT-23 study, never tuned from here on):

- detection runs on ``value`` with ``scipy.signal.find_peaks``, separately on each finite
  segment (the signal is split at every NaN run and nothing is interpolated across a gap);
  no ``wlen``;
- prominence = 0.20 x range, range = p95 - p5 of the clip's finite ``value`` (not the median);
- minimum peak spacing = 0.15 s, converted to samples with each segment's own median frame
  interval, ``max(1, round(0.15 / dt))``;
- a segment shorter than 3 samples, or a clip with fewer than 10 finite samples, gives no taps
  (the summary then shows 0 taps, and NaN range for a clip with too few samples; the parquet
  metadata says ``status``).

TAP DEFINITION. A peak is the maximum aperture. Troughs are the argmin of ``value`` between
neighbouring peaks of the same segment; before the first and after the last peak, the argmin
from the segment start and to the segment end.

- ``amplitude`` = ``value_peak`` - mean(trough_left, trough_right). If a trough sits on the first
  or last sample of its segment (the signal began or ended mid-movement or at a NaN gap), only
  the other trough is used and the tap is flagged ``one_sided``; if both are edges, the tap is
  ``one_sided`` and the lower of the two is used;
- ``amplitude_raw`` = max(``value_raw`` within +-1 sample of the peak) minus the same trough rule
  applied to the min of ``value_raw`` within +-1 sample of each trough; ``raw_disagrees`` when
  amplitude_raw / amplitude is outside 0.90..1.10;
- ``width_half_s`` = width of the peak at half height between the trough level used for the
  amplitude and the peak, by linear interpolation on ``t_s``, bounded by the segment;
  ``narrow`` when it is below 0.10 s. A flag never removes a tap;
- times come from ``t_s`` (seconds from the clip start), never from frame index / fps.
  ``interval_s`` is to the previous peak (NaN for the first), ``rise_s`` from trough_left to the
  peak, ``fall_s`` from the peak to trough_right;
- velocity: ``open_vel`` = max and ``close_vel`` = -min of ``np.gradient(value, t_s)`` (per
  segment) between trough_left and the peak, and between the peak and trough_right, in hand sizes
  per second; ``open_vel_mean`` = amplitude / rise_s and ``close_vel_mean`` = amplitude / fall_s.
  The maximum slope depends on the frame rate (the two 60 fps clips read about 6-7% higher than
  they would at 30 fps) and on the smoothing, so it is comparable inside the 30 fps group only.

FLAGS (``;``-joined in this order, empty if none): ``one_sided``; ``gap_adjacent`` (a peak within
0.2 s of a NaN sample); ``on_filled`` (a ``filled`` sample within +-1 of the peak); ``narrow``;
``raw_disagrees``; ``gap_in_interval`` (the interval to the previous peak contains a NaN run);
``long_interval`` (interval_s > 1.8 x the clip's median interval; needs >= 4 taps);
``possible_hidden_tap_before`` (= gap_in_interval AND long_interval).

Partial open phases cut by a clip edge or a NaN gap are NOT tap rows. ``n_open_edge_ends`` counts
the segment ends (segments of more than 3 samples) whose value is >= p5 + 0.5 x range and which
are at their maximum or moving away from the open phase: at a segment start ``v[0] >= v[1]``, at a
segment end ``v[-1] >= v[-2]``.

OUTPUTS. ``<hash>.parquet`` with ``ref, tap_index, frame, t_peak_s, value_peak, trough_left_s,
trough_right_s, value_trough_left, value_trough_right, amplitude, amplitude_raw, width_half_s,
interval_s, rise_s, fall_s, open_vel, close_vel, open_vel_mean, close_vel_mean, flags`` (a clip
with no taps still writes an empty file with these columns) and in its metadata the parameters,
tool version, sha256 of the source smoothed file, the QC-list hash (or ``unfrozen``), the status
and the clip's summary row. ``taps_summary.csv`` is written ONLY by ``--all`` (a ``--clip`` run
never touches it). Writes are atomic; a clip is skipped when the source hash, the parameters and
the QC list are unchanged, unless ``--force``.

QC LISTS (FT-22). ``models/tapping/docs/qc_retained.csv`` (``ref, verdict, note``): ``ok`` and
``ok_flag`` rows are processed (``qc_status`` retained / retained_flag, the note of an ok_flag row
in ``qc_flag``); clips in ``qc_rejections.csv`` are skipped. If ``qc_retained.csv`` does not exist
every clip is processed with ``qc_status = unfrozen``. If it exists and a smoothed clip is in
neither list, that is an error (exit 1, hashes only); the other clips are still processed.

Nothing here looks at UPDRS or at the CONTROL/ID group. Exit codes: 0 done, 1 a clip failed or the
QC lists are inconsistent with the clips, 2 bad arguments, missing inputs, or an output folder
outside ``data/interim``.

Usage (from the repo root):
    python models/tapping/src/taps.py --all [--force] [--plot]
    python models/tapping/src/taps.py --clip <8-character hash> [--plot]
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from plot_distance import save_figure
from scipy.signal import find_peaks
from signal_processing import file_sha256, is_inside, json_safe, scrub
from smoothing import SmoothingError, read_smoothed

TOOL_VERSION = "taps-0.1.0"
META_KEY = b"tapping.taps"
TMP_MARKER = ".taps-tmp-"

REPO_ROOT = Path(__file__).resolve().parents[3]
INTERIM_ROOT = REPO_ROOT / "data" / "interim"
DOCS_ROOT = REPO_ROOT / "models" / "tapping" / "docs"

RETAINED_NAME = "qc_retained.csv"
REJECTIONS_NAME = "qc_rejections.csv"
SUMMARY_NAME = "taps_summary.csv"
HASH_RE = re.compile(r"[0-9a-f]{8}")

STATUS_OK = "ok"
STATUS_NO_TAPS = "no_taps"
STATUS_FEW_SAMPLES = "too_few_samples"


@dataclass(frozen=True)
class TapParams:
    """The frozen detection and flag parameters (FT-23 study). Do not tune."""

    prominence_frac: float = 0.20  # x range (p95 - p5 of the clip's finite value)
    percentiles: tuple[float, float] = (5.0, 95.0)
    min_spacing_s: float = 0.15
    min_segment_samples: int = 3
    min_finite_samples: int = 10
    gap_adjacent_s: float = 0.2
    narrow_s: float = 0.10
    raw_ratio_low: float = 0.90
    raw_ratio_high: float = 1.10
    long_interval_factor: float = 1.8
    long_interval_min_taps: int = 4
    open_edge_level: float = 0.5  # x range above p5
    low_range: float = 0.6  # summary hint only

    def as_metadata(self) -> dict[str, Any]:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


PARAMS = TapParams()

TAP_COLUMNS = (
    "ref",
    "tap_index",
    "frame",
    "t_peak_s",
    "value_peak",
    "trough_left_s",
    "trough_right_s",
    "value_trough_left",
    "value_trough_right",
    "amplitude",
    "amplitude_raw",
    "width_half_s",
    "interval_s",
    "rise_s",
    "fall_s",
    "open_vel",
    "close_vel",
    "open_vel_mean",
    "close_vel_mean",
    "flags",
)
INT_COLUMNS = ("tap_index", "frame")
SUMMARY_COLUMNS = (
    "ref",
    "fps",
    "duration_s",
    "n_taps",
    "tap_rate_hz",
    "median_interval_s",
    "interval_cv",
    "range",
    "floor_p5",
    "prominence_threshold",
    "n_one_sided",
    "n_gap_adjacent",
    "n_narrow",
    "n_possible_hidden",
    "n_open_edge_ends",
    "n_nan_runs",
    "longest_nan_s",
    "qc_status",
    "qc_flag",
    "low_range",
)
FLAG_ORDER = (
    "one_sided",
    "gap_adjacent",
    "on_filled",
    "narrow",
    "raw_disagrees",
    "gap_in_interval",
    "long_interval",
    "possible_hidden_tap_before",
)


class TapError(ValueError):
    """A clip cannot be processed. The message names the clip by its short hash."""


@dataclass
class TapResult:
    """One clip: the tap table, its summary numbers and a status."""

    table: pd.DataFrame
    stats: dict[str, Any]
    status: str
    peaks: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))


# --------------------------------------------------------------------------- segments


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Runs of True in a boolean mask as ``(start, stop)`` index pairs."""
    if not mask.any():
        return []
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [(int(a), int(b)) for a, b in zip(starts, stops, strict=True)]


def finite_segments(values: np.ndarray) -> list[tuple[int, int]]:
    """Runs of finite samples as ``(start, stop)`` index pairs."""
    return _runs(np.isfinite(values))


def nan_runs(values: np.ndarray) -> list[tuple[int, int]]:
    """Runs of non-finite samples as ``(start, stop)`` index pairs."""
    return _runs(~np.isfinite(values))


def clip_range(values: np.ndarray, params: TapParams = PARAMS) -> tuple[float, float, float] | None:
    """``(p5, p95, range)`` of the finite values, or None with too few finite samples."""
    fin = values[np.isfinite(values)]
    if fin.size < params.min_finite_samples:
        return None
    lo, hi = np.percentile(fin, params.percentiles)
    return float(lo), float(hi), float(hi - lo)


def detect_peaks(t: np.ndarray, v: np.ndarray, params: TapParams = PARAMS) -> np.ndarray:
    """Peak indices (global), found per finite segment. Empty if there is no usable range."""
    stats = clip_range(v, params)
    if stats is None or stats[2] <= 0:
        return np.empty(0, dtype=int)
    prominence = params.prominence_frac * stats[2]
    found: list[int] = []
    for a, b in finite_segments(v):
        if b - a < params.min_segment_samples:
            continue
        dt = float(np.median(np.diff(t[a:b])))
        spacing = max(1, round(params.min_spacing_s / dt))
        idx, _ = find_peaks(v[a:b], prominence=prominence, distance=spacing)
        found += [a + int(i) for i in idx]
    return np.array(sorted(found), dtype=int)


# --------------------------------------------------------------------------- measurements


def _half_width(t: np.ndarray, v: np.ndarray, a: int, b: int, p: int, base: float) -> float:
    """Width at half height between ``base`` and the peak, within the segment ``[a, b)``."""
    h = base + 0.5 * (v[p] - base)
    i = p
    while i > a and v[i - 1] >= h:
        i -= 1
    left = t[a] if i == a else t[i - 1] + (h - v[i - 1]) * (t[i] - t[i - 1]) / (v[i] - v[i - 1])
    j = p
    while j < b - 1 and v[j + 1] >= h:
        j += 1
    right = t[b - 1] if j == b - 1 else t[j] + (v[j] - h) * (t[j + 1] - t[j]) / (v[j] - v[j + 1])
    return float(right - left)


def _base(trough_l: float, trough_r: float, edge_l: bool, edge_r: bool) -> tuple[float, bool]:
    """The trough level for an amplitude and whether the tap is one-sided."""
    if edge_l and edge_r:
        return min(trough_l, trough_r), True
    if edge_l:
        return trough_r, True
    if edge_r:
        return trough_l, True
    return 0.5 * (trough_l + trough_r), False


def _window(x: np.ndarray, centre: int) -> np.ndarray:
    """The finite values within +-1 sample of ``centre``."""
    w = x[max(centre - 1, 0) : centre + 2]
    return w[np.isfinite(w)]


def _wmin(x: np.ndarray, centre: int) -> float:
    w = _window(x, centre)
    return float(w.min()) if w.size else float("nan")


def _wmax(x: np.ndarray, centre: int) -> float:
    w = _window(x, centre)
    return float(w.max()) if w.size else float("nan")


def empty_table() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="float64") for c in TAP_COLUMNS})
    df["ref"] = pd.Series(dtype="object")
    df["flags"] = pd.Series(dtype="object")
    for c in INT_COLUMNS:
        df[c] = pd.Series(dtype="int64")
    return df


def open_edge_ends(v: np.ndarray, params: TapParams = PARAMS) -> int:
    """Segment ends that are still high and at or moving away from their maximum (see module)."""
    stats = clip_range(v, params)
    if stats is None:
        return 0
    level = stats[0] + params.open_edge_level * stats[2]
    count = 0
    for a, b in finite_segments(v):
        if b - a <= 3:
            continue
        count += bool(v[a] >= level and v[a] >= v[a + 1])
        count += bool(v[b - 1] >= level and v[b - 1] >= v[b - 2])
    return count


def extract_taps(
    ref: str,
    t_abs: np.ndarray,
    value: np.ndarray,
    raw: np.ndarray,
    filled: np.ndarray,
    frames: np.ndarray,
    params: TapParams = PARAMS,
) -> TapResult:
    """Detect the taps of one clip and measure them. Pure: the inputs are not changed."""
    t_abs = np.asarray(t_abs, dtype=float)
    v = np.asarray(value, dtype=float)
    raw = np.asarray(raw, dtype=float)
    if len(t_abs) > 1 and not (np.diff(t_abs) > 0).all():
        raise TapError(f"clip {ref}: the time axis is not strictly increasing")
    t = t_abs - t_abs[0] if len(t_abs) else t_abs
    fin = np.isfinite(v)
    runs = nan_runs(v)
    stats = clip_range(v, params)
    rows: list[dict[str, Any]] = []
    status = STATUS_FEW_SAMPLES if stats is None else STATUS_OK
    peaks = detect_peaks(t, v, params)
    nan_t = t[~fin]
    nan_before = np.concatenate(([0], np.cumsum(~fin)))  # NaN samples before each index
    prev_peak: int | None = None
    for a, b in finite_segments(v):
        pk = [int(p) for p in peaks if a <= p < b]
        if not pk:
            continue
        tro = [a + int(np.argmin(v[a : pk[0] + 1]))]
        tro += [x + int(np.argmin(v[x : y + 1])) for x, y in zip(pk[:-1], pk[1:], strict=True)]
        tro.append(pk[-1] + int(np.argmin(v[pk[-1] : b])))
        grad = np.gradient(v[a:b], t[a:b])
        for m, p in enumerate(pk):
            tl, tr = tro[m], tro[m + 1]
            base, one_sided = _base(v[tl], v[tr], tl == a, tr == b - 1)
            amplitude = float(v[p] - base)
            base_raw, _ = _base(_wmin(raw, tl), _wmin(raw, tr), tl == a, tr == b - 1)
            amplitude_raw = _wmax(raw, p) - base_raw
            ratio = amplitude_raw / amplitude if amplitude > 0 else float("nan")
            width = _half_width(t, v, a, b, p, base)
            rise, fall = float(t[p] - t[tl]), float(t[tr] - t[p])
            interval = float(t[p] - t[prev_peak]) if prev_peak is not None else float("nan")
            flags = {
                "one_sided": one_sided,
                "gap_adjacent": bool(nan_t.size)
                and bool(np.abs(nan_t - t[p]).min() <= params.gap_adjacent_s),
                "on_filled": bool(np.asarray(filled)[max(p - 1, 0) : p + 2].any()),
                "narrow": width < params.narrow_s,
                "raw_disagrees": bool(
                    np.isfinite(ratio)
                    and not params.raw_ratio_low <= ratio <= params.raw_ratio_high
                ),
                "gap_in_interval": prev_peak is not None
                and nan_before[p + 1] - nan_before[prev_peak] > 0,
            }
            rows.append(
                {
                    "ref": ref,
                    "tap_index": len(rows),
                    "frame": int(np.asarray(frames)[p]),
                    "t_peak_s": float(t[p]),
                    "value_peak": float(v[p]),
                    "trough_left_s": float(t[tl]),
                    "trough_right_s": float(t[tr]),
                    "value_trough_left": float(v[tl]),
                    "value_trough_right": float(v[tr]),
                    "amplitude": amplitude,
                    "amplitude_raw": amplitude_raw,
                    "width_half_s": width,
                    "interval_s": interval,
                    "rise_s": rise,
                    "fall_s": fall,
                    "open_vel": float(np.max(grad[tl - a : p - a + 1])),
                    "close_vel": float(-np.min(grad[p - a : tr - a + 1])),
                    "open_vel_mean": amplitude / rise if rise > 0 else float("nan"),
                    "close_vel_mean": amplitude / fall if fall > 0 else float("nan"),
                    "_flags": flags,
                }
            )
            prev_peak = p
    table = _finish_table(rows, params)
    stats_row = clip_stats(ref, t, v, table, runs, stats, params)
    if status == STATUS_OK and len(table) == 0:
        status = STATUS_NO_TAPS
    stats_row["n_open_edge_ends"] = open_edge_ends(v, params)
    return TapResult(table, stats_row, status, peaks)


def _finish_table(rows: list[dict[str, Any]], params: TapParams) -> pd.DataFrame:
    """Add the interval flags, join the flag names and fix the column order and types."""
    if not rows:
        return empty_table()
    intervals = np.array([r["interval_s"] for r in rows[1:]], dtype=float)
    median = float(np.median(intervals)) if intervals.size else float("nan")
    use_long = len(rows) >= params.long_interval_min_taps and np.isfinite(median)
    for r in rows:
        flags = r.pop("_flags")
        long_interval = bool(
            use_long
            and np.isfinite(r["interval_s"])
            and r["interval_s"] > params.long_interval_factor * median
        )
        flags["long_interval"] = long_interval
        flags["possible_hidden_tap_before"] = long_interval and bool(flags["gap_in_interval"])
        r["flags"] = ";".join(name for name in FLAG_ORDER if flags[name])
    df = pd.DataFrame(rows)[list(TAP_COLUMNS)]
    for c in INT_COLUMNS:
        df[c] = df[c].astype("int64")
    return df


def clip_stats(
    ref: str,
    t: np.ndarray,
    v: np.ndarray,
    table: pd.DataFrame,
    runs: list[tuple[int, int]],
    stats: tuple[float, float, float] | None,
    params: TapParams,
) -> dict[str, Any]:
    """The per-clip summary numbers (without the QC columns)."""
    nan = float("nan")
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    iv = table["interval_s"].to_numpy(dtype=float)[1:] if len(table) > 1 else np.empty(0)
    flags = table["flags"].str.split(";") if len(table) else pd.Series([], dtype=object)

    def count(name: str) -> int:
        return int(sum(name in f for f in flags))

    longest = max((float(t[min(b, len(t) - 1)] - t[max(a - 1, 0)]) for a, b in runs), default=0.0)
    rng = stats[2] if stats else nan
    return {
        "ref": ref,
        "fps": float(1.0 / np.median(np.diff(t))) if len(t) > 1 else nan,
        "duration_s": duration,
        "n_taps": len(table),
        "tap_rate_hz": len(table) / duration if duration > 0 else nan,
        "median_interval_s": float(np.median(iv)) if iv.size else nan,
        "interval_cv": float(np.std(iv) / np.mean(iv)) if iv.size > 1 else nan,
        "range": rng,
        "floor_p5": stats[0] if stats else nan,
        "prominence_threshold": params.prominence_frac * rng if stats else nan,
        "n_one_sided": count("one_sided"),
        "n_gap_adjacent": count("gap_adjacent"),
        "n_narrow": count("narrow"),
        "n_possible_hidden": count("possible_hidden_tap_before"),
        "n_open_edge_ends": 0,
        "n_nan_runs": len(runs),
        "longest_nan_s": longest,
        "low_range": bool(stats and rng < params.low_range),
    }


# --------------------------------------------------------------------------- QC lists


@dataclass
class QcLists:
    """The frozen FT-22 lists, or ``frozen = False`` when they do not exist yet."""

    frozen: bool
    retained: dict[str, tuple[str, str]] = field(default_factory=dict)  # ref -> (verdict, note)
    rejected: set[str] = field(default_factory=set)
    sha: str = "unfrozen"


class QcError(ValueError):
    """The QC lists cannot be read (exit code 2)."""


def load_qc(docs_dir: Path) -> QcLists:
    retained_path = Path(docs_dir) / RETAINED_NAME
    if not retained_path.is_file():
        return QcLists(frozen=False)
    rejected_path = Path(docs_dir) / REJECTIONS_NAME
    if not rejected_path.is_file():
        raise QcError(f"{REJECTIONS_NAME} is missing next to {RETAINED_NAME}")
    try:
        kept = pd.read_csv(retained_path, dtype=str, keep_default_na=False)
        rej = pd.read_csv(rejected_path, dtype=str, keep_default_na=False)
    except (ValueError, OSError) as err:
        raise QcError(f"cannot read the QC lists: {str(err).splitlines()[0][:80]}") from err
    if not {"ref", "verdict", "note"} <= set(kept.columns) or "ref" not in rej.columns:
        raise QcError("the QC lists lack the columns ref, verdict, note")
    sha = hashlib.sha256(
        (file_sha256(retained_path) + file_sha256(rejected_path)).encode()
    ).hexdigest()
    retained = {
        r: (v, n)
        for r, v, n in zip(kept["ref"], kept["verdict"], kept["note"], strict=True)
        if v in ("ok", "ok_flag")
    }
    return QcLists(True, retained, set(rej["ref"]), sha)


def qc_columns(qc: QcLists, ref: str) -> tuple[str, str]:
    """``(qc_status, qc_flag)`` of a clip."""
    if not qc.frozen:
        return "unfrozen", ""
    verdict, note = qc.retained[ref]
    return ("retained_flag", note) if verdict == "ok_flag" else ("retained", "")


# --------------------------------------------------------------------------- files


def _atomic_write(final: Path, write: Any) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f".{final.name}{TMP_MARKER}{os.getpid()}")
    try:
        write(tmp)
        os.replace(tmp, final)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def write_taps(result: TapResult, path: Path, meta: dict[str, Any]) -> None:
    """Write the tap table with its metadata, atomically."""
    table = pa.Table.from_pandas(result.table, preserve_index=False)
    schema_meta = dict(table.schema.metadata or {})
    schema_meta[META_KEY] = json.dumps(
        {k: json_safe(v) for k, v in meta.items()}, sort_keys=True
    ).encode()
    _atomic_write(path, lambda tmp: pq.write_table(table.replace_schema_metadata(schema_meta), tmp))


def read_taps(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = (pq.read_schema(path).metadata or {}).get(META_KEY)
    if raw is None:
        raise TapError(f"clip {path.stem}: the file has no taps metadata")
    return pd.read_parquet(path), json.loads(raw)


def remove_temporary_files(out_dir: Path) -> None:
    for path in out_dir.glob(f".*{TMP_MARKER}*"):
        with contextlib.suppress(OSError):
            path.unlink()


def summary_row(stats: dict[str, Any], qc_status: str, qc_flag: str) -> dict[str, Any]:
    row = {**stats, "qc_status": qc_status, "qc_flag": qc_flag}
    return {c: row[c] for c in SUMMARY_COLUMNS}


def _clean_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: json_safe(v) for k, v in row.items()}


def process_file(
    source: Path, out_dir: Path, qc: QcLists, *, params: TapParams, force: bool
) -> dict[str, Any]:
    """Process one smoothed file; returns a result row. Never raises for a bad clip."""
    ref = source.stem
    out = out_dir / f"{ref}.parquet"
    qc_status, qc_flag = qc_columns(qc, ref)
    row: dict[str, Any] = {"ref": ref, "action": "failed", "status": "failed", "reason": ""}
    try:
        sha = file_sha256(source)
        key = {
            "tool": TOOL_VERSION,
            "params": params.as_metadata(),
            "source_sha256": sha,
            "qc_list_sha256": qc.sha,
            "qc_status": qc_status,
            "qc_flag": qc_flag,
        }
        if not force and out.is_file():
            try:
                _, old = read_taps(out)
            except (TapError, OSError, pa.ArrowException, ValueError, KeyError):
                old = {}
            if all(old.get(k) == json_safe(v) for k, v in key.items()):
                row.update(action="skipped", status=old["status"], summary=old["summary"])
                return row
        sm = read_smoothed(source)
        tb = sm.table
        result = extract_taps(
            ref,
            tb["t_s"].to_numpy(dtype=float),
            tb["value"].to_numpy(dtype=float),
            tb["value_raw"].to_numpy(dtype=float),
            tb["filled"].to_numpy(dtype=bool),
            tb["frame"].to_numpy(dtype=np.int64),
            params,
        )
        summary = _clean_row(summary_row(result.stats, qc_status, qc_flag))
        write_taps(result, out, {**key, "status": result.status, "summary": summary})
        row.update(action="computed", status=result.status, summary=summary)
    except (TapError, SmoothingError, OSError, pa.ArrowException, ValueError, KeyError) as err:
        row["reason"] = scrub(str(err), source)
    return row


# --------------------------------------------------------------------------- figure


def plot_clip(ref: str, smoothed: pd.DataFrame, taps: pd.DataFrame) -> Figure:
    """The signal with its taps: two rows (first and second half of the clip)."""
    t = smoothed["t_s"].to_numpy(dtype=float)
    t = t - t[0]
    v = smoothed["value"].to_numpy(dtype=float)
    raw = smoothed["value_raw"].to_numpy(dtype=float)
    filled = smoothed["filled"].to_numpy(dtype=bool)
    fig = Figure(figsize=(16, 7.5), dpi=85, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 1)
    half = float(t[-1]) / 2.0
    peak_t = taps["t_peak_s"].to_numpy(dtype=float)
    peak_v = taps["value_peak"].to_numpy(dtype=float)
    trough_t = np.concatenate([taps["trough_left_s"], taps["trough_right_s"]]).astype(float)
    trough_v = np.concatenate([taps["value_trough_left"], taps["value_trough_right"]]).astype(float)
    for k, ax in enumerate(axes):
        ax.plot(t, raw, color="0.78", lw=0.6, label="value_raw")
        ax.plot(t, v, color="tab:blue", lw=1.0, label="value")
        ax.plot(peak_t, peak_v, "v", color="tab:red", ms=6, label="peak (max aperture)")
        ax.plot(trough_t, trough_v, "^", color="tab:green", ms=5, label="trough")
        if filled.any():
            ax.plot(t[filled], v[filled], "o", color="black", ms=4, mfc="none", label="filled")
        for a, b in nan_runs(v):
            ax.axvspan(
                t[max(a - 1, 0)], t[min(b, len(t) - 1)], color="tab:orange", alpha=0.25, lw=0
            )
        ax.set_xlim(k * half - 0.1, (k + 1) * half + 0.1)
        ax.grid(alpha=0.25)
        ax.set_ylabel("distance, hand sizes")
    axes[0].set_title(f"clip {ref}: {len(taps)} taps (prominence 0.20 x range, spacing 0.15 s)")
    axes[1].set_xlabel("time from the clip start, pts (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(color="tab:orange", alpha=0.25))
    labels.append("NaN gap")
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), fontsize=8)
    return fig


# --------------------------------------------------------------------------- command line


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--clip", help="8-character hash of the clip ID")
    source.add_argument("--all", action="store_true", help="every file in --smoothed-dir")
    ap.add_argument("--force", action="store_true", help="recompute even if up to date")
    ap.add_argument("--plot", action="store_true", help="also save an overlay PNG per clip")
    ap.add_argument("--smoothed-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--docs-dir", type=Path, default=None, help="where the QC lists are read from")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    smoothed_dir = Path(args.smoothed_dir or INTERIM_ROOT / "smoothed")
    out_dir = Path(args.out_dir or INTERIM_ROOT / "taps")
    docs_dir = Path(args.docs_dir or DOCS_ROOT)
    if not is_inside(out_dir, INTERIM_ROOT):
        print("error: --out-dir must be inside data/interim", file=sys.stderr)
        return 2
    try:
        return _run(args, smoothed_dir, out_dir, docs_dir)
    except QcError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except OSError as err:
        name = Path(err.filename).name if isinstance(err.filename, str) else "a file"
        print(
            f"error: could not read/write {name}: {err.strerror or type(err).__name__}",
            file=sys.stderr,
        )
        return 2


def _run(args: argparse.Namespace, smoothed_dir: Path, out_dir: Path, docs_dir: Path) -> int:
    qc = load_qc(docs_dir)
    if not qc.frozen:
        print("qc_retained.csv not found: processing ALL clips (QC not frozen)")
    if args.clip is not None:
        if not HASH_RE.fullmatch(args.clip):
            print("error: --clip must be an 8-character hash", file=sys.stderr)
            return 2
        path = smoothed_dir / f"{args.clip}.parquet"
        if not path.is_file():
            print(f"error: no smoothed file for clip {args.clip}", file=sys.stderr)
            return 2
        files = [path]
    else:
        files = sorted(p for p in smoothed_dir.glob("*.parquet") if HASH_RE.fullmatch(p.stem))
        if not files:
            print(f"error: no smoothed files in {smoothed_dir}", file=sys.stderr)
            return 2
    unclassified = sorted(
        p.stem
        for p in files
        if qc.frozen and p.stem not in qc.retained and p.stem not in qc.rejected
    )
    skipped = sorted(p.stem for p in files if qc.frozen and p.stem in qc.rejected)
    todo = [p for p in files if not qc.frozen or p.stem in qc.retained]
    out_dir.mkdir(parents=True, exist_ok=True)
    remove_temporary_files(out_dir)

    started = time.time()
    rows: list[dict[str, Any]] = []
    for path in todo:
        row = process_file(path, out_dir, qc, params=PARAMS, force=args.force)
        rows.append(row)
        if args.plot and row["status"] != "failed":
            tbl, _ = read_taps(out_dir / f"{row['ref']}.parquet")
            png = out_dir / "plots" / f"{row['ref']}_taps.png"
            save_figure(plot_clip(row["ref"], read_smoothed(path).table, tbl), png)
            print(f"saved {png.name}")
    ok = [r for r in rows if r["status"] != "failed"]
    if args.all:  # a --clip run must not replace the batch summary with its one row
        summary = pd.DataFrame([r["summary"] for r in ok], columns=list(SUMMARY_COLUMNS))
        _atomic_write(out_dir / SUMMARY_NAME, lambda tmp: summary.to_csv(tmp, index=False))
    elif ok:
        shown = {k: ok[0]["summary"][k] for k in ("n_taps", "tap_rate_hz", "range", "qc_status")}
        print(f"clip {ok[0]['ref']}: {json.dumps(shown)}")
    elif skipped and not todo:
        print(f"clip {files[0].stem}: rejected in the QC list, skipped")
    failed = [r for r in rows if r["status"] == "failed"]
    print(
        f"done in {time.time() - started:.1f}s: {len(rows)} clips, {len(ok)} ok "
        f"({sum(r['action'] == 'computed' for r in ok)} computed, "
        f"{sum(r['action'] == 'skipped' for r in ok)} up to date), {len(failed)} failed"
        + (f", {len(skipped)} rejected in QC (skipped)" if skipped else "")
    )
    for r in failed:
        print(f"  failed: {r['ref']}: {r['reason']}")
    if unclassified:
        print(f"error: {len(unclassified)} clips are in neither QC list: {', '.join(unclassified)}")
    return 1 if failed or unclassified else 0


if __name__ == "__main__":
    sys.exit(main())
