"""Savitzky-Golay smoothing of the thumb-index signal on its native timestamps (task FT-20).

Input is the FT-19 signal (``signal_processing.py``, ``data/interim/signals/<hash>.parquet``).
Output is ``data/interim/smoothed/<hash>.parquet``. No tap detection, no outlier removal and
no outlier flag: spikes stay as they are. The file is called ``smoothing`` and never ``signal``
because ``src/`` is on ``sys.path``.

The filter is a local polynomial least-squares fit, which on a uniform grid is exactly a
Savitzky-Golay filter (the tests compare it with ``scipy.signal.savgol_filter``), but it works
on the real ``pts_ms`` times, so irregular frame timing is not treated as uniform. Constants,
fixed before any modelling and never tuned on results or labels:

- ``WINDOW_S = 0.167`` s and ``ORDER = 3``;
- the window rule: a sample's window holds every valid sample of its segment with
  ``|t - t_i| <= WINDOW_S / 2 + 1e-9``. That is 5 samples at 30 Hz and 11 at 60 Hz;
- if the window holds fewer than ``ORDER + 2`` samples (segment ends, or a long frame gap) the
  nearest ``ORDER + 2`` samples of the same segment are used instead, and it is counted in
  ``n_widened``. At 30 Hz that is how the first and last two samples of a segment get the same
  fit as scipy's ``mode="interp"``. At 60 Hz an end sample already has at least 6 samples, so
  its window is only cut short (nothing is widened) and the ends differ from scipy's; the
  interior is identical.

Rules:

- A segment is a run of finite samples. The signal is split at EVERY NaN run, each segment is
  smoothed alone, and NaN stays NaN: nothing is smoothed across a gap and nothing becomes 0.
- A segment with fewer than ``ORDER + 2`` samples keeps its raw values and is marked
  ``smoothed = False`` (counted in ``n_unsmoothed``). Rows that are NaN are ``smoothed = False``
  too: there is nothing to smooth.
- FT-19's ``filled`` flag is kept unchanged. Filled samples are inputs to the filter like any
  other sample; ``value_raw`` keeps the FT-19 value next to the smoothed ``value``.
- Nothing here looks at UPDRS or at the CONTROL/ID group.

Output columns: ``frame``, ``t_s``, ``value_raw``, ``value``, ``filled``, ``smoothed``. The
metadata holds ``window_s``, ``order``, ``window_rule``, ``tool``, ``source_sha256`` (of the
FT-19 file), ``n_widened``, ``n_unsmoothed`` and the FT-19 parameters. Writes are atomic, a batch
run skips a clip whose output has the same source and parameters, and every message carries
the hashed clip reference only.

The summary CSV (hashed refs only) is written by ``--all`` only: a ``--clip`` run prints its
numbers and leaves the batch summary untouched.

Exit codes: 0 every selected clip done, 1 some clips failed, 2 an input is missing or the output
folder is outside ``data/interim``.

Usage (from the repo root):
    python models/tapping/src/smoothing.py --clip <8-character hash>
    python models/tapping/src/smoothing.py --all [--force] [--plot]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from plot_distance import save_figure
from signal_processing import (
    DEFAULT_OUT_DIR as SIGNALS_DIR,
)
from signal_processing import (
    Signal,
    SignalError,
    file_sha256,
    find_gaps,
    is_inside,
    json_safe,
    read_signal,
    scrub,
)

TOOL_VERSION = "smoothing-0.1.0"
META_KEY = b"tapping.smoothed"
TMP_MARKER = ".smooth-tmp-"

WINDOW_S = 0.167
ORDER = 3
EPS_S = 1e-9
WINDOW_RULE = "samples of the same segment with |dt| <= window_s / 2 + 1e-9 s"

REPO_ROOT = Path(__file__).resolve().parents[3]
INTERIM_ROOT = REPO_ROOT / "data" / "interim"
DEFAULT_OUT_DIR = INTERIM_ROOT / "smoothed"
DEFAULT_PLOT_DIR = INTERIM_ROOT / "plots"
SUMMARY_NAME = "smoothed_summary.csv"

STATUS_OK = "ok"
STATUS_EMPTY = "no_valid_samples"
TABLE_COLUMNS = ("frame", "t_s", "value_raw", "value", "filled", "smoothed")
HASH_RE = re.compile(r"[0-9a-f]{8}")


class SmoothingError(ValueError):
    """A clip cannot be smoothed. The message names the clip by its short hash."""


class Smoothed:
    """A smoothed clip: the table and the numbers that describe how it was made."""

    def __init__(self, ref: str, table: pd.DataFrame, meta: dict[str, Any]) -> None:
        self.ref = ref
        self.table = table
        self.meta = meta


# --------------------------------------------------------------------------- the filter


def _local_value(
    t: np.ndarray, v: np.ndarray, i: int, window_s: float, order: int
) -> tuple[float, bool]:
    """Local polynomial fit at sample ``i`` of one segment; ``(value, widened)``."""
    half = window_s / 2.0 + EPS_S
    lo = int(np.searchsorted(t, t[i] - half, side="left"))
    hi = int(np.searchsorted(t, t[i] + half, side="right"))
    need = order + 2
    widened = hi - lo < need
    if widened:
        a, b = max(0, lo - need), min(len(t), hi + need)
        nearest = np.argsort(np.abs(t[a:b] - t[i]), kind="stable")[:need]
        idx = np.sort(nearest) + a
    else:
        idx = np.arange(lo, hi)
    x = (t[idx] - t[i]) / (window_s / 2.0)  # scaled to about [-1, 1]: well conditioned
    design = np.vander(x, order + 1, increasing=True)
    coef, *_ = np.linalg.lstsq(design, v[idx], rcond=None)
    return float(coef[0]), widened


def smooth_segment(
    t: np.ndarray, v: np.ndarray, window_s: float = WINDOW_S, order: int = ORDER
) -> tuple[np.ndarray, int]:
    """Smooth one segment (finite values, strictly increasing ``t``); returns it and n_widened.

    The segment must hold at least ``order + 2`` samples.
    """
    if len(t) < order + 2:
        raise ValueError(f"a segment needs at least {order + 2} samples, got {len(t)}")
    out = np.empty(len(v))
    widened = 0
    for i in range(len(v)):
        out[i], was_widened = _local_value(t, v, i, window_s, order)
        widened += was_widened
    return out, widened


def smooth_series(
    t: np.ndarray, values: np.ndarray, window_s: float = WINDOW_S, order: int = ORDER
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Smooth ``values`` on times ``t``, segment by segment. Pure: the inputs are not changed.

    Returns the smoothed array (NaN stays NaN; a too-short segment keeps its raw values), a
    boolean ``smoothed`` mask, and counts ``n_widened``, ``n_unsmoothed``, ``n_segments`` and
    ``n_short_segments``.
    """
    if window_s <= 0 or order < 0:
        raise ValueError("window_s must be positive and order 0 or more")
    t = np.asarray(t, dtype=float)
    values = np.asarray(values, dtype=float)
    if t.shape != values.shape:
        raise ValueError("t and values must have the same length")
    out = values.copy()
    smoothed = np.zeros(len(values), dtype=bool)
    counts = {"n_widened": 0, "n_unsmoothed": 0, "n_segments": 0, "n_short_segments": 0}
    finite = np.isfinite(values)
    gaps = find_gaps(values)
    edges = [0]
    for start, length in gaps:
        edges += [start, start + length]
    edges.append(len(values))
    for a, b in zip(edges[0::2], edges[1::2], strict=True):
        if b <= a or not finite[a:b].all():
            continue
        counts["n_segments"] += 1
        if b - a < order + 2:
            counts["n_short_segments"] += 1
            counts["n_unsmoothed"] += b - a
            continue
        out[a:b], widened = smooth_segment(t[a:b], values[a:b], window_s, order)
        smoothed[a:b] = True
        counts["n_widened"] += widened
    return out, smoothed, counts


# --------------------------------------------------------------------------- from a signal


def smooth_signal(sig: Signal, *, window_s: float = WINDOW_S, order: int = ORDER) -> Smoothed:
    """An FT-19 ``Signal`` to a ``Smoothed`` clip. The input table is not changed."""
    table = sig.table
    t = table["t_s"].to_numpy(dtype=float)
    raw = table["value"].to_numpy(dtype=float)
    if len(t) > 1 and not (np.diff(t) > 0).all():
        raise SmoothingError(f"clip {sig.ref}: the signal time axis is not strictly increasing")
    value, smoothed, counts = smooth_series(t, raw, window_s, order)
    out = pd.DataFrame(
        {
            "frame": table["frame"].to_numpy(dtype=np.int64),
            "t_s": t,
            "value_raw": raw,
            "value": value,
            "filled": table["filled"].to_numpy(dtype=bool),
            "smoothed": smoothed,
        },
        columns=list(TABLE_COLUMNS),
    )
    status = STATUS_OK if smoothed.any() else STATUS_EMPTY
    carried = (
        "scale",
        "scale_mode",
        "scale_window_s",
        "fill_max_gap_frames",
        "fill_method",
        "uniform_hz",
        "trimmed_start_s",
        "trimmed_end_s",
    )
    meta: dict[str, Any] = {
        "tool": TOOL_VERSION,
        "clip_ref": sig.ref,
        "status": status,
        "window_s": window_s,
        "order": order,
        "window_rule": WINDOW_RULE,
        "frames": len(out),
        "n_smoothed": int(smoothed.sum()),
        "n_nan": int((~np.isfinite(raw)).sum()),
        **counts,
        **{f"signal_{k}": sig.meta.get(k) for k in carried},
    }
    return Smoothed(sig.ref, out, meta)


# --------------------------------------------------------------------------- files


def write_smoothed(sm: Smoothed, path: Path, source_sha256: str | None = None) -> None:
    """Write atomically (temporary file, then rename); metadata goes in the parquet file."""
    meta = {k: json_safe(v) for k, v in {**sm.meta, "source_sha256": source_sha256}.items()}
    table = pa.Table.from_pandas(sm.table, preserve_index=False)
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


def read_smoothed(path: Path) -> Smoothed:
    """The table and metadata written by ``write_smoothed``."""
    raw = (pq.read_schema(path).metadata or {}).get(META_KEY)
    if raw is None:
        raise SmoothingError(f"clip {path.stem}: the file has no smoothing metadata")
    meta: dict[str, Any] = json.loads(raw)
    return Smoothed(str(meta["clip_ref"]), pd.read_parquet(path), meta)


def remove_temporary_files(out_dir: Path) -> int:
    removed = 0
    for path in out_dir.glob(f".*{TMP_MARKER}*"):
        with contextlib.suppress(OSError):
            path.unlink()
            removed += 1
    return removed


def up_to_date(path: Path, source_sha: str, params: dict[str, Any]) -> bool:
    """True if ``path`` was made by this tool from this signal file with these parameters."""
    if not path.is_file():
        return False
    try:
        meta = read_smoothed(path).meta
    except (SmoothingError, OSError, pa.ArrowException, ValueError, KeyError):
        return False
    return meta.get("source_sha256") == source_sha and all(
        meta.get(k) == json_safe(v) for k, v in params.items()
    )


ROW_KEYS = ("frames", "n_smoothed", "n_nan", "n_widened", "n_unsmoothed", "n_short_segments")


def process_file(
    source: Path, out_dir: Path, *, window_s: float, order: int, force: bool
) -> dict[str, Any]:
    """Smooth one FT-19 file; return its summary row. Never raises for a bad clip."""
    ref = source.stem
    out = out_dir / f"{ref}.parquet"
    params = {"tool": TOOL_VERSION, "window_s": window_s, "order": order}
    row: dict[str, Any] = {"ref": ref, "action": "failed", "status": "failed", "reason": ""}
    try:
        sha = file_sha256(source)
        if not force and up_to_date(out, sha, params):
            meta = read_smoothed(out).meta
            row.update(action="skipped", status=meta["status"], **{k: meta[k] for k in ROW_KEYS})
            return row
        sm = smooth_signal(read_signal(source), window_s=window_s, order=order)
        write_smoothed(sm, out, sha)
        row.update(action="computed", status=sm.meta["status"], **{k: sm.meta[k] for k in ROW_KEYS})
    except (SignalError, SmoothingError, OSError, pa.ArrowException, ValueError, KeyError) as err:
        row["reason"] = scrub(str(err), source)
    return row


# --------------------------------------------------------------------------- figure


def plot_smoothed(sm: Smoothed) -> Figure:
    """Raw FT-19 signal against the smoothed one; filled samples marked."""
    tab = sm.table
    fig = Figure(figsize=(12, 6.5), dpi=110, layout="constrained")
    FigureCanvasAgg(fig)
    top, bottom = fig.subplots(2, 1, sharex=True)
    top.plot(tab["t_s"], tab["value_raw"], lw=0.8, marker=".", ms=2.5, color="tab:gray")
    top.set_ylabel("before (FT-19), hand sizes")
    top.set_title(
        f"Smoothing, clip {sm.ref}: local polynomial order {sm.meta['order']}, "
        f"window {sm.meta['window_s']} s"
    )
    bottom.plot(tab["t_s"], tab["value_raw"], lw=0.6, color="tab:gray", label="before")
    bottom.plot(tab["t_s"], tab["value"], lw=1.0, color="tab:blue", label="after")
    filled = tab[tab["filled"]]
    bottom.scatter(
        filled["t_s"],
        filled["value"],
        s=28,
        color="tab:red",
        zorder=3,
        label=f"filled ({len(filled)} samples)",
    )
    for ax in (top, bottom):
        ax.grid(alpha=0.25)
        ax.set_ylim(bottom=0)
    bottom.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8)
    bottom.set_ylabel("after, hand sizes")
    bottom.set_xlabel("time from the container timestamps pts_ms (s)")
    return fig


# --------------------------------------------------------------------------- command line


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--clip", help="8-character hash of the clip ID")
    source.add_argument("--all", action="store_true", help="every file in --signals-dir")
    ap.add_argument("--signals-dir", type=Path, default=SIGNALS_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
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
        f"{len(empty)} without a smoothable segment, {len(failed)} failed"
    ]
    lines += [f"  failed: {r['ref']}: {r['reason']}" for r in failed]
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not is_inside(args.out_dir, INTERIM_ROOT):
        print(f"error: --out-dir must be inside {INTERIM_ROOT}", file=sys.stderr)
        return 2
    if args.clip is not None:
        if not HASH_RE.fullmatch(args.clip):
            print(f"error: clip {args.clip!r} is not an 8-character hash", file=sys.stderr)
            return 2
        path = args.signals_dir / f"{args.clip}.parquet"
        if not path.is_file():
            print(
                f"error: no signal file for clip {args.clip} in {args.signals_dir}", file=sys.stderr
            )
            return 2
        files = [path]
    else:
        files = sorted(Path(args.signals_dir).glob("*.parquet"))
        if not files:
            print(f"error: no signal files in {args.signals_dir}", file=sys.stderr)
            return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    remove_temporary_files(args.out_dir)

    started = time.time()
    rows: list[dict[str, Any]] = []
    for path in files:
        row = process_file(path, args.out_dir, window_s=WINDOW_S, order=ORDER, force=args.force)
        rows.append(row)
        if args.plot and row["status"] == STATUS_OK:
            out_png = args.plot_dir / f"{row['ref']}_smoothed.png"
            save_figure(
                plot_smoothed(read_smoothed(args.out_dir / f"{row['ref']}.parquet")), out_png
            )
            print(f"saved {out_png.name}")
    if args.all:  # a --clip run must not replace the batch summary with its one row
        pd.DataFrame(rows).to_csv(args.out_dir / SUMMARY_NAME, index=False)
    elif rows[0]["status"] != "failed":
        print(f"clip {rows[0]['ref']}: {json.dumps({k: rows[0][k] for k in ROW_KEYS})}")
    for line in summary_lines(rows, time.time() - started):
        print(line)
    return 1 if any(r["status"] == "failed" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
