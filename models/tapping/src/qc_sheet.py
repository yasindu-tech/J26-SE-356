"""QC contact sheets of the smoothed thumb-index signals (task FT-21).

One figure shows 24 clips as a 6 x 4 grid, so that a person can look at the signals themselves
(task FT-22 audits EVERY clip this way, to catch silent tracking failures). The sheets only
show; nothing here drops, ranks or sorts clips by quality.

Modes (exactly one):

- ``--sample N --seed S``: ``N`` clips (at most 24) drawn with ``random.Random(S).sample`` from
  all clip hashes sorted ascending, then shown in hash order. Default seed 21. Same seed, same
  clips. The seed is in the figure title and in the CSV, and the CSV is the record to rely on
  (the generator is stable in practice but not promised across Python versions);
- ``--page K``: all clips sorted by hash, 24 per page (234 clips: 9 pages of 24 and a tenth of
  18). A short page hides the unused panels and keeps the grid positions;
- ``--all-pages``: every page.

Panel: the raw FT-19 value in light grey, the smoothed value in blue, filled samples as red
dots, runs without a value (NaN) as shaded orange spans where the line is broken (NaN is never
drawn as 0). The y axis is fixed at ``Y_MIN`` to ``Y_MAX`` hand sizes so that tiny or huge
signals look tiny or huge (``--yscale auto`` autoscales each panel instead); a clip above the
axis gets a red ``max X.X ^`` marker. The x axis is the clip's own duration. Title: the hash,
duration, fps and detection rate. Second line: longest missing run (frames), the manifest's
review flags (names starting with ``info_`` are skipped), filled frames, the longest remaining
gap (frames) and the seconds trimmed at both ends. Third title line: range (p95 - p5), floor (p5)
and max. The text hints below are printed in red under the x axis. No text except the red
``max`` marker sits inside the plotting area, so no valid sample is ever covered by text.

Hints are display hints only: they are never used to drop or sort clips. A clip with no valid
sample gets a red "no valid samples" panel; a clip whose smoothed file is missing or cannot be
read gets an error panel and the exit code is 1. If the smoothed folder and the manifest list
different clips, the counts are printed.

Inputs: ``data/interim/smoothed/<hash>.parquet`` and ``data/interim/tracking_manifest.csv``,
of which ONLY the columns ``ref``, ``duration_s``, ``fps``, ``detection_rate``,
``longest_missing_run`` and ``flags`` are read: no raw clip ID, no group, no hand, no
participant reference. Nothing here looks at UPDRS or at the CONTROL/ID group.

Outputs (under ``data/interim/qc/``, gitignored, always regenerated, atomic writes):
``sheet_seed021.png`` + ``.csv`` or ``sheet_page01of10.png`` + ``.csv``. The CSV lists, per
panel: sheet, panel, row, col, ref, status, range, floor, max, hints, seed, page. It has no
verdict column: FT-22 keeps its verdicts in a separate ``verdicts.csv``.

For a doubtful panel, look closer with ``smoothing.py --clip <hash> --plot``.

Exit codes: 0 done, 1 some clips are missing or unreadable, or the smoothed folder and the manifest
list different clips on ANY page (the sheet is still written),
2 an input is missing, the output folder is outside ``data/interim``, or the arguments are bad.

Usage (from the repo root):
    python models/tapping/src/qc_sheet.py --sample 24 --seed 21
    python models/tapping/src/qc_sheet.py --page 1
    python models/tapping/src/qc_sheet.py --all-pages [--yscale auto]
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import random
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from signal_processing import find_gaps, is_inside
from smoothing import SmoothingError, read_smoothed

GRID_ROWS = 4
GRID_COLS = 6
PER_SHEET = GRID_ROWS * GRID_COLS
FIGSIZE = (24, 14)
DPI = 80
DEFAULT_SEED = 21
TMP_MARKER = ".qc-tmp-"

Y_MIN = -0.1
Y_MAX = 2.4
# The three thresholds below were chosen after viewing the 234 clips; display hints only,
# never used to drop or sort clips.
LOW_RANGE = 0.6  # hint LOW RANGE: p95 - p5 below this (hand sizes)
FLOOR_HIGH = 0.3  # hint FLOOR HIGH: p5 above this (hand sizes)
ABOVE_AXIS = Y_MAX  # hint ABOVE AXIS: max above this (hand sizes)

MANIFEST_COLUMNS = ("ref", "duration_s", "fps", "detection_rate", "longest_missing_run", "flags")
HASH_RE = re.compile(r"[0-9a-f]{8}")

TEXT_BOX = {
    "facecolor": "white",
    "alpha": 0.75,
    "edgecolor": "none",
    "pad": 1.5,
}  # keeps text legible

STATUS_OK = "ok"
STATUS_EMPTY = "no_valid_samples"
STATUS_MISSING = "missing"
STATUS_UNREADABLE = "unreadable"
CSV_COLUMNS = (
    "sheet",
    "panel",
    "row",
    "col",
    "ref",
    "status",
    "range",
    "floor",
    "max",
    "hints",
    "seed",
    "page",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
INTERIM_ROOT = REPO_ROOT / "data" / "interim"


@dataclass
class Panel:
    """Everything one panel needs. Arrays are None for a missing or unreadable clip."""

    ref: str
    status: str
    t: np.ndarray | None = None
    raw: np.ndarray | None = None
    value: np.ndarray | None = None
    filled: np.ndarray | None = None
    info: dict[str, Any] = field(default_factory=dict)
    qc: dict[str, float] = field(default_factory=dict)
    hints: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- numbers


def qc_numbers(value: np.ndarray) -> dict[str, float]:
    """Range (p95 - p5), floor (p5) and max of the finite values; NaN when there are none."""
    finite = np.asarray(value, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"range": float("nan"), "floor": float("nan"), "max": float("nan")}
    p5, p95 = np.percentile(finite, [5, 95])
    return {"range": float(p95 - p5), "floor": float(p5), "max": float(finite.max())}


def hints_for(qc: dict[str, float]) -> list[str]:
    """Text hints from the QC numbers. A NaN number gives no hint."""
    hints = []
    if qc["range"] < LOW_RANGE:
        hints.append("LOW RANGE")
    if qc["floor"] > FLOOR_HIGH:
        hints.append("FLOOR HIGH")
    if qc["max"] > ABOVE_AXIS:
        hints.append("ABOVE AXIS")
    return hints


# --------------------------------------------------------------------------- selection


def all_refs(smoothed_dir: Path, manifest_refs: Sequence[str]) -> list[str]:
    """Every clip hash that has a smoothed file or a manifest row, sorted ascending."""
    on_disk = {p.stem for p in Path(smoothed_dir).glob("*.parquet") if HASH_RE.fullmatch(p.stem)}
    return sorted(on_disk | set(manifest_refs))


def sample_refs(refs: Sequence[str], n: int, seed: int) -> list[str]:
    """``n`` hashes drawn with ``random.Random(seed).sample`` from the sorted list, in hash order."""
    pool = sorted(refs)
    if not 1 <= n <= min(PER_SHEET, len(pool)):
        raise ValueError(f"--sample must be between 1 and {min(PER_SHEET, len(pool))}, got {n}")
    return sorted(random.Random(seed).sample(pool, n))


def n_pages(n_clips: int) -> int:
    return max(1, math.ceil(n_clips / PER_SHEET))


def page_refs(refs: Sequence[str], page: int) -> list[str]:
    """The hashes of page ``page`` (1-based) of the sorted list, 24 per page."""
    pool = sorted(refs)
    total = n_pages(len(pool))
    if not 1 <= page <= total:
        raise ValueError(f"--page must be between 1 and {total}, got {page}")
    return pool[(page - 1) * PER_SHEET : page * PER_SHEET]


# --------------------------------------------------------------------------- loading


def read_manifest(path: Path) -> pd.DataFrame:
    """The manifest, ONLY the columns the sheet shows (never the raw ID, group or hand)."""
    df = pd.read_csv(path, usecols=list(MANIFEST_COLUMNS), dtype={"ref": str})
    return df.set_index("ref")


def review_flags(flags: object) -> list[str]:
    """The manifest's review flags: names starting with ``info_`` are skipped."""
    if not isinstance(flags, str):
        return []
    return [f for f in flags.split(";") if f and not f.startswith("info_")]


def load_panel(ref: str, smoothed_dir: Path, manifest: pd.DataFrame) -> Panel:
    """One clip's panel data. Never raises for a missing or unreadable file."""
    info: dict[str, Any] = {}
    if ref in manifest.index:
        row = manifest.loc[ref]
        info.update(
            duration_s=row["duration_s"],
            fps=row["fps"],
            detection_rate=row["detection_rate"],
            longest_missing_run=row["longest_missing_run"],
            flags=review_flags(row["flags"]),
        )
    path = Path(smoothed_dir) / f"{ref}.parquet"
    if not path.is_file():
        return Panel(ref, STATUS_MISSING, info=info)
    try:
        sm = read_smoothed(path)
    except (SmoothingError, OSError, pa.ArrowException, ValueError, KeyError):
        return Panel(ref, STATUS_UNREADABLE, info=info)
    table = sm.table
    t = table["t_s"].to_numpy(dtype=float)
    value = table["value"].to_numpy(dtype=float)
    raw = table["value_raw"].to_numpy(dtype=float)
    filled = table["filled"].to_numpy(dtype=bool)
    info["filled_frames"] = int(filled.sum())
    info["longest_gap"] = max((length for _, length in find_gaps(value)), default=0)
    info["trimmed_s"] = float(
        (sm.meta.get("signal_trimmed_start_s") or 0.0)
        + (sm.meta.get("signal_trimmed_end_s") or 0.0)
    )
    if not np.isfinite(value).any():
        return Panel(ref, STATUS_EMPTY, t, raw, value, filled, info)
    qc = qc_numbers(value)
    return Panel(ref, STATUS_OK, t, raw, value, filled, info, qc, hints_for(qc))


# --------------------------------------------------------------------------- drawing


def _nan_runs(value: np.ndarray) -> list[tuple[int, int]]:
    return [(int(a), int(a) + int(length)) for a, length in find_gaps(value)]


def _title(panel: Panel) -> str:
    i = panel.info
    if "duration_s" in i:
        head = f"{panel.ref}  {i['duration_s']:.0f}s  {i['fps']:.0f} fps  det {i['detection_rate']:.2f}"
    else:
        head = f"{panel.ref}  (not in the manifest)"
    parts = []
    if "longest_missing_run" in i:
        parts.append(f"miss {int(i['longest_missing_run'])}")
    if i.get("flags"):
        parts.append(",".join(i["flags"]))
    if "filled_frames" in i:
        parts.append(f"fill {i['filled_frames']}")
        parts.append(f"gap {i['longest_gap']}")
        parts.append(f"trim {i['trimmed_s']:.1f}s")
    lines = [head]
    if parts:
        lines.append("  ".join(parts))
    if panel.qc:  # the footer numbers sit above the plotting area, never on the signal
        lines.append(
            f"range {panel.qc['range']:.2f}  floor {panel.qc['floor']:.2f}  max {panel.qc['max']:.2f}"
        )
    return "\n".join(lines)


def _error_panel(ax: Axes, text: str) -> None:
    ax.text(0.5, 0.5, text, color="tab:red", ha="center", va="center", fontsize=12, weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("tab:red")
        spine.set_linewidth(2)


def draw_panel(ax: Axes, panel: Panel, yscale: str) -> None:
    ax.set_title(_title(panel), fontsize=8.5)
    if panel.status == STATUS_MISSING:
        _error_panel(ax, "missing smoothed file")
        return
    if panel.status == STATUS_UNREADABLE:
        _error_panel(ax, "smoothed file unreadable")
        return
    assert panel.t is not None and panel.raw is not None
    assert panel.value is not None and panel.filled is not None
    if panel.status == STATUS_EMPTY:
        _error_panel(ax, "no valid samples")
        return
    t = panel.t - panel.t[0]
    ax.plot(t, panel.raw, color="0.75", lw=0.6)
    ax.plot(t, panel.value, color="tab:blue", lw=0.9)  # NaN stays NaN: the line breaks
    if panel.filled.any():
        ax.scatter(t[panel.filled], panel.value[panel.filled], s=8, color="tab:red", zorder=3)
    for a, b in _nan_runs(panel.value):
        ax.axvspan(t[max(a - 1, 0)], t[min(b, len(t) - 1)], color="tab:orange", alpha=0.25, lw=0)
    ax.set_xlim(0, float(t[-1]))
    if yscale == "fixed":
        ax.set_ylim(Y_MIN, Y_MAX)
        if panel.qc["max"] > Y_MAX:
            ax.text(
                0.99, 0.97, f"max {panel.qc['max']:.1f} ↑", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="tab:red", weight="bold", bbox=TEXT_BOX,
            )  # fmt: skip
    if panel.hints:  # below the x axis (outside the plotting area), laid out by the figure
        ax.set_xlabel(
            "  ".join(panel.hints), color="tab:red", weight="bold", fontsize=8, labelpad=2
        )
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=7)


def make_figure(panels: Sequence[Panel], title: str, yscale: str = "fixed") -> Figure:
    """The 6 x 4 sheet. Unused panels are hidden and keep their grid positions."""
    fig = Figure(figsize=FIGSIZE, dpi=DPI, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(GRID_ROWS, GRID_COLS).ravel()
    for k, ax in enumerate(axes):
        if k < len(panels):
            draw_panel(ax, panels[k], yscale)
        else:
            ax.set_visible(False)
    fig.suptitle(title, fontsize=13)
    return fig


# --------------------------------------------------------------------------- files


def sheet_name(mode: str, seed: int | None, page: int | None, pages: int) -> str:
    return f"sheet_seed{seed:03d}" if mode == "sample" else f"sheet_page{page:02d}of{pages:02d}"


def csv_rows(
    panels: Sequence[Panel], sheet: str, seed: int | None, page: int | None
) -> pd.DataFrame:
    rows = []
    for k, p in enumerate(panels):
        rows.append(
            {
                "sheet": sheet,
                "panel": k + 1,
                "row": k // GRID_COLS + 1,
                "col": k % GRID_COLS + 1,
                "ref": p.ref,
                "status": p.status,
                "range": p.qc.get("range", float("nan")),
                "floor": p.qc.get("floor", float("nan")),
                "max": p.qc.get("max", float("nan")),
                "hints": ";".join(p.hints),
                "seed": seed if seed is not None else "",
                "page": page if page is not None else "",
            }
        )
    return pd.DataFrame(rows, columns=list(CSV_COLUMNS))


def _atomic(final: Path, write: Any) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f".{final.name}{TMP_MARKER}{os.getpid()}")
    try:
        write(tmp)
        os.replace(tmp, final)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def write_sheet(fig: Figure, table: pd.DataFrame, out_dir: Path, name: str) -> tuple[Path, Path]:
    png, csv = out_dir / f"{name}.png", out_dir / f"{name}.csv"
    _atomic(png, lambda tmp: fig.savefig(tmp, format="png"))
    _atomic(csv, lambda tmp: table.to_csv(tmp, index=False))
    return png, csv


def remove_temporary_files(out_dir: Path) -> None:
    for path in out_dir.glob(f".*{TMP_MARKER}*"):
        with contextlib.suppress(OSError):
            path.unlink()


# --------------------------------------------------------------------------- command line


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sample", type=int, metavar="N", help="N random clips (at most 24)")
    mode.add_argument("--page", type=int, metavar="K", help="page K of all clips, 24 per page")
    mode.add_argument("--all-pages", action="store_true", help="every page")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="for --sample (default 21)")
    ap.add_argument("--yscale", choices=("fixed", "auto"), default="fixed")
    ap.add_argument("--smoothed-dir", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    smoothed_dir = args.smoothed_dir or INTERIM_ROOT / "smoothed"
    manifest_path = args.manifest or INTERIM_ROOT / "tracking_manifest.csv"
    out_dir = args.out_dir or INTERIM_ROOT / "qc"
    if not is_inside(out_dir, INTERIM_ROOT):
        print(f"error: --out-dir must be inside {INTERIM_ROOT}", file=sys.stderr)
        return 2
    if not Path(manifest_path).is_file():
        print(f"error: manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    try:
        manifest = read_manifest(Path(manifest_path))
    except (ValueError, OSError) as err:
        print(f"error: cannot read the manifest columns ({err})", file=sys.stderr)
        return 2
    refs = all_refs(Path(smoothed_dir), list(manifest.index))
    if not refs:
        print(f"error: no clips in {smoothed_dir} or the manifest", file=sys.stderr)
        return 2
    on_disk = {p.stem for p in Path(smoothed_dir).glob("*.parquet") if HASH_RE.fullmatch(p.stem)}
    mismatch = on_disk != set(manifest.index)
    if mismatch:
        print(
            f"note: {len(on_disk)} smoothed files, {len(manifest)} manifest rows; "
            f"{len(on_disk - set(manifest.index))} only in the folder, "
            f"{len(set(manifest.index) - on_disk)} only in the manifest"
        )

    pages = n_pages(len(refs))
    jobs: list[tuple[str, list[str], str, int | None, int | None]] = []
    try:
        if args.sample is not None:
            chosen = sample_refs(refs, args.sample, args.seed)
            name = sheet_name("sample", args.seed, None, pages)
            jobs.append(
                (
                    name,
                    chosen,
                    f"QC sheet, seed {args.seed:03d}: {len(chosen)} of {len(refs)} clips",
                    args.seed,
                    None,
                )
            )
        else:
            for k in range(1, pages + 1) if args.all_pages else [args.page]:
                chosen = page_refs(refs, k)
                title = f"QC sheet, page {k} of {pages}, {len(chosen)} clips"
                jobs.append((sheet_name("page", None, k, pages), chosen, title, None, k))
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    remove_temporary_files(out_dir)
    bad = 0
    for name, chosen, title, seed, page in jobs:
        panels = [load_panel(r, Path(smoothed_dir), manifest) for r in chosen]
        bad += sum(p.status in (STATUS_MISSING, STATUS_UNREADABLE) for p in panels)
        fig = make_figure(panels, title, args.yscale)
        png, _ = write_sheet(fig, csv_rows(panels, name, seed, page), out_dir, name)
        print(f"saved {png.name} ({len(panels)} clips)")
    if bad:
        print(f"{bad} clips are missing or unreadable (error panels on the sheet)")
    if mismatch and not bad:
        print("the smoothed folder and the manifest disagree (see the note above): exit code 1")
    return 1 if bad or mismatch else 0


if __name__ == "__main__":
    sys.exit(main())
