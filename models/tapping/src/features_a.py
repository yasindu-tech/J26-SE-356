"""Set A features: rate and static (task FT-27), one row per retained clip.

    python models/tapping/src/features_a.py [--taps-dir data/interim/taps] [--out-dir ...]

Reads the frozen taps (never a glob: the QC retained list decides which clips), builds the
per-tap table (tap_table.py) and computes Set A from it. Outputs under data/interim/features/:
tap_table.parquet, features_a.csv, features_a_meta.json. Anything not computable is NaN and its
reason is in ``nan_reasons``; NaN is never zero. Columns ending ``_sens`` are the sensitivity set
(one-sided taps kept, intervals under 0.2 s dropped). Primary velocity = peak slope.
Labels are never read. Exit codes: 0 ok, 1 taps disagree on parameters, tool version or the
summary, 2 missing input or bad path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tap_table as tt
from signal_processing import file_sha256, is_inside
from taps import QcError, load_qc

TOOL_VERSION = "features-a-0.1.0"
DEFINITIONS_VERSION = "features-a-1"
REPO_ROOT = Path(__file__).resolve().parents[3]
INTERIM_ROOT = REPO_ROOT / "data" / "interim"
DOCS_ROOT = REPO_ROOT / "models" / "tapping" / "docs"
HASH_LEN = 8
MIN_RATE_TAPS = 3  # tap_rate_hz needs n >= 3
MIN_ITI = 2  # iti_mean_s needs 2 valid intervals
MIN_CV_ITI = 3  # iti_cv needs 3 valid intervals
MIN_AMP_TAPS = 3  # means and medians over valid taps need 3
MIN_MAX_TAPS = 1  # amp_max needs 1
SENSITIVITY = "_sens"  # suffix of the sensitivity set (not "_s": that reads as seconds)
OUT_TABLE, OUT_FEATURES, OUT_META = "tap_table.parquet", "features_a.csv", "features_a_meta.json"

CARRIED = ("duration_s", "fps", "qc_status")
VELOCITY = ("open_vel_peak", "close_vel_peak", "open_vel_mean", "close_vel_mean")
FEATURES = (
    "tap_rate_hz", "iti_mean_s", "iti_cv",
    *(f"{v}{kind}" for v in VELOCITY for kind in ("", "_median")),
    "amp_mean", "amp_max",
)  # fmt: skip
COLUMNS = (
    "ref", "n_taps", *CARRIED, "n_over_duration_hz", *FEATURES, "n_one_sided_excluded",
    "n_iti_excluded", "n_short_iti", *(f + SENSITIVITY for f in FEATURES), "nan_reasons",
)  # fmt: skip


class Disagreement(ValueError):
    """Taps files disagree with each other or with the summary (exit code 1)."""


class InputProblem(ValueError):
    """An input is missing or malformed (exit code 2)."""


def _stat(values: np.ndarray, how: str, minimum: int, name: str, reasons: list[str]) -> float:
    """mean / median / max / cv / mean-iti of the finite values, or NaN with a reason."""
    v = values[np.isfinite(values)]
    if v.size < minimum:
        reasons.append(f"{name}:{v.size}<{minimum}")
        return float("nan")
    if how == "mean":
        return float(v.mean())
    if how == "median":
        return float(np.median(v))
    if how == "max":
        return float(v.max())
    mean = float(v.mean())
    return float(v.std(ddof=1) / mean) if mean > 0 else float("nan")


def _set(table: pd.DataFrame, sensitivity: bool, reasons: list[str]) -> dict[str, float]:
    """The features of one clip; ``sensitivity`` keeps one-sided taps and drops short ITIs."""
    sfx = SENSITIVITY if sensitivity else ""
    taps = table if sensitivity else table[table["valid_amp"].to_numpy(dtype=bool)]
    iti_ok = table["valid_iti"].to_numpy(dtype=bool)
    if sensitivity:
        iti_ok = iti_ok & ~table["short_iti"].to_numpy(dtype=bool)
    iti = table.loc[iti_ok, "iti_before_s"].to_numpy(dtype=float)
    out: dict[str, float] = {}
    mark: list[str] = []

    def put(name: str, value: float) -> None:
        out[name + sfx] = value

    iti_mean = _stat(iti, "mean", MIN_ITI, "iti_mean_s" + sfx, mark)
    put("iti_mean_s", iti_mean)
    put("iti_cv", _stat(iti, "cv", MIN_CV_ITI, "iti_cv" + sfx, mark))
    if sensitivity:  # the span form is not defined once intervals are dropped
        put("tap_rate_hz", 1.0 / iti_mean if np.isfinite(iti_mean) and iti_mean > 0 else np.nan)
    else:
        n = len(table)
        t = table["t_peak_s"].to_numpy(dtype=float)
        if n < MIN_RATE_TAPS:
            mark.append(f"tap_rate_hz:{n}<{MIN_RATE_TAPS}")
            put("tap_rate_hz", float("nan"))
        else:
            put("tap_rate_hz", float((n - 1) / (t[-1] - t[0])))
    for v in VELOCITY:
        col = taps[v].to_numpy(dtype=float)
        put(v, _stat(col, "mean", MIN_AMP_TAPS, v + sfx, mark))
        put(v + "_median", _stat(col, "median", MIN_AMP_TAPS, v + "_median" + sfx, mark))
    amp = taps["amplitude"].to_numpy(dtype=float)
    put("amp_mean", _stat(amp, "mean", MIN_AMP_TAPS, "amp_mean" + sfx, mark))
    put("amp_max", _stat(amp, "max", MIN_MAX_TAPS, "amp_max" + sfx, mark))
    reasons += mark
    return out


def clip_features(table: pd.DataFrame, ref: str, summary: pd.Series) -> dict[str, Any]:
    """One features_a.csv row from a clip's tap table and its taps_summary.csv row."""
    n = len(table)
    reasons: list[str] = []
    duration = float(summary["duration_s"])
    row: dict[str, Any] = {"ref": ref, "n_taps": n}
    for c in CARRIED:
        row[c] = summary[c]
    row["n_over_duration_hz"] = n / duration if duration > 0 else float("nan")
    row.update(_set(table, False, reasons))
    row["n_one_sided_excluded"] = int(table["one_sided"].sum())
    later = np.arange(n) > 0  # the first tap has no interval, that is not an exclusion
    row["n_iti_excluded"] = int((~table["valid_iti"].to_numpy(dtype=bool) & later).sum())
    row["n_short_iti"] = int((table["short_iti"] & table["valid_iti"]).sum())
    row.update(_set(table, True, reasons))
    row["nan_reasons"] = ";".join(reasons)
    return {c: row[c] for c in COLUMNS}


def params_hash(params: dict[str, Any]) -> str:
    """sha256[:8] of the canonical JSON of the TapParams stored in a taps file."""
    text = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:HASH_LEN]


def sha8(path: Path) -> str:
    return file_sha256(path)[:HASH_LEN]


def _atomic(final: Path, write: Any) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f".{final.name}.tmp{os.getpid()}")
    try:
        write(tmp)
        os.replace(tmp, final)
    finally:
        if tmp.exists():
            tmp.unlink()


class Loaded(NamedTuple):
    """The retained clips' tap tables and what identifies the taps that were read."""

    refs: list[str]
    tables: list[pd.DataFrame]
    summary: pd.DataFrame
    params_sha8: str
    taps_tool: str


def load_clips(taps_dir: Path, summary_path: Path, docs_dir: Path) -> Loaded:
    """Read the taps of the QC retained clips (never a glob) and build their tap tables.
    Raises Disagreement (exit 1) or InputProblem (exit 2)."""
    try:
        qc = load_qc(docs_dir)
    except QcError as err:
        raise InputProblem(str(err)) from err
    if not qc.frozen:
        raise InputProblem("qc_retained.csv not found: the clip list is not frozen")
    if not summary_path.is_file():
        raise InputProblem("taps_summary.csv not found")
    summary = pd.read_csv(summary_path, dtype={"ref": str}).set_index("ref")
    refs = sorted(qc.retained)
    absent = [r for r in refs if r not in summary.index]
    if absent:
        raise InputProblem(f"{len(absent)} retained clips are not in taps_summary.csv")
    tables, seen = [], set()
    versions: set[tuple[str, str]] = set()
    missing = 0
    for ref in refs:
        path = taps_dir / f"{ref}.parquet"
        if not path.is_file():
            missing += 1
            continue
        taps, meta = tt.read_taps_file(path)
        versions.add((params_hash(meta["params"]), str(meta["tool"])))
        table = tt.build_tap_table(taps)
        if len(table) != int(summary.at[ref, "n_taps"]):
            raise Disagreement("a taps file and taps_summary.csv disagree on the tap count")
        tables.append(table)
        seen.add(ref)
    if missing:
        raise InputProblem(f"{missing} retained clips have no taps file")
    if len(versions) != 1:
        raise Disagreement(f"taps files disagree on parameters or tool version ({len(versions)})")
    ((phash, tool),) = versions
    stale = sorted(p.stem for p in taps_dir.glob("*.parquet") if p.stem not in seen)
    print(f"ignored {len(stale)} taps files that are not in the QC retained list")
    return Loaded(refs, tables, summary, phash, tool)


def run(taps_dir: Path, out_dir: Path, summary_path: Path, docs_dir: Path) -> dict[str, Any]:
    """Compute and write everything; returns the metadata. Raises Disagreement or InputProblem."""
    refs, tables, summary, phash, tool = load_clips(taps_dir, summary_path, docs_dir)
    rows = [clip_features(t, r, summary.loc[r]) for r, t in zip(refs, tables, strict=True)]
    full = pd.concat(tables, ignore_index=True)
    feats = pd.DataFrame(rows, columns=list(COLUMNS))
    meta_out = {
        "taps_folder": taps_dir.name,
        "tap_params_sha8": phash,
        "taps_tool": tool,
        "features_tool": TOOL_VERSION,
        "tap_table_version": tt.TABLE_VERSION,
        "taps_summary_sha8": sha8(summary_path),
        "qc_list_sha8": sha8(docs_dir / "qc_retained.csv"),
        "definitions_version": DEFINITIONS_VERSION,
        "n_clips": len(feats),
        "n_taps": len(full),
    }
    _atomic(
        out_dir / OUT_TABLE,
        lambda tmp: pq.write_table(pa.Table.from_pandas(full, preserve_index=False), tmp),
    )
    _atomic(out_dir / OUT_FEATURES, lambda tmp: feats.to_csv(tmp, index=False, lineterminator="\n"))
    _atomic(
        out_dir / OUT_META,
        lambda tmp: tmp.write_text(json.dumps(meta_out, sort_keys=True, indent=1) + "\n"),
    )
    return meta_out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for name in ("--taps-dir", "--out-dir", "--summary", "--docs-dir"):
        ap.add_argument(name, type=Path, default=None)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    taps_dir = Path(args.taps_dir or INTERIM_ROOT / "taps")
    out_dir = Path(args.out_dir or INTERIM_ROOT / "features")
    summary = Path(args.summary or taps_dir / "taps_summary.csv")
    docs_dir = Path(args.docs_dir or DOCS_ROOT)
    if not all(is_inside(p, INTERIM_ROOT) for p in (taps_dir, out_dir, summary)):
        print(
            "error: --taps-dir, --out-dir and --summary must be inside data/interim",
            file=sys.stderr,
        )
        return 2
    if not is_inside(docs_dir, DOCS_ROOT):
        print("error: --docs-dir must be inside models/tapping/docs", file=sys.stderr)
        return 2
    try:
        meta = run(taps_dir, out_dir, summary, docs_dir)
    except Disagreement as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    except (InputProblem, tt.TableError, OSError) as err:
        print(
            f"error: {type(err).__name__}: {err if not isinstance(err, OSError) else ''}",
            file=sys.stderr,
        )
        return 2
    print(f"wrote {OUT_FEATURES}: {meta['n_clips']} clips, {meta['n_taps']} taps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
