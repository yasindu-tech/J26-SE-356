"""Set B features: the sequence effect (task FT-28), one row per retained clip.

    python models/tapping/src/features_b.py [--taps-dir data/interim/taps] [--out-dir ...]

Slopes, late/early ratio, decrement, hesitations and pauses from the FT-27 tap table. The
definitions are frozen in DEFINITIONS (formula text) and pinned by a test before any rating is
looked at. The pure functions take numpy arrays and return (value, reason): a value that cannot
be computed is NaN with a reason, never 0. Columns ending ``_sens`` are the one sensitivity
variant (one-sided taps kept, taps and intervals under 0.2 s dropped). Labels are never read.
Exit codes: 0 ok, 1 taps disagree on parameters, tool version or the summary, 2 bad input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import features_a as fa
import numpy as np
import pandas as pd
import tap_table as tt
from scipy.stats import theilslopes
from signal_processing import is_inside

TOOL_VERSION = "features-b-0.1.0"
DEFINITIONS_VERSION = "features-b-1"
INTERIM_ROOT = fa.INTERIM_ROOT
DOCS_ROOT = fa.DOCS_ROOT
OUT_FEATURES, OUT_META = "features_b.csv", "features_b_meta.json"
SENS = "_sens"

MIN_TAPS = 8  # slopes, min/initial, decrement: valid taps
MIN_THIRDS = 9  # late/early needs 3 taps in each third
MIN_INTERVALS = 8  # interval features: valid intervals
PER_SECONDS = 10.0  # slopes are fractions of the clip mean per 10 s
INITIAL_TAPS = 3  # the initial level is the median of the first 3 valid taps
ROLL = 3  # rolling median window
DECREMENT_FRACTION = 0.8  # a decrement is a fall below 0.8 x the initial level
DECREMENT_RUN = 3  # ... for at least 3 consecutive taps
DECREMENT_SHARE = 0.8  # ... and at least 80% of the later taps stay below
HESITATION_RATIO = 1.8  # interval / clip median interval above this is a hesitation
HALT_RATIO = 3.0  # above this it is a halt
SHORT_ITI_S = tt.SHORT_ITI_S  # sensitivity set drops taps and intervals shorter than this

DEFINITIONS: dict[str, dict[str, str]] = {
    "amp_slope": {
        "status": "primary",
        "formula": "TheilSen(amplitude vs t_peak_s, scipy defaults) / mean(amplitude) * 10; "
        "valid taps; min 8",
    },
    "open_vel_slope": {
        "status": "primary",
        "formula": "as amp_slope on open_vel_peak, divided by mean(open_vel_peak); min 8",
    },
    "close_vel_slope": {
        "status": "primary",
        "formula": "as amp_slope on close_vel_peak, divided by mean(close_vel_peak); min 8",
    },
    "amp_late_early": {
        "status": "primary",
        "formula": "mean(last k) / mean(first k) of amplitude, k = n_valid // 3; min 9",
    },
    "amp_min_initial": {
        "status": "primary",
        "formula": "min(R) / median(y[:3]); R = centred rolling median 3, min_periods 3; min 8",
    },
    "decrement_present": {
        "status": "primary",
        "formula": "exists i: sm[i:i+3] < 0.8 * median(y[:3]) all, and mean(below[i:]) >= 0.8; "
        "sm = centred rolling median 3, min_periods 1; NaN below 8 valid taps",
    },
    "decrement_onset_frac": {
        "status": "primary",
        "formula": "first such i / n_valid (zero-based); NaN 'no_decrement' when absent",
    },
    "iti_slope": {
        "status": "exploratory",
        "formula": "TheilSen(interval vs t_peak_s of the later tap) / mean(interval) * 10; "
        "valid intervals; min 8; excluded from the primary model if split-half reliability < 0.6",
    },
    "n_hesitations": {
        "status": "primary",
        "formula": "count of valid intervals with 1.8 < r <= 3.0, r = interval / median; min 8",
    },
    "n_halts": {
        "status": "primary",
        "formula": "count of valid intervals with r > 3.0; min 8 valid intervals",
    },
    "longest_pause_ratio": {"status": "primary", "formula": "max(r); min 8 valid intervals"},
    "longest_pause_s": {"status": "primary", "formula": "max valid interval in s; min 8"},
    "sensitivity": {
        "status": "primary",
        "formula": "_sens: one_sided taps kept, taps with preceding interval < 0.2 s dropped; "
        "intervals < 0.2 s dropped before the median and the ratios",
    },
}
NUMERIC = (
    "amp_slope", "open_vel_slope", "close_vel_slope", "amp_late_early", "amp_min_initial",
    "decrement_onset_frac", "iti_slope", "n_hesitations", "n_halts", "longest_pause_ratio",
    "longest_pause_s",
)  # fmt: skip
SPLIT = ("amp_slope", "open_vel_slope", "close_vel_slope", "amp_late_early")
COLUMNS = (
    "ref", "qc_status", "n_valid_taps", *NUMERIC[:5], "decrement_present", *NUMERIC[5:],
    *(f + SENS for f in NUMERIC[:5]), "decrement_present" + SENS, *(f + SENS for f in NUMERIC[5:]),
    "nan_reasons",
)  # fmt: skip

Result = tuple[float, str | None]
NAN = float("nan")


def definitions_sha8() -> str:
    text = json.dumps(DEFINITIONS, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[: fa.HASH_LEN]


# --------------------------------------------------------------------------- pure functions


def _short(minimum: int, unit: str = "n_valid") -> str:
    return f"{unit}<{minimum}"


def slope_per_10s(t: np.ndarray, y: np.ndarray, minimum: int = MIN_TAPS) -> Result:
    """Theil-Sen slope of y on t divided by mean(y), as a fraction of the mean per 10 s."""
    if len(y) < minimum:
        return NAN, _short(minimum)
    mean = float(np.mean(y))
    if not mean > 0:
        return NAN, "mean<=0"
    return float(theilslopes(y, t)[0]) / mean * PER_SECONDS, None


def late_early(y: np.ndarray) -> Result:
    """mean(last k) / mean(first k), k = n // 3."""
    if len(y) < MIN_THIRDS:
        return NAN, _short(MIN_THIRDS)
    k = len(y) // 3
    first = float(np.mean(y[:k]))
    return (float(np.mean(y[-k:])) / first, None) if first > 0 else (NAN, "initial<=0")


def _initial(y: np.ndarray) -> float:
    return float(np.median(y[:INITIAL_TAPS]))


def min_initial(y: np.ndarray) -> Result:
    """min of the interior 3-point rolling median / median of the first 3 taps."""
    if len(y) < MIN_TAPS:
        return NAN, _short(MIN_TAPS)
    roll = pd.Series(y).rolling(ROLL, center=True, min_periods=ROLL).median().dropna()
    base = _initial(y)
    return (float(roll.min()) / base, None) if base > 0 else (NAN, "initial<=0")


def decrement(y: np.ndarray) -> tuple[bool | None, Result]:
    """(present, (onset fraction, reason)). present is None below the minimum."""
    n = len(y)
    if n < MIN_TAPS:
        reason = _short(MIN_TAPS)
        return None, (NAN, reason)
    sm = pd.Series(y).rolling(ROLL, center=True, min_periods=1).median().to_numpy()
    below = sm < DECREMENT_FRACTION * _initial(y)
    for i in range(n - DECREMENT_RUN + 1):
        if below[i : i + DECREMENT_RUN].all() and below[i:].mean() >= DECREMENT_SHARE:
            return True, (i / n, None)
    return False, (NAN, "no_decrement")


def interval_features(t: np.ndarray, iti: np.ndarray) -> dict[str, Result]:
    """Interval features from the valid intervals and the peak times of the later taps."""
    n = len(iti)
    if n < MIN_INTERVALS:
        reason = _short(MIN_INTERVALS, "n_intervals")
        return {k: (NAN, reason) for k in NUMERIC[6:]}
    ratio = iti / float(np.median(iti))
    mean = float(np.mean(iti))
    slope = float(theilslopes(iti, t)[0]) / mean * PER_SECONDS
    return {
        "iti_slope": (slope, None),
        "n_hesitations": (float(((ratio > HESITATION_RATIO) & (ratio <= HALT_RATIO)).sum()), None),
        "n_halts": (float((ratio > HALT_RATIO).sum()), None),
        "longest_pause_ratio": (float(ratio.max()), None),
        "longest_pause_s": (float(iti.max()), None),
    }


# --------------------------------------------------------------------------- one clip


def _clip_set(table: pd.DataFrame, sens: bool) -> tuple[dict[str, Any], list[str]]:
    sfx = SENS if sens else ""
    if sens:
        taps = table[~table["short_iti"].to_numpy(dtype=bool)]
        iv = table[
            table["valid_iti"].to_numpy(dtype=bool) & ~table["short_iti"].to_numpy(dtype=bool)
        ]
    else:
        taps = table[table["valid_amp"].to_numpy(dtype=bool)]
        iv = table[table["valid_iti"].to_numpy(dtype=bool)]
    t = taps["t_peak_s"].to_numpy(dtype=float)
    amp = taps["amplitude"].to_numpy(dtype=float)
    results: dict[str, Result] = {
        "amp_slope": slope_per_10s(t, amp),
        "open_vel_slope": slope_per_10s(t, taps["open_vel_peak"].to_numpy(dtype=float)),
        "close_vel_slope": slope_per_10s(t, taps["close_vel_peak"].to_numpy(dtype=float)),
        "amp_late_early": late_early(amp),
        "amp_min_initial": min_initial(amp),
    }
    present, onset = decrement(amp)
    results["decrement_onset_frac"] = onset
    results.update(
        interval_features(
            iv["t_peak_s"].to_numpy(dtype=float), iv["iti_before_s"].to_numpy(dtype=float)
        )
    )
    out: dict[str, Any] = {"decrement_present" + sfx: present}
    reasons = []
    if present is None:
        reasons.append(f"decrement_present{sfx}:{onset[1]}")
    for name, (value, reason) in results.items():
        out[name + sfx] = value
        if reason:
            reasons.append(f"{name}{sfx}:{reason}")
    return out, reasons


def clip_row(table: pd.DataFrame, ref: str, qc_status: str) -> dict[str, Any]:
    """One features_b.csv row from a clip's tap table."""
    row: dict[str, Any] = {
        "ref": ref,
        "qc_status": qc_status,
        "n_valid_taps": int(table["valid_amp"].sum()),
    }
    reasons: list[str] = []
    for sens in (False, True):
        values, why = _clip_set(table, sens)
        row.update(values)
        reasons += why
    row["nan_reasons"] = ";".join(reasons)
    return {c: row[c] for c in COLUMNS}


# --------------------------------------------------------------------------- files


def run(taps_dir: Path, out_dir: Path, summary_path: Path, docs_dir: Path) -> dict[str, Any]:
    refs, tables, summary, phash, tool = fa.load_clips(taps_dir, summary_path, docs_dir)
    rows = [
        clip_row(t, r, str(summary.at[r, "qc_status"])) for r, t in zip(refs, tables, strict=True)
    ]
    feats = pd.DataFrame(rows, columns=list(COLUMNS))
    meta = {
        "taps_folder": taps_dir.name,
        "tap_params_sha8": phash,
        "taps_tool": tool,
        "features_tool": TOOL_VERSION,
        "tap_table_version": tt.TABLE_VERSION,
        "taps_summary_sha8": fa.sha8(summary_path),
        "qc_list_sha8": fa.sha8(docs_dir / "qc_retained.csv"),
        "definitions_version": DEFINITIONS_VERSION,
        "definitions_sha8": definitions_sha8(),
        "n_clips": len(feats),
    }
    fa._atomic(out_dir / OUT_FEATURES, lambda p: feats.to_csv(p, index=False, lineterminator="\n"))
    fa._atomic(
        out_dir / OUT_META,
        lambda p: p.write_text(json.dumps(meta, sort_keys=True, indent=1) + "\n"),
    )
    return meta


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
    except fa.Disagreement as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    except (fa.InputProblem, tt.TableError, OSError) as err:
        detail = "" if isinstance(err, OSError) else f": {err}"
        print(f"error: {type(err).__name__}{detail}", file=sys.stderr)
        return 2
    print(f"wrote {OUT_FEATURES}: {meta['n_clips']} clips")
    return 0


if __name__ == "__main__":
    sys.exit(main())
