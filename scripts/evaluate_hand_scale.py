#!/usr/bin/env python3
"""Evidence for the FT-13 hand-size normalisation decision, reproducible from landmark files.

Reads the landmark parquet files that ``models/tapping/src/hand_landmarks.py`` already wrote
(no tracking happens here) and prints and writes:

1. the candidate table: for every scale and mode, the scale jitter, the median |r| between
   the scale and the tap (both high-passed), the share of clips with |r| > 0.5, the spread
   of the normalised distance against the world-landmark distance in cm, and the IQR/median
   of the normalised open aperture across clips;
2. the synthetic zoom errors (all pixel coordinates scaled about the wrist by a ramp);
3. the approach/recede table: change of the hand size over each clip and how much of it the
   normalisation removes from the open level;
4. residual scale/tap coupling against the tap rate, and the distribution of tap rates;
5. the open aperture in 2D hand sizes against the world-landmark aperture (cm and hand sizes)
   and the tilt of the wrist-to-knuckle-9 line out of the image plane (a pose reference:
   world landmarks are a model prediction, not a measurement).

The tap rate is the simple 25%/50% crossing counter of ``plot_distance.py``. It is NOT the
tap detector (FT-19 and later). No UPDRS or CONTROL/ID label is read or used anywhere.

Clips are shown by their 8-character hash only (``hubu_fis.pseudonym``); raw clip IDs are
never printed or written to a file name. Output goes to ``data/interim/hand_scale_eval/``
(gitignored) and nowhere else: a different ``--out-dir`` must still be inside the interim root.

Exit codes: 0 ok, 1 a landmark file cannot be read or the output folder is outside the
interim root, 2 the landmark folder is missing or holds no parquet file.

Usage (from the repo root):
    python scripts/evaluate_hand_scale.py
    python scripts/evaluate_hand_scale.py --landmarks-dir data/interim/landmarks
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "models" / "tapping" / "src"))

import hand_landmarks as hl  # noqa: E402
import hand_scale as hs  # noqa: E402
from hubu_fis import pseudonym  # noqa: E402
from plot_distance import (  # noqa: E402
    count_open_close_cycles,
    thumb_index_distance_px,
    time_axis_s,
)

DEFAULT_LANDMARKS_DIR = REPO_ROOT / "data" / "interim" / "landmarks"
INTERIM_ROOT = REPO_ROOT / "data" / "interim"
DEFAULT_OUT_DIR = INTERIM_ROOT / "hand_scale_eval"

# (mode, window_s) rows of the candidate table and their labels
MODE_ROWS: tuple[tuple[str, float | None, str], ...] = (
    ("per_frame", None, "per_frame"),
    ("rolling_median", 2.0, "roll2s"),
    ("rolling_median", 3.0, "roll3s"),
    ("rolling_median", 5.0, "roll5s"),
    ("clip_median", None, "clip_median"),
)
HIGHPASS_S = hs.HIGHPASS_S
LAG_FRAMES = 9  # the tap-phase check looks at lags of up to +-9 frames (0.3 s at 30 fps)
COUPLING_LIMIT = 0.5
SLOW_RATE = 0.7  # cycles per second below which a 3 s window covers fewer than 2 cycles
WINDOW_SWEEP_S = (2.0, 3.0, 5.0, 8.0)
ZOOMS = ((0.85, 1.1), (0.7, 1.4))
FT12_CLIPS = ("245fbb13", "5f05706d", "11054d82", "02519f5d")  # the four FT-11/12 clips
TREND_WINDOW_S = 4.0


# --------------------------------------------------------------------------- helpers


highpass = hs.highpass  # moved to hand_scale.py in FT-15; kept here under the same name
correlation = hs.correlation


def lagged_max_abs_correlation(a: np.ndarray, b: np.ndarray, lags: int = LAG_FRAMES) -> float:
    """Largest |r| between ``a`` and ``b`` shifted by up to +-``lags`` frames."""
    values = [
        abs(correlation(a[max(k, 0) : len(a) + min(k, 0)], b[max(-k, 0) : len(a) - max(k, 0)]))
        for k in range(-lags, lags + 1)
    ]
    return float(np.nanmax(values)) if not np.isnan(values).all() else float("nan")


def load_clips(directory: Path) -> dict[str, pd.DataFrame]:
    """Hash -> landmark table for every parquet file in ``directory``."""
    clips: dict[str, pd.DataFrame] = {}
    for path in sorted(directory.glob("*.parquet")):
        ref = pseudonym(path.stem)
        try:
            df, _ = hl.read_landmarks(path)
        except Exception as err:  # unreadable file: say which clip, by hash only
            raise hl.HandLandmarksError(f"clip {ref}: cannot read the landmarks file") from err
        clips[ref] = df
    return clips


# --------------------------------------------------------------------------- per-clip metrics


def clip_metrics(
    df: pd.DataFrame, scale: str, mode: str, window_s: float | None
) -> dict[str, float]:
    """The per-clip numbers behind the candidate table for one scale and mode."""
    t, _ = time_axis_s(df)
    raw = thumb_index_distance_px(df)
    world_cm = hs.thumb_index_distance_world_cm(df)
    per_frame, _ = hs._guard(hs.scale_length(df, scale))
    slow = (
        pd.Series(per_frame, index=pd.to_datetime(t, unit="s"))
        .rolling("2s", center=True, min_periods=5)
        .median()
        .to_numpy()
    )
    jitter = float(np.nanmedian(np.abs(per_frame - slow)) / np.nanmedian(per_frame))
    used, _ = hs.estimate_scale(df, scale, mode, window_s=window_s or hs.DEFAULT_WINDOW_S)
    normalised = raw / used
    used_hp, raw_hp = highpass(t, used), highpass(t, raw)
    open_half = raw >= np.nanmedian(raw)
    ratio = (normalised / world_cm)[open_half & (world_cm > 0.5)]
    return {
        "jitter": jitter,
        "r": correlation(used_hp, raw_hp),
        "lag_r": lagged_max_abs_correlation(used_hp, raw_hp),
        "world_cv": float(np.nanstd(ratio) / np.nanmean(ratio)),
        "r_world": correlation(normalised, world_cm),
        "open_p95": float(np.nanpercentile(normalised, 95)),
    }


def candidate_table(clips: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-clip rows and the table summarised over clips, one row per scale and mode."""
    rows = []
    for ref, df in clips.items():
        for scale in hs.SCALES:
            for mode, window, label in MODE_ROWS:
                rows.append(
                    {"ref": ref, "scale": scale, "mode": label}
                    | clip_metrics(df, scale, mode, window)
                )
    per_clip = pd.DataFrame(rows)
    grouped = per_clip.groupby(["scale", "mode"], sort=False)

    def iqr_over_median(s: pd.Series) -> float:
        return float((s.quantile(0.75) - s.quantile(0.25)) / s.median())

    table = pd.DataFrame(
        {
            "jitter_median": grouped["jitter"].median(),
            "abs_r_median": grouped["r"].apply(lambda s: s.abs().median()),
            "share_abs_r_above_0.5_pct": grouped["r"].apply(
                lambda s: (
                    100 * float((s.abs() > COUPLING_LIMIT).sum()) / max(int(s.notna().sum()), 1)
                )
            ),
            "lag_r_median": grouped["lag_r"].median(),
            "world_cv_median": grouped["world_cv"].median(),
            "r_world_median": grouped["r_world"].median(),
            "open_p95_iqr_over_median": grouped["open_p95"].apply(iqr_over_median),
        }
    )
    return table, per_clip


# --------------------------------------------------------------------------- zoom


def zoom_clip(df: pd.DataFrame, low: float, high: float) -> pd.DataFrame:
    """All pixel coordinates scaled about the median wrist by a linear ramp ``low`` to ``high``."""
    out = df.copy()
    t = df["pts_ms"].to_numpy(dtype=float)
    ramp = low + (high - low) * (t - np.nanmin(t)) / (np.nanmax(t) - np.nanmin(t))
    cx, cy = np.nanmedian(df["x_px00"]), np.nanmedian(df["y_px00"])
    for i in range(hl.N_LANDMARKS):
        out[f"x_px{i:02d}"] = cx + (df[f"x_px{i:02d}"] - cx) * ramp
        out[f"y_px{i:02d}"] = cy + (df[f"y_px{i:02d}"] - cy) * ramp
    return out


def zoom_table(clips: dict[str, pd.DataFrame], refs: Sequence[str]) -> pd.DataFrame:
    """Relative error of the normalised distance after a synthetic zoom, median over ``refs``."""
    rows = []
    for ref in refs:
        for low, high in ZOOMS:
            zoomed = zoom_clip(clips[ref], low, high)
            for scale in ("w9", "median4"):
                for mode, window, label in MODE_ROWS:
                    w = window or hs.DEFAULT_WINDOW_S
                    a = hs.normalised_distance(clips[ref], scale, mode, window_s=w).values
                    b = hs.normalised_distance(zoomed, scale, mode, window_s=w).values
                    err = np.abs(b / a - 1)
                    err = err[np.isfinite(err)]
                    rows.append(
                        {
                            "zoom": f"{low:g}x to {high:g}x",
                            "scale": scale,
                            "mode": label,
                            "median_err": float(np.median(err)),
                            "p95_err": float(np.percentile(err, 95)),
                            "max_err": float(err.max()),
                        }
                    )
    table = pd.DataFrame(rows).groupby(["zoom", "scale", "mode"], sort=False).median()
    return table


# --------------------------------------------------------------------------- approach / recede


def hand_size_change(df: pd.DataFrame) -> tuple[float, float]:
    """Linear change of the 2 s rolling median of w9 over the clip (fraction) and Spearman rho."""
    t, _ = time_axis_s(df)
    series = pd.Series(hs.scale_length(df, "w9"), index=pd.to_datetime(t, unit="s"))
    slow = series.rolling("2s", center=True, min_periods=5).median().to_numpy()
    ok = np.isfinite(slow) & np.isfinite(t)
    k = np.polyfit(t[ok], slow[ok], 1)
    change = (np.polyval(k, t[ok][-1]) - np.polyval(k, t[ok][0])) / np.nanmedian(slow)
    constant = np.ptp(slow[ok]) == 0
    rho = (
        float("nan") if constant else pd.Series(slow[ok]).corr(pd.Series(t[ok]), method="spearman")
    )
    return float(change), float(rho)


def open_level_trend(t: np.ndarray, x: np.ndarray, window_s: float = TREND_WINDOW_S) -> float:
    """Linear trend of the open level over the whole clip, as a fraction of its mean.

    The open level is the 90th percentile in consecutive ``window_s`` blocks; the slope of a
    line through the blocks is multiplied by the clip duration, so that it is comparable with
    ``hand_size_change`` (which is also a change over the whole clip).
    """
    centres, highs = [], []
    for start in np.arange(np.nanmin(t), np.nanmax(t) - window_s + 1e-9, window_s):
        inside = (t >= start) & (t < start + window_s) & np.isfinite(x)
        if inside.sum() > 20:
            centres.append(start + window_s / 2)
            highs.append(np.percentile(x[inside], 90))
    if len(centres) < 3:
        return float("nan")
    slope = np.polyfit(centres, highs, 1)[0]
    return float(slope * (np.nanmax(t) - np.nanmin(t)) / np.mean(highs))


def approach_table(clips: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per clip: hand-size change and the open-level trend before and after normalisation."""
    rows = []
    for ref, df in clips.items():
        t, _ = time_axis_s(df)
        change, rho = hand_size_change(df)
        raw_trend = open_level_trend(t, thumb_index_distance_px(df))
        norm_trend = open_level_trend(t, hs.normalised_distance(df).values)
        removed = (raw_trend - norm_trend) / change if abs(change) >= 0.05 else float("nan")
        rows.append(
            {
                "ref": ref,
                "hand_change": change,
                "spearman_vs_time": rho,
                "open_trend_raw": raw_trend,
                "open_trend_normalised": norm_trend,
                "camera_trend_removed": removed,
            }
        )
    out = pd.DataFrame(rows).set_index("ref")
    return out.reindex(out["hand_change"].abs().sort_values(ascending=False).index)


# --------------------------------------------------------------------------- tap rate and coupling


def tap_rate(df: pd.DataFrame) -> float:
    """Cycles per second from the simple 25/50 crossing counter (NOT the tap detector)."""
    t, _ = time_axis_s(df)
    finite = t[np.isfinite(t)]
    duration = float(finite.max() - finite.min()) if finite.size else float("nan")
    cycles = count_open_close_cycles(thumb_index_distance_px(df))
    return cycles / duration if duration > 0 else float("nan")


def coupling_table(clips: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per clip: tap rate and |r| between the w9 rolling scale and the tap, for each window."""
    rows = []
    for ref, df in clips.items():
        t, _ = time_axis_s(df)
        raw_hp = highpass(t, thumb_index_distance_px(df))
        row: dict[str, Any] = {"ref": ref, "rate": tap_rate(df)}
        for window in WINDOW_SWEEP_S:
            used, _ = hs.estimate_scale(df, "w9", "rolling_median", window_s=window)
            row[f"abs_r_{window:g}s"] = abs(correlation(highpass(t, used), raw_hp))
        rows.append(row)
    return pd.DataFrame(rows).set_index("ref").sort_values("rate")


def rate_distribution(rates: pd.Series) -> dict[str, float]:
    r = rates.dropna()
    q = r.quantile([0.25, 0.5, 0.75])
    return {
        "n": float(len(r)),
        "min": float(r.min()),
        "q25": float(q[0.25]),
        "median": float(q[0.5]),
        "q75": float(q[0.75]),
        "max": float(r.max()),
        "above_2": float((r > 2).sum()),
        "above_3": float((r > 3).sum()),
    }


def coupling_summary(table: pd.DataFrame) -> dict[str, Any]:
    """Which clips keep |r| > 0.5 at the default window, and how that relates to the rate."""
    col = f"abs_r_{hs.DEFAULT_WINDOW_S:g}s"
    flagged = table[table[col] > COUPLING_LIMIT]
    slow = table[table["rate"] < SLOW_RATE]
    ok = table[["rate", col]].dropna()
    return {
        "window_column": col,
        "flagged": {
            ref: {"rate": float(r.rate), "abs_r": float(r[col])} for ref, r in flagged.iterrows()
        },
        "slow_clips": int(len(slow)),
        "slow_clips_flagged": int((slow[col] > COUPLING_LIMIT).sum()),
        "pearson_rate_vs_abs_r": float(ok["rate"].corr(ok[col])) if len(ok) > 2 else float("nan"),
        "spearman_rate_vs_abs_r": float(ok["rate"].corr(ok[col], method="spearman"))
        if len(ok) > 2
        else float("nan"),
        "two_cycle_window_s_for_slowest": float(2.0 / table["rate"].min())
        if table["rate"].min() > 0
        else float("nan"),
    }


# --------------------------------------------------------------------------- world reference


def tilt_deg(df: pd.DataFrame, a: int, b: int) -> np.ndarray:
    """Angle of the world-landmark line a to b out of the image plane, in degrees."""
    d = np.column_stack(
        [df[f"w{c}{b:02d}"].to_numpy(float) - df[f"w{c}{a:02d}"].to_numpy(float) for c in "xyz"]
    )
    norm = np.linalg.norm(d, axis=1)
    return np.asarray(np.degrees(np.arcsin(np.clip(np.abs(d[:, 2]) / norm, 0, 1))))


def world_aperture_table(clips: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per clip: open aperture (95th percentile) in 2D hand sizes, 3D hand sizes and cm."""
    rows = []
    for ref, df in clips.items():
        normalised = hs.normalised_distance(df).values
        world_cm = hs.thumb_index_distance_world_cm(df)
        w9_cm = hs.scale_length(df, "w9", "world") * 100.0
        rows.append(
            {
                "ref": ref,
                "open_2d_hand_sizes": float(np.nanpercentile(normalised, 95)),
                "open_3d_hand_sizes": float(np.nanpercentile(world_cm / w9_cm, 95)),
                "open_cm": float(np.nanpercentile(world_cm, 95)),
                "w9_cm": float(np.nanmedian(w9_cm)),
                "w9_tilt_deg": float(np.nanmedian(tilt_deg(df, 0, 9))),
            }
        )
    table = pd.DataFrame(rows).set_index("ref")
    table["ratio_2d_over_3d"] = table["open_2d_hand_sizes"] / table["open_3d_hand_sizes"]
    return table


def world_summary(table: pd.DataFrame) -> dict[str, float]:
    inverse_cos = 1 / np.cos(np.radians(table["w9_tilt_deg"]))
    return {
        "ratio_2d_over_3d_min": float(table["ratio_2d_over_3d"].min()),
        "ratio_2d_over_3d_median": float(table["ratio_2d_over_3d"].median()),
        "ratio_2d_over_3d_max": float(table["ratio_2d_over_3d"].max()),
        "corr_ratio_vs_inverse_cos_tilt": float(table["ratio_2d_over_3d"].corr(inverse_cos)),
        "open_cm_min": float(table["open_cm"].min()),
        "open_cm_max": float(table["open_cm"].max()),
        "w9_cm_median": float(table["w9_cm"].median()),
        "w9_cm_min": float(table["w9_cm"].min()),
        "w9_cm_max": float(table["w9_cm"].max()),
    }


# --------------------------------------------------------------------------- command line


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _show(title: str, table: pd.DataFrame, digits: int = 3) -> None:
    print(f"\n== {title}")
    print(table.round(digits).to_string())


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--landmarks-dir", type=Path, default=DEFAULT_LANDMARKS_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--interim-root", type=Path, default=INTERIM_ROOT, help="outputs must stay inside"
    )
    args = ap.parse_args(argv)

    if not args.landmarks_dir.is_dir():
        print(f"error: landmark folder not found: {args.landmarks_dir}", file=sys.stderr)
        return 2
    if not list(args.landmarks_dir.glob("*.parquet")):
        print(f"error: no landmark parquet files in {args.landmarks_dir}", file=sys.stderr)
        return 2
    if not _inside(args.out_dir, args.interim_root):
        print(f"error: output folder must be inside {args.interim_root}", file=sys.stderr)
        return 1
    try:
        clips = load_clips(args.landmarks_dir)
    except hl.HandLandmarksError as err:
        print(f"FAILED: {err}", file=sys.stderr)
        return 1

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    print(f"{len(clips)} clips with landmarks (hashed references only; no labels used)")

    candidates, per_clip = candidate_table(clips)
    _show("1. candidate table (median over clips)", candidates)

    approach = approach_table(clips)
    drift = [r for r in approach.index if abs(approach.loc[r, "hand_change"]) >= 0.05][:4]
    control = [approach["hand_change"].abs().idxmin()]
    zoom_refs = [r for r in dict.fromkeys([*FT12_CLIPS, *drift, *control]) if r in clips]
    _show(
        f"2. synthetic zoom, relative error (median over {len(zoom_refs)} clips)",
        zoom_table(clips, zoom_refs),
        4,
    )
    _show("3. approach/recede (largest hand-size change first; first 8 rows)", approach.head(8))
    print(f"   control (smallest hand-size change): {control[0]}")

    coupling = coupling_table(clips)
    summary = coupling_summary(coupling)
    rates = rate_distribution(coupling["rate"])
    _show("4a. tap rate (25/50 crossing counter, NOT the tap detector) and |r| by window", coupling)
    print("\n   residual coupling:", json.dumps(summary, indent=2, sort_keys=True))
    print("\n== 4b. tap-rate distribution, cycles per second (same exploratory counter)")
    print("  ", json.dumps(rates, sort_keys=True))

    world = world_aperture_table(clips)
    world_stats = world_summary(world)
    _show(
        "5. open aperture, 2D against world landmarks (first 8 clips by hash order)", world.head(8)
    )
    print("\n  ", json.dumps(world_stats, indent=2, sort_keys=True))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    world.to_csv(args.out_dir / "world_aperture.csv")
    candidates.to_csv(args.out_dir / "candidates.csv")
    per_clip.to_csv(args.out_dir / "candidates_per_clip.csv", index=False)
    approach.to_csv(args.out_dir / "approach_recede.csv")
    coupling.to_csv(args.out_dir / "coupling_by_rate.csv")
    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {"coupling": summary, "rates": rates, "clips": len(clips)}, indent=2, sort_keys=True
        )
    )
    print(f"\nwrote 6 files to {args.out_dir} (named by content, clips shown by hash)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
