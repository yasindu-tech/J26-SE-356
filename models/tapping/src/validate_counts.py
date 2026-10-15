"""Compare the detector's tap counts with the blind manual counts (task FT-25).

    python models/tapping/src/validate_counts.py report [--synthetic] [--seed N] [--boot B]

Window = [window_first, window_last], inclusive video frames of the clip's smoothed signal.
Detector count = rows of taps/<ref>.parquet; manual count = tap_frames inside the window (frames
outside are n_outside, edge partials never count); matching is on frame numbers, never times.
d = detector - manual. Primary metric: weighted MAE (weights from the export) on the 40 primary
clips; ICC(2,1) is unweighted; the hard set resamples by participant and is never pooled.
Matching for FT-26: a detector tap owns [trough_left, trough_right]; a second pass pairs the
left-over FN and FP of a clip within round(0.15 * fps) frames as TP_offset (kept apart from TP).
Weights must equal the frame sizes recomputed from the repo (retained minus the seen clips).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import manual_counts as mc
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from plot_distance import save_figure
from scipy import stats
from signal_processing import is_inside

TOOL_VERSION = "validate-0.1.0"
DEFAULT_SEED = 2026
DEFAULT_BOOT = 10_000
LEVEL = 0.95
MAE_TARGET = 1.0
ICC_TARGET = 0.90
SECOND_KINDS = ("inter_rater", "intra_rater_recount")
REPORT_NAME = "validation_report.json"
VALIDATION_DIR = "validation"
SYNTHETIC_DIR = "validation_synthetic"
SYNTHETIC_PREFIX = "SYNTHETIC_"
SET_RATER = {"primary": "R1", "second": "R2", "hard": "R2"}
N_PRIMARY = sum(mc.ALLOCATION.values())
N_SECOND = 3 * sum(mc.SECOND_PER_TERTILE.values())
WINDOW_RULE = (
    "window = [window_first, window_last] inclusive video frames of the smoothed signal; manual "
    "taps outside are n_outside; edge partials are never taps; matching on frame numbers"
)
ICC_WORDING = (
    "ICC(2,1) within the stratified validation sample of 40 clips (one per participant; cells by "
    "detector rate; flagged clips oversampled); not an estimate of the ICC in the full dataset. "
    "ICC is a weak gate here because the between-clip spread of counts (12-71 taps) is large."
)
CLIP_COLUMNS = (
    "ref", "set", "rater", "cell", "weight", "window_first", "window_last", "n_det", "n_man",
    "n_outside", "d", "abs_d", "tp", "tp_offset", "fn", "fp", "manual_edge_partials",
    "detector_edge_ends", "qc_status", "confidence",
)  # fmt: skip
TAP_COLUMNS = (
    "ref", "set", "rater", "kind", "match_pass", "manual_frame", "detector_frame", "offset_frames",
    "position", "in_nan_gap", "detector_flags", "qc_status", "cell",
)  # fmt: skip
KINDS = ("TP", "TP_offset", "FN", "FP")
TOLERANCE_S = 0.15  # the detector's minimum tap spacing


class Refused(ValueError):
    """The counts are incomplete or inconsistent: nothing is analysed (exit code 1)."""


# --------------------------------------------------------------------------- statistics


def icc_stack(x: np.ndarray) -> np.ndarray:
    """ICC(2,1) (two-way random, absolute agreement, single measure) of each n x k matrix in a
    (B, n, k) stack, from the ANOVA mean squares."""
    _, n, k = x.shape
    grand = x.mean(axis=(1, 2), keepdims=True)
    rows = x.mean(axis=2, keepdims=True)
    cols = x.mean(axis=1, keepdims=True)
    msr = k * ((rows - grand) ** 2).sum(axis=(1, 2)) / (n - 1)
    msc = n * ((cols - grand) ** 2).sum(axis=(1, 2)) / (k - 1)
    mse = ((x - rows - cols + grand) ** 2).sum(axis=(1, 2)) / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, (msr - mse) / den, np.nan)


def icc2(man: np.ndarray, det: np.ndarray) -> dict[str, Any]:
    """ICC(2,1) of the (manual, detector) pairs with its F-based 95% CI (McGraw and Wong)."""
    x = np.stack([np.asarray(man, float), np.asarray(det, float)], axis=1)
    n, k = x.shape
    icc = float(icc_stack(x[None])[0])
    grand, rows, cols = x.mean(), x.mean(axis=1), x.mean(axis=0)
    msr = k * ((rows - grand) ** 2).sum() / (n - 1)
    msc = n * ((cols - grand) ** 2).sum() / (k - 1)
    mse = ((x - rows[:, None] - cols[None, :] + grand) ** 2).sum() / ((n - 1) * (k - 1))
    lo = hi = float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        fj = msc / mse
        dfn, dfe = n - 1, (n - 1) * (k - 1)
        a = n * (1 + (k - 1) * icc) - k * icc
        v = dfe * (k * icc * fj + a) ** 2 / (dfn * k**2 * icc**2 * fj**2 + a**2)
        if np.isfinite(v) and v > 0:
            alpha = 1 - LEVEL
            fu, fl = stats.f.ppf(1 - alpha / 2, n - 1, v), stats.f.ppf(1 - alpha / 2, v, n - 1)
            c = k * msc + (k * n - k - n) * mse
            lo = float(n * (msr - fu * mse) / (fu * c + n * msr))
            hi = float(n * (fl * msr - mse) / (c + n * fl * msr))
    return {"icc": icc, "ci_f": [lo, hi], "n": n}


def error_metrics(man: np.ndarray, det: np.ndarray, w: np.ndarray | None = None) -> dict[str, Any]:
    """Error summary of d = detector - manual; ``w`` adds the weighted versions."""
    d = np.asarray(det, float) - np.asarray(man, float)
    a = np.abs(d)
    out: dict[str, Any] = {
        "n": len(d),
        "mae": float(a.mean()),
        "bias": float(d.mean()),
        "max_abs_error": float(a.max()),
        "share_abs_le_1": float((a <= 1).mean()),
        "share_abs_le_2": float((a <= 2).mean()),
    }
    positive = np.asarray(man, float) > 0
    out["relative_mae"] = float((a[positive] / np.asarray(man, float)[positive]).mean())
    out["relative_mae_n"] = int(positive.sum())
    if w is not None:
        w = np.asarray(w, float)
        out["mae_weighted"] = float((w * a).sum() / w.sum())
        out["bias_weighted"] = float((w * d).sum() / w.sum())
        out["share_abs_le_1_weighted"] = float((w * (a <= 1)).sum() / w.sum())
        out["share_abs_le_2_weighted"] = float((w * (a <= 2)).sum() / w.sum())
        out["relative_mae_weighted"] = float(
            (w[positive] * a[positive] / np.asarray(man, float)[positive]).sum() / w[positive].sum()
        )
    return out


def bland_altman(man: np.ndarray, det: np.ndarray) -> dict[str, Any]:
    d = np.asarray(det, float) - np.asarray(man, float)
    n = len(d)
    bias, sd = float(d.mean()), float(d.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1) * sd * np.sqrt(1 / n + 1.96**2 / (2 * (n - 1))))
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
    return {
        "bias": bias, "sd": sd, "loa_lower": lo, "loa_upper": hi,
        "loa_lower_ci": [lo - half, lo + half], "loa_upper_ci": [hi - half, hi + half],
    }  # fmt: skip


def percentile_ci(values: np.ndarray) -> list[float]:
    if np.isnan(values).all():
        return [float("nan"), float("nan")]
    lo, hi = np.nanpercentile(values, [100 * (1 - LEVEL) / 2, 100 * (1 + LEVEL) / 2])
    return [float(lo), float(hi)]


def resample_indices(cells: np.ndarray, boot: int, rng: np.random.Generator) -> np.ndarray:
    """(boot, n) clip indices; every row keeps each cell's size and draws only inside the cell."""
    parts = []
    for cell in mc.CELL_ORDER:
        members = np.flatnonzero(cells == cell)
        if len(members):
            parts.append(members[rng.integers(0, len(members), size=(boot, len(members)))])
    return np.concatenate(parts, axis=1)


def resample_clusters(
    groups: Sequence[str], boot: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """``boot`` index arrays made of whole participants drawn with replacement."""
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(int(inverse.max()) + 1)]
    draws = rng.integers(0, len(members), size=(boot, len(members)))
    return [np.concatenate([members[g] for g in row]) for row in draws]


def boot_primary(
    man: np.ndarray, det: np.ndarray, w: np.ndarray, cells: np.ndarray, boot: int, seed: int
) -> dict[str, list[float]]:
    idx = resample_indices(cells, boot, np.random.default_rng([seed, 1]))
    a = np.abs(det[idx] - man[idx])
    wmae = (w[idx] * a).sum(axis=1) / w[idx].sum(axis=1)
    icc = icc_stack(np.stack([man[idx], det[idx]], axis=2))
    return {"mae_weighted": percentile_ci(wmae), "icc": percentile_ci(icc)}


def boot_simple(
    man: np.ndarray, det: np.ndarray, boot: int, seed: int, stream: int,
    groups: Sequence[str] | None = None,
) -> dict[str, list[float]]:  # fmt: skip
    """Percentile CIs for MAE and ICC; clips are resampled by participant when ``groups`` is given."""
    rng = np.random.default_rng([seed, stream])
    if groups is None:
        n = len(man)
        draws = list(rng.integers(0, n, size=(boot, n)))
    else:
        draws = resample_clusters(groups, boot, rng)
    mae = np.array([np.abs(det[i] - man[i]).mean() for i in draws])
    icc = np.array([icc_stack(np.stack([man[i], det[i]], axis=1)[None])[0] for i in draws])
    return {"mae": percentile_ci(mae), "icc": percentile_ci(icc)}


def decide(
    mae: float, mae_ci: Sequence[float], icc: float, icc_ci: Sequence[float]
) -> dict[str, Any]:
    """The pre-specified decision fields (point estimates and bootstrap CIs)."""
    target_met = bool(mae <= MAE_TARGET and icc >= ICC_TARGET)
    established = bool(mae_ci[1] <= MAE_TARGET and icc_ci[0] >= ICC_TARGET)
    not_met = bool(mae > MAE_TARGET)
    statements: list[str] = []
    if mae <= MAE_TARGET < mae_ci[1]:
        statements.append(
            f"Met on the point estimate; the 95% CI ({mae_ci[0]:.2f}-{mae_ci[1]:.2f}) "
            "includes values above 1.0"
        )
    if icc >= ICC_TARGET > icc_ci[0]:
        statements.append(
            f"Met on the point estimate; the 95% CI ({icc_ci[0]:.2f}-{icc_ci[1]:.2f}) "
            "includes values below 0.90"
        )
    status = (
        "not_met" if not_met
        else "icc_below_target" if not target_met
        else "established" if established
        else "met_on_point_estimate"
    )  # fmt: skip
    return {
        "target_met": target_met, "established": established, "not_met": not_met,
        "status": status, "triggers_ft26": not_met, "statements": statements,
        "decision_ci": "bootstrap percentile, resampled within cells",
        "mae_target": MAE_TARGET, "icc_target": ICC_TARGET,
    }  # fmt: skip


# --------------------------------------------------------------------------- export rows


@dataclass(frozen=True)
class CountRow:
    ref: str
    rater: str
    set: str
    frames: tuple[int, ...]
    n_taps: int
    n_edge: int
    confidence: str
    cell: str
    weight: float | None
    first: int
    last: int


def _int(text: str, what: str) -> int:
    try:
        return int(text)
    except ValueError as err:
        raise mc.InputError(f"the export has a non-integer {what}") from err


def read_export(path: Path) -> pd.DataFrame:
    if not Path(path).is_file():
        raise mc.InputError(f"{mc.EXPORT_NAME} not found; run manual_counts.py export first")
    df = mc.read_text(Path(path))
    if list(df.columns) != list(mc.EXPORT_COLUMNS):
        raise mc.InputError(
            f"{mc.EXPORT_NAME} must have the columns {', '.join(mc.EXPORT_COLUMNS)}"
        )
    return df


def parse_rows(df: pd.DataFrame) -> list[CountRow]:
    rows = []
    for r in df.itertuples(index=False):
        frames = tuple(_int(x, "tap frame") for x in str(r.tap_frames).split(";") if x != "")
        try:
            weight = float(r.weight) if str(r.weight) != "" else None
        except ValueError as err:
            raise mc.InputError("the export has a non-numeric weight") from err
        rows.append(
            CountRow(
                str(r.ref), str(r.rater), str(r.set), frames, _int(str(r.n_taps), "n_taps"),
                _int(str(r.n_edge_partials), "n_edge_partials"), str(r.confidence), str(r.cell),
                weight, _int(str(r.window_first), "window_first"), _int(str(r.window_last), "window_last"),
            )
        )  # fmt: skip
    return rows


def check_export(rows: Sequence[CountRow], frame: dict[str, int]) -> list[str]:
    """Why the counts cannot be analysed. No partial real counts are ever analysed: reading
    unfinished counts would unblind the rater, so R1 must have all 40 primary rows."""
    problems: list[str] = []
    sets = {k: [r for r in rows if r.set == k] for k in SET_RATER}
    odd = sorted(r.ref for r in rows if r.set not in SET_RATER)
    if odd:
        problems.append(f"rows with an unknown set: {mc._listed(odd)}")
    wrong = sorted(r.ref for r in rows if r.set in SET_RATER and r.rater != SET_RATER[r.set])
    if wrong:
        problems.append(f"rows counted by the wrong rater: {mc._listed(wrong)}")
    keys = [(r.set, r.ref) for r in rows]
    dups = sorted({k[1] for k in keys if keys.count(k) > 1})
    if dups:
        problems.append(f"duplicate rows: {mc._listed(dups)}")
    primary = sets["primary"]
    if len(primary) != N_PRIMARY:
        problems.append(
            f"R1 has {len(primary)} primary rows, {N_PRIMARY} are needed (no partial counts)"
        )
    for cell, n in mc.ALLOCATION.items():
        have = sum(r.cell == cell for r in primary)
        if have != n:
            problems.append(f"cell {cell}: {have} primary clips, expected {n}")
        weights = {r.weight for r in primary if r.cell == cell}
        weight = next(iter(weights)) if len(weights) == 1 else None
        if weight is None or not np.isfinite(weight) or weight <= 0:
            problems.append(f"cell {cell}: the weight must be one positive number")
        elif abs(weight * have - frame[cell]) > 1e-9:
            problems.append(
                f"cell {cell}: weight x clips is {weight * have:.6g}, the frame has {frame[cell]}"
            )
    total = sum(r.weight for r in primary if r.weight is not None)
    if abs(total - sum(frame.values())) > 1e-9:
        problems.append(f"the weights sum to {total:.6g}, the frame size is {sum(frame.values())}")
    second, hard = sets["second"], sets["hard"]
    if len(second) not in (0, N_SECOND):
        problems.append(f"R2 has {len(second)} second-rater rows, 0 or {N_SECOND} are needed")
    if not {r.ref for r in second} <= {r.ref for r in primary}:
        problems.append("second-rater clips must be primary clips")
    if len(hard) > len(mc.HARD) or ({r.ref for r in hard} & {r.ref for r in primary}):
        problems.append("hard-set rows are too many or overlap the primary set")
    bad = sorted(
        r.ref
        for r in rows
        if r.n_taps != len(r.frames) or list(r.frames) != sorted(set(r.frames)) or r.first > r.last
    )
    if bad:
        problems.append(f"n_taps, tap frames or window are inconsistent: {mc._listed(bad)}")
    return problems


def expected_frame(docs_root: Path, summary_path: Path) -> dict[str, int]:
    """N_frame per cell recomputed from the repo, not from the export: the retained clips minus
    the seen clips, cell = detector-rate tertile (cuts over all retained) x QC flag."""
    population = mc.load_population(Path(docs_root), Path(summary_path))
    cell_of = mc.cells(population)
    seen = set(mc.PRACTICE) | set(mc.HARD)
    missing = sorted(seen - set(cell_of.index))
    if missing:
        raise Refused(f"seen clips are not in the retained list: {mc._listed(missing)}")
    counts = cell_of.drop(sorted(seen)).value_counts()
    return {cell: int(counts.get(cell, 0)) for cell in mc.CELL_ORDER}


# --------------------------------------------------------------------------- detector data


@dataclass
class Inputs:
    smoothed_dir: Path
    taps_dir: Path
    summary: pd.DataFrame  # indexed by ref
    _signals: dict[str, pd.DataFrame] = field(default_factory=dict)
    _taps: dict[str, pd.DataFrame] = field(default_factory=dict)

    def signal(self, ref: str) -> pd.DataFrame:
        if ref not in self._signals:
            try:
                sig = pd.read_parquet(
                    self.smoothed_dir / f"{ref}.parquet", columns=["frame", "t_s", "value"]
                )
            except (OSError, ValueError, KeyError) as err:
                raise mc.InputError(f"clip {ref}: no usable smoothed file") from err
            if len(sig) < 2:
                raise mc.InputError(f"clip {ref}: the smoothed signal has fewer than 2 samples")
            self._signals[ref] = sig
        return self._signals[ref]

    def taps(self, ref: str) -> pd.DataFrame:
        if ref not in self._taps:
            cols = ["frame", "trough_left_s", "trough_right_s", "flags"]
            try:
                self._taps[ref] = pd.read_parquet(self.taps_dir / f"{ref}.parquet", columns=cols)
            except (OSError, ValueError, KeyError) as err:
                raise mc.InputError(f"clip {ref}: no usable taps file") from err
        return self._taps[ref]


def load_inputs(interim: Path) -> Inputs:
    path = interim / "taps" / "taps_summary.csv"
    if not path.is_file():
        raise mc.InputError("taps_summary.csv not found")
    summary = mc.read_text(path)
    needed = {"ref", "n_taps", "n_open_edge_ends", "qc_status", "fps"}
    if not needed <= set(summary.columns):
        raise mc.InputError(f"taps_summary.csv lacks one of: {', '.join(sorted(needed))}")
    return Inputs(interim / "smoothed", interim / "taps", summary.set_index("ref"))


def detector_spans(
    taps: pd.DataFrame, sig: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Peak, window start and window end of each detector tap, as video frame numbers."""
    if taps.empty:
        e = np.empty(0, dtype=int)
        return e, e, e, []
    frames = sig["frame"].to_numpy(dtype=int)
    t_rel = sig["t_s"].to_numpy(dtype=float) - float(sig["t_s"].iloc[0])  # taps.py clock

    def at(times: np.ndarray) -> np.ndarray:
        i = np.clip(np.searchsorted(t_rel, times), 1, len(t_rel) - 1)
        j = np.where(np.abs(t_rel[i - 1] - times) <= np.abs(t_rel[i] - times), i - 1, i)
        if np.abs(t_rel[j] - times).max() > 1e-6:
            raise Refused("detector trough times do not lie on the signal samples")
        return frames[j]

    left = at(taps["trough_left_s"].to_numpy(dtype=float))
    right = at(taps["trough_right_s"].to_numpy(dtype=float))
    return taps["frame"].to_numpy(dtype=int), left, right, [str(f) for f in taps["flags"]]


# --------------------------------------------------------------------------- matching


def position_of(frame: int, peaks: np.ndarray, first: int, last: int) -> str:
    """first_interval: before the 2nd detector peak; last_interval: after the 2nd-last."""
    if len(peaks) >= 3:
        return (
            "first_interval"
            if frame < peaks[1]
            else "last_interval"
            if frame > peaks[-2]
            else "middle"
        )
    return "first_interval" if frame <= (first + last) / 2 else "last_interval"


def tolerance_frames(fps: float) -> int:
    """Second-pass tolerance: 0.15 s at the nominal frame rate, rounded half up (5 at 30 fps,
    9 at 60 fps; the measured fps of a 30 fps clip varies between 29.6 and 30.04)."""
    return int(np.floor(TOLERANCE_S * round(fps) + 0.5 + 1e-9))


def match_taps(
    manual: Sequence[int], peaks: np.ndarray, left: np.ndarray, right: np.ndarray,
    flags: Sequence[str], gap_frames: set[int], first: int, last: int, tol: int = 0,
) -> list[dict[str, Any]]:  # fmt: skip
    """Pass 1: a detector tap owns [left, right]; a manual tap in a window is a hit. Several
    manual taps in one window: the one nearest the peak is TP, the rest FN. A manual tap on a
    shared trough goes to the window whose peak is nearer. Manual taps in no window are FN;
    detector taps with no manual tap are FP.
    Pass 2 (``tol`` frames, 0 = off): each left-over FN is paired one-to-one, nearest first, with
    a left-over FP whose peak is within ``tol`` frames; the pair becomes one TP_offset row."""
    owner: dict[int, int | None] = {}
    for f in manual:
        cands = [j for j in range(len(peaks)) if left[j] <= f <= right[j]]
        owner[f] = min(cands, key=lambda j: (abs(f - peaks[j]), j)) if cands else None
    rows: list[dict[str, Any]] = []

    def row(kind: str, f: int | None, j: int | None, match_pass: int = 1) -> dict[str, Any]:
        at = f if f is not None else int(peaks[j])  # type: ignore[index]
        return {
            "kind": kind,
            "match_pass": match_pass,
            "manual_frame": f,
            "detector_frame": None if j is None else int(peaks[j]),
            "offset_frames": None if f is None or j is None else f - int(peaks[j]),
            "position": position_of(at, peaks, first, last),
            "in_nan_gap": f is not None and f in gap_frames,
            "detector_flags": "" if j is None else flags[j],
        }

    for j in range(len(peaks)):
        mine = [f for f in manual if owner[f] == j]
        if not mine:
            rows.append(row("FP", None, j))
            continue
        best = min(mine, key=lambda f: (abs(f - peaks[j]), f))
        rows += [row("TP" if f == best else "FN", f, j) for f in mine]
    rows += [row("FN", f, None) for f in manual if owner[f] is None]
    if tol <= 0:
        return rows
    free_fn = [r["manual_frame"] for r in rows if r["kind"] == "FN"]
    free_fp = [r["detector_frame"] for r in rows if r["kind"] == "FP"]
    pairs = sorted((abs(f - p), f, p) for f in free_fn for p in free_fp if abs(f - p) <= tol)
    used_f: set[int] = set()
    used_p: set[int] = set()
    paired: list[tuple[int, int]] = []
    for _, f, p in pairs:
        if f not in used_f and p not in used_p:
            used_f.add(f)
            used_p.add(p)
            paired.append((f, p))
    rows = [
        r
        for r in rows
        if not (r["kind"] == "FN" and r["manual_frame"] in used_f)
        and not (r["kind"] == "FP" and r["detector_frame"] in used_p)
    ]
    index = {int(p): j for j, p in enumerate(peaks)}
    return rows + [row("TP_offset", f, index[p], 2) for f, p in sorted(paired)]


# --------------------------------------------------------------------------- analysis


@dataclass
class Analysis:
    report: dict[str, Any]
    clips: pd.DataFrame
    taps: pd.DataFrame


def clip_result(r: CountRow, inputs: Inputs) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sig = inputs.signal(r.ref)
    frames = sig["frame"].to_numpy(dtype=int)
    if (r.first, r.last) != (int(frames[0]), int(frames[-1])):
        raise Refused(f"clip {r.ref}: the export window differs from the smoothed signal")
    taps = inputs.taps(r.ref)
    summary = inputs.summary.loc[r.ref] if r.ref in inputs.summary.index else None
    if summary is None or int(summary["n_taps"]) != len(taps):
        raise Refused(f"clip {r.ref}: the taps file and taps_summary.csv disagree")
    inside = [f for f in r.frames if r.first <= f <= r.last]
    peaks, left, right, flags = detector_spans(taps, sig)
    gaps = set(frames[~np.isfinite(sig["value"].to_numpy(dtype=float))].tolist())
    tol = tolerance_frames(float(summary["fps"]))
    matches = match_taps(inside, peaks, left, right, flags, gaps, r.first, r.last, tol)
    count = {k: sum(m["kind"] == k for m in matches) for k in KINDS}
    hits = count["TP"] + count["TP_offset"]
    assert hits + count["FN"] == len(inside) and hits + count["FP"] == len(taps)
    qc = str(summary["qc_status"])
    clip = {
        "ref": r.ref, "set": r.set, "rater": r.rater, "cell": r.cell, "weight": r.weight,
        "window_first": r.first, "window_last": r.last, "n_det": len(taps), "n_man": len(inside),
        "n_outside": len(r.frames) - len(inside), "d": len(taps) - len(inside),
        "abs_d": abs(len(taps) - len(inside)), "tp": count["TP"], "tp_offset": count["TP_offset"],
        "fn": count["FN"], "fp": count["FP"], "manual_edge_partials": r.n_edge,
        "detector_edge_ends": int(summary["n_open_edge_ends"]), "qc_status": qc,
        "confidence": r.confidence,
    }  # fmt: skip
    tap_rows = [
        {"ref": r.ref, "set": r.set, "rater": r.rater, **m, "qc_status": qc, "cell": r.cell}
        for m in matches
    ]
    return clip, tap_rows


def _pair(man: np.ndarray, det: np.ndarray, boot: int, seed: int, stream: int) -> dict[str, Any]:
    out = error_metrics(man, det)
    out |= {k: v for k, v in icc2(man, det).items() if k != "n"}
    out["bootstrap_ci"] = boot_simple(man, det, boot, seed, stream)
    return out


def analyse(
    rows: Sequence[CountRow], inputs: Inputs, participants: dict[str, tuple[str, str]],
    frame: dict[str, int], seed: int, boot: int, second_kind: str, synthetic: bool,
) -> Analysis:  # fmt: skip
    problems = check_export(rows, frame)
    if problems:
        raise Refused("; ".join(problems))
    primary = [r for r in rows if r.set == "primary"]
    people = [participants.get(r.ref, ("?", ""))[0] for r in primary]
    if "?" in people or len(set(people)) != len(people):
        raise Refused("the primary clips must come from different, known participants")
    clips, taps = [], []
    for r in rows:
        c, t = clip_result(r, inputs)
        clips.append(c)
        taps += t
    clip_df = pd.DataFrame(clips, columns=list(CLIP_COLUMNS))
    tap_df = pd.DataFrame(taps, columns=list(TAP_COLUMNS))
    for col in ("manual_frame", "detector_frame", "offset_frames"):
        tap_df[col] = tap_df[col].astype("Int64")
    prim = clip_df[clip_df["set"] == "primary"].reset_index(drop=True)
    man, det = prim["n_man"].to_numpy(float), prim["n_det"].to_numpy(float)
    w, cells = prim["weight"].to_numpy(float), prim["cell"].to_numpy()
    metrics = error_metrics(man, det, w)
    ci = boot_primary(man, det, w, cells, boot, seed)
    icc = icc2(man, det)
    report: dict[str, Any] = {
        "data_kind": "SYNTHETIC" if synthetic else "real",
        "tool_version": TOOL_VERSION,
        "parameters": {
            "seed": seed, "boot": boot, "ci_level": LEVEL, "window_rule": WINDOW_RULE,
            "second_rater_kind": second_kind,
            "matching": "pass 1: detector cycle window; pass 2: TP_offset within round(0.15 * fps) frames",
            "second_pass_seconds": TOLERANCE_S,
        },
        "primary": {
            "metrics": metrics, "bland_altman": bland_altman(man, det),
            "mae_weighted_ci_bootstrap": ci["mae_weighted"], "icc": icc,
            "icc_ci_bootstrap": ci["icc"], "icc_wording": ICC_WORDING,
            "decision": decide(metrics["mae_weighted"], ci["mae_weighted"], icc["icc"], ci["icc"]),
            "n_outside": int(prim["n_outside"].sum()),
            "tap_matching": {k: int(prim[k].sum()) for k in ("tp", "tp_offset", "fn", "fp")},
            "edge_partials": {
                "manual": int(prim["manual_edge_partials"].sum()),
                "detector": int(prim["detector_edge_ends"].sum()),
            },
        },
        "per_cell": [],
    }  # fmt: skip
    for cell in mc.CELL_ORDER:
        part = prim[prim["cell"] == cell]
        m = error_metrics(
            part["n_man"].to_numpy(float),
            part["n_det"].to_numpy(float),
            part["weight"].to_numpy(float),
        )
        report["per_cell"].append(
            {"cell": cell, "n": m["n"], "mae_weighted": m["mae_weighted"], "mae": m["mae"],
             "bias": m["bias"], "bias_weighted": m["bias_weighted"]}
        )  # fmt: skip
    second = clip_df[clip_df["set"] == "second"].set_index("ref")
    if len(second):
        refs = sorted(second.index)
        r1 = prim.set_index("ref").loc[refs, "n_man"].to_numpy(float)
        r2 = second.loc[refs, "n_man"].to_numpy(float)
        dt = second.loc[refs, "n_det"].to_numpy(float)
        report["second_rater"] = {
            "n": len(refs), "second_rater_kind": second_kind, "descriptive_only": True,
            "bias_definition": "detector - manual; for r1_vs_r2: R2 - R1",
            "r1_vs_r2": _pair(r1, r2, boot, seed, 2),
            "detector_vs_r1": _pair(r1, dt, boot, seed, 3),
            "detector_vs_r2": _pair(r2, dt, boot, seed, 4),
            "detector_vs_mean_r1_r2": _pair((r1 + r2) / 2, dt, boot, seed, 5),
        }  # fmt: skip
    hard = clip_df[clip_df["set"] == "hard"].reset_index(drop=True)
    if len(hard):
        hm, hd = hard["n_man"].to_numpy(float), hard["n_det"].to_numpy(float)
        groups = [participants.get(ref, ("?", ""))[0] for ref in hard["ref"]]
        if "?" in groups:
            raise mc.InputError("hard-set clips without a known participant")
        report["hard_set"] = {
            "n": len(hard), "n_participants": len(set(groups)), "pooled_with_primary": False,
            "metrics": error_metrics(hm, hd), "icc": icc2(hm, hd),
            "bootstrap_ci_by_participant": boot_simple(hm, hd, boot, seed, 6, groups),
            "tap_matching": {k: int(hard[k].sum()) for k in ("tp", "tp_offset", "fn", "fp")},
        }  # fmt: skip
    return Analysis(report, clip_df, tap_df)


# --------------------------------------------------------------------------- outputs


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def write_json(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}{mc.TMP_MARKER}{os.getpid()}")
    try:
        tmp.write_text(json.dumps(_clean(report), indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def bland_altman_figure(prim: pd.DataFrame, title: str) -> Figure:
    fig = Figure(figsize=(7, 5), dpi=100, layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    man, det = prim["n_man"].to_numpy(float), prim["n_det"].to_numpy(float)
    ba = bland_altman(man, det)
    x, y = (man + det) / 2, det - man
    same = pd.Series(list(zip(x, y, strict=True))).value_counts()  # identical points overlap
    xs, ys = zip(*same.index, strict=True) if len(same) else ((), ())
    ax.scatter(xs, ys, s=22 + 22 * (same.to_numpy() - 1), alpha=0.6, color="tab:blue",
               edgecolors="white", linewidths=0.5, label="clips (larger = identical points)")  # fmt: skip
    ax.axhline(ba["bias"], color="0.2", lw=1, label=f"bias {ba['bias']:+.2f}")
    ax.axhline(ba["loa_lower"], color="0.4", ls="--", lw=1,
               label=f"limits of agreement {ba['loa_lower']:+.2f} to {ba['loa_upper']:+.2f}")  # fmt: skip
    ax.axhline(ba["loa_upper"], color="0.4", ls="--", lw=1)
    for ref, px, py in zip(prim["ref"], x, y, strict=True):
        if abs(py) > 2:
            ax.annotate(ref, (px, py), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("mean of manual and detector count (taps)")
    ax.set_ylabel("detector - manual (taps)")
    ax.set_title(f"{title}: agreement, primary clips (n={len(prim)})")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    return fig


def error_by_cell_figure(prim: pd.DataFrame, title: str) -> Figure:
    fig = Figure(figsize=(9, 5.5), dpi=100, layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    labels = []
    for i, cell in enumerate(mc.CELL_ORDER):
        part = prim[prim["cell"] == cell]
        d = (part["n_det"] - part["n_man"]).to_numpy(float)
        spread = np.linspace(-0.15, 0.15, max(len(d), 1))[: len(d)]
        ax.scatter(np.full(len(d), i) + spread, d, s=24, alpha=0.65, edgecolors="white")
        if len(d):
            ax.plot([i - 0.25, i + 0.25], [d.mean()] * 2, color="black", lw=2,
                    label="mean error" if i == 0 else None)  # fmt: skip
        labels.append(f"{cell}\nn={len(d)}, mean {d.mean() if len(d) else float('nan'):+.2f}")
    ax.axhline(0, color="0.35", lw=1)
    ax.set_xticks(range(len(mc.CELL_ORDER)), labels, fontsize=8)
    ax.set_xlabel("detector-rate tertile / QC flag")
    ax.set_ylabel("detector - manual (taps)")
    ax.set_title(f"{title}: error by cell, primary clips")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    return fig


def write_outputs(
    analysis: Analysis, out_dir: Path, report_path: Path, synthetic: bool
) -> list[str]:
    prefix = SYNTHETIC_PREFIX if synthetic else ""
    title = "SYNTHETIC" if synthetic else "real counts"
    prim = analysis.clips[analysis.clips["set"] == "primary"]
    mc.atomic_csv(analysis.clips, out_dir / f"{prefix}clip_errors.csv")
    mc.atomic_csv(analysis.taps, out_dir / f"{prefix}tap_matches.csv")
    save_figure(bland_altman_figure(prim, title), out_dir / f"{prefix}bland_altman.png")
    save_figure(error_by_cell_figure(prim, title), out_dir / f"{prefix}error_by_cell.png")
    write_json(analysis.report, report_path)
    return [
        f"{prefix}{n}"
        for n in ("clip_errors.csv", "tap_matches.csv", "bland_altman.png", "error_by_cell.png")
    ] + [report_path.name]


def summary_lines(report: dict[str, Any]) -> list[str]:
    p = report["primary"]
    m, d = p["metrics"], p["decision"]
    ci, icc = p["mae_weighted_ci_bootstrap"], p["icc"]
    lines = [
        f"data: {report['data_kind']}"
        + ("  (not real counts)" if report["data_kind"] == "SYNTHETIC" else ""),
        f"primary n={m['n']}: weighted MAE {m['mae_weighted']:.2f} (95% CI {ci[0]:.2f}-{ci[1]:.2f}), "
        f"unweighted {m['mae']:.2f}, bias {m['bias']:.2f}, max |d| {m['max_abs_error']:.0f}, "
        f"|d|<=1 {100 * m['share_abs_le_1']:.0f}%, |d|<=2 {100 * m['share_abs_le_2']:.0f}%",
        f"ICC(2,1) {icc['icc']:.3f} (F-based CI {icc['ci_f'][0]:.3f}-{icc['ci_f'][1]:.3f}; bootstrap "
        f"{p['icc_ci_bootstrap'][0]:.3f}-{p['icc_ci_bootstrap'][1]:.3f})",
        f"decision: {d['status']} (target_met {d['target_met']}, established {d['established']})",
        *d["statements"],
        f"tap matching: TP {p['tap_matching']['tp']}, TP_offset {p['tap_matching']['tp_offset']}, "
        f"FN {p['tap_matching']['fn']}, FP {p['tap_matching']['fp']}; "
        f"manual taps outside the window: {p['n_outside']}",
    ]
    if "second_rater" in report:
        s = report["second_rater"]
        lines.append(
            f"second rater ({s['second_rater_kind']}, n={s['n']}, descriptive): R1-R2 MAE "
            f"{s['r1_vs_r2']['mae']:.2f}, ICC {s['r1_vs_r2']['icc']:.3f}; detector vs R1 "
            f"{s['detector_vs_r1']['mae']:.2f}, vs R2 {s['detector_vs_r2']['mae']:.2f}"
        )
    if "hard_set" in report:
        h = report["hard_set"]
        lines.append(
            f"hard set (separate): n={h['n']} clips, {h['n_participants']} participants, MAE {h['metrics']['mae']:.2f}"
        )
    return lines


# --------------------------------------------------------------------------- synthetic world


def _jitter_manual(
    rng: np.random.Generator, peaks: np.ndarray, left: np.ndarray, right: np.ndarray,
    first: int, last: int, error: int,
) -> list[int]:  # fmt: skip
    """Manual frames that make detector - manual == ``error``: drop taps (error > 0) or add
    double bumps inside a tap's own window (error < 0); every frame stays inside the window."""
    n = len(peaks)
    error = min(error, n)
    keep = sorted(rng.choice(n, n - max(error, 0), replace=False)) if n else []
    frames: list[int] = []
    for j in keep:
        f = int(np.clip(peaks[j] + rng.integers(-2, 3), max(left[j], first), min(right[j], last)))
        frames.append(f)
    for _ in range(max(-error, 0)):
        for _try in range(200):
            j = int(rng.integers(0, n))
            f = int(peaks[j] + rng.choice([-4, -3, 3, 4]))
            if max(left[j], first) <= f <= min(right[j], last) and f not in frames:
                frames.append(f)
                break
        else:
            raise Refused("a synthetic clip has no room for an extra manual tap")
    if len(set(frames)) != len(frames):
        # jitter collided: move the later duplicates by one frame inside the window
        seen: set[int] = set()
        for i, f in enumerate(frames):
            while f in seen:
                f += 1
            frames[i] = f
            seen.add(f)
    return sorted(frames)


def build_synthetic_export(
    selection: pd.DataFrame, inputs: Inputs, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Manual counts built from the real detector taps with injected, known errors."""
    rng = np.random.default_rng([seed, 99])
    errors = np.array([-2, -1, 0, 0, 0, 1, 1, 2])
    info = selection.set_index("ref")
    plan: list[tuple[str, str, str]] = []  # (set, rater, ref)
    plan += [
        ("primary", "R1", r) for r in sorted(selection.loc[selection["set"] == "primary", "ref"])
    ]
    second = selection[(selection["set"] == "primary") & (selection["second_rater"] == "1")]
    plan += [("second", "R2", r) for r in sorted(second["ref"])]
    plan += [("hard", "R2", r) for r in sorted(selection.loc[selection["set"] == "hard", "ref"])]
    r1_frames: dict[str, list[int]] = {}
    out, truth = [], []
    for kind, rater, ref in plan:
        sig, taps = inputs.signal(ref), inputs.taps(ref)
        frames = sig["frame"].to_numpy(dtype=int)
        first, last = int(frames[0]), int(frames[-1])
        peaks, left, right, _ = detector_spans(taps, sig)
        if kind == "second":  # R2 starts from R1's frames, plus one occasional disagreement
            base = list(r1_frames[ref])
            if rng.random() < 0.3 and len(base) > 1:
                base.pop(int(rng.integers(0, len(base))))
            man = sorted(base)
        else:
            man = _jitter_manual(rng, peaks, left, right, first, last, int(rng.choice(errors)))
        if kind == "primary":
            r1_frames[ref] = man
        truth.append({"ref": ref, "set": kind, "injected_error": len(peaks) - len(man)})
        out.append(
            {
                "ref": ref, "rater": rater, "set": kind, "n_taps": len(man),
                "tap_frames": ";".join(map(str, man)),
                "tap_times_s": ";".join(f"{f / 30:.3f}" for f in man),
                "n_edge_partials": int(rng.integers(0, 2)), "confidence": 3, "note": "",
                "cell": info.at[ref, "cell"], "weight": info.at[ref, "weight"] if kind == "primary" else "",
                "window_first": first, "window_last": last,
            }
        )  # fmt: skip
    table = pd.DataFrame(out)[list(mc.EXPORT_COLUMNS)].astype(str)
    return table, pd.DataFrame(truth)


# --------------------------------------------------------------------------- command line


def run_report(args: argparse.Namespace, out_dir: Path, docs_dir: Path) -> int:
    interim = Path(mc.INTERIM_ROOT)
    inputs = load_inputs(interim)
    synthetic = bool(args.synthetic)
    truth: pd.DataFrame | None = None
    if synthetic:
        selection = mc.read_selection(interim / "manual_counts")
        export, truth = build_synthetic_export(selection, inputs, args.seed)
    else:
        export = read_export(docs_dir / mc.EXPORT_NAME)
    participants = mc.participant_keys(Path(mc.VIDEO_ROOT))
    frame = expected_frame(Path(mc.DOCS_ROOT), interim / "taps" / "taps_summary.csv")
    analysis = analyse(
        parse_rows(export),
        inputs,
        participants,
        frame,
        args.seed,
        args.boot,
        args.second_rater_kind,
        synthetic,
    )
    if truth is not None:
        got = analysis.clips.set_index(["set", "ref"])["d"]
        want = truth.set_index(["set", "ref"])["injected_error"]
        if not (got.sort_index() == want.sort_index()).all():
            raise Refused("the synthetic errors were not recovered")
        analysis.report["synthetic_truth_check"] = f"passed for {len(want)} clips"
        mc.atomic_csv(truth, out_dir / f"{SYNTHETIC_PREFIX}truth.csv")
    report_path = (
        out_dir / f"{SYNTHETIC_PREFIX}{REPORT_NAME}" if synthetic else docs_dir / REPORT_NAME
    )
    names = write_outputs(analysis, out_dir, report_path, synthetic)
    print("\n".join(summary_lines(analysis.report)))
    print("wrote: " + ", ".join(names))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)
    rep = sub.add_parser("report", help="MAE, ICC and the tap matching from manual_counts.csv")
    rep.add_argument(
        "--synthetic", action="store_true", help="synthetic counts; never writes docs/"
    )
    rep.add_argument("--out-dir", type=Path, default=None)
    rep.add_argument("--docs-dir", type=Path, default=None)
    rep.add_argument("--seed", type=int, default=DEFAULT_SEED)
    rep.add_argument("--boot", type=int, default=DEFAULT_BOOT)
    rep.add_argument("--second-rater-kind", choices=SECOND_KINDS, default="inter_rater")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.boot < 1:
        print("error: --boot must be at least 1", file=sys.stderr)
        return 2
    interim, docs = Path(mc.INTERIM_ROOT), Path(mc.DOCS_ROOT)
    out_dir = Path(args.out_dir or interim / (SYNTHETIC_DIR if args.synthetic else VALIDATION_DIR))
    docs_dir = Path(args.docs_dir or docs)
    if not is_inside(out_dir, interim):
        print("error: --out-dir must be inside data/interim", file=sys.stderr)
        return 2
    if not is_inside(docs_dir, docs):
        print("error: --docs-dir must be inside models/tapping/docs", file=sys.stderr)
        return 2
    try:
        return run_report(args, out_dir, docs_dir)
    except Refused as err:
        print(f"refused: {err}", file=sys.stderr)
        return 1
    except mc.InputError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except OSError as err:
        name = Path(err.filename).name if isinstance(err.filename, str) else "a file"
        print(f"error: could not read/write {name}: {mc._reason(err)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
