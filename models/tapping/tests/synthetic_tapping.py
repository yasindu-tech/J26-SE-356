"""Synthetic thumb-index signals and the real chain, for test_feature_synthetic.py (task FT-29).

Pure and seeded, numpy only for the signal. ``gen`` makes raised-cosine taps on a floor:
cycle k starts at s_k, y = floor + A_k (0.5 - 0.5 cos(2 pi (t - s_k) / P_k)), at rest on the floor
between cycles, peak at s_k + P_k / 2, maximum slope A_k pi / P_k. ``pauses[k]`` is extra idle time
before tap k. Samples are on the grid i / fps plus uniform jitter; noise goes on the raw values.
``period * fps / 2`` must be a whole number (a peak on a sample), or two samples tie at the top and
the intervals get an irregular 1-frame wobble (iti_cv about 0.02): pass allow_ties=True to waive.
``chain`` runs the shipped code: smoothing -> detector -> tap table -> Set A -> Set B.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import features_a  # noqa: E402
import features_b  # noqa: E402
import smoothing  # noqa: E402
import tap_table  # noqa: E402
import taps  # noqa: E402

REF = "aaaa1111"


def gen(
    rng: np.random.Generator,
    n: int = 29,
    period: float = 2 / 3,
    amps: Any = None,
    periods: Any = None,
    fps: float = 30.0,
    noise: float = 0.0,
    jitter: float = 0.0,
    pauses: dict[int, float] | None = None,
    floor: float = 0.1,
    lead: float = 0.5,
    tail: float = 0.5,
    nan_runs: tuple[tuple[float, float], ...] = (),
    allow_ties: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """(t, raw, truth). ``nan_runs`` are (start_s, end_s) spans set to NaN; ``jitter`` is in s."""
    a = np.full(n, 1.2) if amps is None else np.asarray(amps, float)
    p = np.full(n, period) if periods is None else np.asarray(periods, float)
    if not allow_ties:
        half = p * fps / 2
        assert np.allclose(half, np.round(half), atol=1e-9), "period * fps / 2 must be whole"
    idle = np.zeros(n)
    for k, extra in (pauses or {}).items():
        idle[k] = extra
    start = np.empty(n)
    s = lead
    for k in range(n):
        s += idle[k]
        start[k] = s
        s += p[k]
    peak_t = start + p / 2
    grid = np.arange(int(np.floor((s + tail) * fps)) + 1) / fps
    t = grid + (rng.uniform(-jitter, jitter, grid.size) if jitter else 0.0)
    t[0] = max(t[0], 0.0)
    y = np.full(t.size, floor)
    for k in range(n):
        m = (t >= start[k]) & (t < start[k] + p[k])
        y[m] = floor + a[k] * (0.5 - 0.5 * np.cos(2 * np.pi * (t[m] - start[k]) / p[k]))
    raw = y + (rng.normal(0, noise, y.size) if noise else 0.0)
    for lo, hi in nan_runs:
        raw[(t >= lo) & (t < hi)] = np.nan
    return t, raw, {"peak_t": peak_t, "amps": a, "periods": p, "fps": fps}


def crop(t: np.ndarray, raw: np.ndarray, start_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Drop the samples before ``start_s`` and restart the clock at 0."""
    i = int(np.searchsorted(t, start_s))
    return t[i:] - t[i], raw[i:]


def chain(
    t: np.ndarray, raw: np.ndarray, fps: float, ref: str = REF
) -> tuple[taps.TapResult, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """The shipped code on one synthetic clip: (taps result, tap table, Set A row, Set B row)."""
    value, _, _ = smoothing.smooth_series(t, raw)
    result = taps.extract_taps(
        ref, t, value, raw, np.zeros(t.size, bool), np.arange(t.size), taps.PARAMS
    )
    table = tap_table.build_tap_table(result.table)
    summary = pd.Series(
        {
            "duration_s": result.stats["duration_s"],
            "fps": result.stats["fps"],
            "qc_status": "retained",
        }
    )
    row_a = features_a.clip_features(table, ref, summary)
    row_b = features_b.clip_row(table, ref, "retained")
    return result, table, row_a, row_b
