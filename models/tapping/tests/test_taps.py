"""Tests for models/tapping/src/taps.py (task FT-23).

Synthetic data only: raised-cosine tap trains with known peaks, troughs, amplitudes and times,
under a temporary ``data/interim`` and docs folder (``INTERIM_ROOT`` and ``DOCS_ROOT`` are
patched). No conftest.py on purpose; ``src`` goes on ``sys.path`` here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import smoothing as sm  # noqa: E402
import taps as tp  # noqa: E402

REF = "a1b2c3d4"
FPS = 30.0
T0 = 100.0  # absolute pts of the first sample: times must come out relative to the clip start


# --------------------------------------------------------------------------- synthetic signals


def cycles(
    periods: list[int], amps: list[float], floor: float = 0.1, lead: int = 3, tail: int = 3
) -> tuple[np.ndarray, list[int], list[int]]:
    """Raised-cosine taps. Cycle k has ``periods[k]`` samples and amplitude ``amps[k]``; its peak
    is in the middle and its troughs (value ``floor``) at both ends. ``lead`` and ``tail``
    samples continue the waveform, so no trough is the first or last sample.
    Returns the values, the peak indices and the trough indices."""
    parts: list[float] = []
    peaks: list[int] = []
    troughs: list[int] = [lead]
    start = lead
    for period, amp in zip(periods, amps, strict=True):
        parts += [floor + amp * (0.5 - 0.5 * np.cos(2 * np.pi * i / period)) for i in range(period)]
        peaks.append(start + period // 2)
        start += period
        troughs.append(start)
    parts.append(floor)
    first, last = periods[0], periods[-1]
    lead_v = [
        floor + amps[0] * (0.5 - 0.5 * np.cos(2 * np.pi * j / first)) for j in range(lead, 0, -1)
    ]
    tail_v = [
        floor + amps[-1] * (0.5 - 0.5 * np.cos(2 * np.pi * j / last)) for j in range(1, tail + 1)
    ]
    return np.array(lead_v + parts + tail_v), peaks, troughs


def train(
    n: int = 10, period: int = 16, amp: float = 1.2, floor: float = 0.1
) -> tuple[np.ndarray, list[int], list[int]]:
    return cycles([period] * n, [amp] * n, floor)


def times(n: int, fps: float = FPS) -> np.ndarray:
    return T0 + np.arange(n) / fps


def run(
    v: np.ndarray,
    *,
    t: np.ndarray | None = None,
    raw: np.ndarray | None = None,
    filled: np.ndarray | None = None,
    fps: float = FPS,
) -> tp.TapResult:
    n = len(v)
    return tp.extract_taps(
        REF,
        times(n, fps) if t is None else t,
        v,
        v if raw is None else raw,
        np.zeros(n, dtype=bool) if filled is None else filled,
        np.arange(n),
    )


def flags_of(result: tp.TapResult) -> list[set[str]]:
    return [set(f.split(";")) - {""} for f in result.table["flags"]]


# --------------------------------------------------------------------------- frozen parameters


def test_the_frozen_parameters_are_the_study_values() -> None:
    p = tp.TapParams()
    assert p.prominence_frac == 0.20 and p.percentiles == (5.0, 95.0)
    assert p.min_spacing_s == 0.15 and p.min_segment_samples == 3 and p.min_finite_samples == 10
    assert p.gap_adjacent_s == 0.2 and p.narrow_s == 0.10
    assert (p.raw_ratio_low, p.raw_ratio_high) == (0.90, 1.10)
    assert p.long_interval_factor == 1.8 and p.long_interval_min_taps == 4
    assert p.open_edge_level == 0.5 and p.low_range == 0.6
    assert p == tp.PARAMS
    with pytest.raises(AttributeError):
        p.prominence_frac = 0.25  # type: ignore[misc]  # frozen: assigning must fail


# --------------------------------------------------------------------------- the tap definition


def test_known_train_gives_exact_counts_times_and_amplitudes() -> None:
    v, peaks, troughs = train(10)
    r = run(v)
    tab = r.table
    assert len(tab) == 10 and r.status == tp.STATUS_OK
    np.testing.assert_allclose(tab["amplitude"], 1.2, atol=1e-12)
    np.testing.assert_allclose(
        tab["t_peak_s"], np.array(peaks) / FPS, atol=1e-12
    )  # from the clip start
    np.testing.assert_array_equal(tab["frame"], peaks)
    np.testing.assert_array_equal(tab["tap_index"], np.arange(10))
    assert np.isnan(tab["interval_s"].iloc[0])
    np.testing.assert_allclose(tab["interval_s"].iloc[1:], 16 / FPS, atol=1e-12)
    np.testing.assert_allclose(tab["rise_s"], 8 / FPS, atol=1e-12)
    np.testing.assert_allclose(tab["fall_s"], 8 / FPS, atol=1e-12)
    np.testing.assert_allclose(tab["width_half_s"], 8 / FPS, atol=1e-9)
    np.testing.assert_allclose(tab["trough_left_s"], np.array(troughs[:-1]) / FPS, atol=1e-12)
    np.testing.assert_allclose(tab["trough_right_s"], np.array(troughs[1:]) / FPS, atol=1e-12)
    np.testing.assert_allclose(tab["value_peak"], 1.3, atol=1e-12)
    assert (tab["flags"] == "").all()
    assert r.stats["n_taps"] == 10 and r.stats["n_possible_hidden"] == 0


def test_velocities_follow_the_slope_of_the_signal() -> None:
    v, _, _ = train(8, amp=1.0)
    tab = run(v).table
    expected = 0.5 * FPS * np.sin(2 * np.pi / 16) * 1.0  # steepest central difference of the cosine
    np.testing.assert_allclose(tab["open_vel"], expected, rtol=1e-9)
    np.testing.assert_allclose(tab["close_vel"], expected, rtol=1e-9)
    np.testing.assert_allclose(tab["open_vel_mean"], 1.0 / (8 / FPS), rtol=1e-9)
    np.testing.assert_allclose(tab["close_vel_mean"], 1.0 / (8 / FPS), rtol=1e-9)
    assert (tab["open_vel"] > tab["open_vel_mean"]).all()  # the max slope exceeds the mean slope


@pytest.mark.parametrize(("fps", "period"), [(60.0, 32), (20.0, 16), (30.0, 16)])
def test_velocity_uses_the_pts_time_not_the_index(fps: float, period: int) -> None:
    v, _, _ = cycles([period] * 8, [1.0] * 8, lead=period // 5, tail=period // 5)
    tab = run(v, fps=fps).table
    assert len(tab) == 8
    expected = 0.5 * fps * np.sin(2 * np.pi / period)  # steepest central difference, per second
    np.testing.assert_allclose(tab["open_vel"], expected, rtol=1e-9)
    np.testing.assert_allclose(tab["close_vel"], expected, rtol=1e-9)
    rng = np.random.default_rng(5)  # irregular times: the slope still follows the real time axis
    t = T0 + np.cumsum(np.concatenate(([0.0], 1 / FPS + rng.uniform(-0.008, 0.008, len(v) - 1))))
    irr = run(v, t=t).table
    first = irr.iloc[0]
    seg = slice(int(first["frame"]) - period // 2, int(first["frame"]) + 1)
    assert first["open_vel"] == pytest.approx(np.max(np.gradient(v, t)[seg]), rel=1e-9)


def test_the_spacing_rule_applies_within_a_segment_only() -> None:
    # two peaks 4 samples apart (0.067 s at 60 fps) on opposite sides of a NaN: both are taps
    tail = np.array([0.1, 0.4, 1.3, 0.4, np.nan, 0.4, 1.3, 0.4, 0.1])
    v = np.concatenate([train(4)[0], tail, [0.1] * 3])
    r = run(v, fps=60.0)
    assert len(r.table) == 4 + 2
    assert abs(r.table["interval_s"].iloc[-1] - 4 / 60.0) < 1e-9
    assert "gap_in_interval" in flags_of(r)[-1]


def test_amplitude_is_the_peak_minus_the_mean_of_both_troughs() -> None:
    v, _, _ = train(10)
    v = v + np.linspace(0.0, 0.6, len(v))  # a drift: the two troughs of a tap differ
    tab = run(v).table
    assert (tab["value_trough_left"] < tab["value_trough_right"]).all()
    expected = tab["value_peak"] - 0.5 * (tab["value_trough_left"] + tab["value_trough_right"])
    np.testing.assert_allclose(tab["amplitude"], expected, atol=1e-12)
    one_trough = tab["value_peak"] - tab["value_trough_left"]
    assert not np.allclose(tab["amplitude"], one_trough, atol=1e-3)


def test_the_same_train_with_noise_keeps_count_and_roughly_amplitudes() -> None:
    v, peaks, _ = train(12)
    rng = np.random.default_rng(1)
    noisy = v + rng.normal(0.0, 0.02 * 1.2, len(v))
    tab = run(noisy).table
    assert len(tab) == 12
    assert np.abs(tab["frame"].to_numpy() - np.array(peaks)).max() <= 1
    assert np.abs(tab["amplitude"] - 1.2).max() < 0.15


@pytest.mark.parametrize("scale", [0.3, 2.0, 0.05])
def test_scale_invariance_same_times_and_count(scale: float) -> None:
    v, _, _ = train(10)
    base, scaled = run(v).table, run(v * scale).table
    assert len(base) == len(scaled) == 10
    np.testing.assert_allclose(scaled["t_peak_s"], base["t_peak_s"], atol=1e-12)
    np.testing.assert_allclose(scaled["amplitude"], base["amplitude"] * scale, atol=1e-12)


def test_30_and_60_fps_give_the_same_count_and_times() -> None:
    v30, _, _ = train(10, period=16)
    v60, _, _ = cycles([32] * 10, [1.2] * 10, lead=6, tail=6)
    a, b = run(v30, fps=30.0).table, run(v60, fps=60.0).table
    assert len(a) == len(b) == 10
    assert np.abs(a["t_peak_s"].to_numpy() - b["t_peak_s"].to_numpy()).max() <= 1 / 30.0
    np.testing.assert_allclose(a["amplitude"], b["amplitude"], atol=1e-9)
    assert run(v60, fps=60.0).stats["fps"] == pytest.approx(60.0)


def test_irregular_timestamps_are_used_not_the_frame_index() -> None:
    v, _, _ = train(10)
    rng = np.random.default_rng(3)
    t = T0 + np.cumsum(np.concatenate(([0.0], 1 / FPS + rng.uniform(-0.008, 0.008, len(v) - 1))))
    tab = run(v, t=t).table
    assert len(tab) == 10
    np.testing.assert_allclose(tab["t_peak_s"], t[tab["frame"].to_numpy()] - t[0], atol=1e-12)
    assert not np.allclose(tab["t_peak_s"], tab["frame"] / FPS, atol=1e-4)
    slow = T0 + np.arange(len(v)) * 0.05  # 20 fps: index / 30 would be wrong
    tab20 = run(v, t=slow).table
    np.testing.assert_allclose(tab20["t_peak_s"], tab20["frame"] * 0.05, atol=1e-12)


def test_spacing_is_in_seconds_not_frames() -> None:
    fast30, _, _ = cycles([6] * 14, [1.0] * 14, lead=2, tail=2)
    assert len(run(fast30, fps=30.0).table) == 14  # 0.2 s apart: all kept at 30 fps
    fast60 = fast30  # the same samples at 60 fps are 0.1 s apart: closer than 0.15 s
    tab = run(fast60, fps=60.0).table
    assert 0 < len(tab) < 14
    assert tab["interval_s"].dropna().min() >= 0.15 - 1e-9


def test_amplitude_cut_is_relative_to_the_clip_range() -> None:
    rng = tp.clip_range(cycles([16] * 7, [1.0] * 7)[0])
    assert rng is not None
    cut_at = 0.20 * rng[2]  # the frozen prominence in hand sizes for this clip

    def count(small: float) -> int:
        v, _, _ = cycles([16] * 7, [1.0, 1.0, 1.0, small, 1.0, 1.0, 1.0])
        return len(run(v).table)

    assert count(0.9 * cut_at) == 6  # below 0.20 x range: not a tap
    assert count(1.1 * cut_at) == 7  # above it: a tap
    for scale in (0.05, 20.0):  # the same rule at any signal size
        v, _, _ = cycles([16] * 7, [1.0, 1.0, 1.0, 0.9 * cut_at, 1.0, 1.0, 1.0])
        assert len(run(v * scale).table) == 6


def test_narrow_taps_are_flagged_but_never_removed() -> None:
    v, _, _ = cycles([10] * 9, [1.0] * 9, lead=2, tail=2)
    r = run(v, fps=60.0)
    assert len(r.table) == 9
    assert all("narrow" in f for f in flags_of(r))
    assert (r.table["width_half_s"] < 0.10).all() and r.stats["n_narrow"] == 9
    assert not any("narrow" in f for f in flags_of(run(train(9)[0])))


# --------------------------------------------------------------------------- edges, gaps, filled


def test_a_first_and_a_last_tap_at_the_clip_edge_are_one_sided() -> None:
    v, _, _ = cycles([16] * 6, [1.2] * 6, lead=0, tail=0)  # starts and ends exactly on a trough
    r = run(v)
    f = flags_of(r)
    assert len(r.table) == 6
    assert "one_sided" in f[0] and "one_sided" in f[-1]
    assert not any("one_sided" in x for x in f[1:-1])
    np.testing.assert_allclose(
        r.table["amplitude"], 1.2, atol=1e-12
    )  # the other trough is the floor
    assert r.stats["n_one_sided"] == 2


def test_an_edge_trough_is_dropped_not_averaged() -> None:
    # an isolated segment [0.1 (edge), ..., 1.3, ..., 0.2 (inner trough), 0.4] after a normal train
    seg = np.array([0.1, 0.5, 1.3, 0.6, 0.2, 0.4])
    v = np.concatenate([train(4)[0], [np.nan] * 4, seg, [np.nan] * 3])
    r = run(v)
    last = r.table.iloc[-1]
    assert "one_sided" in flags_of(r)[-1]
    assert last["value_trough_left"] == 0.1 and last["value_trough_right"] == 0.2
    assert last["amplitude"] == pytest.approx(1.3 - 0.2)  # only the inner trough, not the mean 0.15


def test_both_troughs_on_edges_use_the_lower_one() -> None:
    seg = np.array([0.1, 0.5, 1.3, 0.5, 0.2])  # first sample and last sample are the troughs
    v = np.concatenate([train(4)[0], [np.nan] * 4, seg, [np.nan] * 3])
    r = run(v)
    last = r.table.iloc[-1]
    assert "one_sided" in flags_of(r)[-1]
    assert last["value_trough_left"] == 0.1 and last["value_trough_right"] == 0.2
    assert last["amplitude"] == pytest.approx(1.3 - 0.1)  # the LOWER of the two edge troughs


def test_a_nan_gap_in_the_train_splits_the_segments_and_interpolates_nothing() -> None:
    v, peaks, troughs = train(10)
    gap = slice(troughs[5] - 3, troughs[5] + 4)  # around the trough between taps 4 and 5
    v = v.copy()
    v[gap] = np.nan
    r = run(v)
    tab = r.table
    assert len(tab) == 10  # taps on both sides of the gap
    lo, hi = (troughs[5] - 3) / FPS, (troughs[5] + 3) / FPS
    for col in ("t_peak_s", "trough_left_s", "trough_right_s"):
        inside = (tab[col] >= lo - 1e-9) & (tab[col] <= hi + 1e-9)
        assert not inside.any()  # nothing was measured inside the gap
    f = flags_of(r)
    assert "one_sided" in f[4] and "one_sided" in f[5]
    assert "gap_adjacent" in f[4] and "gap_adjacent" in f[5]
    assert "gap_in_interval" in f[5] and "gap_in_interval" not in f[4]
    assert not any("long_interval" in x or "possible_hidden_tap_before" in x for x in f)
    np.testing.assert_allclose(tab["amplitude"], 1.2, atol=1e-12)  # the other trough is the floor
    assert r.stats["n_nan_runs"] == 1 and r.stats["n_gap_adjacent"] == 2


def test_a_missing_tap_under_a_gap_is_a_possible_hidden_tap() -> None:
    v, peaks, _ = train(10)
    v = v.copy()
    v[peaks[5] - 9 : peaks[5] + 10] = np.nan  # the whole 6th tap is hidden
    r = run(v)
    f = flags_of(r)
    assert len(r.table) == 9
    hidden = [i for i, x in enumerate(f) if "possible_hidden_tap_before" in x]
    assert hidden == [5]
    assert {"gap_in_interval", "long_interval"} <= f[5]
    assert r.table["interval_s"].iloc[5] == pytest.approx(32 / FPS)
    assert r.stats["n_possible_hidden"] == 1


def test_a_long_interval_without_a_gap_is_only_long_interval() -> None:
    v, _, _ = cycles([16] * 5 + [34, 34] + [16] * 5, [1.0] * 12)
    f = flags_of(run(v))
    long_idx = [i for i, x in enumerate(f) if "long_interval" in x]
    assert long_idx == [6]  # the 34-sample interval (2.1x the median)
    assert not any("gap_in_interval" in x or "possible_hidden_tap_before" in x for x in f)


def test_the_long_interval_factor_is_1_8() -> None:
    v, _, _ = cycles([16] * 5 + [26, 26] + [16] * 5, [1.0] * 12)  # a 26-sample interval: 1.6x
    assert not any("long_interval" in x for x in flags_of(run(v)))


def test_long_interval_needs_four_taps() -> None:
    v, _, _ = cycles([16, 16, 40], [1.0, 1.0, 1.0])
    r = run(v)
    assert len(r.table) == 3 and not any("long_interval" in x for x in flags_of(r))


def test_filled_samples_next_to_a_peak_are_flagged() -> None:
    v, peaks, _ = train(8)
    filled = np.zeros(len(v), dtype=bool)
    filled[peaks[2] + 1] = True  # within +-1 of tap 2
    filled[peaks[5] + 2] = True  # two samples away: not flagged
    f = flags_of(run(v, filled=filled))
    assert "on_filled" in f[2] and "on_filled" not in f[5]


def test_a_high_end_cut_by_the_clip_edge_is_not_a_tap_but_is_counted() -> None:
    v, _, _ = train(8)
    start = v[3 + 8 :]  # begins at the first peak and goes down: an open phase cut by the edge
    r = run(start)
    assert r.stats["n_open_edge_ends"] >= 1
    assert len(r.table) == 7 and r.table["t_peak_s"].min() > 0.0  # the cut phase is no tap row
    end = v[: 3 + 16 * 7 + 8]  # ends at a peak, rising
    r2 = run(end)
    assert r2.stats["n_open_edge_ends"] >= 1 and len(r2.table) == 7
    assert run(v).stats["n_open_edge_ends"] == 0  # a clip that starts and ends low


def test_open_edge_ends_counts_segment_ends_next_to_a_gap() -> None:
    v, peaks, _ = train(8)
    v = v.copy()
    v[peaks[3] + 1 : peaks[3] + 5] = np.nan  # a gap right after the top of tap 3
    r = run(v)
    assert r.stats["n_open_edge_ends"] == 1  # the open phase cut by the gap
    assert len(r.table) == 7  # that tap is not a row (its top is the segment's last sample)


def test_clip_statistics() -> None:
    v, _, _ = train(10)
    v = v.copy()
    v[80:87] = np.nan  # around the trough at 83: no peak is hidden
    s = run(v).stats
    assert s["n_nan_runs"] == 1 and s["longest_nan_s"] == pytest.approx(8 / FPS)
    assert s["fps"] == pytest.approx(FPS) and s["duration_s"] == pytest.approx((len(v) - 1) / FPS)
    assert s["tap_rate_hz"] == pytest.approx(s["n_taps"] / s["duration_s"])
    assert s["median_interval_s"] == pytest.approx(16 / FPS)
    assert s["interval_cv"] == pytest.approx(0.0, abs=1e-9)
    assert s["prominence_threshold"] == pytest.approx(0.20 * s["range"])
    assert s["low_range"] is False
    assert run(train(10, amp=0.4)[0]).stats["low_range"] is True  # range below 0.6


# --------------------------------------------------------------------------- degenerate signals


@pytest.mark.parametrize(
    ("name", "values", "status"),
    [
        ("ramp", np.linspace(0.0, 1.0, 120), tp.STATUS_NO_TAPS),
        ("constant", np.full(120, 0.7), tp.STATUS_NO_TAPS),
        ("all nan", np.full(60, np.nan), tp.STATUS_FEW_SAMPLES),
        ("five samples", np.array([0.1, 0.9, 0.2, 0.8, 0.1]), tp.STATUS_FEW_SAMPLES),
        (
            "nine finite",
            np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, np.nan]),
            tp.STATUS_FEW_SAMPLES,
        ),
    ],
)
def test_signals_without_taps_give_an_empty_table_not_a_crash(
    name: str, values: np.ndarray, status: str
) -> None:
    r = run(values)
    assert len(r.table) == 0 and r.status == status, name
    assert list(r.table.columns) == list(tp.TAP_COLUMNS)
    assert r.stats["n_taps"] == 0
    if status == tp.STATUS_FEW_SAMPLES:
        assert np.isnan(r.stats["range"]) and not r.stats["low_range"]


def test_a_zero_length_clip_does_not_crash() -> None:
    r = tp.extract_taps(
        REF, np.empty(0), np.empty(0), np.empty(0), np.empty(0, dtype=bool), np.empty(0, dtype=int)
    )
    assert len(r.table) == 0 and r.stats["n_taps"] == 0


def test_a_segment_shorter_than_three_samples_has_no_taps() -> None:
    v, _, _ = train(6)
    cut_v = np.concatenate([v[:40], [np.nan, 5.0, 0.1, np.nan], v[44:]])  # a 2-sample segment
    n = len(cut_v)
    r = tp.extract_taps(REF, times(n), cut_v, cut_v, np.zeros(n, dtype=bool), np.arange(n))
    assert len(r.table) > 0 and (r.table["value_peak"] < 3.0).all()  # the 5.0 gave no tap


def test_a_non_increasing_time_axis_is_refused_by_hashed_reference() -> None:
    v, _, _ = train(6)
    t = times(len(v))
    t[10] = t[9]
    with pytest.raises(tp.TapError, match=REF):
        run(v, t=t)


# --------------------------------------------------------------------------- amplitude on raw


def test_amplitude_raw_uses_value_raw_and_disagreement_is_flagged() -> None:
    v, _, _ = train(8, floor=0.0)
    same = run(v)
    assert (same.table["amplitude_raw"] == same.table["amplitude"]).all()
    assert not any("raw_disagrees" in x for x in flags_of(same))
    near = run(v, raw=v * 1.05)  # inside 0.90..1.10
    np.testing.assert_allclose(
        near.table["amplitude_raw"], near.table["amplitude"] * 1.05, atol=1e-12
    )
    assert not any("raw_disagrees" in x for x in flags_of(near))
    high = run(v, raw=v * 1.3)
    assert all("raw_disagrees" in x for x in flags_of(high))
    low = run(v, raw=v * 0.8)
    assert all("raw_disagrees" in x for x in flags_of(low))
    assert (low.table["flags"].str.contains("raw_disagrees")).all() and len(low.table) == 8


def test_raw_peak_is_the_max_within_one_sample_and_troughs_the_min() -> None:
    v, peaks, troughs = train(6, floor=0.0)
    p, tl = peaks[2], troughs[2]
    base = run(v).table.iloc[2]
    at = v.copy()
    at[p] += 0.3  # a raw spike on the peak sample
    at[tl] -= 0.05  # and a deeper raw trough on the left trough sample
    tab = run(v, raw=at).table
    assert tab["amplitude_raw"].iloc[2] == pytest.approx(base["amplitude"] + 0.3 + 0.025, abs=1e-9)
    near = v.copy()
    near[p + 1] += 0.3  # one sample away: taken, as the max of the +-1 window
    got = run(v, raw=near).table["amplitude_raw"].iloc[2]
    assert got == pytest.approx(base["amplitude"] + (v[p + 1] + 0.3 - v[p]), abs=1e-9)
    far = v.copy()
    far[p + 2] += 0.3  # two samples away: not taken
    assert run(v, raw=far).table["amplitude_raw"].iloc[2] == pytest.approx(base["amplitude"])


# --------------------------------------------------------------------------- files and CLI


def clip_frame(v: np.ndarray, fps: float = FPS, raw: np.ndarray | None = None) -> pd.DataFrame:
    n = len(v)
    return pd.DataFrame(
        {
            "frame": np.arange(n, dtype=np.int64),
            "t_s": T0 + np.arange(n) / fps,
            "value_raw": v if raw is None else raw,
            "value": v,
            "filled": np.zeros(n, dtype=bool),
            "smoothed": np.isfinite(v),
        }
    )


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    interim = tmp_path / "interim"
    (interim / "smoothed").mkdir(parents=True)
    docs = tmp_path / "docs"
    docs.mkdir()
    monkeypatch.setattr(tp, "INTERIM_ROOT", interim)
    monkeypatch.setattr(tp, "DOCS_ROOT", docs)
    return {
        "interim": interim,
        "smoothed": interim / "smoothed",
        "docs": docs,
        "out": interim / "taps",
    }


def write_clip(world: dict[str, Path], ref: str, v: np.ndarray | None = None) -> Path:
    values = train(8)[0] if v is None else v
    path = world["smoothed"] / f"{ref}.parquet"
    sm.write_smoothed(sm.Smoothed(ref, clip_frame(values), {"clip_ref": ref}), path, "x")
    return path


def make_clips(world: dict[str, Path], refs: list[str]) -> None:
    for r in refs:
        write_clip(world, r)


def write_qc(
    world: dict[str, Path], retained: list[tuple[str, str, str]], rejected: list[str]
) -> None:
    pd.DataFrame(retained, columns=["ref", "verdict", "note"]).to_csv(
        world["docs"] / tp.RETAINED_NAME, index=False
    )
    pd.DataFrame(
        {"ref": rejected, "reason": ["other"] * len(rejected), "note": [""] * len(rejected)}
    ).to_csv(world["docs"] / tp.REJECTIONS_NAME, index=False)


REFS = ["0000000a", "0000000b", "0000000c", "0000000d"]


def test_all_run_writes_parquets_and_a_summary_with_exact_columns(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    make_clips(world, REFS)
    assert tp.main(["--all"]) == 0
    out = capsys.readouterr().out
    assert "qc_retained.csv not found: processing ALL clips (QC not frozen)" in out
    assert sorted(p.name for p in world["out"].iterdir()) == sorted(
        [f"{r}.parquet" for r in REFS] + [tp.SUMMARY_NAME]
    )
    summary = pd.read_csv(world["out"] / tp.SUMMARY_NAME, dtype={"ref": str}, keep_default_na=False)
    assert list(summary.columns) == [
        "ref", "fps", "duration_s", "n_taps", "tap_rate_hz", "median_interval_s", "interval_cv",
        "range", "floor_p5", "prominence_threshold", "n_one_sided", "n_gap_adjacent", "n_narrow",
        "n_possible_hidden", "n_open_edge_ends", "n_nan_runs", "longest_nan_s", "qc_status",
        "qc_flag", "low_range",
    ]  # fmt: skip
    assert summary["ref"].tolist() == REFS and (summary["n_taps"] == 8).all()
    assert (summary["qc_status"] == "unfrozen").all() and (summary["qc_flag"] == "").all()
    assert summary["fps"].astype(float).round(3).eq(30.0).all()
    assert summary["low_range"].astype(str).str.lower().eq("false").all()
    tab = pd.read_parquet(world["out"] / f"{REFS[0]}.parquet")
    assert list(tab.columns) == list(tp.TAP_COLUMNS) and (tab["ref"] == REFS[0]).all()


def test_parquet_metadata_records_parameters_source_and_qc(world: dict[str, Path]) -> None:
    path = write_clip(world, REFS[0])
    assert tp.main(["--all"]) == 0
    _, meta = tp.read_taps(world["out"] / f"{REFS[0]}.parquet")
    assert meta["tool"] == tp.TOOL_VERSION and meta["status"] == "ok"
    assert meta["params"]["prominence_frac"] == 0.20 and meta["params"]["min_spacing_s"] == 0.15
    assert meta["source_sha256"] == tp.file_sha256(path) and meta["qc_list_sha256"] == "unfrozen"
    assert meta["summary"]["n_taps"] == 8


def test_a_clip_without_taps_still_writes_an_empty_parquet(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    write_clip(world, REFS[0], np.linspace(0.0, 1.0, 80))
    write_clip(world, REFS[1], np.full(5, 0.5))
    assert tp.main(["--all"]) == 0
    for ref, status in ((REFS[0], "no_taps"), (REFS[1], "too_few_samples")):
        tab, meta = tp.read_taps(world["out"] / f"{ref}.parquet")
        assert (
            len(tab) == 0 and list(tab.columns) == list(tp.TAP_COLUMNS) and meta["status"] == status
        )
        assert (
            str(pq.read_table(world["out"] / f"{ref}.parquet").schema.field("tap_index").type)
            == "int64"
        )
    summary = pd.read_csv(world["out"] / tp.SUMMARY_NAME)
    assert (summary["n_taps"] == 0).all()


def test_qc_lists_retained_and_ok_flag_processed_rejected_skipped(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    make_clips(world, REFS)
    write_qc(
        world,
        [(REFS[0], "ok", ""), (REFS[1], "ok_flag", "slow start"), (REFS[2], "ok", "")],
        [REFS[3]],
    )
    assert tp.main(["--all"]) == 0
    out = capsys.readouterr().out
    assert "not found" not in out and "1 rejected in QC (skipped)" in out
    assert sorted(p.name for p in world["out"].glob("*.parquet")) == [
        f"{r}.parquet" for r in REFS[:3]
    ]
    s = pd.read_csv(world["out"] / tp.SUMMARY_NAME, dtype=str, keep_default_na=False).set_index(
        "ref"
    )
    assert s["qc_status"].to_dict() == {
        REFS[0]: "retained",
        REFS[1]: "retained_flag",
        REFS[2]: "retained",
    }
    assert s["qc_flag"].to_dict() == {REFS[0]: "", REFS[1]: "slow start", REFS[2]: ""}
    _, meta = tp.read_taps(world["out"] / f"{REFS[1]}.parquet")
    assert meta["qc_list_sha256"] != "unfrozen" and meta["qc_status"] == "retained_flag"


def test_a_clip_in_neither_qc_list_is_an_error_listing_hashes_only(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    make_clips(world, REFS)
    write_qc(world, [(REFS[0], "ok", ""), (REFS[1], "ok", "")], [REFS[2]])  # REFS[3] is in neither
    assert tp.main(["--all"]) == 1
    out = capsys.readouterr().out
    assert "1 clips are in neither QC list: 0000000d" in out
    assert (world["out"] / f"{REFS[0]}.parquet").is_file()  # the classified clips were processed
    assert not (world["out"] / f"{REFS[3]}.parquet").exists()


def test_a_single_rejected_clip_is_skipped_by_clip_mode(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    make_clips(world, REFS)
    write_qc(world, [(REFS[0], "ok", "")], [REFS[1]])
    assert tp.main(["--clip", REFS[1]]) == 0
    assert "rejected in the QC list, skipped" in capsys.readouterr().out
    assert not list(world["out"].glob("*.parquet"))
    assert tp.main(["--clip", REFS[2]]) == 1  # in neither list


def test_a_missing_rejections_file_next_to_the_retained_list_exits_2(
    world: dict[str, Path],
) -> None:
    make_clips(world, REFS)
    pd.DataFrame([(REFS[0], "ok", "")], columns=["ref", "verdict", "note"]).to_csv(
        world["docs"] / tp.RETAINED_NAME, index=False
    )
    assert tp.main(["--all"]) == 2


def test_a_clip_run_leaves_the_summary_untouched(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    make_clips(world, REFS)
    assert tp.main(["--all"]) == 0
    before = (world["out"] / tp.SUMMARY_NAME).read_bytes()
    capsys.readouterr()
    assert tp.main(["--clip", REFS[1], "--force"]) == 0
    assert (world["out"] / tp.SUMMARY_NAME).read_bytes() == before
    assert f"clip {REFS[1]}:" in capsys.readouterr().out
    (world["out"] / tp.SUMMARY_NAME).unlink()
    assert tp.main(["--clip", REFS[1], "--force"]) == 0
    assert not (world["out"] / tp.SUMMARY_NAME).exists()  # and it never creates one


def test_resume_skips_unchanged_clips_and_force_or_changes_recompute(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    make_clips(world, REFS)
    assert tp.main(["--all"]) == 0
    first = {p.name: p.read_bytes() for p in world["out"].iterdir()}
    capsys.readouterr()
    assert tp.main(["--all"]) == 0
    assert "0 computed, 4 up to date" in capsys.readouterr().out
    assert {p.name: p.read_bytes() for p in world["out"].iterdir()} == first  # byte-identical
    assert tp.main(["--all", "--force"]) == 0
    assert "4 computed, 0 up to date" in capsys.readouterr().out
    assert {p.name: p.read_bytes() for p in world["out"].iterdir()} == first  # deterministic
    write_clip(world, REFS[0], train(6)[0])  # a changed source file
    assert tp.main(["--all"]) == 0
    assert "1 computed, 3 up to date" in capsys.readouterr().out
    write_qc(world, [(r, "ok", "") for r in REFS], [])  # a new QC list changes every clip's key
    assert tp.main(["--all"]) == 0
    assert "4 computed, 0 up to date" in capsys.readouterr().out
    assert tp.main(["--all"]) == 0 and "0 computed, 4 up to date" in capsys.readouterr().out


def test_the_summary_of_a_resumed_run_equals_the_fresh_one(world: dict[str, Path]) -> None:
    make_clips(world, REFS)
    assert tp.main(["--all"]) == 0
    fresh = (world["out"] / tp.SUMMARY_NAME).read_bytes()
    assert tp.main(["--all"]) == 0
    assert (world["out"] / tp.SUMMARY_NAME).read_bytes() == fresh


def test_exit_codes_for_bad_input(
    world: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_clips(world, REFS)
    assert tp.main(["--all", "--out-dir", str(tmp_path / "elsewhere")]) == 2  # outside interim
    err = capsys.readouterr().err
    assert err.strip() == "error: --out-dir must be inside data/interim"  # no absolute path
    assert not (tmp_path / "elsewhere").exists()
    assert tp.main(["--clip", "CONTROL01_DCHA"]) == 2  # not a hash
    assert "CONTROL" not in capsys.readouterr().err  # and the argument is not echoed
    assert tp.main(["--clip", "ffffffff"]) == 2  # no such file
    empty = tmp_path / "empty"
    empty.mkdir()
    assert tp.main(["--all", "--smoothed-dir", str(empty)]) == 2
    with pytest.raises(SystemExit) as exc:
        tp.main(["--all", "--clip", REFS[0]])
    assert exc.value.code == 2
    with pytest.raises(SystemExit):
        tp.main([])


def test_a_broken_smoothed_file_fails_that_clip_only(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    make_clips(world, REFS[:2])
    (world["smoothed"] / "0000000c.parquet").write_bytes(b"not a parquet file")
    assert tp.main(["--all"]) == 1
    out = capsys.readouterr().out
    assert "failed: 0000000c" in out and "2 ok" in out and str(world["smoothed"]) not in out
    assert (world["out"] / f"{REFS[0]}.parquet").is_file()
    summary = pd.read_csv(world["out"] / tp.SUMMARY_NAME, dtype=str)
    assert summary["ref"].tolist() == REFS[:2]  # the failed clip has no row


def test_a_failing_write_leaves_no_partial_file(
    world: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    make_clips(world, REFS[:1])

    def boom(*args: Any, **kwargs: Any) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(tp.pq, "write_table", boom)
    assert tp.main(["--all"]) == 1
    assert not list(world["out"].glob("*.parquet")) and not list(
        world["out"].glob(f".*{tp.TMP_MARKER}*")
    )


def test_the_plot_option_writes_hashed_png_with_the_legend_outside_the_axes(
    world: dict[str, Path],
) -> None:
    make_clips(world, REFS[:2])
    v = train(8)[0].copy()
    v[60:70] = np.nan
    write_clip(world, REFS[1], v)
    assert tp.main(["--all", "--plot"]) == 0
    assert sorted(p.name for p in (world["out"] / "plots").iterdir()) == [
        f"{r}_taps.png" for r in REFS[:2]
    ]
    tab = pd.read_parquet(world["out"] / f"{REFS[1]}.parquet")
    fig = tp.plot_clip(
        REFS[1], sm.read_smoothed(world["smoothed"] / f"{REFS[1]}.parquet").table, tab
    )
    canvas = fig.canvas
    assert isinstance(canvas, FigureCanvasAgg)  # plot_clip attaches an Agg canvas
    canvas.draw()
    renderer = canvas.get_renderer()
    legend = fig.legends[0].get_window_extent(renderer)
    for ax in fig.axes:
        box = ax.get_window_extent(renderer)
        assert legend.y1 <= box.y0 + 0.5 or legend.y0 >= box.y1 - 0.5  # not over the signal area
    assert len(fig.axes[0].patches) == 1  # the one NaN span is drawn
    peaks = [ln for ln in fig.axes[0].lines if ln.get_marker() == "v"]
    assert len(peaks) == 1 and np.asarray(peaks[0].get_xdata()).size == len(tab)
    assert REFS[1] in fig.axes[0].get_title()


def test_no_raw_ids_labels_or_paths_in_any_output(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    make_clips(world, REFS)
    write_qc(
        world,
        [(REFS[0], "ok", ""), (REFS[1], "ok_flag", "slow start"), (REFS[2], "ok", "")],
        [REFS[3]],
    )
    assert tp.main(["--all", "--plot"]) == 0
    out = capsys.readouterr().out
    names = [p.relative_to(world["out"]).as_posix() for p in world["out"].rglob("*")]
    blob = out + "\n".join(names) + (world["out"] / tp.SUMMARY_NAME).read_text()
    for p in world["out"].glob("*.parquet"):
        blob += json.dumps(tp.read_taps(p)[1])
        blob += "".join(pd.read_parquet(p).columns)
    lower = blob.lower()
    for bad in ("control", "updrs", "group", "clip_id", "participant", "/users", "tmp"):
        assert bad not in lower, bad
    assert all(
        len(Path(n).stem.split("_")[0]) == 8 for n in names if n.endswith((".parquet", ".png"))
    )


def test_flags_never_remove_a_tap() -> None:
    v, _, _ = cycles([10] * 9, [1.0] * 9, lead=2, tail=2)
    r60 = tp.extract_taps(
        REF, times(len(v), 60.0), v, v * 1.4, np.ones(len(v), dtype=bool), np.arange(len(v))
    )
    assert len(r60.table) == 9
    assert all({"narrow", "raw_disagrees", "on_filled"} <= f for f in flags_of(r60))
