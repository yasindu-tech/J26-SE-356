"""Tests for models/tapping/src/smoothing.py (task FT-20).

Synthetic data only: made-up signals, no real clip and no real participant. No conftest.py on
purpose; ``src`` goes on ``sys.path`` here. Nothing is written into the real ``data/interim``:
the command-line tests point ``INTERIM_ROOT`` at a temporary folder.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.signal import find_peaks, savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import signal_processing as sp  # noqa: E402
import smoothing as sm  # noqa: E402

REF = "a1b2c3d4"
OTHER = "0f0e0d0c"


def make_signal(
    t: np.ndarray, value: np.ndarray, *, filled: Any = None, ref: str = REF
) -> sp.Signal:
    n = len(t)
    fl = np.zeros(n, dtype=bool) if filled is None else np.asarray(filled, dtype=bool)
    table = pd.DataFrame(
        {
            "frame": np.arange(n, dtype=np.int64),
            "t_s": np.asarray(t, dtype=float),
            "value": np.asarray(value, dtype=float),
            "measured": np.where(fl, np.nan, value),
            "filled": fl,
            "scale_px": np.full(n, 100.0),
        },
        columns=list(sp.TABLE_COLUMNS),
    )
    meta = {"clip_ref": ref, "scale": "w9", "fill_max_gap_frames": 5, "status": "ok"}
    return sp.Signal(ref, table, meta)


def grid(n: int, fps: float = 30.0) -> np.ndarray:
    return np.arange(n) / fps


def jittered(n: int, fps: float = 30.0, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.arange(n) / fps + rng.uniform(-0.008, 0.008, n)


# --------------------------------------------------------------------------- the filter


def test_matches_scipy_savgol_on_a_30_hz_grid_interior_and_edges() -> None:
    rng = np.random.default_rng(0)
    t = grid(60)
    v = np.sin(2 * np.pi * 1.3 * t) + 0.2 * rng.standard_normal(60)
    out, smoothed, counts = sm.smooth_series(t, v)
    expected = savgol_filter(v, 5, 3, mode="interp")
    np.testing.assert_allclose(out, expected, atol=1e-9)  # every sample, including both ends
    assert smoothed.all()
    assert counts["n_widened"] == 4  # the first and last two samples of the one segment
    assert not np.allclose(out, savgol_filter(v, 7, 3, mode="interp"), atol=1e-6)  # not 7


def test_a_60_hz_window_holds_eleven_samples() -> None:
    rng = np.random.default_rng(1)
    t = grid(120, fps=60.0)
    v = np.cos(2 * np.pi * 2.0 * t) + 0.2 * rng.standard_normal(120)
    out, _, counts = sm.smooth_series(t, v)
    expected = savgol_filter(v, 11, 3, mode="interp")
    np.testing.assert_allclose(out[5:-5], expected[5:-5], atol=1e-8)  # interior: 11 samples
    assert not np.allclose(out[5:-5], savgol_filter(v, 9, 3, mode="interp")[5:-5], atol=1e-6)
    # Edges follow the stated rule, not scipy's: the window is cut short (6 to 10 samples,
    # at least order + 2), so nothing is widened; scipy would fit the first 11 samples instead.
    assert counts["n_widened"] == 0
    for i in (0, 3, 119, 116):
        lo, hi = max(0, i - 5), min(120, i + 6)
        poly = np.polyfit(t[lo:hi] - t[i], v[lo:hi], 3)
        assert out[i] == pytest.approx(poly[-1], abs=1e-8)
    assert not np.allclose(out[:5], expected[:5], atol=1e-6)


def test_a_sinusoid_keeps_its_amplitude_and_frequency() -> None:
    f, n = 2.0, 600
    t = grid(n)
    v = 0.8 + 0.5 * np.sin(2 * np.pi * f * t)
    out, _, _ = sm.smooth_series(t, v)
    mid = slice(10, -10)
    assert (out[mid].max() - out[mid].min()) / (v[mid].max() - v[mid].min()) > 0.99
    up = np.flatnonzero((out[:-1] < 0.8) & (out[1:] >= 0.8))
    crossings = t[up] + (0.8 - out[up]) * (t[up + 1] - t[up]) / (out[up + 1] - out[up])
    assert 1.0 / np.median(np.diff(crossings)) == pytest.approx(f, rel=0.01)


def test_irregular_timestamps_are_used_not_assumed_uniform() -> None:
    n = 400
    t = jittered(n)
    truth = 0.8 + 0.5 * np.sin(2 * np.pi * 2.0 * t)
    out, _, _ = sm.smooth_series(t, truth)
    assert np.abs(out - truth).max() < 0.004  # a cubic fits a 2 Hz sine in 0.167 s closely
    # the same values treated as if they were on the 30 Hz grid are visibly further off
    assumed = savgol_filter(truth, 5, 3, mode="interp")
    assert np.abs(assumed - truth).max() > 2 * np.abs(out - truth).max()


def test_a_time_hole_widens_the_window_and_is_counted() -> None:
    t = grid(40)
    t[20:] += 0.12  # a 120 ms hole: the samples around it have fewer than five neighbours
    v = np.sin(t * 5.0)
    out, smoothed, counts = sm.smooth_series(t, v)
    assert smoothed.all() and np.isfinite(out).all()
    assert counts["n_widened"] > 4  # the four segment-end samples plus those at the hole


def test_a_linear_ramp_is_reproduced_exactly_on_irregular_times() -> None:
    t = jittered(80)
    ramp = 0.3 + 1.7 * t
    out, _, _ = sm.smooth_series(t, ramp)
    np.testing.assert_allclose(out, ramp, atol=1e-9)


# --------------------------------------------------------------------------- tap amplitude


def tap_train(
    amplitudes: list[float], fps: float = 30.0, rate_hz: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Open/close taps with 0.1 s edges and a closed floor: one tap per amplitude."""
    period = 1.0 / rate_hz
    n = int(len(amplitudes) * period * fps)
    t = np.arange(n) / fps
    v = np.full(n, 0.12)
    for k, amp in enumerate(amplitudes):
        s = np.clip((t - k * period) / period, 0.0, 1.0)
        rise = np.clip((s - 0.1) / (0.1 / period * 1.0), 0.0, 1.0)
        fall = np.clip((s - 0.65) / (0.1 / period * 1.0), 0.0, 1.0)
        shape = rise * (1.0 - fall)
        shape = shape * shape * (3.0 - 2.0 * shape)  # smoothstep edges
        v = np.where((s > 0) & (s < 1), 0.12 + (amp - 0.12) * shape, v)
    return t, v


def tap_amplitudes(v: np.ndarray, prominence: float = 0.15) -> list[float]:
    """Per tap: peak minus the lowest value before it (since the previous peak), on one series."""
    peaks, _ = find_peaks(v, prominence=prominence)
    out, prev = [], 0
    for q in peaks:
        out.append(float(v[q] - v[prev : q + 1].min()))
        prev = int(q)
    return out


def test_a_decaying_tap_sequence_keeps_its_last_tap_amplitude() -> None:
    amps = list(np.linspace(1.6, 0.5, 12))  # slow, shrinking taps: a clinical sign, not noise
    t, v = tap_train(amps)
    out, _, _ = sm.smooth_series(t, v)
    raw_amp, sm_amp = tap_amplitudes(v), tap_amplitudes(out)
    assert len(raw_amp) == len(sm_amp) == 12
    ratios = np.array(sm_amp) / np.array(raw_amp)
    assert ratios[-1] >= 0.95
    assert ratios.min() >= 0.95


def test_later_taps_scaled_to_30_percent_are_not_flattened() -> None:
    amps = [1.5] * 6 + [1.5 * 0.3 + 0.12 * 0.7] * 6  # the later taps are scaled in memory
    t, v = tap_train(amps)
    out, _, _ = sm.smooth_series(t, v)
    raw_amp, sm_amp = tap_amplitudes(v, 0.1), tap_amplitudes(out, 0.1)
    assert len(raw_amp) == len(sm_amp) == 12
    assert (np.array(sm_amp) / np.array(raw_amp)).min() >= 0.95


def test_smoothing_never_adds_a_tap_to_a_clean_train() -> None:
    t, v = tap_train([1.4] * 10)
    out, _, _ = sm.smooth_series(t, v)
    assert len(find_peaks(out, prominence=0.15)[0]) == len(find_peaks(v, prominence=0.15)[0])


# --------------------------------------------------------------------------- gaps and flags


def test_gaps_stay_nan_and_are_never_zero() -> None:
    t = grid(60)
    v = np.sin(t * 6.0) + 2.0
    v[20:30] = np.nan
    out, smoothed, _ = sm.smooth_series(t, v)
    assert np.isnan(out[20:30]).all()
    assert not (out == 0).any()
    assert not smoothed[20:30].any()
    assert np.isfinite(out[:20]).all() and np.isfinite(out[30:]).all()


def test_nothing_is_smoothed_across_a_gap() -> None:
    t = grid(60)
    v = np.concatenate([np.full(25, 1.0), np.full(35, 5.0)])  # a step, with a gap at the step
    v[25:30] = np.nan
    out, _, _ = sm.smooth_series(t, v)
    assert np.allclose(out[:25], 1.0, atol=1e-9)  # no bleed of the high segment
    assert np.allclose(out[30:], 5.0, atol=1e-9)  # no bleed of the low segment


def test_each_segment_equals_smoothing_it_alone() -> None:
    rng = np.random.default_rng(5)
    t = grid(80)
    v = rng.standard_normal(80)
    gapped = v.copy()
    gapped[30:34] = np.nan
    out, _, _ = sm.smooth_series(t, gapped)
    first, _, _ = sm.smooth_series(t[:30], v[:30])
    second, _, _ = sm.smooth_series(t[34:], v[34:])
    np.testing.assert_allclose(out[:30], first, atol=1e-12)
    np.testing.assert_allclose(out[34:], second, atol=1e-12)


def test_a_short_segment_keeps_raw_values_and_is_flagged() -> None:
    t = grid(40)
    v = np.sin(t * 4.0)
    v[10:14] = np.nan  # leaves a 10-sample segment before and a 26-sample one after
    v[20:36] = np.nan  # leaves a 6-sample segment between and 4 at the end
    out, smoothed, counts = sm.smooth_series(t, v)
    assert not smoothed[36:].any()
    np.testing.assert_array_equal(out[36:], v[36:])
    assert smoothed[14:20].all() and smoothed[:10].all()
    assert counts["n_short_segments"] == 1 and counts["n_unsmoothed"] == 4
    assert counts["n_segments"] == 3


def test_a_segment_of_exactly_order_plus_two_is_smoothed() -> None:
    t = grid(5)
    out, smoothed, counts = sm.smooth_series(t, np.array([1.0, 2.0, 1.5, 2.5, 2.0]))
    assert smoothed.all() and counts["n_unsmoothed"] == 0


def test_inputs_are_not_changed() -> None:
    t = jittered(50)
    v = np.sin(t * 5.0)
    v[10:13] = np.nan
    t0, v0 = t.copy(), v.copy()
    sm.smooth_series(t, v)
    np.testing.assert_array_equal(t, t0)
    np.testing.assert_array_equal(v, v0)
    sig = make_signal(t, v)
    before = sig.table.copy(deep=True)
    sm.smooth_signal(sig)
    pd.testing.assert_frame_equal(sig.table, before)


def test_the_filled_flag_is_kept_and_filled_samples_are_inputs() -> None:
    t = grid(50)
    v = np.sin(t * 5.0) + 2.0
    filled = np.zeros(50, dtype=bool)
    filled[20:23] = True
    result = sm.smooth_signal(make_signal(t, v, filled=filled))
    tab = result.table
    np.testing.assert_array_equal(tab["filled"].to_numpy(), filled)
    assert tab.loc[filled, "smoothed"].all()
    np.testing.assert_array_equal(tab["value_raw"].to_numpy(), v)
    assert list(tab.columns) == list(sm.TABLE_COLUMNS)


def test_all_nan_signal_gives_a_status_not_a_crash() -> None:
    t = grid(20)
    result = sm.smooth_signal(make_signal(t, np.full(20, np.nan)))
    assert result.meta["status"] == sm.STATUS_EMPTY
    assert result.table["value"].isna().all() and not result.table["smoothed"].any()
    empty = sm.smooth_signal(make_signal(np.empty(0), np.empty(0)))
    assert empty.meta["status"] == sm.STATUS_EMPTY and len(empty.table) == 0


def test_unordered_time_axis_is_refused_by_hashed_reference() -> None:
    t = grid(20)
    t[7] = t[3]
    with pytest.raises(sm.SmoothingError, match=REF):
        sm.smooth_signal(make_signal(t, np.ones(20)))


def test_a_repeated_timestamp_is_refused_by_hashed_reference() -> None:
    t = grid(20)
    t[8] = t[7]  # two consecutive samples with the same time
    with pytest.raises(sm.SmoothingError, match=REF) as err:
        sm.smooth_signal(make_signal(t, np.ones(20)))
    assert "strictly increasing" in str(err.value)


def test_bad_parameters_are_refused() -> None:
    with pytest.raises(ValueError):
        sm.smooth_series(grid(10), np.ones(10), window_s=0.0)
    with pytest.raises(ValueError):
        sm.smooth_series(grid(10), np.ones(10), order=-1)
    with pytest.raises(ValueError):
        sm.smooth_segment(grid(3), np.ones(3))


def test_the_constants_are_the_decided_ones() -> None:
    assert sm.WINDOW_S == 0.167 and sm.ORDER == 3


# --------------------------------------------------------------------------- files


@pytest.fixture
def interim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "interim"
    (root / "signals").mkdir(parents=True)
    monkeypatch.setattr(sm, "INTERIM_ROOT", root)
    return root


def write_source(interim: Path, ref: str = REF, n: int = 90) -> Path:
    t = jittered(n)
    v = np.sin(t * 5.0) + 2.0
    v[30:36] = np.nan
    path = interim / "signals" / f"{ref}.parquet"
    sp.write_signal(make_signal(t, v, ref=ref), path, source_sha256="x")
    return path


def run_cli(interim: Path, *extra: str) -> int:
    return sm.main(
        [
            "--all",
            "--signals-dir",
            str(interim / "signals"),
            "--out-dir",
            str(interim / "smoothed"),
            *extra,
        ]
    )


def test_roundtrip_keeps_table_and_metadata(interim: Path) -> None:
    result = sm.smooth_signal(make_signal(grid(60), np.sin(grid(60) * 4.0)))
    path = interim / "smoothed" / f"{REF}.parquet"
    sm.write_smoothed(result, path, source_sha256="abc")
    back = sm.read_smoothed(path)
    pd.testing.assert_frame_equal(back.table, result.table)
    meta = back.meta
    assert meta["window_s"] == 0.167 and meta["order"] == 3 and meta["source_sha256"] == "abc"
    assert meta["tool"] == sm.TOOL_VERSION and meta["n_widened"] == 4 and meta["n_unsmoothed"] == 0
    assert not list((interim / "smoothed").glob(f".*{sm.TMP_MARKER}*"))


def test_a_failed_write_leaves_no_partial_file(interim: Path) -> None:
    (interim / "smoothed").mkdir()
    bad = sm.Smoothed(REF, pd.DataFrame({"frame": [object()]}), {"clip_ref": REF})
    with pytest.raises(Exception):  # noqa: B017 - any arrow error; the point is the cleanup
        sm.write_smoothed(bad, interim / "smoothed" / f"{REF}.parquet")
    assert list((interim / "smoothed").iterdir()) == []


def test_batch_is_resume_safe_and_metadata_is_idempotent(interim: Path) -> None:
    source = write_source(interim)
    out = interim / "smoothed"
    assert run_cli(interim) == 0
    assert pd.read_csv(out / sm.SUMMARY_NAME)["action"].tolist() == ["computed"]
    first = sm.read_smoothed(out / f"{REF}.parquet")
    assert run_cli(interim) == 0
    assert pd.read_csv(out / sm.SUMMARY_NAME)["action"].tolist() == ["skipped"]
    assert run_cli(interim, "--force") == 0
    assert pd.read_csv(out / sm.SUMMARY_NAME)["action"].tolist() == ["computed"]
    again = sm.read_smoothed(out / f"{REF}.parquet")
    assert again.meta == first.meta  # metadata of a repeated run is identical
    pd.testing.assert_frame_equal(again.table, first.table)
    # a changed source file is recomputed
    t = jittered(90)
    sp.write_signal(make_signal(t, np.sin(t * 7.0) + 2.0), source, source_sha256="y")
    assert run_cli(interim) == 0
    assert pd.read_csv(out / sm.SUMMARY_NAME)["action"].tolist() == ["computed"]


def test_a_clip_run_leaves_the_batch_summary_untouched(
    interim: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_source(interim)
    write_source(interim, OTHER, n=70)
    out = interim / "smoothed"
    assert run_cli(interim) == 0
    before = (out / sm.SUMMARY_NAME).read_text()
    assert len(pd.read_csv(out / sm.SUMMARY_NAME)) == 2
    capsys.readouterr()
    base = ["--signals-dir", str(interim / "signals"), "--out-dir", str(out)]
    assert sm.main(["--clip", REF, *base, "--force"]) == 0
    assert (out / sm.SUMMARY_NAME).read_text() == before
    assert len(pd.read_csv(out / sm.SUMMARY_NAME)) == 2
    assert f"clip {REF}:" in capsys.readouterr().out  # the numbers are still printed


def test_summary_and_files_carry_hashed_refs_only(
    interim: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_source(interim)
    write_source(interim, OTHER, n=70)
    assert run_cli(interim) == 0
    out = interim / "smoothed"
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [f"{OTHER}.parquet", f"{REF}.parquet", sm.SUMMARY_NAME]
    )
    summary = pd.read_csv(out / sm.SUMMARY_NAME)
    assert sorted(summary["ref"]) == sorted([REF, OTHER])
    assert "done in" in capsys.readouterr().out


def test_a_bad_file_is_reported_by_hash_and_the_batch_continues(
    interim: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_source(interim)
    (interim / "signals" / f"{OTHER}.parquet").write_bytes(b"not a parquet file")
    assert run_cli(interim) == 1
    text = capsys.readouterr().out
    assert OTHER in text and "failed" in text
    assert (interim / "smoothed" / f"{REF}.parquet").is_file()
    assert str(interim) not in text  # the path is scrubbed to the hash


def test_plot_option_writes_a_png_named_by_hash(interim: Path) -> None:
    write_source(interim)
    plots = interim / "plots"
    assert run_cli(interim, "--plot", "--plot-dir", str(plots)) == 0
    assert [p.name for p in plots.iterdir()] == [f"{REF}_smoothed.png"]


def test_clip_option_and_exit_code_2_cases(interim: Path, tmp_path: Path) -> None:
    write_source(interim)
    base = ["--signals-dir", str(interim / "signals"), "--out-dir", str(interim / "smoothed")]
    assert sm.main(["--clip", REF, *base]) == 0
    assert sm.main(["--clip", OTHER, *base]) == 2  # no such signal file
    assert sm.main(["--clip", "CONTROL9001_DCHA", *base]) == 2  # a raw ID is not a hash
    assert (
        sm.main(["--all", "--signals-dir", str(interim / "signals"), "--out-dir", str(tmp_path)])
        == 2
    )
    empty = tmp_path / "none"
    empty.mkdir()
    assert (
        sm.main(["--all", "--signals-dir", str(empty), "--out-dir", str(interim / "smoothed")]) == 2
    )
