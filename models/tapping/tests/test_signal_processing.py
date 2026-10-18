"""Tests for models/tapping/src/signal_processing.py (task FT-19).

Synthetic data only: made-up landmark tables, no real clip. Clip names such as
``CONTROL9001_DCHA`` are made up. No conftest.py on purpose; ``src`` goes on ``sys.path`` here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import signal_processing as sp  # noqa: E402
from hubu_fis import pseudonym  # noqa: E402
from test_hand_scale import make_table  # noqa: E402

STEM = "CONTROL9001_DCHA"
REF = pseudonym(STEM)
POLICY = sp.FillPolicy(max_gap_frames=5)


def irregular_times(n: int, fps: float = 30.0, seed: int = 0) -> np.ndarray:
    """Strictly increasing times (s) with up to +-8 ms of jitter around the nominal frames."""
    rng = np.random.default_rng(seed)
    return np.arange(n) / fps + rng.uniform(-0.008, 0.008, n)


def with_gap(values: np.ndarray, start: int, length: int) -> np.ndarray:
    out = values.copy()
    out[start : start + length] = np.nan
    return out


# --------------------------------------------------------------------------- gap runs


def test_find_gaps_lists_start_and_length() -> None:
    v = np.array([1, np.nan, np.nan, 2, 3, np.nan, 4, np.nan])
    assert sp.find_gaps(v) == [(1, 2), (5, 1), (7, 1)]
    assert sp.find_gaps(np.ones(4)) == []


def test_fill_limit_is_inclusive_and_limit_plus_one_is_not_filled() -> None:
    t = np.arange(40) / 30.0
    base = np.linspace(1.0, 2.0, 40)
    at_limit, filled_at = sp.fill_gaps(t, with_gap(base, 10, 5), max_gap_frames=5)
    assert filled_at.sum() == 5 and np.isfinite(at_limit).all()
    over, filled_over = sp.fill_gaps(t, with_gap(base, 10, 6), max_gap_frames=5)
    assert filled_over.sum() == 0
    assert np.isnan(over[10:16]).all()


def test_filling_a_linear_ramp_is_exact_on_irregular_times() -> None:
    t = irregular_times(60)
    ramp = 0.5 + 2.0 * t  # linear in TIME, not in the sample index
    out, filled = sp.fill_gaps(t, with_gap(ramp, 20, 4), max_gap_frames=5)
    assert filled.sum() == 4
    np.testing.assert_allclose(out, ramp, atol=1e-12)


def test_interpolating_in_index_instead_of_time_would_be_wrong_here() -> None:
    # a long time step inside the gap: by index the midpoint would be 0.5, by time it is not
    t = np.array([0.0, 0.1, 0.5, 0.6, 0.7])
    v = np.array([0.0, np.nan, np.nan, 6.0, 7.0])
    out, _ = sp.fill_gaps(t, v, max_gap_frames=5)
    assert out[1] == pytest.approx(1.0) and out[2] == pytest.approx(5.0)


def test_fill_gaps_never_extrapolates_a_run_at_the_start_or_the_end() -> None:
    t = np.arange(12) / 30.0
    v = np.linspace(1.0, 2.0, 12)
    v[:2] = np.nan
    v[-3:] = np.nan
    out, filled = sp.fill_gaps(t, v, max_gap_frames=5)
    assert filled.sum() == 0
    assert np.isnan(out[:2]).all() and np.isnan(out[-3:]).all()


def test_a_gap_in_the_middle_is_never_zero() -> None:
    t = np.arange(50) / 30.0
    v = np.full(50, 1.5)
    long_gap = with_gap(v, 10, 12)
    out, _ = sp.fill_gaps(t, long_gap, max_gap_frames=5)
    assert np.isnan(out[10:22]).all()
    assert not (out == 0).any()
    short = sp.process_series(t, with_gap(v, 10, 3), POLICY)
    assert not (short.value == 0).any()


def test_inputs_are_not_changed() -> None:
    t = np.arange(30) / 30.0
    v = with_gap(np.linspace(1, 2, 30), 5, 3)
    v[0] = np.nan
    t_before, v_before = t.copy(), v.copy()
    sp.process_series(t, v, POLICY)
    sp.fill_gaps(t, v, 5)
    np.testing.assert_array_equal(t, t_before)
    np.testing.assert_array_equal(v, v_before)


# --------------------------------------------------------------------------- trimming


def test_leading_and_trailing_nan_are_trimmed_and_recorded() -> None:
    t = np.arange(40) / 30.0
    v = np.linspace(1, 2, 40)
    v[:4] = np.nan
    v[-7:] = np.nan
    r = sp.process_series(t, v, POLICY)
    assert (r.start, r.stop) == (4, 33)
    assert len(r.value) == 29 and np.isfinite(r.value).all()
    assert r.trimmed_start_s == pytest.approx(4 / 30.0)
    assert r.trimmed_end_s == pytest.approx(7 / 30.0)
    assert r.filled.sum() == 0  # edges are never filled or extrapolated


def test_a_short_gap_at_the_edge_is_trimmed_not_filled() -> None:
    t = np.arange(20) / 30.0
    v = with_gap(np.ones(20), 0, 2)
    r = sp.process_series(t, v, POLICY)
    assert r.start == 2 and r.filled.sum() == 0


def test_processing_twice_equals_processing_once() -> None:
    t = irregular_times(80)
    v = np.sin(2 * np.pi * 2.0 * t)
    v[:3] = np.nan
    v[-2:] = np.nan
    v = with_gap(v, 30, 4)
    v = with_gap(v, 50, 9)
    once = sp.process_series(t, v, POLICY)
    twice = sp.process_series(once.t, once.value, POLICY)
    np.testing.assert_array_equal(once.t, twice.t)
    np.testing.assert_array_equal(once.value, twice.value)
    assert twice.filled.sum() == 0 and twice.trimmed_start_s == 0 and twice.trimmed_end_s == 0


def test_all_nan_gives_an_empty_result_not_a_crash() -> None:
    t = np.arange(10) / 30.0
    r = sp.process_series(t, np.full(10, np.nan), POLICY)
    assert len(r.value) == 0 and r.start == r.stop == 0
    assert np.isnan(r.trimmed_start_s) and np.isnan(r.trimmed_end_s)


# --------------------------------------------------------------------------- whole clips


def clip_table(values: np.ndarray, t_s: np.ndarray, fps: float = 30.0) -> pd.DataFrame:
    df = make_table(values, pts=t_s * 1000.0)
    df["timestamp_ms"] = np.round(np.arange(len(df)) * 1000 / fps).astype(np.int64)
    return df


def test_sinusoid_on_irregular_timestamps_recovers_frequency_and_peak_times() -> None:
    f = 2.0
    n = 300
    t = irregular_times(n)
    hand = 100.0  # make_table's palm length (wrist to knuckle 9), so value = distance / 100
    d = 60.0 + 40.0 * np.sin(2 * np.pi * f * t)
    d[100:103] = np.nan  # a short gap that gets filled
    sig = sp.build_signal(clip_table(d, t), REF, policy=POLICY, mode="clip_median")
    tt = sig.table["t_s"].to_numpy()
    v = sig.table["value"].to_numpy()
    assert np.isfinite(v).all() and len(v) == n
    # frequency: count rising zero crossings of the demeaned signal in true time
    z = v - v.mean()
    up = np.flatnonzero((z[:-1] < 0) & (z[1:] >= 0))
    cross_t = tt[up] + (0 - z[up]) * (tt[up + 1] - tt[up]) / (z[up + 1] - z[up])
    assert 1.0 / np.median(np.diff(cross_t)) == pytest.approx(f, rel=0.01)
    # peak times: local maxima (from the interpolated, unfilled part) sit at t where sin = 1
    peaks = np.flatnonzero((v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:])) + 1
    expected = (np.round(f * tt[peaks] - 0.25) + 0.25) / f
    assert np.abs(tt[peaks] - expected).max() < 1.5 / 30.0
    np.testing.assert_allclose(v[:100] * hand, d[:100], atol=1e-6)


def test_a_uniform_time_assumption_would_misplace_the_samples() -> None:
    n = 120
    t = irregular_times(n)
    d = 60.0 + 40.0 * np.sin(2 * np.pi * 2.0 * t)
    sig = sp.build_signal(clip_table(d, t), REF, policy=POLICY, mode="clip_median")
    assert not np.allclose(sig.table["t_s"], np.arange(n) / 30.0, atol=1e-4)
    np.testing.assert_allclose(sig.table["t_s"], t, atol=1e-9)


def test_filled_samples_are_flagged_and_measured_is_nan_there() -> None:
    n = 100
    t = np.arange(n) / 30.0
    d = np.linspace(40, 80, n)
    d[40:43] = np.nan
    d[70:80] = np.nan  # too long
    sig = sp.build_signal(clip_table(d, t), REF, policy=POLICY, mode="clip_median")
    tab = sig.table
    assert tab["filled"].sum() == 3
    assert np.isnan(tab.loc[tab["filled"], "measured"]).all()
    assert np.isfinite(tab.loc[tab["filled"], "value"]).all()
    assert tab["value"].isna().sum() == 10 and not tab["filled"].iloc[70:80].any()
    assert sig.meta["fraction_filled"] == pytest.approx(3 / n)
    assert sig.meta["longest_remaining_gap_frames"] == 10
    assert sig.meta["fill_max_gap_frames"] == 5 and sig.meta["scale"] == "w9"


def test_clip_trimming_is_recorded_in_the_metadata() -> None:
    n = 100
    t = np.arange(n) / 30.0
    d = np.linspace(40, 80, n)
    d[:6] = np.nan
    d[-9:] = np.nan
    sig = sp.build_signal(clip_table(d, t), REF, policy=POLICY, mode="clip_median")
    assert sig.meta["trimmed_start_frames"] == 6 and sig.meta["trimmed_end_frames"] == 9
    assert sig.meta["trimmed_start_s"] == pytest.approx(6 / 30.0)
    assert sig.meta["trimmed_end_s"] == pytest.approx(9 / 30.0)
    assert sig.meta["frames_in"] == n and sig.meta["frames_out"] == n - 15
    assert sig.table["frame"].iloc[0] == 6


def test_the_landmarks_table_is_not_mutated() -> None:
    t = np.arange(60) / 30.0
    d = with_gap(np.linspace(40, 80, 60), 20, 3)
    df = clip_table(d, t)
    before = df.copy(deep=True)
    sp.build_signal(df, REF, policy=POLICY)
    pd.testing.assert_frame_equal(df, before)


def test_an_all_nan_clip_gives_an_empty_table_with_a_status() -> None:
    t = np.arange(30) / 30.0
    sig = sp.build_signal(clip_table(np.full(30, np.nan), t), REF, policy=POLICY)
    assert sig.meta["status"] == sp.STATUS_EMPTY
    assert len(sig.table) == 0 and list(sig.table.columns) == list(sp.TABLE_COLUMNS)
    assert sig.meta["fraction_filled"] != sig.meta["fraction_filled"]  # NaN, not 0


def test_a_60_fps_clip_works() -> None:
    n = 240
    t = np.arange(n) / 60.0
    d = 60.0 + 30.0 * np.sin(2 * np.pi * 3.0 * t)
    d[50:54] = np.nan
    df = clip_table(d, t, fps=60.0)
    sig = sp.build_signal(df, REF, policy=POLICY, mode="clip_median")
    assert sig.meta["median_rate_hz"] == pytest.approx(60.0)
    assert sig.table["filled"].sum() == 4
    grid = sp.resample_uniform(sig)
    assert grid.meta["uniform_hz"] == 60.0
    np.testing.assert_allclose(np.diff(grid.table["t_s"]), 1 / 60.0, atol=1e-9)


def test_missing_or_unordered_pts_is_refused_by_hashed_reference() -> None:
    t = np.arange(30) / 30.0
    df = clip_table(np.linspace(40, 80, 30), t)
    bad = df.copy()
    bad.loc[5, "pts_ms"] = np.nan
    with pytest.raises(sp.SignalError, match=REF) as err:
        sp.build_signal(bad, REF)
    assert STEM not in str(err.value)
    back = df.copy()
    back.loc[7, "pts_ms"] = back.loc[3, "pts_ms"]
    with pytest.raises(sp.SignalError, match="strictly increasing"):
        sp.build_signal(back, REF)


# --------------------------------------------------------------------------- uniform grid


def test_resample_is_uniform_and_never_bridges_a_long_gap() -> None:
    n = 200
    t = irregular_times(n)
    d = 60.0 + 30.0 * np.sin(2 * np.pi * 1.5 * t)
    d[80:100] = np.nan
    sig = sp.build_signal(clip_table(d, t), REF, policy=POLICY, mode="clip_median")
    grid = sp.resample_uniform(sig, 30.0)
    g = grid.table
    np.testing.assert_allclose(np.diff(g["t_s"]), 1 / 30.0, atol=1e-9)
    in_gap = (g["t_s"] > t[80]) & (g["t_s"] < t[99])
    assert g.loc[in_gap, "value"].isna().all()
    assert (g["value"].dropna() != 0).all()
    assert grid.meta["uniform_hz"] == 30.0


def test_resample_recovers_a_sinusoid_at_true_times() -> None:
    f, n = 1.5, 240
    t = irregular_times(n)
    d = 60.0 + 30.0 * np.sin(2 * np.pi * f * t)
    sig = sp.build_signal(clip_table(d, t), REF, policy=POLICY, mode="clip_median")
    g = sp.resample_uniform(sig, 30.0).table
    expected = (60.0 + 30.0 * np.sin(2 * np.pi * f * g["t_s"].to_numpy())) / 100.0
    # linear interpolation of a 1.5 Hz sine sampled at 30 Hz: a small, bounded error
    assert np.abs(g["value"].to_numpy() - expected).max() < 0.01


# --------------------------------------------------------------------------- files


def test_write_and_read_roundtrip_keeps_parameters(tmp_path: Path) -> None:
    t = np.arange(60) / 30.0
    d = with_gap(np.linspace(40, 80, 60), 20, 3)
    sig = sp.build_signal(clip_table(d, t), REF, policy=sp.FillPolicy(4), mode="clip_median")
    out = tmp_path / f"{REF}.parquet"
    sp.write_signal(sig, out, source_sha256="abc")
    back = sp.read_signal(out)
    pd.testing.assert_frame_equal(back.table, sig.table)
    assert back.meta["fill_max_gap_frames"] == 4 and back.meta["source_sha256"] == "abc"
    assert back.meta["fill_method"] == "linear_in_pts_time"
    assert not list(tmp_path.glob(f".*{sp.TMP_MARKER}*"))


def test_a_failed_write_leaves_no_partial_file(tmp_path: Path) -> None:
    sig = sp.Signal(REF, pd.DataFrame({"frame": [object()]}), {"clip_ref": REF})
    with pytest.raises(Exception):  # noqa: B017 - any arrow error; the point is the cleanup
        sp.write_signal(sig, tmp_path / f"{REF}.parquet")
    assert list(tmp_path.iterdir()) == []


def landmarks_file(tmp_path: Path, stem: str = STEM, **kwargs: Any) -> Path:
    import hand_landmarks as hl

    t = np.arange(90) / 30.0
    d = with_gap(np.linspace(40, 80, 90), 30, 3)
    df = clip_table(d, t)
    path = tmp_path / "landmarks" / f"{stem}.parquet"
    hl.write_landmarks(df, path)
    return path


@pytest.fixture
def interim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway ``data/interim``: the tests never write into the real one."""
    root = tmp_path / "interim"
    root.mkdir()
    monkeypatch.setattr(sp, "INTERIM_ROOT", root)
    return root


def run_cli(tmp_path: Path, *extra: str) -> int:
    out = sp.INTERIM_ROOT / "signals_test_ft19"
    return sp.main(
        ["--all", "--landmarks-dir", str(tmp_path / "landmarks"), "--out-dir", str(out), *extra]
    )


def test_batch_run_is_resume_safe_and_summary_has_hashed_refs_only(
    tmp_path: Path, interim: Path
) -> None:
    landmarks_file(tmp_path)
    out = interim / "signals_test_ft19"
    assert run_cli(tmp_path, "--mode", "clip_median") == 0
    first = pd.read_csv(out / sp.SUMMARY_NAME)
    assert first["action"].tolist() == ["computed"]
    assert run_cli(tmp_path, "--mode", "clip_median") == 0
    assert pd.read_csv(out / sp.SUMMARY_NAME)["action"].tolist() == ["skipped"]
    assert run_cli(tmp_path, "--mode", "clip_median", "--max-gap-frames", "2") == 0
    assert pd.read_csv(out / sp.SUMMARY_NAME)["action"].tolist() == ["computed"]
    names = [p.name for p in out.iterdir()]
    assert REF + ".parquet" in names
    assert all(STEM not in n for n in names)
    assert STEM not in (out / sp.SUMMARY_NAME).read_text()


def test_a_clip_run_leaves_the_batch_summary_untouched(tmp_path: Path, interim: Path) -> None:
    landmarks_file(tmp_path)
    landmarks_file(tmp_path, "CONTROL9003_IZDA")
    out = interim / "signals_test_ft19"
    assert run_cli(tmp_path, "--mode", "clip_median") == 0
    before = (out / sp.SUMMARY_NAME).read_text()
    assert len(pd.read_csv(out / sp.SUMMARY_NAME)) == 2
    code = sp.main(
        [
            "--clip",
            REF,
            "--landmarks-dir",
            str(tmp_path / "landmarks"),
            "--out-dir",
            str(out),
            "--mode",
            "clip_median",
            "--force",
        ]
    )
    assert code == 0
    assert (out / sp.SUMMARY_NAME).read_text() == before
    assert len(pd.read_csv(out / sp.SUMMARY_NAME)) == 2


def test_a_repeated_timestamp_is_refused_by_hashed_reference() -> None:
    t = np.arange(30) / 30.0
    df = clip_table(np.linspace(40, 80, 30), t)
    df.loc[8, "pts_ms"] = df.loc[7, "pts_ms"]  # two consecutive frames with the same time
    with pytest.raises(sp.SignalError, match=REF) as err:
        sp.build_signal(df, REF)
    assert "strictly increasing" in str(err.value) and STEM not in str(err.value)


def test_a_bad_clip_is_reported_by_hash_and_the_batch_continues(
    tmp_path: Path, interim: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import hand_landmarks as hl

    landmarks_file(tmp_path)
    t = np.arange(40) / 30.0
    bad = clip_table(np.linspace(40, 80, 40), t)
    bad.loc[3, "pts_ms"] = np.nan
    hl.write_landmarks(bad, tmp_path / "landmarks" / "CONTROL9002_IZDA.parquet")
    out = interim / "signals_test_ft19"
    assert run_cli(tmp_path, "--mode", "clip_median") == 1
    text = capsys.readouterr().out
    assert pseudonym("CONTROL9002_IZDA") in text and "CONTROL9002" not in text
    assert (out / f"{REF}.parquet").is_file()


def test_out_dir_outside_interim_is_refused(tmp_path: Path) -> None:
    landmarks_file(tmp_path)
    code = sp.main(
        ["--all", "--landmarks-dir", str(tmp_path / "landmarks"), "--out-dir", str(tmp_path)]
    )
    assert code == 2


def test_missing_clip_is_exit_code_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "landmarks").mkdir()
    assert sp.main(["--clip", "deadbeef", "--landmarks-dir", str(tmp_path / "landmarks")]) == 2
    assert "deadbeef" in capsys.readouterr().err


def test_fill_policy_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        sp.FillPolicy(max_gap_frames=-1)
    with pytest.raises(ValueError):
        sp.FillPolicy(method="cubic")


def test_default_fill_limit_is_the_one_tracking_flags_with() -> None:
    from tracking import Gating

    assert sp.DEFAULT_MAX_GAP_FRAMES == Gating().fillable_gap_frames == 5
