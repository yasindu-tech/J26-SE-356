"""Tests for the timing features (GAIT-14)."""

from __future__ import annotations

import math
from pathlib import Path

import check_timing_features as ctf
import numpy as np
import pytest
from carepd_loader import Walk
from cycles import Cycle, CycleCheck, segment_cycles
from feature_contract import check_feature_contract
from preprocess import detect_heel_strikes
from test_preprocess import make_walk
from timing_features import (
    FEATURE_NAMES,
    WINDOW_STRIDES,
    c4,
    middle_window,
    stride_time_cv,
    timing_features,
)


def check_of(*seconds: float, foot: str = "left", ok: bool = True) -> CycleCheck:
    """A cycle check made of cycles with these durations, one after the other."""
    cycles, t = [], 0
    for s in seconds:
        end = t + round(s * 30)
        cycles.append(Cycle(foot, t, end, s, 0.4, ok, ok))  # type: ignore[arg-type]
        t = end
    return CycleCheck(tuple(cycles), 3)


# --------------------------------------------------------------------------- correction
def test_c4_matches_the_textbook_values() -> None:
    assert c4(2) == pytest.approx(0.7979, abs=1e-4)
    assert c4(3) == pytest.approx(0.8862, abs=1e-4)
    assert c4(10) == pytest.approx(0.9727, abs=1e-4)
    assert 0.95 < c4(100) < 1.0


def test_c4_needs_two_values() -> None:
    with pytest.raises(ValueError):
        c4(1)


def test_the_correction_removes_the_bias_in_the_sd() -> None:
    rng = np.random.default_rng(0)
    for n in (3, 6):
        sd = np.mean([np.std(rng.normal(0, 1, n), ddof=1) for _ in range(40000)])
        assert sd / c4(n) == pytest.approx(1.0, abs=0.01)


# --------------------------------------------------------------------------- variability
def test_cv_of_known_strides() -> None:
    # mean 1.1, sample SD 0.1, three values: CV = 100 * 0.1 / c4(3) / 1.1
    assert stride_time_cv(np.array([1.0, 1.1, 1.2])) == pytest.approx(
        100 * 0.1 / c4(3) / 1.1, rel=1e-9
    )


def test_constant_strides_have_no_variability() -> None:
    assert stride_time_cv(np.full(8, 1.2)) == pytest.approx(0.0, abs=1e-9)


def test_window_is_the_middle_of_the_walk() -> None:
    times = np.arange(10, dtype=float)
    np.testing.assert_array_equal(middle_window(times), [2, 3, 4, 5, 6, 7])
    np.testing.assert_array_equal(middle_window(times[:9]), [1, 2, 3, 4, 5, 6])  # 9: start 1
    np.testing.assert_array_equal(middle_window(times[:6]), times[:6])  # short: all of them
    np.testing.assert_array_equal(middle_window(times[:3]), times[:3])


def test_odd_walk_start_up_and_turn_do_not_count() -> None:
    steady = [1.1, 1.12, 1.08, 1.1, 1.09, 1.11]
    walk = np.array([1.9, 1.6, *steady, 1.7, 2.1])  # slow start-up and turn on the outside
    assert stride_time_cv(walk) == pytest.approx(stride_time_cv(np.array(steady)))


def test_window_size_is_checked() -> None:
    with pytest.raises(ValueError):
        middle_window(np.arange(5.0), 1)
    with pytest.raises(ValueError):
        stride_time_cv(np.array([1.1]))


def test_cv_does_not_depend_on_walk_length_for_a_stationary_rhythm() -> None:
    rng = np.random.default_rng(1)
    short = [stride_time_cv(rng.normal(1.1, 0.05, WINDOW_STRIDES)) for _ in range(4000)]
    long = [stride_time_cv(rng.normal(1.1, 0.05, 20)) for _ in range(4000)]
    assert np.mean(short) == pytest.approx(np.mean(long), rel=0.06)


# --------------------------------------------------------------------------- features
def test_features_of_a_regular_walk() -> None:
    f = timing_features(check_of(*[1.2] * 8))
    assert f.stride_time_mean == pytest.approx(1.2)
    assert f.cadence == pytest.approx(100.0)  # 120 / 1.2 steps per minute
    assert f.stride_time_cv == pytest.approx(0.0, abs=1e-9)
    assert f.n_strides == 8


def test_mean_and_cadence_use_every_real_stride_but_cv_only_the_window() -> None:
    times = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0]
    f = timing_features(check_of(*times))
    assert f.stride_time_mean == pytest.approx(np.mean(times))
    assert f.cadence == pytest.approx(120 / np.mean(times))
    assert f.stride_time_cv == pytest.approx(0.0, abs=1e-9)  # the six middle strides are equal


def test_implausible_cycles_are_ignored() -> None:
    good = check_of(1.1, 1.1, 1.1)
    bad = check_of(5.0, 0.2, ok=False)
    both = CycleCheck(good.cycles + bad.cycles, 3)
    assert timing_features(both).stride_time_mean == pytest.approx(1.1)
    assert timing_features(both).n_strides == 3


def test_both_feet_are_pooled_in_time_order() -> None:
    left = Cycle("left", 0, 33, 1.1, 0.4, True, True)
    right = Cycle("right", 16, 52, 1.2, 0.4, True, True)
    left2 = Cycle("left", 33, 70, 1.2, 0.4, True, True)
    f = timing_features(CycleCheck((left, right, left2), 3))
    assert f.stride_time_mean == pytest.approx((1.1 + 1.2 + 1.2) / 3)
    assert f.n_strides == 3


def test_too_few_strides_is_an_error_not_a_made_up_value() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        timing_features(check_of(1.1))
    with pytest.raises(ValueError):
        timing_features(check_of())


def test_two_strides_still_give_a_value() -> None:  # the short KOA-PD-NM clips
    f = timing_features(check_of(1.0, 1.2))
    assert math.isfinite(f.stride_time_cv) and f.stride_time_cv > 0


def test_synthetic_walk_end_to_end() -> None:
    joints, _, _ = make_walk(seconds=8.0)
    f = timing_features(segment_cycles(detect_heel_strikes(joints, 30.0)))
    assert f.stride_time_mean == pytest.approx(1.1, abs=1 / 30)
    assert f.cadence == pytest.approx(120 / f.stride_time_mean)
    assert f.stride_time_cv < 4.0  # a perfectly regular walk, only frame rounding


# --------------------------------------------------------------------------- contract
def test_feature_names_pass_the_feature_contract() -> None:
    check_feature_contract(FEATURE_NAMES, label_name=["UPDRS_GAIT", "severity"])


def test_the_stride_count_is_not_a_model_input() -> None:
    f = timing_features(check_of(*[1.2] * 8))
    assert set(f.as_dict()) == set(FEATURE_NAMES)
    assert "n_strides" not in f.as_dict() and "n_strides" not in FEATURE_NAMES


# --------------------------------------------------------------------------- command line
def synthetic_walks() -> list[Walk]:
    out = []
    for i, (cohort, severity, seconds) in enumerate(
        (("3DGait", 0, 8.0), ("3DGait", 1, 8.0), ("PD-GaM", 2, 8.0), ("PD-GaM", 0, 1.3))
    ):
        joints, _, _ = make_walk(fps=30, seconds=seconds)
        out.append(Walk(cohort, f"secretsubj{i}", "w1", None, 30, joints, severity, severity))
    return out


def test_cli_reports_features_and_writes_the_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(ctf, "load_all", lambda *a, **k: (synthetic_walks(), None))
    rc = ctf.main(["--out-dir", str(tmp_path), "--no-rank-gate"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "stride_time_cv" in out and "secretsubj" not in out
    rows = (tmp_path / "timing_features.csv").read_text().splitlines()
    assert len(rows) == 1 + 3  # the 1.3 s clip fails the cycle check and gets no features


def test_cli_fails_when_a_cohort_has_an_implausible_stride_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(ctf, "load_all", lambda *a, **k: (synthetic_walks(), None))
    monkeypatch.setattr(ctf, "PLAUSIBLE_STRIDE_S", (0.2, 0.5))  # the 1.1 s strides are now "wrong"
    assert ctf.main(["--out-dir", str(tmp_path), "--no-rank-gate"]) == 1
    assert "outside" in capsys.readouterr().out


def test_cli_missing_files_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert ctf.main(["--data-dir", str(tmp_path)]) == 2
