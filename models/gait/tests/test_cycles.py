"""Tests for gait cycles and the cycle check (GAIT-13, Rule B)."""

from __future__ import annotations

from pathlib import Path

import check_cycles as cc
import numpy as np
import pytest
from carepd_format import L_ANKLE, PELVIS
from carepd_loader import Walk
from cycles import (
    CAREPD_MIN_STRIDES,
    KOA_MIN_STRIDES,
    MAX_CYCLE_S,
    MIN_STEP_EXCURSION_M,
    foot_cycles,
    segment_cycles,
)
from preprocess import FootStrikes, HeelStrikes, detect_foot, preprocess
from test_preprocess import make_walk, swing_height

FPS = 30.0


def strikes(*frames: int) -> FootStrikes:
    f = np.array(frames, dtype=int)
    return FootStrikes(f, f + 3)


def hs(left: FootStrikes, right: FootStrikes) -> HeelStrikes:
    return HeelStrikes(left=left, right=right, fps=FPS, method="onset")


# --------------------------------------------------------------------------- cycles
def test_a_cycle_runs_between_strikes_of_the_same_foot() -> None:
    cycles = foot_cycles(strikes(10, 43, 76), FPS, "left")
    assert [(c.start, c.end) for c in cycles] == [(10, 43), (43, 76)]
    assert all(c.foot == "left" and c.plausible for c in cycles)
    assert cycles[0].seconds == pytest.approx(33 / FPS)


def test_cycles_of_both_feet_are_in_time_order() -> None:
    check = segment_cycles(hs(strikes(10, 43, 76), strikes(27, 60)))
    assert [c.start for c in check.cycles] == [10, 27, 43]


def test_too_long_or_too_short_cycles_are_not_counted() -> None:
    long_gap = int((MAX_CYCLE_S + 0.5) * FPS)  # a pause or a missed strike
    check = segment_cycles(hs(strikes(0, 33, 33 + long_gap), strikes(5, 10)))
    assert check.implausible == 2 and check.bad_duration == 2  # one too long, one too short
    assert check.strides() == 1


def test_stepping_on_the_spot_is_not_a_real_stride() -> None:
    forward = np.zeros(100)
    forward[20:50] = np.linspace(0, 0.05, 30)  # the ankle hardly swings past the body
    forward[50:80] = np.linspace(0.05, -0.25, 30)  # a real stride: well behind, then ahead
    forward[80:] = np.linspace(-0.25, 0.25, 20)
    cycles = foot_cycles(strikes(10, 45, 99), FPS, "left", forward)
    assert cycles[0].excursion_m < MIN_STEP_EXCURSION_M and not cycles[0].real_step
    assert cycles[1].real_step and cycles[1].plausible
    assert not cycles[0].plausible and cycles[0].duration_ok


def test_walk_on_the_spot_fails_the_cycle_check() -> None:
    from preprocess import detect_heel_strikes

    joints, _, _ = make_walk(seconds=8.0, stride_len=0.0)  # feet lift, nobody travels
    check = segment_cycles(detect_heel_strikes(joints, FPS))
    # Without a stance (the foot sliding back relative to the pelvis) the lifts are not even
    # separate landings, and any cycle left would fail the real-step test: no strides at all.
    assert check.strides() == 0 and not check.passed
    assert "real strides" in check.reason


def test_short_shuffling_strides_still_count() -> None:
    from preprocess import detect_heel_strikes

    joints, _, _ = make_walk(seconds=8.0, stride_len=0.3)  # 0.3 m strides, like severe PD
    check = segment_cycles(detect_heel_strikes(joints, FPS))
    assert check.passed and check.not_real_step == 0


# --------------------------------------------------------------------------- Rule B
def test_rule_b_counts_strides_on_both_feet_together() -> None:
    # two strides on the left, one on the right: 3 in total -> passes (one foot alone would not)
    check = segment_cycles(hs(strikes(10, 43, 76), strikes(27, 60)))
    assert (check.strides("left"), check.strides("right"), check.strides()) == (2, 1, 3)
    assert check.passed and check.reason == ""


def test_too_few_strides_are_flagged_with_a_reason() -> None:
    check = segment_cycles(hs(strikes(10, 43), strikes(27, 60)))
    assert not check.passed and "2 real strides" in check.reason


def test_koa_clips_need_two_strides() -> None:
    check = segment_cycles(hs(strikes(10, 43), strikes(27, 60)), KOA_MIN_STRIDES)
    assert check.passed
    assert KOA_MIN_STRIDES == 2 and CAREPD_MIN_STRIDES == 3


def test_min_strides_must_be_positive() -> None:
    with pytest.raises(ValueError):
        segment_cycles(hs(strikes(), strikes()), 0)


def test_stride_times_use_only_plausible_cycles() -> None:
    check = segment_cycles(hs(strikes(0, 33, 200), strikes()))
    np.testing.assert_allclose(check.stride_times(), [33 / FPS])


def test_synthetic_walk_end_to_end() -> None:
    joints, _, _ = make_walk(seconds=6.0)
    from preprocess import detect_heel_strikes

    check = segment_cycles(detect_heel_strikes(joints, FPS))
    assert check.passed and check.strides() >= 6
    np.testing.assert_allclose(np.median(check.stride_times()), 1.1, atol=1 / FPS)


# --------------------------------------------------------------------------- first landing
def test_walk_that_starts_mid_stance_gets_no_landing_at_its_lowest_point() -> None:
    fps = 25
    t = np.arange(150) / fps
    h, landings = swing_height(t, 0.6)
    walk = np.zeros((t.size, 17, 3))
    walk[:, PELVIS, 1] = 0.95 + 0.02 * np.cos(4 * np.pi * (t + 0.3) / 1.1)
    walk[:, :, 1] = walk[:, PELVIS, 1][:, None]
    walk[:, L_ANKLE, 1] = h - 0.1 * np.clip(0.6 - t, 0, None)  # still lowering in stance at t=0
    found = detect_foot(preprocess(walk, fps)[:, L_ANKLE, 1], fps).strikes / fps
    assert found.min() > 0.6  # nothing before the first real swing lands


# --------------------------------------------------------------------------- command line
def synthetic_walks() -> list[Walk]:
    out = []
    for i, (cohort, severity, seconds) in enumerate(
        (("3DGait", 0, 6.0), ("3DGait", 1, 6.0), ("PD-GaM", 2, 6.0), ("PD-GaM", 0, 1.3))
    ):
        joints, _, _ = make_walk(fps=30, seconds=seconds)
        out.append(Walk(cohort, f"secretsubj{i}", "w1", None, 30, joints, severity, severity))
    return out


def test_cli_reports_kept_shares_by_class_and_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cc, "load_all", lambda *a, **k: (synthetic_walks(), []))
    monkeypatch.setattr(cc, "MIN_SUBJECTS_TOP_CLASS", 1)
    rc = cc.main(["--out-dir", str(tmp_path)])
    text = capsys.readouterr().out
    assert rc == 0
    assert "Rule B" in text and "1/2 (50%)" in text  # the 1.3 s walk fails
    assert "secretsubj" not in text
    table = (tmp_path / "cycle_check.csv").read_text()
    assert "secretsubj" not in table and table.count("\n") == 5
    assert len(list(tmp_path.glob("eye_check_*.png"))) == 2  # one per cohort, no UPDRS 3 here


def test_cli_fails_when_class_2_3_loses_too_many_subjects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cc, "load_all", lambda *a, **k: (synthetic_walks(), []))
    assert cc.main(["--no-plot", "--out-dir", str(tmp_path)]) == 1  # 1 subject < 10
    assert "fewer than 10 subjects" in capsys.readouterr().out


def test_cli_missing_files_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cc.main(["--data-dir", str(tmp_path), "--no-plot"]) == 2
    assert "MISSING" in capsys.readouterr().out
