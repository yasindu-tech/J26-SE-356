"""Tests for the step, joint-range and arm-swing features (GAIT-15)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import check_gait_features as cgf
import numpy as np
import pytest
from carepd_format import L_ANKLE, L_HIP, L_KNEE, PELVIS, R_ANKLE, R_HIP, R_KNEE
from carepd_loader import Walk
from cycles import Cycle, CycleCheck, segment_cycles
from feature_contract import check_feature_contract
from gait_features import (
    EXPERIMENTAL_NAMES,
    FEATURE_NAMES,
    L_SHOULDER,
    L_WRIST,
    R_SHOULDER,
    R_WRIST,
    gait_features,
    interior_angle,
    leg_length,
    thigh_angle,
)
from preprocess import detect_heel_strikes, preprocess
from test_preprocess import make_walk

FPS = 30.0


def full_walk(
    seconds: float = 8.0,
    stride_len: float = 1.2,
    knee_bend: float = 0.08,
    arm_left: float = 0.15,
    arm_right: float = 0.15,
    scale: float = 1.0,
) -> np.ndarray:
    """``make_walk`` plus knees (halfway between hip and ankle, pushed forward by ``knee_bend``)
    and arms (each wrist swings forward-back about its shoulder once per stride, opposite sides
    in opposite phase, amplitude ``arm_left`` / ``arm_right`` in metres)."""
    joints, _, _ = make_walk(fps=int(FPS), seconds=seconds, stride_len=stride_len)
    t = np.arange(joints.shape[0]) / FPS
    for hip, knee, ankle in ((L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE)):
        joints[:, knee] = (joints[:, hip] + joints[:, ankle]) / 2
        joints[:, knee, 2] += knee_bend
    swing = 2 * np.pi * t / 1.1
    for shoulder, wrist, x, amp, sign in (
        (L_SHOULDER, L_WRIST, 0.2, arm_left, 1),
        (R_SHOULDER, R_WRIST, -0.2, arm_right, -1),
    ):
        joints[:, shoulder, 0] = x
        joints[:, shoulder, 1] = joints[:, PELVIS, 1] + 0.5
        joints[:, wrist, 0] = x
        joints[:, wrist, 1] = joints[:, shoulder, 1] - 0.5
        joints[:, wrist, 2] = joints[:, shoulder, 2] + sign * amp * np.sin(swing)
    return joints * scale


def features_of(joints: np.ndarray):  # noqa: ANN201
    return gait_features(joints, FPS, segment_cycles(detect_heel_strikes(joints, FPS)))


# --------------------------------------------------------------------------- geometry
def test_interior_angle_of_known_shapes() -> None:
    a, b, c = np.array([[1.0, 0, 0]]), np.array([[0.0, 0, 0]]), np.array([[0.0, 1, 0]])
    assert interior_angle(a, b, c)[0] == pytest.approx(90.0)
    assert interior_angle(a, b, -a)[0] == pytest.approx(180.0)  # a straight leg


def test_thigh_angle_is_zero_when_vertical_and_positive_when_forward() -> None:
    facing = np.array([[0.0, 0.0, 1.0]])
    hip = np.zeros((1, 3))
    assert thigh_angle(hip, np.array([[0.0, -0.4, 0.0]]), facing)[0] == pytest.approx(0.0)
    assert thigh_angle(hip, np.array([[0.0, -0.4, 0.4]]), facing)[0] == pytest.approx(45.0)
    assert thigh_angle(hip, np.array([[0.0, -0.4, -0.4]]), facing)[0] == pytest.approx(-45.0)


def test_leg_length_is_hip_knee_plus_knee_ankle() -> None:
    p = np.zeros((10, 17, 3))
    p[:, L_HIP], p[:, R_HIP] = (0.1, 0, 0), (-0.1, 0, 0)
    p[:, L_KNEE], p[:, R_KNEE] = (0.1, -0.4, 0), (-0.1, -0.4, 0)
    p[:, L_ANKLE], p[:, R_ANKLE] = (0.1, -0.85, 0), (-0.1, -0.85, 0)
    assert leg_length(p) == pytest.approx(0.85)


# --------------------------------------------------------------------------- features
def test_features_of_a_regular_walk_are_sensible() -> None:
    f = features_of(full_walk())
    assert 0.9 < f.leg_length_m < 1.15  # synthetic pelvis height 0.95 m, knees bent forward
    assert f.step_length_norm * f.leg_length_m == pytest.approx(0.6, abs=0.1)  # half a stride
    assert f.step_length_asymmetry < 0.15  # symmetric walk, only frame rounding
    assert f.arm_swing_asymmetry < 0.1
    assert 0 < f.knee_flexion_range_deg < 60 and 0 < f.hip_flexion_range_deg < 90
    assert f.arm_swing_range_norm * f.leg_length_m == pytest.approx(0.3, abs=0.04)  # 2 x 0.15 m


def test_every_length_feature_is_in_leg_lengths_and_does_not_change_with_body_size() -> None:
    small, big = features_of(full_walk(scale=0.8)), features_of(full_walk(scale=1.25))
    assert big.leg_length_m / small.leg_length_m == pytest.approx(1.25 / 0.8, rel=1e-3)
    for name in ("step_length_norm", "arm_swing_range_norm", "step_width_norm"):
        assert getattr(big, name) == pytest.approx(getattr(small, name), rel=0.02)
    for name in ("knee_flexion_range_deg", "hip_flexion_range_deg", "step_length_asymmetry"):
        assert getattr(big, name) == pytest.approx(getattr(small, name), rel=0.02, abs=1e-3)


def test_bigger_arm_swing_and_longer_steps_give_bigger_numbers() -> None:
    base = features_of(full_walk())
    assert features_of(full_walk(arm_left=0.05, arm_right=0.05)).arm_swing_range_norm < (
        base.arm_swing_range_norm * 0.5
    )
    assert features_of(full_walk(stride_len=0.8)).step_length_norm < base.step_length_norm


def test_knee_flexion_range_grows_with_the_knee_bend() -> None:
    straight = features_of(full_walk(knee_bend=0.0)).knee_flexion_range_deg
    bent = features_of(full_walk(knee_bend=0.15)).knee_flexion_range_deg
    assert bent > straight + 3


def test_arm_asymmetry_is_zero_for_equal_arms_and_large_for_one_still_arm() -> None:
    assert features_of(full_walk()).arm_swing_asymmetry < 0.1
    one_arm = features_of(full_walk(arm_left=0.15, arm_right=0.02))
    assert one_arm.arm_swing_asymmetry > 1.0  # |0.30 - 0.04| / mean 0.17
    # the model input (mean of both arms) drops, the asymmetry stays out of the model inputs
    assert "arm_swing_asymmetry" not in one_arm.as_dict()


def test_features_do_not_depend_on_where_the_walk_is_in_the_room() -> None:
    joints = full_walk()
    moved = joints + np.array([3.0, 0.2, -5.0])
    a, b = features_of(joints).as_dict(), features_of(moved).as_dict()
    for name in FEATURE_NAMES:
        assert b[name] == pytest.approx(a[name], rel=1e-6, abs=1e-9)


def test_only_real_strides_are_used() -> None:
    joints = full_walk()
    check = segment_cycles(detect_heel_strikes(joints, FPS))
    base = gait_features(joints, FPS, check)
    junk = Cycle("left", 0, 10, 0.33, 0.0, False, False)  # not a stride
    noisy = CycleCheck(check.cycles + (junk,), check.min_strides)
    again = gait_features(joints, FPS, noisy)
    assert again.n_strides == base.n_strides
    assert again.as_dict() == base.as_dict()


# --------------------------------------------------------------------------- errors
def test_no_real_strides_is_an_error() -> None:
    joints = full_walk()
    with pytest.raises(ValueError, match="no real strides"):
        gait_features(joints, FPS, CycleCheck((), 3))


def test_strides_on_one_foot_only_cannot_give_asymmetry() -> None:
    joints = full_walk()
    check = segment_cycles(detect_heel_strikes(joints, FPS))
    left_only = CycleCheck(tuple(c for c in check.cycles if c.foot == "left"), 3)
    with pytest.raises(ValueError, match="one foot only"):
        gait_features(joints, FPS, left_only)


def test_collapsed_leg_joints_are_an_error() -> None:
    joints = full_walk()
    check = segment_cycles(detect_heel_strikes(joints, FPS))
    flat = joints.copy()
    for j in (L_HIP, L_KNEE, L_ANKLE, R_HIP, R_KNEE, R_ANKLE):
        flat[:, j] = flat[:, PELVIS]
    flat[:, L_HIP, 0], flat[:, R_HIP, 0] = 0.01, -0.01  # keep a facing direction
    with pytest.raises(ValueError, match="implausible"):
        gait_features(flat, FPS, check)


# --------------------------------------------------------------------------- contract
def test_feature_names_pass_the_feature_contract() -> None:
    check_feature_contract(FEATURE_NAMES, label_name=["UPDRS_GAIT", "severity"])


def test_only_the_five_model_features_are_inputs() -> None:
    f = features_of(full_walk())
    assert set(f.as_dict()) == set(FEATURE_NAMES) and len(FEATURE_NAMES) == 5
    assert set(f.experimental_dict()) == set(EXPERIMENTAL_NAMES)
    assert not set(FEATURE_NAMES) & set(EXPERIMENTAL_NAMES)
    for banned in ("n_strides", "leg_length_m"):
        assert banned not in f.as_dict() and banned not in FEATURE_NAMES


def test_preprocessed_walk_is_what_the_features_use() -> None:
    joints = full_walk()
    assert leg_length(preprocess(joints, FPS)) == pytest.approx(features_of(joints).leg_length_m)


# --------------------------------------------------------------------------- command line
def synthetic_walks() -> list[Walk]:
    out = []
    # (cohort, severity, arm amplitude, stride length): class 2-3 walks smaller, as in PD
    spec = (
        ("3DGait", 0, 0.15, 1.2),
        ("3DGait", 2, 0.05, 0.8),
        ("PD-GaM", 0, 0.15, 1.2),
        ("PD-GaM", 2, 0.05, 0.8),
    )
    for i, (cohort, severity, arm, stride_len) in enumerate(spec):
        joints = full_walk(arm_left=arm, arm_right=arm, stride_len=stride_len)
        out.append(Walk(cohort, f"secretsubj{i}", "w1", None, 30, joints, severity, severity))
    joints = full_walk(seconds=1.3)
    out.append(Walk("PD-GaM", "secretsubj9", "w1", None, 30, joints, 0, 0))  # fails the cycle check
    return out


def test_cli_reports_features_and_writes_the_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cgf, "load_all", lambda *a, **k: (synthetic_walks(), None))
    monkeypatch.setattr(cgf, "MIN_WALKS_PER_CLASS", 1)
    rc = cgf.main(["--out-dir", str(tmp_path), "--no-rank-gate"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "arm swing" in out and "NOT model inputs" in out and "secretsubj" not in out
    rows = (tmp_path / "gait_features.csv").read_text().splitlines()
    assert len(rows) == 1 + 4  # the 1.3 s clip fails the cycle check and gets no features
    assert "step_width_norm" in rows[0] and "n_strides" in rows[0]


def test_cli_fails_when_a_feature_goes_the_wrong_way(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    walks = synthetic_walks()
    walks = [replace(w, severity=2 - w.severity) for w in walks]  # class 2-3 now walks "better"
    monkeypatch.setattr(cgf, "load_all", lambda *a, **k: (walks, None))
    monkeypatch.setattr(cgf, "MIN_WALKS_PER_CLASS", 1)
    assert cgf.main(["--out-dir", str(tmp_path), "--no-rank-gate"]) == 1
    assert "wrong way" in capsys.readouterr().out


def test_cli_missing_files_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cgf.main(["--data-dir", str(tmp_path)]) == 2
    assert "MISSING" in capsys.readouterr().out
