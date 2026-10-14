"""Tests for the single walk -> eight features function and the feature table (GAIT-17)."""

from __future__ import annotations

import csv
import dataclasses
import math
from pathlib import Path

import build_feature_table as bft
import gait_extract as ge
import mediapipe_mapping as mm
import numpy as np
import pytest
from carepd_loader import Walk
from cycles import KOA_MIN_STRIDES, Cycle, CycleCheck, segment_cycles
from gait_extract import FEATURE_NAMES, extract_features, extract_from_trace, lr_difference
from gait_features import gait_features
from preprocess import detect_heel_strikes
from test_gait_features import full_walk
from test_mediapipe_mapping import mediapipe_world, trace_of
from test_preprocess import make_walk
from timing_features import timing_features

FPS = 30.0


def cycle(foot: str, start: int, seconds: float) -> Cycle:
    end = start + round(seconds * 30)
    return Cycle(foot, start, end, seconds, 0.4, True, True)  # type: ignore[arg-type]


def check_with(left: list[float], right: list[float]) -> CycleCheck:
    cycles, t = [], 0
    for foot, times in (("left", left), ("right", right)):
        for s in times:
            cycles.append(cycle(foot, t, s))
            t += round(s * 30)
    return CycleCheck(tuple(cycles), 3)


# --------------------------------------------------------------------------- the features
def test_the_eight_names_are_in_the_documented_order() -> None:
    assert FEATURE_NAMES == (
        "stride_time_mean",
        "cadence",
        "stride_time_cv",
        "step_length_norm",
        "step_length_asymmetry",
        "knee_flexion_range_deg",
        "hip_flexion_range_deg",
        "arm_swing_range_norm",
    )


def test_extract_features_equals_the_hand_assembled_gait14_and_gait15_features() -> None:
    joints = full_walk()
    result = extract_features(joints, FPS)
    check = segment_cycles(detect_heel_strikes(joints, FPS), 3)
    by_hand = {**timing_features(check).as_dict(), **gait_features(joints, FPS, check).as_dict()}
    assert result.ok and result.reason == "" and result.flags == ()
    assert tuple(result.features) == FEATURE_NAMES
    for name in FEATURE_NAMES:
        assert result.features[name] == pytest.approx(by_hand[name])
    np.testing.assert_allclose(result.vector(), [by_hand[n] for n in FEATURE_NAMES])
    assert set(result.experimental) == {"step_width_norm", "arm_swing_asymmetry"}


def test_checks_are_returned_but_are_not_model_inputs() -> None:
    result = extract_features(full_walk(), FPS)
    assert result.checks is not None
    assert result.checks.n_strides >= 3 and result.checks.n_left > 0 and result.checks.n_right > 0
    assert 0.9 < result.checks.leg_length_m < 1.15
    assert "n_strides" not in FEATURE_NAMES and "leg_length_m" not in FEATURE_NAMES


# --------------------------------------------------------------------------- rejection
def test_a_short_walk_is_rejected_with_a_reason_not_an_error() -> None:
    result = extract_features(full_walk(seconds=1.3), FPS)
    assert not result.ok and result.status == "rejected"
    assert "real strides" in result.reason and result.features == {}
    with pytest.raises(ValueError, match="rejected"):
        result.vector()


def test_strides_on_one_foot_only_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    one_foot = check_with([1.1] * 4, [])
    monkeypatch.setattr(ge, "segment_cycles", lambda *a, **k: one_foot)
    result = extract_features(full_walk(), FPS)
    assert not result.ok and "one foot" in result.reason
    assert result.checks is not None and result.checks.n_right == 0


def test_collapsed_legs_are_rejected() -> None:
    result = extract_features(full_walk(scale=0.1), FPS)
    assert not result.ok
    assert "leg length" in result.reason or "real strides" in result.reason


@pytest.mark.parametrize("fps", [10.0, 100.0])
def test_a_frame_rate_outside_the_range_raises(fps: float) -> None:
    with pytest.raises(ValueError, match="frame rate"):
        extract_features(full_walk(), fps)


def test_bad_input_raises() -> None:
    with pytest.raises(ValueError):
        extract_features(np.zeros((100, 5, 3)), FPS)
    joints = full_walk()
    joints[10, 3, 1] = np.nan
    with pytest.raises(ValueError):
        extract_features(joints, FPS)
    with pytest.raises(ValueError, match="min_strides"):
        extract_features(full_walk(), FPS, min_strides=1)


# --------------------------------------------------------------------------- left/right flag
def test_lr_difference_is_zero_for_equal_feet() -> None:
    assert lr_difference(check_with([1.1] * 3, [1.1] * 3)) == pytest.approx(0.0)


def test_lr_difference_is_a_share_of_the_mean() -> None:
    # medians 1.0 and 1.5 -> |diff| / mean = 0.5 / 1.25
    assert lr_difference(check_with([1.0] * 3, [1.5] * 3)) == pytest.approx(0.4)


def test_lr_difference_is_not_assessed_with_fewer_than_three_strides_a_foot() -> None:
    assert math.isnan(lr_difference(check_with([1.0] * 3, [1.5] * 2)))
    assert math.isnan(lr_difference(check_with([1.0] * 2, [1.5] * 2)))


def test_a_large_lr_difference_is_flagged_but_changes_nothing_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    joints = full_walk()
    plain = extract_features(joints, FPS)
    monkeypatch.setattr(ge, "lr_difference", lambda check: 0.4)
    flagged = extract_features(joints, FPS)
    assert flagged.ok and len(flagged.flags) == 1 and "40%" in flagged.flags[0]
    assert flagged.features == plain.features


def test_a_small_or_unassessed_difference_is_not_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    joints = full_walk()
    for value in (0.2, math.nan):
        monkeypatch.setattr(ge, "lr_difference", lambda check, v=value: v)
        assert extract_features(joints, FPS).flags == ()


# --------------------------------------------------------------------------- video
def test_video_trace_gives_the_same_features_as_the_walk_itself() -> None:
    joints = full_walk(seconds=10.0)
    result, walk = extract_from_trace(trace_of(mediapipe_world(joints)))
    assert result.ok and walk is not None and walk.orientation_ok
    direct = extract_features(walk.joints, walk.fps, min_strides=KOA_MIN_STRIDES)
    assert result.features == direct.features
    assert result.features["step_length_norm"] > 0


def test_video_without_a_person_is_rejected() -> None:
    world = mediapipe_world(full_walk(seconds=6.0))
    result, walk = extract_from_trace(trace_of(world, np.zeros(world.shape[0], dtype=bool)))
    assert not result.ok and walk is None and "no usable stretch" in result.reason


def test_a_mirrored_mapping_is_rejected_not_silently_negative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = mediapipe_world(full_walk(seconds=10.0))
    monkeypatch.setattr(mm, "AXES", np.array([1.0, -1.0, 1.0]))  # y flipped only: a mirror
    result, walk = extract_from_trace(trace_of(world))
    assert not result.ok and walk is not None and "orientation" in result.reason


# --------------------------------------------------------------------------- table builder
def synthetic_walks() -> list[Walk]:
    out = []
    spec = (("3DGait", 0, 8.0), ("3DGait", 1, 8.0), ("PD-GaM", 2, 8.0), ("PD-GaM", 0, 1.3))
    for i, (cohort, severity, seconds) in enumerate(spec):
        joints, _, _ = make_walk(fps=30, seconds=seconds)
        out.append(Walk(cohort, f"secretsubj{i}", "w1", None, 30, joints, severity, severity))
    return out


def test_cli_writes_both_tables_and_never_prints_subject_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(bft, "load_all", lambda *a, **k: (synthetic_walks(), None))
    assert bft.main(["--out-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "secretsubj" not in out and "rejected" in out
    with (tmp_path / "feature_table.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3  # the 1.3 s clip is rejected
    assert list(rows[0])[:5] == bft.META
    assert all(n in rows[0] for n in FEATURE_NAMES)
    assert all(math.isfinite(float(r[n])) for r in rows for n in FEATURE_NAMES)
    assert "secretsubj" not in (tmp_path / "feature_table.csv").read_text()
    rejected = (tmp_path / "rejected_walks.csv").read_text()
    assert "secretsubj" not in rejected and "real strides" in rejected


def test_cli_leaves_unassessed_lr_difference_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bft, "load_all", lambda *a, **k: (synthetic_walks()[:1], None))
    monkeypatch.setattr(ge, "lr_difference", lambda check: math.nan)
    assert bft.main(["--out-dir", str(tmp_path)]) == 0
    with (tmp_path / "feature_table.csv").open() as fh:
        row = next(csv.DictReader(fh))
    assert row["lr_difference"] == "" and row["lr_flag"] == "0"


def test_cli_fails_on_a_non_finite_feature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real = bft.extract_features

    def broken(*a, **k):  # type: ignore[no-untyped-def]
        r = real(*a, **k)
        return dataclasses.replace(r, features={**r.features, "cadence": math.nan})

    monkeypatch.setattr(bft, "load_all", lambda *a, **k: (synthetic_walks()[:1], None))
    monkeypatch.setattr(bft, "extract_features", broken)
    assert bft.main(["--out-dir", str(tmp_path)]) == 1
    assert "non-finite" in capsys.readouterr().out


def test_cli_missing_files_exit_2(tmp_path: Path) -> None:
    assert bft.main(["--data-dir", str(tmp_path), "--out-dir", str(tmp_path)]) == 2
