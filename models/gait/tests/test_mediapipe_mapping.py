"""Tests for the MediaPipe -> 17-joint mapping (GAIT-16). Synthetic landmarks with known answers."""

from __future__ import annotations

from pathlib import Path

import check_mediapipe_mapping as cmm
import mediapipe_mapping as mm
import numpy as np
import pytest
from carepd_format import L_ANKLE, L_HIP, L_KNEE, PELVIS, R_ANKLE, R_HIP, R_KNEE
from cycles import segment_cycles
from gait_features import L_SHOULDER, L_WRIST, R_SHOULDER, R_WRIST, gait_features
from mediapipe_mapping import (
    AXES,
    MappedWalk,
    Segment,
    facing_agreement,
    fill_gaps,
    map_landmarks,
    usable_segments,
    walks_from_trace,
)
from pose_probe import PoseTrace, VideoInfo
from preprocess import detect_heel_strikes
from test_gait_features import full_walk
from timing_features import timing_features

FPS = 30.0


def mediapipe_world(joints: np.ndarray, heel_back: float = 0.05, toe_ahead: float = 0.12):
    """MediaPipe-style world landmarks (frames, 33, 3) for a walk in h36m-style joints.

    Hip-centred, in MediaPipe's own axes (y down). Feet point forward (+z in the h36m frame),
    toes ahead of the heels.
    """
    j = joints - ((joints[:, L_HIP] + joints[:, R_HIP]) / 2)[:, None, :]
    w = np.zeros((j.shape[0], 33, 3))
    head = j[:, 10] + np.array([0.0, 0.2, 0.0])
    w[:, 0] = head
    w[:, 7], w[:, 8] = head + [0.07, 0, 0], head - [0.07, 0, 0]  # ears, left = +x
    w[:, 11], w[:, 12] = j[:, L_SHOULDER], j[:, R_SHOULDER]
    w[:, 15], w[:, 16] = j[:, L_WRIST], j[:, R_WRIST]
    w[:, 13], w[:, 14] = (w[:, 11] + w[:, 15]) / 2, (w[:, 12] + w[:, 16]) / 2
    w[:, 23], w[:, 24] = j[:, L_HIP], j[:, R_HIP]
    w[:, 25], w[:, 26] = j[:, L_KNEE], j[:, R_KNEE]
    w[:, 27], w[:, 28] = j[:, L_ANKLE], j[:, R_ANKLE]
    w[:, 29], w[:, 30] = w[:, 27] - [0, 0, heel_back], w[:, 28] - [0, 0, heel_back]
    w[:, 31], w[:, 32] = w[:, 27] + [0, 0, toe_ahead], w[:, 28] + [0, 0, toe_ahead]
    return w * AXES  # to MediaPipe's axes (the rotation is its own inverse)


def trace_of(world: np.ndarray, detected: np.ndarray | None = None, fps: float = FPS) -> PoseTrace:
    n = world.shape[0]
    det = np.ones(n, dtype=bool) if detected is None else detected
    w = world.copy()
    w[~det] = np.nan
    vis = np.where(det[:, None], 1.0, np.nan) * np.ones((n, 33))
    return PoseTrace(fps, 1920, 1080, det, vis, np.zeros((n, 33, 2)), w)


# --------------------------------------------------------------------------- axes
def test_axes_are_a_rotation_not_a_mirror() -> None:
    assert np.linalg.det(np.diag(AXES)) == pytest.approx(1.0)  # a mirror would be -1
    assert tuple(AXES) == (1.0, -1.0, -1.0)


def test_mapping_recovers_the_original_joints() -> None:
    joints = full_walk()
    mapped = map_landmarks(mediapipe_world(joints))
    centred = joints - joints[:, PELVIS : PELVIS + 1]
    assert mapped.shape == (joints.shape[0], 17, 3)
    np.testing.assert_allclose(mapped[:, PELVIS], 0, atol=1e-12)
    for joint in (L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, 11, 13, 14, 16):
        np.testing.assert_allclose(mapped[:, joint], centred[:, joint], atol=1e-9)


def test_left_stays_left_and_up_is_up() -> None:
    mapped = map_landmarks(mediapipe_world(full_walk()))
    assert (mapped[:, L_HIP, 0] > mapped[:, R_HIP, 0]).all()  # +x is the subject's left
    assert (mapped[:, 10, 1] > mapped[:, L_ANKLE, 1]).all()  # head above ankle: y is up


def test_facing_agreement_is_one_for_the_right_axes_and_zero_for_a_mirror() -> None:
    world = mediapipe_world(full_walk())
    assert facing_agreement(world) == pytest.approx(1.0)
    mirrored = world * np.array([1.0, -1.0, 1.0]) * AXES  # y flipped only: the trap
    assert facing_agreement(mirrored) == pytest.approx(0.0)
    assert facing_agreement(world * AXES) == pytest.approx(0.0)  # axes left unconverted twice


def test_facing_agreement_does_not_depend_on_the_walking_direction() -> None:
    world = mediapipe_world(full_walk())
    turned = world.copy()  # walk the other way: rotate 180 degrees about the vertical axis
    turned[:, :, 0] *= -1
    turned[:, :, 2] *= -1
    assert facing_agreement(turned) == pytest.approx(1.0)


def test_facing_agreement_ignores_missing_frames_and_handles_none() -> None:
    world = mediapipe_world(full_walk())
    world[5:20] = np.nan
    assert facing_agreement(world) == pytest.approx(1.0)
    assert np.isnan(facing_agreement(np.full((10, 33, 3), np.nan)))


def test_bad_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="33, 3"):
        map_landmarks(np.zeros((10, 17, 3)))


def test_nan_frames_stay_nan_after_mapping() -> None:
    world = mediapipe_world(full_walk())
    world[3] = np.nan
    mapped = map_landmarks(world)
    assert np.isnan(mapped[3]).all() and np.isfinite(mapped[4]).all()


# --------------------------------------------------------------------------- end to end
def test_a_mapped_walk_gives_the_same_features_as_the_original() -> None:
    joints = full_walk()
    mapped = map_landmarks(mediapipe_world(joints))
    a = segment_cycles(detect_heel_strikes(joints, FPS))
    b = segment_cycles(detect_heel_strikes(mapped, FPS))
    assert a.strides() == b.strides() > 3
    assert timing_features(b).stride_time_mean == pytest.approx(timing_features(a).stride_time_mean)
    fa, fb = gait_features(joints, FPS, a).as_dict(), gait_features(mapped, FPS, b).as_dict()
    for name, value in fa.items():
        assert fb[name] == pytest.approx(value, rel=1e-6, abs=1e-9)
    assert fb["step_length_norm"] > 0.1  # positive: forward is forward


def test_a_mirrored_mapping_cannot_give_sensible_step_lengths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reason for the orientation check. On real clips (11 Oct 2026) the step length was
    +0.39 with the rotation and -0.38 with only y flipped: no error, just wrong. On a clean
    synthetic walk the mirror finds no real strides at all. Either way it is not +0.4."""
    world = mediapipe_world(full_walk())

    def step_length() -> float:
        joints = map_landmarks(world)
        check = segment_cycles(detect_heel_strikes(joints, FPS))
        try:
            return gait_features(joints, FPS, check).step_length_norm
        except ValueError:  # no real strides
            return float("-inf")

    right = step_length()
    monkeypatch.setattr(mm, "AXES", np.array([1.0, -1.0, 1.0]))  # y flipped only: a mirror
    assert right > 0.1 and step_length() < 0.0


# --------------------------------------------------------------------------- usable stretches
def flags(pattern: str) -> np.ndarray:
    return np.array([c == "1" for c in pattern])


def test_short_gaps_are_bridged_long_gaps_split() -> None:
    fps = 10.0  # 0.5 s = 5 frames
    pattern = "1" * 30 + "0" * 5 + "1" * 30 + "0" * 6 + "1" * 30
    segs = usable_segments(flags(pattern), fps, min_s=2.0)
    assert segs == [Segment(0, 65), Segment(71, 101)]


def test_gaps_at_the_ends_are_not_bridged_and_short_stretches_are_dropped() -> None:
    fps = 10.0
    segs = usable_segments(flags("0" * 3 + "1" * 30 + "0" * 6 + "1" * 10), fps, min_s=2.0)
    assert segs == [Segment(3, 33)]  # the 10-frame stretch is shorter than 2 s
    assert usable_segments(flags(""), fps) == []
    assert usable_segments(flags("0" * 50), fps) == []


def test_fill_gaps_interpolates_linearly() -> None:
    w = np.zeros((10, 33, 3))
    w[:, :, 0] = np.arange(10)[:, None] * 2.0  # a ramp
    ok = np.ones(10, dtype=bool)
    ok[3:6] = False
    w[3:6] = 99.0  # junk in the unusable frames
    filled = fill_gaps(w, ok)
    np.testing.assert_allclose(filled[:, 0, 0], np.arange(10) * 2.0)
    assert (w[3:6] == 99.0).all()  # the input is not changed


def test_fill_gaps_needs_usable_end_frames() -> None:
    w = np.zeros((10, 33, 3))
    ok = np.ones(10, dtype=bool)
    ok[0] = False
    with pytest.raises(ValueError, match="first and last"):
        fill_gaps(w, ok)


# --------------------------------------------------------------------------- from a trace
def test_walks_from_trace_fills_a_short_loss_and_maps_the_stretch() -> None:
    world = mediapipe_world(full_walk(seconds=10.0))
    det = np.ones(world.shape[0], dtype=bool)
    det[90:100] = False  # 0.33 s lost: bridged
    det[:15] = False  # walker not yet in view
    walks = walks_from_trace(trace_of(world, det))
    assert len(walks) == 1
    w = walks[0]
    assert w.start == 15 and w.stop == world.shape[0] and w.filled_frames == 10
    assert np.isfinite(w.joints).all() and w.orientation_ok
    assert w.seconds == pytest.approx((world.shape[0] - 15) / FPS)


def test_walks_from_trace_splits_at_a_long_loss() -> None:
    world = mediapipe_world(full_walk(seconds=12.0))
    det = np.ones(world.shape[0], dtype=bool)
    det[150:180] = False  # 1 s lost: two walks
    walks = walks_from_trace(trace_of(world, det))
    assert [(w.start, w.stop) for w in walks] == [(0, 150), (180, world.shape[0])]
    assert all(w.filled_frames == 0 for w in walks)


def test_frames_with_few_visible_legs_are_not_used() -> None:
    world = mediapipe_world(full_walk(seconds=8.0))
    trace = trace_of(world)
    trace.visibility[:60, 23:33] = 0.1  # found, but legs not visible, for the first 2 s
    walks = walks_from_trace(trace)
    assert walks[0].start == 60


def test_no_person_gives_no_walk() -> None:
    world = mediapipe_world(full_walk())
    assert walks_from_trace(trace_of(world, np.zeros(world.shape[0], dtype=bool))) == []


def test_mapped_walk_flags_a_mirrored_orientation(monkeypatch: pytest.MonkeyPatch) -> None:
    trace = trace_of(mediapipe_world(full_walk()))
    assert walks_from_trace(trace)[0].orientation_ok
    monkeypatch.setattr(mm, "AXES", np.array([1.0, -1.0, 1.0]))  # the trap: y flipped only
    walk = walks_from_trace(trace)[0]
    assert not walk.orientation_ok and isinstance(walk, MappedWalk)


# --------------------------------------------------------------------------- command line
def fake_probe(world: np.ndarray):  # noqa: ANN201
    def probe(path, model, **kwargs):  # noqa: ANN001, ANN202
        info = VideoInfo(FPS, 1920, 1080, world.shape[0])
        return info, trace_of(world)

    return probe


def clip_files(tmp_path: Path) -> list[str]:
    model = tmp_path / "model.task"
    clip = tmp_path / "secretname.mov"
    model.write_bytes(b"x")
    clip.write_bytes(b"x")
    return ["--model", str(model), "--clip", f"normal={clip}", "--out-dir", str(tmp_path / "out")]


def test_cli_reports_a_walk_and_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cmm, "probe_video", fake_probe(mediapipe_world(full_walk(seconds=10.0))))
    rc = cmm.main(clip_files(tmp_path))
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "normal" in out and "ok" in out and "secretname" not in out
    assert (tmp_path / "out" / "eye_check_normal.png").stat().st_size > 1000
    rows = (tmp_path / "out" / "mapped_clip_features.csv").read_text().splitlines()
    assert len(rows) == 2 and "step_length_norm" in rows[0]
    assert list((tmp_path / "out").glob("trace_*.npz"))  # the landmarks were cached


def test_cli_uses_the_cache_on_a_second_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    world = mediapipe_world(full_walk(seconds=10.0))
    monkeypatch.setattr(cmm, "probe_video", fake_probe(world))
    assert cmm.main([*clip_files(tmp_path), "--no-plot"]) == 0

    def boom(*a, **k):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("MediaPipe must not run again")

    monkeypatch.setattr(cmm, "probe_video", boom)
    assert cmm.main([*clip_files(tmp_path), "--no-plot"]) == 0
    capsys.readouterr()


def test_cli_fails_on_a_mirrored_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cmm, "probe_video", fake_probe(mediapipe_world(full_walk(seconds=10.0))))
    monkeypatch.setattr(mm, "AXES", np.array([1.0, -1.0, 1.0]))  # the trap: y flipped only
    assert cmm.main([*clip_files(tmp_path), "--no-plot"]) == 1
    assert "mirrored" in capsys.readouterr().out


def test_cli_says_when_there_is_no_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    world = mediapipe_world(full_walk(seconds=4.0))
    world[:] = np.nan

    def probe(path, model, **kwargs):  # noqa: ANN001, ANN202
        det = np.zeros(world.shape[0], dtype=bool)
        return VideoInfo(FPS, 1920, 1080, world.shape[0]), trace_of(world, det)

    monkeypatch.setattr(cmm, "probe_video", probe)
    assert cmm.main([*clip_files(tmp_path), "--no-plot"]) == 0
    assert "no usable stretch" in capsys.readouterr().out


def test_cli_missing_files_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cmm.main(["--data-dir", str(tmp_path), "--model", str(tmp_path / "none.task")]) == 2
    assert "MISSING" in capsys.readouterr().out


def test_module_constants_agree_with_the_pose_probe() -> None:
    assert mm.MIN_FACING_AGREEMENT == 0.9 and mm.MIN_WALK_S >= 2.0
    assert mm.MAX_GAP_S == 0.5
