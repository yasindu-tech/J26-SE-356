"""Tests for video to joints (GAIT-25). A fake MediaPipe trace stands in for the video."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import check_video_to_joints as cvj
import numpy as np
import pytest
import video_to_joints as vj
from mediapipe_mapping import walks_from_trace
from pose_probe import PoseTrace, VideoInfo
from test_gait_features import full_walk
from test_mediapipe_mapping import FPS, mediapipe_world, trace_of
from video_to_joints import VideoError, video_to_joints


def make_trace(seconds: float = 8.0, lost: tuple[int, int] | None = None) -> PoseTrace:
    world = mediapipe_world(full_walk(seconds=seconds))
    det = np.ones(world.shape[0], dtype=bool)
    if lost:
        det[lost[0] : lost[1]] = False
    return trace_of(world, det)


def cut(trace: PoseTrace, frames: int) -> PoseTrace:
    return PoseTrace(
        trace.fps,
        trace.width,
        trace.height,
        trace.detected[:frames],
        trace.visibility[:frames],
        trace.image[:frames],
        trace.world[:frames],
    )


class Fake:
    """Replaces the three calls that touch a real video, and records the time limit asked for."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, trace: PoseTrace) -> None:
        self.asked: list[float | None] = []
        self.trace = trace
        self.frames = trace.n_frames
        self.reported = (trace.width, trace.height)  # what the file says its size is
        self.real = (trace.width, trace.height)  # the size of its frames
        self.video = tmp_path / "secret_name.mov"
        self.video.write_bytes(b"not a real video")
        self.model = tmp_path / "pose.task"
        self.model.write_bytes(b"not a real model")
        monkeypatch.setattr(vj, "read_info", lambda _p: self.info())
        monkeypatch.setattr(vj, "read_rgb_frames", self.read_frames)
        monkeypatch.setattr(vj, "probe_video", self.probe)

    def info(self) -> VideoInfo:
        return VideoInfo(self.trace.fps, *self.reported, self.frames)

    def read_frames(self, _path: Path) -> Iterator[np.ndarray]:
        yield np.zeros((self.real[1], self.real[0], 3), dtype=np.uint8)

    def probe(
        self, _path: Path, _model: Path, *, max_seconds: float | None = None
    ) -> tuple[VideoInfo, PoseTrace]:
        self.asked.append(max_seconds)
        trace = self.trace
        if max_seconds is not None:
            trace = cut(trace, int(max_seconds * trace.fps))
        return self.info(), trace


def run(fake: Fake, **kwargs: float | None) -> vj.VideoWalk:
    return video_to_joints(fake.video, fake.model, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- the walk
def test_the_walk_is_the_one_the_gait_16_path_gives_and_the_facts_are_filled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trace = make_trace(8.0, lost=(90, 100))  # a 0.33 s loss is bridged
    walk = run(Fake(monkeypatch, tmp_path, trace))
    ref = walks_from_trace(trace)[0]
    np.testing.assert_array_equal(walk.joints, ref.joints)
    assert walk.joints.shape == (240, 17, 3) and walk.fps == FPS
    assert (walk.start_s, walk.end_s, walk.seconds) == (0.0, 8.0, 8.0)
    assert walk.video_seconds == 8.0 and walk.analysed_seconds == 8.0 and not walk.truncated
    assert (walk.width, walk.height) == (1920, 1080)
    assert walk.person_found == pytest.approx(230 / 240)
    assert walk.stretches == 1 and walk.unused_seconds == 0.0
    assert walk.filled_frames == 10 and walk.facing_agreement == 1.0
    assert walk.leg_visibility == pytest.approx(230 / 240)  # the 10 lost frames count as unclear


def test_leg_visibility_counts_landmarks_that_are_not_clearly_visible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trace = make_trace(8.0)
    trace.visibility[:, 31:33] = 0.1  # both toes unclear: 8 of the 10 leg landmarks are clear
    walk = run(Fake(monkeypatch, tmp_path, trace))
    assert walk.leg_visibility == pytest.approx(0.8)


def test_leg_visibility_counts_only_the_walk_not_the_rest_of_the_clip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trace = make_trace(12.0, lost=(150, 180))  # the walk is the second stretch, frames 180 to 360
    trace.visibility[:150, 23:33] = 0.1  # the first stretch (not the walk) has unclear legs
    walk = run(Fake(monkeypatch, tmp_path, trace))
    assert walk.start_s == 6.0 and walk.leg_visibility == 1.0


def test_the_longest_stretch_is_the_walk_and_the_rest_is_reported_as_unused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trace = make_trace(12.0, lost=(150, 180))  # 5 s, a 1 s loss, then 6 s
    walk = run(Fake(monkeypatch, tmp_path, trace))
    assert walk.stretches == 2
    assert (walk.start_s, walk.end_s) == (6.0, 12.0)
    assert walk.seconds == 6.0 and walk.unused_seconds == pytest.approx(5.0)


def test_the_first_stretch_wins_when_it_is_the_longer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trace = make_trace(12.0, lost=(210, 240))  # 7 s, a 1 s loss, then 4 s
    walk = run(Fake(monkeypatch, tmp_path, trace))
    assert (walk.start_s, walk.end_s) == (0.0, 7.0) and walk.unused_seconds == pytest.approx(4.0)


# --------------------------------------------------------------------------- the time limit
def test_the_default_limit_is_30_s_and_can_be_changed_or_removed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = Fake(monkeypatch, tmp_path, make_trace(8.0))
    run(fake)
    run(fake, max_seconds=5.0)
    run(fake, max_seconds=None)
    assert fake.asked == [vj.MAX_SECONDS, 5.0, None] and vj.MAX_SECONDS == 30.0


def test_a_clip_longer_than_the_limit_is_flagged_as_cut(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    walk = run(Fake(monkeypatch, tmp_path, make_trace(12.0)), max_seconds=6.0)
    assert walk.truncated and walk.analysed_seconds == 6.0 and walk.video_seconds == 12.0
    assert walk.end_s == 6.0


@pytest.mark.parametrize(("extra", "cut_flag"), [(0, False), (1, False), (2, False), (3, True)])
def test_a_frame_count_off_by_a_frame_or_two_is_not_a_cut(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extra: int, cut_flag: bool
) -> None:
    fake = Fake(monkeypatch, tmp_path, make_trace(8.0))
    fake.frames = fake.trace.n_frames + extra
    assert run(fake, max_seconds=None).truncated is cut_flag


# --------------------------------------------------------------------------- the video cannot be used
def test_missing_or_empty_files_are_refused(tmp_path: Path) -> None:
    model = tmp_path / "pose.task"
    model.write_bytes(b"x")
    with pytest.raises(VideoError, match="missing or empty"):
        video_to_joints(tmp_path / "nope.mov", model)
    empty = tmp_path / "empty.mov"
    empty.write_bytes(b"")
    with pytest.raises(VideoError, match="missing or empty"):
        video_to_joints(empty, model)
    video = tmp_path / "v.mov"
    video.write_bytes(b"x")
    with pytest.raises(VideoError, match="pose model"):
        video_to_joints(video, tmp_path / "nope.task")


def test_a_video_that_cannot_be_opened_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = Fake(monkeypatch, tmp_path, make_trace())

    def broken(_p: Path) -> VideoInfo:
        raise ValueError("could not open the video")

    monkeypatch.setattr(vj, "read_info", broken)
    with pytest.raises(VideoError, match="could not be opened"):
        run(fake)


@pytest.mark.parametrize("fps", [0.0, -1.0])
def test_a_video_without_a_frame_rate_or_frames_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fps: float
) -> None:
    fake = Fake(monkeypatch, tmp_path, make_trace())
    monkeypatch.setattr(vj, "read_info", lambda _p: VideoInfo(fps, 1920, 1080, 240))
    with pytest.raises(VideoError, match="no frame rate"):
        run(fake)
    monkeypatch.setattr(vj, "read_info", lambda _p: VideoInfo(30.0, 1920, 1080, 0))
    with pytest.raises(VideoError, match="no frame rate or no frames"):
        run(fake)


def test_frames_of_another_size_than_the_file_reports_are_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = Fake(monkeypatch, tmp_path, make_trace())
    fake.real = (1080, 1920)  # a phone video whose rotation was not applied
    with pytest.raises(VideoError, match="rotation was not applied"):
        run(fake)
    assert fake.asked == []  # refused before the slow step


def test_a_video_with_no_readable_frame_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = Fake(monkeypatch, tmp_path, make_trace())

    def nothing(_p: Path) -> Iterator[np.ndarray]:
        return
        yield

    monkeypatch.setattr(vj, "read_rgb_frames", nothing)
    with pytest.raises(VideoError, match="no frame could be read"):
        run(fake)


def test_a_failure_inside_the_pose_step_becomes_a_video_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = Fake(monkeypatch, tmp_path, make_trace())

    def fail(*_a: object, **_k: object) -> tuple[VideoInfo, PoseTrace]:
        raise ValueError("no frames could be read from the video")

    monkeypatch.setattr(vj, "probe_video", fail)
    with pytest.raises(VideoError, match="no frames could be read"):
        run(fake)


def test_no_person_is_refused_and_the_share_found_is_in_the_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trace = make_trace(8.0)
    trace.detected[:] = False
    with pytest.raises(VideoError, match=r"no walk found.*0%"):
        run(Fake(monkeypatch, tmp_path, trace))
    partial = make_trace(8.0)
    partial.visibility[:, 23:33] = 0.1  # found everywhere, legs never visible
    with pytest.raises(VideoError, match=r"seen in 100% of the frames.*legs in clear view"):
        run(Fake(monkeypatch, tmp_path, partial))


def test_a_stretch_shorter_than_the_minimum_is_not_a_walk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = Fake(monkeypatch, tmp_path, make_trace(8.0))
    with pytest.raises(VideoError, match="no walk found"):
        run(fake, max_seconds=1.5)  # 1.5 s is under the 2 s minimum
    assert run(fake, max_seconds=2.5).seconds == 2.5


def test_a_bad_time_limit_is_a_programming_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = Fake(monkeypatch, tmp_path, make_trace())
    for bad in (0.0, -3.0):
        with pytest.raises(ValueError, match="positive"):
            run(fake, max_seconds=bad)


# --------------------------------------------------------------------------- command line
def test_cli_prints_the_facts_and_never_the_file_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = Fake(monkeypatch, tmp_path, make_trace(12.0, lost=(150, 180)))
    assert vj.main([str(fake.video), "--model", str(fake.model)]) == 0
    out = capsys.readouterr().out
    assert "secret_name" not in out
    assert "w-" in out and "1920x1080" in out and "6.0 s (6.0 to 12.0 s)" in out
    assert "2 usable stretch(es)" in out and "5.0 s unused" in out and "facing 100%" in out
    assert "first" not in out  # not cut


def test_cli_says_when_the_clip_was_cut_and_can_read_all_of_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = Fake(monkeypatch, tmp_path, make_trace(12.0))
    argv = [str(fake.video), "--model", str(fake.model)]
    assert vj.main([*argv, "--max-seconds", "6"]) == 0
    assert "read the first 6 s only" in capsys.readouterr().out
    assert vj.main([*argv, "--no-cap"]) == 0
    assert "read the first" not in capsys.readouterr().out
    assert fake.asked == [6.0, None]


def test_cli_saves_the_joints(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    trace = make_trace(8.0)
    fake = Fake(monkeypatch, tmp_path, trace)
    out = tmp_path / "sub" / "walk.npz"
    assert vj.main([str(fake.video), "--model", str(fake.model), "--save", str(out)]) == 0
    with np.load(out) as z:
        np.testing.assert_array_equal(z["joints"], walks_from_trace(trace)[0].joints)
        assert float(z["fps"]) == FPS


def test_cli_gives_the_reason_and_exit_2_for_an_unusable_video(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = tmp_path / "pose.task"
    model.write_bytes(b"x")
    assert vj.main([str(tmp_path / "secret_name.mov"), "--model", str(model)]) == 2
    out = capsys.readouterr().out
    assert "missing or empty" in out and "secret_name" not in out


# --------------------------------------------------------------------------- the GAIT-16 comparison script
def patch_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, trace: PoseTrace, shift: float = 0.0
) -> list[str]:
    for name in ("a.mov", "b.mov", "pose.task"):
        (tmp_path / name).write_bytes(b"x")
    ref = walks_from_trace(trace)[0]

    def fake_walk(_p: Path, _m: Path, *, max_seconds: float | None = None) -> vj.VideoWalk:
        return vj.VideoWalk(
            ref.joints + shift, ref.fps, 0, 8, 8, 8, False, 1920, 1080, 1.0, 1, 0, 0, 1, 1.0
        )

    monkeypatch.setattr(cvj, "video_to_joints", fake_walk)
    monkeypatch.setattr(cvj, "load_or_run", lambda *_a: (trace, True))
    return [
        "--model", str(tmp_path / "pose.task"),
        "--clip", f"one={tmp_path / 'a.mov'}",
        "--clip", f"two={tmp_path / 'b.mov'}",
    ]  # fmt: skip


def test_the_comparison_passes_when_the_joints_agree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = patch_check(monkeypatch, tmp_path, make_trace(8.0))
    assert cvj.main(argv) == 0
    assert capsys.readouterr().out.count("same") == 2


def test_the_comparison_fails_when_the_joints_differ_by_more_than_a_millimetre(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = patch_check(monkeypatch, tmp_path, make_trace(8.0), shift=0.01)
    assert cvj.main(argv) == 1
    assert "DIFFERENT" in capsys.readouterr().out
    argv = patch_check(monkeypatch, tmp_path, make_trace(8.0), shift=0.0005)
    assert cvj.main(argv) == 0


def test_the_comparison_fails_on_different_lengths_and_on_a_video_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = patch_check(monkeypatch, tmp_path, make_trace(8.0))
    monkeypatch.setattr(cvj, "load_or_run", lambda *_a: (make_trace(6.0), True))
    assert cvj.main(argv) == 1
    assert "DIFFERENT shapes" in capsys.readouterr().out

    def refuse(*_a: object, **_k: object) -> vj.VideoWalk:
        raise VideoError("no walk found")

    monkeypatch.setattr(cvj, "video_to_joints", refuse)
    assert cvj.main(argv) == 1
    assert "no walk found" in capsys.readouterr().out


def test_the_comparison_exits_2_for_a_missing_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cvj.main(["--model", str(tmp_path / "no.task"), "--clip", "x=" + str(tmp_path)]) == 2
    assert "MISSING" in capsys.readouterr().out
