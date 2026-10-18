"""Tests for the quality gate (GAIT-26). Synthetic walks with known answers."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import quality_gate as qg
from mediapipe_mapping import MIN_FACING_AGREEMENT, walks_from_trace
from quality_gate import GateResult, check_walk, reject
from test_gait_features import full_walk
from test_mediapipe_mapping import mediapipe_world, trace_of
from test_video_to_joints import Fake, make_trace
from video_to_joints import VideoError, VideoWalk, video_to_joints

BANNED = re.compile(r"diagnos|parkinson|disease|patient|\bpd\b|chance|indicat|cause|due to", re.I)


def good(seconds: float = 8.0, **changes: Any) -> VideoWalk:
    walk = walks_from_trace(trace_of(mediapipe_world(full_walk(seconds=seconds))))[0]
    base = VideoWalk(
        joints=walk.joints,
        fps=walk.fps,
        start_s=0.0,
        end_s=walk.seconds,
        video_seconds=walk.seconds,
        analysed_seconds=walk.seconds,
        truncated=False,
        width=1920,
        height=1080,
        person_found=1.0,
        stretches=1,
        unused_seconds=0.0,
        filled_frames=0,
        facing_agreement=1.0,
        leg_visibility=1.0,
    )
    return replace(base, **changes)


# --------------------------------------------------------------------------- a good walk
def test_a_clean_walk_passes_with_only_the_one_person_warning() -> None:
    result = check_walk(good())
    assert result.passed and result.reasons == ()
    assert result.warnings == (qg.ONE_PERSON_ASSUMED,)
    assert set(result.checks) == {
        "frame_rate",
        "leg_visibility",
        "filled_share",
        "facing",
        "strides",
    }
    assert all(c["ok"] for c in result.checks.values())
    assert result.checks["strides"]["value"] == 12 and result.checks["strides"]["limit"] == 3


def test_the_record_is_plain_json_and_copies_the_checks() -> None:
    result = check_walk(good(), people_in_view=1)
    record = result.as_record()
    assert json.loads(json.dumps(record)) == record
    assert set(record) == {"passed", "reasons", "warnings", "checks"}
    record["checks"]["strides"]["ok"] = False
    assert result.checks["strides"]["ok"] is True  # the record is a copy


# --------------------------------------------------------------------------- each limit, at and beyond it
@pytest.mark.parametrize(
    ("rule", "changes", "ok"),
    [
        ("leg_visibility", {"leg_visibility": 0.80}, True),
        ("leg_visibility", {"leg_visibility": 0.7999}, False),
        ("filled_share", {"filled_frames": 48}, True),  # 48 of 240 frames = 20%
        ("filled_share", {"filled_frames": 49}, False),
        ("facing", {"facing_agreement": MIN_FACING_AGREEMENT}, True),
        ("facing", {"facing_agreement": MIN_FACING_AGREEMENT - 0.0001}, False),
        ("frame_rate", {"fps": 20.0}, True),
        ("frame_rate", {"fps": 19.9}, False),
        ("frame_rate", {"fps": 60.0}, True),
        ("frame_rate", {"fps": 60.1}, False),
    ],
)
def test_each_limit_is_inclusive_and_fails_just_beyond_it(
    rule: str, changes: dict[str, Any], ok: bool
) -> None:
    result = check_walk(good(**changes))
    assert result.checks[rule]["ok"] is ok
    if rule not in ("frame_rate",):  # a changed frame rate also changes the step timing
        assert result.passed is ok
        assert (len(result.reasons) == 0) is ok


def test_strides_pass_at_the_limit_and_fail_below_it() -> None:
    assert check_walk(good(3.0)).passed  # 3 strides
    short = check_walk(good(2.5))  # 2 strides
    assert not short.passed
    assert short.reasons == ("The walk has 2 real strides on both feet; at least 3 are needed.",)
    assert check_walk(good(2.5), min_strides=2).passed
    assert not check_walk(good(8.0), min_strides=13).passed


@pytest.mark.parametrize(("fps", "limit"), [(18.0, 20.0), (120.0, 60.0)])
def test_a_frame_rate_outside_20_to_60_is_rejected_with_the_range_in_words(
    fps: float, limit: float
) -> None:
    result = check_walk(good(fps=fps))
    assert not result.passed and "strides" not in result.checks
    assert result.reasons == (
        f"The video has {fps:.0f} frames per second; 20 to 60 are supported.",
    )
    assert result.checks["frame_rate"] == {"value": fps, "limit": limit, "ok": False}


def test_a_frame_rate_too_low_for_the_step_filter_is_rejected_without_a_crash() -> None:
    result = check_walk(good(fps=5.0))
    assert (
        not result.passed
        and len(result.reasons) == 1
        and "5 frames per second" in result.reasons[0]
    )
    assert "strides" not in result.checks  # the steps were not looked for


def test_a_walk_with_too_few_frames_to_filter_counts_as_no_strides() -> None:
    base = good()
    result = check_walk(replace(base, joints=base.joints[:10]))
    assert not result.passed and result.checks["strides"]["value"] == 0


def test_each_reason_names_its_own_problem_and_several_can_apply_at_once() -> None:
    result = check_walk(good(leg_visibility=0.5, filled_frames=100, facing_agreement=0.5, fps=10.0))
    assert not result.passed and len(result.reasons) >= 4
    text = " ".join(result.reasons)
    for part in ("frames per second", "legs were clearly visible", "filled in", "faces"):
        assert part in text
    assert [c["ok"] for n, c in result.checks.items() if n != "strides"] == [False] * 4


# --------------------------------------------------------------------------- the person-count slot
def test_without_a_count_the_gate_warns_that_one_person_is_assumed() -> None:
    result = check_walk(good())
    assert qg.ONE_PERSON_ASSUMED in result.warnings and "people" not in result.checks


def test_a_count_of_one_removes_the_warning_and_two_rejects() -> None:
    one = check_walk(good(), people_in_view=1)
    assert one.passed and one.warnings == () and one.checks["people"]["ok"]
    two = check_walk(good(), people_in_view=2)
    assert not two.passed and two.reasons == (
        "2 people are in view; the video must show one person walking.",
    )
    assert not two.checks["people"]["ok"] and two.warnings == ()


def test_a_count_of_zero_is_not_an_objection_and_a_negative_one_is_an_error() -> None:
    assert check_walk(good(), people_in_view=0).passed
    with pytest.raises(ValueError, match="negative"):
        check_walk(good(), people_in_view=-1)


# --------------------------------------------------------------------------- warnings
def test_a_cut_clip_and_unused_stretches_are_warned_about_and_do_not_reject() -> None:
    result = check_walk(
        good(
            truncated=True,
            analysed_seconds=30.0,
            video_seconds=321.2,
            stretches=3,
            unused_seconds=4.5,
        )
    )
    assert result.passed
    assert "Only the first 30 s of the 321 s video were read." in result.warnings
    assert any("2 other stretch(es) of usable video (4.5 s)" in w for w in result.warnings)
    assert len(result.warnings) == 3  # and the one-person warning


def test_a_single_other_stretch_is_already_worth_a_warning() -> None:
    result = check_walk(good(stretches=2, unused_seconds=2.4))
    assert any("1 other stretch(es) of usable video (2.4 s)" in w for w in result.warnings)
    assert not any("other stretch" in w for w in check_walk(good(stretches=1)).warnings)


def test_a_clip_that_was_not_cut_has_no_cut_warning() -> None:
    assert not any("Only the first" in w for w in check_walk(good()).warnings)


# --------------------------------------------------------------------------- video errors
def test_an_unusable_video_becomes_a_rejection_with_the_reason() -> None:
    result = reject(VideoError("the video could not be opened (damaged, or not a video file)"))
    assert result == GateResult(
        False, reasons=("the video could not be opened (damaged, or not a video file)",)
    )
    assert result.checks == {} and result.as_record()["passed"] is False


def test_a_video_with_no_person_goes_through_video_to_joints_and_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trace = make_trace(8.0)
    trace.detected[:] = False
    fake = Fake(monkeypatch, tmp_path, trace)
    with pytest.raises(VideoError) as err:
        video_to_joints(fake.video, fake.model)
    result = reject(err.value)
    assert not result.passed and "no walk found" in result.reasons[0]


def test_a_good_fake_video_goes_through_video_to_joints_and_passes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = Fake(monkeypatch, tmp_path, make_trace(8.0))
    result = check_walk(video_to_joints(fake.video, fake.model))
    assert result.passed and result.checks["strides"]["value"] == 12


def test_a_walk_with_a_lot_filled_in_is_rejected_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trace = make_trace(10.0)
    for start in range(30, 270, 30):  # eight 0.33 s losses, each bridged: 80 of 300 frames filled
        trace.detected[start : start + 10] = False
        trace.world[start : start + 10] = np.nan
    fake = Fake(monkeypatch, tmp_path, trace)
    result = check_walk(video_to_joints(fake.video, fake.model))
    assert not result.passed and result.checks["filled_share"]["value"] == pytest.approx(80 / 300)


# --------------------------------------------------------------------------- wording
def test_no_message_talks_about_diagnosis_or_cause() -> None:
    walks = [
        good(leg_visibility=0.1, filled_frames=200, facing_agreement=0.1, fps=5.0),
        good(2.5, truncated=True, stretches=4, unused_seconds=9.0),
        good(),
    ]
    messages = [
        m
        for w in walks
        for r in (check_walk(w), check_walk(w, people_in_view=3))
        for m in (*r.reasons, *r.warnings)
    ]
    messages.append(
        reject(VideoError("no walk found: a person was seen in 0% of the frames")).reasons[0]
    )
    assert len(messages) > 10
    assert not [m for m in messages if BANNED.search(m)]


# --------------------------------------------------------------------------- command line
def test_cli_prints_the_verdict_the_checks_and_never_the_file_name(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(qg, "video_to_joints", lambda *_a, **_k: good())
    assert qg.main(["secret_name.mov"]) == 0
    out = capsys.readouterr().out
    assert "PASSED" in out and "secret_name" not in out and "w-" in out
    assert "strides" in out and "warning: One person in view is assumed" in out


def test_cli_rejects_with_exit_1_and_the_reasons(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(qg, "video_to_joints", lambda *_a, **_k: good(2.5))
    assert qg.main(["v.mov"]) == 1
    out = capsys.readouterr().out
    assert "REJECTED" in out and "reason: The walk has 2 real strides" in out and "FAIL" in out


def test_cli_passes_its_options_through(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    asked: list[float | None] = []

    def fake(_p: Path, _m: Path, *, max_seconds: float | None = None) -> VideoWalk:
        asked.append(max_seconds)
        return good(2.5)

    monkeypatch.setattr(qg, "video_to_joints", fake)
    assert (
        qg.main(["v.mov", "--min-strides", "2", "--max-seconds", "12"]) == 0
    )  # 2 strides now pass
    assert asked == [12.0]
    assert qg.main(["v.mov", "--people", "2", "--min-strides", "2"]) == 1
    assert "2 people are in view" in capsys.readouterr().out


def test_cli_turns_a_video_error_into_a_rejection(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refuse(*_a: object, **_k: object) -> VideoWalk:
        raise VideoError("the video file is missing or empty")

    monkeypatch.setattr(qg, "video_to_joints", refuse)
    assert qg.main(["secret_name.mov"]) == 1
    out = capsys.readouterr().out
    assert "REJECTED" in out and "reason: the video file is missing or empty" in out
    assert "secret_name" not in out
