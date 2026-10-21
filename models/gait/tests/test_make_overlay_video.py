"""Tests for the skeleton and heel-strike overlay. No video or pose model is used."""

from __future__ import annotations

from pathlib import Path

import make_overlay_video as mov
import numpy as np
import pytest
from cycles import Cycle, CycleCheck
from test_gait_features import full_walk
from test_mediapipe_mapping import mediapipe_world, trace_of


def cycle(foot: str, start: int, end: int, ok: bool = True) -> Cycle:
    return Cycle(foot, start, end, (end - start) / 30, 0.3, ok, ok)  # type: ignore[arg-type]


def test_counted_strikes_keep_only_plausible_cycles_and_add_the_offset() -> None:
    check = CycleCheck(
        (cycle("left", 5, 35), cycle("right", 20, 50), cycle("left", 35, 40, ok=False)), 3
    )
    assert mov.counted_strikes(check, 100) == {
        105: ["left"],
        120: ["right"],
        135: ["left"],
        150: ["right"],
    }


def test_a_strike_shared_by_two_cycles_is_listed_once() -> None:
    check = CycleCheck((cycle("left", 5, 35), cycle("left", 35, 65)), 3)
    assert mov.counted_strikes(check, 0) == {5: ["left"], 35: ["left"], 65: ["left"]}


def test_a_normal_walk_is_scored_with_strikes_inside_the_walk() -> None:
    plan = mov.build_plan(trace_of(mediapipe_world(full_walk(seconds=8.0))))
    assert plan.state == "scored" and plan.strides >= 3
    assert plan.strikes and all(plan.start <= f < plan.stop for f in plan.strikes)


def test_a_clip_with_no_usable_stretch_is_not_scored() -> None:
    world = mediapipe_world(full_walk(seconds=8.0))
    plan = mov.build_plan(trace_of(world, np.zeros(world.shape[0], dtype=bool)))
    assert plan.state == "no_walk" and plan.strikes == {}


def test_a_frame_rate_outside_the_supported_range_is_not_scored() -> None:
    plan = mov.build_plan(trace_of(mediapipe_world(full_walk(seconds=8.0)), fps=120.0))
    assert plan.state == "frame_rate"


def test_the_banner_says_what_happened_to_each_part_of_the_clip() -> None:
    scored = mov.Plan("scored", 30, 90, strides=5)
    assert mov.caption(scored, 30.0, 40) == (
        "SCORED WALK 1.0 s to 3.0 s | 5 strides counted",
        mov.GREEN,
    )
    assert mov.caption(scored, 30.0, 10)[1] == mov.GREY  # before the walk: not used
    few = mov.Plan("few_strides", 0, 60, strides=2)
    assert "too few strides (2)" in mov.caption(few, 30.0, 10)[0]
    assert mov.caption(mov.Plan("no_walk"), 30.0, 0)[0].startswith("NOT SCORED")


def test_a_ring_grows_after_the_strike_and_is_gone_afterwards() -> None:
    assert mov.ring_radius(-1, 30.0, 1000) is None
    first, later = mov.ring_radius(0, 30.0, 1000), mov.ring_radius(8, 30.0, 1000)
    assert first is not None and later is not None and later > first
    assert mov.ring_radius(11, 30.0, 1000) is None  # 0.35 s at 30 fps is 10.5 frames


def test_a_large_frame_is_scaled_down_to_even_sides_and_a_small_one_is_left_alone() -> None:
    big = np.zeros((2160, 3840, 3), dtype=np.uint8)
    out = mov.fit(big)
    assert max(out.shape[:2]) == mov.MAX_SIDE and all(side % 2 == 0 for side in out.shape[:2])
    assert out.shape[0] / out.shape[1] == pytest.approx(2160 / 3840, abs=0.01)
    small = np.zeros((480, 640, 3), dtype=np.uint8)
    assert mov.fit(small) is small


def drawn(frame: int, plan: mov.Plan) -> tuple[np.ndarray, np.ndarray]:
    bgr = np.full((120, 200, 3), 200, dtype=np.uint8)
    image = np.full((33, 2), 0.5)
    image[:, 0] = np.linspace(0.3, 0.7, 33)
    out = mov.draw_frame(bgr, image, np.ones(33), plan, frame, 30.0)
    return bgr, out


def test_drawing_leaves_the_input_alone_and_colours_the_border_by_state() -> None:
    plan = mov.Plan("scored", 10, 50, strides=4, strikes={20: ["left"]})
    bgr, inside = drawn(20, plan)
    assert (bgr == 200).all() and inside.shape == bgr.shape
    assert tuple(inside[0, 0]) == mov.GREEN
    assert tuple(drawn(60, plan)[1][0, 0]) == mov.GREY  # after the walk


def test_a_landmark_that_is_not_visible_draws_nothing() -> None:
    plan = mov.Plan("scored", 0, 50, strides=4)
    bgr = np.full((120, 200, 3), 200, dtype=np.uint8)
    image = np.full((33, 2), 0.5)
    hidden = mov.draw_frame(bgr, image, np.zeros(33), plan, 10, 30.0)
    shown = mov.draw_frame(bgr, image, np.ones(33), plan, 10, 30.0)
    assert (hidden != shown).any()
    assert (hidden[40:80, 60:140] == 200).all()  # nothing drawn in the middle


def test_outputs_are_named_by_hash_unless_a_name_is_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    written: list[Path] = []

    def fake(
        video: Path, out: Path, model: Path = Path(), trace_file: Path | None = None
    ) -> mov.Plan:
        written.append(out)
        return mov.Plan("scored", 0, 10, strides=3)

    monkeypatch.setattr(mov, "make_overlay", fake)
    assert mov.main(["secret_name.MOV", "--out-dir", str(tmp_path)]) == 0
    assert written[0].parent == tmp_path and "secret_name" not in written[0].name
    assert mov.main(["secret_name.MOV", "--out", str(tmp_path / "mine.mp4")]) == 0
    assert written[1].name == "mine.mp4"
    with pytest.raises(SystemExit):
        mov.main(["a.MOV", "b.MOV", "--out", str(tmp_path / "x.mp4")])


def test_a_video_that_cannot_be_read_is_reported_and_the_rest_carry_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake(
        video: Path, out: Path, model: Path = Path(), trace_file: Path | None = None
    ) -> mov.Plan:
        if video.name == "bad.MOV":
            raise ValueError("no frames could be read from the video")
        return mov.Plan("scored", 0, 10, strides=3)

    monkeypatch.setattr(mov, "make_overlay", fake)
    assert mov.main(["bad.MOV", "good.MOV", "--out-dir", str(tmp_path)]) == 2
    assert capsys.readouterr().out.count("->") == 1


def test_the_banner_is_at_the_top_where_player_controls_cannot_cover_it() -> None:
    _, out = drawn(20, mov.Plan("scored", 10, 50, strides=4, strikes={20: ["left"]}))
    assert tuple(out[5, 185]) == (30, 30, 30)  # the banner bar
    assert tuple(out[-8, 100]) == (200, 200, 200)  # nothing is drawn along the bottom


def test_a_video_with_no_walk_gets_no_overlay_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    world = mediapipe_world(full_walk(seconds=8.0))
    trace = trace_of(world, np.zeros(world.shape[0], dtype=bool))
    monkeypatch.setattr(mov, "read_info", lambda _p: None)
    monkeypatch.setattr(mov, "load_trace", lambda _p: trace)
    out = tmp_path / "o.mp4"
    plan = mov.make_overlay(tmp_path / "v.mov", out, trace_file=tmp_path / "t.npz")
    assert plan.state == "no_walk" and not out.exists()


def test_the_main_program_says_when_it_wrote_no_overlay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(mov, "make_overlay", lambda *a, **k: mov.Plan("no_walk"))
    assert mov.main(["v.MOV", "--out-dir", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert "no overlay written" in printed and "->" not in printed


def test_a_saved_trace_can_be_given_for_one_video_only(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        mov.main(["a.MOV", "b.MOV", "--trace", str(tmp_path / "t.npz")])
