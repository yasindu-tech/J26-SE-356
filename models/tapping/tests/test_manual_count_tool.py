"""Tests for models/tapping/src/manual_count_tool.py (task FT-24 counting tool).

No real clip and no window: a scripted FakeUI replaces the OpenCV window, a fake clock drives
playback, and clips are synthetic (in memory, or a small video written to a temporary folder).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import manual_count_tool as tool  # noqa: E402
import manual_counts as mc  # noqa: E402

REFS = ("aaaa1111", "bbbb2222", "cccc3333", "dddd4444")
POISON = "POISONCELL"


class FakeUI:
    """Plays back a key script; records every overlay line the tool composes."""

    def __init__(self, keys: Iterable[int | str] = ()) -> None:
        self.keys = [ord(k) if isinstance(k, str) else k for k in keys]
        self.shown = 0
        self.closed = False

    def show(self, image: np.ndarray) -> None:
        self.shown += 1

    def key(self, wait_ms: int) -> int:
        if not self.keys:
            raise RuntimeError("key script exhausted")
        return self.keys.pop(0)

    def close(self) -> None:
        self.closed = True


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 0.01
        return self.now


def tiny_clip(n: int = 30, fps: float = 30.0) -> tool.Clip:
    frames = [tool.encode(np.full((16, 12, 3), 8 * i % 256, np.uint8)) for i in range(n)]
    return tool.make_clip(frames, [1000 * i / fps + 50 for i in range(n)])


@pytest.fixture
def lines(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []
    real = tool.compose

    def spy(image: np.ndarray, text: list[str]) -> np.ndarray:
        seen.extend(text)
        return real(image, text)

    monkeypatch.setattr(tool, "compose", spy)
    return seen


@pytest.fixture
def out(tmp_path: Path) -> Path:
    """A selection.csv with poisoned cell and weight columns the tool must never show."""
    out = tmp_path / "interim" / "manual_counts"
    out.mkdir(parents=True)
    rows = [
        {"ref": REFS[0], "set": "primary", "second_rater": "1"},
        {"ref": REFS[1], "set": "primary", "second_rater": "0"},
        {"ref": REFS[2], "set": "primary", "second_rater": "1"},
        {"ref": REFS[3], "set": "practice", "second_rater": "0"},
    ]
    sel = pd.DataFrame(rows)
    sel["cell"] = f"{POISON} slow/flagged"
    sel["cell_rank"] = "0"
    sel["weight"] = "987654"
    sel["replaced_for"] = ""
    sel["seed"] = "24"
    sel[list(mc.SELECTION_COLUMNS)].to_csv(out / mc.SELECTION_NAME, index=False)
    return out


def session(
    out: Path,
    keys: Sequence[int | str],
    rater: str = "R1",
    kind: str = "main",
    resume: bool = False,
    load=None,
) -> FakeUI:  # type: ignore[no-untyped-def]
    ui = FakeUI(keys)
    tool.run_session(
        rater,
        kind,
        resume,
        out_dir=out,
        ui=ui,
        load=load or (lambda ref: tiny_clip()),
        clock=Clock(),
    )
    return ui


SAVE_TWO: list[int | str] = [
    "m",
    ".",
    ".",
    "m",
    "e",
    "3",
    "s",
]  # marks at frames 0 and 2, one edge partial


# --------------------------------------------------------------------------- frame cache


def test_frame_cache_returns_frames_in_order(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), 30.0, (64, 48))
    for i in range(40):
        writer.write(np.full((48, 64, 3), 5 * i, np.uint8))
    writer.release()
    clip = tool.decode_clip(path)
    assert len(clip) == 40
    means = [float(clip.image(i).mean()) for i in range(40)]
    assert np.allclose(means, [5 * i for i in range(40)], atol=3)
    assert clip.times[0] == 0 and (np.diff(clip.times) > 0).all()
    assert clip.times[-1] == pytest.approx(39 / 30, abs=0.002)


def test_frames_are_scaled_to_the_long_side() -> None:
    img = tool.Clip([tool.encode(np.zeros((1920, 1080, 3), np.uint8))] * 2, np.array([0.0, 0.1]))
    assert img.image(0).shape[:2] == (960, 540)


def test_bad_times_or_tiny_clips_are_refused() -> None:
    frame = tool.encode(np.zeros((8, 8, 3), np.uint8))
    with pytest.raises(tool.DecodeError):
        tool.make_clip([frame, frame], [0.0, 0.0])
    with pytest.raises(tool.DecodeError):
        tool.make_clip([frame], [0.0])
    with pytest.raises(tool.DecodeError):
        tool.decode_clip(Path("does-not-exist.mp4"))


# --------------------------------------------------------------------------- player and marks


def test_player_follows_the_clock_at_its_speed() -> None:
    p = tool.Player(np.arange(61) / 30.0)
    assert p.speed == 0.5
    p.play(100.0)
    assert p.tick(101.0) == 15  # 1 s of wall time at 0.5x = 0.5 s = frame 15
    p.change_speed(+1, 101.0)
    assert p.tick(102.0) == 45  # then 1 s at 1x
    p.change_speed(+5, 102.0)
    assert p.speed == 1.0
    p.change_speed(-9, 102.0)
    assert p.speed == 0.125
    assert p.tick(1000.0) == 60 and not p.playing  # stops at the last frame
    p.step(-1)
    assert p.index == 59
    p.step(-100)
    assert p.index == 0


def test_marks_undo_and_edges() -> None:
    c = tool.Count()
    assert c.mark(5) and c.mark(2) and not c.mark(5)
    c.edge()
    assert c.marks() == [2, 5] and c.edges() == 1
    assert c.undo() and c.edges() == 0
    assert c.undo() and c.marks() == [5]
    assert c.undo() and not c.undo()


# --------------------------------------------------------------------------- sessions


def test_full_session_writes_rows_in_the_rater_order(out: Path) -> None:
    session(out, SAVE_TWO * 3)
    rows = mc.read_rows(out / "R1.csv")
    order = mc.list_refs(mc.read_selection(out), "R1", "main")
    assert rows["ref"].tolist() == order and rows["order"].tolist() == ["1", "2", "3"]
    assert (rows["n_taps"] == "2").all() and (rows["tap_frames"] == "0;2").all()
    assert (rows["tap_times_s"] == "0.000;0.067").all() and (rows["n_edge_partials"] == "1").all()
    assert mc.check_rows(rows, "R1", order, "R1.csv") == []
    assert not (out / ".R1.journal.json").exists()


def test_save_needs_a_confidence(out: Path, lines: list[str]) -> None:
    session(out, ["m", "s", "2", "s", *SAVE_TWO * 2])
    assert any("set the confidence" in line for line in lines)
    assert mc.read_rows(out / "R1.csv")["confidence"].tolist()[0] == "2"


def test_quit_keeps_the_journal_and_resume_restores_it(out: Path) -> None:
    session(out, ["m", ".", "m", "q"])
    assert (out / ".R1.journal.json").exists() and not (out / "R1.csv").exists()
    with pytest.raises(tool.SessionError, match="--resume"):
        session(out, [])
    session(out, ["u", "1", "s", *SAVE_TWO * 2], resume=True)
    first = mc.read_rows(out / "R1.csv").iloc[0]
    assert first["n_taps"] == "1" and first["tap_frames"] == "0" and first["confidence"] == "1"


def test_a_crash_loses_no_mark(out: Path) -> None:
    with pytest.raises(RuntimeError):
        session(out, ["m", ".", ".", "m"])  # the script ends right after a mark: a crash
    saved = json.loads((out / ".R1.journal.json").read_text())
    assert saved["count"]["actions"] == [["m", 0], ["m", 2]]
    with pytest.raises(RuntimeError):
        session(out, ["e"], resume=True)
    with pytest.raises(RuntimeError):
        session(out, ["3"], resume=True)
    session(out, ["s", *SAVE_TWO * 2], resume=True)
    first = mc.read_rows(out / "R1.csv").iloc[0]
    assert first["tap_frames"] == "0;2" and first["n_edge_partials"] == "1"
    assert first["confidence"] == "3"


def test_resume_skips_saved_clips(out: Path) -> None:
    session(out, [*SAVE_TWO, "q"])
    session(out, SAVE_TWO * 2, resume=True)
    assert len(mc.read_rows(out / "R1.csv")) == 3


def test_notes_are_checked_before_they_are_kept(out: Path, lines: list[str]) -> None:
    keys: list[int | str] = ["n", *"control", 13, 27, "n", *"blur late", 13, *SAVE_TWO]
    session(out, keys + SAVE_TWO * 2)
    assert any("note not kept" in line for line in lines)
    assert not any("control" in line and "note not kept" in line for line in lines)
    assert mc.read_rows(out / "R1.csv")["note"].tolist()[0] == "blur late"


def test_unreadable_clips_are_listed_without_naming_them(
    out: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    order = mc.list_refs(mc.read_selection(out), "R1", "main")

    def load(ref: str) -> tool.Clip:
        if ref == order[1]:
            raise tool.DecodeError("cannot open /Users/someone/CONTROL01_DCHA.mp4")
        return tiny_clip()

    session(out, SAVE_TWO * 2, load=load)
    assert mc.read_unreadable(out) == {order[1]}
    text = capsys.readouterr().out
    assert "clip 2 of 3 could not be decoded" in text and order[1] not in text


def test_lists_per_rater(out: Path) -> None:
    session(out, SAVE_TWO * 2, rater="R2")
    assert set(mc.read_rows(out / "R2.csv")["ref"]) == {REFS[0], REFS[2]}
    session(out, SAVE_TWO, kind="practice")
    assert mc.read_rows(out / "R1_practice.csv")["ref"].tolist() == [REFS[3]]
    with pytest.raises(mc.InputError):
        session(out, [], rater="R2", kind="practice")


def test_nothing_shown_or_printed_names_a_clip(
    out: Path, lines: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    session(
        out, ["n", *"ok", 13, "]", " ", -1, -1, " ", "[", "x", "u", "u", *SAVE_TWO, *SAVE_TWO * 2]
    )
    text = capsys.readouterr()
    shown = "\n".join(lines) + text.out + text.err + tool.WINDOW_TITLE
    for secret in (*REFS, str(out), out.name, POISON, "slow/", "flagged", "987654", "weight"):
        assert secret not in shown
    assert tool.WINDOW_TITLE == "FT-24 counter" == tool.CvUI.title
    assert tool.TOOL_COLUMNS == ("ref", "set", "second_rater", "replaced_for")


def test_overlay_text_is_only_counter_state() -> None:
    p, c = tool.Player(np.arange(10) / 30.0), tool.Count()
    c.mark(0)
    text = tool.overlay_lines(3, 40, p, c, None, "")
    assert text[0].startswith("clip 3 of 40") and "taps marked 1" in text[1]
    assert "[this frame is marked]" in text[1]


def test_main_hides_error_details(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    def broken() -> FakeUI:
        raise RuntimeError("/Users/someone/data/raw/HUBU-FIS/videos/CONTROL01_DCHA.mp4")

    assert tool.main(["--selftest"], ui_factory=broken) == 2
    err = capsys.readouterr().err
    assert "RuntimeError" in err and "Users" not in err and "CONTROL" not in err


def test_main_argument_rules(tmp_path: Path) -> None:
    assert tool.main([]) == 2
    assert tool.main(["--rater", "R1", "--out-dir", str(tmp_path)]) == 2
    assert tool.main(["--rater", "R2", "--practice"]) == 2
    assert tool.main(["--rater", "R1", "--hard"]) == 2


# --------------------------------------------------------------------------- self-test


def test_selftest_clip_has_the_stated_taps() -> None:
    t, a = tool.selftest_aperture()
    peaks = tool.selftest_peaks(a)
    assert len(peaks) == tool.SELFTEST_TAPS == 7
    assert list(peaks) == [round(s * tool.SELFTEST_FPS) for s in tool.SELFTEST_PEAK_TIMES]
    assert a[-1] > a[-2] > a[-3]  # still opening at the last frame: one edge partial
    assert a[0] == a[1] == pytest.approx(0.08)  # starts closed: no edge partial at the start
    second = a[peaks[3] : peaks[4] + 1]
    assert a[peaks[3]] - second.min() > 0.3  # the double bump has a clear reversal
    assert len(tool.make_selftest_clip()) == len(t) == 181


def test_selftest_scores_a_correct_count(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    keys: list[int | str] = []
    at = 0
    for frame in (round(s * tool.SELFTEST_FPS) for s in tool.SELFTEST_PEAK_TIMES):
        keys += ["."] * (frame - at) + ["m"]
        at = frame
    keys += ["e", "3", "s"]
    assert tool.main(["--selftest"], ui_factory=lambda: FakeUI(keys), clock=Clock()) == 0
    text = capsys.readouterr().out
    assert "expected 7 taps and 1 edge partial; you counted 7 taps and 1 edge partials" in text
    assert "selftest passed" in text
    assert list(tmp_path.iterdir()) == []  # nothing written


def test_selftest_reports_a_wrong_count(capsys: pytest.CaptureFixture[str]) -> None:
    assert tool.main(["--selftest"], ui_factory=lambda: FakeUI(["m", "3", "s"]), clock=Clock()) == 1
    assert "counts differ" in capsys.readouterr().out


# --------------------------------------------------------------------------- key handler


def keys_for(n: int = 30) -> tuple[tool.Keys, tool.Player, tool.Count]:
    player, count = tool.Player(tiny_clip(n).times), tool.Count()
    return tool.Keys(player, count, lambda: None), player, count


def press(keys: tool.Keys, text: str, now: float = 0.0, shown: int = 0) -> list[str | None]:
    return [keys.handle(ord(c), now, shown) for c in text]


def test_repeated_mark_on_one_frame_is_one_mark() -> None:
    keys, _, count = keys_for()
    press(keys, "mmm", shown=5)
    assert count.marks() == [5] and count.actions == [["m", 5]]
    assert keys.message == "this frame is already marked"


def test_held_key_while_playing_gives_one_mark_and_pauses() -> None:
    keys, player, count = keys_for()
    player.play(0.0)
    assert player.playing
    press(keys, "m", shown=7)
    assert not player.playing and player.index == 7 and count.marks() == [7]
    press(keys, "m" * 40, shown=7)  # auto-repeat: the video no longer moves
    assert count.marks() == [7] and len(count.actions) == 1


def test_two_different_frames_and_undo_and_edge() -> None:
    keys, _, count = keys_for()
    press(keys, "m", shown=3)
    press(keys, "m", shown=9)
    assert count.marks() == [3, 9]
    press(keys, "e")
    assert count.edges() == 1 and count.marks() == [3, 9]  # an edge is not a tap
    press(keys, "u")  # undoes the edge
    assert count.edges() == 0 and count.marks() == [3, 9]
    press(keys, "u")  # undoes the last mark (9), not the first
    assert count.marks() == [3]
    press(keys, "uu")
    assert count.marks() == [] and keys.message == "nothing to undo"


def test_save_needs_confidence_and_quit_returns() -> None:
    keys, _, _ = keys_for()
    assert press(keys, "s") == [None] and "confidence" in keys.message
    assert press(keys, "3s") == [None, "save"]
    assert press(keys, "q") == ["quit"]


def test_overlay_shows_the_zero_based_frame_and_the_rater_counts() -> None:
    p, c = tool.Player(np.arange(10) / 30.0, index=4), tool.Count()
    c.mark(4)
    c.edge()
    text = tool.overlay_lines(1, 1, p, c, None, "")
    assert "frame 4 (0-9)" in text[0]
    assert "taps marked 1" in text[1] and "edge partials 1" in text[1]


def test_selftest_diagnosis_lists_offsets_and_possible_repeat() -> None:
    expected = [21, 39, 57, 75, 90, 120, 150]
    lines = tool.selftest_diagnosis([41, 73, 74, 89, 95], expected)
    assert "  mark 41: nearest expected 39, offset +2" in lines
    assert "  mark 74: nearest expected 75, offset -1" in lines
    assert any("possible key repeat" in x and "(73, 74)" in x for x in lines)
    assert not any("possible key repeat" in x for x in tool.selftest_diagnosis([21, 39], expected))


def test_failed_selftest_prints_the_diagnosis(capsys: pytest.CaptureFixture[str]) -> None:
    keys: list[int | str] = [".", "m", "m", ".", "m", "3", "s"]
    assert tool.main(["--selftest"], ui_factory=lambda: FakeUI(keys), clock=Clock()) == 1
    out = capsys.readouterr().out
    assert "nearest expected" in out and "possible key repeat" in out
