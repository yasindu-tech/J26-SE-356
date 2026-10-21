"""Tests for the quick pre-check of a video. The pose model is not run."""

from __future__ import annotations

from pathlib import Path

import prescreen as ps
import pytest


def test_the_frames_checked_are_spread_over_the_part_that_will_be_read() -> None:
    picked = ps.sample_indices(900, 30.0, None)
    assert len(picked) == ps.SAMPLES and picked == sorted(set(picked))
    assert picked[0] < 900 * 0.1 and picked[-1] > 900 * 0.9
    assert max(ps.sample_indices(900, 30.0, 10.0)) < 300  # only the first 10 s count


def test_a_very_short_or_empty_video_gives_fewer_frames_or_none() -> None:
    assert ps.sample_indices(5, 30.0, None) == [0, 1, 2, 3, 4]
    assert ps.sample_indices(0, 30.0, None) == []


def test_a_video_with_the_legs_in_two_of_twelve_frames_goes_on() -> None:
    assert ps.refusal([True] * 2 + [False] * 10) is None
    assert ps.refusal([True] * 12) is None


def test_a_video_with_hardly_any_legs_is_refused_with_a_plain_reason() -> None:
    reason = ps.refusal([True] + [False] * 11)
    assert reason is not None and "only 1 of the 12 frames" in reason
    assert "head to feet" in reason
    none = ps.refusal([False] * 12) or ""
    assert "no person with the legs in view was found in any of the 12 frames" in none
    assert "legs were clearly visible" not in none  # it may be no person at all, e.g. a hand
    assert ps.refusal([False] * 12) is not None
    assert ps.refusal([False, False]) is not None and ps.refusal([True, False]) is None


def test_nothing_looked_at_means_no_opinion() -> None:
    assert ps.refusal([]) is None


def test_a_video_that_cannot_be_read_is_left_to_the_full_analysis(tmp_path: Path) -> None:
    junk = tmp_path / "junk.mov"
    junk.write_bytes(b"not a video")
    assert ps.prescreen(junk, tmp_path / "no_model.task") is None
    assert ps.prescreen(tmp_path / "missing.mov", tmp_path / "no_model.task") is None


def test_prescreen_returns_the_reason_for_a_video_without_legs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ps, "_legs_clear", lambda *a: [False] * 12)
    assert "does not look like a video of someone walking" in (
        ps.prescreen(tmp_path / "v.mov", tmp_path) or ""
    )
    monkeypatch.setattr(ps, "_legs_clear", lambda *a: [True] * 12)
    assert ps.prescreen(tmp_path / "v.mov", tmp_path) is None
