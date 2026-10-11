"""Tests for clip ID parsing.

The failing cases matter more than the passing ones: a clip ID that cannot be
parsed must raise, because a wrong participant key silently leaks one hand of
a participant across a fold boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clip_id import UnparseableClipId, parse_clip_id  # noqa: E402


def test_control_right_hand() -> None:
    key = parse_clip_id("CONTROL01_DCHA")
    assert key.participant == "CONTROL01"
    assert key.group == "CONTROL"
    assert key.hand == "right"
    assert key.number == 1


def test_patient_left_hand() -> None:
    key = parse_clip_id("ID1004_IZDA")
    assert key.participant == "ID1004"
    assert key.group == "ID"
    assert key.hand == "left"


def test_both_hands_share_one_participant_key() -> None:
    """The whole point: both clips of one person group together."""
    left = parse_clip_id("ID110_IZDA")
    right = parse_clip_id("ID110_DCHA")
    assert left.participant == right.participant


def test_leading_zeros_give_different_keys_but_same_number() -> None:
    """hubu_fis.load_labels uses ``number`` to reject "1" vs "01" spellings."""
    a, b = parse_clip_id("CONTROL1_DCHA"), parse_clip_id("CONTROL01_DCHA")
    assert a.participant != b.participant
    assert a.number == b.number == 1


def test_control_and_patient_numbers_do_not_collide() -> None:
    """CONTROL101 and ID101 are different people."""
    assert parse_clip_id("CONTROL101_DCHA").participant != parse_clip_id("ID101_DCHA").participant


@pytest.mark.parametrize(
    "bad",
    [
        "CONTROL01",  # no hand
        "CONTROL01_LEFT",  # wrong hand token
        "SUBJECT01_DCHA",  # wrong group token
        "CONTROL_DCHA",  # no number
        "CONTROL01_DCHA.mp4",  # extension not stripped
        "control01_dcha",  # wrong case
        "CONTROL01_DCHA\n",  # trailing newline ($ alone would accept this)
        " CONTROL01_DCHA",  # leading whitespace
        "CONTROL01_DCHA ",  # trailing whitespace
        "ID-1_DCHA",  # sign is not a digit
        "",
    ],
)
def test_unparseable_ids_raise(bad: str) -> None:
    with pytest.raises(UnparseableClipId):
        parse_clip_id(bad)
