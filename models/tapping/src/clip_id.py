"""Parse HUBU-FIS clip identifiers into participant and hand.

A clip ID looks like CONTROL01_DCHA or ID1004_IZDA:

    CONTROL | ID   cohort group (control or patient)
    <digits>       participant number within that group
    DCHA | IZDA    hand: derecha (right) or izquierda (left)

The participant key returned here is what cross-validation folds are grouped
on. Two clips from the same person must never straddle train and test, so a
clip ID that cannot be parsed is an error, never a silently skipped row.
"""

from __future__ import annotations

import re
from typing import NamedTuple

_CLIP_RE = re.compile(r"(?P<group>CONTROL|ID)(?P<number>\d+)_(?P<hand>DCHA|IZDA)")

_HAND_EN = {"DCHA": "right", "IZDA": "left"}


class ClipKey(NamedTuple):
    """The parts of a clip ID that identify who and which hand."""

    participant: str
    group: str
    hand: str
    number: int  # participant number as an integer, to catch "01" vs "1" spellings


class UnparseableClipId(ValueError):
    """Raised when a clip ID does not match the HUBU-FIS naming scheme."""


def parse_clip_id(clip_id: str) -> ClipKey:
    """Split a HUBU-FIS clip ID into participant, cohort group and hand.

    Args:
        clip_id: e.g. "CONTROL01_DCHA" or "ID1004_IZDA", without file extension.

    Returns:
        ClipKey(participant="CONTROL01", group="CONTROL", hand="right", number=1)

    Raises:
        UnparseableClipId: if the ID does not match the scheme exactly. Grouping
            folds on a wrong key would leak one hand of a participant into the
            test set, so this fails loudly rather than guessing. Whitespace and
            trailing newlines are not stripped: they are errors in the source.
    """
    match = _CLIP_RE.fullmatch(clip_id)
    if match is None:
        raise UnparseableClipId(
            f"clip ID {clip_id!r} does not match <CONTROL|ID><number>_<DCHA|IZDA>"
        )
    group = match.group("group")
    return ClipKey(
        participant=f"{group}{match.group('number')}",
        group=group,
        hand=_HAND_EN[match.group("hand")],
        number=int(match.group("number")),
    )
