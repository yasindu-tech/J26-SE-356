"""GAIT-13: cut walks into gait cycles and flag walks with too few of them.

A gait cycle (a stride) runs from one heel strike to the next heel strike of the same foot. A
cycle is counted only if it is a real walking stride:

- it lasts ``MIN_CYCLE_S`` to ``MAX_CYCLE_S`` (a pause or a missed strike makes it too long);
- the ankle swings at least ``MIN_STEP_EXCURSION_M`` forward and back relative to the pelvis,
  along the direction the hips face. Stepping on the spot, turning in place or freezing lifts
  the foot without a stride. The swing is about half the stride length, so 0.1 m keeps short
  shuffling strides: on CARE-PD class 2-3 cycles it removes 61% of cycles where the pelvis
  moved under 0.08 m (stationary) but only 6% of 0.15-0.30 m strides and none of the longer
  ones (checked 9 Oct 2026). 0.2 m was tried first and removed 86% of short shuffling strides,
  the very gait PD causes. The rule uses only pelvis-relative positions, so it works on
  MediaPipe video as well.

The cycle check (Rule B, confirmed by the plan owner on 9 Oct 2026): a walk passes if it has at
least ``CAREPD_MIN_STRIDES`` plausible strides counting both feet together; ``KOA_MIN_STRIDES``
for the short KOA-PD-NM clips. The sheet's first wording, at least 3 cycles on one foot, was
not used because in BMCLab it kept 9% of class 0 walks but 53% of class 2-3 walks: faster,
healthier walkers cross the fixed capture area in fewer strides, so a strict cycle count removes
walks by severity. See ``models/gait/docs/cycle_check.md``.

Walks that fail are flagged, not deleted; GAIT-17 builds features only for walks that pass.
Walk duration and cycle count must never be model features: they carry the capture-area effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from preprocess import FootStrikes, HeelStrikes

MIN_CYCLE_S = 0.5
MAX_CYCLE_S = 3.0
MIN_STEP_EXCURSION_M = 0.1  # a real stride swings the ankle past the body by at least this
CAREPD_MIN_STRIDES = 3  # plausible strides, both feet together
KOA_MIN_STRIDES = 2  # KOA-PD-NM clips are single short passes

Foot = Literal["left", "right"]


@dataclass(frozen=True)
class Cycle:
    foot: Foot
    start: int  # frame of the heel strike that opens the cycle
    end: int  # frame of the next heel strike of the same foot
    seconds: float
    excursion_m: float  # forward-back swing of the ankle relative to the pelvis; nan if unknown
    duration_ok: bool
    real_step: bool

    @property
    def plausible(self) -> bool:
        """Counted as a stride: a plausible duration and a real step."""
        return self.duration_ok and self.real_step


@dataclass(frozen=True)
class CycleCheck:
    cycles: tuple[Cycle, ...]
    min_strides: int

    def strides(self, foot: Foot | None = None) -> int:
        """Plausible strides, for one foot or both."""
        return sum(1 for c in self.cycles if c.plausible and (foot is None or c.foot == foot))

    @property
    def implausible(self) -> int:
        return sum(1 for c in self.cycles if not c.plausible)

    @property
    def bad_duration(self) -> int:
        return sum(1 for c in self.cycles if not c.duration_ok)

    @property
    def not_real_step(self) -> int:
        """Cycles with a plausible duration but no real stride (stepping on the spot)."""
        return sum(1 for c in self.cycles if c.duration_ok and not c.real_step)

    @property
    def passed(self) -> bool:
        return self.strides() >= self.min_strides

    @property
    def reason(self) -> str:
        if self.passed:
            return ""
        return f"{self.strides()} real strides on both feet (need {self.min_strides})"

    def stride_times(self, foot: Foot | None = None) -> np.ndarray:
        return np.array(
            [c.seconds for c in self.cycles if c.plausible and (foot is None or c.foot == foot)]
        )


def foot_cycles(
    strikes: FootStrikes, fps: float, foot: Foot, forward: np.ndarray | None = None
) -> list[Cycle]:
    """Cycles of one foot. ``forward`` is the ankle ahead of the pelvis per frame; without it the
    real-step test cannot be made and every cycle counts as a real step (synthetic use only)."""
    s = strikes.strikes
    out = []
    for a, b in zip(s[:-1], s[1:], strict=True):
        seconds = (int(b) - int(a)) / fps
        if forward is None:
            excursion, real = float("nan"), True
        else:
            seg = forward[int(a) : int(b) + 1]
            excursion = float(seg.max() - seg.min())
            real = excursion >= MIN_STEP_EXCURSION_M
        out.append(
            Cycle(
                foot,
                int(a),
                int(b),
                seconds,
                excursion,
                MIN_CYCLE_S <= seconds <= MAX_CYCLE_S,
                real,
            )
        )
    return out


def segment_cycles(hs: HeelStrikes, min_strides: int = CAREPD_MIN_STRIDES) -> CycleCheck:
    """All cycles of both feet, in time order, with the pass/flag decision."""
    if min_strides < 1:
        raise ValueError("min_strides must be at least 1")
    cycles = foot_cycles(hs.left, hs.fps, "left", hs.left_forward) + foot_cycles(
        hs.right, hs.fps, "right", hs.right_forward
    )
    cycles.sort(key=lambda c: (c.start, c.foot))
    return CycleCheck(tuple(cycles), min_strides)
