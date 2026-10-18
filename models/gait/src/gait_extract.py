"""GAIT-17: one function from a walk to the eight model inputs.

``extract_features(joints, fps)`` is the only place where a walk becomes model inputs. The
training table (CARE-PD) and the demo (MediaPipe video) both call it, so there is no second code
path that could drift. It runs the heel-strike detector and the cycle check (GAIT-12/13), then
the timing features (GAIT-14) and the gait features (GAIT-15), and returns them in the fixed
order of ``FEATURE_NAMES``.

Decisions (checked on 2,950 CARE-PD walks and 8 video clips, 12 Oct 2026, see
``models/gait/docs/extract_features.md``):

1. *A walk that cannot give the features is a result, not an error.* Too few real strides,
   strides on one foot only, or an implausible leg length come back as ``status="rejected"``
   with a reason, so the caller can count them by cohort and class (rejection depends on severity:
   BMCLab rejects 34% of class 0 but 4% of class 2-3; PD-GaM the other way round). Bad input (wrong
   shape, NaN, a frame rate outside 20-60 fps) is a programming error and raises ``ValueError``.
2. *Never impute.* A rejected walk has no features at all.
3. *The left/right stride-time difference is a flag, not a filter.* On CARE-PD it is 1-2% (median)
   and does not depend on class, so it measures detection quality, not disease. Above
   ``LR_FLAG`` (25%) one foot probably has extra or missed heel strikes (video: 64% on a clip with
   a doubled foot, 25-28% on two severe-PD clips, 1-4% on clean clips). It needs at least
   ``LR_MIN_STRIDES`` real strides on each foot to be judged (with 2-3 the difference is noise,
   12-19% on clean clips); otherwise it is "not assessed". A flag is listed in ``flags`` and
   changes nothing else.
4. *Video: the longest usable stretch, no turn handling.* ``extract_from_trace`` takes the
   longest stretch of usable frames (GAIT-16) and uses the KOA-PD-NM stride minimum. Turns and
   pauses inside a long clip are not cut out; record the demo as one straight walk of 10 s or
   more.
5. *Values are rounded to ``DECIMALS`` (6) decimals here, once* (day-6 review, 14 Oct 2026). The
   training table stores 6 decimals, so without this the demo would feed the model slightly
   different numbers from the ones it was trained on: near a tree split that changed the class of
   5 of 2,738 CARE-PD walks. Rounding here makes a walk's ``vector()`` equal its table row exactly,
   and turns floating-point noise (a stride-time CV of 1e-14 for equal strides) into 0. Six
   decimals is far below what the measurements resolve.

Check values (``Checks``) are returned for the table and for the demo's warnings. They are never
model inputs: ``n_strides`` and ``leg_length_m`` carry the capture-area effect and the cohort.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from cycles import CAREPD_MIN_STRIDES, KOA_MIN_STRIDES, CycleCheck, segment_cycles
from gait_features import EXPERIMENTAL_NAMES, MIN_LEG_M, gait_features, leg_length
from gait_features import FEATURE_NAMES as GAIT_FEATURE_NAMES
from mediapipe_mapping import MappedWalk, walks_from_trace
from pose_probe import PoseTrace
from preprocess import detect_heel_strikes, preprocess
from timing_features import FEATURE_NAMES as TIMING_FEATURE_NAMES
from timing_features import timing_features

FEATURE_NAMES = (*TIMING_FEATURE_NAMES, *GAIT_FEATURE_NAMES)  # the eight model inputs, in order
MIN_FPS, MAX_FPS = 20.0, 60.0  # data runs at 25, 30 and 50 fps
LR_MIN_STRIDES = 3  # real strides needed on EACH foot to judge the left/right difference
LR_FLAG = 0.25  # flag a walk whose feet differ in median stride time by more than this share
DECIMALS = 6  # every feature value is rounded to this, once, so the demo equals the training table

Status = Literal["ok", "rejected"]


@dataclass(frozen=True)
class Checks:
    """Values for checking and warnings. Never model inputs."""

    n_strides: int
    n_left: int
    n_right: int
    leg_length_m: float
    lr_difference: float  # |left - right| / mean of the median stride times; nan = not assessed


@dataclass(frozen=True)
class WalkFeatures:
    status: Status
    reason: str = ""  # why a walk was rejected; empty when ok
    features: dict[str, float] = field(default_factory=dict)  # FEATURE_NAMES, in order, if ok
    checks: Checks | None = None  # None if the walk was rejected before they were known
    flags: tuple[str, ...] = ()  # warnings on an accepted walk
    experimental: dict[str, float] = field(default_factory=dict)  # not model inputs

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def vector(self) -> np.ndarray:
        """The eight inputs as an array in ``FEATURE_NAMES`` order (ok walks only)."""
        if not self.ok:
            raise ValueError(f"no features: walk rejected ({self.reason})")
        return np.array([self.features[n] for n in FEATURE_NAMES])


def rounded(values: dict[str, float]) -> dict[str, float]:
    """The values rounded to ``DECIMALS`` decimals, the precision of the training table."""
    return {name: round(float(v), DECIMALS) for name, v in values.items()}


def rejected(reason: str, checks: Checks | None = None) -> WalkFeatures:
    return WalkFeatures(status="rejected", reason=reason, checks=checks)


def lr_difference(check: CycleCheck) -> float:
    """Left/right difference in median stride time as a share of their mean.

    NaN (not assessed) unless each foot has at least ``LR_MIN_STRIDES`` real strides.
    """
    left, right = check.stride_times("left"), check.stride_times("right")
    if left.size < LR_MIN_STRIDES or right.size < LR_MIN_STRIDES:
        return math.nan
    a, b = float(np.median(left)), float(np.median(right))
    return abs(a - b) / ((a + b) / 2)


def extract_features(
    joints: np.ndarray, fps: float, *, min_strides: int = CAREPD_MIN_STRIDES
) -> WalkFeatures:
    """The eight model inputs of one walk, or a rejection with its reason.

    ``joints`` is ``(frames, 17, 3)`` in the h36m-style convention of CARE-PD (see
    ``mediapipe_mapping`` for video). ``min_strides`` is the cycle-check minimum (Rule B).
    """
    if not MIN_FPS <= fps <= MAX_FPS:
        raise ValueError(
            f"frame rate {fps} fps is outside {MIN_FPS:g}-{MAX_FPS:g}; read it from the data"
        )
    if min_strides < 2:
        raise ValueError("min_strides must be at least 2 (a variability needs two strides)")
    leg = leg_length(preprocess(joints, fps))  # also rejects a bad shape or NaN
    check = segment_cycles(detect_heel_strikes(joints, fps), min_strides)
    n_left, n_right = check.strides("left"), check.strides("right")
    checks = Checks(check.strides(), n_left, n_right, leg, lr_difference(check))
    if not check.passed:
        return rejected(check.reason, checks)
    if n_left == 0 or n_right == 0:
        return rejected("real strides on one foot only", checks)
    if leg < MIN_LEG_M:
        return rejected(f"leg length {leg:.2f} m is implausible", checks)

    timing = timing_features(check)
    gait = gait_features(joints, fps, check)
    values = {**timing.as_dict(), **gait.as_dict()}
    flags = []
    if not math.isnan(checks.lr_difference) and checks.lr_difference > LR_FLAG:
        flags.append(
            f"left and right stride times differ by {checks.lr_difference:.0%} "
            f"(over {LR_FLAG:.0%}): one foot may have extra or missed heel strikes"
        )
    return WalkFeatures(
        status="ok",
        features=rounded({name: values[name] for name in FEATURE_NAMES}),
        checks=checks,
        flags=tuple(flags),
        experimental=rounded({name: gait.experimental_dict()[name] for name in EXPERIMENTAL_NAMES}),
    )


def extract_from_trace(trace: PoseTrace) -> tuple[WalkFeatures, MappedWalk | None]:
    """Features of the longest usable stretch of a MediaPipe trace (video).

    Returns the result and the stretch used (None if there was none). Uses the KOA-PD-NM stride
    minimum, because the clips are short single passes.
    """
    walks = walks_from_trace(trace)
    if not walks:
        return rejected("no usable stretch (person not found, or legs not visible)"), None
    walk = max(walks, key=lambda w: w.joints.shape[0])
    if not walk.orientation_ok:
        return rejected(f"orientation check failed ({walk.facing_agreement:.0%})"), walk
    return extract_features(walk.joints, walk.fps, min_strides=KOA_MIN_STRIDES), walk
