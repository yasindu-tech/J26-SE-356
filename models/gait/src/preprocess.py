"""GAIT-12: pelvis-centre and smooth a walk, then find heel strikes.

Works on any h36m 17-joint walk ``(frames, 17, 3)`` in metres with y up: CARE-PD skeletons now,
and MediaPipe landmarks mapped to the same 17 joints later (GAIT-16). That is why heel strikes
are found on the ankle height *relative to the pelvis*: MediaPipe's world landmarks are
hip-centred and have no floor, so a floor-based detector would only work on CARE-PD.

Steps:

1. ``centre_on_pelvis``: subtract the pelvis from every joint, so the walk is translation-free.
2. ``smooth``: zero-phase low-pass Butterworth filter (4th order, 6 Hz). Zero-phase, so event
   times do not move; the cutoff is in Hz, so it adapts to 25 or 30 fps.
3. ``detect_heel_strikes``: per foot, find the dips of the ankle height with SciPy
   ``find_peaks`` on the inverted signal. Two dips of the same foot must be at least
   ``MIN_STRIDE_S`` apart (converted to frames with the walk's fps), and a dip must be deep
   relative to the walk's own range, so noise is not counted.

Where in the dip is the heel strike? The ankle is lowest around mid-stance, after the foot has
landed. With ``method="onset"`` (the default) each strike is moved back from the minimum to the
end of the fast downward movement, which is the moment the foot lands. ``method="minimum"``
keeps the plain minimum. All times are frame indices; divide by fps for seconds.

No starting value here is tuned on the UPDRS labels (that would leak the label into the
pipeline). They are typical gait values, checked against the overlay plots.

Checked on CARE-PD (9 Oct 2026, ``check_heel_strikes.py``): median stride times equal those of a
coordinate-based method (ankle furthest ahead of the pelvis) in all four cohorts, and our strikes
come 2 to 3 frames before it. With ``forward`` given (``detect_heel_strikes`` does this), two
landings of one foot also need a stance between them, which removes dips in mid-swing (3% of
strikes on UPDRS 2 walks, 20% on UPDRS 3, almost none on UPDRS 0-1). Known limit: on UPDRS 2-3
walks about 4-8% of strikes still have no match in the coordinate method (under 0.5% for UPDRS
0-1); part of this is the reference failing on shuffling steps. GAIT-13 (``cycles.py``) then
counts only real strides, which drops stepping on the spot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from carepd_format import L_ANKLE, L_HIP, PELVIS, R_ANKLE, R_HIP
from scipy.signal import butter, filtfilt, find_peaks

SMOOTH_CUTOFF_HZ = 6.0  # common low-pass cutoff for walking kinematics
SMOOTH_ORDER = 4
MIN_STRIDE_S = 0.5  # two strikes of the same foot are at least this far apart
REL_DEPTH = 0.25  # a dip must be at least this share of the trace's 5th-95th percentile range
ABS_DEPTH_M = 0.005  # and at least this deep in metres (5 mm keeps severe shuffling steps)
FIRST_DROP_FRACTION = 0.5  # the first landing of a walk must drop at least this share of a
#                            typical dip; otherwise it is a slow drift, not a step
FIRST_SPEED_FRACTION = 0.5  # ... and end a descent at least this share as fast as a typical one
FIRST_SETTLE_FRAMES = 1  # ... and come before its dip's lowest point (not on it)
ONSET_SPEED_FRACTION = 0.2  # the foot has landed once its downward speed drops below this share
#                             of the fastest downward speed in that dip

STANCE_BACK_M = 0.05  # between two landings of one foot there is a stance, in which the ankle
#                       moves back relative to the pelvis by at least this much

Method = Literal["onset", "minimum"]


@dataclass(frozen=True)
class FootStrikes:
    strikes: np.ndarray  # frame index of each heel strike, ascending
    minima: np.ndarray  # frame index of each dip's lowest point (same length as strikes)

    def stride_times(self, fps: float) -> np.ndarray:
        """Seconds between consecutive strikes of this foot."""
        return np.diff(self.strikes) / fps


@dataclass(frozen=True)
class HeelStrikes:
    left: FootStrikes
    right: FootStrikes
    fps: float
    method: str
    # Each ankle's position ahead of the pelvis along the facing direction (metres, per frame).
    # Used by GAIT-13 to tell real steps from stepping on the spot; None if not computed.
    left_forward: np.ndarray | None = None
    right_forward: np.ndarray | None = None

    def all_stride_times(self) -> np.ndarray:
        return np.concatenate([self.left.stride_times(self.fps), self.right.stride_times(self.fps)])


# --------------------------------------------------------------------------- preprocessing
def _check_walk(joints: np.ndarray) -> np.ndarray:
    j = np.asarray(joints, dtype=np.float64)
    if j.ndim != 3 or j.shape[1:] != (17, 3):
        raise ValueError(f"expected a walk of shape (frames, 17, 3), got {j.shape}")
    if not np.isfinite(j).all():
        raise ValueError("walk contains NaN or inf; fill gaps before preprocessing")
    return j


def centre_on_pelvis(joints: np.ndarray) -> np.ndarray:
    """Every joint relative to the pelvis in the same frame (pelvis becomes the origin)."""
    j = _check_walk(joints)
    return j - j[:, PELVIS : PELVIS + 1, :]


def smooth(joints: np.ndarray, fps: float, cutoff_hz: float = SMOOTH_CUTOFF_HZ) -> np.ndarray:
    """Zero-phase low-pass filter along time for every joint and axis."""
    j = _check_walk(joints)
    nyquist = fps / 2
    if not 0 < cutoff_hz < nyquist:
        raise ValueError(f"cutoff {cutoff_hz} Hz must be between 0 and {nyquist} Hz at {fps} fps")
    b, a = butter(SMOOTH_ORDER, cutoff_hz / nyquist)
    pad = 3 * max(len(a), len(b))
    if j.shape[0] <= pad:
        raise ValueError(f"walk of {j.shape[0]} frames is too short to filter (need > {pad})")
    return np.asarray(filtfilt(b, a, j, axis=0))


def preprocess(joints: np.ndarray, fps: float) -> np.ndarray:
    """Pelvis-centred, smoothed walk: what every later step uses."""
    return smooth(centre_on_pelvis(joints), fps)


def ankle_forward(prepared: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Left and right ankle position ahead of the pelvis, along the direction the hips face.

    The facing direction is horizontal and perpendicular to the hip line (h36m convention: +x is
    the subject's left, y up), so it follows turns and needs no global position: it works on
    hip-centred MediaPipe landmarks too.
    """
    hip = prepared[:, L_HIP] - prepared[:, R_HIP]
    facing = np.stack([-hip[:, 2], np.zeros(len(hip)), hip[:, 0]], axis=1)
    norm = np.linalg.norm(facing, axis=1, keepdims=True)
    if (norm < 1e-6).any():
        raise ValueError("left and right hip coincide; the facing direction is undefined")
    facing /= norm
    left = (prepared[:, L_ANKLE] * facing).sum(axis=1)
    right = (prepared[:, R_ANKLE] * facing).sum(axis=1)
    return left, right


def ankle_heights(prepared: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Left and right ankle height relative to the pelvis (metres, up positive)."""
    return prepared[:, L_ANKLE, 1], prepared[:, R_ANKLE, 1]


# --------------------------------------------------------------------------- heel strikes
def _onset(height: np.ndarray, minimum: int, start: int) -> int | None:
    """Walk back from a dip's minimum to the frame where the fast descent ended.

    Returns None when the descent is not inside the walk (the walk starts mid-dip), because the
    landing then happened before the first frame and cannot be placed.
    """
    v = np.diff(height)  # v[k] = height[k+1] - height[k]; negative = moving down
    seg = v[start:minimum]
    if seg.size == 0 or seg.min() >= 0:
        return None
    fastest = int(np.argmin(seg)) + start
    threshold = ONSET_SPEED_FRACTION * v[fastest]  # negative number
    k = fastest
    while k + 1 < minimum and v[k + 1] < threshold:
        k += 1
    return k + 1


def detect_foot(
    height: np.ndarray,
    fps: float,
    method: Method = "onset",
    abs_depth_m: float = ABS_DEPTH_M,
    forward: np.ndarray | None = None,
) -> FootStrikes:
    """Heel strikes of one foot from its ankle height (any vertical offset, y up).

    ``abs_depth_m`` is the smallest dip counted, in metres; noisier input (video) may need more.
    ``forward`` is the ankle's position ahead of the pelvis (``ankle_forward``). If given, two
    landings of one foot need a stance between them (the ankle moving back relative to the
    pelvis); a dip in the middle of a swing, with no stance before the next one, is dropped.
    """
    x = np.asarray(height, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("ankle height contains NaN or inf")
    if x.size < 3:
        return FootStrikes(np.array([], dtype=int), np.array([], dtype=int))
    n = x.size
    lo, hi = np.percentile(x, [5, 95])
    depth = max(abs_depth_m, REL_DEPTH * (hi - lo))
    gap = max(int(round(MIN_STRIDE_S * fps)), 1)
    # Pretend the foot lifts again just after the last frame, so a walk that ends during stance
    # keeps its last landing (find_peaks only accepts a dip that rises on both sides). The
    # landing itself must still lie inside the walk; see the onset check below.
    padded = np.concatenate([x, np.full(gap, x.max())])
    minima, props = find_peaks(-padded, distance=gap, prominence=depth)
    typical = float(np.median(props["prominences"])) if minima.size else 0.0
    # Fastest downward speed before each dip; a real landing ends a fast descent.
    v = np.diff(x)
    speeds = [
        -float(v[lb : min(int(m), n - 1)].min())
        for m, lb in zip(minima, props["left_bases"], strict=True)
        if min(int(m), n - 1) > lb
    ]
    typical_speed = float(np.median([sp for sp in speeds if sp > 0])) if speeds else 0.0
    strikes: list[int] = []
    kept: list[int] = []
    for i, (m_pad, left_base) in enumerate(zip(minima, props["left_bases"], strict=True)):
        m = min(int(m_pad), n - 1)
        first = i == 0  # the walk's first dip: nothing before it shows a full swing
        start = max(int(left_base), kept[-1]) if kept else min(int(left_base), m)
        # A landing needs a swing before it: the ankle must have been at least `depth` higher
        # since the previous dip (or since the walk started). This merges two dips of one
        # stance and drops a slow drift at the start of a walk.
        if x[start : m + 1].max() - x[m] < depth:
            continue
        # A walk that starts high and drifts down is not a step. Nothing before the first
        # landing shows the swing in full, so it must drop at least half a typical dip of this
        # foot (a real first swing does; a slow drift at the start does not).
        if first and x[start : m + 1].max() - x[m] < FIRST_DROP_FRACTION * typical:
            continue
        # A walk can also start with the foot already in stance, slowly lowering: that dip is
        # mid-stance, not a landing. The first landing must end a descent at least half as fast
        # as this foot's typical one.
        if first and (
            m <= start or -float(v[start:m].min()) < FIRST_SPEED_FRACTION * typical_speed
        ):
            continue
        if method == "minimum":
            if m < n - 1:  # the last frame is not a real minimum
                strikes.append(m)
                kept.append(m)
            continue
        onset = _onset(x, m, start)
        # A real landing is followed by stance, where the ankle keeps settling, so it comes
        # before the dip's lowest point. A first "landing" on the lowest point itself is a walk
        # that started mid-stance.
        if onset is not None and first and m - onset < FIRST_SETTLE_FRAMES:
            continue
        # One landing per swing: between two landings of the same foot there must be a stance,
        # in which the ankle moves back relative to the pelvis. If it did not, the previous
        # "landing" was a dip in mid-swing; drop it and keep this one.
        if (
            onset is not None
            and forward is not None
            and strikes
            and onset > strikes[-1]
            and forward[strikes[-1]] - forward[strikes[-1] : onset + 1].min() < STANCE_BACK_M
        ):
            strikes.pop()
            kept.pop()
        # The landing must be inside the walk with at least one frame after it; a walk that
        # ends mid-descent has not landed yet.
        if onset is not None and onset < n - 1 and (not strikes or onset > strikes[-1]):
            strikes.append(onset)
            kept.append(m)
    return FootStrikes(np.array(strikes, dtype=int), np.array(kept, dtype=int))


def detect_heel_strikes(joints: np.ndarray, fps: float, method: Method = "onset") -> HeelStrikes:
    """Preprocess a raw walk and find heel strikes of both feet."""
    prepared = preprocess(joints, fps)
    left, right = ankle_heights(prepared)
    left_fwd, right_fwd = ankle_forward(prepared)
    return HeelStrikes(
        left=detect_foot(left, fps, method, forward=left_fwd),
        right=detect_foot(right, fps, method, forward=right_fwd),
        fps=fps,
        method=method,
        left_forward=left_fwd,
        right_forward=right_fwd,
    )
