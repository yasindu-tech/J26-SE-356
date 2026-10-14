"""GAIT-16: turn MediaPipe world landmarks into the 17-joint walk the gait pipeline expects.

The pipeline (``preprocess``, ``cycles``, the GAIT-14/15 features) works on CARE-PD walks: 17
joints in h36m order, metres, y up, +z the way the subject faces, +x the subject's left. This
module maps a MediaPipe Pose Landmarker trace (33 landmarks per frame) onto that format so a
video walk goes through the same code.

What is decided here, and why (measured on 8 KOA-PD-NM clips on 11 Oct 2026, see
``models/gait/docs/mediapipe_mapping.md``):

1. *World landmarks, not image coordinates.* They are in metres and centred on the hips (the
   pelvis never moves), and need no camera angle. Image coordinates have no real depth.
2. *Axes: ``(x, -y, -z)``.* MediaPipe's world y points DOWN. Flipping y alone is a mirror image:
   the skeleton is reflected, so the forward direction taken from the hips points backwards and
   step length turns negative without any error (toe ahead of heel along the hip-derived facing
   direction in 0-4% of frames). Flipping y and z together is a rotation (180 degrees about x):
   the same test gives 96-100%. ``facing_agreement`` repeats this test on every mapped walk.
3. *Missing or poor frames.* ``preprocess`` rejects NaN, and MediaPipe often loses the walker
   (entering the frame) or sees only a few leg landmarks. Frames count as usable when the person
   is found and at least 80% of the leg landmarks are visible (``pose_probe.clean_mask``). Gaps
   of up to ``MAX_GAP_S`` between usable frames are filled by linear interpolation; longer gaps
   split the clip. Each remaining stretch of at least ``MIN_WALK_S`` is one candidate walk.
   Choosing which stretch to analyse in a long clip is left to GAIT-17.
4. *Joints MediaPipe lacks.* Pelvis = midpoint of the hips, thorax = midpoint of the shoulders,
   spine = midpoint of pelvis and thorax, head = midpoint of the ears, neck = midpoint of thorax
   and head. Only the pelvis matters for the features; spine, neck and head are rough and would
   only matter for trunk lean, which is not used yet.

MediaPipe's body size is close to metric but not exact (median leg length 0.76-0.78 m on these
clips against 0.86 m in CARE-PD), which is why the length features are divided by leg length.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pose_probe import MAX_GAP_S, PoseTrace, clean_mask

N_LANDMARKS = 33
N_JOINTS = 17

# MediaPipe Pose landmark indices
L_EAR, R_EAR = 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_TOE, R_TOE = 31, 32

# MediaPipe world (x, y down, z) -> h36m-style (x, y up, z): a rotation, not a mirror.
AXES = np.array([1.0, -1.0, -1.0])

MIN_WALK_S = 2.0  # a stretch shorter than this cannot hold two strides
MIN_FACING_AGREEMENT = 0.9  # share of frames with the toe ahead of the heel (see module doc)


def _check_world(world: np.ndarray) -> np.ndarray:
    w = np.asarray(world, dtype=np.float64)
    if w.ndim != 3 or w.shape[1:] != (N_LANDMARKS, 3):
        raise ValueError(f"expected world landmarks of shape (frames, 33, 3), got {w.shape}")
    return w


def map_landmarks(world: np.ndarray) -> np.ndarray:
    """``(frames, 33, 3)`` MediaPipe world landmarks -> ``(frames, 17, 3)`` h36m-style joints.

    NaN stays NaN; filling gaps is ``walks_from_trace``'s job.
    """
    w = _check_world(world) * AXES
    pelvis = (w[:, L_HIP] + w[:, R_HIP]) / 2
    thorax = (w[:, L_SHOULDER] + w[:, R_SHOULDER]) / 2
    head = (w[:, L_EAR] + w[:, R_EAR]) / 2
    return np.stack(
        [
            pelvis,
            w[:, R_HIP],
            w[:, R_KNEE],
            w[:, R_ANKLE],
            w[:, L_HIP],
            w[:, L_KNEE],
            w[:, L_ANKLE],
            (pelvis + thorax) / 2,  # spine
            thorax,
            (thorax + head) / 2,  # neck
            head,
            w[:, L_SHOULDER],
            w[:, L_ELBOW],
            w[:, L_WRIST],
            w[:, R_SHOULDER],
            w[:, R_ELBOW],
            w[:, R_WRIST],
        ],
        axis=1,
    )


def facing_agreement(world: np.ndarray) -> float:
    """Share of frames in which the toes are ahead of the heels along the hip-derived facing.

    The pipeline takes the facing direction from the hip line (perpendicular to it, with +x the
    subject's left). If the axes were mirrored, it would point backwards and this share would be
    near 0. A correct mapping gives near 1. Uses the landmarks only, frame by frame, with NaN
    frames ignored. Returns NaN if there is no usable frame.
    """
    w = _check_world(world) * AXES
    hip = w[:, L_HIP] - w[:, R_HIP]
    side = np.stack([hip[:, 0], np.zeros(len(hip)), hip[:, 2]], axis=1)
    norm = np.linalg.norm(side, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        side = side / norm
    facing = np.stack([-side[:, 2], np.zeros(len(side)), side[:, 0]], axis=1)
    foot_l = ((w[:, L_TOE] - w[:, L_HEEL]) * facing).sum(axis=1)
    foot_r = ((w[:, R_TOE] - w[:, R_HEEL]) * facing).sum(axis=1)
    ahead = np.concatenate([foot_l, foot_r])
    ahead = ahead[np.isfinite(ahead)]
    return float(np.mean(ahead > 0)) if ahead.size else float("nan")


# --------------------------------------------------------------------------- usable stretches
@dataclass(frozen=True)
class Segment:
    start: int  # first frame
    stop: int  # one past the last frame

    @property
    def length(self) -> int:
        return self.stop - self.start


def _runs(flags: np.ndarray) -> list[tuple[int, int, bool]]:
    """Runs of equal values as ``(start, stop, value)``."""
    out: list[tuple[int, int, bool]] = []
    start = 0
    for i in range(1, len(flags) + 1):
        if i == len(flags) or flags[i] != flags[start]:
            out.append((start, i, bool(flags[start])))
            start = i
    return out


def usable_segments(
    usable: np.ndarray, fps: float, *, max_gap_s: float = MAX_GAP_S, min_s: float = MIN_WALK_S
) -> list[Segment]:
    """Stretches of usable frames, bridging gaps of at most ``max_gap_s`` seconds between them.

    A gap at the start or end of the clip is never bridged (there is nothing to interpolate
    from). Stretches shorter than ``min_s`` are dropped.
    """
    flags = np.asarray(usable, dtype=bool)
    if flags.size == 0:
        return []
    max_gap = int(max_gap_s * fps)
    bridged = flags.copy()
    runs = _runs(flags)
    for k, (a, b, value) in enumerate(runs):
        if not value and 0 < k < len(runs) - 1 and b - a <= max_gap:
            bridged[a:b] = True
    return [Segment(a, b) for a, b, value in _runs(bridged) if value and b - a >= int(min_s * fps)]


def fill_gaps(world: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """Replace the unusable frames of ``world`` by linear interpolation between usable ones.

    The first and last frame must be usable (``usable_segments`` guarantees it). Frames that
    were "found" but unusable are replaced too: their landmarks jump around.
    """
    w = _check_world(world).copy()
    good = np.asarray(usable, dtype=bool)
    if good.shape != (w.shape[0],) or not (good[0] and good[-1]):
        raise ValueError("usable must flag every frame, with the first and last frame usable")
    if good.all():
        return w
    t = np.arange(w.shape[0])
    flat = w.reshape(w.shape[0], -1)
    for col in range(flat.shape[1]):
        flat[~good, col] = np.interp(t[~good], t[good], flat[good, col])
    return flat.reshape(w.shape)


@dataclass(frozen=True)
class MappedWalk:
    joints: np.ndarray  # (frames, 17, 3), h36m-style, ready for the gait pipeline
    fps: float
    start: int  # first frame in the clip
    stop: int  # one past the last frame in the clip
    filled_frames: int  # frames replaced by interpolation
    facing_agreement: float  # see ``facing_agreement``

    @property
    def seconds(self) -> float:
        return self.joints.shape[0] / self.fps

    @property
    def orientation_ok(self) -> bool:
        return bool(self.facing_agreement >= MIN_FACING_AGREEMENT)


def walks_from_trace(trace: PoseTrace, *, min_s: float = MIN_WALK_S) -> list[MappedWalk]:
    """Every usable stretch of a MediaPipe trace as a walk, in time order (maybe none)."""
    usable = clean_mask(trace)
    walks = []
    for seg in usable_segments(usable, trace.fps, min_s=min_s):
        sl = slice(seg.start, seg.stop)
        flags = usable[sl]
        world = fill_gaps(trace.world[sl], flags)
        walks.append(
            MappedWalk(
                joints=map_landmarks(world),
                fps=trace.fps,
                start=seg.start,
                stop=seg.stop,
                filled_frames=int((~flags).sum()),
                facing_agreement=facing_agreement(world),
            )
        )
    return walks
