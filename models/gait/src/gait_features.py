"""GAIT-15: step, joint-range and arm-swing features from the real strides of one walk.

Five model features (``FEATURE_NAMES``), all from pelvis-relative positions so they exist on
CARE-PD and on MediaPipe video alike. Each is a median over the real strides of GAIT-13 (both
feet), so it does not depend on the stride count:

- ``step_length_norm``: landing foot ahead of the other foot at heel strike, in leg lengths.
- ``step_length_asymmetry``: ``|left - right| / mean`` of the median step length of each foot.
- ``knee_flexion_range_deg``, ``hip_flexion_range_deg``: knee angle and thigh swing ranges
  within a stride, mean of both sides.
- ``arm_swing_range_norm``: forward-back wrist swing about the shoulder, in leg lengths.

Step width and arm swing asymmetry are computed but are not model inputs
(``EXPERIMENTAL_NAMES``). Leg length only scales and is returned for checks, like the stride
count; it is never a feature. A walk with no real strides, strides on one foot only, or an
implausible leg length (under ``MIN_LEG_M``) raises ``ValueError``; GAIT-17 only calls this for
walks that pass the cycle check.

The reasons, the ablation and the results are in ``models/gait/docs/gait_features.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from carepd_format import L_ANKLE, L_HIP, L_KNEE, R_ANKLE, R_HIP, R_KNEE
from cycles import CycleCheck
from preprocess import ankle_forward, preprocess

# h36m indices not exported by carepd_format (see its joint order)
L_SHOULDER, L_WRIST, R_SHOULDER, R_WRIST = 11, 13, 14, 16

FEATURE_NAMES = (
    "step_length_norm",
    "step_length_asymmetry",
    "knee_flexion_range_deg",
    "hip_flexion_range_deg",
    "arm_swing_range_norm",
)
EXPERIMENTAL_NAMES = ("step_width_norm", "arm_swing_asymmetry")  # reported, never model inputs
MIN_LEG_M = 0.2  # a shorter leg means broken joints or wrong units (metres expected)


@dataclass(frozen=True)
class GaitFeatures:
    step_length_norm: float  # share of leg length
    step_length_asymmetry: float  # 0 = equal steps with both feet
    knee_flexion_range_deg: float
    hip_flexion_range_deg: float
    arm_swing_range_norm: float  # share of leg length
    # experimental: not model inputs
    step_width_norm: float
    arm_swing_asymmetry: float
    # checks: not model inputs
    leg_length_m: float
    n_strides: int

    def as_dict(self) -> dict[str, float]:
        """The model inputs only."""
        return {name: getattr(self, name) for name in FEATURE_NAMES}

    def experimental_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in EXPERIMENTAL_NAMES}


# --------------------------------------------------------------------------- geometry
def _length(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(np.linalg.norm(a - b, axis=-1))


def leg_length(prepared: np.ndarray) -> float:
    """Median over frames of the mean hip-knee + knee-ankle length of both legs (metres)."""
    left = _length(prepared[:, L_HIP], prepared[:, L_KNEE]) + _length(
        prepared[:, L_KNEE], prepared[:, L_ANKLE]
    )
    right = _length(prepared[:, R_HIP], prepared[:, R_KNEE]) + _length(
        prepared[:, R_KNEE], prepared[:, R_ANKLE]
    )
    return float(np.median((left + right) / 2))


def interior_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at ``b`` between the segments b->a and b->c, in degrees, per frame."""
    u, v = a - b, c - b
    cos = (u * v).sum(axis=-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-12)
    return np.asarray(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def facing_and_side(prepared: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal unit vectors, per frame: the way the hips face, and the subject's left."""
    hip = prepared[:, L_HIP] - prepared[:, R_HIP]
    side = np.stack([hip[:, 0], np.zeros(len(hip)), hip[:, 2]], axis=1)
    norm = np.linalg.norm(side, axis=1, keepdims=True)
    if (norm < 1e-6).any():
        raise ValueError("left and right hip coincide; the facing direction is undefined")
    side = side / norm
    facing = np.stack([-side[:, 2], np.zeros(len(side)), side[:, 0]], axis=1)
    return facing, side


def thigh_angle(hip: np.ndarray, knee: np.ndarray, facing: np.ndarray) -> np.ndarray:
    """Thigh angle from the vertical in the walking direction, degrees per frame (+ = forward)."""
    seg = knee - hip
    forward = (seg * facing).sum(axis=1)
    return np.asarray(np.degrees(np.arctan2(forward, -seg[:, 1])))


def _asymmetry(a: float, b: float) -> float:
    mean = (abs(a) + abs(b)) / 2
    return abs(a - b) / mean if mean > 1e-9 else 0.0


# --------------------------------------------------------------------------- features
def gait_features(joints: np.ndarray, fps: float, check: CycleCheck) -> GaitFeatures:
    """Features of one walk from its joints and its cycle check (real strides, both feet)."""
    cycles = [c for c in check.cycles if c.plausible]
    if not cycles:
        raise ValueError("no real strides; gait features need at least one")
    strikes: dict[str, set[int]] = {"left": set(), "right": set()}
    for c in cycles:
        strikes[c.foot].update((c.start, c.end))
    if not strikes["left"] or not strikes["right"]:
        raise ValueError("real strides on one foot only; step asymmetry needs both feet")

    p = preprocess(joints, fps)
    leg = leg_length(p)
    if leg < MIN_LEG_M:
        raise ValueError(f"leg length {leg:.2f} m is implausible; check the joints and the units")
    facing, side = facing_and_side(p)
    fwd_left, fwd_right = ankle_forward(p)

    steps: dict[str, list[float]] = {"left": [], "right": []}
    widths: list[float] = []
    for foot, own, other in (("left", fwd_left, fwd_right), ("right", fwd_right, fwd_left)):
        for s in sorted(strikes[foot]):
            steps[foot].append(float(own[s] - other[s]))
            across = p[s, L_ANKLE] - p[s, R_ANKLE]
            widths.append(abs(float(across @ side[s])))
    step_left, step_right = float(np.median(steps["left"])), float(np.median(steps["right"]))
    step = float(np.median(steps["left"] + steps["right"]))

    signals = {
        "knee_l": interior_angle(p[:, L_HIP], p[:, L_KNEE], p[:, L_ANKLE]),
        "knee_r": interior_angle(p[:, R_HIP], p[:, R_KNEE], p[:, R_ANKLE]),
        "hip_l": thigh_angle(p[:, L_HIP], p[:, L_KNEE], facing),
        "hip_r": thigh_angle(p[:, R_HIP], p[:, R_KNEE], facing),
        "arm_l": ((p[:, L_WRIST] - p[:, L_SHOULDER]) * facing).sum(axis=1),
        "arm_r": ((p[:, R_WRIST] - p[:, R_SHOULDER]) * facing).sum(axis=1),
    }
    ranges = {
        name: float(np.median([np.ptp(sig[c.start : c.end + 1]) for c in cycles]))
        for name, sig in signals.items()
    }
    arm_mean = (ranges["arm_l"] + ranges["arm_r"]) / 2
    return GaitFeatures(
        step_length_norm=step / leg,
        step_length_asymmetry=_asymmetry(step_left, step_right),
        knee_flexion_range_deg=(ranges["knee_l"] + ranges["knee_r"]) / 2,
        hip_flexion_range_deg=(ranges["hip_l"] + ranges["hip_r"]) / 2,
        arm_swing_range_norm=arm_mean / leg,
        step_width_norm=float(np.median(widths)) / leg,
        arm_swing_asymmetry=_asymmetry(ranges["arm_l"], ranges["arm_r"]),
        leg_length_m=leg,
        n_strides=len(cycles),
    )
