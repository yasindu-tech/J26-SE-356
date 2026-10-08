"""Read and sanity-check the CARE-PD h36m world-coordinate files (GAIT-07, Check A).

The rules about axes, units and joint order live in
``models/gait/configs/carepd_format.toml``. This module loads that config and
checks real arrays against it, so a wrong assumption fails loudly here instead
of silently corrupting every gait feature later.
"""

from __future__ import annotations

import hashlib
import pickle
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, field_validator

N_JOINTS = 17
_DOWN_SUFFIX = re.compile(r"_down\d+$")

# Joint indices in the h36m order (checked against joint_order in the config).
PELVIS, R_HIP, R_KNEE, R_ANKLE, L_HIP, L_KNEE, L_ANKLE = 0, 1, 2, 3, 4, 5, 6
HEAD = 10


# --------------------------------------------------------------------------- config
class Skeleton(BaseModel):
    format: str
    joint_order: list[str]

    @field_validator("joint_order")
    @classmethod
    def _seventeen_joints(cls, v: list[str]) -> list[str]:
        if len(v) != N_JOINTS or len(set(v)) != N_JOINTS:
            raise ValueError(f"joint_order must list {N_JOINTS} distinct joints, got {len(v)}")
        return v


class Coordinates(BaseModel):
    units: str
    up_axis: str
    forward_axis: str
    lateral_axis: str
    floor_y: float


class Checks(BaseModel):
    n_sample_walks: int
    min_frames: int
    unit_bounds_height_m: tuple[float, float]
    cohort_median_height_m: tuple[float, float]
    floor_tolerance_m: float
    min_hip_separation_m: float
    min_fraction_pass: float


class Labels(BaseModel):
    field: str
    class_map: dict[int, int]


class Cohort(BaseModel):
    fps: int
    world_file: str
    label_file: str
    has_down_copies: bool


class FormatConfig(BaseModel):
    skeleton: Skeleton
    coordinates: Coordinates
    checks: Checks
    labels: Labels
    cohorts: dict[str, Cohort]


def load_config(path: Path) -> FormatConfig:
    """Load and validate the TOML config; a typo in a key raises immediately."""
    with path.open("rb") as f:
        raw = tomllib.load(f)
    return FormatConfig.model_validate(raw)


# --------------------------------------------------------------------------- keys
def split_key(key: str) -> tuple[str, str]:
    """Return ``(subject, walk)`` for ``subject__walk[_downK]`` (the copy suffix is dropped)."""
    subject, sep, walk = key.partition("__")
    if not sep or not subject or not walk:
        raise ValueError(f"not a 'subject__walk' key: {key!r}")
    return subject, _DOWN_SUFFIX.sub("", walk)


def short_id(key: str) -> str:
    """Short hash for printing, so subject IDs never appear in terminal logs (CLAUDE.md §4)."""
    return "w-" + hashlib.sha1(key.encode()).hexdigest()[:8]


def sample_keys(keys: list[str], n: int) -> list[str]:
    """Pick ``n`` keys spread evenly over the sorted keys. Deterministic, no cherry-picking."""
    ordered = sorted(keys)
    if n >= len(ordered):
        return ordered
    idx = np.linspace(0, len(ordered) - 1, n).round().astype(int)
    return [ordered[i] for i in idx]


# --------------------------------------------------------------------------- per-walk checks
@dataclass(frozen=True)
class WalkMetrics:
    n_frames: int
    height_m: float  # median head height above the lower ankle
    floor_offset_m: float  # lowest ankle point over the whole walk (should be ~0)
    knee_minus_hip_m: float  # median knee y minus hip y (negative when Y is up)
    hip_separation_m: float  # median (left hip x - right hip x) (positive when +x is left)
    net_forward_m: float  # pelvis displacement along +z from first to last frame


def walk_metrics(walk: np.ndarray) -> WalkMetrics:
    """Compute the orientation and unit metrics for one (frames, 17, 3) walk."""
    a = np.asarray(walk, dtype=np.float64)
    ankle_y = a[:, [R_ANKLE, L_ANKLE], 1]
    return WalkMetrics(
        n_frames=a.shape[0],
        height_m=float(np.median(a[:, HEAD, 1] - ankle_y.min(axis=1))),
        floor_offset_m=float(ankle_y.min()),
        knee_minus_hip_m=float(
            np.median(a[:, [R_KNEE, L_KNEE], 1].mean(axis=1) - a[:, [R_HIP, L_HIP], 1].mean(axis=1))
        ),
        hip_separation_m=float(np.median(a[:, L_HIP, 0] - a[:, R_HIP, 0])),
        net_forward_m=float(a[-1, PELVIS, 2] - a[0, PELVIS, 2]),
    )


def walk_violations(walk: np.ndarray, cfg: FormatConfig) -> list[str]:
    """Return a list of broken rules for one walk (empty list means it passes)."""
    c = cfg.checks
    problems: list[str] = []
    if walk.ndim != 3 or walk.shape[1:] != (N_JOINTS, 3):
        return [f"shape {walk.shape} is not (frames, {N_JOINTS}, 3)"]
    if not np.isfinite(walk).all():
        return ["contains NaN or inf"]
    if walk.shape[0] < c.min_frames:
        problems.append(f"only {walk.shape[0]} frames (< {c.min_frames})")
    m = walk_metrics(walk)
    lo, hi = c.unit_bounds_height_m
    if not lo <= m.height_m <= hi:
        problems.append(f"height {m.height_m:.2f} m outside {lo}-{hi} m (wrong units?)")
    if abs(m.floor_offset_m - cfg.coordinates.floor_y) > c.floor_tolerance_m:
        problems.append(f"lowest ankle at y={m.floor_offset_m:+.2f} m, floor should be y=0")
    if m.knee_minus_hip_m >= 0:
        problems.append("knees are not below hips (Y not up, or joint order wrong)")
    if m.hip_separation_m <= c.min_hip_separation_m:
        problems.append("left hip is not at larger x than right hip (left/right swapped?)")
    return problems


# --------------------------------------------------------------------------- cohort checks
@dataclass(frozen=True)
class CohortSummary:
    n_entries: int
    n_walks: int
    n_subjects: int
    median_height_m: float
    frac_floor_ok: float
    frac_knee_ok: float
    frac_hip_ok: float
    frac_forward_ok: float


def summarise_cohort(walks: dict[str, np.ndarray], cfg: FormatConfig) -> CohortSummary:
    """Aggregate the orientation metrics over every walk in a cohort file."""
    c = cfg.checks
    metrics = [walk_metrics(w) for w in walks.values()]
    splits = [split_key(k) for k in walks]
    return CohortSummary(
        n_entries=len(walks),
        n_walks=len(set(splits)),
        n_subjects=len({s for s, _ in splits}),
        median_height_m=float(np.median([m.height_m for m in metrics])),
        frac_floor_ok=float(
            np.mean(
                [
                    abs(m.floor_offset_m - cfg.coordinates.floor_y) <= c.floor_tolerance_m
                    for m in metrics
                ]
            )
        ),
        frac_knee_ok=float(np.mean([m.knee_minus_hip_m < 0 for m in metrics])),
        frac_hip_ok=float(np.mean([m.hip_separation_m > c.min_hip_separation_m for m in metrics])),
        frac_forward_ok=float(np.mean([m.net_forward_m > 0 for m in metrics])),
    )


def cohort_violations(s: CohortSummary, cfg: FormatConfig) -> list[str]:
    """Return the cohort-level rules that fail (empty list means the cohort passes)."""
    c = cfg.checks
    problems: list[str] = []
    lo, hi = c.cohort_median_height_m
    if not lo <= s.median_height_m <= hi:
        problems.append(f"median height {s.median_height_m:.2f} m outside {lo}-{hi} m")
    for name, frac in (
        ("lowest ankle at floor", s.frac_floor_ok),
        ("knees below hips (Y up)", s.frac_knee_ok),
        ("left hip at larger x (+x = left)", s.frac_hip_ok),
        ("net movement towards +z", s.frac_forward_ok),
    ):
        if frac < c.min_fraction_pass:
            problems.append(f"{name}: only {frac:.1%} of walks (need {c.min_fraction_pass:.0%})")
    return problems


# --------------------------------------------------------------------------- labels
class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Unpickler that refuses anything except numpy arrays and plain containers.

    ``pickle`` can run arbitrary code when loading a file. The CARE-PD label
    files only need numpy, so anything else is blocked instead of executed.
    """

    _PLAIN = {("collections", "OrderedDict"), ("builtins", "set"), ("builtins", "frozenset")}

    def find_class(self, module: str, name: str) -> Any:
        if module.split(".")[0] == "numpy" or (module, name) in self._PLAIN:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def load_label_pickle(path: Path) -> dict[Any, dict[str, dict[str, Any]]]:
    """Load a CARE-PD cohort ``.pkl`` (subject -> walk -> fields) with the restricted unpickler."""
    with path.open("rb") as f:
        data = _NumpyOnlyUnpickler(f).load()
    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: expected a dict of subjects, got {type(data).__name__}")
    return data


@dataclass(frozen=True)
class LabelCoverage:
    n_h36m_walks: int
    n_label_walks: int
    h36m_without_label: int  # must be 0, otherwise the label join is broken
    labels_without_h36m: int  # expected: walks CARE-PD dropped for being too short
    class_counts: dict[str, int]  # walk-level, after the 2-3 merge; 'unlabelled' if None


def label_coverage(
    h36m_keys: list[str], labels: dict[Any, dict[str, dict[str, Any]]], cfg: FormatConfig
) -> LabelCoverage:
    """Match h36m walks to label-file walks by (subject, walk) and count severity classes."""
    wanted = {split_key(k) for k in h36m_keys}
    have = {(str(subj), walk) for subj, walks in labels.items() for walk in walks}
    counts: dict[str, int] = {}
    for walks in labels.values():
        for fields in walks.values():
            raw = fields.get(cfg.labels.field)
            cls = "unlabelled" if raw is None else str(cfg.labels.class_map[int(raw)])
            counts[cls] = counts.get(cls, 0) + 1
    return LabelCoverage(
        n_h36m_walks=len(wanted),
        n_label_walks=len(have),
        h36m_without_label=len(wanted - have),
        labels_without_h36m=len(have - wanted),
        class_counts=dict(sorted(counts.items())),
    )
