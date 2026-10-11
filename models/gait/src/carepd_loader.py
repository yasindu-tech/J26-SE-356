"""GAIT-11: load the CARE-PD h36m walks and join them to subject, cohort and UPDRS gait label.

Run from the repo root (venv active):

    python models/gait/src/carepd_loader.py

It prints how many walks were loaded per cohort and severity class, checks the totals against
the GAIT-08 label counts, and saves one skeleton plot to ``models/gait/artifacts/loader``
(git-ignored). Exit code 0 = loaded and counts match, 1 = counts do not match GAIT-08,
2 = a file is missing.

What a ``Walk`` holds: the skeleton ``(frames, 17, 3)`` in metres (y up, floor at 0, walking
towards +z), the cohort, subject and walk name, the frame rate from the config, the raw UPDRS
gait score (0 to 3) and the model class (0, 1, 2-3). Nothing else from the label files is
read: the ``medication`` and ``other`` fields, and the SMPL fields, are never touched, so they
cannot leak into model inputs (CLAUDE.md section 3.1). The UPDRS gait score is carried only as
the label (``score`` and ``severity``) and must never be turned into an input feature.

BMCLab stores each walk five times (``_down0`` to ``_down4``, a 150 fps recording downsampled
at five offsets). By default only the first copy is loaded, so each real walk counts once and
the counts match GAIT-08. ``all_copies=True`` loads all five; use it only for augmentation
inside training folds.

Subject IDs are never printed (CLAUDE.md section 4); walks are shown by a short hash.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from carepd_format import (
    N_JOINTS,
    FormatConfig,
    load_config,
    load_label_pickle,
    short_id,
    split_key,
)
from count_labels import ClassCount, count_cohort, usable_walks

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "carepd_format.toml"
DEFAULT_DATA = REPO_ROOT / "data" / "raw" / "CARE-PD"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "artifacts" / "loader"

_COPY_SUFFIX = re.compile(r"_down(\d+)$")
LABEL_FIELD_FPS = "fps"  # the only other label-file field the loader reads (a consistency check)
MIN_PLOT_FRAMES = 90  # prefer a walk of at least 3 s at 30 fps for the example plot

# h36m bones as (joint, joint) names; indices come from the config's joint_order.
BONES = (
    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("pelvis", "spine"),
    ("spine", "thorax"),
    ("thorax", "neck"),
    ("neck", "head"),
    ("thorax", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("thorax", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
)


@dataclass(frozen=True)
class Walk:
    cohort: str
    subject: str  # unique only within a cohort; use ``group`` for splits
    walk: str  # walk name without the _downK suffix
    copy: int | None  # which _downK copy this is; None if the cohort has no copies
    fps: int
    joints: np.ndarray  # (frames, 17, 3), metres, y up
    score: int  # raw UPDRS gait score 0..3: a LABEL, never an input
    severity: int  # model class 0, 1, 2 (2 means "2-3"): a LABEL, never an input

    @property
    def n_frames(self) -> int:
        return int(self.joints.shape[0])

    @property
    def group(self) -> str:
        """Subject key for subject-wise splits; includes the cohort because IDs repeat."""
        return f"{self.cohort}:{self.subject}"

    @property
    def tag(self) -> str:
        """Short hash that is safe to print instead of the subject / walk names."""
        return short_id(f"{self.group}__{self.walk}")


@dataclass(frozen=True)
class CohortReport:
    name: str
    n_walks: int
    n_subjects: int
    severity: dict[int, ClassCount]  # model class -> subjects and walks actually loaded
    copies_dropped: int  # extra _downK copies not loaded
    unlabelled: int  # skeletons whose label has no UPDRS gait score
    skeleton_without_label: int  # skeletons with no entry in the label file
    labelled_without_skeleton: int  # labelled walks CARE-PD dropped (shorter than 30 frames)
    fps_mismatch: int  # walks whose label-file fps is not the value the config expects
    matches_gait08: bool | None  # None when all copies were loaded (counts then differ)


# --------------------------------------------------------------------------- helpers
def copy_index(key: str) -> int | None:
    m = _COPY_SUFFIX.search(key)
    return int(m.group(1)) if m else None


def class_counts(walks: list[Walk], *, raw: bool = False) -> dict[int, ClassCount]:
    """Subjects and walks per class (model class, or raw score if ``raw``)."""
    subjects: dict[int, set[str]] = {}
    n_walks: dict[int, int] = {}
    for w in walks:
        c = w.score if raw else w.severity
        subjects.setdefault(c, set()).add(w.group)
        n_walks[c] = n_walks.get(c, 0) + 1
    return {c: ClassCount(len(subjects[c]), n_walks[c]) for c in sorted(subjects)}


def _score_and_fps(fields: Any, label_field: str) -> tuple[int | None, int | None]:
    """Read only the UPDRS gait score and fps from one walk's label fields."""
    value = fields.get(label_field)
    fps = fields.get(LABEL_FIELD_FPS)
    return (
        None if value is None else int(value),
        None if fps is None else int(round(float(fps))),
    )


# --------------------------------------------------------------------------- loading
def load_cohort(
    name: str, data_dir: Path, cfg: FormatConfig, *, all_copies: bool = False
) -> tuple[list[Walk], CohortReport]:
    """Load one cohort's walks, joined to their label. Raises FileNotFoundError if a file is missing."""
    cohort = cfg.cohorts[name]
    world_path = data_dir / "h36m" / name / cohort.world_file
    label_path = data_dir / cohort.label_file
    for p in (world_path, label_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    labels = load_label_pickle(label_path)
    label_of: dict[tuple[str, str], tuple[int | None, int | None]] = {}
    for subject, walks_of in labels.items():
        for walk_name, fields in walks_of.items():
            label_of[(str(subject), walk_name)] = _score_and_fps(fields, cfg.labels.field)

    walks: list[Walk] = []
    copies_dropped = unlabelled = no_label = fps_mismatch = 0
    with np.load(world_path, allow_pickle=False) as npz:
        keys = list(npz.files)
        index: dict[tuple[str, str], dict[int, str]] = {}
        for key in keys:
            idx = copy_index(key)
            index.setdefault(split_key(key), {})[-1 if idx is None else idx] = key
        gait08 = count_cohort(name, labels, usable_walks(keys), cfg)

        for (subject, walk_name), copies in sorted(index.items()):
            if (subject, walk_name) not in label_of:
                no_label += 1
                continue
            score, label_fps = label_of[(subject, walk_name)]
            if score is None:
                unlabelled += 1
                continue
            if score not in cfg.labels.class_map:
                raise ValueError(f"{name}: score {score} is not in the class map")
            chosen = sorted(copies) if all_copies else [min(copies)]
            copies_dropped += len(copies) - len(chosen)
            if label_fps is not None and label_fps != (cohort.label_fps or cohort.fps):
                fps_mismatch += 1
            for c in chosen:
                joints = np.asarray(npz[copies[c]])
                if joints.ndim != 3 or joints.shape[1:] != (N_JOINTS, 3):
                    raise ValueError(f"{name}: expected (frames, 17, 3), got {joints.shape}")
                walks.append(
                    Walk(
                        cohort=name,
                        subject=subject,
                        walk=walk_name,
                        copy=None if c == -1 else c,
                        fps=cohort.fps,
                        joints=joints,
                        score=score,
                        severity=cfg.labels.class_map[score],
                    )
                )

    labelled_without_skeleton = sum(
        1 for k, (s, _) in label_of.items() if s is not None and k not in index
    )
    severity = class_counts(walks)
    matches = (
        None
        if all_copies
        else severity == gait08.merged and class_counts(walks, raw=True) == gait08.raw
    )
    report = CohortReport(
        name=name,
        n_walks=len(walks),
        n_subjects=len({w.group for w in walks}),
        severity=severity,
        copies_dropped=copies_dropped,
        unlabelled=unlabelled,
        skeleton_without_label=no_label,
        labelled_without_skeleton=labelled_without_skeleton,
        fps_mismatch=fps_mismatch,
        matches_gait08=matches,
    )
    return walks, report


def load_all(
    data_dir: Path,
    cfg: FormatConfig,
    cohorts: list[str] | None = None,
    *,
    all_copies: bool = False,
) -> tuple[list[Walk], list[CohortReport]]:
    """Load several cohorts (all configured ones by default)."""
    walks: list[Walk] = []
    reports: list[CohortReport] = []
    for name in cohorts or list(cfg.cohorts):
        w, r = load_cohort(name, data_dir, cfg, all_copies=all_copies)
        walks.extend(w)
        reports.append(r)
    return walks, reports


# --------------------------------------------------------------------------- plotting
def pick_walk_for_plot(walks: list[Walk]) -> Walk:
    """A reproducible example: the median-length walk of at least 3 s, in loading order."""
    long_enough = [w for w in walks if w.n_frames >= MIN_PLOT_FRAMES] or walks
    if not long_enough:
        raise ValueError("no walks to plot")
    ranked = sorted(long_enough, key=lambda w: w.n_frames)
    return ranked[len(ranked) // 2]


def plot_walk(walk: Walk, cfg: FormatConfig, out_png: Path, n_snapshots: int = 9) -> None:
    """Side view of the skeleton at evenly spaced frames, plus both ankle heights over time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = {j: i for i, j in enumerate(cfg.skeleton.joint_order)}
    j = walk.joints  # x lateral, y up, z forward
    t = np.arange(walk.n_frames) / walk.fps
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(11, 7.5), gridspec_kw={"height_ratios": [2, 1]})

    frames = np.linspace(0, walk.n_frames - 1, n_snapshots).round().astype(int)
    for rank, f in enumerate(frames):
        shade = 0.25 + 0.75 * rank / max(len(frames) - 1, 1)
        for a, b in BONES:
            ax_a.plot(
                [j[f, order[a], 2], j[f, order[b], 2]],
                [j[f, order[a], 1], j[f, order[b], 1]],
                color=(0.1, 0.25, 0.6, shade),
                lw=1.6,
            )
    ax_a.axhline(cfg.coordinates.floor_y, color="gray", lw=0.8, ls="--")
    ax_a.set_aspect("equal")
    ax_a.set_xlabel("distance walked, towards +z (m)")
    ax_a.set_ylabel("height (m)")
    ax_a.set_title(
        f"{walk.cohort} walk {walk.tag}: UPDRS gait {walk.score}, "
        f"{walk.fps} fps, {walk.n_frames} frames ({n_snapshots} snapshots, later = darker)"
    )

    ax_b.plot(t, j[:, order["left_ankle"], 1], color="tab:blue", lw=1, label="left ankle")
    ax_b.plot(t, j[:, order["right_ankle"], 1], color="tab:red", lw=1, label="right ankle")
    ax_b.set_xlabel("time (s)")
    ax_b.set_ylabel("ankle height (m)")
    ax_b.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- command line
def _cell(c: ClassCount | None) -> str:
    return f"{c.walks} / {c.subjects}" if c else "-"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CARE-PD loader (GAIT-11)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--all-copies", action="store_true", help="load all five BMCLab copies of each walk"
    )
    parser.add_argument("--no-plot", action="store_true", help="skip the skeleton plot")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    try:
        walks, reports = load_all(args.data_dir, cfg, all_copies=args.all_copies)
    except FileNotFoundError as err:
        print(f"MISSING: {err}")
        return 2

    which = "all copies" if args.all_copies else "first copy of each walk"
    print(f"CARE-PD h36m walks joined to UPDRS gait labels ({which})")
    print("Cells show walks / subjects. Model classes: 0, 1, 2 (= scores 2 and 3 merged).\n")
    print(
        f"{'cohort':<10}{'walks':>7}{'subjects':>9}{'class 0':>12}{'class 1':>12}{'class 2-3':>12}"
        f"{'copies dropped':>16}{'label, no skel.':>16}{'fps differs':>13}"
    )
    for r in reports:
        print(
            f"{r.name:<10}{r.n_walks:>7}{r.n_subjects:>9}"
            f"{_cell(r.severity.get(0)):>12}{_cell(r.severity.get(1)):>12}"
            f"{_cell(r.severity.get(2)):>12}{r.copies_dropped:>16}"
            f"{r.labelled_without_skeleton:>16}{r.fps_mismatch:>13}"
        )
    pooled = class_counts(walks)
    print(
        f"{'ALL':<10}{len(walks):>7}{len({w.group for w in walks}):>9}"
        f"{_cell(pooled.get(0)):>12}{_cell(pooled.get(1)):>12}{_cell(pooled.get(2)):>12}"
    )
    print(
        "\nNot loaded: the label files' medication and other fields, and the SMPL fields. "
        "The UPDRS score is a label only."
    )
    problems = [
        f"  {r.name}: skeletons without a label {r.skeleton_without_label}, "
        f"without a UPDRS score {r.unlabelled}"
        for r in reports
        if r.skeleton_without_label or r.unlabelled
    ]
    if problems:
        print("\nWalks skipped (expected: none):\n" + "\n".join(problems))

    status = 0
    if args.all_copies:
        print("\nCounts are not compared with GAIT-08 because extra copies were loaded.")
    else:
        bad = [r.name for r in reports if not r.matches_gait08]
        if bad:
            print(f"\nCounts DO NOT match GAIT-08 for: {', '.join(bad)}. The join is wrong.")
            status = 1
        else:
            print("\nCounts match GAIT-08 for every cohort.")

    if not args.no_plot and walks:
        pick = pick_walk_for_plot(walks)
        out = args.out_dir / "example_walk.png"
        plot_walk(pick, cfg, out)
        print(
            f"Saved skeleton plot: {out} (walk {pick.tag}, {pick.cohort}, "
            f"UPDRS gait {pick.score}, {pick.n_frames} frames)"
        )
    return status


if __name__ == "__main__":
    sys.exit(main())
