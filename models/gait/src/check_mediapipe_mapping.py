"""GAIT-16: run KOA-PD-NM clips through MediaPipe, the mapping and the gait pipeline.

Run from the repository root (venv active)::

    python models/gait/src/check_mediapipe_mapping.py

For each clip (default: three normal walkers and three PD clips; pick others with
``--clip ROLE=PATH``, repeatable) it runs the MediaPipe Pose Landmarker on the first
``--max-seconds`` of video (default 30; the heavy model is slow on a CPU, roughly a minute or
two per clip), maps the landmarks to 17 joints (``mediapipe_mapping``), takes the longest
usable stretch as the walk, finds heel strikes and cycles, and computes the GAIT-14 and
GAIT-15 features. The landmarks of each clip are cached in ``artifacts/mediapipe`` (git-ignored)
so a re-run does not repeat MediaPipe; ``--rerun`` ignores the cache.

Checks (exit 1 if one fails, for a clip that has a walk):

- orientation: the toe is ahead of the heel along the hip-derived facing in at least 90% of
  frames (a mirrored mapping gives about 0%);
- leg length from the mapped joints is between 0.6 and 1.1 m.

KOA-PD-NM has no severity labels, so this shows that the pipeline runs sensibly on video, not
that the model is accurate. One plot per clip is written to ``artifacts/mediapipe`` for an eye
check: ankle height over time with the detected heel strikes marked. Look at each plot and
confirm the markers sit on real steps. Clips are shown by role and a short hash, never by file
name, and no video frames are saved.

Exit code 0 = all clips with a walk pass, 1 = a check failed, 2 = a file is missing.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from carepd_format import short_id
from cycles import KOA_MIN_STRIDES, CycleCheck, segment_cycles
from gait_features import FEATURE_NAMES as GAIT_FEATURES
from gait_features import gait_features, leg_length
from mediapipe_mapping import MappedWalk, walks_from_trace
from pose_probe import PoseTrace, probe_video
from preprocess import ankle_heights, detect_heel_strikes, preprocess
from timing_features import FEATURE_NAMES as TIMING_FEATURES
from timing_features import timing_features

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = REPO_ROOT / "data" / "raw" / "KOA-PD-NM"
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "artifacts" / "pose_landmarker_heavy.task"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "artifacts" / "mediapipe"
# Single short passes (4 to 20 s), relative to the data folder; change them with --clip.
DEFAULT_CLIPS = {
    "normal-1": "NM/001_NM_01.MOV",
    "normal-2": "NM/029_NM_02.MOV",
    "normal-3": "NM/003_NM_02.MOV",
    "pd-ml": "PD/PD_ML/002_PD_01_ML.MOV",
    "pd-md": "PD/PD_MD/004_PD_02_MD.MOV",
    "koa-el": "KOA/KOA_EL/009_KOA_01_EL.MOV",
}
DEFAULT_MAX_SECONDS = 30.0
LEG_LENGTH_M = (0.6, 1.1)  # plausible leg length of an adult in the mapped joints
TRACE_FIELDS = ("detected", "visibility", "image", "world")


def parse_clip(text: str) -> tuple[str, Path]:
    role, sep, path = text.partition("=")
    if not sep or not role or not path:
        raise argparse.ArgumentTypeError("use ROLE=PATH, for example normal=data/raw/.../clip.MOV")
    return role, Path(path)


def load_or_run(
    path: Path, model: Path, cache: Path, max_seconds: float, rerun: bool
) -> tuple[PoseTrace, bool]:
    """The clip's MediaPipe trace, from the cache when there is one. Returns (trace, cached)."""
    if cache.is_file() and not rerun:
        with np.load(cache) as z:
            return (
                PoseTrace(
                    fps=float(z["fps"]),
                    width=int(z["width"]),
                    height=int(z["height"]),
                    **{name: z[name] for name in TRACE_FIELDS},
                ),
                True,
            )
    _, trace = probe_video(path, model, max_seconds=max_seconds)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        fps=trace.fps,
        width=trace.width,
        height=trace.height,
        **{name: getattr(trace, name) for name in TRACE_FIELDS},
    )
    return trace, False


def plot_eye_check(walk: MappedWalk, check: CycleCheck, title: str, out_png: Path) -> None:
    """Ankle height (relative to the pelvis) with the heel strikes of the real strides marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prepared = preprocess(walk.joints, walk.fps)
    left, right = ankle_heights(prepared)
    t = np.arange(len(left)) / walk.fps
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t, left, label="left ankle", lw=1.2)
    ax.plot(t, right, label="right ankle", lw=1.2)
    for foot, colour, series in (("left", "tab:blue", left), ("right", "tab:orange", right)):
        strikes = sorted(
            {f for c in check.cycles if c.plausible and c.foot == foot for f in (c.start, c.end)}
        )
        ax.plot(
            np.array(strikes) / walk.fps,
            series[strikes],
            "v",
            color=colour,
            ms=8,
            label=f"{foot} heel strike (real strides)",
        )
    ax.set_xlabel("time in this stretch (s)")
    ax.set_ylabel("height rel. to pelvis (m)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, ncols=2)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MediaPipe mapping on KOA-PD-NM (GAIT-16)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--clip", type=parse_clip, action="append", help="ROLE=PATH (repeatable)")
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    parser.add_argument("--rerun", action="store_true", help="ignore cached MediaPipe results")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    clips = (
        args.clip
        if args.clip
        else [(role, args.data_dir / rel) for role, rel in DEFAULT_CLIPS.items()]
    )
    missing = [
        p for p in [args.model, *[c[1] for c in clips]] if not p.is_file() or p.stat().st_size == 0
    ]
    if missing:
        for p in missing:
            print(f"MISSING or empty: {p}")
        return 2

    print(f"MediaPipe mapping check on {len(clips)} clips (first {args.max_seconds:.0f} s each).")
    print("Walk = longest stretch of usable frames; strides need >= 2 (short KOA-PD-NM clips).\n")
    print(
        f"{'role':<10}{'id':<12}{'fps':>4}{'found':>7}{'walks':>6}{'sec':>6}{'filled':>7}"
        f"{'facing':>8}{'leg m':>7}{'strides':>8}{'stride s':>9}  result"
    )
    status = 0
    rows: list[list[object]] = []
    for role, path in clips:
        tag = short_id(path.name)
        cache = args.out_dir / f"trace_{tag}_{int(args.max_seconds)}s.npz"
        try:
            trace, _ = load_or_run(path, args.model, cache, args.max_seconds, args.rerun)
        except ValueError as err:
            print(f"{role:<10}{tag:<12}  could not process: {err}")
            continue
        walks = walks_from_trace(trace)
        found = float(trace.detected.mean())
        head = f"{role:<10}{tag:<12}{trace.fps:>4.0f}{found:>7.0%}{len(walks):>6}"
        if not walks:
            print(head + "  no usable stretch (person not found, or legs not visible)")
            continue
        walk = max(walks, key=lambda w: w.joints.shape[0])
        leg = leg_length(preprocess(walk.joints, walk.fps))
        check = segment_cycles(detect_heel_strikes(walk.joints, walk.fps), KOA_MIN_STRIDES)
        problems = []
        if not walk.orientation_ok:
            problems.append("mirrored or backwards")
        if not LEG_LENGTH_M[0] <= leg <= LEG_LENGTH_M[1]:
            problems.append("leg length off")
        stride_s = float("nan")
        values: dict[str, float] = {}
        if check.strides() >= KOA_MIN_STRIDES:
            timing = timing_features(check)
            stride_s = timing.stride_time_mean
            values = {**timing.as_dict(), **gait_features(walk.joints, walk.fps, check).as_dict()}
        else:
            problems.append(f"{check.strides()} real strides")
        verdict = "ok" if not problems else "; ".join(problems)
        print(
            head + f"{walk.seconds:>6.1f}{walk.filled_frames:>7}{walk.facing_agreement:>8.0%}"
            f"{leg:>7.2f}{check.strides():>8}{stride_s:>9.2f}  {verdict}"
        )
        if problems and (not walk.orientation_ok or "leg length off" in problems):
            status = 1
        if not args.no_plot:
            plot_eye_check(walk, check, f"{role} ({tag})", args.out_dir / f"eye_check_{role}.png")
        rows.append(
            [role, tag, f"{trace.fps:.0f}", f"{walk.seconds:.2f}", check.strides()]
            + [
                f"{values[n]:.4f}" if n in values else ""
                for n in (*TIMING_FEATURES, *GAIT_FEATURES)
            ]
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "mapped_clip_features.csv"
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["role", "id", "fps", "seconds", "strides", *TIMING_FEATURES, *GAIT_FEATURES]
        )
        writer.writerows(rows)
    print(f"\nFeatures per clip (no labels: plausibility only): {out}")
    if not args.no_plot:
        print(f"Eye-check plots: {args.out_dir}/eye_check_<role>.png (look at each one)")
    return status


if __name__ == "__main__":
    sys.exit(main())
