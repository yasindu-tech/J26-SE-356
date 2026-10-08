"""GAIT-09 / Check C: does MediaPipe find a walker in the KOA-PD-NM videos filmed at 8 m?

Run from the repo root (venv active):

    python models/gait/src/check_mediapipe_clips.py

By default it uses one healthy, one mild-PD and one severe-PD clip from
``data/raw/KOA-PD-NM``. Pick other clips with ``--clip ROLE=PATH`` (repeat it), and if
detection is poor try ``--follow`` (a crop that follows the walker) or
``--crop x0,y0,x1,y1`` (fractions of the frame). Each clip is printed by its role and a short
hash, never by file name. Plots and a results file go to ``models/gait/artifacts/check_c``
(git-ignored); no video frames are saved.

Pass mark (the plan's suggestion): person found in at least 90% of frames and at least 2 gait
cycles in the world ankle trace. Exit code 0 = all clips pass, 1 = a clip failed,
2 = a file is missing.

The heavy model runs on the CPU; allow roughly a minute or two per clip.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from carepd_format import short_id
from pose_probe import (
    CLEAN_LEG_VISIBILITY,
    CYCLES_PASS,
    DETECTION_PASS,
    MIN_CLEAN_S,
    ClipMetrics,
    evaluate,
    plot_clip,
    probe_video,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = REPO_ROOT / "data" / "raw" / "KOA-PD-NM"
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "artifacts" / "pose_landmarker_heavy.task"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "artifacts" / "check_c"

# One clip per role, relative to the data folder. Chosen from the file sizes and lengths
# measured earlier (single passes, 4 to 20 s); change them freely with --clip.
DEFAULT_CLIPS = {
    "healthy": "NM/001_NM_01.MOV",
    "mild": "PD/PD_ML/004_PD_01_ML.MOV",
    "severe": "PD/PD_SV/001_PD_01_SV.MOV",
}


def parse_clip(text: str) -> tuple[str, Path]:
    role, sep, path = text.partition("=")
    if not sep or not role or not path:
        raise argparse.ArgumentTypeError("use ROLE=PATH, for example healthy=data/raw/.../clip.MOV")
    return role, Path(path)


def parse_crop(text: str) -> tuple[float, float, float, float]:
    try:
        x0, y0, x1, y1 = (float(v) for v in text.split(","))
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            "use x0,y0,x1,y1 as fractions, e.g. 0.1,0.2,0.9,1.0"
        ) from err
    return x0, y0, x1, y1


def metrics_to_dict(
    role: str, tag: str, mode: str, fps: float, size: str, m: ClipMetrics
) -> dict[str, object]:
    return {
        "role": role,
        "id": tag,
        "mode": mode,
        "fps": fps,
        "size": size,
        "frames": m.n_frames,
        "detection_rate": round(m.detection_rate, 4),
        "first_found_s": None if m.first_found_s is None else round(m.first_found_s, 2),
        "in_view_rate": round(m.in_view_rate, 4),
        "longest_gap_s": round(m.longest_gap_s, 2),
        "passed_in_view": m.passed_in_view,
        "clean_s": round(m.clean_s, 2),
        "raw_world_cycles": m.raw_world_cycles,
        "visible_all": round(m.visible_all, 4),
        "visible_legs": round(m.visible_legs, 4),
        "world_cycles": {"left": m.world_left.cycles, "right": m.world_right.cycles},
        "image_cycles": {"left": m.image_left.cycles, "right": m.image_right.cycles},
        "interval_cv_world": [m.world_left.interval_cv, m.world_right.interval_cv],
        "passed": m.passed,
        "reasons": list(m.reasons),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MediaPipe on KOA-PD-NM clips (GAIT-09, Check C)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--clip", type=parse_clip, action="append", help="ROLE=PATH (repeatable)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--follow", action="store_true", help="crop that follows the moving walker")
    group.add_argument("--crop", type=parse_crop, help="fixed crop, fractions x0,y0,x1,y1")
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

    mode = "follow-crop" if args.follow else "fixed crop" if args.crop else "full frame"
    slug = {"follow-crop": "follow", "fixed crop": "crop", "full frame": "full"}[mode]
    print(f"MediaPipe Pose Landmarker (heavy, video mode), {mode}")
    print(
        f"pass mark: found in >= {DETECTION_PASS:.0%} of frames, >= {MIN_CLEAN_S:.0f} s of clean "
        f"frames and >= {CYCLES_PASS} gait cycles"
    )
    print(
        f"clean = person found and >= {CLEAN_LEG_VISIBILITY:.0%} of leg landmarks visible; "
        "cycles are counted on clean frames only ('raw' = cycles with no such filter)."
    )
    print("'in view' counts only frames from the first to the last frame the person is found.\n")
    header = (
        f"{'role':<9}{'id':<12}{'fps':>5} {'size':<10}{'sec':>6}{'frames':>7}"
        f"{'found':>7}{'first(s)':>9}{'in view':>8}{'legs vis':>9}{'clean(s)':>9}"
        f"{'cycles L/R':>11}{'raw':>5}  "
        f"{'image L/R':<10}{'result':<7}in view"
    )
    print(header)

    rows: list[dict[str, object]] = []
    n_whole = n_in_view = 0
    failures: list[str] = []
    for role, path in clips:
        tag = short_id(path.name)
        try:
            info, trace = probe_video(path, args.model, crop=args.crop, follow=args.follow)
        except ValueError as err:
            print(f"{role:<9}{tag:<12}  could not process: {err}")
            continue
        m = evaluate(trace)
        size = f"{info.width}x{info.height}"
        first = "never" if m.first_found_s is None else f"{m.first_found_s:.1f}"
        print(
            f"{role:<9}{tag:<12}{trace.fps:>5.0f} {size:<10}{m.n_frames / trace.fps:>6.1f}{m.n_frames:>7}"
            f"{m.detection_rate:>7.0%}{first:>9}{m.in_view_rate:>8.0%}{m.visible_legs:>9.0%}"
            f"{m.clean_s:>9.1f}{m.world_left.cycles:>6}/{m.world_right.cycles:<4}{m.raw_world_cycles:>4}  "
            f"{m.image_left.cycles}/{m.image_right.cycles:<8}"
            f"{'PASS' if m.passed else 'FAIL':<7}{'PASS' if m.passed_in_view else 'FAIL'}"
        )
        plot_clip(trace, m, f"{role} ({tag})", args.out_dir / f"check_c_{role}_{slug}.png")
        rows.append(metrics_to_dict(role, tag, mode, trace.fps, size, m))
        n_whole += m.passed
        n_in_view += m.passed_in_view
        if not m.passed:
            failures.append(
                f"  {role}: "
                + "; ".join(m.reasons)
                + f" (longest gap once in view: {m.longest_gap_s:.1f} s)"
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"check_c_results_{slug}.json").write_text(json.dumps(rows, indent=2))
    print(f"\nPlots and results saved in {args.out_dir} (files end in _{slug})")
    if failures:
        print("\nFailed the plan's mark:")
        print("\n".join(failures))
    print(
        f"\nCheck C, plan's mark on the whole clip: {n_whole}/{len(clips)} pass"
        f"\nCheck C, counted from when the person is found: {n_in_view}/{len(clips)} pass"
    )
    return 0 if n_whole == len(clips) else 1


if __name__ == "__main__":
    sys.exit(main())
