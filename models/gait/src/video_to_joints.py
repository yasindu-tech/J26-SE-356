"""GAIT-25: a walking video in, the 17-joint walk out (what ``predict_video`` will call).

Runs MediaPipe (``pose_probe``), maps the longest usable stretch to the pipeline's joints
(``mediapipe_mapping``) and reports facts about the video for the quality gate (GAIT-26), which
decides whether to trust it; nothing is judged here. Only the first ``MAX_SECONDS`` are read by
default. The video must show one person: MediaPipe can merge a walker and a helper into one
skeleton, and nothing here detects that.

    python models/gait/src/video_to_joints.py VIDEO [--save walk.npz]

Exit code 0 = a walk was found, 2 = the video or the pose model cannot be used.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from carepd_format import short_id
from mediapipe_mapping import MIN_WALK_S, walks_from_trace
from pose_probe import (
    LEG_IDX,
    VISIBLE_THRESHOLD,
    PoseTrace,
    VideoInfo,
    probe_video,
    read_info,
    read_rgb_frames,
)

MAX_SECONDS = 30.0
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "artifacts" / "pose_landmarker_heavy.task"
FRAME_COUNT_SLACK = 2  # a container's frame count can be a frame or two off


class VideoError(ValueError):
    """The video cannot give a walk. The message says why, in plain words."""


@dataclass(frozen=True)
class VideoWalk:
    joints: np.ndarray  # (frames, 17, 3), h36m-style, metres: ready for the gait pipeline
    fps: float
    start_s: float  # where the walk begins in the clip
    end_s: float
    video_seconds: float  # whole clip, as the container reports it
    analysed_seconds: float  # how much of it was read
    truncated: bool  # True when the clip is longer than what was read
    width: int
    height: int
    person_found: float  # share of the analysed frames with a person
    stretches: int  # usable stretches found; the longest is the walk
    unused_seconds: float  # usable video in the other stretches
    filled_frames: int  # walk frames replaced by interpolation
    facing_agreement: float  # see ``mediapipe_mapping.facing_agreement``
    leg_visibility: (
        float  # share of the walk's leg landmarks clearly visible (not found counts as no)
    )

    @property
    def seconds(self) -> float:
        return self.joints.shape[0] / self.fps


def _clip_info(path: Path) -> VideoInfo:
    try:
        info = read_info(path)
    except ValueError as err:
        raise VideoError(
            "The video could not be opened (it may be damaged, or not a video file)."
        ) from err
    if info.fps <= 0 or info.n_frames <= 0:
        raise VideoError("The video has no readable frames or no frame rate.")
    return info


def _check_frame_size(path: Path, info: VideoInfo) -> None:
    """The size the file reports must be the size of its frames (a rotation flag can break it)."""
    frames = cast(Generator[np.ndarray, None, None], read_rgb_frames(path))  # it is a generator
    try:
        first = next(frames, None)
    finally:
        frames.close()
    if first is None:
        raise VideoError("No frame could be read from the video.")
    height, width = first.shape[:2]
    if (width, height) != (info.width, info.height):
        raise VideoError(
            f"The video's rotation setting does not match its picture ({info.width}x{info.height} "
            f"reported, {width}x{height} found), so it cannot be read reliably. Save or export "
            "the video in its normal orientation and try again."
        )


def video_to_joints(
    path: Path,
    model: Path = DEFAULT_MODEL,
    *,
    max_seconds: float | None = MAX_SECONDS,
    min_walk_s: float = MIN_WALK_S,
    on_trace: Callable[[PoseTrace], None] | None = None,
) -> VideoWalk:
    """The longest usable walk in ``path`` as joints, with facts about the video.

    ``max_seconds`` limits how much video is read (``None`` reads all of it; the heavy model
    runs at roughly 15 to 20 frames per second on a CPU). ``on_trace`` gets the pose trace as soon
    as it exists, even when no walk is found in it.
    """
    path, model = Path(path), Path(model)
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be positive or None")
    if not path.is_file() or path.stat().st_size == 0:
        raise VideoError("the video file is missing or empty")
    if not model.is_file():
        raise VideoError("the pose model file is missing")
    _check_frame_size(path, _clip_info(path))
    try:
        info, trace = probe_video(path, model, max_seconds=max_seconds)
    except ValueError as err:
        raise VideoError(str(err)) from err
    if on_trace is not None:
        on_trace(trace)

    found = float(trace.detected.mean())
    walks = walks_from_trace(trace, min_s=min_walk_s)
    if not walks:
        raise VideoError(
            f"No walk was found: no stretch of {min_walk_s:g} seconds or more showed one person's "
            "legs clearly. Film the whole body, head to feet, from the side."
        )
    walk = max(walks, key=lambda w: w.joints.shape[0])
    legs = trace.visibility[walk.start : walk.stop][:, list(LEG_IDX)]
    return VideoWalk(
        joints=walk.joints,
        fps=walk.fps,
        start_s=walk.start / walk.fps,
        end_s=walk.stop / walk.fps,
        video_seconds=info.duration_s,
        analysed_seconds=trace.n_frames / trace.fps,
        truncated=info.n_frames - trace.n_frames > FRAME_COUNT_SLACK,
        width=info.width,
        height=info.height,
        person_found=found,
        stretches=len(walks),
        unused_seconds=sum(w.seconds for w in walks if w is not walk),
        filled_frames=walk.filled_frames,
        facing_agreement=walk.facing_agreement,
        leg_visibility=float(np.mean(legs >= VISIBLE_THRESHOLD)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Video to joints (GAIT-25)")
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    parser.add_argument("--no-cap", action="store_true", help="read the whole video")
    parser.add_argument("--save", type=Path, help="write the joints and frame rate to this .npz")
    args = parser.parse_args(argv)

    tag = short_id(args.video.name)  # never the file name
    try:
        walk = video_to_joints(
            args.video, args.model, max_seconds=None if args.no_cap else args.max_seconds
        )
    except VideoError as err:
        print(f"{tag}: {err}")
        return 2
    cut = f", read the first {walk.analysed_seconds:.0f} s only" if walk.truncated else ""
    print(
        f"{tag}: {walk.width}x{walk.height}, {walk.fps:.0f} fps, {walk.video_seconds:.1f} s{cut}\n"
        f"walk {walk.seconds:.1f} s ({walk.start_s:.1f} to {walk.end_s:.1f} s); person found in "
        f"{walk.person_found:.0%} of frames; {walk.stretches} usable stretch(es), "
        f"{walk.unused_seconds:.1f} s unused; {walk.filled_frames} frames filled; "
        f"facing {walk.facing_agreement:.0%}; legs clear {walk.leg_visibility:.0%}"
    )
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save, joints=walk.joints, fps=walk.fps)
        print(f"Wrote {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
