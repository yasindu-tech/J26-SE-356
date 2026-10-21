"""A quick look at a few frames before the full analysis, so a video that is not a walk is
refused in seconds instead of after minutes of pose estimation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from pose_probe import (
    CLEAN_LEG_VISIBILITY,
    LEG_IDX,
    VISIBLE_THRESHOLD,
    landmarks_from_result,
    read_info,
)

SAMPLES = 12  # frames looked at, spread over the part of the video that will be read
# At least this share of them must show the legs clearly (a walker can leave the frame).
MIN_SHARE = 0.15


def sample_indices(
    n_frames: int, fps: float, max_seconds: float | None, count: int = SAMPLES
) -> list[int]:
    """Frame numbers at the middle of ``count`` equal slices of the part that will be read."""
    last = n_frames if max_seconds is None else min(n_frames, round(max_seconds * fps))
    if last <= 0:
        return []
    count = min(count, last)
    return sorted({int(i * last / count + last / (2 * count)) for i in range(count)})


def refusal(legs_clear: list[bool]) -> str | None:
    """The plain reason to refuse the video, or None when enough frames show the legs."""
    if not legs_clear:
        return None
    seen = sum(legs_clear)
    if seen >= max(1, math.ceil(MIN_SHARE * len(legs_clear))):
        return None
    found = (
        f"no person with the legs in view was found in any of the {len(legs_clear)} frames checked"
        if seen == 0
        else f"a person with the legs in view was found in only {seen} of the "
        f"{len(legs_clear)} frames checked"
    )
    return (
        f"This does not look like a video of someone walking: {found}. The video must show one "
        "person walking, filmed from the side, head to feet."
    )


def prescreen(path: Path, model: Path, max_seconds: float | None = None) -> str | None:
    """A reason to refuse the video, or None to go on. Never refuses a video it could not read:
    the full analysis reports those itself."""
    try:
        return refusal(_legs_clear(Path(path), Path(model), max_seconds))
    except (ValueError, OSError, RuntimeError):
        return None


def _legs_clear(path: Path, model: Path, max_seconds: float | None) -> list[bool]:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

    info = read_info(path)
    if info.fps <= 0 or info.n_frames <= 0:
        return []
    wanted = set(sample_indices(info.n_frames, info.fps, max_seconds))
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
    )
    clear: list[bool] = []
    cap = cv2.VideoCapture(str(path))
    try:
        with PoseLandmarker.create_from_options(options) as landmarker:
            for index in range(max(wanted, default=-1) + 1):
                if index not in wanted:
                    if not cap.grab():
                        break
                    continue
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                found = landmarks_from_result(
                    landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                )
                legs = found is not None and (
                    float(np.mean(found[0][list(LEG_IDX)] >= VISIBLE_THRESHOLD))
                    >= CLEAN_LEG_VISIBILITY
                )
                clear.append(legs)
    finally:
        cap.release()
    return clear


def main(argv: list[str] | None = None) -> int:
    """``python src/prescreen.py VIDEO...``: how many of the frames checked show the legs."""
    import argparse

    from carepd_format import short_id
    from video_to_joints import DEFAULT_MODEL, MAX_SECONDS

    parser = argparse.ArgumentParser(description="Quick pre-check of walking videos")
    parser.add_argument("videos", type=Path, nargs="+")
    args = parser.parse_args(argv)
    status = 0
    for video in args.videos:
        try:
            clear = _legs_clear(video, DEFAULT_MODEL, MAX_SECONDS)
        except (ValueError, OSError, RuntimeError) as err:
            print(f"{short_id(video.name)}: could not be read ({err})")
            continue
        reason = refusal(clear)
        status = status or int(reason is not None)
        print(f"{short_id(video.name)}: legs clear in {sum(clear)} of {len(clear)} frames: "
              f"{'REFUSED' if reason else 'goes on'}")  # fmt: skip
    return status


if __name__ == "__main__":
    raise SystemExit(main())
