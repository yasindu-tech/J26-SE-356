"""GAIT-25: ``video_to_joints`` must give the joints the GAIT-16 path gives for the same clip.

    python models/gait/src/check_video_to_joints.py [--clip ROLE=PATH ...]

For each clip it runs ``video_to_joints`` and compares its joints with the longest walk of the
clip's cached MediaPipe trace (``check_mediapipe_mapping``; the trace is made if missing). The
two must agree to within 1 mm. Default clips: normal-1 and pd-ml (a minute or two each).

Exit code 0 = they agree, 1 = they differ, 2 = a file is missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from carepd_format import short_id
from check_mediapipe_mapping import (
    DEFAULT_CLIPS,
    DEFAULT_DATA,
    DEFAULT_MODEL,
    DEFAULT_OUT,
    load_or_run,
    parse_clip,
)
from mediapipe_mapping import walks_from_trace
from video_to_joints import MAX_SECONDS, VideoError, video_to_joints

TOLERANCE_M = 1e-3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="video_to_joints against the GAIT-16 path")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--clip", type=parse_clip, action="append", help="ROLE=PATH (repeatable)")
    args = parser.parse_args(argv)

    clips = args.clip or [(r, args.data_dir / DEFAULT_CLIPS[r]) for r in ("normal-1", "pd-ml")]
    missing = [p for p in [args.model, *[c[1] for c in clips]] if not p.is_file()]
    if missing:
        for p in missing:
            print(f"MISSING: {p}")
        return 2

    status = 0
    for role, path in clips:
        tag = short_id(path.name)
        try:
            new = video_to_joints(path, args.model, max_seconds=MAX_SECONDS)
        except VideoError as err:
            print(f"{role:<10}{tag:<12}video_to_joints: {err}")
            status = 1
            continue
        cache = args.out_dir / f"trace_{tag}_{int(MAX_SECONDS)}s.npz"
        trace, _ = load_or_run(path, args.model, cache, MAX_SECONDS, False)
        old = max(walks_from_trace(trace), key=lambda w: w.joints.shape[0])
        if new.joints.shape != old.joints.shape:
            print(f"{role:<10}{tag:<12}DIFFERENT shapes {new.joints.shape} and {old.joints.shape}")
            status = 1
            continue
        worst = float(np.nanmax(np.abs(new.joints - old.joints)))
        same = worst <= TOLERANCE_M
        status = status or int(not same)
        print(
            f"{role:<10}{tag:<12}{new.seconds:5.1f} s walk, {new.stretches} stretch(es), "
            f"found {new.person_found:.0%}; largest difference {worst:.2g} m  "
            f"{'same' if same else 'DIFFERENT'}"
        )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
