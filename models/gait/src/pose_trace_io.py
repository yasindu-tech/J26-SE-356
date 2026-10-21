"""Save and load a pose trace, so a later step can use it without running the pose model again."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pose_probe import PoseTrace


def save_trace(trace: PoseTrace, path: Path) -> None:
    """Write the trace to ``path`` (an .npz file); full precision, so results match exactly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        fps=trace.fps,
        width=trace.width,
        height=trace.height,
        detected=trace.detected,
        visibility=trace.visibility,
        image=trace.image,
        world=trace.world,
    )


def load_trace(path: Path) -> PoseTrace:
    with np.load(path) as f:
        return PoseTrace(
            float(f["fps"]),
            int(f["width"]),
            int(f["height"]),
            f["detected"],
            f["visibility"],
            f["image"],
            f["world"],
        )
