"""A saved pose trace comes back exactly as it was."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pose_trace_io import load_trace, save_trace
from test_gait_features import full_walk
from test_mediapipe_mapping import mediapipe_world, trace_of


def test_a_trace_round_trips_exactly_including_missing_frames(tmp_path: Path) -> None:
    world = mediapipe_world(full_walk(seconds=4.0))
    det = np.ones(world.shape[0], dtype=bool)
    det[20:30] = False
    trace = trace_of(world, det)
    save_trace(trace, tmp_path / "deep" / "t.npz")
    back = load_trace(tmp_path / "deep" / "t.npz")
    assert (back.fps, back.width, back.height) == (trace.fps, trace.width, trace.height)
    for name in ("detected", "visibility", "image", "world"):
        np.testing.assert_array_equal(getattr(back, name), getattr(trace, name))
