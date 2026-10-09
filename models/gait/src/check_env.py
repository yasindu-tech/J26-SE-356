"""Check that the gait environment is ready (task GAIT-05).

Run from the repo root:
    python models/gait/src/check_env.py

It imports every package the gait pipeline needs, prints its version, and,
if the MediaPipe pose model file is present, loads it once to prove pose
estimation will run on this machine.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

REQUIRED = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "shap",
    "xgboost",
    "cv2",
    "mediapipe",
    "matplotlib",
]

POSE_MODEL = Path("models/gait/artifacts/pose_landmarker_heavy.task")


def check_imports() -> list[str]:
    """Import each required package and return the names that failed."""
    failed: list[str] = []
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
        except Exception as err:  # broad on purpose: report every failure, not just ImportError
            print(f"  FAIL  {name}: {err}")
            failed.append(name)
            continue
        print(f"  ok    {name} {getattr(module, '__version__', '')}")
    return failed


def check_pose_model() -> bool:
    """Load the MediaPipe Pose Landmarker once. Returns False if it cannot load."""
    if not POSE_MODEL.exists():
        print(f"  MISSING  {POSE_MODEL} (download it, see models/gait/README.md)")
        return False
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=RunningMode.VIDEO,
    )
    with PoseLandmarker.create_from_options(options):
        print(f"  ok    pose model loads: {POSE_MODEL.name}")
    return True


def main() -> int:
    print(f"Python {sys.version.split()[0]}")
    if sys.version_info < (3, 11):
        print("  FAIL  Python 3.11 or newer is required (see pyproject.toml)")
        return 1
    print("Packages:")
    failed = check_imports()
    print("Pose model:")
    model_ok = check_pose_model() if "mediapipe" not in failed else False
    if failed or not model_ok:
        print("\nNot ready.")
        return 1
    print("\nEnvironment ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
