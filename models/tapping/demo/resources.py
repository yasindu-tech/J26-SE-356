"""Demo resources (task FT-47): the trained demo model (checked against its meta) and the hand
landmarker handle. No streamlit import; the app caches both with st.cache_resource.
The landmarker handle is a verified model path plus a factory: track_clip closes its landmarker
after each clip and VIDEO-mode timestamps restart per clip, so one instance cannot be reused.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv_harness as ch  # noqa: E402
import cv_splits as cvs  # noqa: E402
import lightgbm  # noqa: E402
from data_access import DemoDataError, processed_root  # noqa: E402

FEATURE_SET = "A"
MODEL_FILE = "model_{target}_A.txt"
META_FILE = "model_{target}_A_meta.json"
TOOL = "demo-model-1"
NOTE = "Demo model; not used for any reported metric"
TRAIN_HINT = "run models/tapping/demo/train_demo_model.py once"


class DemoModelError(DemoDataError):
    """The demo model is missing or does not match its meta (message safe to show)."""


def demo_dir(root: Path | None = None) -> Path:
    return (processed_root() if root is None else root) / "demo"


def model_paths(target: str, root: Path | None = None) -> tuple[Path, Path]:
    folder = demo_dir(root)
    return folder / MODEL_FILE.format(target=target), folder / META_FILE.format(target=target)


def model_sha8(target: str, root: Path | None = None) -> str:
    """sha8 of the model file: the cache key of the loaded model."""
    model, meta = model_paths(target, root)
    if not model.is_file() or not meta.is_file():
        raise DemoModelError(f"no demo model for {target}: {TRAIN_HINT}")
    return cvs.sha8_of(model.read_bytes())


def load_demo_model(target: str, root: Path | None = None) -> lightgbm.Booster:
    root = processed_root() if root is None else root
    model, meta_path = model_paths(target, root)
    sha = model_sha8(target, root)
    meta: dict[str, Any] = json.loads(meta_path.read_text())
    cols = list(ch.FEATURE_SETS[FEATURE_SET])
    features = root / "features.csv"
    expected = {
        "tool": TOOL,
        "target": target,
        "model_sha8": sha,
        "features_sha8": cvs.sha8_of(json.dumps(cols)),
        "grid_sha8": ch.grid_sha8("lgbm", ch.grid_for("lgbm")),
        "lightgbm": lightgbm.__version__,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise DemoModelError(f"demo model for {target}: {key} differs; retrain the model")
    if not features.is_file():
        raise DemoModelError("the features file is missing")
    if meta.get("inputs_sha8", {}).get("features") != cvs.file_sha256(features)[: cvs.HASH_LEN]:
        raise DemoModelError(f"demo model for {target}: features file changed; retrain the model")
    booster = lightgbm.Booster(model_file=str(model))
    if booster.num_feature() != len(cols):
        raise DemoModelError(f"demo model for {target}: wrong number of features")
    return booster


FOLD_META = "folds_meta.json"
FOLD_FILE = "fold_r{repeat}_f{fold}.txt"
N_FOLD_MODELS = 25


def fold_dir(target: str, root: Path | None = None) -> Path:
    return demo_dir(root) / "folds" / target


def fold_sha8(target: str, root: Path | None = None) -> str:
    """sha8 of the folds meta file: the cache key of the loaded fold models."""
    path = fold_dir(target, root) / FOLD_META
    if not path.is_file():
        raise DemoModelError(f"no fold models for {target}: {TRAIN_HINT}")
    return cvs.sha8_of(path.read_bytes())


def load_fold_models(target: str, root: Path | None = None) -> list[lightgbm.Booster]:
    """The 25 outer-fold refits of the official run, each checked against the folds meta."""
    root = processed_root() if root is None else root
    folder = fold_dir(target, root)
    meta_path = folder / FOLD_META
    if not meta_path.is_file():
        raise DemoModelError(f"no fold models for {target}: {TRAIN_HINT}")
    meta: dict[str, Any] = json.loads(meta_path.read_text())
    cols = list(ch.FEATURE_SETS[FEATURE_SET])
    expected = {
        "tool": TOOL, "target": target, "n_models": N_FOLD_MODELS,
        "features_sha8": cvs.sha8_of(json.dumps(cols)),
        "grid_sha8": ch.grid_sha8("lgbm", ch.grid_for("lgbm")),
        "lightgbm": lightgbm.__version__,
    }  # fmt: skip
    for key, value in expected.items():
        if meta.get(key) != value:
            raise DemoModelError(f"fold models for {target}: {key} differs; retrain the models")
    features = root / "features.csv"
    if (
        not features.is_file()
        or meta.get("inputs_sha8", {}).get("features") != cvs.file_sha256(features)[: cvs.HASH_LEN]
    ):
        raise DemoModelError(f"fold models for {target}: features file changed; retrain")
    models = []
    for repeat in range(5):
        for fold in range(5):
            name = FOLD_FILE.format(repeat=repeat, fold=fold)
            path = folder / name
            if not path.is_file():
                raise DemoModelError(f"fold models for {target}: {name} is missing; retrain")
            if cvs.sha8_of(path.read_bytes()) != meta.get("files_sha8", {}).get(name):
                raise DemoModelError(f"fold models for {target}: {name} sha8 differs; retrain")
            models.append(lightgbm.Booster(model_file=str(path)))
    return models


class LandmarkerHandle:
    """A verified landmarker model file; calling it makes a fresh MediaPipe landmarker."""

    def __init__(self, model_path: Path, sha256: str) -> None:
        self.model_path = model_path
        self.sha256 = sha256

    def __call__(self, model_path: Path, **options: Any) -> Any:
        import hand_landmarks as hl

        return hl.MediaPipeHandLandmarker(model_path, **options)


def make_landmarker(model_path: Path | None = None) -> LandmarkerHandle:
    import hand_landmarks as hl

    path = Path(hl.DEFAULT_MODEL if model_path is None else model_path)
    if not path.is_file():
        raise DemoDataError(f"hand landmarker model not found. {hl.MODEL_HINT}")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != hl.EXPECTED_MODEL_SHA256:
        raise DemoDataError("the hand landmarker model file is not the expected one")
    return LandmarkerHandle(path, sha)
