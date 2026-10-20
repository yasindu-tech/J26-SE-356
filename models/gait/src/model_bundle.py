"""GAIT-22: save and load the demo model together with its feature list and version.

A bundle is a folder with ``model.json`` (XGBoost's own format) and ``bundle.json`` (version,
feature order, classes, settings, training fingerprints). ``train_xgboost.py`` writes it and
checks that a reload gives identical predictions; GAIT-23 (SHAP) and GAIT-27 (``predict_video``)
read it with ``load_bundle``. The model was fitted on every subject: its own score is never a result.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xgboost
from gait_extract import FEATURE_NAMES
from make_folds import CLASS_LABELS
from xgboost import XGBClassifier

MODEL_FILE = "model.json"
INFO_FILE = "bundle.json"
REQUIRED = (
    "model_version",
    "model_sha1",
    "features",
    "classes",
    "xgboost_version",
    "probabilities",
)
CLASSES = tuple(CLASS_LABELS)


class BundleError(ValueError):
    """The bundle is incomplete, changed, or does not match the code that reads it."""


@dataclass(frozen=True)
class Bundle:
    model: XGBClassifier
    info: dict[str, Any]

    @property
    def version(self) -> str:
        return str(self.info["model_version"])

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(self.info["features"])

    @property
    def reference(self) -> dict[str, Any]:
        """Training-walk numbers the plain-words step needs (written by ``train_xgboost.py``)."""
        if "explanation_reference" not in self.info:
            raise BundleError("the bundle has no explanation reference: rerun train_xgboost.py")
        return dict(self.info["explanation_reference"])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Class probabilities (rows x 3, uncalibrated) for one vector or a matrix of the inputs."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if x.shape[1] != len(self.features):
            raise BundleError(f"expected {len(self.features)} inputs per walk, got {x.shape[1]}")
        if not np.isfinite(x).all():
            raise BundleError("an input is missing or infinite")
        return np.asarray(self.model.predict_proba(x))


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def save_bundle(model: XGBClassifier, directory: Path, details: Mapping[str, Any]) -> Bundle:
    """Write ``model`` and its description to ``directory``. ``details`` adds free fields."""
    if tuple(int(c) for c in model.classes_) != CLASSES:
        raise BundleError(f"the model must have classes {CLASSES}, has {list(model.classes_)}")
    directory.mkdir(parents=True, exist_ok=True)
    model.save_model(directory / MODEL_FILE)
    sha = _sha1(directory / MODEL_FILE)
    info = {
        **details,
        "model_version": f"gait-xgb-{sha[:8]}",
        "model_sha1": sha,
        "features": list(FEATURE_NAMES),
        "classes": {str(c): CLASS_LABELS[c] for c in CLASSES},
        "xgboost_version": xgboost.__version__,
        "probabilities": "uncalibrated",
        "note": "fitted on every subject for the demo; its score is never a result",
    }
    (directory / INFO_FILE).write_text(json.dumps(info, indent=2, sort_keys=True))
    return Bundle(model, info)


def load_bundle(directory: Path) -> Bundle:
    """Read a bundle, refusing one that is incomplete, altered or built for other inputs."""
    for name in (MODEL_FILE, INFO_FILE):
        if not (directory / name).is_file():
            raise BundleError(f"{directory / name} is missing")
    info = json.loads((directory / INFO_FILE).read_text())
    missing = [k for k in REQUIRED if k not in info]
    if missing:
        raise BundleError(f"bundle.json lacks {missing}")
    if info["features"] != list(FEATURE_NAMES):
        raise BundleError(
            f"the bundle's inputs {info['features']} differ from the code's {list(FEATURE_NAMES)}"
        )
    if _sha1(directory / MODEL_FILE) != info["model_sha1"]:
        raise BundleError("model.json does not match its recorded hash: the model was changed")
    if info["xgboost_version"] != xgboost.__version__:
        warnings.warn(
            f"bundle saved with XGBoost {info['xgboost_version']}, loaded with {xgboost.__version__}",
            stacklevel=2,
        )
    model = XGBClassifier()
    model.load_model(directory / MODEL_FILE)
    if tuple(int(c) for c in model.classes_) != CLASSES:
        raise BundleError(f"the model must have classes {CLASSES}, has {list(model.classes_)}")
    return Bundle(model, info)
