"""GAIT-24: SHAP explanations of one walk in plain words.

``build_reference`` (called by ``train_xgboost.py``) records, from the training walks, what each
input usually looks like, how big a push counts as slight, moderate or strong, and whether an
input's push for a class follows its value steadily. It is stored in the model bundle.
``describe`` turns the top three inputs of one walk into one sentence each. The sentences say
what the model used, never why a person walks that way.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from gait_extract import FEATURE_NAMES

if TYPE_CHECKING:
    from explain_shap import Explanation

TOP = 3
PERCENTILES = (10, 25, 75, 90)
BINS = 5
FLAT = 0.02  # log-odds: a change between value bands smaller than this is not a change
VARIES = " (its effect varies with the other inputs)"
CLASS_WORDS = {
    0: "no gait impairment pattern",
    1: "a mild impairment pattern",
    2: "a moderate-to-severe impairment pattern",
}
NAMES = {
    "stride_time_mean": "stride time",
    "cadence": "walking pace",
    "stride_time_cv": "stride-time variability",
    "step_length_norm": "step length",
    "step_length_asymmetry": "left-right step difference",
    "knee_flexion_range_deg": "knee bend range",
    "hip_flexion_range_deg": "hip swing range",
    "arm_swing_range_norm": "arm swing",
}
SHOWN: dict[str, Callable[[float], str]] = {
    "stride_time_mean": lambda v: f"{v:.2f} s",
    "cadence": lambda v: f"{v:.0f} steps/min",
    "stride_time_cv": lambda v: f"{v:.1f}%",
    "step_length_norm": lambda v: f"{v:.2f} leg lengths",
    "step_length_asymmetry": lambda v: f"{100 * v:.0f}%",
    "knee_flexion_range_deg": lambda v: f"{v:.0f}°",
    "hip_flexion_range_deg": lambda v: f"{v:.0f}°",
    "arm_swing_range_norm": lambda v: f"{v:.2f} leg lengths",
}


@dataclass(frozen=True)
class Entry:
    feature: str
    value: float
    shap: float
    direction: str
    strength: str
    varies: bool
    sentence: str

    def as_record(self) -> dict[str, Any]:
        return {
            "name": self.feature,
            "value": self.value,
            "shap": self.shap,
            "direction": self.direction,
            "strength": self.strength,
            "sentence": self.sentence,
        }


@dataclass(frozen=True)
class WordedExplanation:
    predicted_class: int
    probabilities: tuple[float, ...]
    entries: tuple[Entry, ...]

    @property
    def sentences(self) -> list[str]:
        return [e.sentence for e in self.entries]


def build_reference(x: np.ndarray, values: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    """Reference numbers from the training walks (``values``: walks, inputs, classes)."""
    n = len(x)
    own = np.abs(values[np.arange(n), :, predicted])
    top_sizes = -np.sort(-own, axis=1)[:, :TOP]
    cuts = np.percentile(top_sizes, [100 / 3, 200 / 3])
    percentiles: dict[str, dict[str, float]] = {}
    steady: dict[str, dict[str, bool]] = {}
    for j, name in enumerate(FEATURE_NAMES):
        percentiles[name] = {f"p{p}": float(np.percentile(x[:, j], p)) for p in PERCENTILES}
        band = pd.qcut(pd.Series(x[:, j]).rank(method="first"), BINS, labels=False).to_numpy()
        steady[name] = {}
        for c in range(values.shape[2]):
            steps = np.diff([values[band == k, j, c].mean() for k in range(BINS)])
            steady[name][str(c)] = not ((steps > FLAT).any() and (steps < -FLAT).any())
    return {
        "walks": n,
        "percentiles": percentiles,
        "strength_cuts": [float(c) for c in cuts],
        "steady": steady,
    }


def where_it_sits(value: float, marks: dict[str, float]) -> str:
    if value <= marks["p10"]:
        return "lower than about 9 in 10 of the training walks"
    if value <= marks["p25"]:
        return "on the low side of the training walks"
    if value < marks["p75"]:
        return "in the middle range of the training walks"
    if value < marks["p90"]:
        return "on the high side of the training walks"
    return "higher than about 9 in 10 of the training walks"


def strength_word(size: float, cuts: list[float]) -> str:
    return "slightly" if size < cuts[0] else "moderately" if size < cuts[1] else "strongly"


def describe(explanation: Explanation, reference: dict[str, Any]) -> WordedExplanation:
    """One sentence per input of ``explanation`` (``explain_shap.Explainer.explain``)."""
    klass = explanation.predicted_class
    entries = []
    for c in explanation.top:
        varies = not reference["steady"][c.feature][str(klass)]
        strength = strength_word(abs(c.shap), reference["strength_cuts"])
        push = "towards" if c.direction == "towards" else "away from"
        sentence = (
            f"The model used {NAMES[c.feature]} ({SHOWN[c.feature](c.value)}), "
            f"{where_it_sits(c.value, reference['percentiles'][c.feature])}, "
            f"which {strength} pushed its score {push} {CLASS_WORDS[klass]}"
            f"{VARIES if varies else ''}."
        )
        entries.append(Entry(c.feature, c.value, c.shap, c.direction, strength, varies, sentence))
    return WordedExplanation(klass, explanation.probabilities, tuple(entries))
