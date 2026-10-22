"""Score and explanation for the demo (task FT-49), no streamlit import. Set A features of an
upload come from the official features_a code; the score is the demo model's output with a band
from the 25 cross-validation fold models; SHAP is the tree-path-dependent raw (log-odds) value.
Texts come from fixed templates only: contributions to the model's score, never causes.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv_harness as ch  # noqa: E402
import data_access as da  # noqa: E402
import features_a  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import tap_table  # noqa: E402
from cv_splits import GuardFailure  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from scipy.stats import percentileofscore  # noqa: E402

SET_A = list(ch.FEATURE_SETS["A"])
BAND_PERCENTILES = (5.0, 95.0)
THRESHOLD = 0.5  # the fixed study threshold
FPS_NOTE_ABOVE = 45.0  # a frame rate above this counts as 60 fps
ADDITIVITY_TOL = 1e-6
EXPLAINER = {"feature_perturbation": "tree_path_dependent", "model_output": "raw"}
PLAIN = {
    "tap_rate_hz": ("Tapping rate", "taps/s"),
    "iti_cv": ("Rhythm irregularity", "CV"),
    "open_vel_peak": ("Opening speed", "hand sizes/s"),
    "close_vel_peak": ("Closing speed", "hand sizes/s"),
    "amp_mean": ("Mean amplitude", "hand sizes"),
}
NEGLIGIBLE = 0.005
UP, DOWN = "#c4622d", "#3b6ea5"  # higher score / lower score
X_LABEL = "contribution to the score (log-odds); right = higher score"


class ScoreError(da.DemoDataError):
    """The clip cannot be scored; the message is safe to show on screen."""


class ScoreResult(TypedDict):
    score: float
    band_low: float
    band_high: float
    n_models: int


class Explanation(TypedDict):
    base_value: float
    raw_margin: float
    shap: dict[str, float]


class Contribution(TypedDict):
    feature: str
    value: float
    shap: float
    percentile: float
    in_range: bool


def upload_features(result: Any) -> dict[str, float]:
    """The 5 Set A values of an upload, by the official tap table and features_a code."""
    table = tap_table.build_tap_table(result.taps_table)
    summary = pd.Series(
        {"duration_s": result.stats["duration_s"], "fps": result.stats["fps"], "qc_status": ""}
    )
    row = features_a.clip_features(table, result.upload_ref, summary)
    values = {c: float(row[c]) for c in SET_A}
    if not all(np.isfinite(v) for v in values.values()):
        raise ScoreError("Not enough valid taps to score this clip.")
    return values


def _row(features: dict[str, float]) -> np.ndarray:
    return np.array([[features[c] for c in SET_A]], dtype=float)


def score(features: dict[str, float], model: Any, fold_models: list[Any]) -> ScoreResult:
    x = _row(features)
    folds = np.array([float(m.predict(x)[0]) for m in fold_models])
    low, high = np.percentile(folds, BAND_PERCENTILES)  # linear interpolation
    return {
        "score": float(model.predict(x)[0]), "band_low": float(low), "band_high": float(high),
        "n_models": len(fold_models),
    }  # fmt: skip


def explain(features: dict[str, float], model: Any) -> Explanation:
    x = _row(features)
    explainer = shap.TreeExplainer(model, **EXPLAINER)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="LightGBM binary classifier with TreeExplainer")
        values = explainer.shap_values(x, check_additivity=False)
    if not isinstance(values, np.ndarray) or values.shape != x.shape:
        raise GuardFailure("TreeExplainer returned an unexpected shape for a binary classifier")
    base = np.asarray(explainer.expected_value, dtype=float).ravel()
    raw = float(model.predict(x, raw_score=True)[0])
    if base.size != 1 or abs(float(base[0]) + float(values.sum()) - raw) > ADDITIVITY_TOL:
        raise GuardFailure("base value plus SHAP differs from the model margin")
    return {
        "base_value": float(base[0]), "raw_margin": raw,
        "shap": {c: float(values[0, i]) for i, c in enumerate(SET_A)},
    }  # fmt: skip


def load_reference(root: Path | None = None) -> pd.DataFrame:
    """The Set A columns of the research clips; usecols keeps every other column unread."""
    path = (da.processed_root() if root is None else root) / "features.csv"
    if not path.is_file():
        raise da.DemoDataError("missing input: features file")
    return pd.read_csv(path, usecols=SET_A)[SET_A]


def context(features: dict[str, float], reference: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for c in SET_A:
        ref = reference[c].to_numpy(dtype=float)
        ref = ref[np.isfinite(ref)]
        v = features[c]
        if not np.isfinite(v):  # a missing value has no percentile and is not 'outside'
            out[c] = {"percentile": float("nan"), "in_range": True}
            continue
        out[c] = {
            "percentile": float(percentileofscore(ref, v, kind="mean")),
            "in_range": bool(ref.min() <= v <= ref.max()),
        }
    return out


def example_panel(target: str, ref: str, root: Path | None = None) -> dict[str, Any]:
    """Held-out view of a research clip: mean score and min/max band over the 5 repeats."""
    s = da.clip_summary(target, ref, root)
    probas = da.heldout_probas(target, ref, root)
    return {
        "score": s["proba"], "band_low": min(probas), "band_high": max(probas),
        "n_models": len(probas), "base_value": s["base_value"], "shap": s["shap"],
        "values": s["value"],
    }  # fmt: skip


def cv_auroc(target: str, root: Path | None = None) -> dict[str, float]:
    path = (da.processed_root() if root is None else root) / "cv" / "results" / "results_table.csv"
    if not path.is_file():
        raise da.DemoDataError("missing input: results table")
    table = pd.read_csv(path, comment="#")
    hit = table[
        (table["section"] == "performance") & (table["target"] == target)
        & (table["subject"] == "lgbm A") & (table["metric"] == "auroc")
    ]  # fmt: skip
    if len(hit) != 1:
        raise GuardFailure(f"results table: expected one lgbm A AUROC row for {target}")
    r = hit.iloc[0]
    return {"estimate": float(r["estimate"]), "ci_low": float(r["ci_low"]),
            "ci_high": float(r["ci_high"])}  # fmt: skip


def contributions(
    values: dict[str, float], shap_values: dict[str, float], ctx: dict[str, dict[str, Any]]
) -> list[Contribution]:
    """Per feature, sorted by absolute contribution (largest first)."""
    rows: list[Contribution] = [
        {"feature": c, "value": values[c], "shap": shap_values[c],
         "percentile": ctx[c]["percentile"], "in_range": ctx[c]["in_range"]}
        for c in SET_A
    ]  # fmt: skip
    return sorted(rows, key=lambda r: (-abs(r["shap"]), SET_A.index(r["feature"])))


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def value_text(r: Contribution) -> str:
    unit = PLAIN[r["feature"]][1]
    return f"{r['value']:.2f} {unit}" if np.isfinite(r["value"]) else "not available"


def percentile_text(r: Contribution) -> str:
    if not np.isfinite(r["percentile"]):
        return "percentile not available"
    return f"{ordinal(int(round(r['percentile'])))} percentile of the research clips"


def wording(rows: list[Contribution]) -> list[str]:
    """One sentence per feature from fixed templates: a contribution to the score, no cause."""
    out = []
    for r in rows:
        head = f"{PLAIN[r['feature']][0]} ({value_text(r)}, {percentile_text(r)})"
        if abs(r["shap"]) < NEGLIGIBLE:
            text = f"{head} had almost no effect on the score ({r['shap']:+.2f} log-odds)"
        else:
            direction = "up" if r["shap"] > 0 else "down"
            text = f"{head} pushed the score {direction} by {abs(r['shap']):.2f} (log-odds)"
        if not r["in_range"]:
            text += "; outside the range of the research clips"
        out.append(text + ".")
    return out


def shap_figure(rows: list[Contribution], title: str, compact: bool = False) -> Figure:
    """The bar figure; ``compact`` is the same content at page size with larger relative fonts."""
    big, mid, small = (10, 9.5, 11) if compact else (12, 11, 14)
    fig = Figure(figsize=(7.0, 2.7) if compact else (11.5, 4.2), dpi=110, layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    labels = [f"{PLAIN[r['feature']][0]} ({r['value']:.2f} {PLAIN[r['feature']][1]})" for r in rows]
    vals = np.array([r["shap"] for r in rows])
    y = np.arange(len(rows))[::-1]  # largest contribution on top
    ax.barh(y, vals, color=[UP if v > 0 else DOWN for v in vals], height=0.6)
    ax.axvline(0, color="#555555", lw=1)
    ax.set_yticks(y, labels)
    span = max(float(np.abs(vals).max()), 0.1) * 1.15
    ax.set_xlim(-span, span)
    ax.set_xlabel(X_LABEL, fontsize=big)
    ax.tick_params(labelsize=mid)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=small)
    return fig
