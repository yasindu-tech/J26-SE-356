"""Demo inputs (task FT-47): loads the held-out SHAP outputs of data/processed/xai for the
Streamlit demo, with the same guards as shap_figures (oof_reproduced, sha8, row counts). No
streamlit import. Only hashed clip refs and feature columns are read; labels never are.
Errors: DemoDataError = input missing (friendly message), GuardFailure = inputs disagree.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv_splits as cvs  # noqa: E402
import pandas as pd  # noqa: E402
import shap_directions as sd  # noqa: E402
import shap_figures as sf  # noqa: E402
import treeshap_heldout as tsh  # noqa: E402
from cv_splits import GuardFailure, InputProblem  # noqa: E402

ENV_ROOT = "TAPPING_DEMO_PROCESSED"  # tests point the app at a synthetic data/processed
FEATURE_SET = "A"  # the demo shows Set A only
TARGETS = ("ge2", "ge1")
GLOBAL_MEAN = "mean_abs_shap_mean_over_repeats"
REF_RE = re.compile(r"^[0-9a-f]{8}$")


class DemoDataError(ValueError):
    """A demo input is missing; the message is safe to show on screen."""


class ClipSummary(TypedDict):
    clip_ref: str
    proba: float
    base_value: float
    raw_margin: float
    shap: dict[str, float]
    value: dict[str, float]


class DirectionRow(TypedDict):
    feature: str
    global_rank: int
    mean_abs_shap: float
    direction: str
    rho_value_vs_shap: float


def processed_root() -> Path:
    env = os.environ.get(ENV_ROOT)
    return Path(env) if env else cvs.PROCESSED_ROOT


def _paths(target: str, root: Path | None) -> dict[str, Path]:
    if target not in TARGETS:
        raise DemoDataError(f"unknown target: {target}")
    root = processed_root() if root is None else root
    folder = root / "xai" / target / FEATURE_SET
    paths = {n: folder / n for n in sf.SHAP_INPUTS}
    paths["official_meta"] = root / "cv" / target / f"meta_{tsh.MODEL}_{FEATURE_SET}.json"
    paths["directions"] = root / "xai" / "directions" / target / FEATURE_SET / sd.OUT_FILE
    paths["directions_meta"] = paths["directions"].with_name(sd.META_FILE)
    return paths


def cache_key(target: str, root: Path | None = None) -> tuple[tuple[str, str], ...]:
    """(file name, sha8 of its bytes) per input: a changed file changes the key."""
    paths = _paths(target, root)
    optional = ("directions", "directions_meta")  # keyed when present; directions() requires them
    for name, path in paths.items():
        if not path.is_file() and name not in optional:
            raise DemoDataError(f"missing input: {name} for {target}")
    paths = {n: p for n, p in paths.items() if p.is_file()}
    return tuple((name, cvs.sha8_of(path.read_bytes())) for name, path in sorted(paths.items()))


def _clips(target: str, root: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    paths = _paths(target, root)
    cols = list(tsh.ch.FEATURE_SETS[FEATURE_SET])
    try:
        inputs = sf.read_inputs(paths[tsh.META_FILE].parent, cols)
        if not paths["official_meta"].is_file():
            raise InputProblem("missing input: official meta")
    except InputProblem as err:
        raise DemoDataError(str(err)) from err
    official = json.loads(paths["official_meta"].read_text())
    sf.check_meta(inputs["meta"], official, target, FEATURE_SET)
    sf.check_tables(inputs, cols, f"{target} {FEATURE_SET}")
    clips = sf.per_clip_means(inputs["values"], cols)
    if not clips["clip_ref"].map(lambda r: bool(REF_RE.match(r))).all():
        raise GuardFailure(f"{target}: a clip ref is not an 8-hex hashed ref")
    return clips, inputs["glob"], cols


def available_clips(target: str, root: Path | None = None) -> list[str]:
    clips, _, _ = _clips(target, root)
    return sorted(clips["clip_ref"])


def clip_summary(target: str, ref: str, root: Path | None = None) -> ClipSummary:
    clips, _, cols = _clips(target, root)
    row = clips[clips["clip_ref"] == ref]
    if len(row) != 1:
        raise DemoDataError(f"unknown clip ref: {ref}")
    r = row.iloc[0]
    shap = {c: float(r[f"shap_{c}"]) for c in cols}
    if abs(float(r["base_value"]) + sum(shap.values()) - float(r["raw_margin"])) > (
        tsh.ADDITIVITY_TOL
    ):
        raise GuardFailure(f"{target}: base value plus SHAP differs from the model margin")
    return {
        "clip_ref": ref, "proba": float(r["proba"]), "base_value": float(r["base_value"]),
        "raw_margin": float(r["raw_margin"]), "shap": shap,
        "value": {c: float(r[f"value_{c}"]) for c in cols},  # NaN stays NaN
    }  # fmt: skip


def heldout_probas(target: str, ref: str, root: Path | None = None) -> list[float]:
    """The held-out model score of one clip in each of the 5 repeats, in repeat order."""
    _clips(target, root)  # the same guards as every other loader
    paths = _paths(target, root)
    cols = list(tsh.ch.FEATURE_SETS[FEATURE_SET])
    values = sf.read_inputs(paths[tsh.META_FILE].parent, cols)["values"]
    rows = values[values["clip_ref"] == ref].sort_values("repeat")
    if len(rows) != tsh.REPEATS:
        raise DemoDataError(f"unknown clip ref: {ref}")
    return [float(p) for p in rows["proba"]]


def global_importance(target: str, root: Path | None = None) -> list[dict[str, Any]]:
    _, glob, _ = _clips(target, root)
    ordered = glob.sort_values(["rank", "feature"], kind="stable")
    return [
        {"feature": str(f), "rank": int(k), "mean_abs_shap": float(m)}
        for f, k, m in zip(ordered["feature"], ordered["rank"], ordered[GLOBAL_MEAN], strict=True)
    ]


def directions(target: str, root: Path | None = None) -> list[DirectionRow]:
    paths = _paths(target, root)
    _clips(target, root)  # the SHAP inputs must pass their guards first
    for name in ("directions", "directions_meta"):
        if not paths[name].is_file():
            raise DemoDataError(f"missing input: {name} for {target}")
    meta = json.loads(paths["directions_meta"].read_text())
    current = {n: cvs.sha8_of(paths[n].read_bytes()) for n in sf.SHAP_INPUTS}
    if meta.get("inputs_sha8") != current:
        raise GuardFailure(f"{target}: directions were computed from different SHAP inputs")
    if meta.get("files_sha8", {}).get(sd.OUT_FILE) != cvs.sha8_of(paths["directions"].read_bytes()):
        raise GuardFailure(f"{target}: directions file differs from its meta")
    table = pd.read_csv(paths["directions"])
    return [
        {
            "feature": str(r.feature), "global_rank": int(r.global_rank),
            "mean_abs_shap": float(r.mean_abs_shap), "direction": str(r.direction),
            "rho_value_vs_shap": float(r.rho_value_vs_shap),
        }
        for r in table.itertuples()
    ]  # fmt: skip
