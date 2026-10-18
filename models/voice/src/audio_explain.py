"""Plain-language explanations for an audio voice result (task VOICE-24).

Every output of ``analyze_audio`` gets an answer to "why?":

- ``top_reasons``: the 5 SHAP drivers, each with its plain meaning, the
  person's value, and where that value sits among healthy and PD TRAINING
  voices (percentiles stored in the bundle, never training rows);
- ``uncertainty``: why the confidence interval is as wide as it is;
- ``quality``: what the quality gate measured, and why a file was refused;
- ``model_overview``: what the model relies on in general (mean |SHAP|).

Wording rules: explanations describe what the model used, never what causes
Parkinson's disease, and never use diagnostic words (CLAUDE.md section 1).
"""

from __future__ import annotations

from typing import Any

import audio_features
import numpy as np

LOSSY_FORMATS = {"mp3", "m4a", "aac", "ogg", "opus", "wma", "amr", "3gp"}
WIDE_CI = 0.30  # CI wider than this is called out as wide
SHORT_VOWEL_S = 2.0

FeatureInfo = dict[str, tuple[str, str]]
Reference = dict[str, dict[str, list[float]]]


def percentile_among(value: float, percentiles: list[float]) -> float:
    """Where ``value`` sits (0-100) in a population described by its 101 percentiles."""
    points = np.asarray(percentiles, dtype=float)
    if value < points[0]:
        return 0.0
    if value > points[-1]:
        return 100.0
    # Average the lowest and highest matching rank, so ties land in the middle.
    low = np.interp(value, points, np.arange(101), left=0, right=100)
    high = 100 - np.interp(-value, -points[::-1], np.arange(101), left=0, right=100)
    return round(float((low + high) / 2), 1)


def _comparison(pct_healthy: float) -> str:
    if pct_healthy >= 99.5:
        return "higher than all healthy training voices"
    if pct_healthy <= 0.5:
        return "lower than all healthy training voices"
    if pct_healthy >= 90:
        return f"higher than {pct_healthy:.0f}% of healthy training voices"
    if pct_healthy <= 10:
        return f"lower than {100 - pct_healthy:.0f}% of healthy training voices"
    return "within the usual range of healthy training voices"


def describe_driver(
    feature: str,
    value: float,
    shap_log_odds: float,
    reference: Reference,
    feature_info: FeatureInfo = audio_features.FEATURE_INFO,
    comparable: bool = True,
) -> dict[str, Any]:
    group, meaning = feature_info[feature]
    pct_healthy = percentile_among(value, reference["healthy"][feature])
    pct_pd = percentile_among(value, reference["pd"][feature])
    effect = "raised" if shap_log_odds > 0 else "lowered"
    comparison = (
        _comparison(pct_healthy)
        if comparable
        else "measured differently from the training data, so it cannot be compared"
    )
    return {
        "comparable": comparable,
        "feature": feature,
        "meaning": meaning,
        "group": group,
        "value": round(float(value), 9),
        "shap_log_odds": round(float(shap_log_odds), 4),
        "effect": effect,
        "percentile_among_healthy": pct_healthy,
        "percentile_among_pd": pct_pd,
        "sentence": f"{meaning[0].upper()}{meaning[1:]} ({feature}): "
        f"{comparison}. The model used this, and it {effect} the score.",
    }


def outside_training(features: dict[str, float], reference: Reference) -> list[str]:
    """Features whose value is outside the 1st-99th percentile of ALL training voices."""
    unusual = []
    for name, value in features.items():
        if name not in reference["healthy"]:
            continue
        low = min(reference["healthy"][name][1], reference["pd"][name][1])
        high = max(reference["healthy"][name][99], reference["pd"][name][99])
        if value < low or value > high:
            unusual.append(name)
    return unusual


def uncertainty_reasons(
    result: dict[str, Any],
    quality_info: dict[str, Any],
    unusual: list[str],
    n_training: int,
    datasets: list[str],
    incomparable_used: list[str] | None = None,
) -> list[str]:
    reasons = [
        f"The model was trained on only {n_training} people ({', '.join(datasets)}), "
        "so every estimate is uncertain."
    ]
    if incomparable_used:
        reasons.append(
            f"This score relies on measures that new recordings give on a different scale "
            f"from the training data ({', '.join(incomparable_used[:3])}"
            f"{', ...' if len(incomparable_used) > 3 else ''}); treat it with extra caution."
        )
    if result.get("ci_lower") is not None:
        width = result["ci_upper"] - result["ci_lower"]
        if width > WIDE_CI:
            reasons.append(
                f"The 95% range is wide ({result['ci_lower']:.2f} to {result['ci_upper']:.2f}): "
                "models trained on slightly different people disagree about this voice."
            )
    fmt = quality_info.get("source_format", "")
    if fmt in LOSSY_FORMATS:
        reasons.append(
            f"The file is compressed ({fmt}); fine voice measures such as jitter and "
            "shimmer are less exact than from WAV or FLAC."
        )
    vowel_s = quality_info.get("vowel_s")
    if vowel_s is not None and vowel_s < SHORT_VOWEL_S:
        reasons.append(
            f"The steady 'aaah' is short ({vowel_s:.1f} s); longer gives steadier measures."
        )
    if unusual:
        shown = ", ".join(unusual[:3]) + (", ..." if len(unusual) > 3 else "")
        reasons.append(
            f"{len(unusual)} measurement(s) are outside the range seen in training ({shown}); "
            "the model is less reliable for this voice."
        )
    return reasons


def model_overview(global_importance: dict[str, Any], top_n: int = 3) -> dict[str, Any]:
    groups = {g: v for g, v in global_importance["per_group"].items() if v > 0}
    total = sum(groups.values()) or 1.0
    top = list(groups.items())[:top_n]
    return {
        "top_groups": [{"group": g, "share": round(v / total, 3)} for g, v in top],
        "sentence": "In general the model relies most on "
        + ", ".join(f"{g} ({v / total:.0%})" for g, v in top)
        + " measures.",
    }


def explain(
    result: dict[str, Any],
    features: dict[str, float] | None,
    quality: audio_features.QualityReport,
    reference: Reference,
    global_importance: dict[str, Any],
    n_training: int,
    datasets: list[str],
    feature_info: FeatureInfo = audio_features.FEATURE_INFO,
    incomparable: list[str] | None = None,
) -> dict[str, Any]:
    """The explanation block attached to one ``analyze_audio`` result."""
    incomparable = incomparable or []
    explanation: dict[str, Any] = {
        "quality": {
            "passed": quality.passed,
            "problems": quality.problems,
            "measured": quality.info,
        },
        "model_overview": model_overview(global_importance),
    }
    if features is None or result.get("abstained"):
        explanation["summary"] = (
            "No score: the recording did not pass the quality check. "
            + (quality.problems[0] if quality.problems else "")
        ).strip()
        explanation["top_reasons"] = []
        explanation["uncertainty"] = []
        return explanation

    reasons = [
        describe_driver(
            d["feature"],
            features[d["feature"]],
            d["shap_log_odds"],
            reference,
            feature_info,
            comparable=d["feature"] not in incomparable,
        )
        for d in result["shap_top5"]
    ]
    # Out-of-range is only meaningful for measures on the same scale as training.
    unusual = [n for n in outside_training(features, reference) if n not in incomparable]
    incomparable_used = [r["feature"] for r in reasons if not r["comparable"]]
    explanation["top_reasons"] = reasons
    explanation["outside_training"] = unusual
    explanation["uncertainty"] = uncertainty_reasons(
        result, quality.info, unusual, n_training, datasets, incomparable_used
    )
    lead = reasons[0] if reasons else None
    explanation["summary"] = (
        f"Risk score {result['risk_score']:.2f} (95% range {result['ci_lower']:.2f}-"
        f"{result['ci_upper']:.2f}). "
        + (f"Biggest single reason: {lead['sentence']}" if lead else "")
    )
    return explanation
