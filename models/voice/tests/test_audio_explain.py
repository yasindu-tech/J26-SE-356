"""Tests for the plain-language explanations (VOICE-24). All values are synthetic."""

from __future__ import annotations

import re

import audio_explain
import audio_features
import pytest

NAMES = list(audio_features.FEATURE_NAMES)
BANNED = re.compile(r"diagnos|positive|negative|cleared|discharged|causes", re.IGNORECASE)


def reference(low: float = 0.0, high: float = 100.0) -> dict[str, dict[str, list[float]]]:
    """Every feature uniform from ``low`` to ``high`` in both classes (101 percentiles)."""
    step = (high - low) / 100
    ladder = [low + i * step for i in range(101)]
    return {cls: {name: ladder for name in NAMES} for cls in ("healthy", "pd")}


def test_percentile_among_places_values_on_the_population() -> None:
    points = reference()["healthy"]["jitter_local"]
    assert audio_explain.percentile_among(-5, points) == 0.0
    assert audio_explain.percentile_among(500, points) == 100.0
    assert audio_explain.percentile_among(50, points) == pytest.approx(50, abs=1)


def test_percentile_among_puts_ties_in_the_middle() -> None:
    assert audio_explain.percentile_among(1.0, [1.0] * 101) == pytest.approx(50, abs=1)


@pytest.mark.parametrize(
    ("value", "shap", "expect"),
    [
        (99.9, 0.4, ("higher than all healthy", "raised")),
        (95, 0.4, ("higher than 95%", "raised")),
        (50, -0.2, ("within the usual range", "lowered")),
        (3, -0.2, ("lower than 97%", "lowered")),
    ],
)
def test_driver_sentences_are_plain_and_never_diagnostic(
    value: float, shap: float, expect: tuple[str, str]
) -> None:
    driver = audio_explain.describe_driver("jitter_local", value, shap, reference())
    assert driver["effect"] == expect[1]
    assert expect[0] in driver["sentence"]
    assert "jitter_local" in driver["sentence"]
    assert "model used this" in driver["sentence"]
    assert not BANNED.search(driver["sentence"])


def test_incomparable_measure_is_not_compared_with_training() -> None:
    driver = audio_explain.describe_driver("jitter_local", 99.9, 0.4, reference(), comparable=False)
    assert "cannot be compared" in driver["sentence"]
    assert "healthy training voices" not in driver["sentence"]


def test_values_outside_the_training_range_are_flagged() -> None:
    features = dict.fromkeys(NAMES, 50.0)
    features["hnr_db"] = 1000.0
    assert audio_explain.outside_training(features, reference()) == ["hnr_db"]


def _reasons(fmt: str, vowel_s: float, ci: tuple[float, float], unusual: list[str]) -> list[str]:
    result = {"risk_score": sum(ci) / 2, "ci_lower": ci[0], "ci_upper": ci[1]}
    info = {"source_format": fmt, "vowel_s": vowel_s, "snr_db": 25.0}
    return audio_explain.uncertainty_reasons(result, info, unusual, 60, ["figshare"])


def test_uncertainty_always_names_the_training_size_and_other_voice_conditions() -> None:
    reasons = _reasons("wav", 3.0, (0.4, 0.5), [])
    assert len(reasons) == 2
    assert "60 people" in reasons[0]
    assert "cold" in reasons[1] and "asthma" in reasons[1]
    assert not BANNED.search(reasons[1])


def test_uncertainty_says_when_background_noise_could_not_be_checked() -> None:
    result = {"risk_score": 0.5, "ci_lower": 0.4, "ci_upper": 0.6}
    unchecked = {"source_format": "wav", "vowel_s": 3.0, "snr_db": None}
    checked = {**unchecked, "snr_db": 25.0}
    text = " ".join(audio_explain.uncertainty_reasons(result, unchecked, [], 60, ["figshare"]))
    assert "could not be checked" in text
    text = " ".join(audio_explain.uncertainty_reasons(result, checked, [], 60, ["figshare"]))
    assert "could not be checked" not in text


def test_voice_far_outside_every_training_voice_is_refused() -> None:
    features = dict.fromkeys(NAMES, 50.0)
    # Range is 1..99, so "far" starts beyond 1 - 49 and 99 + 49.
    features.update(hnr_db=200.0, f0_mean_hz=-100.0, jitter_local=160.0)
    far = audio_explain.far_outside_training(features, reference())
    assert far == ["f0_mean_hz", "jitter_local", "hnr_db"]
    problems = audio_explain.unlike_training_problem(far)
    assert len(problems) == 1 and "natural 'aaah'" in problems[0]
    assert not BANNED.search(problems[0])


def test_a_few_unusual_measures_do_not_refuse_a_voice() -> None:
    features = dict.fromkeys(NAMES, 50.0)
    features.update(hnr_db=120.0, f0_mean_hz=200.0, jitter_local=-80.0)  # 1 near, 2 far
    far = audio_explain.far_outside_training(features, reference())
    assert len(far) == audio_explain.MAX_FAR_OUTSIDE - 1
    assert audio_explain.unlike_training_problem(far) == []
    assert audio_explain.far_outside_training({"x": 1e9, **features}, reference()) == far


def test_measures_on_another_scale_are_not_counted_as_far_outside() -> None:
    features = dict.fromkeys(NAMES, 50.0)
    features.update(hnr_db=500.0, f0_mean_hz=500.0, jitter_local=500.0)
    far = audio_explain.far_outside_training(features, reference(), skip=["hnr_db"])
    assert "hnr_db" not in far


def test_uncertainty_names_compression_short_vowel_wide_range_and_unusual_values() -> None:
    reasons = " ".join(_reasons("mp3", 1.2, (0.2, 0.8), ["hnr_db"]))
    assert "compressed (mp3)" in reasons
    assert "short" in reasons
    assert "wide" in reasons
    assert "outside the range seen in training" in reasons


def test_explanation_warns_when_the_score_uses_incomparable_measures() -> None:
    quality = audio_features.QualityReport(True, [], {"source_format": "wav", "vowel_s": 3.0})
    result = {
        "risk_score": 0.6,
        "ci_lower": 0.5,
        "ci_upper": 0.7,
        "shap_top5": [{"feature": "jitter_local", "shap_log_odds": 0.5}],
    }
    features = dict.fromkeys(NAMES, 50.0)
    importance = {"per_group": {"jitter": 1.0}}
    explanation = audio_explain.explain(
        result,
        features,
        quality,
        reference(),
        importance,
        60,
        ["figshare"],
        incomparable=["jitter_local"],
    )
    assert not explanation["top_reasons"][0]["comparable"]
    assert any("different scale" in r for r in explanation["uncertainty"])


def test_refused_recording_gets_reasons_but_no_drivers() -> None:
    quality = audio_features.QualityReport(False, ["The steady 'aaah' is only 0.4 s long."])
    importance = {"per_group": {"mfcc": 2.0, "jitter": 1.0}}
    explanation = audio_explain.explain(
        {"abstained": True}, None, quality, reference(), importance, 60, ["figshare"]
    )
    assert explanation["top_reasons"] == []
    assert explanation["summary"].startswith("No score")
    assert "0.4 s" in explanation["summary"]
    assert explanation["quality"]["passed"] is False
