"""End-to-end tests: train tiny models, then analyse audio files (VOICE-23 to VOICE-25).

Models are trained on synthetic feature tables (no real people); the analysed
files are synthetic vowels from test_audio_features.
"""

from __future__ import annotations

import json
from pathlib import Path

import analyze_audio
import audio_features
import audio_model
import joblib
import numpy as np
import pandas as pd
import pytest
import uci_praat
import voice_module
from test_audio_features import synthetic_vowel, write_wav

N_PEOPLE = 60


def _synthetic_table(columns: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    labels = np.array([0, 1] * (N_PEOPLE // 2))
    features = rng.normal(size=(N_PEOPLE, len(columns)))
    features[:, :5] += labels[:, None] * 1.5  # a learnable signal in the first features
    table = pd.DataFrame(features, columns=columns)
    table.insert(0, "person", [f"synthetic:{i}" for i in range(N_PEOPLE)])
    table["label"] = labels
    table["dataset"] = "synthetic"
    table["sex"] = np.where(np.arange(N_PEOPLE) % 4 < 2, "F", "M")
    table["path"] = [f"data/raw/synthetic/{i}.wav" for i in range(N_PEOPLE)]
    return table


@pytest.fixture(scope="module")
def bundle() -> audio_model.AudioBundle:
    table = _synthetic_table(list(audio_features.FEATURE_NAMES))
    return audio_model.train_audio_model(table, n_ensemble=3, score="anova_f", n_boot=50)


@pytest.fixture(scope="module")
def uci_bundle() -> audio_model.AudioBundle:
    """A bundle shaped like uci_transfer's: the 16 UCI Praat measures, jitter flagged."""
    columns = list(uci_praat.UCI_PRAAT_FEATURES)
    table = _synthetic_table(columns)
    X, y = table[columns], table.label.to_numpy()
    voice, test = voice_module.train_voice_model(
        X,
        y,
        table.person.to_numpy(),
        table.sex.to_numpy(),
        n_ensemble=3,
        score="anova_f",
        n_boot=50,
        k_grid=(4, 8, 16),
    )
    train = np.setdiff1d(np.arange(len(y)), test)
    return audio_model.AudioBundle(
        voice=voice,
        reference=audio_model.feature_percentiles(X.iloc[train], y[train]),
        global_importance=audio_model.global_importance(
            voice, X.iloc[train], uci_praat.UCI_FEATURE_INFO
        ),
        datasets=["synthetic"],
        feature_set="uci_praat",
        feature_info=uci_praat.UCI_FEATURE_INFO,
        incomparable=["locPctJitter"],
    )


def test_bundle_holds_references_but_no_person_rows(bundle: audio_model.AudioBundle) -> None:
    assert set(bundle.reference) == {"healthy", "pd"}
    assert all(len(p) == 101 for p in bundle.reference["pd"].values())
    assert bundle.pipeline_version.startswith("voice-audio-")
    assert bundle.global_importance["per_feature"]
    assert len(bundle.holdout_paths) > 0
    assert "synthetic:0" not in json.dumps(bundle.model_card)


@pytest.mark.parametrize("which", ["bundle", "uci_bundle"])
def test_clean_vowel_gets_a_full_explained_result(
    which: str, request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    model: audio_model.AudioBundle = request.getfixturevalue(which)
    path = write_wav(tmp_path / "Jane_Doe_visit.wav", synthetic_vowel(pad_s=0.3))
    result = analyze_audio.analyze_audio(path, model)

    for name in voice_module.RESULT_FIELDS:
        assert name in result
    assert result["abstained"] is False
    assert result["ci_lower"] <= result["risk_score"] <= result["ci_upper"]
    assert result["triage_priority"] in voice_module.TRIAGE_VALUES
    assert result["pipeline_version"] == model.pipeline_version
    explanation = result["explanations"]
    assert len(explanation["top_reasons"]) == len(result["shap_top5"]) > 0
    assert explanation["uncertainty"]
    assert explanation["quality"]["passed"] is True
    # Privacy: the file name (which may be a person's name) never reaches the output.
    assert "Jane" not in json.dumps(result)
    assert result["person"].startswith("upload-")


def test_uci_model_scores_with_uci_column_names(
    uci_bundle: audio_model.AudioBundle, tmp_path: Path
) -> None:
    path = write_wav(tmp_path / "a.wav", synthetic_vowel(pad_s=0.3))
    result = analyze_audio.analyze_audio(path, uci_bundle)
    assert {d["feature"] for d in result["shap_top5"]} <= set(uci_praat.UCI_PRAAT_FEATURES)
    assert uci_bundle.pipeline_version.startswith("voice-uci-praat-")


def test_unusable_and_non_audio_files_abstain_with_reasons(
    bundle: audio_model.AudioBundle, tmp_path: Path
) -> None:
    silent = write_wav(tmp_path / "silent.wav", np.zeros(32000))
    fake = tmp_path / "fake.mp3"
    fake.write_text("not audio")
    for path in (silent, fake):
        result = analyze_audio.analyze_audio(path, bundle)
        assert result["abstained"] is True
        assert result["risk_score"] is None and result["triage_priority"] is None
        assert result["availability_mask"] == {"voice": False}
        assert result["problems"]
        assert result["explanations"]["summary"].startswith("No score")


def test_command_writes_json_the_desktop_can_open(
    bundle: audio_model.AudioBundle, tmp_path: Path
) -> None:
    bundle_path = tmp_path / "bundle.joblib"
    joblib.dump(bundle, bundle_path)
    wav = write_wav(tmp_path / "a.wav", synthetic_vowel(pad_s=0.3))
    out = tmp_path / "results.json"
    assert analyze_audio.main([str(wav), "--bundle", str(bundle_path), "--out", str(out)]) == 0
    results = json.loads(out.read_text())
    assert isinstance(results, list) and results[0]["modality"] == "voice"


def test_command_fails_clearly_without_a_trained_model(tmp_path: Path) -> None:
    missing = tmp_path / "none.joblib"
    assert analyze_audio.main([str(tmp_path / "a.wav"), "--bundle", str(missing)]) == 1


@pytest.mark.parametrize(
    ("score", "ci", "expect", "widened"),
    [
        (0.5, (0.4, 0.6), (0.4, 0.6), False),
        (0.04, (0.07, 0.3), (0.04, 0.3), True),
        (0.95, (0.6, 0.9), (0.6, 0.95), True),
    ],
)
def test_range_is_widened_never_narrowed_to_include_the_score(
    score: float, ci: tuple[float, float], expect: tuple[float, float], widened: bool
) -> None:
    result = {"risk_score": score, "ci_lower": ci[0], "ci_upper": ci[1]}
    assert analyze_audio._widen_to_include_score(result) is widened
    assert (result["ci_lower"], result["ci_upper"]) == expect


def test_abstained_result_is_left_alone() -> None:
    result = {"risk_score": None, "ci_lower": None, "ci_upper": None}
    assert analyze_audio._widen_to_include_score(result) is False
