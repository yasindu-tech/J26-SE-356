"""Tests for models/voice/src/voice_module.py and demo_voice.py. Synthetic data only."""

from __future__ import annotations

import json

import demo_voice
import numpy as np
import pandas as pd
import pytest
import voice_module as vm

FAST = dict(score="anova_f", n_ensemble=5, n_boot=50, k_grid=(10,), c_grid=(0.5,), leaves_grid=(4,))
N_PEOPLE = 120


def voice_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """120 people x 3 rows x 20 features; f0-f2 carry signal, the rest is noise."""
    rng = np.random.default_rng(42)
    groups = np.repeat(np.arange(3000, 3000 + N_PEOPLE), 3)
    y = np.repeat((rng.random(N_PEOPLE) < 0.7).astype(int), 3)
    gender = np.repeat(rng.integers(0, 2, N_PEOPLE), 3)
    X = pd.DataFrame(rng.normal(size=(len(y), 20)), columns=[f"f{i}" for i in range(20)])
    X[["f0", "f1", "f2"]] += y[:, None] * np.array([1.5, -1.2, 1.0])
    return X, y, groups, gender


@pytest.fixture(scope="module", params=["l1_logistic", "lightgbm"])
def trained(request) -> tuple[vm.VoiceBundle, np.ndarray]:
    X, y, groups, gender = voice_data()
    return vm.train_voice_model(X, y, groups, gender, model=request.param, **FAST)


def one_person(index: np.ndarray) -> pd.DataFrame:
    X, _, groups, _ = voice_data()
    return X[groups == groups[index[0]]]


def test_result_vector_has_every_d5_field(trained) -> None:
    bundle, test = trained
    result = vm.predict_voice(one_person(test), bundle)
    assert tuple(result) == vm.RESULT_FIELDS
    assert 0 <= result["ci_lower"] <= result["ci_upper"] <= 1
    assert 0 <= result["risk_score"] <= 1
    assert result["triage_priority"] in vm.TRIAGE_VALUES
    assert result["availability_mask"] == {"voice": True}
    assert len(result["shap_top5"]) == vm.TOP_K
    assert result["quality_gate_passed"] and not result["abstained"]
    assert result["sex_fairness_note"].startswith("Exploratory")


def test_holdout_people_are_never_trained_on(trained) -> None:
    bundle, test = trained
    _, _, groups, _ = voice_data()
    card = bundle.model_card
    assert card["n_training_people"] + card["n_holdout_people"] == N_PEOPLE
    assert card["holdout_check"]["n_people"] == card["n_holdout_people"]
    assert bundle.n_training == N_PEOPLE - len(np.unique(groups[test]))


def test_signal_is_found_on_holdout(trained) -> None:
    bundle, _ = trained
    assert bundle.model_card["holdout_check"]["auc"]["value"] > 0.8


def test_triage_wording_is_never_diagnostic(trained) -> None:
    bundle, test = trained
    result = vm.predict_voice(one_person(test), bundle)
    text = json.dumps(result).lower() + json.dumps(bundle.model_card).lower()
    for word in (
        "diagnosed with",
        "positive",
        "negative",
        "cleared",
        "discharged",
        "healthy result",
    ):
        assert word not in text
    assert set(vm.TRIAGE_VALUES) == {"prioritised_referral", "not_prioritised"}


@pytest.mark.parametrize(
    ("breakage", "match"),
    [
        (lambda r: r.drop(columns=["f3"]), "missing"),
        (lambda r: r.assign(f4=np.nan), "blank or non-numeric"),
        (lambda r: r.assign(gender=1), "D7"),
        (lambda r: r.assign(UPDRS_III=10.0), "UPDRS"),
        (lambda r: pd.concat([r, r]), "at most"),
        (lambda r: r.iloc[:0], "no recordings"),
    ],
)
def test_bad_input_abstains_instead_of_guessing(trained, breakage, match: str) -> None:
    """Red test: every broken input must abstain, with no score and no triage."""
    bundle, test = trained
    result = vm.predict_voice(breakage(one_person(test)), bundle)
    assert result["abstained"] and not result["quality_gate_passed"]
    assert result["risk_score"] is None and result["triage_priority"] is None
    assert result["ci_lower"] is None and result["ci_upper"] is None
    assert result["availability_mask"] == {"voice": False}  # absent, never zero-filled
    assert any(match in p for p in result["problems"])
    assert tuple(result) == vm.RESULT_FIELDS


def test_shap_values_add_up_to_the_model_output(trained) -> None:
    """Local accuracy: baseline + sum of SHAP values = the model's log-odds, per recording."""
    bundle, test = trained
    X = one_person(test)[bundle.feature_names].to_numpy(dtype=float)
    shap = vm.shap_values(bundle.model, X, bundle.feature_names)
    total = shap.sum(axis=1).to_numpy()
    p = bundle.model.predict_proba(X)[:, 1]
    assert np.allclose(total, np.log(p / (1 - p)), atol=1e-6)


def test_explanation_is_faithful_by_deletion(trained) -> None:
    """Faithfulness: resetting the top-5 features to the training mean moves the
    model output more than resetting 5 other selected features does."""
    bundle, test = trained
    X, _, groups, _ = voice_data()
    shift_top, shift_other = [], []
    for person in np.unique(groups[test])[:10]:
        rows = X[groups == person][bundle.feature_names].to_numpy(dtype=float)
        shap = vm.shap_values(bundle.model, rows, bundle.feature_names).drop(columns="_baseline")
        order = shap.mean().abs().sort_values(ascending=False).index
        base = bundle.model.predict_proba(rows)[:, 1].mean()
        mean = bundle.model["scale"].mean_
        for chosen, out in ((order[:5], shift_top), (order[-5:], shift_other)):
            edited = rows.copy()
            idx = [bundle.feature_names.index(c) for c in chosen]
            edited[:, idx] = mean[idx]
            out.append(abs(bundle.model.predict_proba(edited)[:, 1].mean() - base))
    assert np.mean(shift_top) > np.mean(shift_other)


def test_calibration_is_monotonic() -> None:
    rng = np.random.default_rng(0)
    raw = rng.random(200)
    y = (raw + rng.normal(0, 0.2, 200) > 0.5).astype(int)
    calibrator = vm.fit_platt(raw, y)
    grid = np.linspace(0.01, 0.99, 50)
    assert np.all(np.diff(vm.calibrate(calibrator, grid)) > 0)


def test_gender_as_training_input_is_refused() -> None:
    X, y, groups, gender = voice_data()
    with pytest.raises(ValueError, match="D7"):
        vm.train_voice_model(X.assign(gender=gender), y, groups, gender, **FAST)


def test_training_is_reproducible() -> None:
    X, y, groups, gender = voice_data()
    a, test = vm.train_voice_model(X, y, groups, gender, **FAST)
    b, _ = vm.train_voice_model(X, y, groups, gender, **FAST)
    rows = one_person(test)
    assert vm.predict_voice(rows, a) == vm.predict_voice(rows, b)


def test_demo_scores_five_people_and_one_abstention() -> None:
    X, y, groups, gender = voice_data()
    _, results = demo_voice.run_demo(X, y, groups, gender, **FAST)
    scored = [r for r in results if not r["abstained"]]
    assert len(scored) == demo_voice.N_DEMO
    assert {r["reference_label"] for r in scored} == {"PD", "healthy"}
    assert results[-1]["abstained"]
    text = json.dumps(results)
    assert not any(str(pid) in text for pid in range(3000, 3000 + N_PEOPLE))
    assert "|" in demo_voice.summary_table(results)
