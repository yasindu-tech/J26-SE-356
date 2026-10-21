"""Tests for predict_video (GAIT-27). The video steps are faked; the model is a small real one."""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import explain_words as ew
import model_bundle as mb
import numpy as np
import predict_video as pv
import pytest
import train_xgboost as tx
from gait_extract import FEATURE_NAMES, extract_features, rejected
from predict_video import GaitPredictor, predict_video
from test_evaluate_cohorts import FAST, cohort_table
from test_video_to_joints import Fake, make_trace
from video_to_joints import video_to_joints

BANNED = re.compile(r"diagnos|parkinson|disease|patient|\bpd\b|chance|indicat|cause|due to", re.I)


@pytest.fixture(scope="module")
def bundle_dir() -> Path:
    import shap

    table = cohort_table()
    model = tx.fit_final_model(table, params=FAST)
    x = table[list(FEATURE_NAMES)].to_numpy(float)
    values = tx.shap_array(shap.TreeExplainer(model).shap_values(x))
    ref = ew.build_reference(x, values, model.predict_proba(x).argmax(axis=1))
    folder = Path(tempfile.mkdtemp())
    mb.save_bundle(model, folder, {"explanation_reference": ref})
    return folder


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Fake:
    return Fake(monkeypatch, tmp_path, make_trace())


@pytest.fixture
def predictor(bundle_dir: Path, fake: Fake) -> GaitPredictor:
    return GaitPredictor(bundle_dir, fake.model)


# --------------------------------------------------------------------------- a scored walk
def test_a_good_walk_is_scored_with_the_whole_record(predictor: GaitPredictor, fake: Fake) -> None:
    r = predictor.predict(fake.video)
    assert r["available"] is True and r["modality"] == "gait" and r["reasons"] == []
    assert r["model_version"] == predictor.bundle.version
    assert r["class_labels"] == ["class 0", "class 1", "class 2-3"]
    assert r["predicted_label"] == r["class_labels"][r["predicted_class"]]
    assert len(r["top_features"]) == 3 and all(e["sentence"] for e in r["top_features"])
    assert list(r["feature_vector"]) == list(FEATURE_NAMES)
    assert r["gate"]["passed"] is True and r["walk"]["seconds"] > 0
    assert r["probabilities_calibrated"] is False
    json.dumps(r, allow_nan=False)  # JSON-safe


def test_the_scores_are_the_models_scores_for_the_eight_inputs_of_that_walk(
    predictor: GaitPredictor, fake: Fake
) -> None:
    r = predictor.predict(fake.video)
    walk = video_to_joints(fake.video, fake.model)
    x = extract_features(walk.joints, walk.fps).vector()
    assert np.allclose(list(r["feature_vector"].values()), x)
    expected = predictor.bundle.predict_proba(x)[0]
    assert np.allclose(r["probabilities"], expected) and sum(r["probabilities"]) == pytest.approx(1)
    assert r["predicted_class"] == int(expected.argmax())
    assert r["expected_severity"] == pytest.approx(expected[1] + 2 * expected[2])
    assert 0 <= r["expected_severity"] <= 2


def test_the_one_person_warning_goes_when_a_count_of_one_is_given(
    predictor: GaitPredictor, fake: Fake
) -> None:
    assert any(
        "One person in view is assumed" in w for w in predictor.predict(fake.video)["warnings"]
    )
    r = predictor.predict(fake.video, people_in_view=1)
    assert r["available"] and not any("assumed" in w for w in r["warnings"])


def test_warnings_of_the_feature_step_join_the_gates(
    predictor: GaitPredictor, fake: Fake, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = pv.extract_features
    monkeypatch.setattr(
        pv, "extract_features", lambda *a, **k: replace(real(*a, **k), flags=("odd",))
    )
    r = predictor.predict(fake.video)
    assert r["available"] and "odd" in r["warnings"] and len(r["warnings"]) == 2


def test_the_record_never_carries_the_file_name(predictor: GaitPredictor, fake: Fake) -> None:
    assert "secret_name" not in json.dumps(predictor.predict(fake.video))
    assert re.fullmatch(r"w-[0-9a-f]{8}", predictor.predict(fake.video)["video"])


# --------------------------------------------------------------------------- not scored
def assert_not_scored(r: dict[str, Any]) -> None:
    assert r["available"] is False and r["reasons"]
    for key in ("predicted_class", "predicted_label", "probabilities", "expected_severity"):
        assert r[key] is None
    assert r["top_features"] == [] and r["feature_vector"] is None
    assert r["model_version"]
    json.dumps(r, allow_nan=False)


def test_a_video_without_a_walk_is_not_scored_and_nothing_is_raised(
    bundle_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trace = make_trace()
    trace = replace(trace, detected=np.zeros_like(trace.detected))
    f = Fake(monkeypatch, tmp_path, trace)
    r = GaitPredictor(bundle_dir, f.model).predict(f.video)
    assert_not_scored(r)
    assert "No walk was found" in r["reasons"][0] and r["walk"] is None


def test_a_missing_video_is_not_scored(predictor: GaitPredictor, tmp_path: Path) -> None:
    r = predictor.predict(tmp_path / "nothing.mov")
    assert_not_scored(r)
    assert "missing or empty" in r["reasons"][0]


def test_a_gate_rejection_gives_its_reasons_and_the_walk(
    predictor: GaitPredictor, fake: Fake
) -> None:
    r = predictor.predict(fake.video, people_in_view=2)
    assert_not_scored(r)
    assert r["reasons"] == ["2 people are in view; the video must show one person walking."]
    assert r["gate"]["passed"] is False and r["walk"]["seconds"] > 0


def test_a_feature_rejection_is_explained_in_plain_words(
    predictor: GaitPredictor, fake: Fake, monkeypatch: pytest.MonkeyPatch
) -> None:
    for reason, start in (
        ("real strides on one foot only", "Real strides were found on only one foot"),
        ("leg length 0.10 m is implausible", "The leg length measured"),
        ("2 real strides on both feet (need 3)", "The walk could not be turned"),
    ):
        monkeypatch.setattr(pv, "extract_features", lambda *a, _r=reason, **k: rejected(_r))
        r = predictor.predict(fake.video)
        assert_not_scored(r)
        assert r["reasons"][0].startswith(start) and r["gate"]["passed"] is True


def test_a_frame_rate_the_inputs_were_not_made_for_is_not_scored(
    predictor: GaitPredictor, fake: Fake, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_a: object, **_k: object) -> None:
        raise ValueError("frame rate 120 fps is outside 20-60")

    monkeypatch.setattr(pv, "extract_features", refuse)
    r = predictor.predict(fake.video)
    assert_not_scored(r)
    assert "20 to 60" in r["reasons"][0]


# --------------------------------------------------------------------------- the predictor
def test_a_missing_pose_model_or_reference_stops_the_start(
    bundle_dir: Path, tmp_path: Path, fake: Fake
) -> None:
    with pytest.raises(FileNotFoundError, match="pose model"):
        GaitPredictor(bundle_dir, tmp_path / "none.task")
    table = cohort_table()
    bare = mb.save_bundle(tx.fit_final_model(table, params=FAST), tmp_path / "bare", {})
    assert bare.version
    with pytest.raises(mb.BundleError, match="reference"):
        GaitPredictor(tmp_path / "bare", fake.model)


def test_predict_video_uses_one_loaded_predictor(
    predictor: GaitPredictor, fake: Fake, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pv, "default_predictor", lambda: predictor)
    r = predict_video(fake.video, people_in_view=2)
    assert r["available"] is False and r["reasons"][0].startswith("2 people")


def test_the_time_limit_is_passed_on(bundle_dir: Path, fake: Fake) -> None:
    GaitPredictor(bundle_dir, fake.model, max_seconds=5.0).predict(fake.video)
    assert fake.asked == [5.0]


# --------------------------------------------------------------------------- command line
def run_cli(fake: Fake, bundle_dir: Path, *args: str) -> int:
    return pv.main(
        [str(fake.video), "--bundle", str(bundle_dir), "--model", str(fake.model), *args]
    )


def test_cli_prints_the_verdict_the_scores_and_the_sentences(
    bundle_dir: Path, fake: Fake, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_cli(fake, bundle_dir) == 0
    out = capsys.readouterr().out
    assert "SCORED" in out and "uncalibrated" in out and "not a UPDRS score" in out
    assert out.count("  - The model used") == 3 and "secret_name" not in out
    assert "One person in view is assumed" in out
    assert not BANNED.search(out)


def test_cli_json_is_the_record(
    bundle_dir: Path, fake: Fake, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_cli(fake, bundle_dir, "--json") == 0
    record = json.loads(capsys.readouterr().out)
    assert record["available"] and record["modality"] == "gait"


def test_cli_exit_codes_for_a_rejection_and_for_a_broken_setup(
    bundle_dir: Path, fake: Fake, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_cli(fake, bundle_dir, "--people", "2") == 1
    out = capsys.readouterr().out
    assert "NOT SCORED" in out and "2 people are in view" in out and not BANNED.search(out)
    assert (
        pv.main([str(fake.video), "--bundle", str(tmp_path / "nope"), "--model", str(fake.model)])
        == 2
    )
    assert "Cannot start" in capsys.readouterr().out
    assert (
        pv.main([str(fake.video), "--bundle", str(bundle_dir), "--model", str(tmp_path / "x")]) == 2
    )


# --------------------------------------------------------------------------- GAIT-28 additions
def test_the_record_carries_its_schema_version_and_the_confirmation(
    predictor: GaitPredictor, fake: Fake
) -> None:
    r = predictor.predict(fake.video)
    assert r["schema_version"] == "gait-result/1" and r["one_person_confirmed"] is False
    assert list(r)[:2] == ["modality", "schema_version"]
    gone = predictor.predict(fake.video, people_in_view=2)
    assert gone["schema_version"] == "gait-result/1" and gone["modality"] == "gait"


def test_a_confirmed_upload_changes_the_one_person_warning_and_nothing_else(
    predictor: GaitPredictor, fake: Fake
) -> None:
    plain = predictor.predict(fake.video)
    confirmed = predictor.predict(fake.video, one_person_confirmed=True)
    assert confirmed["one_person_confirmed"] is True
    assert pv.ONE_PERSON_CONFIRMED in confirmed["warnings"]
    assert not any("is assumed" in w for w in confirmed["warnings"])
    assert confirmed["probabilities"] == plain["probabilities"]
    assert confirmed["gate"] == plain["gate"]  # the gate's own record is left as it was
    assert not BANNED.search(pv.ONE_PERSON_CONFIRMED)


def test_a_bundle_with_a_banned_input_is_refused_before_any_scoring(
    bundle_dir: Path, fake: Fake, monkeypatch: pytest.MonkeyPatch
) -> None:
    from feature_contract import FeatureContractError

    real = mb.load_bundle

    def with_label_input(directory: Path) -> mb.Bundle:
        b = real(directory)
        return mb.Bundle(b.model, {**b.info, "features": [*b.features[:-1], "UPDRS_GAIT"]})

    monkeypatch.setattr(pv, "load_bundle", with_label_input)
    with pytest.raises(FeatureContractError):
        GaitPredictor(bundle_dir, fake.model)


def test_cli_writes_the_record_to_a_file_with_the_confirmation(
    bundle_dir: Path, fake: Fake, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "deep" / "result.json"
    assert run_cli(fake, bundle_dir, "--out", str(out), "--one-person-confirmed") == 0
    record = json.loads(out.read_text())
    assert record["one_person_confirmed"] is True and record["available"] is True
    assert "You confirmed one person is walking" in capsys.readouterr().out
    assert run_cli(fake, bundle_dir, "--out", str(out), "--people", "2") == 1
    assert json.loads(out.read_text())["available"] is False


# --------------------------------------------------------------------------- quick pre-check
def test_a_video_refused_by_the_pre_check_is_not_scored_and_the_pose_pass_is_skipped(
    predictor: GaitPredictor, fake: Fake, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reason = "This does not look like a video of someone walking: no person with the legs in view was found."
    monkeypatch.setattr(pv, "prescreen", lambda *a, **k: reason)
    r = predictor.predict(fake.video, trace_out=tmp_path / "t.npz")
    assert_not_scored(r)
    assert r["reasons"] == [reason] and r["gate"]["passed"] is False and r["walk"] is None
    assert fake.asked == [] and not (tmp_path / "t.npz").exists()  # no pose pass, no trace


def test_the_pre_check_looks_at_the_part_that_will_be_read(
    predictor: GaitPredictor, fake: Fake, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[Path, Path, float | None]] = []
    monkeypatch.setattr(pv, "prescreen", lambda p, m, s=None: seen.append((p, m, s)))
    predictor.predict(fake.video)
    assert seen == [(fake.video, fake.model, predictor.max_seconds)]


def test_the_pose_trace_is_saved_when_asked_for(
    predictor: GaitPredictor, fake: Fake, tmp_path: Path
) -> None:
    from pose_trace_io import load_trace

    target = tmp_path / "out" / "t.npz"
    assert predictor.predict(fake.video, trace_out=target)["available"] is True
    back = load_trace(target)
    assert back.fps == fake.trace.fps and back.n_frames > 0
    assert predictor.predict(fake.video)["available"] is True  # and not when it is not


def test_the_pose_trace_is_saved_even_when_no_walk_is_found(
    bundle_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trace = make_trace()
    f = Fake(monkeypatch, tmp_path, replace(trace, detected=np.zeros_like(trace.detected)))
    r = GaitPredictor(bundle_dir, f.model).predict(f.video, trace_out=tmp_path / "t.npz")
    assert "No walk was found" in r["reasons"][0] and (tmp_path / "t.npz").is_file()
