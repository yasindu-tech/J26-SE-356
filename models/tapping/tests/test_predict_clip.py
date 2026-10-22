"""Tests for src/predict_clip.py (task FT-63): the schema of one result, the failed paths, no
path or file name in any output, NaN as null, and byte-identical output. Synthetic world and
a faked tracker, as in the demo tests; no real video."""

from __future__ import annotations

import base64
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent / "demo"))
sys.path.insert(0, str(HERE))

import data_access as da  # noqa: E402
import predict_clip as pc  # noqa: E402
import resources as res  # noqa: E402
import scoring as sc  # noqa: E402
import upload_pipeline as up  # noqa: E402
from test_demo_resources import add_folds, add_model  # noqa: E402
from test_demo_upload import DATA, Factory, fake_pipeline, synthetic_df  # noqa: E402
from test_shap_figures import make_world  # noqa: E402

HEX8 = re.compile(r"^[0-9a-f]{8}$")
SECRET = "patient_jane_doe_clip"
TODAY = "2026-10-21"
TOP = {
    "schema_version", "status", "clip_ref", "target", "recording", "quality", "score",
    "contributions", "curve", "signal", "preview_frames", "model", "pdf_base64",
}  # fmt: skip


def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, models: bool = True) -> Path:
    w = make_world(tmp_path, monkeypatch)
    add_model(w["proc"])
    if models:
        add_folds(w["proc"])
    monkeypatch.setenv(da.ENV_ROOT, str(w["proc"]))
    monkeypatch.setattr(res, "make_landmarker", lambda: Factory())
    fake_pipeline(monkeypatch)
    monkeypatch.setattr(
        up, "preview_frames", lambda p, idx, lm: [np.zeros((60, 400, 3), np.uint8)] * len(idx)
    )
    return w["proc"]


def video(tmp_path: Path, name: str = f"{SECRET}.mp4", data: bytes = DATA) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def stages() -> tuple[list[tuple[int, str]], Any]:
    seen: list[tuple[int, str]] = []
    return seen, lambda n, label: seen.append((n, label))


def keys_sorted(obj: Any) -> bool:
    if isinstance(obj, dict):
        return list(obj) == sorted(obj) and all(keys_sorted(v) for v in obj.values())
    if isinstance(obj, list):
        return all(keys_sorted(v) for v in obj)
    return True


def test_a_scored_result_has_the_documented_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world(tmp_path, monkeypatch)
    seen, progress = stages()
    out = pc.run(video(tmp_path), "ge2", progress, TODAY)
    assert set(out) == TOP and out["schema_version"] == "tapping-result/1"
    assert out["status"] == "scored" and out["target"] == "ge2"
    assert HEX8.match(out["clip_ref"])
    assert set(out["recording"]) == {"duration_s", "fps", "hand_found_pct", "n_taps"}
    assert set(out["quality"]) == {"flags", "notes"} and isinstance(out["quality"]["notes"], list)
    score = out["score"]
    assert set(score) == {
        "value", "band_low", "band_high", "band_basis", "threshold", "above_threshold",
        "cv_auroc", "cv_ci_low", "cv_ci_high",
    }  # fmt: skip
    assert 0.0 <= score["band_low"] <= score["band_high"] <= 1.0 and score["threshold"] == 0.5
    assert isinstance(score["above_threshold"], bool)
    rows = out["contributions"]
    assert len(rows) == 5
    for row in rows:
        assert set(row) == {"feature_label", "value_text", "percentile_text", "shap", "sentence"}
        assert isinstance(row["shap"], float) and row["sentence"].endswith(".")
    assert [abs(r["shap"]) for r in rows] == sorted((abs(r["shap"]) for r in rows), reverse=True)
    curve = out["curve"]
    assert len(curve["times"]) == len(curve["amps"]) and len(curve["line_x"]) == len(
        curve["line_y"]
    )
    assert isinstance(curve["usable"], bool) and curve["slope_sentence"] and curve["count_sentence"]
    sig = out["signal"]
    assert len(sig["t"]) == len(sig["smoothed"]) <= pc.MAX_POINTS
    assert len(sig["peak_t"]) == len(sig["peak_v"])
    assert out["model"].keys() == {"demo_model_sha8", "folds_sha8"}
    assert all(HEX8.match(v) for v in out["model"].values())
    assert [n for n, _ in seen] == [1, 2, 3, 4, 5]


def test_sentences_and_texts_come_from_the_demo_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world(tmp_path, monkeypatch)
    out = pc.run(video(tmp_path), "ge2", lambda n, label: None, TODAY)
    a = up.analyse_upload(DATA, ".mp4", Factory()).result
    features = sc.upload_features(a)
    model = res.load_demo_model("ge2")
    rows = sc.contributions(
        features, sc.explain(features, model)["shap"], sc.context(features, sc.load_reference())
    )
    assert [r["sentence"] for r in out["contributions"]] == sc.wording(rows)
    assert out["score"]["band_basis"] == pc.tx.BAND_UPLOAD
    assert out["score"]["value"] == pytest.approx(
        sc.score(features, model, res.load_fold_models("ge2"))["score"], abs=1e-6
    )  # noqa: E501


def test_the_preview_frames_are_small_jpegs_and_the_pdf_is_a_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world(tmp_path, monkeypatch)
    out = pc.run(video(tmp_path), "ge2", lambda n, label: None, TODAY)
    frames = out["preview_frames"]
    assert len(frames) == up.N_PREVIEW
    import cv2

    for f in frames:
        raw = base64.b64decode(f["jpeg_base64"])
        assert raw[:2] == b"\xff\xd8"
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        assert img is not None and img.shape[1] <= pc.PREVIEW_WIDTH
    assert base64.b64decode(out["pdf_base64"])[:5] == b"%PDF-"


def test_two_runs_give_identical_bytes_with_sorted_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world(tmp_path, monkeypatch)
    path = video(tmp_path)
    first = pc.to_json(pc.run(path, "ge2", lambda n, label: None, TODAY))
    second = pc.to_json(pc.run(path, "ge2", lambda n, label: None, TODAY))
    assert first == second
    assert keys_sorted(json.loads(first))
    assert "NaN" not in first and "Infinity" not in first


@pytest.mark.parametrize("case", ["too_long", "no_hand", "unreadable"])
def test_a_bad_video_gives_a_failed_result_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    world(tmp_path, monkeypatch)
    if case == "too_long":
        fake_pipeline(monkeypatch, probe=(30.0, 30 * 61))
    elif case == "no_hand":
        fake_pipeline(monkeypatch, df=synthetic_df(detected_share=0.2))
    else:

        def unreadable(path: Any) -> Any:
            raise up.UploadError("The video could not be read. Try an mp4 or mov file.")

        monkeypatch.setattr(up, "probe_video", unreadable)
    out = pc.run(video(tmp_path), "ge2", lambda n, label: None, TODAY)
    assert out["status"] == "failed" and HEX8.match(out["clip_ref"])
    assert set(out) == {"schema_version", "status", "reason", "clip_ref", "target"}
    assert out["reason"] and SECRET not in out["reason"] and str(tmp_path) not in out["reason"]
    assert "score" not in out and "contributions" not in out


def test_main_writes_exactly_one_json_line_and_only_stage_lines_on_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    world(tmp_path, monkeypatch)
    code = pc.main(["--video", str(video(tmp_path)), "--target", "ge2"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.count("\n") == 1 and json.loads(captured.out)["status"] == "scored"
    lines = captured.err.strip().splitlines()
    assert lines == [f"STAGE {i + 1}/5 {label}" for i, label in enumerate(pc.STAGES)]
    for text in (captured.out, captured.err):
        assert SECRET not in text and str(tmp_path) not in text


def test_main_exits_zero_with_failed_json_when_the_file_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    world(tmp_path, monkeypatch)
    code = pc.main(["--video", str(tmp_path / f"{SECRET}_missing.mp4")])
    captured = capsys.readouterr()
    out = json.loads(captured.out)
    assert code == 0 and out["status"] == "failed" and out["clip_ref"] is None
    assert SECRET not in captured.out + captured.err and str(tmp_path) not in captured.out


def test_the_cli_in_a_real_process_prints_only_json_on_stdout(tmp_path: Path) -> None:
    missing = tmp_path / f"{SECRET}.mp4"
    done = subprocess.run(
        [sys.executable, str(HERE.parent / "src" / "predict_clip.py"), "--video", str(missing)],
        capture_output=True, text=True, timeout=120, check=False,
    )  # fmt: skip
    assert done.returncode == 0
    out = json.loads(done.stdout)
    assert out["status"] == "failed" and done.stdout.count("\n") == 1
    assert SECRET not in done.stdout + done.stderr and str(tmp_path) not in done.stdout


def test_a_wrong_extension_is_refused_without_naming_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world(tmp_path, monkeypatch)
    out = pc.run(video(tmp_path, f"{SECRET}.avi"), "ge2", lambda n, label: None, TODAY)
    assert out["status"] == "failed" and out["reason"] == "Use an mp4, mov or m4v video."
    assert SECRET not in json.dumps(out)


def test_missing_models_give_a_failed_result_without_a_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world(tmp_path, monkeypatch, models=False)
    out = pc.run(video(tmp_path), "ge2", lambda n, label: None, TODAY)
    assert out["status"] == "failed" and out["reason"] == pc.MODELS_MISSING
    assert "/" not in out["reason"] and str(tmp_path) not in json.dumps(out)


def test_an_unexpected_error_is_a_fixed_text_never_its_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world(tmp_path, monkeypatch)

    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError(f"failed on {tmp_path}/{SECRET}.mp4")

    monkeypatch.setattr(up, "analyse_upload", boom)
    out = pc.run(video(tmp_path), "ge2", lambda n, label: None, TODAY)
    assert out["status"] == "failed" and out["reason"] == pc.GENERIC
    assert SECRET not in json.dumps(out)


def test_a_known_message_with_a_path_is_replaced() -> None:
    assert pc.safe_reason(up.UploadError("cannot open /Users/x/clip.mp4")) == pc.GENERIC
    assert pc.safe_reason(up.UploadError("The video is longer than 60 s.")) == (
        "The video is longer than 60 s."
    )


def test_nan_is_written_as_null_never_as_zero() -> None:
    assert pc.num(float("nan")) is None and pc.num(float("inf")) is None and pc.num(None) is None
    assert pc.num(0.0) == 0.0
    assert pc.nums([1.0, float("nan"), 2.5]) == [1.0, None, 2.5]
    text = pc.to_json({"a": pc.nums([float("nan")])})
    assert text == '{"a":[null]}'
    with pytest.raises(ValueError):
        pc.to_json({"a": float("nan")})


def test_thinning_keeps_the_ends_and_the_limit() -> None:
    assert list(pc.thin(5)) == [0, 1, 2, 3, 4]
    idx = pc.thin(10_000)
    assert len(idx) <= pc.MAX_POINTS and idx[0] == 0 and idx[-1] == 9_999
    assert list(idx) == sorted(set(idx))
