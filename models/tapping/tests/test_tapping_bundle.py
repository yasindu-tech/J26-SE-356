"""Tests for src/tapping_bundle.py (model-bundle-export): a bundle that loads, and one that is
refused for each broken rule: a changed, extra or missing file, a pickle, a downgrade, a
forbidden feature, person-level keys or clip identifiers. Synthetic world, no real data."""

from __future__ import annotations

import datetime
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.stats import percentileofscore

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent / "demo"))
sys.path.insert(0, str(HERE))

import data_access as da  # noqa: E402
import hand_landmarks as hl  # noqa: E402
import predict_clip as pc  # noqa: E402
import resources as res  # noqa: E402
import tapping_bundle as tb  # noqa: E402
from test_demo_resources import add_folds, add_model  # noqa: E402
from test_predict_clip import TODAY, video  # noqa: E402
from test_predict_clip import world as clip_world  # noqa: E402
from test_shap_figures import make_world  # noqa: E402

T0 = datetime.datetime(2026, 10, 22, 12, 0, 0)
COMMIT = "abc1234"


def processed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A synthetic data/processed with both targets' models and a fake landmarker file."""
    proc = make_world(tmp_path, monkeypatch)["proc"]
    for target in da.TARGETS:
        add_model(proc, target)
        add_folds(proc, target)
    fake = tmp_path / "hand_landmarker.task"
    fake.write_bytes(b"not a real landmarker")
    monkeypatch.setattr(hl, "EXPECTED_MODEL_SHA256", hashlib.sha256(fake.read_bytes()).hexdigest())
    monkeypatch.setattr(res, "make_landmarker", lambda path=None: res.LandmarkerHandle(fake, ""))
    return proc


def built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    proc = processed(tmp_path, monkeypatch)
    root = tmp_path / "models" / "tapping"
    return proc, tb.build(proc, root, now=T0, commit=COMMIT)


def rehash(folder: Path) -> None:
    """Rewrite the manifest's hashes, as someone hiding an edit would."""
    path = folder / tb.MANIFEST
    m = json.loads(path.read_text())
    m["files"] = {p.relative_to(folder).as_posix(): tb.sha256_file(p)
                  for p in sorted(folder.rglob("*")) if p.is_file() and p.name != tb.MANIFEST}  # fmt: skip
    path.write_text(json.dumps(m))


# --------------------------------------------------------------------------- a good bundle


def test_a_built_bundle_loads_and_lists_every_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, folder = built(tmp_path, monkeypatch)
    root = folder.parent
    assert (root / tb.POINTER).read_text().strip() == folder.name == f"20261022_120000_{COMMIT}"
    b = tb.load(root)
    m = b.manifest
    on_disk = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    assert set(m["files"]) == on_disk - {tb.MANIFEST}
    assert len([n for n in m["files"] if n.startswith("folds/ge2/fold_")]) == 25
    assert m["feature_names"] == tb.sc.SET_A and m["contains_person_level_data"] is False
    assert m["git_commit"] == COMMIT and m["libraries"]["lightgbm"]
    assert all(len(q) == 101 for q in b.reference.values())
    assert len(b.folds("ge2")) == 25 and b.model("ge1").num_feature() == 5
    assert "diagnos" not in json.dumps(m).replace("Not a diagnostic device", "")


def test_the_bundle_holds_no_research_rows_or_identifiers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc, folder = built(tmp_path, monkeypatch)
    assert not list(folder.rglob("*.csv")) and not list(folder.rglob("*.parquet"))
    assert tb.leaked_identifiers(folder, proc) == []
    for p in folder.rglob("*.json"):
        assert tb.person_keys(json.loads(p.read_text())) == set(), p.name


def test_the_score_from_the_bundle_equals_the_score_from_data_processed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip_world(tmp_path, monkeypatch)  # legacy: no bundle installed
    before = pc.run(video(tmp_path), "ge2", lambda n, label: None, TODAY)
    proc = da.processed_root()
    add_model(proc, "ge1")
    add_folds(proc, "ge1")
    fake = tmp_path / "lm.task"
    fake.write_bytes(b"x")
    monkeypatch.setattr(hl, "EXPECTED_MODEL_SHA256", tb.sha256_file(fake))
    real_make = res.make_landmarker
    monkeypatch.setattr(res, "make_landmarker", lambda path=None: res.LandmarkerHandle(fake, ""))
    root = tmp_path / "bundles"
    tb.build(proc, root, now=T0, commit=COMMIT)
    monkeypatch.setattr(res, "make_landmarker", real_make)
    clip_world(tmp_path / "w2", monkeypatch)  # the same synthetic clip, a fresh processed root
    monkeypatch.setenv(tb.ENV_ROOT, str(root))
    shutil.rmtree(da.processed_root())  # the bundle must not need data/processed at all
    after = pc.run(video(tmp_path), "ge2", lambda n, label: None, TODAY)
    assert before["status"] == after["status"] == "scored"
    assert after["score"] == before["score"] and after["model"] == before["model"]
    for a, b in zip(after["contributions"], before["contributions"], strict=True):
        assert a["shap"] == b["shap"] and a["feature_label"] == b["feature_label"]


def test_bundle_percentiles_stay_within_one_point_of_the_full_table() -> None:
    rng = np.random.default_rng(0)
    values = rng.lognormal(size=233)
    values[:20] = values[0]  # ties, as tap counts have
    q = tb.quantiles(values)
    for x in np.linspace(values.min() - 1, values.max() + 1, 2001):
        assert abs(tb.percentile_of(float(x), q) - percentileofscore(values, x, kind="mean")) < 1.0


def test_a_missing_feature_value_has_no_percentile_never_zero() -> None:
    ref = {c: list(np.linspace(0.0, 1.0, 101)) for c in tb.sc.SET_A}
    features = {c: 0.5 for c in tb.sc.SET_A} | {"iti_cv": float("nan")}
    ctx = tb.context(features, ref)
    assert np.isnan(ctx["iti_cv"]["percentile"]) and ctx["tap_rate_hz"]["percentile"] == 50.0


# --------------------------------------------------------------------------- refused bundles


def test_a_changed_file_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, folder = built(tmp_path, monkeypatch)
    target = folder / "folds" / "ge2" / "fold_r2_f3.txt"
    target.write_bytes(target.read_bytes().replace(b"leaf_value=", b"leaf_value=1", 1))
    with pytest.raises(tb.BundleError, match="fold_r2_f3.txt does not match its hash"):
        tb.load(folder.parent)


def test_an_extra_or_missing_file_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, folder = built(tmp_path, monkeypatch)
    (folder / "notes.txt").write_text("hello")
    with pytest.raises(tb.BundleError, match="unlisted files"):
        tb.load(folder.parent)
    (folder / "notes.txt").unlink()
    (folder / tb.REFERENCE).unlink()
    with pytest.raises(tb.BundleError, match="missing"):
        tb.load(folder.parent)


def test_a_pickle_is_refused_even_with_a_matching_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, folder = built(tmp_path, monkeypatch)
    (folder / "model.joblib").write_bytes(b"\x80\x04")
    rehash(folder)
    with pytest.raises(tb.BundleError, match="pickled files are refused"):
        tb.load(folder.parent)


@pytest.mark.parametrize(
    "name",
    ["updrs_part3", "hoehn_yahr_stage", "clinician_rating", "medication_on", "levodopa_dose",
     "datscan_sbr", "qsm_putamen", "ledd_mg", "on_off_state"],
)  # fmt: skip
def test_forbidden_feature_names_are_refused(name: str) -> None:
    with pytest.raises(tb.BundleError, match="forbidden"):
        tb.check_feature_names(["tap_rate_hz", name])


def test_a_bundle_with_a_forbidden_feature_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, folder = built(tmp_path, monkeypatch)
    m = json.loads((folder / tb.MANIFEST).read_text())
    m["feature_names"] = [*m["feature_names"], "updrs_iii_total"]
    (folder / tb.MANIFEST).write_text(json.dumps(m))
    with pytest.raises(tb.BundleError, match="forbidden"):
        tb.load(folder.parent)


def test_a_downgrade_or_a_rebuild_of_the_same_version_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc, folder = built(tmp_path, monkeypatch)
    root = folder.parent
    with pytest.raises(tb.BundleError, match="not newer"):
        tb.build(proc, root, now=T0 - datetime.timedelta(days=1), commit=COMMIT)
    with pytest.raises(tb.BundleError, match="not newer"):
        tb.build(proc, root, now=T0, commit=COMMIT)
    newer = tb.build(proc, root, now=T0 + datetime.timedelta(hours=1), commit=COMMIT)
    assert tb.current_version(root) == newer.name and tb.load(root).version == newer.name


def test_person_level_keys_stop_the_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proc = processed(tmp_path, monkeypatch)
    real = tb.model_card

    def leaky(*a: Any, **k: Any) -> dict[str, Any]:
        return real(*a, **k) | {"examples": [{"participant": "aaaaaaaa", "rating": 3}]}

    monkeypatch.setattr(tb, "model_card", leaky)
    root = tmp_path / "models"
    with pytest.raises(tb.BundleError, match="person-level keys"):
        tb.build(proc, root, now=T0, commit=COMMIT)
    assert not root.exists() or not any(root.iterdir())  # nothing half-written, no CURRENT


def test_a_clip_identifier_in_any_file_is_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc, folder = built(tmp_path, monkeypatch)
    (folder / "folds" / "ge2" / "fold_r0_f0.txt").write_text("trained on 0000000a\n")
    assert tb.leaked_identifiers(folder, proc) == ["fold_r0_f0.txt"]


def test_no_bundle_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(tb.BundleError, match="no tapping model bundle installed"):
        tb.load(tmp_path / "empty")


def test_a_wrong_lightgbm_version_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, folder = built(tmp_path, monkeypatch)
    monkeypatch.setattr(tb.lightgbm, "__version__", "0.0.1")
    with pytest.raises(tb.BundleError, match="needs lightgbm"):
        tb.load(folder.parent)


def test_a_tampered_bundle_fails_the_clip_and_never_falls_back_to_data_processed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip_world(tmp_path, monkeypatch)  # data/processed would score this clip
    proc = da.processed_root()
    add_model(proc, "ge1")
    add_folds(proc, "ge1")
    fake = tmp_path / "lm.task"
    fake.write_bytes(b"x")
    monkeypatch.setattr(hl, "EXPECTED_MODEL_SHA256", tb.sha256_file(fake))
    factory = res.make_landmarker
    monkeypatch.setattr(res, "make_landmarker", lambda path=None: res.LandmarkerHandle(fake, ""))
    root = tmp_path / "bundles"
    folder = tb.build(proc, root, now=T0, commit=COMMIT)
    monkeypatch.setattr(res, "make_landmarker", factory)
    (folder / res.MODEL_FILE.format(target="ge2")).write_text("changed")
    monkeypatch.setenv(tb.ENV_ROOT, str(root))
    out = pc.run(video(tmp_path), "ge2", lambda n, label: None, TODAY)
    assert out["status"] == "failed" and out["reason"] == pc.MODELS_CHANGED
    assert str(tmp_path) not in json.dumps(out)
