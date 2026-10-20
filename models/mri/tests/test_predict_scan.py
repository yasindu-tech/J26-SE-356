"""Tests for models/mri/src/predict_scan.py. The image tools (dcm2niix, FSL, ANTs) are
replaced by fakes, so these run anywhere; the scoring uses a real (synthetic) bundle."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("lightgbm")
nib = pytest.importorskip("nibabel")

import mri_bundle as mb  # noqa: E402
import predict_scan as ps  # noqa: E402
from test_mri_bundle import cv_payload, scanners  # noqa: E402
from test_train_mri import T1_COLS, cohort  # noqa: E402


@pytest.fixture(scope="module")
def trained():
    d = cohort(n=40, signal=2.0)
    s = mb.build_set(d, scanners(d), cv_payload() | {"run": "r1"}, n_splits=4, n_repeats=1)
    return d, mb.Bundle("v-test", {"t1": s})


def write_series(folder: Path, stem: str, shape, desc: str, model: str, bvals=None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4))
    path = folder / f"{stem}.nii.gz"
    nib.save(img, str(path))
    # the key dcm2niix really writes (BIDS spelling, with an s)
    meta = {
        "SeriesDescription": desc,
        "SeriesNumber": int(stem[1]),
        "ManufacturersModelName": model,
    }
    (folder / f"{stem}.json").write_text(json.dumps(meta))
    if bvals is not None:
        (folder / f"{stem}.bval").write_text(" ".join(map(str, bvals)))
        (folder / f"{stem}.bvec").write_text("0\n0\n0\n")
    return path


# ------------------------------------------------------------------ sorting series


def test_series_are_sorted_by_content_not_name(tmp_path: Path) -> None:
    t1 = write_series(tmp_path, "s2_MPRAGE", (10, 10, 10), "MPRAGE GRAPPA", "Prisma_fit")
    dti = write_series(
        tmp_path, "s3_DTI", (5, 5, 5, 8), "DTI_gated", "Prisma_fit", [0, 0] + [1000] * 6
    )
    few = write_series(tmp_path, "s4_DTI", (5, 5, 5, 3), "DTI", "Prisma_fit", [0, 1000, 1000])
    loc = write_series(tmp_path, "s1_loc", (10, 10, 3), "localizer", "Prisma_fit")
    adc = write_series(tmp_path, "s3_DTI_ADC", (5, 5, 5), "DTI_ADC", "Prisma_fit")
    found = {p.name: ps.classify(p) for p in (t1, dti, few, loc, adc)}
    assert found["s2_MPRAGE.nii.gz"].kind == "t1"
    assert found["s3_DTI.nii.gz"].kind == "dti" and found["s3_DTI.nii.gz"].n_shell_dirs == 6
    assert found["s1_loc.nii.gz"].kind == "other"
    assert found["s3_DTI_ADC.nii.gz"] is None  # scanner-made map, never an input
    series = [s for s in found.values() if s]
    assert ps.choose_t1(series).description == "MPRAGE GRAPPA"
    assert ps.choose_dti(series).description == "DTI_gated"  # 3 directions is too few


def test_zip_with_paths_outside_its_folder_is_refused(tmp_path: Path) -> None:
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("../escape.dcm", b"x")
    with pytest.raises(ps.UploadError, match="outside"):
        ps.extract_upload(bad, tmp_path / "out")
    not_zip = tmp_path / "a.txt"
    not_zip.write_text("hello")
    with pytest.raises(ps.UploadError, match="neither"):
        ps.extract_upload(not_zip, tmp_path / "out2")


def test_feature_labels_are_plain_words() -> None:
    assert ps.feature_label("L_putamen_grey_frac") == "Left putamen: grey-matter share"
    assert ps.feature_label("R_caudate_t1_entropy").startswith("Right caudate: texture")
    assert (
        ps.feature_label("dti_L_cst_fa_mean") == "Left corticospinal tract: FA (fibre organisation)"
    )
    assert ps.feature_label("dti_CC_genu_md_mean").startswith("corpus callosum (front part)")
    assert ps.feature_label("dti_wm_fa_median").startswith("All white matter")
    assert ps.feature_label("brain_volume_ml") == "Whole-brain volume"


# ------------------------------------------------------------------ scoring


def prepared_from(d, i: int, model: str = "Prisma_fit") -> ps.Prepared:
    p = ps.Prepared()
    p.features = d.features.iloc[i].to_dict()
    p.scanner = {"t1": model if d.batches["t1"][i] == "A" else "TrioTim"}
    p.available = {"t1"}
    return p


def test_score_matches_the_final_model_and_has_an_interval(trained) -> None:
    d, b = trained
    chosen, parts = ps.score_person(b, prepared_from(d, 0), float(d.age[0]), float(d.sex[0]))
    assert chosen is b.sets["t1"]
    low, high = parts["interval"]
    assert low <= high and parts["n_copies"] == 4
    assert sum(parts["contrib"].values()) + parts["base"] == pytest.approx(parts["value"])


def test_unseen_scanner_model_is_not_scored(trained) -> None:
    d, b = trained
    chosen, reason = ps.score_person(b, prepared_from(d, 0, "Ingenia"), 65.0, 1.0)
    assert chosen is None and "not in the training data" in reason


def test_nothing_usable_is_not_scored(trained) -> None:
    _, b = trained
    chosen, reason = ps.score_person(b, ps.Prepared(), 65.0, 1.0)
    assert chosen is None and "nothing to score" in reason


# ------------------------------------------------------------------ the whole run (faked tools)


def fake_tools(monkeypatch, d, t1_status: str = "ok") -> None:
    import preprocess_dti
    import preprocess_t1
    import t1_features

    def dcm2niix(upload: Path, out: Path) -> None:
        write_series(out, "s2_MPRAGE", (10, 10, 10), "MPRAGE", "Prisma_fit")

    def t1_pipeline(nifti, sid, out_root, recorder=None, **_):
        with recorder.step(sid, "n4"):
            pass
        return SimpleNamespace(
            status=t1_status,
            reason="" if t1_status == "ok" else "mask too small",
            brain_mask_ml=1500.0,
            template_correlation=0.75,
        )

    monkeypatch.setattr(ps, "run_dcm2niix", dcm2niix)
    monkeypatch.setattr(preprocess_t1, "preprocess_t1", t1_pipeline)
    monkeypatch.setattr(preprocess_dti, "preprocess_dti", lambda *a, **k: pytest.fail("no DTI"))
    monkeypatch.setattr(t1_features, "load_atlas", lambda *a: None)
    monkeypatch.setattr(
        t1_features,
        "extract_subject",
        lambda *a, **k: SimpleNamespace(
            status="ok", reason="", features=d.features.iloc[0].to_dict()
        ),
    )


def test_full_run_gives_vector_score_explanation_and_card(monkeypatch, tmp_path, trained) -> None:
    d, b = trained
    fake_tools(monkeypatch, d)
    upload = tmp_path / "dicom"
    upload.mkdir()
    status = tmp_path / "status.json"
    r = ps.predict_scan(
        upload, float(d.age[0]), "M" if d.sex[0] else "F", tmp_path / "w", b, status_file=status
    )
    assert r["status"] == "scored" and r["model_set"] == "t1" and r["model_version"] == "v-test"
    fv = r["feature_vector"]
    assert fv["names"] == list(T1_COLS) and len(fv["raw"]) == len(fv["harmonised"]) == 6
    assert r["score"]["kind"] == "log_odds" and r["score"]["calibrated"] is False
    assert r["explanation"]["top_drivers"] and r["model_card"]["purpose"].endswith(
        "Not a diagnosis."
    )
    assert r["inputs"]["t1"]["found"] and not r["inputs"]["dti"]["found"]
    assert r["availability"] == {"t1": True, "dti": False}
    assert "no DTI series" in " ".join(r["validation"]["problems"])
    done = json.loads(status.read_text())["done"]
    assert done == ["convert", "t1", "dti", "features", "score", "explain"]
    text = json.dumps(r).lower()
    assert "diagnosis" not in text.replace("not a diagnosis", "")  # screening wording only


def test_failed_qc_abstains_instead_of_guessing(monkeypatch, tmp_path, trained) -> None:
    d, b = trained
    fake_tools(monkeypatch, d, t1_status="failed")
    upload = tmp_path / "dicom"
    upload.mkdir()
    r = ps.predict_scan(upload, 70.0, "F", tmp_path / "w", b)
    assert r["status"] == "abstained" and r["score"] is None
    assert any("T1 failed" in p for p in r["validation"]["problems"])


def test_missing_age_or_sex_abstains(tmp_path, trained) -> None:
    _, b = trained
    r = ps.predict_scan(tmp_path, None, "M", tmp_path / "w", b)
    assert r["status"] == "abstained" and "age and sex" in r["abstain_reason"]
    r = ps.predict_scan(tmp_path, 60.0, "X", tmp_path / "w", b)
    assert r["status"] == "abstained"


def test_scanner_model_read_from_either_sidecar_spelling(tmp_path: Path) -> None:
    bids = write_series(tmp_path, "s2_MPRAGE", (10, 10, 10), "MPRAGE", "TrioTim")
    assert ps.classify(bids).model == "TrioTim"
    old = write_series(tmp_path / "old", "s2_MPRAGE", (10, 10, 10), "MPRAGE", "x")
    meta_path = old.with_name("s2_MPRAGE.json")
    meta = json.loads(meta_path.read_text())
    meta["ManufacturerModelName"] = meta.pop("ManufacturersModelName").replace("x", "Prisma")
    meta_path.write_text(json.dumps(meta))
    assert ps.classify(old).model == "Prisma"
