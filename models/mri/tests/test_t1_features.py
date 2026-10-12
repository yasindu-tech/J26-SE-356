"""Tests for models/mri/src/t1_features.py, on tiny synthetic images (no real scans).

Each guard is shown to fail on bad input as well as pass on good input.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import t1_features as f
from scipy.io import savemat

SHAPE = (30, 30, 30)
AFFINE = np.diag([1.0, 1.0, 1.0, 1.0])


def save(path: Path, data: np.ndarray, affine: np.ndarray = AFFINE) -> None:
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine), str(path))


def write_atlas(tmp: Path, shape=SHAPE, drop: str | None = None) -> tuple[Path, Path]:
    """Maxprob image + XML with the 8 regions as 4x4x4 cubes, label value = index + 1."""
    labels = [name for name in f.REGIONS.values() if name != drop]
    data = np.zeros(shape)
    xml = ["<atlas><data>"]
    for i, name in enumerate(labels):
        x = 2 + (i % 4) * 7
        z = 2 + (i // 4) * 7
        data[x : x + 4, 10:14, z : z + 4] = i + 1
        xml.append(f'<label index="{i}" x="0" y="0" z="0">{name}</label>')
    xml.append("</data></atlas>")
    save(tmp / "atlas.nii.gz", data)
    (tmp / "atlas.xml").write_text("".join(xml))
    return tmp / "atlas.nii.gz", tmp / "atlas.xml"


def write_subject(tmp: Path, sid: str = "S0001", scale: float = 1.0) -> Path:
    d = tmp / sid
    (d / "transforms").mkdir(parents=True)
    seg = np.full(SHAPE, 3)  # white matter everywhere (enough voxels for normalising)
    seg[:, :, :2] = 1
    t1 = np.where(seg == 3, 800.0, 100.0) * scale
    t1[:, 10:14, :] = 400.0 * scale  # the region slab: half the white-matter brightness
    save(d / f"{sid}_mni.nii.gz", t1)
    save(d / f"{sid}_fast_seg.nii.gz", seg)
    save(d / f"{sid}_fast_pve_0.nii.gz", np.full(SHAPE, 0.1))
    save(d / f"{sid}_fast_pve_1.nii.gz", np.full(SHAPE, 0.6))
    save(d / f"{sid}_fast_pve_2.nii.gz", np.full(SHAPE, 0.3))
    mask = np.zeros((10, 10, 10))
    mask[:5] = 1  # 500 voxels of 2 mm^3 -> 4 mL
    save(d / f"{sid}_n4_brain_mask.nii.gz", mask, np.diag([2.0, 2.0, 2.0, 1.0]))
    params = np.r_[np.diag([1.1, 1.0, 0.9]).ravel(), [1.0, 2.0, 3.0]].reshape(12, 1)
    savemat(
        d / "transforms" / f"{sid}_fwd_affine.mat",
        {"AffineTransform_float_3_3": params, "fixed": np.zeros((3, 1))},
    )
    return d


@pytest.fixture
def atlas(tmp_path: Path) -> f.Atlas:
    return f.load_atlas(*write_atlas(tmp_path))


# --- feature list ---------------------------------------------------------------


def test_feature_count_and_names() -> None:
    names = f.feature_names()
    assert len(names) == 74 == len(set(names))
    assert "L_putamen_grey_frac" in names and "R_thalamus_t1_entropy" in names


# --- first-order statistics -------------------------------------------------------


def test_first_order_known_values() -> None:
    out = f.first_order(np.array([1.0, 2.0, 3.0, 4.0]), bin_width=1.0)
    assert out["mean"] == pytest.approx(2.5)
    assert out["median"] == pytest.approx(2.5)
    assert out["std"] == pytest.approx(np.std([1, 2, 3, 4]))
    assert out["skewness"] == pytest.approx(0.0)
    assert out["kurtosis"] == pytest.approx(1.64)  # Pearson kurtosis of 1..4
    assert out["entropy"] == pytest.approx(2.0)  # 4 equally full bins


def test_first_order_constant_region() -> None:
    out = f.first_order(np.full(20, 0.5))
    assert (out["std"], out["skewness"], out["kurtosis"], out["entropy"]) == (0, 0, 0, 0)


def test_first_order_empty_region_fails() -> None:
    with pytest.raises(f.FeatureError, match="no voxels"):
        f.first_order(np.array([]))


# --- brightness normalisation -----------------------------------------------------


def test_normalisation_removes_scanner_gain() -> None:
    seg = np.full(SHAPE, 3)
    t1 = np.full(SHAPE, 800.0)
    a = f.normalise_brightness(t1, seg)
    b = f.normalise_brightness(t1 * 2.4, seg)  # same tissue, scanner with 2.4x gain
    assert np.allclose(a, b) and np.allclose(a, 1.0)


def test_normalisation_fails_without_white_matter() -> None:
    with pytest.raises(f.FeatureError, match="white-matter voxels"):
        f.normalise_brightness(np.ones(SHAPE), np.ones(SHAPE, dtype=int))


# --- atlas ------------------------------------------------------------------------


def test_atlas_label_value_is_index_plus_one(tmp_path: Path) -> None:
    _, xml = write_atlas(tmp_path)
    values = f.atlas_label_values(xml)
    assert values["Left Caudate"] == 1  # index 0 in the XML


def test_atlas_missing_region_fails(tmp_path: Path) -> None:
    with pytest.raises(f.FeatureError, match="Right Pallidum"):
        f.load_atlas(*write_atlas(tmp_path, drop="Right Pallidum"))


def test_atlas_missing_file_fails(tmp_path: Path) -> None:
    with pytest.raises(f.FeatureError, match="not found"):
        f.load_atlas(tmp_path / "nope.nii.gz", tmp_path / "nope.xml")


def test_grid_mismatch_fails(atlas: f.Atlas) -> None:
    with pytest.raises(f.FeatureError, match="shape"):
        f.check_same_grid(AFFINE, (31, 30, 30), atlas)
    with pytest.raises(f.FeatureError, match="voxel-to-world"):
        f.check_same_grid(np.diag([2.0, 1, 1, 1]), SHAPE, atlas)


# --- whole person -----------------------------------------------------------------


def test_extract_subject_values(tmp_path: Path, atlas: f.Atlas) -> None:
    r = f.extract_subject(write_subject(tmp_path), "S0001", atlas)
    assert r.status == "ok", r.reason
    feats = r.features
    assert set(feats) == set(f.feature_names())
    assert feats["L_putamen_grey_frac"] == pytest.approx(0.6)
    assert feats["L_putamen_t1_mean"] == pytest.approx(0.5)  # 400 / 800
    assert feats["brain_volume_ml"] == pytest.approx(4.0)
    assert feats["affine_det"] == pytest.approx(1.1 * 1.0 * 0.9)


def test_scanner_gain_does_not_change_features(tmp_path: Path, atlas: f.Atlas) -> None:
    a = f.extract_subject(write_subject(tmp_path / "a"), "S0001", atlas).features
    b = f.extract_subject(write_subject(tmp_path / "b", scale=2.4), "S0001", atlas).features
    assert a == pytest.approx(b)


def test_missing_output_gives_failed_result_not_exception(tmp_path: Path, atlas: f.Atlas) -> None:
    d = write_subject(tmp_path)
    (d / "S0001_fast_pve_1.nii.gz").unlink()
    r = f.extract_subject(d, "S0001", atlas)
    assert r.status == "failed" and "pve_1" in r.reason and r.features == {}


def test_bad_affine_file_fails(tmp_path: Path) -> None:
    savemat(tmp_path / "x.mat", {"other": np.zeros((3, 1))})
    with pytest.raises(f.FeatureError, match="no affine"):
        f.affine_determinant(tmp_path / "x.mat")


# --- table ------------------------------------------------------------------------


def test_table_uses_pseudonyms_and_no_label(tmp_path: Path, atlas: f.Atlas) -> None:
    r = f.extract_subject(write_subject(tmp_path), "S0001", atlas)
    meta = {
        f.pseudonym("S0001"): {"scanner_model": "Prisma", "manufacturer": "SIEMENS", "site_key": ""}
    }
    table = f.build_table([r], meta)
    assert list(table.columns[:4]) == list(f.META_COLUMNS)
    assert table.loc[0, "subject"] == f.pseudonym("S0001")
    assert "S0001" not in table.to_csv()
    assert not {"label", "diagnosis", "group", "cohort"} & set(table.columns)


def test_table_rejects_duplicate_person(tmp_path: Path, atlas: f.Atlas) -> None:
    r = f.extract_subject(write_subject(tmp_path), "S0001", atlas)
    with pytest.raises(f.FeatureError, match="twice"):
        f.build_table([r, r], {})


def test_blank_site_key_stays_blank(tmp_path: Path) -> None:
    conv = tmp_path / "conv.csv"
    pd.DataFrame(
        {
            "subject_id": ["1", "2"],
            "status": ["ok", "ok"],
            "kind": ["t1", "t1"],
            "model": ["Prisma", "Verio"],
            "manufacturer": ["S", "S"],
            "site_key": ["32", None],
        }
    ).to_csv(conv, index=False)
    meta = f.scanner_metadata(conv)
    assert meta[f.pseudonym("2")]["site_key"] == ""
    assert meta[f.pseudonym("1")]["site_key"] == "32"
