"""Tests for models/mri/src/convert_dicom.py.

Each check is proved to FAIL on bad input as well as pass on good input
No real PPMI data is used: tiny synthetic NIfTI files
are written to a temp folder.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "convert_dicom.py"
spec = importlib.util.spec_from_file_location("convert_dicom", SRC)
assert spec and spec.loader
convert_dicom = importlib.util.module_from_spec(spec)
sys.modules["convert_dicom"] = convert_dicom
spec.loader.exec_module(convert_dicom)


def write_nifti(path: Path, shape: tuple[int, ...], voxel: float = 1.0) -> None:
    img = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.diag([voxel, voxel, voxel, 1]))
    nib.save(img, path)


def test_t1_ok(tmp_path: Path) -> None:
    write_nifti(tmp_path / "s_I1.nii.gz", (8, 8, 8))
    kind, n_vol, shape, voxel = convert_dicom.check_output(tmp_path, "s_I1")
    assert (kind, n_vol, shape) == ("t1", 1, (8, 8, 8))
    assert voxel == (1.0, 1.0, 1.0)


def test_no_nifti_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected 1 NIfTI"):
        convert_dicom.check_output(tmp_path, "s_I1")


def test_two_niftis_fail(tmp_path: Path) -> None:
    write_nifti(tmp_path / "s_I1.nii.gz", (8, 8, 8))
    write_nifti(tmp_path / "s_I1_e2.nii.gz", (8, 8, 8))
    with pytest.raises(ValueError, match="expected 1 NIfTI"):
        convert_dicom.check_output(tmp_path, "s_I1")


def test_t1_must_be_3d(tmp_path: Path) -> None:
    write_nifti(tmp_path / "s_I1.nii.gz", (8, 8, 8, 3))
    with pytest.raises(ValueError, match="3D"):
        convert_dicom.check_output(tmp_path, "s_I1")


def test_t1_odd_voxel_size_fails(tmp_path: Path) -> None:
    write_nifti(tmp_path / "s_I1.nii.gz", (8, 8, 8), voxel=5.0)
    with pytest.raises(ValueError, match="voxel"):
        convert_dicom.check_output(tmp_path, "s_I1")


def test_dti_ok(tmp_path: Path) -> None:
    write_nifti(tmp_path / "s_I2.nii.gz", (8, 8, 8, 4))
    (tmp_path / "s_I2.bval").write_text("0 1000 1000 1000\n")
    (tmp_path / "s_I2.bvec").write_text("0 1 0 0\n0 0 1 0\n0 0 0 1\n")
    kind, n_vol, _, _ = convert_dicom.check_output(tmp_path, "s_I2")
    assert (kind, n_vol) == ("dti", 4)


def test_dti_volume_count_mismatch_fails(tmp_path: Path) -> None:
    write_nifti(tmp_path / "s_I2.nii.gz", (8, 8, 8, 4))
    (tmp_path / "s_I2.bval").write_text("0 1000 1000\n")  # 3 values, 4 volumes
    (tmp_path / "s_I2.bvec").write_text("0 1 0\n0 0 1\n0 0 0\n")
    with pytest.raises(ValueError, match="bval"):
        convert_dicom.check_output(tmp_path, "s_I2")


def test_dti_missing_bvec_fails(tmp_path: Path) -> None:
    write_nifti(tmp_path / "s_I2.nii.gz", (8, 8, 8, 2))
    (tmp_path / "s_I2.bval").write_text("0 1000\n")
    with pytest.raises(ValueError, match="bvec"):
        convert_dicom.check_output(tmp_path, "s_I2")


def test_parse_path() -> None:
    p = Path("/x/PPMI/3365/MPRAGE_T1_SAG/2011-08-25_09_19_02.0/I269488")
    assert convert_dicom.parse_path(p) == (
        "3365",
        "269488",
        "MPRAGE_T1_SAG",
        "2011-08-25_09_19_02.0",
    )


def test_site_key_missing_stays_missing(tmp_path: Path) -> None:
    (tmp_path / "with_site.xml").write_text("<r><siteKey>096</siteKey><imageUID>111</imageUID></r>")
    (tmp_path / "no_site.xml").write_text("<r><imageUID>222</imageUID></r>")
    (tmp_path / "broken.xml").write_text("<r><unclosed>")
    keys = convert_dicom.load_site_keys([tmp_path])
    assert keys == {"111": "096"}  # 222 is absent, not defaulted


def test_pseudonym_hides_id() -> None:
    assert "3365" not in convert_dicom.pseudonym("3365")
    assert convert_dicom.pseudonym("3365") == convert_dicom.pseudonym("3365")


def test_equalised_copy_fails_loudly(tmp_path: Path) -> None:
    write_nifti(tmp_path / "s_I1.nii.gz", (8, 8, 8))
    write_nifti(tmp_path / "s_I1_Eq_1.nii.gz", (8, 8, 8))
    with pytest.raises(ValueError, match="unequal slice spacing"):
        convert_dicom.check_output(tmp_path, "s_I1")


def test_field_strength_one_spelling() -> None:
    assert convert_dicom.field_strength("3") == convert_dicom.field_strength("3.0") == "3"
    assert convert_dicom.field_strength("") == ""


def test_scanner_adc_map_is_removed_and_dti_passes(tmp_path: Path) -> None:
    write_nifti(tmp_path / "s_I3.nii.gz", (8, 8, 8, 2))
    write_nifti(tmp_path / "s_I3_ADC.nii.gz", (8, 8, 8))
    (tmp_path / "s_I3.bval").write_text("0 1000\n")
    (tmp_path / "s_I3.bvec").write_text("0 1\n0 0\n0 0\n")
    removed = convert_dicom.remove_scanner_derived(tmp_path, "s_I3")
    assert removed == ["s_I3_ADC.nii.gz"]
    assert convert_dicom.check_output(tmp_path, "s_I3")[0] == "dti"


def test_raw_scan_is_not_mistaken_for_derived(tmp_path: Path) -> None:
    write_nifti(tmp_path / "s_I4.nii.gz", (8, 8, 8))
    assert convert_dicom.remove_scanner_derived(tmp_path, "s_I4") == []
    assert (tmp_path / "s_I4.nii.gz").exists()


def test_b0_only_diffusion_scan_is_labelled(tmp_path: Path) -> None:
    write_nifti(tmp_path / "s_I5.nii.gz", (8, 8, 8), voxel=2.0)
    kind = convert_dicom.check_output(tmp_path, "s_I5", series_name="DTI_revB0_AP")[0]
    assert kind == "dti_b0_only"


def test_non_diffusion_4d_without_bval_still_fails(tmp_path: Path) -> None:
    write_nifti(tmp_path / "s_I6.nii.gz", (8, 8, 8, 3))
    with pytest.raises(ValueError, match="3D"):
        convert_dicom.check_output(tmp_path, "s_I6", series_name="MPRAGE")
