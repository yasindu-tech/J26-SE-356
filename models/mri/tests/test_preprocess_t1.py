"""Tests for models/mri/src/preprocess_t1.py.

Only the checks and helpers are tested here, with tiny synthetic NIfTI files.
The heavy steps (N4, SynthStrip, ANTs, FAST) need real tools and are checked by
eye on real scans. Each check is shown to FAIL on bad input as well as pass.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "preprocess_t1.py"
spec = importlib.util.spec_from_file_location("preprocess_t1", SRC)
assert spec and spec.loader
p = importlib.util.module_from_spec(spec)
sys.modules["preprocess_t1"] = p
spec.loader.exec_module(p)


def write(path: Path, data: np.ndarray, voxel: float = 1.0) -> None:
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.diag([voxel, voxel, voxel, 1])), path)


# --- mask volume ---------------------------------------------------------------


def test_mask_volume_ok() -> None:
    p.check_mask_volume(1500.0)


@pytest.mark.parametrize("ml", [0.0, 200.0, 5000.0])
def test_mask_volume_out_of_range_fails(ml: float) -> None:
    with pytest.raises(p.StepError, match="brain mask"):
        p.check_mask_volume(ml)


def test_mask_volume_ml_counts_voxels(tmp_path: Path) -> None:
    data = np.zeros((10, 10, 10))
    data[:5] = 1  # 500 voxels of 1 mm^3 = 0.5 mL
    write(tmp_path / "m.nii.gz", data)
    assert p.mask_volume_ml(tmp_path / "m.nii.gz") == pytest.approx(0.5)


# --- template match ------------------------------------------------------------


def test_template_correlation_ok() -> None:
    p.check_template_correlation(0.9)


@pytest.mark.parametrize("corr", [0.1, -0.5, float("nan")])
def test_template_correlation_bad_fails(corr: float) -> None:
    with pytest.raises(p.StepError, match="template"):
        p.check_template_correlation(corr)


def test_template_correlation_identical_is_one(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    data = rng.random((6, 6, 6)) + 1
    write(tmp_path / "a.nii.gz", data)
    assert p.template_correlation(tmp_path / "a.nii.gz", tmp_path / "a.nii.gz") == pytest.approx(
        1.0
    )


def test_template_correlation_shape_mismatch_fails(tmp_path: Path) -> None:
    write(tmp_path / "a.nii.gz", np.ones((6, 6, 6)))
    write(tmp_path / "b.nii.gz", np.ones((5, 6, 6)))
    with pytest.raises(p.StepError, match="shape"):
        p.template_correlation(tmp_path / "a.nii.gz", tmp_path / "b.nii.gz")


# --- tissue volumes ------------------------------------------------------------


def test_tissue_volumes(tmp_path: Path) -> None:
    seg = np.zeros((10, 10, 10))
    seg[:1] = 1  # 100 voxels CSF
    seg[1:3] = 2  # 200 voxels grey
    seg[3:6] = 3  # 300 voxels white
    write(tmp_path / "s.nii.gz", seg)
    assert p.tissue_volumes_ml(tmp_path / "s.nii.gz") == pytest.approx((0.1, 0.2, 0.3))


# --- tools and failure handling --------------------------------------------------


def test_missing_template_is_reported(tmp_path: Path) -> None:
    cfg = p.T1Config(template=tmp_path / "nope.nii.gz", skullstrip="bet")
    with pytest.raises(p.StepError, match="template not found"):
        p.check_tools(cfg)


def test_unknown_skullstrip_method_rejected(tmp_path: Path) -> None:
    template = tmp_path / "t.nii.gz"
    template.write_bytes(b"x")
    cfg = p.T1Config(template=template, skullstrip="magic", fast_command="python3")
    with pytest.raises(p.StepError, match="unknown skull-strip"):
        p.check_tools(cfg)


def test_missing_synthstrip_script_is_reported(tmp_path: Path) -> None:
    template = tmp_path / "t.nii.gz"
    template.write_bytes(b"x")
    cfg = p.T1Config(
        template=template, synthstrip_script=tmp_path / "missing", fast_command="python3"
    )
    with pytest.raises(p.StepError, match="SynthStrip wrapper"):
        p.check_tools(cfg)


def test_bad_scan_gives_failed_result_not_exception(tmp_path: Path) -> None:
    cfg = p.T1Config(template=tmp_path / "nope.nii.gz")
    result = p.preprocess_t1(tmp_path / "missing.nii.gz", "123", tmp_path / "out", cfg)
    assert result.status == "failed"
    assert result.reason  # a reason is always given
    assert result.mni_brain == ""  # no output is claimed for a failed scan


# --- input selection -------------------------------------------------------------


def test_t1_inputs_skips_failed_and_dti(tmp_path: Path) -> None:
    table = tmp_path / "conv.csv"
    pd.DataFrame(
        {
            "subject_id": ["1", "2", "3"],
            "status": ["ok", "failed", "ok"],
            "kind": ["t1", "t1", "dti"],
            "nifti_path": ["a.nii.gz", "b.nii.gz", "c.nii.gz"],
        }
    ).to_csv(table, index=False)
    assert [s for s, _ in p.t1_inputs(table)] == ["1"]


def test_pseudonym_hides_id() -> None:
    assert "3365" not in p.pseudonym("3365")
