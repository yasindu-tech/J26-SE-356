"""Tests for models/mri/src/input_cleanup.py. No real data; everything lives in tmp_path.

The cleanup must delete only what it is meant to and refuse anything unsafe.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from input_cleanup import (
    SeriesRef,
    UnsafeDeleteError,
    delete_inputs,
)

SID = "S0001"


def make_dicom(raw: Path, subject: str, series: str, image_id: str, n: int = 3) -> Path:
    d = raw / subject / series / "2020-01-01_00_00_00.0" / f"I{image_id}"
    d.mkdir(parents=True)
    for i in range(n):
        (d / f"{i}.dcm").write_bytes(b"x" * 100)
    return d


def make_nifti(root: Path, subject: str, image_id: str, extra: tuple[str, ...] = ()) -> Path:
    folder = root / subject
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / f"{subject}_I{image_id}"
    for suffix in (".nii.gz", ".bval", ".bvec", ".json", *extra):
        stem.with_name(stem.name + suffix).write_bytes(b"y" * 50)
    return stem.with_name(stem.name + ".nii.gz")


@pytest.fixture
def env(tmp_path: Path):
    raw = tmp_path / "data" / "raw"
    nifti = tmp_path / "data" / "processed" / "nifti"
    raw.mkdir(parents=True)
    nifti.mkdir(parents=True)
    return raw, nifti


def test_deletes_converted_series_and_keeps_everything_else(env) -> None:
    raw, nifti = env
    make_dicom(raw / "set3", SID, "Ax_DTI", "111")
    unconverted = make_dicom(raw / "set3", SID, "Other_series", "222")
    other_person = make_dicom(raw / "set3", "S0002", "Ax_DTI", "333")
    n = make_nifti(nifti, SID, "111")
    t1 = nifti / SID / f"{SID}_I999.nii.gz"  # the same person's T1: different name
    t1.write_bytes(b"t1")

    report = delete_inputs(
        SID, [SeriesRef("111", n)], [raw / "set3"], raw_root=raw, nifti_root=nifti
    )

    assert not (raw / "set3" / SID / "Ax_DTI").exists()  # emptied folders are tidied up
    assert unconverted.exists() and other_person.exists()
    assert not n.exists() and not n.with_suffix("").with_suffix(".bval").exists()
    assert t1.exists()  # T1 untouched
    assert report.bytes_freed == 3 * 100 + 4 * 50
    assert all(SID not in name for name in report.removed)


def test_dry_run_deletes_nothing_but_reports_size(env) -> None:
    raw, nifti = env
    d = make_dicom(raw / "set3", SID, "Ax_DTI", "111")
    n = make_nifti(nifti, SID, "111")
    report = delete_inputs(
        SID, [SeriesRef("111", n)], [raw / "set3"], dry_run=True, raw_root=raw, nifti_root=nifti
    )
    assert d.exists() and n.exists() and report.bytes_freed == 3 * 100 + 4 * 50


def test_empty_subject_folder_is_removed_but_raw_root_stays(env) -> None:
    raw, nifti = env
    make_dicom(raw / "set3", SID, "Ax_DTI", "111")
    n = make_nifti(nifti, SID, "111")
    delete_inputs(SID, [SeriesRef("111", n)], [raw / "set3"], raw_root=raw, nifti_root=nifti)
    assert not (raw / "set3" / SID).exists() and (raw / "set3").exists()
    assert not (nifti / SID).exists()  # nothing left in the person's NIfTI folder


def test_nifti_folder_with_other_files_is_kept(env) -> None:
    raw, nifti = env
    n = make_nifti(nifti, SID, "111")
    (nifti / SID / "notes.txt").write_text("keep")
    delete_inputs(SID, [SeriesRef("111", n)], [], raw_root=raw, nifti_root=nifti)
    assert (nifti / SID / "notes.txt").exists()


# --- the guards: each must refuse and delete nothing ---------------------------------------


def test_raw_dir_outside_data_raw_is_refused(env, tmp_path: Path) -> None:
    raw, nifti = env
    elsewhere = tmp_path / "Documents"
    d = make_dicom(elsewhere, SID, "Ax_DTI", "111")
    n = make_nifti(nifti, SID, "111")
    with pytest.raises(UnsafeDeleteError, match="not inside"):
        delete_inputs(SID, [SeriesRef("111", n)], [elsewhere], raw_root=raw, nifti_root=nifti)
    assert d.exists() and n.exists()  # nothing at all was deleted


def test_nifti_outside_nifti_root_is_refused_before_any_delete(env, tmp_path: Path) -> None:
    raw, nifti = env
    d = make_dicom(raw / "set3", SID, "Ax_DTI", "111")
    stray = make_nifti(tmp_path / "somewhere_else", SID, "111")
    with pytest.raises(UnsafeDeleteError, match="outside"):
        delete_inputs(
            SID, [SeriesRef("111", stray)], [raw / "set3"], raw_root=raw, nifti_root=nifti
        )
    assert d.exists() and stray.exists()  # the DICOM was not deleted either


def test_symlinked_series_folder_is_not_followed(env, tmp_path: Path) -> None:
    raw, nifti = env
    target = tmp_path / "precious"
    target.mkdir()
    (target / "keep.dcm").write_bytes(b"z")
    link_parent = raw / "set3" / SID / "Ax_DTI" / "2020-01-01_00_00_00.0"
    link_parent.mkdir(parents=True)
    (link_parent / "I111").symlink_to(target)
    with pytest.raises(UnsafeDeleteError, match="symlink"):
        delete_inputs(
            SID,
            [SeriesRef("111", nifti / "x.nii.gz")],
            [raw / "set3"],
            raw_root=raw,
            nifti_root=nifti,
        )
    assert (target / "keep.dcm").exists()


def test_other_image_ids_are_never_matched(env) -> None:
    raw, nifti = env
    keep = make_dicom(raw / "set3", SID, "Ax_DTI", "1111")  # "111" must not match "1111"
    delete_inputs(
        SID, [SeriesRef("111", nifti / "x.nii.gz")], [raw / "set3"], raw_root=raw, nifti_root=nifti
    )
    assert keep.exists()
