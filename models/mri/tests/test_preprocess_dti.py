"""Tests for models/mri/src/preprocess_dti.py.

The choices and checks are tested with tiny synthetic files. eddy, BET and ANTs
need real tools and are checked by eye on real scans. Each check is shown to FAIL
on bad input as well as pass on good input.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "preprocess_dti.py"
spec = importlib.util.spec_from_file_location("preprocess_dti", SRC)
assert spec and spec.loader
d = importlib.util.module_from_spec(spec)
sys.modules["preprocess_dti"] = d
spec.loader.exec_module(d)


def write_dwi(folder: Path, stem: str, bvals: list[float]) -> tuple[Path, Path, Path]:
    n = len(bvals)
    data = np.arange(4 * 4 * 4 * n, dtype=np.float32).reshape(4, 4, 4, n)
    nifti = folder / f"{stem}.nii.gz"
    nib.save(nib.Nifti1Image(data, np.diag([2.0, 2.0, 2.0, 1.0])), nifti)
    bval, bvec = folder / f"{stem}.bval", folder / f"{stem}.bvec"
    bval.write_text(" ".join(str(b) for b in bvals) + "\n")
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(3, n))
    vecs /= np.linalg.norm(vecs, axis=0)
    np.savetxt(bvec, vecs)
    return nifti, bval, bvec


# --- shell selection -----------------------------------------------------------


def test_select_shell_keeps_b0_and_b1000_only() -> None:
    bvals = np.array([0, 700, 1000, 1000, 1000, 1000, 1000, 1000, 2000, 5])
    assert d.select_shell(bvals).tolist() == [0, 2, 3, 4, 5, 6, 7, 9]


def test_select_shell_without_b0_fails() -> None:
    with pytest.raises(d.StepError, match="no b0"):
        d.select_shell(np.array([1000.0] * 10))


def test_select_shell_too_few_directions_fails() -> None:
    with pytest.raises(d.StepError, match="need 6"):
        d.select_shell(np.array([0, 1000, 1000, 1000, 2000, 2000, 2000]))


def test_write_selected_drops_other_shells(tmp_path: Path) -> None:
    nifti, bval, bvec = write_dwi(tmp_path, "s", [0, 2000] + [1000] * 6)
    dwi, bv, bc, kept = d.write_selected(nifti, bval, bvec, tmp_path, "out")
    assert kept.tolist() == [0] + [1000] * 6
    assert nib.load(str(dwi)).shape[-1] == 7
    assert np.loadtxt(bc).shape == (3, 7)


def test_write_selected_mismatched_files_fail(tmp_path: Path) -> None:
    nifti, bval, bvec = write_dwi(tmp_path, "s", [0] + [1000] * 6)
    bval.write_text("0 1000 1000\n")  # 3 values for 7 volumes
    with pytest.raises(d.StepError, match="disagree"):
        d.write_selected(nifti, bval, bvec, tmp_path, "out")


# --- choosing one series per person ------------------------------------------------


def test_choose_series_lowest_usable_image_id() -> None:
    rows = pd.DataFrame({"image_id": ["300", "100", "200"], "n_shell_dirs": [32, 0, 32]})
    assert d.choose_series(rows)["image_id"] == "200"  # 100 has no b~1000 shell


def test_choose_series_none_usable_fails() -> None:
    rows = pd.DataFrame({"image_id": ["1", "2"], "n_shell_dirs": [0, 3]})
    with pytest.raises(d.StepError, match="no DTI series"):
        d.choose_series(rows)


def test_choose_series_sorts_numerically_not_as_text() -> None:
    rows = pd.DataFrame({"image_id": ["900", "1000"], "n_shell_dirs": [32, 32]})
    assert d.choose_series(rows)["image_id"] == "900"


# --- phase encoding and readout ------------------------------------------------------


@pytest.mark.parametrize(
    ("sidecar", "name", "axis", "source"),
    [
        ({"PhaseEncodingDirection": "j-"}, "DTI_gated", "j-", "json"),
        ({}, "DTI_RL", "i", "series_name"),
        ({}, "AX_DTI___L_-_R", "i", "series_name"),
        ({}, "DTI_B1000_64dir_PA", "j", "series_name"),
        ({}, "DTI_gated", "j", "assumed"),
    ],
)
def test_phase_encoding_sources(sidecar: dict, name: str, axis: str, source: str) -> None:
    got_axis, vec, got_source = d.phase_encoding(sidecar, name)
    assert (got_axis, got_source) == (axis, source)
    assert sum(abs(v) for v in vec) == 1


def test_readout_from_json_or_fallback() -> None:
    assert d.readout_time({"TotalReadoutTime": 0.04}) == (0.04, "json")
    assert d.readout_time({}) == (d.FALLBACK_READOUT_S, "fallback")
    assert d.readout_time({"TotalReadoutTime": 0}) == (d.FALLBACK_READOUT_S, "fallback")


def test_eddy_inputs(tmp_path: Path) -> None:
    acqp, index = d.write_eddy_inputs(tmp_path, "s", [1, 0, 0], 0.05, 4)
    assert acqp.read_text().split() == ["1", "0", "0", "0.05"]
    assert index.read_text().split() == ["1", "1", "1", "1"]


def test_eddy_motion_parsed(tmp_path: Path) -> None:
    (tmp_path / "s_eddy.eddy_movement_rms").write_text("0.5 0.0\n1.0 0.2\n1.5 0.4\n")
    abs_mm, rel_mm = d.eddy_motion(tmp_path / "s_eddy")
    assert abs_mm == pytest.approx(1.0)
    assert rel_mm == pytest.approx(0.3)


def test_eddy_motion_missing_file_fails(tmp_path: Path) -> None:
    with pytest.raises(d.StepError, match="movement"):
        d.eddy_motion(tmp_path / "s_eddy")


# --- checks ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("check", "good", "bad"),
    [
        (d.check_mask_volume, 1500.0, 100.0),
        (d.check_median_fa, 0.25, 0.9),
        (d.check_motion, 0.8, 5.0),
        (d.check_template_correlation, 0.8, 0.1),
    ],
)
def test_checks_pass_good_and_fail_bad(check, good: float, bad: float) -> None:
    check(good)
    with pytest.raises(d.StepError):
        check(bad)


@pytest.mark.parametrize(
    "check", [d.check_mask_volume, d.check_median_fa, d.check_motion, d.check_template_correlation]
)
def test_checks_fail_on_nan(check) -> None:
    with pytest.raises(d.StepError):
        check(float("nan"))


def test_missing_template_reported(tmp_path: Path) -> None:
    with pytest.raises(d.StepError, match="FA template"):
        d.check_tools(d.DTIConfig(fa_template=tmp_path / "nope.nii.gz"))


def test_bad_scan_gives_failed_result(tmp_path: Path) -> None:
    cfg = d.DTIConfig(fa_template=tmp_path / "nope.nii.gz")
    res = d.preprocess_dti(tmp_path / "x.nii.gz", "123", "1", "DTI", tmp_path / "out", cfg)
    assert res.status == "failed" and res.reason and res.fa_mni == ""


# --- input selection --------------------------------------------------------------------


def test_dti_inputs_chooses_one_per_person_and_reports_unusable(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(d, "REPO_ROOT", tmp_path)
    write_dwi(tmp_path, "a_I10", [0] + [1000] * 6)
    write_dwi(tmp_path, "a_I11", [0] + [1000] * 6)
    write_dwi(tmp_path, "b_I20", [0, 2000, 2000])
    table = tmp_path / "conv.csv"
    pd.DataFrame(
        {
            "subject_id": ["a", "a", "b", "c"],
            "image_id": ["11", "10", "20", "30"],
            "series_name": ["DTI_RL", "DTI_LR", "DTI_B2000", "MPRAGE"],
            "status": ["ok", "ok", "ok", "ok"],
            "kind": ["dti", "dti", "dti", "t1"],
            "nifti_path": ["a_I11.nii.gz", "a_I10.nii.gz", "b_I20.nii.gz", "c.nii.gz"],
        }
    ).to_csv(table, index=False)
    chosen, skipped = d.dti_inputs(table)
    assert [(c[0], c[1]) for c in chosen] == [("a", "10")]
    assert [s[0] for s in skipped] == ["b"]  # c is a T1, not a DTI person


def test_eddy_arguments_threads_and_repol() -> None:
    p = Path("x")
    fast = d.DTIConfig(eddy_threads=8, repol=False)
    args = d.eddy_arguments(p, p, p, p, p, p, p, fast)
    assert "--nthr=8" in args and "--repol" not in args
    default = d.eddy_arguments(p, p, p, p, p, p, p, d.DTIConfig())
    assert "--repol" in default and not any(a.startswith("--nthr") for a in default)


def test_threads_requested_but_unsupported_is_reported(tmp_path: Path, monkeypatch) -> None:
    template = tmp_path / "t.nii.gz"
    template.write_bytes(b"x")
    monkeypatch.setattr(d.shutil, "which", lambda cmd: "/bin/" + cmd)
    monkeypatch.setattr(d, "eddy_supports_threads", lambda cmd: False)
    with pytest.raises(d.StepError, match="no --nthr"):
        d.check_tools(d.DTIConfig(fa_template=template, eddy_threads=4))


def test_streaming_run_passes_lines_and_reports_failure() -> None:
    seen: list[str] = []
    d._run_streaming(["sh", "-c", "echo one; echo two"], "demo", seen.append)
    assert [x.strip() for x in seen] == ["one", "two"]
    with pytest.raises(d.StepError, match="boom"):
        d._run_streaming(["sh", "-c", "echo boom; exit 3"], "demo", lambda _: None)


def _touch_outputs(out: Path, stem: str, suffixes: tuple[str, ...]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for sfx in suffixes:
        (out / f"{stem}{sfx}").write_bytes(b"x" * 10)
    (out / "transforms").mkdir(exist_ok=True)
    (out / "transforms" / f"{stem}_fwd_affine.mat").write_bytes(b"x")
    (out / "transforms" / f"{stem}_fwd_warp.nii.gz").write_bytes(b"x" * 50)


def test_cleanup_keeps_results_and_deletes_intermediates(tmp_path: Path) -> None:
    _touch_outputs(tmp_path, "s", d.KEEP_SUFFIXES)
    for junk in (
        "_dwi.nii.gz",
        "_eddy.nii.gz",
        "_eddy.eddy_outlier_free_data.nii.gz",
        "_b0mean.nii.gz",
    ):
        (tmp_path / f"s{junk}").write_bytes(b"x" * 100)
    freed = d.remove_intermediates(tmp_path, "s")
    assert freed == 450  # 4 in-between files + the warp field
    left = sorted(f.name for f in tmp_path.iterdir() if f.is_file())
    assert left == sorted(f"s{sfx}" for sfx in d.KEEP_SUFFIXES)
    assert (tmp_path / "transforms" / "s_fwd_affine.mat").exists()  # transforms untouched
    assert d.final_outputs_exist(tmp_path, "s")  # a rerun reuses these, no new eddy


def test_final_outputs_missing_means_not_finished(tmp_path: Path) -> None:
    _touch_outputs(tmp_path, "s", d.FINAL_SUFFIXES)
    (tmp_path / "s_FA_mni.nii.gz").unlink()
    assert not d.final_outputs_exist(tmp_path, "s")


def test_median_in_mask(tmp_path: Path) -> None:
    fa = np.zeros((4, 4, 4), dtype=np.float32)
    fa[:2] = 0.3
    fa[2:] = 0.9
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[:2] = 1  # only the 0.3 half is brain
    nib.save(nib.Nifti1Image(fa, np.eye(4)), tmp_path / "fa.nii.gz")
    nib.save(nib.Nifti1Image(mask, np.eye(4)), tmp_path / "m.nii.gz")
    assert d.median_in_mask(tmp_path / "fa.nii.gz", tmp_path / "m.nii.gz") == pytest.approx(0.3)
