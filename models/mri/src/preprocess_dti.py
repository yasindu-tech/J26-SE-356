"""MRI-15: DTI preprocessing pipeline.

One diffusion scan in, FA and MD maps in standard space out:

1. pick one diffusion series per person, by the same rule for everyone, and keep
   only its b0 volumes and its b~1000 shell (so every person is measured alike);
2. brain mask from the mean b0 (FSL BET);
3. motion and eddy-current correction with FSL eddy, which also rotates the
   gradient directions to match;
4. tensor fit with DIPY -> FA (fractional anisotropy) and MD (mean diffusivity);
5. registration of FA to the FMRIB58_FA template with ANTs; MD follows the same
   transform.

Design choice for PP1: NO susceptibility-distortion correction (topup) for anyone.
Only some sites acquired the reverse phase-encode data topup needs, so correcting
only those people would make processing differ by site, which is the very effect
harmonisation tries to remove. This is a stated limitation, not an oversight.

Nothing here learns from the dataset (templates are fixed external files), so it is
safe before any train/test split. A failed step or check gives ``status="failed"``
with the reason, never zeros.

Needs: FSL (bet, eddy), Python packages antspyx, dipy, nibabel.

Usage (from the repo root, venv active):
    python models/mri/src/preprocess_dti.py --subject 100890
    python models/mri/src/preprocess_dti.py --limit 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from progress import NullRecorder, RunRecorder

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TABLE_IN = REPO_ROOT / "data" / "interim" / "mri_conversion_table.csv"
DEFAULT_TABLE_OUT = REPO_ROOT / "data" / "interim" / "dti_preprocess_table.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "dti"

B0_MAX = 50.0  # volumes with b below this are treated as b0
SHELL_TARGET = 1000.0
SHELL_TOLERANCE = 200.0  # keep volumes with b in 800-1200
MIN_DIRECTIONS = 6  # the tensor has 6 unknowns, so fewer directions cannot fit it
# Used when the scan does not record its readout time (Philips often doesn't).
# Without topup this only scales eddy's internal field model; recorded per person.
FALLBACK_READOUT_S = 0.05

# PROVISIONAL limits, set before seeing the 38 scans. Check them against the batch
# results and adjust, writing down why. They are not from a paper.
MASK_ML_RANGE = (800.0, 2400.0)  # BET mask on b0, includes CSF
MEDIAN_FA_RANGE = (0.10, 0.50)  # median FA inside the brain mask
MAX_MEAN_ABS_MOTION_MM = 3.0  # eddy's mean absolute displacement
MIN_TEMPLATE_CORRELATION = 0.5  # warped FA vs FMRIB58_FA, inside the template

log = logging.getLogger("preprocess_dti")


class StepError(Exception):
    """A pipeline step failed or a check did not pass. The message is the reason."""


def pseudonym(subject_id: str) -> str:
    """Short hash for log lines. Never log the raw PPMI subject ID."""
    return hashlib.sha256(subject_id.encode()).hexdigest()[:8]


def default_fa_template() -> Path:
    fsldir = Path(os.environ.get("FSLDIR", Path.home() / "fsl"))
    return fsldir / "data" / "standard" / "FMRIB58_FA_1mm.nii.gz"


@dataclass(frozen=True)
class DTIConfig:
    fa_template: Path = field(default_factory=default_fa_template)
    bet_command: str = "bet"
    eddy_command: str = "eddy"  # FSL picks the CPU build; "eddy_cpu" also works
    bet_fraction: float = 0.3  # BET strictness for b0 images (T1 used 0.4)
    keep_intermediates: bool = False  # False: delete big in-between files once a person passes
    eddy_threads: int = 1  # eddy --nthr; only newer FSL builds support it
    repol: bool = True  # eddy outlier replacement: better data, somewhat slower


@dataclass
class DTIResult:
    subject_id: str
    image_id: str
    input_nifti: str
    status: str = "failed"
    reason: str = ""
    b_value: float | None = None
    n_directions: int | None = None
    n_b0: int | None = None
    pe_axis: str = ""
    pe_source: str = ""  # "json", "series_name" or "assumed"
    readout_s: float | None = None
    readout_source: str = ""  # "json" or "fallback"
    brain_mask_ml: float | None = None
    mean_abs_motion_mm: float | None = None
    mean_rel_motion_mm: float | None = None
    median_fa: float | None = None
    template_correlation: float | None = None
    fa_mni: str = ""
    md_mni: str = ""
    transform_dir: str = ""
    qc_image: str = ""


# ---------------------------------------------------------------- choices (pure)


def choose_series(rows: pd.DataFrame) -> pd.Series:
    """One DTI series per person: the lowest image ID that has a usable b~1000 shell.

    ``rows`` are this person's conversion-table rows with kind == "dti" and a
    ``n_shell_dirs`` column. Raises StepError when none qualify.
    """
    usable = rows[rows["n_shell_dirs"] >= MIN_DIRECTIONS]
    if usable.empty:
        raise StepError(
            f"no DTI series with >= {MIN_DIRECTIONS} directions at b~{SHELL_TARGET:.0f}"
        )
    return usable.sort_values("image_id", key=lambda s: s.astype(int)).iloc[0]


def select_shell(bvals: np.ndarray) -> np.ndarray:
    """Indices of the b0 volumes plus the b~1000 shell. Raises StepError if unusable."""
    b0 = bvals < B0_MAX
    shell = np.abs(bvals - SHELL_TARGET) <= SHELL_TOLERANCE
    if not b0.any():
        raise StepError("no b0 volume in this series")
    if shell.sum() < MIN_DIRECTIONS:
        raise StepError(
            f"only {int(shell.sum())} volumes at b~{SHELL_TARGET:.0f}, need {MIN_DIRECTIONS}"
        )
    return np.flatnonzero(b0 | shell)


def count_shell_dirs(bval_file: Path) -> int:
    bvals = np.atleast_1d(np.loadtxt(bval_file))
    return int((np.abs(bvals - SHELL_TARGET) <= SHELL_TOLERANCE).sum())


_AXIS_FROM_JSON = {
    "i": ("i", 1),
    "i-": ("i", -1),
    "j": ("j", 1),
    "j-": ("j", -1),
    "k": ("k", 1),
    "k-": ("k", -1),
}


def phase_encoding(sidecar: dict, series_name: str) -> tuple[str, list[int], str]:
    """(axis label, acqparams vector, source). The JSON is trusted first, then the
    series name (LR/RL -> i axis, AP/PA -> j axis), and only then an assumption."""
    pe = sidecar.get("PhaseEncodingDirection")
    if pe in _AXIS_FROM_JSON:
        axis, sign = _AXIS_FROM_JSON[pe]
        vec = [0, 0, 0]
        vec["ijk".index(axis)] = sign
        return pe, vec, "json"
    name = series_name.upper()
    if re.search(r"(^|[^A-Z])(LR|RL|L_-_R|R_-_L)([^A-Z]|$)|FAT_SHIFT_[LR]", name):
        return "i", [1, 0, 0], "series_name"
    if re.search(r"(^|[^A-Z])(AP|PA|A_P)([^A-Z]|$)", name):
        return "j", [0, 1, 0], "series_name"
    return "j", [0, 1, 0], "assumed"  # most axial EPI is phase-encoded anterior-posterior


def readout_time(sidecar: dict) -> tuple[float, str]:
    value = sidecar.get("TotalReadoutTime")
    if isinstance(value, int | float) and value > 0:
        return float(value), "json"
    return FALLBACK_READOUT_S, "fallback"


# ---------------------------------------------------------------- checks (pure)


def _in_range(value: float, limits: tuple[float, float], what: str) -> None:
    if not (limits[0] <= value <= limits[1]):  # NaN fails too
        raise StepError(f"{what} is {value:.3g}, outside {limits[0]:g}-{limits[1]:g}")


def check_mask_volume(ml: float) -> None:
    _in_range(ml, MASK_ML_RANGE, "brain mask (mL)")


def check_median_fa(fa: float) -> None:
    _in_range(fa, MEDIAN_FA_RANGE, "median FA in brain")


def check_motion(mean_abs_mm: float) -> None:
    if not mean_abs_mm <= MAX_MEAN_ABS_MOTION_MM:
        raise StepError(
            f"head motion too large: mean {mean_abs_mm:.2f} mm > {MAX_MEAN_ABS_MOTION_MM} mm"
        )


def check_template_correlation(corr: float) -> None:
    if not corr >= MIN_TEMPLATE_CORRELATION:
        raise StepError(f"aligned FA matches template poorly (correlation {corr:.2f})")


def check_tools(config: DTIConfig) -> None:
    if not config.fa_template.exists():
        raise StepError(f"FA template not found: {config.fa_template}")
    for cmd in (config.bet_command, config.eddy_command):
        if shutil.which(cmd) is None:
            raise StepError(f"FSL '{cmd}' not found (is FSL installed and on PATH?)")
    if config.eddy_threads > 1 and not eddy_supports_threads(config.eddy_command):
        raise StepError(
            f"'{config.eddy_command}' has no --nthr option; run with --eddy-threads 1 and --jobs N"
        )


def eddy_supports_threads(eddy_command: str) -> bool:
    """True when this eddy build accepts --nthr (newer FSL versions)."""
    proc = subprocess.run([eddy_command, "--help"], capture_output=True, text=True, check=False)
    return "--nthr" in (proc.stdout + proc.stderr)


def eddy_arguments(
    dwi: Path,
    mask: Path,
    bval: Path,
    bvec: Path,
    acqp: Path,
    index: Path,
    prefix: Path,
    config: DTIConfig,
) -> list[str]:
    cmd = [
        config.eddy_command,
        f"--imain={dwi}",
        f"--mask={mask}",
        f"--acqp={acqp}",
        f"--index={index}",
        f"--bvecs={bvec}",
        f"--bvals={bval}",
        f"--out={prefix}",
        "--data_is_shelled",
    ]
    if config.repol:
        cmd.append("--repol")
    if config.eddy_threads > 1:
        cmd.append(f"--nthr={config.eddy_threads}")
    return cmd


# ---------------------------------------------------------------- steps


def _run_streaming(cmd: list[str], what: str, on_line: Callable[[str], None]) -> None:
    """Run a long tool, passing each output line to ``on_line`` as it appears."""
    tail: deque[str] = deque(maxlen=20)
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            tail.append(line.rstrip())
            on_line(line)
        code = proc.wait()
    if code != 0:
        last = next((t for t in reversed(tail) if t.strip()), "")
        raise StepError(f"{what} failed (exit {code}): {last}")


def _run(cmd: list[str], what: str) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or [""]
        raise StepError(f"{what} failed (exit {proc.returncode}): {tail[0]}")


def write_selected(
    nifti: Path, bval: Path, bvec: Path, out: Path, stem: str
) -> tuple[Path, Path, Path, np.ndarray]:
    """Keep only the b0 + b~1000 volumes. Returns new (dwi, bval, bvec, bvals)."""
    import nibabel as nib

    bvals = np.atleast_1d(np.loadtxt(bval))
    bvecs = np.loadtxt(bvec).reshape(3, -1)  # FSL layout: 3 rows x N volumes
    img = nib.load(str(nifti))
    if img.shape[-1] != bvals.size or bvecs.shape[1] != bvals.size:
        raise StepError(
            f"volumes {img.shape[-1]}, bvals {bvals.size}, bvecs {bvecs.shape[1]} disagree"
        )
    keep = select_shell(bvals)
    data = np.asarray(img.dataobj)[..., keep]
    dwi, bv, bc = out / f"{stem}_dwi.nii.gz", out / f"{stem}_dwi.bval", out / f"{stem}_dwi.bvec"
    nib.save(nib.Nifti1Image(data.astype(np.float32), img.affine, img.header), str(dwi))
    np.savetxt(bv, bvals[keep][None, :], fmt="%g")
    np.savetxt(bc, bvecs[:, keep], fmt="%.6f")
    return dwi, bv, bc, bvals[keep]


def brain_mask(dwi: Path, bvals: np.ndarray, out: Path, stem: str, config: DTIConfig) -> Path:
    import nibabel as nib

    img = nib.load(str(dwi))
    mean_b0 = np.asarray(img.dataobj)[..., bvals < B0_MAX].mean(axis=-1)
    b0_file = out / f"{stem}_b0mean.nii.gz"
    nib.save(nib.Nifti1Image(mean_b0.astype(np.float32), img.affine), str(b0_file))
    brain = out / f"{stem}_b0brain"
    _run(
        [config.bet_command, str(b0_file), str(brain), "-f", str(config.bet_fraction), "-m", "-n"],
        "BET on b0",
    )
    mask = out / f"{stem}_b0brain_mask.nii.gz"
    if not mask.exists():
        raise StepError("BET produced no mask")
    return mask


def mask_volume_ml(mask: Path) -> float:
    import nibabel as nib

    img = nib.load(str(mask))
    voxel_ml = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
    return float((np.asarray(img.dataobj) > 0).sum()) * voxel_ml


def write_eddy_inputs(
    out: Path, stem: str, pe_vec: list[int], readout_s: float, n_vol: int
) -> tuple[Path, Path]:
    acqp, index = out / f"{stem}_acqparams.txt", out / f"{stem}_index.txt"
    acqp.write_text(" ".join(str(v) for v in pe_vec) + f" {readout_s:g}\n")
    index.write_text(" ".join(["1"] * n_vol) + "\n")
    return acqp, index


def run_eddy(
    dwi: Path,
    mask: Path,
    bval: Path,
    bvec: Path,
    acqp: Path,
    index: Path,
    prefix: Path,
    config: DTIConfig,
    on_line: Callable[[str], None] | None = None,
) -> tuple[Path, Path]:
    cmd = eddy_arguments(dwi, mask, bval, bvec, acqp, index, prefix, config)
    if on_line is None:
        _run(cmd, "FSL eddy")
    else:
        _run_streaming(cmd + ["--verbose"], "FSL eddy", on_line)
    corrected = prefix.with_name(prefix.name + ".nii.gz")
    rotated = prefix.with_name(prefix.name + ".eddy_rotated_bvecs")
    if not corrected.exists() or not rotated.exists():
        raise StepError("eddy did not produce corrected data and rotated bvecs")
    return corrected, rotated


def eddy_motion(prefix: Path) -> tuple[float, float]:
    """Mean absolute and relative displacement (mm) from eddy's movement_rms file."""
    rms = prefix.with_name(prefix.name + ".eddy_movement_rms")
    if not rms.exists():
        raise StepError("eddy movement file missing")
    values = np.atleast_2d(np.loadtxt(rms))
    return float(values[:, 0].mean()), float(values[1:, 1].mean() if len(values) > 1 else 0.0)


def fit_tensor(dwi: Path, bval: Path, bvec: Path, mask: Path, fa_out: Path, md_out: Path) -> float:
    """DIPY tensor fit. Writes FA and MD; returns the median FA inside the mask."""
    import nibabel as nib
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel

    img = nib.load(str(dwi))
    data = np.asarray(img.dataobj, dtype=np.float32)
    m = np.asarray(nib.load(str(mask)).dataobj) > 0
    gtab = gradient_table(np.atleast_1d(np.loadtxt(bval)), bvecs=np.loadtxt(bvec).reshape(3, -1).T)
    fit = TensorModel(gtab).fit(data, mask=m)
    fa, md = np.asarray(fit.fa, dtype=np.float32), np.asarray(fit.md, dtype=np.float32)
    bad = ~np.isfinite(fa[m])
    if bad.mean() > 0.01:
        raise StepError(f"tensor fit failed in {bad.mean():.1%} of brain voxels")
    nib.save(nib.Nifti1Image(fa, img.affine), str(fa_out))
    nib.save(nib.Nifti1Image(md, img.affine), str(md_out))
    inside = fa[m][np.isfinite(fa[m])]
    if inside.min() < 0 or inside.max() > 1.0001:
        raise StepError(f"FA outside 0-1 (min {inside.min():.3f}, max {inside.max():.3f})")
    return float(np.median(inside))


def register_to_template(
    fa: Path, md: Path, template: Path, fa_mni: Path, md_mni: Path, tdir: Path, stem: str
) -> None:
    import ants

    fixed = ants.image_read(str(template))
    reg = ants.registration(
        fixed=fixed, moving=ants.image_read(str(fa)), type_of_transform="SyN", verbose=False
    )
    ants.image_write(reg["warpedmovout"], str(fa_mni))
    md_w = ants.apply_transforms(
        fixed=fixed,
        moving=ants.image_read(str(md)),
        transformlist=reg["fwdtransforms"],
        interpolator="linear",
    )
    ants.image_write(md_w, str(md_mni))
    tdir.mkdir(parents=True, exist_ok=True)
    for kind, files in (("fwd", reg["fwdtransforms"]), ("inv", reg["invtransforms"])):
        for f in files:
            name = "warp.nii.gz" if f.endswith(".nii.gz") else "affine.mat"
            shutil.copy(f, tdir / f"{stem}_{kind}_{name}")


def template_correlation(img_path: Path, template: Path) -> float:
    import nibabel as nib

    a = np.asarray(nib.load(str(img_path)).dataobj, dtype=float)
    b = np.asarray(nib.load(str(template)).dataobj, dtype=float)
    if a.shape != b.shape:
        raise StepError(f"aligned shape {a.shape} differs from template {b.shape}")
    inside = b > 0
    return float(np.corrcoef(a[inside], b[inside])[0, 1])


def save_qc_image(fa_mni: Path, md_mni: Path, dst: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib

    fa = np.asarray(nib.load(str(fa_mni)).dataobj, dtype=float)
    md = np.asarray(nib.load(str(md_mni)).dataobj, dtype=float)
    c = [n // 2 for n in fa.shape]
    md_hi = np.percentile(md[md > 0], 99) if (md > 0).any() else 1.0
    fig, ax = plt.subplots(2, 3, figsize=(12, 8), facecolor="black")
    for row, (vol, vmax) in enumerate([(fa, 0.8), (md, md_hi)]):
        for col, sl in enumerate([vol[c[0]], vol[:, c[1]], vol[:, :, c[2]]]):
            ax[row, col].imshow(np.rot90(sl), cmap="gray", vmin=0, vmax=vmax)
            ax[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(dst, dpi=80, facecolor="black")
    plt.close(fig)


# ---------------------------------------------------------------- the pipeline


DTI_STEP_KEYS = ["select", "mask", "eddy", "tensor", "align", "qc"]


def preprocess_dti(
    nifti: Path,
    subject_id: str,
    image_id: str,
    series_name: str,
    out_root: Path,
    config: DTIConfig | None = None,
    redo: bool = False,
    recorder: RunRecorder | NullRecorder | None = None,
) -> DTIResult:
    """Run all steps on one DTI series. Never raises for a bad scan: returns a failed result."""
    config = config or DTIConfig()
    rec = recorder or NullRecorder()
    res = DTIResult(subject_id=subject_id, image_id=image_id, input_nifti=str(nifti))
    out = out_root / subject_id
    stem = subject_id
    try:
        check_tools(config)
        bval = nifti.with_name(nifti.name.removesuffix(".nii.gz") + ".bval")
        bvec = bval.with_suffix(".bvec")
        sidecar_file = bval.with_suffix(".json")
        for f in (nifti, bval, bvec):
            if not f.exists():
                raise StepError(f"input not found: {f.name}")
        sidecar = json.loads(sidecar_file.read_text()) if sidecar_file.exists() else {}
        out.mkdir(parents=True, exist_ok=True)

        res.pe_axis, pe_vec, res.pe_source = phase_encoding(sidecar, series_name)
        res.readout_s, res.readout_source = readout_time(sidecar)

        prefix = out / f"{stem}_eddy"
        corrected = prefix.with_name(prefix.name + ".nii.gz")
        rotated = prefix.with_name(prefix.name + ".eddy_rotated_bvecs")
        fa, md = out / f"{stem}_FA.nii.gz", out / f"{stem}_MD.nii.gz"
        fa_mni, md_mni, tdir = (
            out / f"{stem}_FA_mni.nii.gz",
            out / f"{stem}_MD_mni.nii.gz",
            out / "transforms",
        )
        kept_mask, kept_bval = out / f"{stem}_b0brain_mask.nii.gz", out / f"{stem}_dwi.bval"
        # Finished earlier and cleaned up: reuse the kept results instead of redoing eddy.
        finished = not redo and final_outputs_exist(out, stem)

        with rec.step(subject_id, "select"):
            if finished:
                bv, bc = kept_bval, out / f"{stem}_dwi.bvec"
                bvals = np.atleast_1d(np.loadtxt(kept_bval))
            else:
                dwi, bv, bc, bvals = write_selected(nifti, bval, bvec, out, stem)
            res.n_b0 = int((bvals < B0_MAX).sum())
            res.n_directions = int(bvals.size - res.n_b0)
            res.b_value = float(np.median(bvals[bvals >= B0_MAX]))

        with rec.step(subject_id, "mask"):
            mask = kept_mask if finished else brain_mask(dwi, bvals, out, stem, config)
            res.brain_mask_ml = mask_volume_ml(mask)
            check_mask_volume(res.brain_mask_ml)

        with rec.step(subject_id, "eddy"):
            if not finished and (redo or not (corrected.exists() and rotated.exists())):
                acqp, index = write_eddy_inputs(out, stem, pe_vec, res.readout_s, int(bvals.size))
                corrected, rotated = run_eddy(
                    dwi,
                    mask,
                    bv,
                    bc,
                    acqp,
                    index,
                    prefix,
                    config,
                    on_line=lambda line: rec.note(subject_id, "eddy", line),
                )
            res.mean_abs_motion_mm, res.mean_rel_motion_mm = eddy_motion(prefix)
            check_motion(res.mean_abs_motion_mm)

        with rec.step(subject_id, "tensor"):
            if finished:
                res.median_fa = median_in_mask(fa, mask)
            else:
                # rotated bvecs from eddy, not the originals
                res.median_fa = fit_tensor(corrected, bv, rotated, mask, fa, md)
            check_median_fa(res.median_fa)

        with rec.step(subject_id, "align"):
            if redo or not (fa_mni.exists() and md_mni.exists() and any(tdir.glob("*"))):
                register_to_template(fa, md, config.fa_template, fa_mni, md_mni, tdir, stem)
            res.template_correlation = template_correlation(fa_mni, config.fa_template)
            check_template_correlation(res.template_correlation)

        qc = out / f"{stem}_qc.png"
        with rec.step(subject_id, "qc"):
            save_qc_image(fa_mni, md_mni, qc)
        res.fa_mni, res.md_mni, res.transform_dir, res.qc_image = (
            _rel(fa_mni),
            _rel(md_mni),
            _rel(tdir),
            _rel(qc),
        )
        res.status = "ok"
        if not config.keep_intermediates:
            remove_intermediates(out, stem)
    except StepError as exc:
        res.reason = str(exc)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        res.reason = f"{type(exc).__name__}: {exc}"
    log.info("subject %s -> %s %s", pseudonym(subject_id), res.status, res.reason)
    return res


# Files kept for each person after a successful run (everything else is deleted unless
# --keep-intermediates). These are what feature extraction and the report need.
KEEP_SUFFIXES = (
    "_FA.nii.gz",
    "_MD.nii.gz",
    "_FA_mni.nii.gz",
    "_MD_mni.nii.gz",
    "_b0brain_mask.nii.gz",
    "_dwi.bval",
    "_dwi.bvec",
    "_eddy.eddy_rotated_bvecs",
    "_eddy.eddy_movement_rms",
    "_eddy.eddy_parameters",
    "_acqparams.txt",
    "_index.txt",
    "_qc.png",
)
FINAL_SUFFIXES = (
    "_FA.nii.gz",
    "_MD.nii.gz",
    "_FA_mni.nii.gz",
    "_MD_mni.nii.gz",
    "_b0brain_mask.nii.gz",
    "_dwi.bval",
    "_dwi.bvec",
    "_eddy.eddy_rotated_bvecs",
    "_eddy.eddy_movement_rms",
)


def final_outputs_exist(out: Path, stem: str) -> bool:
    """True when this person's results are complete (with or without in-between files)."""
    tdir = out / "transforms"
    files_ok = all((out / f"{stem}{sfx}").exists() for sfx in FINAL_SUFFIXES)
    return files_ok and tdir.is_dir() and any(tdir.iterdir())


def remove_intermediates(out: Path, stem: str) -> int:
    """Delete in-between files (selected DWI copy, eddy output, b0 mean...). Returns bytes freed.

    Only called after every check passed, and never touches the kept results or transforms.
    """
    freed = 0
    for f in out.iterdir():
        if f.is_file() and f.name.startswith(stem) and not f.name.endswith(KEEP_SUFFIXES):
            freed += f.stat().st_size
            f.unlink()
    # Non-linear warp fields are ~70 MB each. The aligned images are already saved and
    # features are measured on them, so only the small affine files are kept.
    tdir = out / "transforms"
    if tdir.is_dir():
        for f in tdir.glob("*_warp.nii.gz"):
            freed += f.stat().st_size
            f.unlink()
    return freed


def median_in_mask(fa: Path, mask: Path) -> float:
    import nibabel as nib

    values = np.asarray(nib.load(str(fa)).dataobj)[np.asarray(nib.load(str(mask)).dataobj) > 0]
    values = values[np.isfinite(values)]
    return float(np.median(values))


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------- command line


def dti_inputs(table: Path) -> tuple[list[tuple[str, str, str, Path]], list[tuple[str, str]]]:
    """Chosen series per person, plus (subject, reason) for people with no usable DTI."""
    df = pd.read_csv(table, dtype=str)
    df = df[(df["status"] == "ok") & (df["kind"] == "dti")].copy()
    df["path"] = df["nifti_path"].map(lambda p: REPO_ROOT / p)
    df["n_shell_dirs"] = df["path"].map(
        lambda p: count_shell_dirs(p.with_name(p.name.removesuffix(".nii.gz") + ".bval"))
    )
    chosen, skipped = [], []
    for subject, rows in df.groupby("subject_id"):
        try:
            r = choose_series(rows)
            chosen.append((str(subject), str(r.image_id), str(r.series_name), Path(r.path)))
        except StepError as exc:
            skipped.append((str(subject), str(exc)))
    return chosen, skipped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--table", type=Path, default=DEFAULT_TABLE_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--results", type=Path, default=DEFAULT_TABLE_OUT)
    ap.add_argument("--subject", help="only this subject ID")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fa-template", type=Path, default=default_fa_template())
    ap.add_argument("--eddy", default="eddy", help="eddy command (eddy or eddy_cpu)")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--events-url", help="also send progress events to this dashboard URL")
    ap.add_argument(
        "--eddy-threads",
        type=int,
        default=1,
        help="CPU threads per eddy (needs a newer FSL; check: eddy --help | grep nthr)",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="people processed at the same time (each runs its own eddy)",
    )
    ap.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="keep big in-between files (eddy output etc.) instead of deleting them",
    )
    ap.add_argument(
        "--no-repol",
        action="store_true",
        help="skip eddy outlier replacement (faster; applies to everyone in the run)",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = DTIConfig(
        fa_template=args.fa_template,
        eddy_command=args.eddy,
        eddy_threads=args.eddy_threads,
        repol=not args.no_repol,
        keep_intermediates=args.keep_intermediates,
    )
    try:
        check_tools(config)  # fail once, clearly, before starting anyone
    except StepError as exc:
        log.error("%s", exc)
        return 2

    chosen, skipped = dti_inputs(args.table)
    if args.subject:
        chosen = [c for c in chosen if c[0] == args.subject]
        skipped = [s for s in skipped if s[0] == args.subject]
    if args.limit:
        chosen = chosen[: args.limit]

    everyone = [c[0] for c in chosen] + [s for s, _ in skipped]
    rec = RunRecorder("dti", everyone, DTI_STEP_KEYS, events_url=args.events_url, jobs=args.jobs)
    print(f"run events: {_rel(rec.path)}  (watch live: python models/mri/src/pipeline_monitor.py)")
    results = []
    for subject, reason in skipped:  # no usable DTI series: recorded as failed straight away
        rec.subject_start(subject)
        rec.subject_end(subject, "failed", reason)
        results.append(DTIResult(subject_id=subject, image_id="", input_nifti="", reason=reason))

    def one(item: tuple[str, str, str, Path]) -> DTIResult:
        s, i, n, p = item
        rec.subject_start(s)
        r = preprocess_dti(p, s, i, n, args.out, config, args.redo, recorder=rec)
        rec.subject_end(
            s,
            r.status,
            r.reason,
            metrics={
                "brain_mask_ml": r.brain_mask_ml,
                "mean_abs_motion_mm": r.mean_abs_motion_mm,
                "median_fa": r.median_fa,
                "template_correlation": r.template_correlation,
            },
        )
        return r

    # Threads are enough: the slow part (eddy) runs as a separate FSL process.
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        results += list(pool.map(one, chosen))
    rec.run_end()
    if not results:
        log.error("no DTI series selected from %s", args.table)
        return 2

    table = pd.DataFrame([asdict(r) for r in results])
    args.results.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.results, index=False)
    ok = int((table["status"] == "ok").sum())
    print(f"DTI preprocessing: {ok} ok, {len(table) - ok} failed, of {len(table)}")
    for r in results:
        if r.status != "ok":
            print(f"FAILED subject {pseudonym(r.subject_id)}: {r.reason}")
    print(f"table: {_rel(args.results)}")
    return 0 if ok == len(table) else 1


if __name__ == "__main__":
    sys.exit(main())
