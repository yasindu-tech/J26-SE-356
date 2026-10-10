"""MRI-14: T1 preprocessing pipeline.

One scan in, one clean brain in standard space out. Four steps, in this order:

1. N4 bias-field correction on the whole head (evens out brightness);
2. skull-stripping with SynthStrip (or FSL BET as a fallback);
3. registration of the brain to the MNI152 template with ANTs (affine + SyN);
4. tissue labelling (CSF / grey / white) with FSL FAST.

Nothing here learns from the dataset. The template is a fixed external file, so
this runs safely before any train/test split, and the same function can later
process a single new upload.

A step that fails, or a result that fails a check, gives ``status="failed"`` with
the reason. It is never replaced by zeros or a default.

Needs: FSL (``fast``, optionally ``bet``), the SynthStrip Docker wrapper script,
and the Python packages ``antspyx`` and ``nibabel``.

Usage (from the repo root, with the venv active):
    python models/mri/src/preprocess_t1.py --subject <patno>
    python models/mri/src/preprocess_t1.py --limit 3
    python models/mri/src/preprocess_t1.py --synthstrip ~/synthstrip-docker
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
from progress import NullRecorder, RunRecorder

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TABLE_IN = REPO_ROOT / "data" / "interim" / "mri_conversion_table.csv"
DEFAULT_TABLE_OUT = REPO_ROOT / "data" / "interim" / "t1_preprocess_table.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "t1"

# PROVISIONAL limits, chosen before looking at all 60 scans. Check them against the
# batch results and adjust, writing down why (they are not from a paper).
MASK_ML_RANGE = (900.0, 2100.0)  # SynthStrip mask incl. CSF; one subject gave 1524 mL
MIN_TEMPLATE_CORRELATION = 0.5  # warped brain vs template, inside the template brain

log = logging.getLogger("preprocess_t1")


class StepError(Exception):
    """A pipeline step failed or a check did not pass. The message is the reason."""


def pseudonym(subject_id: str) -> str:
    """Short hash for log lines. Never log the raw PPMI subject ID."""
    return hashlib.sha256(subject_id.encode()).hexdigest()[:8]


def default_template() -> Path:
    fsldir = Path(os.environ.get("FSLDIR", Path.home() / "fsl"))
    return fsldir / "data" / "standard" / "MNI152_T1_1mm_brain.nii.gz"


@dataclass(frozen=True)
class T1Config:
    template: Path = field(default_factory=default_template)
    skullstrip: str = "synthstrip"  # "synthstrip" or "bet"
    synthstrip_script: Path = Path.home() / "synthstrip-docker"
    fast_command: str = "fast"
    keep_intermediates: bool = False  # False: delete big in-between files once a person passes
    bet_command: str = "bet"


@dataclass
class T1Result:
    subject_id: str
    input_nifti: str
    status: str = "failed"  # "ok" or "failed"
    reason: str = ""  # why it failed; empty when ok
    skullstrip: str = ""
    brain_mask_ml: float | None = None
    template_correlation: float | None = None
    csf_ml_mni: float | None = None  # volumes are in template space, so they are
    grey_ml_mni: float | None = None  # NOT biological volumes. Use for sanity checks only.
    white_ml_mni: float | None = None
    mni_brain: str = ""
    tissue_seg: str = ""
    transform_dir: str = ""
    qc_image: str = ""


# ---------------------------------------------------------------- checks (pure)


def check_mask_volume(ml: float, limits: tuple[float, float] = MASK_ML_RANGE) -> None:
    """Fail on an empty or implausibly sized brain mask."""
    if not (limits[0] <= ml <= limits[1]):
        raise StepError(f"brain mask is {ml:.0f} mL, outside {limits[0]:.0f}-{limits[1]:.0f} mL")


def check_template_correlation(corr: float, minimum: float = MIN_TEMPLATE_CORRELATION) -> None:
    """Fail when the aligned brain barely resembles the template."""
    if not corr >= minimum:  # also true for NaN, so NaN fails
        raise StepError(
            f"aligned brain matches template poorly (correlation {corr:.2f} < {minimum})"
        )


def check_tools(config: T1Config) -> None:
    """Fail early, with a clear message, when something the pipeline needs is missing."""
    if not config.template.exists():
        raise StepError(f"template not found: {config.template}")
    if shutil.which(config.fast_command) is None:
        raise StepError(f"FSL '{config.fast_command}' not found (is FSL installed and on PATH?)")
    if config.skullstrip == "synthstrip" and not config.synthstrip_script.exists():
        raise StepError(f"SynthStrip wrapper not found: {config.synthstrip_script}")
    if config.skullstrip == "bet" and shutil.which(config.bet_command) is None:
        raise StepError(f"FSL '{config.bet_command}' not found")
    if config.skullstrip not in {"synthstrip", "bet"}:
        raise StepError(f"unknown skull-strip method: {config.skullstrip}")


# ---------------------------------------------------------------- steps


def n4_correct(src: Path, dst: Path) -> None:
    import ants

    image = ants.image_read(str(src))
    corrected = ants.n4_bias_field_correction(image)
    ants.image_write(corrected.clone("float"), str(dst))


def _run(cmd: list[str], what: str) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or [""]
        raise StepError(f"{what} failed (exit {proc.returncode}): {tail[0]}")


def skull_strip(src: Path, brain: Path, mask: Path, config: T1Config) -> None:
    if config.skullstrip == "synthstrip":
        # sys.executable: the wrapper's own '#!/usr/bin/env python' line fails on a Mac with only python3
        _run(
            [
                sys.executable,
                str(config.synthstrip_script),
                "-i",
                str(src),
                "-o",
                str(brain),
                "-m",
                str(mask),
            ],
            "SynthStrip",
        )
    else:
        stem = brain.with_name(brain.name.removesuffix(".nii.gz"))
        _run([config.bet_command, str(src), str(stem), "-R", "-f", "0.4", "-m"], "BET")
        made = stem.with_name(stem.name + "_mask.nii.gz")
        if made != mask:
            shutil.move(made, mask)
    if not brain.exists() or not mask.exists():
        raise StepError("skull-stripping produced no output files")


def mask_volume_ml(mask: Path) -> float:
    import nibabel as nib
    import numpy as np

    img = nib.load(str(mask))
    voxel_ml = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
    return float((np.asarray(img.dataobj) > 0).sum()) * voxel_ml


def register_to_template(
    brain: Path, template: Path, mni_out: Path, transform_dir: Path, stem: str
) -> None:
    """Affine + SyN to the MNI template. Transforms are copied somewhere permanent:
    ANTs leaves them in a temp folder that macOS clears."""
    import ants

    fixed = ants.image_read(str(template))
    moving = ants.image_read(str(brain))
    reg = ants.registration(fixed=fixed, moving=moving, type_of_transform="SyN", verbose=False)
    ants.image_write(reg["warpedmovout"], str(mni_out))
    transform_dir.mkdir(parents=True, exist_ok=True)
    for kind, files in (("fwd", reg["fwdtransforms"]), ("inv", reg["invtransforms"])):
        for f in files:
            name = "warp.nii.gz" if f.endswith(".nii.gz") else "affine.mat"
            shutil.copy(f, transform_dir / f"{stem}_{kind}_{name}")


def template_correlation(mni: Path, template: Path) -> float:
    import nibabel as nib
    import numpy as np

    a = np.asarray(nib.load(str(mni)).dataobj, dtype=float)
    b = np.asarray(nib.load(str(template)).dataobj, dtype=float)
    if a.shape != b.shape:
        raise StepError(f"aligned brain shape {a.shape} differs from template {b.shape}")
    inside = b > 0
    return float(np.corrcoef(a[inside], b[inside])[0, 1])


def segment(mni: Path, prefix: Path, config: T1Config) -> Path:
    _run([config.fast_command, "-t", "1", "-n", "3", "-o", str(prefix), str(mni)], "FSL FAST")
    seg = prefix.with_name(prefix.name + "_seg.nii.gz")
    if not seg.exists():
        raise StepError("FAST produced no segmentation file")
    return seg


def tissue_volumes_ml(seg: Path) -> tuple[float, float, float]:
    import nibabel as nib
    import numpy as np

    img = nib.load(str(seg))
    voxel_ml = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
    data = np.asarray(img.dataobj)
    return tuple(float((data == k).sum()) * voxel_ml for k in (1, 2, 3))  # type: ignore[return-value]


def save_qc_image(mni: Path, seg: Path, dst: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np

    a = np.asarray(nib.load(str(mni)).dataobj, dtype=float)
    s = np.asarray(nib.load(str(seg)).dataobj)
    lo, hi = np.percentile(a[a > 0], [1, 99.5])
    c = [n // 2 for n in a.shape]
    fig, ax = plt.subplots(2, 3, figsize=(12, 8), facecolor="black")
    for row, (vol, cmap, vmin, vmax) in enumerate([(a, "gray", lo, hi), (s, "viridis", 0, 3)]):
        for col, sl in enumerate([vol[c[0]], vol[:, c[1]], vol[:, :, c[2]]]):
            ax[row, col].imshow(np.rot90(sl), cmap=cmap, vmin=vmin, vmax=vmax)
            ax[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(dst, dpi=80, facecolor="black")
    plt.close(fig)


# ---------------------------------------------------------------- the pipeline


T1_STEP_KEYS = ["n4", "skull", "align", "tissue", "qc"]


def preprocess_t1(
    nifti: Path,
    subject_id: str,
    out_root: Path,
    config: T1Config | None = None,
    redo: bool = False,
    recorder: RunRecorder | NullRecorder | None = None,
) -> T1Result:
    """Run the four steps on one T1 scan. Never raises for a bad scan: returns a failed result."""
    config = config or T1Config()
    rec = recorder or NullRecorder()
    result = T1Result(subject_id=subject_id, input_nifti=str(nifti), skullstrip=config.skullstrip)
    out = out_root / subject_id
    try:
        check_tools(config)
        if not nifti.exists():
            raise StepError(f"input not found: {nifti}")
        out.mkdir(parents=True, exist_ok=True)
        n4 = out / f"{subject_id}_n4.nii.gz"
        brain = out / f"{subject_id}_n4_brain.nii.gz"
        mask = out / f"{subject_id}_n4_brain_mask.nii.gz"
        mni = out / f"{subject_id}_mni.nii.gz"
        seg_prefix = out / f"{subject_id}_fast"
        tdir = out / "transforms"
        qc = out / f"{subject_id}_qc.png"

        # Finished earlier and cleaned up: reuse kept results instead of redoing N4/skull-strip.
        finished = not redo and final_outputs_exist(out, subject_id)
        with rec.step(subject_id, "n4"):
            if not finished and (redo or not n4.exists()):
                n4_correct(nifti, n4)
        with rec.step(subject_id, "skull"):
            if not finished and (redo or not (brain.exists() and mask.exists())):
                skull_strip(n4, brain, mask, config)
            result.brain_mask_ml = mask_volume_ml(mask)
            check_mask_volume(result.brain_mask_ml)

        with rec.step(subject_id, "align"):
            if redo or not mni.exists() or not any(tdir.glob("*")):
                register_to_template(brain, config.template, mni, tdir, subject_id)
            result.template_correlation = template_correlation(mni, config.template)
            check_template_correlation(result.template_correlation)

        seg = seg_prefix.with_name(seg_prefix.name + "_seg.nii.gz")
        with rec.step(subject_id, "tissue"):
            if redo or not seg.exists():
                seg = segment(mni, seg_prefix, config)
            result.csf_ml_mni, result.grey_ml_mni, result.white_ml_mni = tissue_volumes_ml(seg)
        with rec.step(subject_id, "qc"):
            save_qc_image(mni, seg, qc)

        result.mni_brain, result.tissue_seg = _rel(mni), _rel(seg)
        result.transform_dir, result.qc_image = _rel(tdir), _rel(qc)
        result.status = "ok"
        if not config.keep_intermediates:
            remove_intermediates(out, subject_id)
    except StepError as exc:
        result.reason = str(exc)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        result.reason = f"{type(exc).__name__}: {exc}"
    log.info("subject %s -> %s %s", pseudonym(subject_id), result.status, result.reason)
    return result


# Kept per person after a successful run; everything else is deleted unless
# --keep-intermediates. Feature extraction needs the aligned brain, tissue maps and mask.
KEEP_SUFFIXES = (
    "_n4_brain_mask.nii.gz",
    "_mni.nii.gz",
    "_fast_seg.nii.gz",
    "_fast_pve_0.nii.gz",
    "_fast_pve_1.nii.gz",
    "_fast_pve_2.nii.gz",
    "_qc.png",
)
FINAL_SUFFIXES = ("_n4_brain_mask.nii.gz", "_mni.nii.gz", "_fast_seg.nii.gz")


def final_outputs_exist(out: Path, stem: str) -> bool:
    """True when this person's results are complete (with or without in-between files)."""
    tdir = out / "transforms"
    files_ok = all((out / f"{stem}{sfx}").exists() for sfx in FINAL_SUFFIXES)
    return files_ok and tdir.is_dir() and any(tdir.iterdir())


def remove_intermediates(out: Path, stem: str) -> int:
    """Delete in-between files (N4 image, unmasked brain copy...). Returns bytes freed."""
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


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------- command line


def t1_inputs(table: Path) -> list[tuple[str, Path]]:
    """(subject, NIfTI path) for every T1 that the conversion step marked ok."""
    df = pd.read_csv(table, dtype=str)
    df = df[(df["status"] == "ok") & (df["kind"] == "t1")]
    return [(str(r.subject_id), REPO_ROOT / r.nifti_path) for r in df.itertuples()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--table",
        type=Path,
        default=DEFAULT_TABLE_IN,
        help="conversion table from convert_dicom.py",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--results", type=Path, default=DEFAULT_TABLE_OUT)
    ap.add_argument("--subject", help="only this subject ID")
    ap.add_argument("--limit", type=int, default=0, help="only the first N subjects")
    ap.add_argument("--skullstrip", choices=["synthstrip", "bet"], default="synthstrip")
    ap.add_argument(
        "--synthstrip",
        type=Path,
        default=Path.home() / "synthstrip-docker",
        help="path to the SynthStrip Docker wrapper script",
    )
    ap.add_argument("--template", type=Path, default=default_template())
    ap.add_argument("--redo", action="store_true", help="recompute steps even if outputs exist")
    ap.add_argument("--events-url", help="also send progress events to this dashboard URL")
    ap.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="keep big in-between files (N4 image etc.) instead of deleting them",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = T1Config(
        template=args.template,
        skullstrip=args.skullstrip,
        synthstrip_script=args.synthstrip,
        keep_intermediates=args.keep_intermediates,
    )

    inputs = t1_inputs(args.table)
    if args.subject:
        inputs = [i for i in inputs if i[0] == args.subject]
    if args.limit:
        inputs = inputs[: args.limit]
    if not inputs:
        log.error("no T1 scans selected from %s", args.table)
        return 2

    rec = RunRecorder("t1", [s for s, _ in inputs], T1_STEP_KEYS, events_url=args.events_url)
    print(f"run events: {_rel(rec.path)}  (watch live: python models/mri/src/pipeline_monitor.py)")
    results = []
    for subject, nifti in inputs:
        rec.subject_start(subject)
        r = preprocess_t1(nifti, subject, args.out, config, args.redo, recorder=rec)
        rec.subject_end(
            subject,
            r.status,
            r.reason,
            metrics={
                "brain_mask_ml": r.brain_mask_ml,
                "template_correlation": r.template_correlation,
            },
        )
        results.append(r)
    rec.run_end()
    table = pd.DataFrame([asdict(r) for r in results])
    args.results.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.results, index=False)

    ok = int((table["status"] == "ok").sum())
    print(f"T1 preprocessing: {ok} ok, {len(table) - ok} failed, of {len(table)}")
    for r in results:
        if r.status != "ok":
            print(f"FAILED subject {pseudonym(r.subject_id)}: {r.reason}")
    print(f"table: {_rel(args.results)}")
    return 0 if ok == len(table) else 1


if __name__ == "__main__":
    sys.exit(main())
