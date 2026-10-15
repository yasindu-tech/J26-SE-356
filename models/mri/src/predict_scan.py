"""Score one person's uploaded MRI: the MRI module's single-scan path, used by the backend.

Input: a folder (or .zip) holding one person's DICOM files, in any folder layout, plus age
and sex (the two covariates ComBat was fitted with). Steps, each reported to the dashboard
page /predict and to an optional status file the backend polls:

    convert   dcm2niix on the whole upload; series sorted into T1 / DTI by what they
              contain (a .bval file means diffusion), not by their typed-in names
    t1        the T1 pipeline (preprocess_t1.py) on the chosen T1
    dti       the DTI pipeline (preprocess_dti.py) on the chosen b~1000 series
    features  the same feature code as training (t1_features.py, dti_features.py)
    score     frozen ComBat + LightGBM from the model bundle (mri_bundle.py)
    explain   exact TreeSHAP contributions of every feature, in plain language

Output (``MriResult`` JSON, schema ``mri-result/1``), following the Modality Contract:

* ``feature_vector``: every model input by name, raw and after harmonisation. This is
  the MRI embedding the future fusion model takes.
* ``score``: the model's raw score in log-odds (0 = the boundary it learned) with a 95%
  interval from the 50 fold copies of the pipeline. Deliberately not a percentage: it is
  not calibrated, and the fusion model will decide how it is combined.
* ``explanation``: per-feature contributions to that score, plus the top drivers in
  words. They describe what moved this model's score, never a cause.
* ``model_card``: training people and cross-validated performance with 95% CIs.
* ``validation``, ``inputs``: which series were used, the quality checks, QC pictures.

The module abstains (``status: "abstained"``, no score) rather than guess when there is no
usable T1 or DTI, a quality check fails, the scanner model was not in the training data
(ComBat cannot harmonise it), or age/sex are missing. It never says "diagnosis": this is a
screening and triage aid.

Command line:

    python models/mri/src/predict_scan.py --input upload.zip --age 67 --sex M --out result.json
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from convert_dicom import DIFFUSION_NAME, SCANNER_DERIVED
from mri_bundle import Bundle, BundleError, SetBundle, load_bundle, normalise_model
from progress import NullRecorder, RunRecorder

SCHEMA_VERSION = "mri-result/1"
STEPS = ["convert", "t1", "dti", "features", "score", "explain"]
SUBJECT = "upload"  # fixed working name; nothing about the person goes into paths or logs
T1_NAME = re.compile(r"MPRAGE|MP-RAGE|SPGR|BRAVO|TFE|T1", re.IGNORECASE)
MAX_UPLOAD_BYTES = 4 * 1024**3
TOP_DRIVERS = 8
DISCLAIMER = (
    "Screening and triage aid for prioritising specialist review. Not a diagnosis. "
    "The score and contributions describe this model, trained on few people; see the model card."
)


class UploadError(ValueError):
    """The upload itself is unusable (not a zip, unsafe paths, too large)."""


# ------------------------------------------------------------------ progress


class Progress:
    """Reports steps to the run recorder (dashboard) and to a JSON status file (backend)."""

    def __init__(self, recorder: RunRecorder | NullRecorder, status_file: Path | None) -> None:
        self.rec = recorder
        self.status_file = status_file
        self.done: list[str] = []
        self.current: str | None = None
        self.note_text = ""
        self._write()

    def _write(self) -> None:
        if self.status_file is None:
            return
        tmp = self.status_file.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "steps": STEPS,
                    "done": self.done,
                    "current": self.current,
                    "note": self.note_text,
                    "t": time.time(),
                }
            )
        )
        tmp.replace(self.status_file)  # atomic: the backend never reads half a file

    @contextmanager
    def step(self, name: str) -> Iterator[None]:
        self.current, self.note_text = name, ""
        self._write()
        with self.rec.step(SUBJECT, name):
            yield
        self.done.append(name)
        self.current = None
        self._write()

    def note(self, text: str) -> None:
        self.note_text = text
        self._write()
        if self.current:
            self.rec.note(SUBJECT, self.current, text, every_s=1.0)


class _InnerSteps:
    """Lets preprocess_t1/dti report their own steps as notes on the outer step."""

    LABELS = {
        "n4": "evening out brightness",
        "skull": "removing the skull",
        "align": "aligning to the standard brain",
        "tissue": "labelling tissue",
        "qc": "saving the check picture",
        "select": "picking the b~1000 scan",
        "mask": "finding the brain",
        "eddy": "correcting motion (eddy, the slow step)",
        "tensor": "computing FA and MD",
    }

    def __init__(self, progress: Progress, prefix: str) -> None:
        self.progress, self.prefix = progress, prefix

    @contextmanager
    def step(self, subject_id: str, step: str) -> Iterator[None]:
        self.progress.note(f"{self.prefix}: {self.LABELS.get(step, step)}")
        yield

    def note(self, subject_id: str, step: str, text: str, every_s: float = 10.0) -> None:
        return  # tool output may contain file names; not forwarded

    def subject_start(self, *a: Any) -> None:
        return

    def subject_end(self, *a: Any, **k: Any) -> None:
        return

    def report(self, *a: Any, **k: Any) -> None:
        return

    def run_end(self) -> None:
        return


# ------------------------------------------------------------------ upload and series


def extract_upload(src: Path, dest: Path, max_bytes: int = MAX_UPLOAD_BYTES) -> Path:
    """A folder is used as it is. A zip is extracted safely into ``dest``."""
    if src.is_dir():
        return src
    if not zipfile.is_zipfile(src):
        raise UploadError("the upload is neither a folder nor a zip file")
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with zipfile.ZipFile(src) as z:
        total = sum(i.file_size for i in z.infolist())
        if total > max_bytes:
            raise UploadError(f"the zip unpacks to {total / 1e9:.1f} GB, over the limit")
        for info in z.infolist():
            target = (dest / info.filename).resolve()
            if not target.is_relative_to(root):
                raise UploadError("the zip contains a path outside its own folder")
        z.extractall(dest)
    return dest


@dataclass
class Series:
    nifti: Path
    kind: str  # "t1", "dti" or "other"
    description: str
    series_number: int
    manufacturer: str
    model: str
    field_strength: str
    voxel_mm: tuple[float, ...]
    shape: tuple[int, ...]
    n_shell_dirs: int = 0

    def summary(self) -> dict:
        return {
            "series_description": self.description,
            "series_number": self.series_number,
            "manufacturer": self.manufacturer,
            "scanner_model": self.model,
            "field_strength_t": self.field_strength,
            "voxel_mm": [round(v, 2) for v in self.voxel_mm],
            "shape": list(self.shape),
            **({"b1000_directions": self.n_shell_dirs} if self.kind == "dti" else {}),
        }


def run_dcm2niix(upload: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["dcm2niix", "-z", "y", "-b", "y", "-ba", "y", "-f", "s%s_%d", "-o", str(out)]
    result = subprocess.run([*cmd, str(upload)], capture_output=True, text=True, check=False)
    if result.returncode != 0 and not any(out.glob("*.nii.gz")):
        raise RuntimeError("dcm2niix found no images it could convert in the upload")


def classify(nifti: Path) -> Series | None:
    """What one converted series is, from its content and JSON sidecar."""
    import nibabel as nib
    from preprocess_dti import count_shell_dirs

    if SCANNER_DERIVED.search(nifti.name):
        return None  # scanner-made maps (ADC, FA...) are never inputs
    stem = nifti.name.removesuffix(".nii.gz")
    sidecar_path = nifti.with_name(stem + ".json")
    meta = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {}
    img = nib.load(str(nifti))
    shape = tuple(int(s) for s in img.shape)
    voxel = tuple(float(v) for v in img.header.get_zooms()[:3])
    desc = str(meta.get("SeriesDescription", ""))
    bval = nifti.with_name(stem + ".bval")
    kind = "other"
    shell = 0
    if "_Eq" in nifti.name:
        kind = "other"  # interpolated copy for uneven slices; never guessed into use
    elif bval.exists():
        kind, shell = "dti", count_shell_dirs(bval)
    elif (
        len(shape) == 3
        and T1_NAME.search(desc)
        and not DIFFUSION_NAME.search(desc)
        and min(voxel) >= 0.5
        and max(voxel) <= 2.5
    ):
        kind = "t1"
    return Series(
        nifti=nifti,
        kind=kind,
        description=desc,
        series_number=int(meta.get("SeriesNumber", 0) or 0),
        manufacturer=str(meta.get("Manufacturer", "")),
        model=str(meta.get("ManufacturerModelName", "")),
        field_strength=str(meta.get("MagneticFieldStrength", "")),
        voxel_mm=voxel,
        shape=shape,
        n_shell_dirs=shell,
    )


def choose_t1(series: list[Series]) -> Series | None:
    """The finest-resolution 3D T1; ties go to the larger volume, then the lower number."""
    t1 = [s for s in series if s.kind == "t1"]
    if not t1:
        return None
    return sorted(t1, key=lambda s: (max(s.voxel_mm), -int(np.prod(s.shape)), s.series_number))[0]


def choose_dti(series: list[Series]) -> Series | None:
    """The DTI series with the most b~1000 directions (at least the 6 a tensor needs)."""
    from preprocess_dti import MIN_DIRECTIONS

    dti = [s for s in series if s.kind == "dti" and s.n_shell_dirs >= MIN_DIRECTIONS]
    if not dti:
        return None
    return sorted(dti, key=lambda s: (-s.n_shell_dirs, s.series_number))[0]


# ------------------------------------------------------------------ plain language

SIDES = {"L": "Left", "R": "Right"}
TISSUE_WORDS = {
    "csf": "fluid (CSF) share",
    "grey": "grey-matter share",
    "white": "white-matter share",
}
STAT_WORDS = {
    "mean": "average T1 brightness",
    "median": "median T1 brightness",
    "std": "spread of T1 brightness",
    "skewness": "skew of T1 brightness",
    "kurtosis": "peakedness of T1 brightness",
    "entropy": "texture variety (entropy) of T1 brightness",
}
TRACT_WORDS = {
    "CC_genu": "corpus callosum (front part)",
    "CC_body": "corpus callosum (middle part)",
    "CC_splenium": "corpus callosum (back part)",
    "cerebral_peduncle": "cerebral peduncle",
    "cst": "corticospinal tract",
    "alic": "internal capsule (front limb)",
    "plic": "internal capsule (back limb)",
}


def feature_label(name: str) -> str:
    """Feature column -> words a clinician can read. Unknown names are returned as they are."""
    if name == "brain_volume_ml":
        return "Whole-brain volume"
    if name == "affine_det":
        return "Head-size scaling to the standard brain"
    if name == "dti_wm_fa_median":
        return "All white matter: typical FA (fibre organisation)"
    if name == "dti_wm_md_median":
        return "All white matter: typical MD (water diffusion)"
    m = re.fullmatch(r"([LR])_(\w+?)_(csf|grey|white)_frac", name)
    if m:
        return f"{SIDES[m[1]]} {m[2]}: {TISSUE_WORDS[m[3]]}"
    m = re.fullmatch(r"([LR])_(\w+?)_t1_(\w+)", name)
    if m and m[3] in STAT_WORDS:
        return f"{SIDES[m[1]]} {m[2]}: {STAT_WORDS[m[3]]}"
    m = re.fullmatch(r"dti_(?:(L|R)_)?(\w+?)_(fa|md)_mean", name)
    if m:
        what = "FA (fibre organisation)" if m[3] == "fa" else "MD (water diffusion)"
        region = TRACT_WORDS.get(m[2], m[2])
        side = f"{SIDES[m[1]]} " if m[1] else ""
        return f"{side}{region}: {what}"
    return name


# ------------------------------------------------------------------ scoring


@dataclass
class Prepared:
    """What the image steps produced for one person."""

    features: dict[str, float] = field(default_factory=dict)
    scanner: dict[str, str] = field(default_factory=dict)  # modality -> scanner model
    available: set[str] = field(default_factory=set)


def score_person(
    bundle: Bundle, prepared: Prepared, age: float, sex: float
) -> tuple[SetBundle, dict] | tuple[None, str]:
    """(set used, scored parts) or (None, reason to abstain)."""
    chosen = bundle.choose(prepared.available)
    if chosen is None:
        return None, "no usable T1 or DTI series, so there is nothing to score"
    batch = {}
    for g in chosen.modalities:
        model = normalise_model(prepared.scanner.get(g, ""))
        group = chosen.scanner_map[g].get(model)
        if group is None:
            return None, (
                f"the {g.upper()} scanner model ({model or 'unknown'}) was not in the training "
                "data, so the scan cannot be harmonised with the training scans"
            )
        batch[g] = group
    final = chosen.final
    row = final.harmonise(prepared.features, batch, age, sex)
    value = final.score(row)
    copies = [m.score(m.harmonise(prepared.features, batch, age, sex)) for m in chosen.members]
    low, high = np.percentile(copies, [2.5, 97.5]) if copies else (value, value)
    contrib, base = final.contributions(row)
    outside = [
        c
        for c in final.columns
        if not (chosen.feature_ranges[c][0] <= float(row[c].iloc[0]) <= chosen.feature_ranges[c][1])
    ]
    return chosen, {
        "batch": batch,
        "row": row,
        "value": value,
        "interval": [float(low), float(high)],
        "n_copies": len(copies),
        "contrib": contrib,
        "base": base,
        "outside": outside,
    }


def explanation(contrib: dict[str, float], base: float, raw: dict[str, float]) -> dict:
    ranked = sorted(contrib.items(), key=lambda kv: (-abs(kv[1]), kv[0]))
    drivers = []
    for name, c in ranked[:TOP_DRIVERS]:
        if c == 0:
            continue
        drivers.append(
            {
                "feature": name,
                "label": feature_label(name),
                "value": raw.get(name),
                "contribution": c,
                "direction": "towards PD" if c > 0 else "towards control",
                "text": (
                    f"{feature_label(name)} "
                    f"{'raised' if c > 0 else 'lowered'} this model's score by {abs(c):.2f}"
                ),
            }
        )
    return {
        "kind": "attribution",
        "method": "TreeSHAP (exact, LightGBM pred_contrib), in log-odds",
        "base_value": base,
        "feature_attributions": dict(ranked),
        "top_drivers": drivers,
        "caveat": "Contributions show what moved this model's score, not a biological cause.",
    }


def _qc_picture(path: Path) -> str | None:
    if not path.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


# ------------------------------------------------------------------ the whole run


def predict_scan(
    upload: Path,
    age: float | None,
    sex: str | None,
    work: Path,
    bundle: Bundle | None = None,
    recorder: RunRecorder | NullRecorder | None = None,
    status_file: Path | None = None,
    eddy_threads: int = 1,
) -> dict:
    """Run every step for one upload and return the MriResult. Never raises for bad input."""
    from preprocess_dti import DTIConfig, preprocess_dti
    from preprocess_t1 import preprocess_t1

    progress = Progress(recorder or NullRecorder(), status_file)
    result: dict[str, Any] = {
        "modality": "mri",
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "abstain_reason": None,
        "error": None,
        "model_version": None,
        "model_set": None,
        "inputs": {"t1": {"found": False}, "dti": {"found": False}},
        # which MRI parts produced the score (the availability mask fusion needs)
        "availability": {"t1": False, "dti": False},
        "validation": {"usable": False, "problems": []},
        "feature_vector": None,
        "score": None,
        "explanation": None,
        "model_card": None,
        "disclaimer": DISCLAIMER,
    }
    problems: list[str] = result["validation"]["problems"]

    def abstain(reason: str) -> dict:
        result["status"], result["abstain_reason"] = "abstained", reason
        return result

    try:
        bundle = bundle or load_bundle()
        result["model_version"] = bundle.version
        sex_code = {"M": 1.0, "F": 0.0}.get(str(sex or "").strip().upper()[:1])
        if age is None or not np.isfinite(age) or sex_code is None:
            return abstain("age and sex are needed: the scanner adjustment was fitted with them")

        with progress.step("convert"):
            files = extract_upload(upload, work / "upload")
            run_dcm2niix(files, work / "nifti")
            series = [s for s in map(classify, sorted((work / "nifti").glob("*.nii.gz"))) if s]
            if any("_Eq" in p.name for p in (work / "nifti").glob("*.nii.gz")):
                problems.append("a series has unequal slice spacing and was not used")
            progress.note(f"{len(series)} series found")
        t1, dti = choose_t1(series), choose_dti(series)
        prepared = Prepared()

        with progress.step("t1"):
            if t1 is None:
                problems.append("no 3D T1 series found")
            else:
                result["inputs"]["t1"] = {"found": True, **t1.summary()}
                r = preprocess_t1(
                    t1.nifti, SUBJECT, work / "t1", recorder=_InnerSteps(progress, "T1")
                )
                checks = {
                    "brain_mask_ml": r.brain_mask_ml,
                    "template_correlation": r.template_correlation,
                }
                result["inputs"]["t1"] |= {"status": r.status, "checks": checks}
                result["inputs"]["t1"]["qc_image"] = _qc_picture(
                    work / "t1" / SUBJECT / f"{SUBJECT}_qc.png"
                )
                if r.status != "ok":
                    problems.append(f"T1 failed a processing step or check: {r.reason}")
                else:
                    prepared.scanner["t1"] = t1.model

        with progress.step("dti"):
            if dti is None:
                problems.append("no DTI series with at least 6 directions at b~1000")
            else:
                result["inputs"]["dti"] = {"found": True, **dti.summary()}
                config = DTIConfig(eddy_threads=eddy_threads)
                r = preprocess_dti(
                    dti.nifti,
                    SUBJECT,
                    str(dti.series_number),
                    dti.description,
                    work / "dti",
                    config=config,
                    recorder=_InnerSteps(progress, "DTI"),
                )
                checks = {
                    "brain_mask_ml": r.brain_mask_ml,
                    "mean_abs_motion_mm": r.mean_abs_motion_mm,
                    "median_fa": r.median_fa,
                    "template_correlation": r.template_correlation,
                    "resampled_to_2mm": r.resampled,
                }
                result["inputs"]["dti"] |= {"status": r.status, "checks": checks}
                result["inputs"]["dti"]["qc_image"] = _qc_picture(
                    work / "dti" / SUBJECT / f"{SUBJECT}_qc.png"
                )
                if r.status != "ok":
                    problems.append(f"DTI failed a processing step or check: {r.reason}")
                else:
                    prepared.scanner["dti"] = dti.model

        with progress.step("features"):
            if "t1" in prepared.scanner:
                from t1_features import DEFAULT_ATLAS, DEFAULT_ATLAS_XML, load_atlas
                from t1_features import extract_subject as t1_extract

                f = t1_extract(
                    work / "t1" / SUBJECT, SUBJECT, load_atlas(DEFAULT_ATLAS, DEFAULT_ATLAS_XML)
                )
                if f.status == "ok":
                    prepared.features |= f.features
                    prepared.available.add("t1")
                else:
                    problems.append(f"T1 features could not be measured: {f.reason}")
            if "dti" in prepared.scanner:
                from dti_features import (
                    DEFAULT_ATLAS,
                    DEFAULT_ATLAS_XML,
                    DEFAULT_JHU,
                    DEFAULT_JHU_XML,
                    load_atlases,
                )
                from dti_features import extract_subject as dti_extract

                atl = load_atlases(DEFAULT_JHU, DEFAULT_JHU_XML, DEFAULT_ATLAS, DEFAULT_ATLAS_XML)
                f = dti_extract(work / "dti" / SUBJECT, SUBJECT, atl)
                if f.status == "ok":
                    prepared.features |= f.features
                    prepared.available.add("dti")
                else:
                    problems.append(f"DTI features could not be measured: {f.reason}")

        with progress.step("score"):
            chosen, parts = score_person(bundle, prepared, float(age), sex_code)
        if chosen is None:
            return abstain(str(parts))
        assert isinstance(parts, dict)
        result["validation"]["usable"] = True
        result["model_set"] = chosen.name
        result["availability"] = {g: g in chosen.modalities for g in ("t1", "dti")}
        result["model_card"] = chosen.card
        cols = chosen.final.columns
        result["feature_vector"] = {
            "names": cols,
            "labels": [feature_label(c) for c in cols],
            "raw": [prepared.features[c] for c in cols],
            "harmonised": [float(parts["row"][c].iloc[0]) for c in cols],
            "scanner_batch": parts["batch"],
            "outside_training_range": parts["outside"],
        }
        result["score"] = {
            "kind": "log_odds",
            "value": parts["value"],
            "interval_95": parts["interval"],
            "interval_method": f"spread over {parts['n_copies']} fold copies of the pipeline",
            "calibrated": False,
            "n_training": chosen.card["training_people"]["total"],
            "meaning": chosen.card["score_meaning"],
        }
        with progress.step("explain"):
            result["explanation"] = explanation(parts["contrib"], parts["base"], prepared.features)
        if parts["outside"]:
            problems.append(
                f"{len(parts['outside'])} feature(s) fall outside the training range; "
                "treat the score with extra caution"
            )
        result["status"] = "scored"
    except (UploadError, BundleError) as e:
        result["error"] = str(e)
    except (RuntimeError, OSError, ValueError) as e:
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", type=Path, required=True, help="DICOM folder or .zip")
    ap.add_argument("--age", type=float, required=True)
    ap.add_argument("--sex", required=True, choices=["M", "F"])
    ap.add_argument("--out", type=Path, required=True, help="result JSON")
    ap.add_argument("--status-file", type=Path, default=None, help="progress JSON for a caller")
    ap.add_argument("--bundle", type=Path, default=None, help="default: data/models/mri/LATEST")
    ap.add_argument("--work", type=Path, default=None, help="default: a temporary folder")
    ap.add_argument("--keep-work", action="store_true", help="keep converted and processed images")
    ap.add_argument("--eddy-threads", type=int, default=1)
    ap.add_argument("--events-url", default=None)
    args = ap.parse_args(argv)

    work = args.work or Path(tempfile.mkdtemp(prefix="mri_predict_"))
    rec = RunRecorder("predict", [SUBJECT], STEPS, events_url=args.events_url)
    rec.subject_start(SUBJECT)
    try:
        bundle = load_bundle(args.bundle) if args.bundle else None
        result = predict_scan(
            args.input,
            args.age,
            args.sex,
            work,
            bundle=bundle,
            recorder=rec,
            status_file=args.status_file,
            eddy_threads=args.eddy_threads,
        )
    except BundleError as e:
        result = {
            "modality": "mri",
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "error": str(e),
        }
    finally:
        if not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)  # converted images never stay behind
    status = result["status"]
    rec.subject_end(SUBJECT, "ok" if status != "failed" else "failed", status)
    rec.run_end()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=float))
    print(
        f"{status}: {result.get('abstain_reason') or result.get('error') or 'scored'} -> {args.out}"
    )
    return 1 if status == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
