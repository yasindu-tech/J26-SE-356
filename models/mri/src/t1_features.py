"""T1 feature extraction: one row of numbers per person from the preprocessed T1.

Input is what ``preprocess_t1.py`` leaves in ``data/processed/t1/<subject>/``:

    <subject>_mni.nii.gz             brain on the MNI152 1 mm grid
    <subject>_fast_seg.nii.gz        hard tissue labels (1 CSF, 2 grey, 3 white)
    <subject>_fast_pve_0/1/2.nii.gz  soft tissue maps (CSF, grey, white)
    <subject>_n4_brain_mask.nii.gz   brain mask in the person's own (native) space
    transforms/<subject>_fwd_affine.mat

Features, for each of 8 regions (caudate, putamen, pallidum, thalamus; left and right) from
the Harvard-Oxford subcortical atlas that ships with FSL:

    tissue composition   mean CSF, grey and white share from the soft maps     (3)
    brightness summary   mean, median, std, skewness, kurtosis, entropy of the
                         T1 brightness divided by the person's own median
                         white-matter brightness                                (6)

plus two whole-brain numbers that survive registration: ``brain_volume_ml`` (native-space
mask) and ``affine_det`` (overall scale of the affine part of the registration). 74 in total.

Design decisions (see the T1 feature guide in the vault):

* No shape features. The regions come from the atlas, so every person has the same shape.
* No texture features yet. Too many correlated numbers for 59 people.
* No regional volumes. FAST ran on the template-space image and the non-linear warps were
  not kept, so a template-space volume is not a biological volume.
* First-order features are written here in numpy rather than through PyRadiomics, to avoid
  a compiled dependency. Definitions follow PyRadiomics' first-order documentation: population
  standard deviation, Pearson (not excess) kurtosis, and entropy in bits over a histogram with
  a fixed bin width.

Nothing here is fitted to data, so running it on everyone before the CV split is not leakage:
each row depends only on that person's own scan.

Subject IDs in the output are pseudonyms (the same hash used by the progress events). The
diagnosis label is never written to this table; it lives in a separate file.

Command line:

    python models/mri/src/t1_features.py                 # everyone with status ok
    python models/mri/src/t1_features.py --limit 3       # first 3 people only
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from progress import NullRecorder, RunRecorder

REPO = Path(__file__).resolve().parents[3]
FSL_DIR = Path(os.environ.get("FSLDIR", str(Path.home() / "fsl")))
ATLAS_DIR = FSL_DIR / "data" / "atlases"
DEFAULT_ATLAS = ATLAS_DIR / "HarvardOxford" / "HarvardOxford-sub-maxprob-thr50-1mm.nii.gz"
DEFAULT_ATLAS_XML = ATLAS_DIR / "HarvardOxford-Subcortical.xml"

# Feature-table region name -> label name in the Harvard-Oxford XML.
REGIONS: dict[str, str] = {
    "L_caudate": "Left Caudate",
    "L_putamen": "Left Putamen",
    "L_pallidum": "Left Pallidum",
    "L_thalamus": "Left Thalamus",
    "R_caudate": "Right Caudate",
    "R_putamen": "Right Putamen",
    "R_pallidum": "Right Pallidum",
    "R_thalamus": "Right Thalamus",
}
TISSUES = ("csf", "grey", "white")  # FAST pve_0, pve_1, pve_2 for a T1 image
STATS = ("mean", "median", "std", "skewness", "kurtosis", "entropy")
GLOBAL_FEATURES = ("brain_volume_ml", "affine_det")
META_COLUMNS = ("subject", "scanner_model", "manufacturer", "site_key")

MIN_REGION_VOXELS = 50  # an atlas region smaller than this means the atlas did not load right
MIN_WHITE_VOXELS = 10_000  # far below any real brain; guards the brightness normalisation
ENTROPY_BIN_WIDTH = 0.025  # in units of "median white-matter brightness"


class FeatureError(ValueError):
    """A person's features cannot be computed; the reason says why."""


def feature_names() -> list[str]:
    """Every feature column, in table order."""
    names = []
    for region in REGIONS:
        names += [f"{region}_{t}_frac" for t in TISSUES]
        names += [f"{region}_t1_{s}" for s in STATS]
    return names + list(GLOBAL_FEATURES)


def pseudonym(subject_id: str) -> str:
    """Same hash as progress.pseudonym, so tables and run events can be joined."""
    return hashlib.sha256(subject_id.encode()).hexdigest()[:8]


# ------------------------------------------------------------------ atlas


@dataclass
class Atlas:
    masks: dict[str, np.ndarray]  # region -> boolean mask on the MNI grid
    affine: np.ndarray
    shape: tuple[int, ...]


def atlas_label_values(xml_path: Path) -> dict[str, int]:
    """Label name -> voxel value in the maxprob image (the XML index plus one)."""
    if not xml_path.exists():
        raise FeatureError(f"atlas label file not found: {xml_path}")
    root = ET.parse(xml_path).getroot()
    values = {}
    for label in root.iter("label"):
        name = (label.text or "").strip()
        values[name] = int(label.attrib["index"]) + 1
    return values


def load_atlas(atlas_path: Path, xml_path: Path) -> Atlas:
    import nibabel as nib

    if not atlas_path.exists():
        raise FeatureError(f"atlas image not found: {atlas_path}")
    values = atlas_label_values(xml_path)
    missing = [name for name in REGIONS.values() if name not in values]
    if missing:
        raise FeatureError(f"atlas has no label for: {', '.join(missing)}")
    img = nib.load(str(atlas_path))
    data = np.asarray(img.dataobj).astype(int)
    masks = {}
    for region, name in REGIONS.items():
        mask = data == values[name]
        if mask.sum() < MIN_REGION_VOXELS:
            raise FeatureError(f"atlas region {region} has only {int(mask.sum())} voxels")
        masks[region] = mask
    return Atlas(masks=masks, affine=img.affine, shape=data.shape)


def check_same_grid(img_affine: np.ndarray, img_shape: tuple[int, ...], atlas: Atlas) -> None:
    """The brain and the atlas must sit on the same voxel grid, or every region is wrong."""
    if tuple(img_shape[:3]) != tuple(atlas.shape[:3]):
        raise FeatureError(f"image shape {img_shape[:3]} differs from atlas {atlas.shape[:3]}")
    if not np.allclose(img_affine, atlas.affine, atol=1e-3):
        raise FeatureError("image and atlas have different voxel-to-world matrices")


# ------------------------------------------------------------------ features


def first_order(values: np.ndarray, bin_width: float = ENTROPY_BIN_WIDTH) -> dict[str, float]:
    """Mean, median, std, skewness, kurtosis and entropy of one region's brightness."""
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        raise FeatureError("region has no voxels")
    mean = float(v.mean())
    std = float(v.std())  # population SD, as PyRadiomics
    if std > 0:
        z = (v - mean) / std
        skew = float(np.mean(z**3))
        kurt = float(np.mean(z**4))  # Pearson kurtosis: 3 for a normal distribution
    else:
        skew, kurt = 0.0, 0.0
    # Fixed bin width, binned the PyRadiomics way: floor(x / w) - floor(min / w).
    bins = np.floor(v / bin_width).astype(np.int64)
    counts = np.bincount(bins - bins.min())
    p = counts[counts > 0] / v.size
    entropy = float(-(p * np.log2(p)).sum())
    return {
        "mean": mean,
        "median": float(np.median(v)),
        "std": std,
        "skewness": skew,
        "kurtosis": kurt,
        "entropy": entropy,
    }


def normalise_brightness(t1: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Divide by the median white-matter brightness, so scanners with a different gain agree."""
    white = t1[seg == 3]
    if white.size < MIN_WHITE_VOXELS:
        raise FeatureError(f"only {white.size} white-matter voxels; segmentation looks wrong")
    ref = float(np.median(white))
    if ref <= 0:
        raise FeatureError("median white-matter brightness is not positive")
    return t1 / ref


def tissue_features(pves: list[np.ndarray], atlas: Atlas) -> dict[str, float]:
    """Mean CSF, grey and white share inside each region."""
    out: dict[str, float] = {}
    for region, mask in atlas.masks.items():
        for tissue, pve in zip(TISSUES, pves, strict=True):
            out[f"{region}_{tissue}_frac"] = float(pve[mask].mean())
    return out


def brightness_features(t1_norm: np.ndarray, atlas: Atlas) -> dict[str, float]:
    """First-order statistics of normalised brightness inside each region."""
    out: dict[str, float] = {}
    for region, mask in atlas.masks.items():
        for stat, value in first_order(t1_norm[mask]).items():
            out[f"{region}_t1_{stat}"] = value
    return out


def region_features(t1_norm: np.ndarray, pves: list[np.ndarray], atlas: Atlas) -> dict[str, float]:
    return tissue_features(pves, atlas) | brightness_features(t1_norm, atlas)


def mask_volume_ml(mask_path: Path) -> float:
    import nibabel as nib

    img = nib.load(str(mask_path))
    voxel_ml = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
    return float((np.asarray(img.dataobj) > 0).sum()) * voxel_ml


def affine_determinant(mat_path: Path) -> float:
    """|det| of the 3x3 part of an ANTs affine (.mat). One number for overall scale."""
    from scipy.io import loadmat

    m = loadmat(str(mat_path))
    key = next((k for k in m if k.startswith("AffineTransform")), None)
    if key is None:
        raise FeatureError(f"no affine parameters in {mat_path.name}")
    params = np.asarray(m[key], dtype=float).ravel()
    if params.size != 12:
        raise FeatureError(f"expected 12 affine parameters, found {params.size}")
    return float(abs(np.linalg.det(params[:9].reshape(3, 3))))


# ------------------------------------------------------------------ one person


@dataclass
class SubjectFeatures:
    subject: str  # pseudonym, never the raw ID
    status: str = "ok"
    reason: str = ""
    features: dict[str, float] = field(default_factory=dict)


def subject_files(subject_dir: Path, subject_id: str) -> dict[str, Path]:
    return {
        "t1": subject_dir / f"{subject_id}_mni.nii.gz",
        "seg": subject_dir / f"{subject_id}_fast_seg.nii.gz",
        "pve_0": subject_dir / f"{subject_id}_fast_pve_0.nii.gz",
        "pve_1": subject_dir / f"{subject_id}_fast_pve_1.nii.gz",
        "pve_2": subject_dir / f"{subject_id}_fast_pve_2.nii.gz",
        "mask": subject_dir / f"{subject_id}_n4_brain_mask.nii.gz",
        "affine": subject_dir / "transforms" / f"{subject_id}_fwd_affine.mat",
    }


FEATURE_STEPS = ["load", "tissue", "brightness", "size"]


def extract_subject(
    subject_dir: Path,
    subject_id: str,
    atlas: Atlas,
    recorder: RunRecorder | NullRecorder | None = None,
) -> SubjectFeatures:
    """All features for one person. Never raises for a bad scan: returns status 'failed'."""
    import nibabel as nib

    rec = recorder or NullRecorder()
    result = SubjectFeatures(subject=pseudonym(subject_id))
    try:
        files = subject_files(subject_dir, subject_id)
        with rec.step(subject_id, "load"):
            missing = [k for k, p in files.items() if not p.exists()]
            if missing:
                raise FeatureError(f"missing outputs: {', '.join(missing)}")
            t1_img = nib.load(str(files["t1"]))
            check_same_grid(t1_img.affine, t1_img.shape, atlas)
            t1 = np.asarray(t1_img.dataobj, dtype=float)
            seg = np.asarray(nib.load(str(files["seg"])).dataobj).astype(int)
            pves = [
                np.asarray(nib.load(str(files[f"pve_{i}"])).dataobj, dtype=float) for i in range(3)
            ]
            for arr in (seg, *pves):
                if arr.shape != t1.shape:
                    raise FeatureError("tissue maps and T1 have different shapes")
        with rec.step(subject_id, "tissue"):
            feats = tissue_features(pves, atlas)
        with rec.step(subject_id, "brightness"):
            feats |= brightness_features(normalise_brightness(t1, seg), atlas)
        with rec.step(subject_id, "size"):
            feats["brain_volume_ml"] = mask_volume_ml(files["mask"])
            feats["affine_det"] = affine_determinant(files["affine"])
        bad = [k for k, v in feats.items() if not np.isfinite(v)]
        if bad:
            raise FeatureError(f"non-finite values in {len(bad)} features, e.g. {bad[0]}")
        result.features = feats
    except (FeatureError, OSError, ValueError) as e:
        result.status, result.reason = "failed", str(e)
    return result


# ------------------------------------------------------------------ table


def build_table(results: list[SubjectFeatures], meta: dict[str, dict[str, str]]):
    """Feature table (ok people only) with metadata columns first. Raises on duplicates."""
    import pandas as pd

    ok = [r for r in results if r.status == "ok"]
    subjects = [r.subject for r in ok]
    if len(set(subjects)) != len(subjects):
        raise FeatureError("the same person appears twice in the feature table")
    rows = []
    for r in ok:
        m = meta.get(r.subject, {})
        rows.append(
            {
                **{c: m.get(c, "") for c in META_COLUMNS if c != "subject"},
                "subject": r.subject,
                **r.features,
            }
        )
    table = pd.DataFrame(rows, columns=[*META_COLUMNS, *feature_names()])
    if table[feature_names()].isna().any().any():
        raise FeatureError("feature table has missing values")
    return table


def scanner_metadata(conversion_table: Path) -> dict[str, dict[str, str]]:
    """Pseudonym -> scanner model, manufacturer and site key, from the conversion table."""
    import pandas as pd

    df = pd.read_csv(conversion_table, dtype=str).fillna("")
    df = df[(df["status"] == "ok") & (df["kind"] == "t1")]
    meta = {}
    for _, row in df.iterrows():
        meta[pseudonym(row["subject_id"])] = {
            "scanner_model": row.get("model", ""),
            "manufacturer": row.get("manufacturer", ""),
            "site_key": row.get("site_key", ""),  # blank stays blank, never "no site"
        }
    return meta


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--t1-dir", type=Path, default=REPO / "data/processed/t1")
    ap.add_argument("--table", type=Path, default=REPO / "data/interim/t1_preprocess_table.csv")
    ap.add_argument(
        "--conversion", type=Path, default=REPO / "data/interim/mri_conversion_table.csv"
    )
    ap.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    ap.add_argument("--atlas-xml", type=Path, default=DEFAULT_ATLAS_XML)
    ap.add_argument("--out", type=Path, default=REPO / "data/processed/features/t1_features.csv")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--events-url", default=None, help="send progress to a remote dashboard")
    args = ap.parse_args(argv)

    try:
        atlas = load_atlas(args.atlas, args.atlas_xml)
    except FeatureError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    done = pd.read_csv(args.table, dtype=str)
    ids = done.loc[done["status"] == "ok", "subject_id"].tolist()[: args.limit]
    rec = RunRecorder("t1features", ids, FEATURE_STEPS, events_url=args.events_url)
    print("watch live: python models/mri/src/pipeline_monitor.py, then open /t1features")
    results = []
    for i, sid in enumerate(ids, 1):
        rec.subject_start(sid)
        r = extract_subject(args.t1_dir / sid, sid, atlas, recorder=rec)
        metrics = {k: r.features.get(k) for k in GLOBAL_FEATURES}
        rec.subject_end(sid, r.status, r.reason, metrics=metrics)
        print(f"[{i}/{len(ids)}] {r.subject}: {r.status} {r.reason}".rstrip())
        results.append(r)
    rec.run_end()
    table = build_table(results, scanner_metadata(args.conversion))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    failed = [r for r in results if r.status != "ok"]
    report = args.out.with_name(args.out.stem + "_failures.csv")
    pd.DataFrame([vars(r) | {"features": ""} for r in failed]).to_csv(report, index=False)
    print(f"{len(table)} people x {len(feature_names())} features -> {args.out}")
    print(f"{len(failed)} failed -> {report}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
