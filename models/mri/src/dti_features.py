"""DTI feature extraction: one row of numbers per person from the preprocessed DTI.

Input is what ``preprocess_dti.py`` keeps in ``data/processed/dti/<subject>/``: the FA and
MD maps registered to the FMRIB58 template (``<subject>_FA_mni.nii.gz``,
``<subject>_MD_mni.nii.gz``), which share the MNI152 1 mm grid with FSL's atlases.

Features (32):

    white-matter tracts   mean FA and mean MD in 11 JHU ICBM-DTI-81 labels: genu, body
                          and splenium of the corpus callosum; left and right cerebral
                          peduncle, corticospinal tract, anterior and posterior limb of
                          the internal capsule                                      (22)
    deep grey matter      mean MD in the 8 Harvard-Oxford regions used for T1
                          (caudate, putamen, pallidum, thalamus; left and right)     (8)
    whole white matter    median FA and median MD over all JHU-labelled voxels       (2)

MD is reported in 10^-3 mm^2/s (DIPY gives mm^2/s; a healthy white-matter value is near
0.7). FA in grey matter is low and noisy, so deep grey matter gets MD only.

The tract list is a proposal kept small for about 60 people; it covers the motor pathways
and the corpus callosum and can be changed in ``TRACTS``. Each name must exist in the atlas
label file or the run stops, so a typo cannot silently drop a region.

A region counts only voxels the scan actually covers (FA and MD above zero after
registration). If less than half of a region is covered (for example a short scan that
misses the brainstem), that person fails with the reason instead of getting a value from a
sliver.

Nothing is fitted to data, so running this on everyone before the CV split is not leakage.
Subject IDs in the output are pseudonyms; the diagnosis label is never in this table.

Command line:

    python models/mri/src/dti_features.py              # everyone with status ok
    python models/mri/src/dti_features.py --limit 3
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from progress import NullRecorder, RunRecorder
from t1_features import (
    ATLAS_DIR,
    DEFAULT_ATLAS,
    DEFAULT_ATLAS_XML,
    META_COLUMNS,
    REGIONS,
    Atlas,
    FeatureError,
    check_same_grid,
    pseudonym,
)

REPO = Path(__file__).resolve().parents[3]
DEFAULT_JHU = ATLAS_DIR / "JHU" / "JHU-ICBM-labels-1mm.nii.gz"
DEFAULT_JHU_XML = ATLAS_DIR / "JHU-labels.xml"

# Feature-table name -> label name in JHU-labels.xml.
TRACTS: dict[str, str] = {
    "CC_genu": "Genu of corpus callosum",
    "CC_body": "Body of corpus callosum",
    "CC_splenium": "Splenium of corpus callosum",
    "L_cerebral_peduncle": "Cerebral peduncle L",
    "R_cerebral_peduncle": "Cerebral peduncle R",
    "L_cst": "Corticospinal tract L",
    "R_cst": "Corticospinal tract R",
    "L_alic": "Anterior limb of internal capsule L",
    "R_alic": "Anterior limb of internal capsule R",
    "L_plic": "Posterior limb of internal capsule L",
    "R_plic": "Posterior limb of internal capsule R",
}
MD_SCALE = 1000.0  # mm^2/s -> 10^-3 mm^2/s
MIN_COVERAGE = 0.5  # share of a region's voxels the scan must cover
MIN_REGION_VOXELS = 20
FEATURE_STEPS = ["load", "tracts", "deepgrey", "whole"]


def feature_names() -> list[str]:
    names = []
    for t in TRACTS:
        names += [f"dti_{t}_fa_mean", f"dti_{t}_md_mean"]
    names += [f"dti_{r}_md_mean" for r in REGIONS]
    return names + ["dti_wm_fa_median", "dti_wm_md_median"]


# ------------------------------------------------------------------ atlases


def label_values(xml_path: Path) -> dict[str, int]:
    """Label name -> voxel value. FSL label atlases store the value as the XML index;
    probabilistic atlases (whose maxprob images we use) store index + 1."""
    if not xml_path.exists():
        raise FeatureError(f"atlas label file not found: {xml_path}")
    root = ET.parse(xml_path).getroot()
    kind = (root.findtext("header/type") or "").strip().lower()
    offset = 1 if kind == "probabilistic" else 0
    return {
        (lab.text or "").strip(): int(lab.attrib["index"]) + offset for lab in root.iter("label")
    }


def load_label_atlas(atlas_path: Path, xml_path: Path, wanted: dict[str, str]) -> Atlas:
    """Boolean mask per wanted region. Fails loudly on a missing name or an empty region."""
    import nibabel as nib

    if not atlas_path.exists():
        raise FeatureError(f"atlas image not found: {atlas_path}")
    values = label_values(xml_path)
    missing = [n for n in wanted.values() if n not in values]
    if missing:
        raise FeatureError(f"atlas has no label for: {', '.join(missing)}")
    img = nib.load(str(atlas_path))
    data = np.asarray(img.dataobj).astype(int)
    masks = {}
    for key, name in wanted.items():
        mask = data == values[name]
        if mask.sum() < MIN_REGION_VOXELS:
            raise FeatureError(f"atlas region {key} has only {int(mask.sum())} voxels")
        masks[key] = mask
    return Atlas(masks=masks, affine=img.affine, shape=data.shape)


@dataclass
class DtiAtlases:
    tracts: Atlas
    deep_grey: Atlas
    white_matter: np.ndarray  # every voxel with any JHU label


def load_atlases(jhu: Path, jhu_xml: Path, ho: Path, ho_xml: Path) -> DtiAtlases:
    import nibabel as nib

    tracts = load_label_atlas(jhu, jhu_xml, TRACTS)
    deep = load_label_atlas(ho, ho_xml, REGIONS)
    if tracts.shape != deep.shape or not np.allclose(tracts.affine, deep.affine, atol=1e-3):
        raise FeatureError("JHU and Harvard-Oxford atlases are on different grids")
    wm = np.asarray(nib.load(str(jhu)).dataobj).astype(int) > 0
    return DtiAtlases(tracts, deep, wm)


# ------------------------------------------------------------------ measurements


def covered_mean(values: np.ndarray, mask: np.ndarray, covered: np.ndarray, name: str) -> float:
    """Mean over the region's voxels that the scan covers; fail if too little is covered."""
    inside = mask & covered
    share = inside.sum() / mask.sum()
    if share < MIN_COVERAGE:
        raise FeatureError(f"region {name} only {share:.0%} covered by the scan")
    return float(values[inside].mean())


def tract_features(
    fa: np.ndarray, md: np.ndarray, covered: np.ndarray, atlas: Atlas
) -> dict[str, float]:
    out = {}
    for t, mask in atlas.masks.items():
        out[f"dti_{t}_fa_mean"] = covered_mean(fa, mask, covered, t)
        out[f"dti_{t}_md_mean"] = covered_mean(md, mask, covered, t) * MD_SCALE
    return out


def deep_grey_features(md: np.ndarray, covered: np.ndarray, atlas: Atlas) -> dict[str, float]:
    return {
        f"dti_{r}_md_mean": covered_mean(md, mask, covered, r) * MD_SCALE
        for r, mask in atlas.masks.items()
    }


def whole_wm_features(
    fa: np.ndarray, md: np.ndarray, covered: np.ndarray, wm: np.ndarray
) -> dict[str, float]:
    inside = wm & covered
    if inside.sum() < MIN_COVERAGE * wm.sum():
        raise FeatureError("scan covers less than half of the white matter")
    return {
        "dti_wm_fa_median": float(np.median(fa[inside])),
        "dti_wm_md_median": float(np.median(md[inside])) * MD_SCALE,
    }


# ------------------------------------------------------------------ one person


@dataclass
class SubjectFeatures:
    subject: str  # pseudonym, never the raw ID
    status: str = "ok"
    reason: str = ""
    features: dict[str, float] = field(default_factory=dict)


def extract_subject(
    subject_dir: Path,
    subject_id: str,
    atlases: DtiAtlases,
    recorder: RunRecorder | NullRecorder | None = None,
) -> SubjectFeatures:
    """All DTI features for one person. Never raises for a bad scan: returns 'failed'."""
    import nibabel as nib

    rec = recorder or NullRecorder()
    result = SubjectFeatures(subject=pseudonym(subject_id))
    try:
        with rec.step(subject_id, "load"):
            fa_p = subject_dir / f"{subject_id}_FA_mni.nii.gz"
            md_p = subject_dir / f"{subject_id}_MD_mni.nii.gz"
            missing = [p.name.split("_", 1)[1] for p in (fa_p, md_p) if not p.exists()]
            if missing:
                raise FeatureError(f"missing outputs: {', '.join(missing)}")
            fa_img = nib.load(str(fa_p))
            check_same_grid(fa_img.affine, fa_img.shape, atlases.tracts)
            fa = np.asarray(fa_img.dataobj, dtype=float)
            md = np.asarray(nib.load(str(md_p)).dataobj, dtype=float)
            if md.shape != fa.shape:
                raise FeatureError("FA and MD maps have different shapes")
            covered = (fa > 0) & (md > 0)
        with rec.step(subject_id, "tracts"):
            feats = tract_features(fa, md, covered, atlases.tracts)
        with rec.step(subject_id, "deepgrey"):
            feats |= deep_grey_features(md, covered, atlases.deep_grey)
        with rec.step(subject_id, "whole"):
            feats |= whole_wm_features(fa, md, covered, atlases.white_matter)
        bad = [k for k, v in feats.items() if not np.isfinite(v)]
        if bad:
            raise FeatureError(f"non-finite values in {len(bad)} features, e.g. {bad[0]}")
        result.features = feats
    except (FeatureError, OSError, ValueError) as e:
        result.status, result.reason = "failed", str(e)
    return result


# ------------------------------------------------------------------ inputs and table


def finished_people(result_tables: list[Path]) -> dict[str, str]:
    """subject_id -> image_id for everyone with an ok DTI result. A later table wins."""
    import pandas as pd

    people: dict[str, str] = {}
    for t in result_tables:
        df = pd.read_csv(t, dtype=str).fillna("")
        for _, row in df[df["status"] == "ok"].iterrows():
            people[row["subject_id"]] = row["image_id"]
    return people


def scanner_metadata(
    conversion_tables: list[Path], people: dict[str, str]
) -> dict[str, dict[str, str]]:
    """Pseudonym -> scanner details of the DTI series that was actually used."""
    import pandas as pd

    meta = {}
    for t in conversion_tables:
        df = pd.read_csv(t, dtype=str).fillna("")
        for _, row in df.iterrows():
            if people.get(row["subject_id"]) == row["image_id"]:
                meta[pseudonym(row["subject_id"])] = {
                    "scanner_model": row.get("model", ""),
                    "manufacturer": row.get("manufacturer", ""),
                    "site_key": row.get("site_key", ""),  # blank stays blank
                }
    return meta


def build_table(results: list[SubjectFeatures], meta: dict[str, dict[str, str]]):
    """Feature table (ok people only), metadata columns first. Raises on duplicates."""
    import pandas as pd

    ok = [r for r in results if r.status == "ok"]
    if len({r.subject for r in ok}) != len(ok):
        raise FeatureError("the same person appears twice in the feature table")
    rows = []
    for r in ok:
        m = meta.get(r.subject, {})
        rows.append({**{c: m.get(c, "") for c in META_COLUMNS}, "subject": r.subject, **r.features})
    table = pd.DataFrame(rows, columns=[*META_COLUMNS, *feature_names()])
    if table[feature_names()].isna().any().any():
        raise FeatureError("feature table has missing values")
    return table


def default_tables(interim: Path) -> tuple[list[Path], list[Path]]:
    """Every DTI results table (oldest first, so the newest wins) and conversion table."""
    results = [*interim.glob("dti*_results.csv"), *interim.glob("dti_preprocess_table.csv")]
    results.sort(key=lambda p: p.stat().st_mtime)
    return results, sorted(interim.glob("dti*_conversion.csv"))


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dti-dir", type=Path, default=REPO / "data/processed/dti")
    ap.add_argument("--interim", type=Path, default=REPO / "data/interim")
    ap.add_argument("--jhu", type=Path, default=DEFAULT_JHU)
    ap.add_argument("--jhu-xml", type=Path, default=DEFAULT_JHU_XML)
    ap.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    ap.add_argument("--atlas-xml", type=Path, default=DEFAULT_ATLAS_XML)
    ap.add_argument("--out", type=Path, default=REPO / "data/processed/features/dti_features.csv")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--events-url", default=None, help="send progress to a remote dashboard")
    args = ap.parse_args(argv)

    results_tables, conversion_tables = default_tables(args.interim)
    if not results_tables:
        print(f"error: no dti*_results.csv in {args.interim}", file=sys.stderr)
        return 2
    try:
        atlases = load_atlases(args.jhu, args.jhu_xml, args.atlas, args.atlas_xml)
    except FeatureError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    people = finished_people(results_tables)
    ids = sorted(people)[: args.limit]
    rec = RunRecorder("dtifeatures", ids, FEATURE_STEPS, events_url=args.events_url)
    print("watch live: python models/mri/src/pipeline_monitor.py, then open /dtifeatures")
    results = []
    for i, sid in enumerate(ids, 1):
        rec.subject_start(sid)
        r = extract_subject(args.dti_dir / sid, sid, atlases, recorder=rec)
        metrics = {k: r.features.get(k) for k in ("dti_wm_fa_median", "dti_wm_md_median")}
        rec.subject_end(sid, r.status, r.reason, metrics=metrics)
        print(f"[{i}/{len(ids)}] {r.subject}: {r.status} {r.reason}".rstrip())
        results.append(r)
    rec.run_end()

    table = build_table(results, scanner_metadata(conversion_tables, people))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    failed = [r for r in results if r.status != "ok"]
    report = args.out.with_name(args.out.stem + "_failures.csv")
    pd.DataFrame([{"subject": r.subject, "reason": r.reason} for r in failed]).to_csv(
        report, index=False
    )
    print(f"{len(table)} people x {len(feature_names())} features -> {args.out}")
    print(f"{len(failed)} failed -> {report}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
