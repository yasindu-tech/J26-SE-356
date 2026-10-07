"""convert PPMI DICOM series to NIfTI and build the scanner / site table.

For every series folder found under the raw PPMI download this script:

1. runs ``dcm2niix`` (one compressed NIfTI per series, plus ``.json`` sidecar,
   plus ``.bval`` / ``.bvec`` for diffusion scans);
2. reads the scanner details from one DICOM header (manufacturer, model, field
   strength, software, sequence settings);
3. looks up the IDA site key from the metadata XML files, when one exists;
4. checks the result (exactly one volume, 3D for T1, volumes == bval entries
   for DTI) and records ``ok`` or ``failed`` with the reason.

Outputs go to ``data/interim/`` and ``data/processed/``, both git-ignored.
They hold PPMI subject IDs, so NOTHING they contain may be committed or quoted
in logs. This script only logs hashed subject IDs.

A missing value is written as an empty cell with an explicit ``site_source``
of ``missing``. It is never filled with a default.

Usage (from the repo root):
    python models/mri/src/convert_dicom.py
    python models/mri/src/convert_dicom.py --limit 3        # try three series first
    python models/mri/src/convert_dicom.py --raw-dir "data/raw/PPMI" --raw-dir "data/raw/PPMI 3"
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA = REPO_ROOT / "data"
DEFAULT_OUT_NIFTI = DATA / "processed" / "nifti"
DEFAULT_TABLE = DATA / "interim" / "mri_conversion_table.csv"

IMAGE_DIR = re.compile(r"^I(\d+)$")
log = logging.getLogger("convert_dicom")


def shown(path: Path) -> str:
    """Path relative to the repo when inside it, otherwise as given."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def pseudonym(subject_id: str) -> str:
    """Short hash for log lines. Never log the raw PPMI subject ID."""
    return hashlib.sha256(subject_id.encode()).hexdigest()[:8]


@dataclass
class SeriesRecord:
    subject_id: str
    image_id: str
    series_name: str
    visit_date: str
    n_dicom: int  # 1 file can hold a whole volume (multi-frame DICOM)
    status: str = "failed"  # "ok" or "failed"
    reason: str = ""  # why it failed; empty when ok
    kind: str = ""  # "t1" or "dti" (decided from the output, not the name)
    nifti_path: str = ""
    n_volumes: int | None = None
    shape: str = ""
    voxel_mm: str = ""
    manufacturer: str = ""
    model: str = ""
    field_strength_t: str = ""
    software_versions: str = ""
    tr_ms: str = ""
    te_ms: str = ""
    site_key: str = ""
    site_source: str = "missing"  # "ida_xml" or "missing"


def find_series(raw_dirs: list[Path]) -> list[Path]:
    """Every folder named I<digits> that holds .dcm files.

    Expected layout: <raw>/<subject>/<series name>/<date>/I<image id>/*.dcm
    """
    found: list[Path] = []
    for raw in raw_dirs:
        for d in sorted(raw.rglob("I*")):
            if d.is_dir() and IMAGE_DIR.match(d.name) and any(d.glob("*.dcm")):
                found.append(d)
    return found


def parse_path(series_dir: Path) -> tuple[str, str, str, str]:
    """(subject_id, image_id, series_name, visit_date) from the folder layout."""
    image_id = IMAGE_DIR.match(series_dir.name).group(1)  # type: ignore[union-attr]
    date_dir = series_dir.parent
    series_dir_name = date_dir.parent
    subject = series_dir_name.parent
    return subject.name, image_id, series_dir_name.name, date_dir.name


def load_site_keys(metadata_dirs: list[Path]) -> dict[str, str]:
    """Map image_id -> IDA site key from the downloaded metadata XML files.

    Only some XML files carry a <siteKey>. Where it is absent the image simply
    has no entry; the caller records that as missing, never as a default.
    """
    keys: dict[str, str] = {}
    for meta in metadata_dirs:
        for xml_file in meta.rglob("*.xml"):
            try:
                root = ET.parse(xml_file).getroot()
            except ET.ParseError:
                log.warning("unreadable XML skipped: %s", xml_file.name)
                continue
            site = next((e.text for e in root.iter() if e.tag.endswith("siteKey")), None)
            image = next((e.text for e in root.iter() if e.tag.endswith("imageUID")), None)
            if site and image:
                keys[image.strip()] = site.strip()
    return keys


def field_strength(raw: str) -> str:
    """'3' and '3.0' are the same scanner. Write one form; keep empty as empty."""
    try:
        return f"{float(raw):g}"
    except ValueError:
        return ""


def read_header(series_dir: Path) -> dict[str, str]:
    """Scanner details from the first DICOM file (pixel data is not read)."""
    import pydicom

    first = next(series_dir.glob("*.dcm"))
    ds = pydicom.dcmread(first, stop_before_pixels=True, force=True)

    def get(name: str) -> str:
        value = getattr(ds, name, "")
        return "" if value is None else str(value).strip()

    return {
        "manufacturer": get("Manufacturer"),
        "model": get("ManufacturerModelName"),
        "field_strength_t": field_strength(get("MagneticFieldStrength")),
        "software_versions": get("SoftwareVersions"),
        "tr_ms": get("RepetitionTime"),
        "te_ms": get("EchoTime"),
    }


def run_dcm2niix(series_dir: Path, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "dcm2niix",
        "-z",
        "y",
        "-b",
        "y",
        "-ba",
        "y",
        "-f",
        stem,
        "-o",
        str(out_dir),
        str(series_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"dcm2niix exit code {result.returncode}")


def check_output(out_dir: Path, stem: str) -> tuple[str, int, tuple[int, ...], tuple[float, ...]]:
    """Validate the conversion. Returns (kind, n_volumes, shape, voxel sizes).

    Raises ValueError with a plain reason when something is wrong.
    """
    import nibabel as nib
    import numpy as np

    niftis = sorted(out_dir.glob(f"{stem}*.nii.gz"))
    if any("_Eq" in n.name for n in niftis):
        raise ValueError(
            "unequal slice spacing: dcm2niix also wrote an interpolated '_Eq' copy. "
            "Inspect this series by hand before choosing which file to use"
        )
    if len(niftis) != 1:
        raise ValueError(f"expected 1 NIfTI, found {len(niftis)}")
    img = nib.load(niftis[0])
    shape = tuple(int(s) for s in img.shape)
    voxel = tuple(float(v) for v in img.header.get_zooms()[:3])
    n_vol = shape[3] if len(shape) == 4 else 1
    bval = niftis[0].with_name(niftis[0].name.removesuffix(".nii.gz") + ".bval")
    bvec = bval.with_suffix(".bvec")

    if bval.exists():  # diffusion scan
        if not bvec.exists():
            raise ValueError("DTI has .bval but no .bvec")
        n_bval = len(bval.read_text().split())
        if n_bval != n_vol:
            raise ValueError(f"DTI volumes ({n_vol}) != bval entries ({n_bval})")
        return "dti", n_vol, shape, voxel

    if len(shape) != 3:
        raise ValueError(f"T1 must be 3D, got shape {shape}")
    if not (min(voxel) >= 0.5 and max(voxel) <= 2.5):
        raise ValueError(f"unusual voxel size {voxel} mm")
    if not np.isfinite(img.header.get_zooms()[0]):
        raise ValueError("invalid voxel size")
    return "t1", n_vol, shape, voxel


def convert_one(
    series_dir: Path, out_root: Path, site_keys: dict[str, str], redo: bool = False
) -> SeriesRecord:
    subject, image_id, series_name, visit_date = parse_path(series_dir)
    rec = SeriesRecord(
        subject_id=subject,
        image_id=image_id,
        series_name=series_name,
        visit_date=visit_date,
        n_dicom=sum(1 for _ in series_dir.glob("*.dcm")),
    )
    stem = f"{subject}_I{image_id}"
    out_dir = out_root / subject
    try:
        for key, value in read_header(series_dir).items():
            setattr(rec, key, value)
        if image_id in site_keys:
            rec.site_key, rec.site_source = site_keys[image_id], "ida_xml"
        already_done = any(out_dir.glob(f"{stem}*.nii.gz"))
        if redo or not already_done:  # a finished conversion is still re-checked below
            for old in out_dir.glob(f"{stem}*"):
                old.unlink()
            run_dcm2niix(series_dir, out_dir, stem)
        kind, n_vol, shape, voxel = check_output(out_dir, stem)
        rec.kind, rec.n_volumes = kind, n_vol
        rec.shape = "x".join(map(str, shape))
        rec.voxel_mm = "x".join(f"{v:.2f}" for v in voxel)
        rec.nifti_path = shown(next(out_dir.glob(f"{stem}*.nii.gz")))
        rec.status = "ok"
    except (ValueError, RuntimeError, OSError) as exc:
        rec.status, rec.reason = "failed", str(exc)
    log.info("subject %s series I%s -> %s %s", pseudonym(subject), image_id, rec.status, rec.reason)
    return rec


def summarise(table: pd.DataFrame) -> str:
    ok = table[table["status"] == "ok"]
    lines = [
        f"series found: {len(table)}   ok: {len(ok)}   failed: {len(table) - len(ok)}",
        f"people: {table['subject_id'].nunique()}",
    ]
    if not ok.empty:
        lines.append("scanners (manufacturer / model / field T):")
        scanners = ok.groupby(["manufacturer", "model", "field_strength_t"]).size()
        lines.append(scanners.to_string())
        with_site = (ok["site_source"] == "ida_xml").sum()
        lines.append(
            f"site key known for {with_site} of {len(ok)} series (rest: missing, not zero)"
        )
    for _, row in table[table["status"] == "failed"].iterrows():
        lines.append(
            f"FAILED subject {pseudonym(row['subject_id'])} I{row['image_id']}: {row['reason']}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--raw-dir",
        action="append",
        type=Path,
        help="folder holding <subject>/<series>/<date>/I<id>/*.dcm. Repeatable. "
        "Default: every data/raw/PPMI* folder.",
    )
    ap.add_argument(
        "--metadata-dir",
        action="append",
        type=Path,
        help="folder with IDA metadata XML files. Repeatable. Default: data/raw/PPMI*.",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_NIFTI)
    ap.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    ap.add_argument(
        "--limit", type=int, default=0, help="only convert the first N series (for a trial run)"
    )
    ap.add_argument(
        "--redo", action="store_true", help="convert again even if the NIfTI already exists"
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if shutil.which("dcm2niix") is None:
        log.error("dcm2niix not found. Install it with: brew install dcm2niix")
        return 2

    default_dirs = sorted(p for p in (DATA / "raw").glob("PPMI*") if p.is_dir())
    raw_dirs = args.raw_dir or default_dirs
    meta_dirs = args.metadata_dir or default_dirs
    series = find_series(raw_dirs)
    if not series:
        log.error("no DICOM series found under %s", [str(d) for d in raw_dirs])
        return 2
    if args.limit:
        series = series[: args.limit]

    site_keys = load_site_keys(meta_dirs)
    log.info("%d series to convert, %d image IDs with a site key", len(series), len(site_keys))

    records: list[SeriesRecord] = []
    for s in series:
        subject, image_id, *_ = parse_path(s)
        records.append(convert_one(s, args.out, site_keys, redo=args.redo))

    table = pd.DataFrame(
        [asdict(r) for r in records], columns=[f.name for f in fields(SeriesRecord)]
    )
    args.table.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.table, index=False)
    print(summarise(table))
    print(f"\ntable: {shown(args.table)}")
    return 0 if (table["status"] == "ok").all() else 1


if __name__ == "__main__":
    sys.exit(main())
