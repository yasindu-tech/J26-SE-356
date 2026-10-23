"""Export trained model bundles as zips that teammates can unpack and use.

    python scripts/export_models.py                       # every module with a bundle here
    python scripts/export_models.py --module tapping      # one module, the version in CURRENT
    python scripts/export_models.py --module mri --version 20261022_220745_b948f12 --out ~/Desktop

Writes one pdxai_models_<module>_<version>.zip per module, containing data/models/<module>/
<version>/, the version pointer (LATEST for MRI, CURRENT for the others), a README.txt and a
top-level MANIFEST.json listing every file with its SHA-256.

To use it, a teammate unzips it at the repository root (the zip paths start with data/models/):

    unzip pdxai_models_<module>_<version>.zip -d <repo root>

Every bundle is checked against its own hashes when it loads, so a changed or partly copied
file is refused. Share the zips privately (team drive), never through git: the models are
derived from data under data use agreements. This script refuses to write a zip that is not
the bundle it checked, and a bundle may hold model parameters only: no scans, recordings,
labels, subject-level rows or pickles (see the checks below).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "data" / "models"
MRI_FILES = ("bundle.json", "manifest.json", "model_card.json")
# Modules whose bundles follow pdxai-bundle/1: manifest.json lists every file with its SHA-256.
LISTED = {"tapping"}
POINTER = {"mri": "LATEST"}  # every other module: CURRENT
# Keys a model bundle must not carry. A bundle with any of these holds person-level data.
FORBIDDEN_KEYS = {"subject_id", "subject", "subjects", "labels", "y", "rows", "patient_id", "person",
                  "participant", "participants", "patno", "ref", "clip_ref", "rating", "label"}  # fmt: skip
# Never inside a bundle: data tables, raw inputs, or pickles (loading a pickle runs code).
FORBIDDEN_SUFFIXES = {".csv", ".tsv", ".parquet", ".xlsx", ".nii", ".gz", ".dcm", ".wav", ".mp4",
                      ".mov", ".pkl", ".pickle", ".joblib", ".npy", ".npz", ".pt", ".pth"}  # fmt: skip
README = """PD-XAI {module} model - version {version}
Screening and triage only. Not a diagnostic device. Share only on the team drive; never commit.

Install:
1. Unzip at the repository root:  unzip {zip_name} -d <repo root>
   (the paths start with data/models/, which git ignores).
2. The bundle is checked against its SHA-256 hashes every time it loads. A changed or partly
   copied file is refused. MANIFEST.json lists the hash of every file in this zip.
3. Python packages: the versions the bundle was built with are in its manifest.json.

Contents: model parameters and a model card only. No recordings, videos, scans, per-person
features, labels or file names.
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def person_keys(o: object) -> set[str]:
    found: set[str] = set()
    if isinstance(o, dict):
        for k, v in o.items():
            if str(k).lower() in FORBIDDEN_KEYS:
                found.add(str(k))
            found |= person_keys(v)
    elif isinstance(o, list):
        for v in o:
            found |= person_keys(v)
    return found


def verify_mri(folder: Path) -> tuple[dict, list[str]]:
    """Refuse an MRI bundle that is incomplete, changed, or holds person-level data."""
    missing = [f for f in MRI_FILES if not (folder / f).is_file()]
    if missing:
        raise SystemExit(f"{folder.name}: missing {', '.join(missing)}")
    body = (folder / "bundle.json").read_text()
    expected = json.loads((folder / "manifest.json").read_text())["bundle_sha256"]
    if hashlib.sha256(body.encode()).hexdigest() != expected:
        raise SystemExit(f"{folder.name}: bundle.json does not match its manifest hash")
    data = json.loads(body)
    found = person_keys(data)
    if found:
        raise SystemExit(
            f"{folder.name}: bundle has person-level keys {sorted(found)}; not exporting"
        )
    return {"git_commit": data.get("commit", ""), "created": data.get("created", "")}, list(
        MRI_FILES
    )


def verify_listed(folder: Path) -> tuple[dict, list[str]]:
    """Refuse a pdxai-bundle/1 bundle whose files are not exactly the listed, unchanged ones,
    or that holds a data table, a raw input, a pickle or person-level keys."""
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("schema") != "pdxai-bundle/1" or manifest.get("version") != folder.name:
        raise SystemExit(f"{folder.name}: not a pdxai-bundle/1 bundle of this version")
    if manifest.get("contains_person_level_data") is not False:
        raise SystemExit(
            f"{folder.name}: the manifest does not state it holds no person-level data"
        )
    listed: dict[str, str] = manifest["files"]
    present = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    present.discard("manifest.json")
    if present != set(listed):
        raise SystemExit(f"{folder.name}: files differ from the manifest "
                         f"(unlisted {sorted(present - set(listed))}, missing {sorted(set(listed) - present)})")  # fmt: skip
    for name, digest in listed.items():
        if Path(name).suffix.lower() in FORBIDDEN_SUFFIXES:
            raise SystemExit(
                f"{folder.name}: {name} is a data table, raw input or pickle; not exporting"
            )
        if sha256(folder / name) != digest:
            raise SystemExit(f"{folder.name}: {name} does not match its manifest hash")
        if name.endswith(".json"):
            found = person_keys(json.loads((folder / name).read_text()))
            if found:
                raise SystemExit(f"{folder.name}: {name} has person-level keys {sorted(found)}")
    info = {"git_commit": manifest.get("git_commit", ""), "created": manifest.get("created_at", "")}
    return info, ["manifest.json", *sorted(listed)]


def export(module: str, version: str | None, out: Path) -> Path:
    root = MODELS / module
    pointer = POINTER.get(module, "CURRENT")
    if not version:
        if not (root / pointer).is_file():
            raise SystemExit(f"no data/models/{module}/{pointer}; build a bundle first")
        version = (root / pointer).read_text().strip()
    folder = root / version
    if not folder.is_dir():
        raise SystemExit(f"no bundle at {folder}")
    info, files = verify_listed(folder) if module in LISTED else verify_mri(folder)

    entries = {f"data/models/{module}/{version}/{f}": folder / f for f in files}
    zip_name = f"pdxai_models_{module}_{version}.zip"
    readme = README.format(module=module, version=version, zip_name=zip_name)
    extra = {f"data/models/{module}/{pointer}": version + "\n"}
    if module in LISTED:
        extra["README.txt"] = readme
    manifest = {
        "schema": "pdxai-models/1",
        "module": module,
        "version": version,
        **info,
        "files": {name: sha256(p) for name, p in entries.items()}
        | {name: hashlib.sha256(text.encode()).hexdigest() for name, text in extra.items()},
        "note": "Unzip at the repo root. Model parameters only. Share privately, never commit.",
    }
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / zip_name
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
        for name, p in entries.items():
            z.write(p, name)
        for name, text in extra.items():
            z.writestr(name, text)
    with zipfile.ZipFile(zip_path) as z:  # the zip must hold exactly what was checked
        names = set(z.namelist()) - {"MANIFEST.json"}
        bad = [n for n in names if n not in manifest["files"]
               or hashlib.sha256(z.read(n)).hexdigest() != manifest["files"][n]]  # fmt: skip
        if bad or names != set(manifest["files"]):
            zip_path.unlink()
            raise SystemExit(f"the written zip does not match its manifest: {bad}")

    print(f"wrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB, {len(names)} files)")
    shown = list(manifest["files"].items())
    for name, digest in shown[:6]:
        print(f"  {digest[:12]}  {name}")
    if len(shown) > 6:
        print(f"  ... and {len(shown) - 6} more, all listed in MANIFEST.json")
    return zip_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--module", action="append", help="module to export (repeatable; default: all)")
    ap.add_argument("--version", help="bundle version (default: the one named in the pointer)")
    ap.add_argument("--out", type=Path, default=Path.cwd(), help="folder for the zips")
    args = ap.parse_args(argv)

    modules = args.module or sorted(
        p.name for p in MODELS.glob("*") if (p / POINTER.get(p.name, "CURRENT")).is_file()
    )
    if not modules:
        raise SystemExit("no model bundles under data/models; build one first")
    if args.version and len(modules) != 1:
        raise SystemExit("--version needs exactly one --module")
    for module in modules:
        export(module, args.version, args.out)
    print("Teammates: unzip at the repo root. Share privately, never through git.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
