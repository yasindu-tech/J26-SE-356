"""Export the trained MRI model as one zip that teammates can unpack and use.

    python scripts/export_models.py                 # the version named in data/models/mri/LATEST
    python scripts/export_models.py --version 20261022_220745_b948f12 --out ~/Desktop

Writes pdxai_models_mri_<version>.zip containing data/models/mri/<version>/ (bundle.json,
manifest.json, model_card.json), the LATEST pointer, and a top-level MANIFEST.json listing
every file with its SHA-256.

To use it, a teammate unzips it at the repository root (the zip paths start with data/models/):

    unzip pdxai_models_mri_<version>.zip -d <repo root>

The MRI bundle is checked against its own hash when it loads, so a changed or partly
copied file is refused. Share the zip privately (team drive), never through git: the model
is derived from PPMI data under a data use agreement. This script refuses to write a zip
that is not the bundle it checked, and the bundle holds model parameters only: no scans,
labels, or subject-level rows (see the check below).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUNDLES = REPO / "data" / "models" / "mri"
FILES = ("bundle.json", "manifest.json", "model_card.json")
# Keys a model bundle must not carry. A bundle with any of these holds person-level data.
FORBIDDEN_KEYS = {"subject_id", "subject", "subjects", "labels", "y", "rows", "patient_id", "person"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(folder: Path) -> dict:
    """Refuse a bundle that is incomplete, changed, or holds person-level data."""
    missing = [f for f in FILES if not (folder / f).is_file()]
    if missing:
        raise SystemExit(f"{folder.name}: missing {', '.join(missing)}")
    body = (folder / "bundle.json").read_text()
    expected = json.loads((folder / "manifest.json").read_text())["bundle_sha256"]
    if hashlib.sha256(body.encode()).hexdigest() != expected:
        raise SystemExit(f"{folder.name}: bundle.json does not match its manifest hash")
    found: set[str] = set()

    def walk(o: object) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                if str(k).lower() in FORBIDDEN_KEYS:
                    found.add(str(k))
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    data = json.loads(body)
    walk(data)
    if found:
        raise SystemExit(f"{folder.name}: bundle has person-level keys {sorted(found)}; not exporting")
    return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", help="bundle version (default: the one named in LATEST)")
    ap.add_argument("--out", type=Path, default=Path.cwd(), help="folder for the zip")
    args = ap.parse_args(argv)

    version = args.version
    if not version:
        latest = BUNDLES / "LATEST"
        if not latest.is_file():
            raise SystemExit("no data/models/mri/LATEST; train a bundle first (mri_bundle.py)")
        version = latest.read_text().strip()
    folder = BUNDLES / version
    if not folder.is_dir():
        raise SystemExit(f"no bundle at {folder}")
    data = verify(folder)

    entries = {f"data/models/mri/{version}/{f}": folder / f for f in FILES}
    manifest = {
        "schema": "pdxai-models/1",
        "module": "mri",
        "version": version,
        "git_commit": data.get("commit", ""),
        "created": data.get("created", ""),
        "files": {name: sha256(p) for name, p in entries.items()},
        "note": "Unzip at the repo root. Model parameters only. Share privately, never commit.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    zip_path = args.out / f"pdxai_models_mri_{version}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
        for name, p in entries.items():
            z.write(p, name)
        z.writestr("data/models/mri/LATEST", version + "\n")

    print(f"wrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
    for name, digest in manifest["files"].items():
        print(f"  {digest[:12]}  {name}")
    print("Teammates: unzip it at the repo root. Share privately, never through git.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
