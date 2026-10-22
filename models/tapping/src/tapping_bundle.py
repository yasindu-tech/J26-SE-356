"""The tapping models packed for another laptop: data/models/tapping/<version>/ (model-bundle-export).

`build` copies, for each target, the demo model and its 25 cross-validation fold models (both
checked against data/processed by resources.py first), the hand landmarker, 101 percentiles per
Set A feature and a model card, then writes manifest.json (SHA-256 of every other file) and
data/models/tapping/CURRENT. It never copies features.csv or any per-clip row: the percentiles
replace the research-clip table when a new clip is placed among the research clips.
`load` refuses a bundle with a missing, extra or changed file, a pickle, or a forbidden feature.

    python models/tapping/src/tapping_bundle.py build    # a new version; CURRENT points to it
    python models/tapping/src/tapping_bundle.py check    # loads CURRENT and prints its version

Share the bundle only privately (scripts/export_models.py --module tapping); never commit it.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "demo"))

import data_access as da  # noqa: E402
import hand_landmarks as hl  # noqa: E402
import lightgbm  # noqa: E402
import numpy as np  # noqa: E402
import resources as res  # noqa: E402
import scoring as sc  # noqa: E402

REPO_ROOT = HERE.parents[2]
BUNDLES = REPO_ROOT / "data" / "models" / "tapping"
ENV_ROOT = "TAPPING_BUNDLE_ROOT"  # tests point the scorer at a temporary bundle folder
SCHEMA = "pdxai-bundle/1"
MODULE = "tapping"
MANIFEST = "manifest.json"
POINTER = "CURRENT"
REFERENCE = "reference_percentiles.json"
CARD = "model_card.json"
LANDMARKER = "hand_landmarker.task"
N_PERCENTILES = 101
VERSION_RE = re.compile(r"^\d{8}_\d{6}_[0-9a-f]{7}$")
PICKLE_SUFFIXES = {".pkl", ".pickle", ".joblib", ".npy", ".npz", ".pt", ".pth"}
# CLAUDE.md 3.1: never a model input. Matched anywhere in a lower-case feature name.
FORBIDDEN_FEATURE = re.compile(
    r"updrs|hoehn|yahr|rating|clinician|impression|medicat|levodopa|ledd|on_?off|datscan"
    r"|dat_?spect|\bsbr\b|qsm|diagnos"
)
# A JSON key with one of these names holds person-level data; no bundle file may have one.
PERSON_KEYS = {"ref", "clip_ref", "participant", "participants", "subject", "subject_id",
               "patno", "patient_id", "rating", "label", "labels", "y", "rows", "path", "file"}  # fmt: skip


class BundleError(ValueError):
    """The bundle is missing, incomplete, changed or not allowed (message safe to show)."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_feature_names(names: list[str]) -> None:
    bad = [n for n in names if FORBIDDEN_FEATURE.search(n.lower())]
    if bad:
        raise BundleError(f"forbidden model input features: {bad}")


def person_keys(obj: Any) -> set[str]:
    """Keys anywhere in a JSON object whose name marks person-level data."""
    found: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in PERSON_KEYS:
                found.add(str(k))
            found |= person_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            found |= person_keys(v)
    return found


# --------------------------------------------------------------------------- percentiles


def quantiles(values: np.ndarray) -> list[float]:
    """101 percentiles (0, 1, ..., 100) of the finite values; p0 and p100 are the min and max."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        raise BundleError("no finite reference values")
    return [float(x) for x in np.percentile(v, np.arange(N_PERCENTILES))]


def percentile_of(value: float, q: list[float]) -> float:
    """Where a value falls among the reference clips, from their 101 percentiles; ties average,
    as scipy's percentileofscore(kind="mean") does on the full table."""
    arr = np.asarray(q, dtype=float)
    grid = np.linspace(0.0, 100.0, arr.size)
    if value < arr[0]:
        return 0.0
    if value > arr[-1]:
        return 100.0
    lo, hi = np.searchsorted(arr, value, "left"), np.searchsorted(arr, value, "right")
    if hi > lo:  # equal to one or more percentiles
        return float(grid[lo:hi].mean())
    a, b = arr[lo - 1], arr[lo]
    return float(grid[lo - 1] + (value - a) / (b - a) * (grid[lo] - grid[lo - 1]))


def context(features: dict[str, float], reference: dict[str, list[float]]) -> dict[str, Any]:
    """sc.context with the bundle's percentiles in place of the research-clip table."""
    out: dict[str, dict[str, Any]] = {}
    for c in sc.SET_A:
        v, q = features[c], reference[c]
        if not np.isfinite(v):  # a missing value has no percentile and is not 'outside'
            out[c] = {"percentile": float("nan"), "in_range": True}
            continue
        out[c] = {"percentile": percentile_of(v, q), "in_range": bool(q[0] <= v <= q[-1])}
    return out


# --------------------------------------------------------------------------- loading


@dataclass
class TappingBundle:
    folder: Path
    manifest: dict[str, Any]
    reference: dict[str, list[float]]
    card: dict[str, Any]
    _models: dict[str, Any] = field(default_factory=dict)
    _folds: dict[str, list[Any]] = field(default_factory=dict)

    @property
    def version(self) -> str:
        return str(self.manifest["version"])

    @property
    def landmarker(self) -> Path:
        return self.folder / LANDMARKER

    def _target(self, target: str) -> str:
        if target not in self.manifest["targets"]:
            raise BundleError(f"the bundle has no model for target {target}")
        return target

    def model(self, target: str) -> lightgbm.Booster:
        if target not in self._models:
            path = self.folder / res.MODEL_FILE.format(target=self._target(target))
            self._models[target] = lightgbm.Booster(model_file=str(path))
        return self._models[target]

    def folds(self, target: str) -> list[lightgbm.Booster]:
        if target not in self._folds:
            folder = self.folder / "folds" / self._target(target)
            self._folds[target] = [
                lightgbm.Booster(model_file=str(folder / res.FOLD_FILE.format(repeat=r, fold=f)))
                for r in range(5) for f in range(5)
            ]  # fmt: skip
        return self._folds[target]

    def model_sha8(self, target: str) -> str:
        """The same value resources.model_sha8 gives for the original file."""
        name = res.MODEL_FILE.format(target=self._target(target))
        return self.manifest["files"][name][:8]

    def folds_sha8(self, target: str) -> str:
        name = f"folds/{self._target(target)}/{res.FOLD_META}"
        return self.manifest["files"][name][:8]

    def features_sha8(self, target: str) -> str:
        """sha8 of the features.csv the model was trained from (recorded, not shipped)."""
        meta = json.loads((self.folder / res.META_FILE.format(target=target)).read_text())
        return str(meta["inputs_sha8"]["features"])

    def auroc(self, target: str) -> dict[str, float]:
        return dict(self.card["targets"][self._target(target)]["cv_auroc"])


def bundle_root() -> Path:
    env = os.environ.get(ENV_ROOT)
    return Path(env) if env else BUNDLES


def current_version(root: Path = BUNDLES) -> str | None:
    pointer = root / POINTER
    return pointer.read_text().strip() if pointer.is_file() else None


def installed(root: Path = BUNDLES) -> bool:
    return current_version(root) is not None


def verify(folder: Path) -> dict[str, Any]:
    """The manifest of a bundle folder whose files are exactly the listed ones, unchanged."""
    path = folder / MANIFEST
    if not path.is_file():
        raise BundleError(f"tapping bundle {folder.name}: no {MANIFEST}")
    manifest: dict[str, Any] = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("module") != MODULE:
        raise BundleError(f"tapping bundle {folder.name}: not a {SCHEMA} {MODULE} bundle")
    if manifest.get("version") != folder.name:
        raise BundleError(f"tapping bundle {folder.name}: manifest names another version")
    listed: dict[str, str] = manifest.get("files", {})
    present = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    present.discard(MANIFEST)
    if present - set(listed):
        raise BundleError(
            f"tapping bundle {folder.name}: unlisted files {sorted(present - set(listed))}"
        )
    if set(listed) - present:
        raise BundleError(f"tapping bundle {folder.name}: missing {sorted(set(listed) - present)}")
    pickles = [n for n in listed if Path(n).suffix.lower() in PICKLE_SUFFIXES]
    if pickles:
        raise BundleError(f"tapping bundle {folder.name}: pickled files are refused {pickles}")
    for name, digest in listed.items():
        if sha256_file(folder / name) != digest:
            raise BundleError(f"tapping bundle {folder.name}: {name} does not match its hash")
    check_feature_names(list(manifest.get("feature_names", [])))
    if manifest.get("feature_names") != sc.SET_A:
        raise BundleError(f"tapping bundle {folder.name}: features differ from this code's Set A")
    return manifest


def load(root: Path = BUNDLES, version: str | None = None) -> TappingBundle:
    """The bundle named by CURRENT (or ``version``), every file checked before use."""
    version = version or current_version(root)
    if not version:
        raise BundleError("no tapping model bundle installed (data/models/tapping/CURRENT)")
    folder = root / version
    if not folder.is_dir():
        raise BundleError(f"tapping bundle {version} is not installed")
    manifest = verify(folder)
    wanted = manifest["libraries"]["lightgbm"]
    if wanted != lightgbm.__version__:
        raise BundleError(f"tapping bundle needs lightgbm {wanted}, found {lightgbm.__version__}")
    if sha256_file(folder / LANDMARKER) != hl.EXPECTED_MODEL_SHA256:
        raise BundleError("the bundled hand landmarker is not the expected file")
    reference = json.loads((folder / REFERENCE).read_text())["percentiles"]
    if sorted(reference) != sorted(sc.SET_A) or any(
        len(q) != N_PERCENTILES for q in reference.values()
    ):
        raise BundleError("the bundled reference percentiles are incomplete")
    card = json.loads((folder / CARD).read_text())
    return TappingBundle(folder, manifest, reference, card)


# --------------------------------------------------------------------------- building


def git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"], cwd=REPO_ROOT,
                         capture_output=True, text=True, check=False)  # fmt: skip
    commit = out.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{7}", commit):
        raise BundleError("not in a git checkout: the bundle must name the code commit")
    return commit


def library_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": platform.python_version()}
    for name in ("lightgbm", "numpy", "shap", "mediapipe", "scipy", "pandas"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not installed"
    return out


def model_card(processed: Path, metas: dict[str, dict[str, Any]], n_reference: dict[str, int],
               created: str) -> dict[str, Any]:  # fmt: skip
    """Everything a reader of a result needs; every number is copied from a script's output."""
    targets = {}
    for t, meta in metas.items():
        targets[t] = {
            "definition": {"ge2": "clinician finger-tapping rating >= 2",
                           "ge1": "clinician finger-tapping rating >= 1"}[t],
            "cv_auroc": sc.cv_auroc(t, processed),
            "cv_auroc_source": "data/processed/cv/results/results_table.csv (lgbm A, outer CV, "
                               "participant bootstrap); not a metric of this fitted model",
            "n_training_clips": meta["n_clips"],
            "params": meta["params"],
            "selection_rule": meta["selection_rule"],
        }  # fmt: skip
    return {
        "module": MODULE,
        "created": created,
        "intended_use": "Screening and triage only. Not a diagnostic device. A score above the "
        "threshold means prioritised referral; below it means not prioritised.",
        "dataset": "HUBU-FIS (Zenodo, DOI 10.5281/zenodo.17738775)",
        "input": "one smartphone video of finger tapping (mp4, mov or m4v), one hand",
        "features": {c: {"plain": sc.PLAIN[c][0], "unit": sc.PLAIN[c][1]} for c in sc.SET_A},
        "targets": targets,
        "threshold": sc.THRESHOLD,
        "threshold_note": "fixed study threshold; scores are not calibrated to screening prevalence",
        "band": "5-95% of the 25 cross-validation fold models' scores; this is the spread of "
        "refits, not a bootstrap confidence interval",
        "explanation": "TreeSHAP, tree-path-dependent, raw log-odds; contributions to the "
        "model's score, never causes",
        "reference": {
            "what": "101 percentiles per feature over the research clips",
            "n_clips": n_reference,
            "note": "p0 and p100 equal the smallest and largest clip value of each "
            "feature; no clip row, identifier or label is included",
        },  # fmt: skip
        "known_limits": [
            "The label is a clinician's MDS-UPDRS finger-tapping rating, used only as the "
            "target; the features are measured from the same movement that rating scores.",
            "The demo model is fitted on all clips; its own score is never a reported metric. "
            "The AUROC shown is the outer cross-validation estimate of the same recipe.",
            "In HUBU-FIS some video formats occur only in one group (see "
            "models/tapping/README.md): resolution or frame rate may act as a shortcut.",
            "Trained on people already diagnosed versus controls: no evidence for early or "
            "prodromal detection.",
        ],
    }


def leaked_identifiers(folder: Path, processed: Path) -> list[str]:
    """Clip refs or participant ids of the research table that appear in any bundle text file.
    Only ids with a letter are searched, so digits inside a model's numbers cannot match."""
    import pandas as pd

    table = pd.read_csv(processed / "features.csv", usecols=["ref", "participant"], dtype=str)
    ids = {x for col in ("ref", "participant") for x in table[col].dropna()
           if re.search(r"[a-z]", x, re.I)}  # fmt: skip
    hits = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix != ".task":
            text = p.read_text(errors="ignore")
            hits += [f"{p.name}" for x in ids if x in text]
    return sorted(set(hits))


def build(processed: Path | None = None, root: Path = BUNDLES, now: datetime.datetime | None = None,
          commit: str | None = None) -> Path:  # fmt: skip
    """Write a new bundle version and point CURRENT to it; refuses a downgrade or an overwrite."""
    processed = da.processed_root() if processed is None else processed
    now = now or datetime.datetime.now()
    version = f"{now:%Y%m%d_%H%M%S}_{commit or git_commit()}"
    previous = current_version(root)
    if previous and version <= previous:
        raise BundleError(f"version {version} is not newer than the installed {previous}")
    out = root / version
    if out.exists():
        raise BundleError(f"bundle {version} already exists")
    tmp = root / f".{version}.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        metas: dict[str, dict[str, Any]] = {}
        trained: dict[str, str] = {}
        for target in da.TARGETS:
            res.load_demo_model(target, processed)  # every check of the demo itself
            res.load_fold_models(target, processed)
            model, meta = res.model_paths(target, processed)
            shutil.copy2(model, tmp / model.name)
            shutil.copy2(meta, tmp / meta.name)
            metas[target] = json.loads(meta.read_text())
            stamp = datetime.datetime.fromtimestamp(model.stat().st_mtime)
            trained[target] = stamp.isoformat(timespec="seconds")  # the meta has no date
            src = res.fold_dir(target, processed)
            dst = tmp / "folds" / target
            dst.mkdir(parents=True)
            names = [res.FOLD_META] + [res.FOLD_FILE.format(repeat=r, fold=f)
                                       for r in range(5) for f in range(5)]  # fmt: skip
            for name in names:
                shutil.copy2(src / name, dst / name)
        handle = res.make_landmarker()  # checks the landmarker hash
        shutil.copy2(handle.model_path, tmp / LANDMARKER)
        table = sc.load_reference(processed)  # Set A columns only
        percentiles = {c: quantiles(table[c].to_numpy(dtype=float)) for c in sc.SET_A}
        n_reference = {c: int(np.isfinite(table[c].to_numpy(dtype=float)).sum()) for c in sc.SET_A}
        (tmp / REFERENCE).write_text(json.dumps(
            {"what": "101 percentiles (0..100) per Set A feature", "percentiles": percentiles},
            indent=1))  # fmt: skip
        created = now.isoformat(timespec="seconds")
        (tmp / CARD).write_text(
            json.dumps(model_card(processed, metas, n_reference, created), indent=2)
        )
        for p in tmp.rglob("*.json"):
            keys = person_keys(json.loads(p.read_text()))
            if keys:
                raise BundleError(f"{p.name} has person-level keys {sorted(keys)}; not bundled")
        leaks = leaked_identifiers(tmp, processed)
        if leaks:
            raise BundleError(f"research clip identifiers found in {leaks}; not bundled")
        files = {p.relative_to(tmp).as_posix(): sha256_file(p)
                 for p in sorted(tmp.rglob("*")) if p.is_file()}  # fmt: skip
        manifest = {
            "schema": SCHEMA, "module": MODULE, "version": version, "git_commit": version[-7:],
            "trained_at": trained, "trained_at_basis": "model file time",
            "created_at": created, "files": files, "feature_names": sc.SET_A,
            "targets": list(da.TARGETS), "default_target": "ge2",
            "input_schema": {"input": "one finger-tapping video", "formats": sorted(res_suffixes()),
                             "max_upload_mb": max_upload_mb()},
            "libraries": library_versions(), "model_card": CARD,
            "contains_person_level_data": False,
            "intended_use": "Screening and triage only. Not a diagnostic device.",
        }  # fmt: skip
        (tmp / MANIFEST).write_text(json.dumps(manifest, indent=2))
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    load(root, version)  # the written bundle must load before CURRENT moves to it
    (root / POINTER).write_text(version + "\n")
    return out


def res_suffixes() -> set[str]:
    import upload_pipeline as up

    return set(up.SUFFIXES)


def max_upload_mb() -> int:
    import upload_pipeline as up

    return int(up.MAX_BYTES // (1024 * 1024))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("action", choices=("build", "check"))
    ap.add_argument("--root", type=Path, default=bundle_root())
    args = ap.parse_args(argv)
    try:
        if args.action == "build":
            print(f"wrote {build(root=args.root)}")
        b = load(args.root)
    except (BundleError, da.DemoDataError) as err:
        print(f"tapping bundle: {err}", file=sys.stderr)
        return 1
    print(f"tapping bundle {b.version}: {len(b.manifest['files'])} files, hashes OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
