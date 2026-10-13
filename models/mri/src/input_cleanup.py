"""Delete a person's input images once their results are safely kept.

Disk space, not time, is the limit for the DTI batches. When a person has passed every
check and their final outputs exist, the big inputs are no longer needed:

* the raw DICOM folders of the series that were converted, and
* the converted NIfTI files (.nii.gz, .bval, .bvec, .json) of that person's DTI series.

This is opt-in (``--delete-inputs`` on ``preprocess_dti.py``) and deliberately narrow:

* Only called for a person whose status is ok and whose final outputs exist.
* Raw DICOM: only the ``I<image id>`` folders of series listed as converted in the
  conversion table, only under a ``--raw-dir`` you pass, and only if that folder is inside
  ``data/raw``. A series that was never converted is left alone. Symlinks are never followed.
* NIfTI: only the four file types of that person's converted DTI series, only inside
  ``data/processed/nifti``. T1 files in the same folders have different names and are kept.
* People who failed keep all their inputs, so the failure can be looked at or rerun.
* ``dry_run=True`` lists what would go and deletes nothing.

After this, rerunning a finished person still works (their results are reused), but
``--redo`` on them cannot, because the input is gone. Download it again for that.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

NIFTI_SUFFIXES = (".nii.gz", ".bval", ".bvec", ".json")
REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = REPO_ROOT / "data" / "raw"
NIFTI_ROOT = REPO_ROOT / "data" / "processed" / "nifti"


class UnsafeDeleteError(ValueError):
    """A path is outside the allowed folders, or is a symlink. Nothing was deleted."""


@dataclass
class SeriesRef:
    image_id: str
    nifti_path: Path  # the converted .nii.gz of this series


@dataclass
class CleanupReport:
    bytes_freed: int = 0
    removed: list[str] = field(default_factory=list)  # names only, never full paths with IDs

    def add(self, name: str, size: int) -> None:
        self.removed.append(name)
        self.bytes_freed += size


def _inside(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _check(path: Path, root: Path) -> None:
    if path.is_symlink():
        raise UnsafeDeleteError(f"refusing to delete a symlink: {path.name}")
    if not _inside(path, root):
        raise UnsafeDeleteError(f"{path.name} is outside {root.name}")


def dicom_folders(
    subject_id: str, series: Iterable[SeriesRef], raw_dirs: Iterable[Path], raw_root: Path
) -> list[Path]:
    """The I<image id> folders of converted series, for this person only."""
    found = []
    for raw in raw_dirs:
        if not _inside(raw, raw_root):
            raise UnsafeDeleteError(f"--raw-dir {raw.name} is not inside {raw_root}")
        person = raw / subject_id
        if not person.is_dir():
            continue
        for s in series:
            # layout: <raw>/<subject>/<series name>/<date>/I<image id>
            found += [p for p in person.glob(f"*/*/I{s.image_id}") if p.is_dir() or p.is_symlink()]
    return found


def nifti_files(series: Iterable[SeriesRef]) -> list[Path]:
    files = []
    for s in series:
        base = s.nifti_path.name.removesuffix(".nii.gz")
        for suffix in NIFTI_SUFFIXES:
            f = s.nifti_path.with_name(base + suffix)
            if f.exists() or f.is_symlink():
                files.append(f)
    return files


def _prune_empty(folder: Path, stop: Path) -> None:
    """Remove empty folders from ``folder`` upward, never above ``stop`` or ``stop`` itself."""
    folder = folder.resolve()
    stop = stop.resolve()
    while folder != stop and stop in folder.parents:
        try:
            folder.rmdir()
        except OSError:  # not empty: something else lives here, so we leave it
            return
        folder = folder.parent


def delete_inputs(
    subject_id: str,
    series: list[SeriesRef],
    raw_dirs: list[Path],
    dry_run: bool = False,
    raw_root: Path | None = None,
    nifti_root: Path | None = None,
) -> CleanupReport:
    """Delete the inputs of one finished person. Call only after their results are kept.

    All paths are checked before the first delete, so one unsafe path means nothing is removed.
    """
    raw_root = raw_root or RAW_ROOT
    nifti_root = nifti_root or NIFTI_ROOT
    dicoms = dicom_folders(subject_id, series, raw_dirs, raw_root)
    niftis = nifti_files(series)
    for p in dicoms:
        _check(p, raw_root)
    for p in niftis:
        _check(p, nifti_root)

    report = CleanupReport()
    for p in dicoms:
        report.add(f"dicom:{p.name}", _size(p))
        if not dry_run:
            shutil.rmtree(p)
            _prune_empty(p.parent, p.parents[2])  # up to, not including, <raw>/<subject>
    for p in niftis:
        report.add(f"nifti:{p.name.rsplit('_', 1)[-1]}", _size(p))
        if not dry_run:
            p.unlink()
    if not dry_run:
        for root in {p.parent for p in niftis}:
            if root.is_dir() and not any(root.iterdir()):
                root.rmdir()
        for raw in raw_dirs:
            person = raw / subject_id
            if person.is_dir():
                _prune_empty(person, raw)
    return report
