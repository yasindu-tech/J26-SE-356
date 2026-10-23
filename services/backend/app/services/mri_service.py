"""MRI analyses: accept an upload, start the analysis, report progress, serve the result.

Flow: create_analysis saves the zip under data/backend/mri/<id>/ and hands it to the
runner. The runner's process writes status.json while it works and result.json at the
end. Reads (status, result) pick those files up, store the result in the database and
delete the files, so nothing about the upload is left on disk once it has been read.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
import zipfile
from pathlib import Path
from typing import BinaryIO

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import (
    InvalidUploadError,
    MriAnalysisNotFoundError,
    MriResultNotReadyError,
    NoDicomInUploadError,
    UploadTooLargeError,
)
from app.core.time import utc_now
from app.models.audit_log import AuditLogEntry
from app.models.clinician import Clinician
from app.models.mri_analysis import MriAnalysis
from app.schemas.mri import FINAL_STATUSES, MriAnalysisStatus
from app.workers.mri_runner import MriJob, MriRunner

STEPS = ["convert", "t1", "dti", "features", "score", "explain"]  # predict_scan.STEPS
CHUNK = 1024 * 1024
DICOM_MAGIC_OFFSET = 128  # a DICOM file starts with a 128-byte preamble, then "DICM"
DICOM_MAGIC = b"DICM"
DICOM_SCAN_LIMIT = 500  # files looked at before giving up; one hit is enough
IGNORED_NAMES = ("__MACOSX/", ".DS_Store", "Thumbs.db")


def _folder(settings: Settings, analysis_id: uuid.UUID) -> Path:
    return settings.mri_work_dir / str(analysis_id)


def _audit(db: Session, clinician: Clinician, action: str, analysis_id: uuid.UUID) -> None:
    db.add(AuditLogEntry(clinician_id=clinician.id, action=action, detail=str(analysis_id)))


def save_upload(src: BinaryIO, dest: Path, max_bytes: int) -> None:
    """Copy the uploaded zip to disk in chunks, refusing anything too large or not a zip."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with dest.open("wb") as out:
        while chunk := src.read(CHUNK):
            size += len(chunk)
            if size > max_bytes:
                out.close()
                dest.unlink(missing_ok=True)
                raise UploadTooLargeError(f"The upload is over {max_bytes // CHUNK} MB")
            out.write(chunk)
    if size == 0 or not zipfile.is_zipfile(dest):
        dest.unlink(missing_ok=True)
        raise InvalidUploadError
    if not looks_like_dicom_zip(dest):
        dest.unlink(missing_ok=True)
        raise NoDicomInUploadError


def looks_like_dicom_zip(path: Path) -> bool:
    """True when the zip holds at least one file that is a DICOM file.

    A file counts if bytes 128-131 read "DICM" (the DICOM preamble), or if its name ends in
    ".dcm" (some writers omit the preamble; dcm2niix decides later). Only the first 132 bytes
    of each file are read, and at most DICOM_SCAN_LIMIT files, so a large zip is not unpacked.
    This is a cheap guard against a wrong file, not a check that the series are usable or
    that they belong to one person: the analysis still decides that.
    """
    with zipfile.ZipFile(path) as z:
        scanned = 0
        for info in z.infolist():
            name = info.filename
            if info.is_dir() or any(part in name for part in IGNORED_NAMES):
                continue
            if name.lower().endswith(".dcm"):
                return True
            with z.open(info) as f:
                head = f.read(DICOM_MAGIC_OFFSET + len(DICOM_MAGIC))
            if head[DICOM_MAGIC_OFFSET:] == DICOM_MAGIC:
                return True
            scanned += 1
            if scanned >= DICOM_SCAN_LIMIT:
                return False
    return False


def create_analysis(
    db: Session,
    clinician: Clinician,
    upload: BinaryIO,
    age: float,
    sex: str,
    runner: MriRunner,
    settings: Settings,
    eddy_profile: str = "full",
    subject_label: str | None = None,
) -> MriAnalysis:
    row = MriAnalysis(
        id=uuid.uuid4(),
        clinician_id=clinician.id,
        status="queued",
        age=age,
        sex=sex,
        eddy_profile=eddy_profile,
        subject_label=subject_label,
    )
    folder = _folder(settings, row.id)
    try:
        save_upload(upload, folder / "upload.zip", settings.mri_max_upload_mb * CHUNK)
    except (InvalidUploadError, UploadTooLargeError):
        shutil.rmtree(folder, ignore_errors=True)
        raise
    db.add(row)
    _audit(db, clinician, "mri_upload", row.id)
    db.commit()
    db.refresh(row)
    runner.submit(
        MriJob(analysis_id=row.id, folder=folder, age=age, sex=sex, eddy_profile=eddy_profile)
    )
    return row


def _own(db: Session, clinician: Clinician, analysis_id: uuid.UUID) -> MriAnalysis:
    row = db.get(MriAnalysis, analysis_id)
    if row is None or row.clinician_id != clinician.id:
        raise MriAnalysisNotFoundError
    return row


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def refresh_from_disk(db: Session, row: MriAnalysis, settings: Settings) -> dict | None:
    """Move a finished result from disk into the database. Returns live progress, if any."""
    if row.status in FINAL_STATUSES:
        return None
    folder = _folder(settings, row.id)
    result = _read_json(folder / "result.json")
    if result is not None:
        status = result.get("status")
        row.status = status if status in FINAL_STATUSES else "failed"
        row.error = (str(result.get("error"))[:500]) if result.get("error") else None
        row.result_json = json.dumps(result)
        row.finished_at = utc_now()
        db.commit()
        shutil.rmtree(folder, ignore_errors=True)
        return None
    progress = _read_json(folder / "status.json")
    stale_after = settings.mri_timeout_hours * 3600
    if progress is not None:
        if time.time() - float(progress.get("t", 0)) > stale_after:
            row.status, row.error, row.finished_at = "failed", "the analysis stopped", utc_now()
            db.commit()
            return None
        if row.status != "running":
            row.status = "running"
            db.commit()
        return progress
    if not folder.exists():  # nothing on disk and no result: the backend lost the job
        row.status, row.error, row.finished_at = "failed", "the analysis was interrupted", utc_now()
        db.commit()
    return None


def _status(row: MriAnalysis, progress: dict | None) -> MriAnalysisStatus:
    result = json.loads(row.result_json) if row.result_json else {}
    done = list((progress or {}).get("done", []))
    if row.status == "scored":
        done = STEPS
    return MriAnalysisStatus(
        id=row.id,
        status=row.status,  # type: ignore[arg-type]
        eddy_profile=row.eddy_profile,  # type: ignore[arg-type]
        subject_label=row.subject_label,
        created_at=row.created_at,
        finished_at=row.finished_at,
        steps=STEPS,
        done=done,
        current_step=(progress or {}).get("current"),
        note=str((progress or {}).get("note", "")),
        error=row.error,
        abstain_reason=result.get("abstain_reason"),
    )


def get_status(
    db: Session, clinician: Clinician, analysis_id: uuid.UUID, settings: Settings
) -> MriAnalysisStatus:
    row = _own(db, clinician, analysis_id)
    return _status(row, refresh_from_disk(db, row, settings))


def list_analyses(
    db: Session, clinician: Clinician, settings: Settings, limit: int, offset: int
) -> tuple[list[MriAnalysisStatus], int]:
    q = db.query(MriAnalysis).filter(MriAnalysis.clinician_id == clinician.id)
    total = q.count()
    rows = q.order_by(MriAnalysis.created_at.desc()).offset(offset).limit(limit).all()
    return [_status(r, refresh_from_disk(db, r, settings)) for r in rows], total


def get_result(
    db: Session, clinician: Clinician, analysis_id: uuid.UUID, settings: Settings
) -> dict:
    row = _own(db, clinician, analysis_id)
    refresh_from_disk(db, row, settings)
    if row.status not in FINAL_STATUSES or row.result_json is None:
        raise MriResultNotReadyError
    _audit(db, clinician, "mri_result_view", row.id)
    db.commit()
    return json.loads(row.result_json)
