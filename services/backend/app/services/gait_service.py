"""Gait analyses: accept a video, start the analysis, report its status, serve the result.

Flow: create_analysis saves the video under data/backend/gait/<id>/ (named by the analysis ID,
never by its own name) and hands it to the runner. The runner deletes the video when the
process ends and leaves result.json. Reads pick that file up, store the result in the database
and delete the folder, so nothing of the upload is left on disk once it has been read.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import BinaryIO

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import (
    GaitAnalysisNotFoundError,
    GaitResultNotReadyError,
    InvalidUploadError,
    OnePersonNotConfirmedError,
    UploadTooLargeError,
)
from app.core.time import utc_now
from app.models.audit_log import AuditLogEntry
from app.models.clinician import Clinician
from app.models.gait_analysis import GaitAnalysis
from app.schemas.gait import GAIT_VIDEO_TYPES, GaitAnalysisStatus
from app.schemas.mri import FINAL_STATUSES
from app.workers.gait_runner import GaitJob, GaitRunner

CHUNK = 1024 * 1024
NOT_A_VIDEO = f"The upload must be a video file ({', '.join(GAIT_VIDEO_TYPES)})"


def _folder(settings: Settings, analysis_id: uuid.UUID) -> Path:
    return settings.gait_work_dir / str(analysis_id)


def _audit(db: Session, clinician: Clinician, action: str, analysis_id: uuid.UUID) -> None:
    db.add(AuditLogEntry(clinician_id=clinician.id, action=action, detail=str(analysis_id)))


def video_suffix(filename: str | None) -> str:
    """The upload's extension if it is an accepted video type (its name is not kept)."""
    suffix = Path(filename or "").suffix.lower()
    if suffix not in GAIT_VIDEO_TYPES:
        raise InvalidUploadError(NOT_A_VIDEO)
    return suffix


def save_upload(src: BinaryIO, dest: Path, max_bytes: int) -> None:
    """Copy the uploaded video to disk in chunks, refusing an empty or too large file."""
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
    if size == 0:
        dest.unlink(missing_ok=True)
        raise InvalidUploadError("The uploaded video is empty")


def create_analysis(
    db: Session,
    clinician: Clinician,
    upload: BinaryIO,
    filename: str | None,
    one_person_confirmed: bool,
    runner: GaitRunner,
    settings: Settings,
    subject_label: str | None = None,
) -> GaitAnalysis:
    if not one_person_confirmed:
        raise OnePersonNotConfirmedError
    suffix = video_suffix(filename)
    row = GaitAnalysis(
        id=uuid.uuid4(),
        clinician_id=clinician.id,
        status="queued",
        one_person_confirmed=one_person_confirmed,
        subject_label=subject_label,
        created_at=utc_now(),
    )
    folder = _folder(settings, row.id)
    video = folder / f"{row.id}{suffix}"
    try:
        save_upload(upload, video, settings.gait_max_upload_mb * CHUNK)
    except (InvalidUploadError, UploadTooLargeError):
        shutil.rmtree(folder, ignore_errors=True)
        raise
    db.add(row)
    _audit(db, clinician, "gait_upload", row.id)
    db.commit()
    db.refresh(row)
    runner.submit(GaitJob(row.id, folder, video, one_person_confirmed))
    return row


def _own(db: Session, clinician: Clinician, analysis_id: uuid.UUID) -> GaitAnalysis:
    row = db.get(GaitAnalysis, analysis_id)
    if row is None or row.clinician_id != clinician.id:
        raise GaitAnalysisNotFoundError
    return row


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _finish(db: Session, row: GaitAnalysis, status: str, error: str | None) -> None:
    row.status, row.error, row.finished_at = status, error, utc_now()
    db.commit()


def refresh_from_disk(db: Session, row: GaitAnalysis, settings: Settings) -> None:
    """Move a finished result from disk into the database, or notice a lost analysis."""
    if row.status in FINAL_STATUSES:
        return
    folder = _folder(settings, row.id)
    result = _read_json(folder / "result.json")
    if result is not None:
        if "available" in result:  # a predict_video record
            status = "scored" if result["available"] else "abstained"
            error = None
        else:  # the runner's own failure record
            status, error = "failed", str(result.get("error") or "the gait analysis failed")[:500]
        row.result_json = json.dumps({**result, "status": status})
        _finish(db, row, status, error)
        shutil.rmtree(folder, ignore_errors=True)
        return
    started = _read_json(folder / "started.json")
    if started is not None:
        if time.time() - float(started.get("t", 0)) > settings.gait_timeout_minutes * 60 + 60:
            _finish(db, row, "failed", "the analysis stopped")
            shutil.rmtree(folder, ignore_errors=True)
        elif row.status != "running":
            row.status = "running"
            db.commit()
        return
    if not folder.exists():  # nothing on disk and no result: the backend lost the job
        _finish(db, row, "failed", "the analysis was interrupted")


def _status(row: GaitAnalysis) -> GaitAnalysisStatus:
    result = json.loads(row.result_json) if row.result_json else {}
    reasons = result.get("reasons") or []
    end = row.finished_at or utc_now()
    return GaitAnalysisStatus(
        id=row.id,
        status=row.status,  # type: ignore[arg-type]
        created_at=row.created_at,
        finished_at=row.finished_at,
        elapsed_s=max(0.0, (end - row.created_at).total_seconds()),
        one_person_confirmed=row.one_person_confirmed,
        subject_label=row.subject_label,
        error=row.error,
        abstain_reason=" ".join(reasons) if row.status == "abstained" and reasons else None,
    )


def get_status(
    db: Session, clinician: Clinician, analysis_id: uuid.UUID, settings: Settings
) -> GaitAnalysisStatus:
    row = _own(db, clinician, analysis_id)
    refresh_from_disk(db, row, settings)
    return _status(row)


def list_analyses(
    db: Session, clinician: Clinician, settings: Settings, limit: int, offset: int
) -> tuple[list[GaitAnalysisStatus], int]:
    q = db.query(GaitAnalysis).filter(GaitAnalysis.clinician_id == clinician.id)
    total = q.count()
    rows = q.order_by(GaitAnalysis.created_at.desc()).offset(offset).limit(limit).all()
    for r in rows:
        refresh_from_disk(db, r, settings)
    return [_status(r) for r in rows], total


def get_result(
    db: Session, clinician: Clinician, analysis_id: uuid.UUID, settings: Settings
) -> dict:
    row = _own(db, clinician, analysis_id)
    refresh_from_disk(db, row, settings)
    if row.status not in FINAL_STATUSES or row.result_json is None:
        raise GaitResultNotReadyError
    _audit(db, clinician, "gait_result_view", row.id)
    db.commit()
    return json.loads(row.result_json)
