"""Finger-tapping analyses: accept a video, start the analysis, report its status, serve the result.

Flow: create_analysis saves the video under data/backend/tapping/<id>/ (named by the analysis ID,
never by its own name) and hands it to the runner. The runner deletes the video when the process
ends and leaves result.json. Reads pick result.json up, store the result in the database
(PostgreSQL in the Docker stack) and delete the folder, so the upload itself is not kept.
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
    InvalidUploadError,
    TappingAnalysisNotFoundError,
    TappingResultNotReadyError,
    UploadTooLargeError,
)
from app.core.time import utc_now
from app.models.audit_log import AuditLogEntry
from app.models.clinician import Clinician
from app.models.tapping_analysis import TappingAnalysis
from app.schemas.mri import FINAL_STATUSES
from app.schemas.tapping import TAPPING_VIDEO_TYPES, TappingAnalysisStatus
from app.workers.stoppable import STOPPED_MESSAGE, request_stop
from app.workers.tapping_runner import TappingJob, TappingRunner, failed_result

CHUNK = 1024 * 1024
NOT_A_VIDEO = f"The upload must be a video file ({', '.join(TAPPING_VIDEO_TYPES)})"


def _folder(settings: Settings, analysis_id: uuid.UUID) -> Path:
    return settings.tapping_work_dir / str(analysis_id)


def _audit(db: Session, clinician: Clinician, action: str, analysis_id: uuid.UUID) -> None:
    db.add(AuditLogEntry(clinician_id=clinician.id, action=action, detail=str(analysis_id)))


def video_suffix(filename: str | None) -> str:
    """The upload's extension if it is an accepted video type (its name is not kept)."""
    suffix = Path(filename or "").suffix.lower()
    if suffix not in TAPPING_VIDEO_TYPES:
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
    runner: TappingRunner,
    settings: Settings,
    target: str = "ge2",
    subject_label: str | None = None,
) -> TappingAnalysis:
    suffix = video_suffix(filename)
    row = TappingAnalysis(
        id=uuid.uuid4(),
        clinician_id=clinician.id,
        status="queued",
        target=target,
        subject_label=subject_label,
        created_at=utc_now(),
    )
    folder = _folder(settings, row.id)
    video = folder / f"{row.id}{suffix}"
    try:
        save_upload(upload, video, settings.tapping_max_upload_mb * CHUNK)
    except (InvalidUploadError, UploadTooLargeError):
        shutil.rmtree(folder, ignore_errors=True)
        raise
    db.add(row)
    _audit(db, clinician, "tapping_upload", row.id)
    db.commit()
    db.refresh(row)
    runner.submit(TappingJob(row.id, folder, video, target))
    return row


def _own(db: Session, clinician: Clinician, analysis_id: uuid.UUID) -> TappingAnalysis:
    row = db.get(TappingAnalysis, analysis_id)
    if row is None or row.clinician_id != clinician.id:
        raise TappingAnalysisNotFoundError
    return row


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _finish(db: Session, row: TappingAnalysis, status: str, error: str | None) -> None:
    row.status, row.error, row.finished_at = status, error, utc_now()
    db.commit()


def _outcome(result: dict) -> tuple[str, str | None]:
    """(status, error) of a finished result.

    A video the program read but could not score (it names its clip) is "abstained" with the
    reason, like a gait video the model declines. No clip ref means the analysis itself could not
    run (timeout, program missing): "failed".
    """
    if result.get("status") == "scored":
        return "scored", None
    reason = str(result.get("reason") or "the analysis failed")[:500]
    return ("abstained", None) if result.get("clip_ref") else ("failed", reason)


def refresh_from_disk(db: Session, row: TappingAnalysis, settings: Settings) -> None:
    """Move a finished result from disk into the database, or notice a lost analysis."""
    if row.status in FINAL_STATUSES:
        return
    folder = _folder(settings, row.id)
    result = _read_json(folder / "result.json")
    if result is not None:
        status, error = _outcome(result)
        row.result_json = json.dumps(result)
        _finish(db, row, status, error)
        shutil.rmtree(folder, ignore_errors=True)
        return
    started = _read_json(folder / "started.json")
    if started is not None:
        if time.time() - float(started.get("t", 0)) > settings.tapping_timeout_minutes * 60 + 60:
            _finish(db, row, "failed", "the analysis stopped")
            shutil.rmtree(folder, ignore_errors=True)
        elif row.status != "running":
            row.status = "running"
            db.commit()
        return
    if not folder.exists():  # nothing on disk and no result: the backend lost the job
        _finish(db, row, "failed", "the analysis was interrupted")


def _status(row: TappingAnalysis) -> TappingAnalysisStatus:
    result = json.loads(row.result_json) if row.result_json else {}
    end = row.finished_at or utc_now()
    return TappingAnalysisStatus(
        id=row.id,
        status=row.status,  # type: ignore[arg-type]
        target=row.target,  # type: ignore[arg-type]
        created_at=row.created_at,
        finished_at=row.finished_at,
        elapsed_s=max(0.0, (end - row.created_at).total_seconds()),
        subject_label=row.subject_label,
        error=row.error,
        abstain_reason=result.get("reason") if row.status == "abstained" else None,
    )


def get_status(
    db: Session, clinician: Clinician, analysis_id: uuid.UUID, settings: Settings
) -> TappingAnalysisStatus:
    row = _own(db, clinician, analysis_id)
    refresh_from_disk(db, row, settings)
    return _status(row)


def stop_analysis(
    db: Session, clinician: Clinician, analysis_id: uuid.UUID, settings: Settings
) -> TappingAnalysisStatus:
    """Stop a queued or running analysis: recorded as failed at once; the worker ends the
    program and deletes what it left. An analysis that has already ended is returned as it is."""
    row = _own(db, clinician, analysis_id)
    refresh_from_disk(db, row, settings)
    if row.status not in FINAL_STATUSES:
        request_stop(_folder(settings, row.id))
        row.result_json = json.dumps(failed_result(STOPPED_MESSAGE, row.target))
        _audit(db, clinician, "tapping_stop", row.id)
        _finish(db, row, "failed", STOPPED_MESSAGE)
    return _status(row)


def list_analyses(
    db: Session, clinician: Clinician, settings: Settings, limit: int, offset: int
) -> tuple[list[TappingAnalysisStatus], int]:
    q = db.query(TappingAnalysis).filter(TappingAnalysis.clinician_id == clinician.id)
    total = q.count()
    rows = q.order_by(TappingAnalysis.created_at.desc()).offset(offset).limit(limit).all()
    for r in rows:
        refresh_from_disk(db, r, settings)
    return [_status(r) for r in rows], total


def get_result(
    db: Session, clinician: Clinician, analysis_id: uuid.UUID, settings: Settings
) -> dict:
    row = _own(db, clinician, analysis_id)
    refresh_from_disk(db, row, settings)
    if row.status not in FINAL_STATUSES or row.result_json is None:
        raise TappingResultNotReadyError
    _audit(db, clinician, "tapping_result_view", row.id)
    db.commit()
    return json.loads(row.result_json)
