"""Voice analyses: accept one recording, run the voice module on it, return its result.

Flow: the recording is saved under data/backend/voice/<id>/ (named by a fresh ID, never by its
own name), analyze_audio.py runs on it, and the folder is deleted before the response is sent,
whatever happened. Nothing of the upload or its result is kept; only an audit entry records
that an analysis was made.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import BinaryIO

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import InvalidUploadError, UploadTooLargeError
from app.models.audit_log import AuditLogEntry
from app.models.clinician import Clinician
from app.schemas.voice import VOICE_AUDIO_TYPES
from app.workers.voice_runner import VoiceJob, VoiceRunner

CHUNK = 1024 * 1024
NOT_AUDIO = f"The upload must be an audio file ({', '.join(VOICE_AUDIO_TYPES)})"


def audio_suffix(filename: str | None) -> str:
    """The upload's extension if it is an accepted audio type (its name is not kept)."""
    suffix = Path(filename or "").suffix.lower()
    if suffix not in VOICE_AUDIO_TYPES:
        raise InvalidUploadError(NOT_AUDIO)
    return suffix


def save_upload(src: BinaryIO, dest: Path, max_bytes: int) -> None:
    """Copy the uploaded recording to disk in chunks, refusing an empty or too large file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with dest.open("wb") as out:
        while chunk := src.read(CHUNK):
            size += len(chunk)
            if size > max_bytes:
                raise UploadTooLargeError(f"The upload is over {max_bytes // CHUNK} MB")
            out.write(chunk)
    if size == 0:
        raise InvalidUploadError("The uploaded recording is empty")


def analyze(
    db: Session,
    clinician: Clinician,
    upload: BinaryIO,
    filename: str | None,
    runner: VoiceRunner,
    settings: Settings,
) -> dict:
    suffix = audio_suffix(filename)
    analysis_id = uuid.uuid4()
    folder = settings.voice_work_dir / str(analysis_id)
    audio = folder / f"{analysis_id}{suffix}"
    try:
        save_upload(upload, audio, settings.voice_max_upload_mb * CHUNK)
        result = runner.run(VoiceJob(folder, audio))
    finally:
        shutil.rmtree(folder, ignore_errors=True)  # the recording is never kept
    db.add(
        AuditLogEntry(clinician_id=clinician.id, action="voice_analysis", detail=str(analysis_id))
    )
    db.commit()
    return result
