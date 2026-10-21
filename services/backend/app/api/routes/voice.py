from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_clinician, get_voice_runner
from app.core.config import Settings, get_settings
from app.core.database import get_db
from app.models.clinician import Clinician
from app.schemas.errors import ErrorResponse
from app.schemas.voice import VoiceResult
from app.services import voice_service
from app.workers.voice_runner import VoiceRunner

router = APIRouter(prefix="/voice", tags=["voice"])


@router.post(
    "/analyses",
    response_model=VoiceResult,
    status_code=status.HTTP_200_OK,
    summary="Analyse one sustained 'aaah' recording and return the explained voice result",
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def analyze(
    file: Annotated[
        UploadFile, File(description="One person's sustained 'aaah', any audio format")
    ],
    clinician: Clinician = Depends(get_current_clinician),
    db: Session = Depends(get_db),
    runner: VoiceRunner = Depends(get_voice_runner),
    settings: Settings = Depends(get_settings),
) -> VoiceResult:
    result = voice_service.analyze(db, clinician, file.file, file.filename, runner, settings)
    return VoiceResult.model_validate(result)
