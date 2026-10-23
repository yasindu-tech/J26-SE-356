import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_clinician, get_gait_runner
from app.core.config import Settings, get_settings
from app.core.database import get_db
from app.models.clinician import Clinician
from app.schemas.errors import ErrorResponse
from app.schemas.gait import GaitAnalysisCreated, GaitAnalysisList, GaitAnalysisStatus, GaitResult
from app.schemas.mri import SUBJECT_LABEL_PATTERN
from app.services import gait_service
from app.workers.gait_runner import GaitRunner

router = APIRouter(prefix="/gait", tags=["gait"])


@router.post(
    "/analyses",
    response_model=GaitAnalysisCreated,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload one walking video and start the gait analysis",
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
def create_analysis(
    file: Annotated[UploadFile, File(description="One person walking, filmed from the side")],
    one_person_confirmed: Annotated[
        bool, Form(description="Must be true: the clinician confirms one person is walking")
    ],
    subject_label: Annotated[
        str | None,
        Form(
            pattern=SUBJECT_LABEL_PATTERN,
            description="Optional pseudonymous label to tell one person's analyses apart "
            "(letters, digits, '-' and '_'; never a name)",
        ),
    ] = None,
    clinician: Clinician = Depends(get_current_clinician),
    db: Session = Depends(get_db),
    runner: GaitRunner = Depends(get_gait_runner),
    settings: Settings = Depends(get_settings),
) -> GaitAnalysisCreated:
    row = gait_service.create_analysis(
        db,
        clinician,
        file.file,
        file.filename,
        one_person_confirmed,
        runner,
        settings,
        subject_label,
    )
    return GaitAnalysisCreated(
        id=row.id,
        status=row.status,  # type: ignore[arg-type]
        subject_label=row.subject_label,
    )


@router.get(
    "/analyses",
    response_model=GaitAnalysisList,
    status_code=status.HTTP_200_OK,
    summary="The signed-in clinician's gait analyses, newest first",
    responses={401: {"model": ErrorResponse}},
)
def list_analyses(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    clinician: Clinician = Depends(get_current_clinician),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> GaitAnalysisList:
    items, total = gait_service.list_analyses(db, clinician, settings, limit, offset)
    return GaitAnalysisList(items=items, total=total)


@router.get(
    "/analyses/{analysis_id}",
    response_model=GaitAnalysisStatus,
    status_code=status.HTTP_200_OK,
    summary="Status of one gait analysis (poll until the status is final)",
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
def get_status(
    analysis_id: uuid.UUID,
    clinician: Clinician = Depends(get_current_clinician),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> GaitAnalysisStatus:
    return gait_service.get_status(db, clinician, analysis_id, settings)


@router.post(
    "/analyses/{analysis_id}/stop",
    response_model=GaitAnalysisStatus,
    status_code=status.HTTP_200_OK,
    summary="Stop a queued or running gait analysis (it is recorded as failed: stopped)",
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
def stop_analysis(
    analysis_id: uuid.UUID,
    clinician: Clinician = Depends(get_current_clinician),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> GaitAnalysisStatus:
    return gait_service.stop_analysis(db, clinician, analysis_id, settings)


@router.get(
    "/analyses/{analysis_id}/result",
    response_model=GaitResult,
    status_code=status.HTTP_200_OK,
    summary="The gait module's output: scores, plain-words explanation, or why it was not scored",
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
)
def get_result(
    analysis_id: uuid.UUID,
    clinician: Clinician = Depends(get_current_clinician),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> GaitResult:
    return GaitResult.model_validate(gait_service.get_result(db, clinician, analysis_id, settings))


@router.get(
    "/analyses/{analysis_id}/overlay",
    response_class=FileResponse,
    status_code=status.HTTP_200_OK,
    summary="The walking video with the skeleton and the counted heel strikes drawn on it (mp4)",
    responses={
        200: {"content": {"video/mp4": {}}},
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
def get_overlay(
    analysis_id: uuid.UUID,
    clinician: Clinician = Depends(get_current_clinician),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    path = gait_service.get_overlay(db, clinician, analysis_id, settings)
    return FileResponse(path, media_type="video/mp4")
