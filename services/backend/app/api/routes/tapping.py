import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_clinician, get_tapping_runner
from app.core.config import Settings, get_settings
from app.core.database import get_db
from app.models.clinician import Clinician
from app.schemas.errors import ErrorResponse
from app.schemas.mri import SUBJECT_LABEL_PATTERN
from app.schemas.tapping import (
    TappingAnalysisCreated,
    TappingAnalysisList,
    TappingAnalysisStatus,
    TappingResult,
    TappingTarget,
)
from app.services import tapping_service
from app.workers.tapping_runner import TappingRunner

router = APIRouter(prefix="/tapping", tags=["tapping"])


@router.post(
    "/analyses",
    response_model=TappingAnalysisCreated,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload one finger-tapping video and start the analysis",
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
def create_analysis(
    file: Annotated[UploadFile, File(description="One hand tapping thumb to index finger")],
    target: Annotated[
        TappingTarget,
        Form(description="Study target: ge2 (primary) or ge1 (secondary)"),
    ] = "ge2",
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
    runner: TappingRunner = Depends(get_tapping_runner),
    settings: Settings = Depends(get_settings),
) -> TappingAnalysisCreated:
    row = tapping_service.create_analysis(
        db, clinician, file.file, file.filename, runner, settings, target, subject_label
    )
    return TappingAnalysisCreated(
        id=row.id,
        status=row.status,  # type: ignore[arg-type]
        target=row.target,  # type: ignore[arg-type]
        subject_label=row.subject_label,
    )


@router.get(
    "/analyses",
    response_model=TappingAnalysisList,
    status_code=status.HTTP_200_OK,
    summary="The signed-in clinician's finger-tapping analyses, newest first",
    responses={401: {"model": ErrorResponse}},
)
def list_analyses(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    clinician: Clinician = Depends(get_current_clinician),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> TappingAnalysisList:
    items, total = tapping_service.list_analyses(db, clinician, settings, limit, offset)
    return TappingAnalysisList(items=items, total=total)


@router.get(
    "/analyses/{analysis_id}",
    response_model=TappingAnalysisStatus,
    status_code=status.HTTP_200_OK,
    summary="Status of one tapping analysis (poll until the status is final)",
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
def get_status(
    analysis_id: uuid.UUID,
    clinician: Clinician = Depends(get_current_clinician),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> TappingAnalysisStatus:
    return tapping_service.get_status(db, clinician, analysis_id, settings)


@router.post(
    "/analyses/{analysis_id}/stop",
    response_model=TappingAnalysisStatus,
    status_code=status.HTTP_200_OK,
    summary="Stop a queued or running tapping analysis (it is recorded as failed: stopped)",
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
def stop_analysis(
    analysis_id: uuid.UUID,
    clinician: Clinician = Depends(get_current_clinician),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> TappingAnalysisStatus:
    return tapping_service.stop_analysis(db, clinician, analysis_id, settings)


@router.get(
    "/analyses/{analysis_id}/result",
    response_model=TappingResult,
    # Only what the program wrote: a scored result has no "reason" and a failed one has no score,
    # and the desktop app checks the exact shape (apps/desktop/src/shared/tappingResult.ts).
    response_model_exclude_unset=True,
    status_code=status.HTTP_200_OK,
    summary="The tapping module's output: score, contributions, curves, PDF, or why it was not scored",
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
) -> TappingResult:
    return TappingResult.model_validate(
        tapping_service.get_result(db, clinician, analysis_id, settings)
    )
