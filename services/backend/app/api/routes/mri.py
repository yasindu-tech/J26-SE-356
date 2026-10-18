import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_clinician, get_mri_runner
from app.core.config import Settings, get_settings
from app.core.database import get_db
from app.models.clinician import Clinician
from app.schemas.errors import ErrorResponse
from app.schemas.mri import (
    EddyProfile,
    MriAnalysisCreated,
    MriAnalysisList,
    MriAnalysisStatus,
    MriResult,
)
from app.services import mri_service
from app.workers.mri_runner import MriRunner

router = APIRouter(prefix="/mri", tags=["mri"])


@router.post(
    "/analyses",
    response_model=MriAnalysisCreated,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload one person's MRI (zip of DICOM files) and start the analysis",
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
    },
)
def create_analysis(
    file: Annotated[
        UploadFile, File(description="Zip of one person's DICOM files (T1 and/or DTI)")
    ],
    age: Annotated[
        float, Form(ge=18, le=110, description="Years; a covariate of the scanner adjustment")
    ],
    sex: Annotated[Literal["M", "F"], Form(description="A covariate of the scanner adjustment")],
    eddy_profile: Annotated[
        EddyProfile,
        Form(
            description="DTI motion correction: 'full' (default, what the model was trained on) "
            "or 'fast' (fewer iterations, quicker, not validated against 'full')"
        ),
    ] = "full",
    clinician: Clinician = Depends(get_current_clinician),
    db: Session = Depends(get_db),
    runner: MriRunner = Depends(get_mri_runner),
    settings: Settings = Depends(get_settings),
) -> MriAnalysisCreated:
    row = mri_service.create_analysis(
        db, clinician, file.file, age, sex, runner, settings, eddy_profile
    )
    return MriAnalysisCreated(
        id=row.id,
        status=row.status,  # type: ignore[arg-type]
        eddy_profile=row.eddy_profile,  # type: ignore[arg-type]
    )


@router.get(
    "/analyses",
    response_model=MriAnalysisList,
    status_code=status.HTTP_200_OK,
    summary="The signed-in clinician's MRI analyses, newest first",
    responses={401: {"model": ErrorResponse}},
)
def list_analyses(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    clinician: Clinician = Depends(get_current_clinician),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> MriAnalysisList:
    items, total = mri_service.list_analyses(db, clinician, settings, limit, offset)
    return MriAnalysisList(items=items, total=total)


@router.get(
    "/analyses/{analysis_id}",
    response_model=MriAnalysisStatus,
    status_code=status.HTTP_200_OK,
    summary="Progress of one MRI analysis (poll until the status is final)",
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
def get_status(
    analysis_id: uuid.UUID,
    clinician: Clinician = Depends(get_current_clinician),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> MriAnalysisStatus:
    return mri_service.get_status(db, clinician, analysis_id, settings)


@router.get(
    "/analyses/{analysis_id}/result",
    response_model=MriResult,
    status_code=status.HTTP_200_OK,
    summary="The MRI module's output: feature vector, score with 95% interval, explanation",
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
) -> MriResult:
    return MriResult.model_validate(mri_service.get_result(db, clinician, analysis_id, settings))
