from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_clinician
from app.core.database import get_db
from app.models.clinician import Clinician
from app.schemas.auth import LoginRequest, LogoutRequest, MeResponse, RefreshRequest, TokenPair
from app.schemas.errors import ErrorResponse
from app.services import auth_service

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/login",
    response_model=TokenPair,
    status_code=status.HTTP_200_OK,
    summary="Sign in a clinician",
    responses={401: {"model": ErrorResponse}},
)
def login(body: LoginRequest, db: Session = Depends(get_db)) -> TokenPair:
    return auth_service.authenticate_and_issue_tokens(db, body.username, body.password)


@router.post(
    "/refresh",
    response_model=TokenPair,
    status_code=status.HTTP_200_OK,
    summary="Exchange a refresh token for a new token pair",
    responses={401: {"model": ErrorResponse}},
)
def refresh(body: RefreshRequest, db: Session = Depends(get_db)) -> TokenPair:
    return auth_service.refresh_tokens(db, body.refresh_token)


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke a refresh token",
)
def logout(body: LogoutRequest, db: Session = Depends(get_db)) -> None:
    auth_service.revoke_refresh_token(db, body.refresh_token)


@router.get(
    "/me",
    response_model=MeResponse,
    status_code=status.HTTP_200_OK,
    summary="The signed-in clinician",
    responses={401: {"model": ErrorResponse}},
)
def me(clinician: Clinician = Depends(get_current_clinician)) -> MeResponse:
    return MeResponse(
        id=clinician.id,
        username=clinician.username,
        full_name=clinician.full_name,
        role=clinician.role,
    )
