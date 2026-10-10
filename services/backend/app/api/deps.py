import jwt
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.errors import NotAuthenticatedError
from app.core.security import decode_access_token
from app.models.clinician import Clinician

_bearer_scheme = HTTPBearer()


def get_current_clinician(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> Clinician:
    """Dependency for any route that needs an authenticated clinician.
    Wire it in as `Depends(get_current_clinician)` on every route except
    `/health`, `/auth/login`, `/auth/refresh` and `/auth/logout`."""
    try:
        clinician_id = decode_access_token(credentials.credentials)
    except jwt.InvalidTokenError:
        raise NotAuthenticatedError from None

    clinician = db.query(Clinician).filter(Clinician.id == clinician_id).first()
    if clinician is None or not clinician.is_active:
        raise NotAuthenticatedError
    return clinician
