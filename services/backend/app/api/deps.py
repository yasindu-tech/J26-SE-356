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


_mri_runner = None


def get_mri_runner():  # noqa: ANN201 (returns the app-wide MriRunner)
    """The one background runner for MRI analyses (tests override this dependency)."""
    global _mri_runner
    if _mri_runner is None:
        from app.core.config import get_settings
        from app.workers.mri_runner import SubprocessRunner

        _mri_runner = SubprocessRunner(get_settings())
    return _mri_runner


_gait_runner = None


def get_gait_runner():  # noqa: ANN201 (returns the app-wide GaitRunner)
    """The one background runner for gait analyses (tests override this dependency)."""
    global _gait_runner
    if _gait_runner is None:
        from app.core.config import get_settings
        from app.workers.gait_runner import SubprocessGaitRunner

        _gait_runner = SubprocessGaitRunner(get_settings())
    return _gait_runner
