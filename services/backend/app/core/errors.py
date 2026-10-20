"""Domain errors and the one JSON shape every error response uses.

Services raise an `AppError` subclass; routes never build an `HTTPException`
by hand (see API-Standards.md §1.4). The handlers in `main.py` turn any
`AppError` — and FastAPI's own validation errors — into:

    {"error": {"code": "INVALID_CREDENTIALS", "message": "...", "details": null}}

`code` is mirrored in `packages/shared/src/api/errorCodes.ts`; keep the two
lists in sync (see tests/test_error_codes.py).
"""

from typing import Any


class AppError(Exception):
    """Base domain error. Subclass per failure case; don't raise this directly."""

    status_code = 400
    code = "BAD_REQUEST"
    message = "Bad request"

    def __init__(self, message: str | None = None, details: Any = None) -> None:
        super().__init__(message or self.message)
        if message is not None:
            self.message = message
        self.details = details


class InvalidCredentialsError(AppError):
    status_code, code, message = 401, "INVALID_CREDENTIALS", "Incorrect username or password"


class InvalidRefreshTokenError(AppError):
    status_code, code, message = (
        401,
        "INVALID_REFRESH_TOKEN",
        "Refresh token is invalid, expired, or already used",
    )


class NotAuthenticatedError(AppError):
    status_code, code, message = 401, "NOT_AUTHENTICATED", "Not authenticated"


class MriAnalysisNotFoundError(AppError):
    # Also returned for another clinician's analysis, so IDs cannot be probed.
    status_code, code, message = 404, "MRI_ANALYSIS_NOT_FOUND", "MRI analysis not found"


class MriResultNotReadyError(AppError):
    status_code, code, message = 409, "MRI_RESULT_NOT_READY", "The MRI analysis has not finished"


class GaitAnalysisNotFoundError(AppError):
    # Also returned for another clinician's analysis, so IDs cannot be probed.
    status_code, code, message = 404, "GAIT_ANALYSIS_NOT_FOUND", "Gait analysis not found"


class GaitResultNotReadyError(AppError):
    status_code, code, message = 409, "GAIT_RESULT_NOT_READY", "The gait analysis has not finished"


class OnePersonNotConfirmedError(AppError):
    # A form field the request did fill in, but not with the one accepted answer.
    status_code, code, message = (
        422,
        "VALIDATION_ERROR",
        "Confirm that the video shows one person walking, filmed from the side",
    )


class InvalidUploadError(AppError):
    status_code, code, message = 400, "INVALID_UPLOAD", "The upload must be a zip of DICOM files"


class UploadTooLargeError(AppError):
    status_code, code, message = 413, "UPLOAD_TOO_LARGE", "The upload is larger than allowed"


# Every AppError code, for the parity test against errorCodes.ts.
ERROR_CODES: tuple[str, ...] = (
    AppError.code,
    InvalidCredentialsError.code,
    InvalidRefreshTokenError.code,
    NotAuthenticatedError.code,
    MriAnalysisNotFoundError.code,
    MriResultNotReadyError.code,
    GaitAnalysisNotFoundError.code,
    GaitResultNotReadyError.code,
    InvalidUploadError.code,
    UploadTooLargeError.code,
    "VALIDATION_ERROR",
)
