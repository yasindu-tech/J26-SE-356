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


# Every AppError code, for the parity test against errorCodes.ts.
ERROR_CODES: tuple[str, ...] = (
    AppError.code,
    InvalidCredentialsError.code,
    InvalidRefreshTokenError.code,
    NotAuthenticatedError.code,
    "VALIDATION_ERROR",
)
