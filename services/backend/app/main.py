from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes.auth import router as auth_router
from app.core.config import INSECURE_DEV_JWT_SECRET, get_settings
from app.core.errors import AppError

settings = get_settings()

if settings.jwt_secret_key == INSECURE_DEV_JWT_SECRET and settings.environment != "dev":
    raise RuntimeError(
        "JWT_SECRET_KEY is still the insecure dev default outside environment=dev "
        f"(environment={settings.environment!r}). Set a real secret before starting."
    )

app = FastAPI(title="PD-XAI Backend")

# CORS origins come from settings, never "*" (API-Standards.md §1.5) — the
# desktop/mobile apps' own origins, not every site on the web.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix=settings.api_prefix)


def _error_envelope(code: str, message: str, details: object = None) -> dict:
    return {"error": {"code": code, "message": message, "details": details}}


@app.exception_handler(AppError)
def handle_app_error(_request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_envelope(exc.code, exc.message, exc.details),
    )


@app.exception_handler(RequestValidationError)
def handle_validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=_error_envelope("VALIDATION_ERROR", "Invalid request", exc.errors()),
    )


# Catches HTTPExceptions we didn't raise ourselves — e.g. HTTPBearer rejecting
# a missing/malformed Authorization header before get_current_clinician ever
# runs — so every error response uses the one envelope, not a bare {"detail": ...}.
@app.exception_handler(HTTPException)
def handle_http_exception(_request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_envelope(f"HTTP_{exc.status_code}", str(exc.detail)),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
