import sys
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[4]

# The fallback JWT secret. Fine for environment="dev" (local dev/tests
# without a .env file); the app refuses to start with it in any other
# environment — see main.py's startup check.
INSECURE_DEV_JWT_SECRET = "dev-only-insecure-secret-change-me"


class Settings(BaseSettings):
    """Backend configuration, loaded from environment variables / `.env`."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "dev"  # "dev" | "staging" | "production"
    api_prefix: str = "/api/v1"

    database_url: str = "postgresql+psycopg2://pdxai:pdxai@localhost:5432/pdxai"

    # JWT signing. jwt_secret_key has no safe default — it must be set via
    # the environment in any real deployment; the fallback below is only so
    # local dev/tests can run without a .env file.
    jwt_secret_key: str = INSECURE_DEV_JWT_SECRET
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7

    # Never "*" — a browser preflight only succeeds for an origin listed here.
    cors_origins: list[str] = ["http://localhost:5173"]

    # MRI analyses (models/mri/src/predict_scan.py runs as a separate process).
    # Uploads and working files live under data/ (git-ignored, never committed);
    # the uploaded DICOM files are deleted as soon as the analysis ends.
    mri_work_dir: Path = REPO_ROOT / "data" / "backend" / "mri"
    mri_python: str = sys.executable  # interpreter with the models/mri dependencies
    mri_script: Path = REPO_ROOT / "models" / "mri" / "src" / "predict_scan.py"
    mri_max_upload_mb: int = 4096
    mri_eddy_threads: int = 2
    mri_timeout_hours: float = 6.0

    # Gait analyses (models/gait/src/predict_video.py runs as a separate process, one video at a
    # time, in its own worker so a gait video never waits behind an MRI scan). The uploaded
    # video is deleted when the analysis ends; the overlay video drawn from it is kept.
    gait_work_dir: Path = REPO_ROOT / "data" / "backend" / "gait"
    gait_python: str = sys.executable  # interpreter with the models/gait dependencies
    gait_script: Path = REPO_ROOT / "models" / "gait" / "src" / "predict_video.py"
    gait_overlay_script: Path = REPO_ROOT / "models" / "gait" / "src" / "make_overlay_video.py"
    gait_overlay_dir: Path = REPO_ROOT / "data" / "backend" / "gait_overlays"
    gait_bundle_dir: Path = (
        REPO_ROOT / "models" / "gait" / "artifacts" / "models" / "xgboost" / "bundle"
    )
    gait_max_upload_mb: int = 500
    gait_timeout_minutes: float = 15.0

    # Voice analyses (models/voice/src/analyze_audio.py runs as a separate process). A recording
    # takes a few seconds, so the request waits for the result. The uploaded audio is deleted as
    # soon as the analysis ends and the result is not stored.
    voice_work_dir: Path = REPO_ROOT / "data" / "backend" / "voice"
    voice_python: str = sys.executable  # interpreter with the models/voice dependencies
    voice_script: Path = REPO_ROOT / "models" / "voice" / "src" / "analyze_audio.py"
    voice_model: str = "audio"  # analyze_audio.py --model: "audio" (raw-audio) or "uci"
    voice_max_upload_mb: int = 50
    voice_timeout_seconds: float = 120.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
