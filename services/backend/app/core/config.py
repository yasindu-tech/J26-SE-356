from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
