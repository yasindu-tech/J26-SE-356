import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Literal
from uuid import UUID

import bcrypt
import jwt

from app.core.config import get_settings
from app.core.time import utc_now

TokenType = Literal["access", "refresh"]

# bcrypt's algorithm ignores any bytes past the 72nd — recent versions of the
# `bcrypt` package raise instead of silently truncating, so truncate
# ourselves rather than letting a long password 500 the login route.
_BCRYPT_MAX_BYTES = 72


def hash_password(plain_password: str) -> str:
    truncated = plain_password.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    return bcrypt.hashpw(truncated, bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    truncated = plain_password.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    return bcrypt.checkpw(truncated, hashed_password.encode("utf-8"))


def create_access_token(clinician_id: UUID) -> tuple[str, datetime]:
    settings = get_settings()
    expires_at = utc_now() + timedelta(minutes=settings.access_token_expire_minutes)
    token = _encode(clinician_id, "access", expires_at)
    return token, expires_at


def decode_access_token(token: str) -> UUID:
    """Returns the clinician id if `token` is a valid, unexpired access token.

    Raises `jwt.InvalidTokenError` (or a subclass) otherwise — callers turn
    that into a 401, not a 500; see api/deps.py.
    """
    payload = jwt.decode(
        token, get_settings().jwt_secret_key, algorithms=[get_settings().jwt_algorithm]
    )
    if payload.get("type") != "access":
        raise jwt.InvalidTokenError("not an access token")
    return UUID(payload["sub"])


def _encode(clinician_id: UUID, token_type: TokenType, expires_at: datetime) -> str:
    settings = get_settings()
    # jti makes each token unique even if issued within the same second as
    # another one for the same clinician (e.g. login immediately followed by
    # refresh) — without it, two tokens with an identical exp are byte-identical.
    payload = {
        "sub": str(clinician_id),
        "type": token_type,
        "exp": expires_at,
        "jti": secrets.token_hex(8),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def generate_refresh_token_secret() -> str:
    """A high-entropy opaque string — not a JWT.

    The refresh token is looked up by its hash in the database (see
    models/refresh_token.py), so it can actually be revoked on logout.
    A self-contained JWT refresh token can't be revoked without a separate
    denylist anyway, so there's no point encoding claims into it.
    """
    return secrets.token_urlsafe(48)


def hash_refresh_token(raw_token: str) -> str:
    """SHA-256 is fine here (unlike for passwords) — the input is already a
    48-byte random secret, not low-entropy user input, so there's nothing
    for a slow KDF to protect against."""
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
