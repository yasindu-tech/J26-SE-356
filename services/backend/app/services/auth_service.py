import logging
import uuid
from datetime import timedelta

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import InvalidCredentialsError, InvalidRefreshTokenError
from app.core.security import (
    create_access_token,
    generate_refresh_token_secret,
    hash_refresh_token,
    verify_password,
)
from app.core.time import utc_now
from app.models.audit_log import AuditLogEntry
from app.models.clinician import Clinician
from app.models.refresh_token import RefreshToken
from app.schemas.auth import TokenPair

logger = logging.getLogger(__name__)


def authenticate_and_issue_tokens(db: Session, username: str, password: str) -> TokenPair:
    clinician = db.query(Clinician).filter(Clinician.username == username).first()

    if (
        clinician is None
        or not clinician.is_active
        or not verify_password(password, clinician.hashed_password)
    ):
        # Log the attempted username, not which check failed — same reasoning
        # as InvalidCredentialsError not distinguishing the two.
        db.add(
            AuditLogEntry(clinician_id=None, action="login_failed", detail=f"username={username!r}")
        )
        db.commit()
        raise InvalidCredentialsError

    tokens = _issue_token_pair(db, clinician)
    db.add(AuditLogEntry(clinician_id=clinician.id, action="login"))
    db.commit()
    logger.info("clinician login", extra={"clinician_id": str(clinician.id)})
    return tokens


def refresh_tokens(db: Session, raw_refresh_token: str) -> TokenPair:
    record = _find_refresh_token(db, raw_refresh_token)
    if record is None or record.expires_at < utc_now():
        raise InvalidRefreshTokenError

    if record.revoked_at is not None:
        # Reuse of a token that was already rotated away (or explicitly logged
        # out). That can only happen if it was copied somewhere it shouldn't
        # have been, so the whole chain is suspect: revoke every other active
        # refresh token for this clinician too, not just reject this one.
        # Otherwise a stolen-and-already-rotated token's successor keeps working
        # even after the theft is detected.
        _revoke_all_refresh_tokens(db, record.clinician_id)
        db.add(
            AuditLogEntry(clinician_id=record.clinician_id, action="refresh_token_reuse_detected")
        )
        db.commit()
        raise InvalidRefreshTokenError

    # Rotate: revoke the used token and issue a new pair, so a stolen refresh
    # token can only be replayed once before the legitimate client's next
    # refresh reveals the theft (the legitimate client's token will already
    # be revoked, which is handled by the reuse branch above).
    record.revoked_at = utc_now()
    clinician = db.query(Clinician).filter(Clinician.id == record.clinician_id).one()
    if not clinician.is_active:
        db.commit()
        raise InvalidRefreshTokenError

    tokens = _issue_token_pair(db, clinician)
    db.commit()
    return tokens


def revoke_refresh_token(db: Session, raw_refresh_token: str) -> None:
    record = _get_active_refresh_token(db, raw_refresh_token)
    if record is not None:
        record.revoked_at = utc_now()
        db.add(AuditLogEntry(clinician_id=record.clinician_id, action="logout"))
        db.commit()
    # Logging out with an already-invalid token is a no-op, not an error —
    # the caller's goal (be logged out) is already satisfied.


def _find_refresh_token(db: Session, raw_refresh_token: str) -> RefreshToken | None:
    token_hash = hash_refresh_token(raw_refresh_token)
    return db.query(RefreshToken).filter(RefreshToken.token_hash == token_hash).first()


def _get_active_refresh_token(db: Session, raw_refresh_token: str) -> RefreshToken | None:
    record = _find_refresh_token(db, raw_refresh_token)
    if record is None or record.revoked_at is not None or record.expires_at < utc_now():
        return None
    return record


def _revoke_all_refresh_tokens(db: Session, clinician_id: uuid.UUID) -> None:
    db.query(RefreshToken).filter(
        RefreshToken.clinician_id == clinician_id, RefreshToken.revoked_at.is_(None)
    ).update({"revoked_at": utc_now()})


def _issue_token_pair(db: Session, clinician: Clinician) -> TokenPair:
    settings = get_settings()
    access_token, access_expires_at = create_access_token(clinician.id)

    raw_refresh_token = generate_refresh_token_secret()
    db.add(
        RefreshToken(
            clinician_id=clinician.id,
            token_hash=hash_refresh_token(raw_refresh_token),
            expires_at=utc_now() + timedelta(days=settings.refresh_token_expire_days),
        )
    )

    expires_in = int((access_expires_at - utc_now()).total_seconds())
    return TokenPair(
        access_token=access_token, refresh_token=raw_refresh_token, expires_in=expires_in
    )
