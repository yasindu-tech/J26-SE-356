import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class RefreshToken(Base):
    """Server-side record of an issued refresh token, keyed by its hash (the
    raw token is never stored). Logout revokes by setting `revoked_at` —
    this is what makes /auth/logout actually do something, rather than being
    a no-op the client just forgets a JWT for."""

    __tablename__ = "refresh_tokens"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    clinician_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("clinicians.id"), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime())
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(), default=None)
