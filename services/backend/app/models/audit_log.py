import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class AuditLogEntry(Base):
    """Who / when / what, per Security-And-Data-Protection.md's audit-logging
    control. Scoped to authentication events (login, failed login, logout)
    in this PR — logging every *patient-data* access is broader and belongs
    with the Patient/Assessment services, not this one."""

    __tablename__ = "audit_log"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    # Nullable: a failed login with an unrecognised username has no clinician to attribute it to.
    clinician_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("clinicians.id"), nullable=True
    )
    # "login" | "login_failed" | "logout" | "refresh_token_reuse_detected"
    action: Mapped[str] = mapped_column(String(50))
    detail: Mapped[str | None] = mapped_column(String(255), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(), server_default=func.now())
