import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Clinician(Base):
    """A clinician account. Provisioning (how an account gets created) is not
    part of this PR — see Security-And-Data-Protection.md, which specifies
    login/authorization requirements but not a self-signup flow. Accounts are
    expected to be created out-of-band (admin/DB seed) for now."""

    __tablename__ = "clinicians"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str] = mapped_column(String(255))
    # Single role today ("clinician"). Security-And-Data-Protection.md says
    # authorization is role-based but doesn't define a role hierarchy —
    # this column exists so that can be added without a schema change.
    role: Mapped[str] = mapped_column(String(50), default="clinician")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(), server_default=func.now())
