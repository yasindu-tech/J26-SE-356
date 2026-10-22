import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.core.time import utc_now


class TappingAnalysis(Base):
    """One uploaded finger-tapping video and its result (see services/tapping_service.py).

    Holds no patient identifiers and no file name (only the optional pseudonymous label). The
    result JSON carries the score, the contributions, the signal and the one-page PDF; the video
    is deleted from disk when the analysis ends.
    """

    __tablename__ = "tapping_analyses"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    clinician_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("clinicians.id"), index=True)
    # "queued" | "running" | "scored" | "abstained" | "failed"
    status: Mapped[str] = mapped_column(String(20), default="queued")
    # "ge2" (primary study target) or "ge1" (secondary).
    target: Mapped[str] = mapped_column(String(3), default="ge2")
    # The same pseudonymous label as on MRI and gait analyses. Never a name.
    subject_label: Mapped[str | None] = mapped_column(String(32), default=None)
    error: Mapped[str | None] = mapped_column(String(500), default=None)
    result_json: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(), default=utc_now, server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(), default=None)
