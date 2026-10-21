import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.core.time import utc_now


class GaitAnalysis(Base):
    """One uploaded walking video and its result (see services/gait_service.py).

    Holds no patient identifiers and no file name (only the optional pseudonymous label): the result JSON carries the walk's
    features, scores and explanation only. The video is deleted from disk when the
    analysis ends.
    """

    __tablename__ = "gait_analyses"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    clinician_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("clinicians.id"), index=True)
    # "queued" | "running" | "scored" | "abstained" | "failed"
    status: Mapped[str] = mapped_column(String(20), default="queued")
    one_person_confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    # The same pseudonymous label as on MRI analyses (validated at the API). Never a name.
    subject_label: Mapped[str | None] = mapped_column(String(32), default=None)
    error: Mapped[str | None] = mapped_column(String(500), default=None)
    result_json: Mapped[str | None] = mapped_column(Text, default=None)
    # Set in Python (naive UTC, app/core/time.py) so the elapsed time is right on any database.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(), default=utc_now, server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(), default=None)
