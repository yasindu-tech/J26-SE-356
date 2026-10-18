import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class MriAnalysis(Base):
    """One uploaded MRI study and its result (see services/mri_service.py).

    Holds no patient identifiers: age and sex are the two covariates the MRI model's
    scanner adjustment needs, and the result JSON carries features, scores and
    explanations only. The uploaded DICOM files are not stored here and are deleted
    from disk when the analysis ends.
    """

    __tablename__ = "mri_analyses"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    clinician_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("clinicians.id"), index=True)
    # "queued" | "running" | "scored" | "abstained" | "failed"
    status: Mapped[str] = mapped_column(String(20), default="queued")
    age: Mapped[float] = mapped_column(Float)
    sex: Mapped[str] = mapped_column(String(1))  # "M" | "F"
    # "full" (what the model was trained on) | "fast" (fewer eddy iterations, not validated)
    eddy_profile: Mapped[str] = mapped_column(String(10), default="full", server_default="full")
    error: Mapped[str | None] = mapped_column(String(500), default=None)
    result_json: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(), default=None)
