"""Shapes on the wire for gait analyses. Field names follow models/gait/src/predict_video.py
(schema gait-result/1). Screening wording only: no field says "diagnosis"."""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from app.schemas.mri import AnalysisStatus

GAIT_VIDEO_TYPES: tuple[str, ...] = (".mp4", ".mov", ".m4v", ".avi")


class GaitAnalysisCreated(BaseModel):
    id: uuid.UUID
    status: AnalysisStatus
    subject_label: str | None = None


class GaitAnalysisStatus(BaseModel):
    id: uuid.UUID
    status: AnalysisStatus
    created_at: datetime
    finished_at: datetime | None = None
    elapsed_s: float  # since the upload; the analysis usually takes about a minute
    one_person_confirmed: bool
    subject_label: str | None = None
    error: str | None = None
    abstain_reason: str | None = None  # why the video could not be scored, in plain words


class GaitAnalysisList(BaseModel):
    items: list[GaitAnalysisStatus]
    total: int


class GaitCheck(BaseModel):
    value: float
    limit: float
    ok: bool


class GaitGate(BaseModel):
    passed: bool
    reasons: list[str]
    warnings: list[str]
    checks: dict[str, GaitCheck]


class GaitWalk(BaseModel):
    seconds: float
    start_s: float
    end_s: float
    fps: float


class GaitTopFeature(BaseModel):
    name: str
    value: float
    shap: float  # log-odds of the predicted class
    direction: Literal["towards", "away"]
    strength: str
    sentence: str


class GaitResult(BaseModel):
    modality: Literal["gait"]
    schema_version: str
    status: Literal["scored", "abstained", "failed"]
    error: str | None = None
    available: bool = False
    video: str | None = None  # a short hash, never the file name
    reasons: list[str] = []
    warnings: list[str] = []
    gate: GaitGate | None = None
    walk: GaitWalk | None = None
    predicted_class: int | None = None
    predicted_label: str | None = None
    class_labels: list[str] = []
    probabilities: list[float] | None = None  # uncalibrated model scores, not chances
    probabilities_calibrated: bool = False
    expected_severity: float | None = None
    expected_severity_note: str = ""
    top_features: list[GaitTopFeature] = []
    feature_vector: dict[str, float] | None = None
    model_version: str | None = None
    one_person_confirmed: bool = False
