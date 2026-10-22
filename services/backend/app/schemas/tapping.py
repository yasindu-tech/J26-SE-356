"""Shapes on the wire for finger-tapping analyses. Field names follow
models/tapping/src/predict_clip.py (schema tapping-result/1). Screening wording only: no field
says "diagnosis". A number that was not available is null, never 0."""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from app.schemas.mri import AnalysisStatus

TAPPING_VIDEO_TYPES: tuple[str, ...] = (".mp4", ".mov", ".m4v")
TappingTarget = Literal["ge2", "ge1"]


class TappingAnalysisCreated(BaseModel):
    id: uuid.UUID
    status: AnalysisStatus
    target: TappingTarget = "ge2"
    subject_label: str | None = None


class TappingAnalysisStatus(BaseModel):
    id: uuid.UUID
    status: AnalysisStatus
    target: TappingTarget = "ge2"
    created_at: datetime
    finished_at: datetime | None = None
    elapsed_s: float  # since the upload; the analysis usually takes under a minute
    subject_label: str | None = None
    error: str | None = None
    abstain_reason: str | None = None  # why the video could not be scored, in plain words


class TappingAnalysisList(BaseModel):
    items: list[TappingAnalysisStatus]
    total: int


class TappingRecording(BaseModel):
    duration_s: float | None = None
    fps: float | None = None
    hand_found_pct: float | None = None
    n_taps: int


class TappingFlag(BaseModel):
    name: str
    text: str


class TappingQuality(BaseModel):
    flags: list[TappingFlag]
    notes: list[str]


class TappingScore(BaseModel):
    value: float  # a model score on a 0 to 1 scale: not a calibrated probability
    band_low: float
    band_high: float
    band_basis: str  # the band is a spread of 25 cross-validation models, not a confidence interval
    threshold: float
    above_threshold: bool
    cv_auroc: float | None = None
    cv_ci_low: float | None = None
    cv_ci_high: float | None = None


class TappingContribution(BaseModel):
    feature_label: str
    value_text: str
    percentile_text: str
    shap: float | None = None  # log-odds
    sentence: str


class TappingCurve(BaseModel):
    times: list[float | None]
    amps: list[float | None]
    line_x: list[float | None]
    line_y: list[float | None]
    slope_sentence: str
    count_sentence: str
    note: str
    usable: bool


class TappingSignal(BaseModel):
    t: list[float | None]
    smoothed: list[float | None]
    peak_t: list[float | None]
    peak_v: list[float | None]


class TappingPreview(BaseModel):
    t: float | None = None
    jpeg_base64: str


class TappingModelInfo(BaseModel):
    demo_model_sha8: str
    folds_sha8: str


class TappingResult(BaseModel):
    """Either a scored result, or (status "failed") the plain reason it could not be scored."""

    schema_version: str
    status: Literal["scored", "failed"]
    clip_ref: str | None = None  # sha256[:8] of the video bytes, never the file name
    target: TappingTarget
    reason: str | None = None
    recording: TappingRecording | None = None
    quality: TappingQuality | None = None
    score: TappingScore | None = None
    contributions: list[TappingContribution] = []
    curve: TappingCurve | None = None
    signal: TappingSignal | None = None
    preview_frames: list[TappingPreview] = []
    model: TappingModelInfo | None = None
    pdf_base64: str | None = None
