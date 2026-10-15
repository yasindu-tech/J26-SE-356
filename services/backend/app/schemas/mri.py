"""Shapes on the wire for MRI analyses. Field names follow models/mri/src/predict_scan.py
(MriResult, schema mri-result/1). Screening wording only: no field says "diagnosis"."""

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

AnalysisStatus = Literal["queued", "running", "scored", "abstained", "failed"]
FINAL_STATUSES: tuple[str, ...] = ("scored", "abstained", "failed")


class MriAnalysisCreated(BaseModel):
    id: uuid.UUID
    status: AnalysisStatus


class MriAnalysisStatus(BaseModel):
    id: uuid.UUID
    status: AnalysisStatus
    created_at: datetime
    finished_at: datetime | None = None
    steps: list[str]  # every step, in order
    done: list[str]  # steps finished so far
    current_step: str | None = None
    note: str = ""  # e.g. "DTI: correcting motion (eddy, the slow step)"
    error: str | None = None
    abstain_reason: str | None = None


class MriAnalysisList(BaseModel):
    items: list[MriAnalysisStatus]
    total: int


class MriScore(BaseModel):
    kind: Literal["log_odds"]  # raw model score; 0 is the learned boundary
    value: float
    interval_95: list[float]
    interval_method: str
    calibrated: bool
    n_training: int
    meaning: str


class MriFeatureVector(BaseModel):
    names: list[str]
    labels: list[str]
    raw: list[float]
    harmonised: list[float]  # the MRI embedding for the fusion model
    scanner_batch: dict[str, str]
    outside_training_range: list[str]


class MriDriver(BaseModel):
    feature: str
    label: str
    value: float | None = None
    contribution: float  # log-odds
    direction: Literal["towards PD", "towards control"]
    text: str


class MriExplanation(BaseModel):
    kind: str
    method: str
    base_value: float
    feature_attributions: dict[str, float]
    top_drivers: list[MriDriver]
    caveat: str


class MriSeriesInput(BaseModel):
    model_config = ConfigDict(extra="allow")  # series details, checks, QC picture
    found: bool


class MriValidation(BaseModel):
    usable: bool
    problems: list[str]


class MriResult(BaseModel):
    modality: Literal["mri"]
    schema_version: str
    status: Literal["scored", "abstained", "failed"]
    abstain_reason: str | None = None
    error: str | None = None
    model_version: str | None = None
    model_set: str | None = None
    availability: dict[str, bool] = {}
    inputs: dict[str, MriSeriesInput] = {}
    validation: MriValidation | None = None
    feature_vector: MriFeatureVector | None = None
    score: MriScore | None = None
    explanation: MriExplanation | None = None
    model_card: dict[str, Any] | None = None
    disclaimer: str = ""
