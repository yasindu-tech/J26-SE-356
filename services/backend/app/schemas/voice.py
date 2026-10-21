"""Shapes on the wire for voice analyses. Field names follow models/voice/src/analyze_audio.py
(the PP1 result vector, decision D5, plus its explanations). Screening wording only: no field
says "diagnosis"."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

# Formats analyze_audio.py can open (libsndfile, or ffmpeg for the compressed ones).
VOICE_AUDIO_TYPES: tuple[str, ...] = (
    ".wav",
    ".flac",
    ".ogg",
    ".mp3",
    ".m4a",
    ".aac",
    ".webm",
)


class VoiceResult(BaseModel):
    """One recording's result. Abstained results have no score, range or triage."""

    # The voice module's output passes through whole; the fields below are checked.
    model_config = ConfigDict(extra="allow")

    person: str  # a hash of the file's bytes: never the file name
    modality: Literal["voice"]
    risk_score: float | None
    ci_lower: float | None
    ci_upper: float | None
    triage_priority: Literal["prioritised_referral", "not_prioritised"] | None
    abstained: bool
    quality_gate_passed: bool
    problems: list[str]
    pipeline_version: str
    n_training: int
    explanations: dict[str, Any]
