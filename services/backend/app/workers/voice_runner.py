"""Runs models/voice/src/analyze_audio.py for one recording, as a separate process.

A sustained "aaah" takes a few seconds (Praat and librosa on the CPU), so unlike MRI and gait the
request waits for it. The process uses the voice module's Python, writes its result to a JSON
file in the job folder, and the caller deletes the folder afterwards.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.core.config import REPO_ROOT, Settings
from app.core.errors import VoiceAnalysisFailedError


@dataclass(frozen=True)
class VoiceJob:
    folder: Path  # data/backend/voice/<id>/ : the audio and result.json
    audio: Path

    @property
    def result_file(self) -> Path:
        return self.folder / "result.json"


class VoiceRunner(Protocol):
    def run(self, job: VoiceJob) -> dict: ...


class SubprocessVoiceRunner:
    """One analyze_audio.py process per request."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def command(self, job: VoiceJob) -> list[str]:
        s = self.settings
        return [
            s.voice_python,
            str(s.voice_script),
            str(job.audio),
            "--model", s.voice_model,
            "--out", str(job.result_file),
        ]  # fmt: skip

    def run(self, job: VoiceJob) -> dict:
        try:
            done = subprocess.run(
                self.command(job),
                cwd=REPO_ROOT,
                capture_output=True,  # tool output is not logged
                timeout=self.settings.voice_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise VoiceAnalysisFailedError("The voice analysis timed out") from None
        except OSError:
            raise VoiceAnalysisFailedError("The voice analysis could not start") from None
        if done.returncode != 0 or not job.result_file.exists():
            raise VoiceAnalysisFailedError(
                "The voice analysis could not run: the voice model may not be trained yet"
            )
        try:
            results = json.loads(job.result_file.read_text())
        except (OSError, ValueError):
            raise VoiceAnalysisFailedError(
                "The voice analysis wrote an unreadable result"
            ) from None
        if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
            raise VoiceAnalysisFailedError("The voice analysis wrote an unexpected result")
        return results[0]
