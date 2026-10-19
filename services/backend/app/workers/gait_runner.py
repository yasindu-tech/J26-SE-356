"""Runs models/gait/src/predict_video.py for one video, outside the request.

A video takes about a minute (MediaPipe on the CPU), so a request never waits on it. One video
runs at a time, in a separate process with the gait module's Python, in a worker of its own so a
gait video never queues behind an MRI scan. The worker writes started.json when the process
begins; the script writes result.json. If the backend restarts while a video runs, the analysis
is marked failed when it is next read (see gait_service.refresh_from_disk).
"""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.core.config import REPO_ROOT, Settings

SCHEMA_VERSION = "gait-result/1"
CANNOT_START = 2  # predict_video.py's exit code when the model bundle or pose model is unusable


@dataclass(frozen=True)
class GaitJob:
    analysis_id: uuid.UUID
    folder: Path  # data/backend/gait/<id>/ : the video, started.json, result.json
    video: Path
    one_person_confirmed: bool

    @property
    def started_file(self) -> Path:
        return self.folder / "started.json"

    @property
    def result_file(self) -> Path:
        return self.folder / "result.json"


class GaitRunner(Protocol):
    def submit(self, job: GaitJob) -> None: ...


def failed_result(error: str) -> dict:
    return {
        "modality": "gait",
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "error": error,
    }


class SubprocessGaitRunner:
    """One predict_video.py process at a time, in a background thread."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gait")

    def submit(self, job: GaitJob) -> None:
        self._pool.submit(self.run, job)

    def command(self, job: GaitJob) -> list[str]:
        s = self.settings
        cmd = [
            s.gait_python,
            str(s.gait_script),
            str(job.video),
            "--bundle", str(s.gait_bundle_dir),
            "--out", str(job.result_file),
        ]  # fmt: skip
        if job.one_person_confirmed:
            cmd.append("--one-person-confirmed")
        return cmd

    def run(self, job: GaitJob) -> None:
        error = "the gait analysis ended without a result"
        try:
            job.started_file.write_text(json.dumps({"t": time.time()}))
            done = subprocess.run(
                self.command(job),
                cwd=REPO_ROOT,
                capture_output=True,  # tool output is not logged
                timeout=self.settings.gait_timeout_minutes * 60,
                check=False,
            )
            if done.returncode == CANNOT_START:
                error = "the gait analysis could not start: the model files are missing or unusable"
        except subprocess.TimeoutExpired:
            error = "the gait analysis timed out"
        except OSError:
            error = "the gait analysis could not start"
        finally:
            job.video.unlink(missing_ok=True)  # the video is never kept
        if not job.result_file.exists():
            job.result_file.write_text(json.dumps(failed_result(error)))
