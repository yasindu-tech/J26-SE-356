"""Runs models/gait/src/predict_video.py for one video, outside the request.

A video takes about a minute (MediaPipe on the CPU), so a request never waits on it. One video
runs at a time, in separate processes with the gait module's Python, in a worker of its own so a
gait video never queues behind an MRI scan. The worker writes started.json when the work begins.
The scoring script writes predict.json and the pose trace (trace.npz); make_overlay_video.py then
draws overlay.mp4 from that trace without running the pose model again. predict.json becomes
result.json once both have ended, so a result never shows up before its overlay. A video the
scoring refuses early leaves no trace and so gets no overlay. If the backend restarts while a
video runs, the analysis is marked failed when it is next read (see
gait_service.refresh_from_disk).
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
    def predict_file(self) -> Path:
        return self.folder / "predict.json"

    @property
    def trace_file(self) -> Path:
        return self.folder / "trace.npz"

    @property
    def result_file(self) -> Path:
        return self.folder / "result.json"

    @property
    def overlay_file(self) -> Path:
        return self.folder / "overlay.mp4"


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
            "--out", str(job.predict_file),
            "--trace", str(job.trace_file),
        ]  # fmt: skip
        if job.one_person_confirmed:
            cmd.append("--one-person-confirmed")
        return cmd

    def overlay_command(self, job: GaitJob) -> list[str]:
        s = self.settings
        return [
            s.gait_python,
            str(s.gait_overlay_script),
            str(job.video),
            "--trace",
            str(job.trace_file),
            "--out",
            str(job.overlay_file),
        ]

    def run(self, job: GaitJob) -> None:
        error = "the gait analysis ended without a result"
        deadline = time.monotonic() + self.settings.gait_timeout_minutes * 60
        try:
            job.started_file.write_text(json.dumps({"t": time.time()}))
            done = self._run(self.command(job), deadline)
            if done.returncode == CANNOT_START:
                error = "the gait analysis could not start: the model files are missing or unusable"
            self._draw_overlay(job, deadline)
        except subprocess.TimeoutExpired:
            error = "the gait analysis timed out"
        except OSError:
            error = "the gait analysis could not start"
        finally:
            job.video.unlink(missing_ok=True)  # the uploaded video itself is never kept
            job.trace_file.unlink(missing_ok=True)
        if job.predict_file.exists():
            job.predict_file.replace(job.result_file)
        else:
            job.overlay_file.unlink(missing_ok=True)
            job.result_file.write_text(json.dumps(failed_result(error)))

    @staticmethod
    def _run(cmd: list[str], deadline: float) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,  # tool output is not logged
            timeout=max(1.0, deadline - time.monotonic()),
            check=False,
        )

    def _draw_overlay(self, job: GaitJob, deadline: float) -> None:
        """The overlay is a bonus: any failure here leaves the result as it is, without one."""
        if not job.trace_file.exists():
            return
        try:
            made = self._run(self.overlay_command(job), deadline).returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            made = False
        if not made:
            job.overlay_file.unlink(missing_ok=True)
