"""Runs models/tapping/src/predict_clip.py for one video, outside the request.

A video takes under a minute (MediaPipe on the CPU), so a request never waits on it. One video
runs at a time, in a separate process with the tapping module's Python, in a worker of its own so
a tapping video never queues behind an MRI scan or a gait video. The worker writes started.json
when the work begins. predict_clip.py prints one JSON object on stdout and always exits 0 for a
video it cannot score (status "failed" with a plain reason); that object becomes result.json. If
the program cannot run at all, or the backend restarts while a video runs, the analysis is marked
failed when it is next read (see tapping_service.refresh_from_disk).
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
from app.workers.stoppable import Stopped, clean_up_if_stopped, run_stoppable

SCHEMA_VERSION = "tapping-result/1"
MAX_RESULT_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class TappingJob:
    analysis_id: uuid.UUID
    folder: Path  # data/backend/tapping/<id>/ : the video, started.json, result.json
    video: Path
    target: str

    @property
    def started_file(self) -> Path:
        return self.folder / "started.json"

    @property
    def result_file(self) -> Path:
        return self.folder / "result.json"


class TappingRunner(Protocol):
    def submit(self, job: TappingJob) -> None: ...


def failed_result(reason: str, target: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "reason": reason,
        "clip_ref": None,
        "target": target,
    }


def valid_result(text: str) -> dict | None:
    """The program's object, or None when its output is not one JSON object of a sane size."""
    if not text.strip() or len(text) > MAX_RESULT_BYTES:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        return None
    return data if data.get("status") in ("scored", "failed") else None


class SubprocessTappingRunner:
    """One predict_clip.py process at a time, in a background thread."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tapping")

    def submit(self, job: TappingJob) -> None:
        self._pool.submit(self.run, job)

    def command(self, job: TappingJob) -> list[str]:
        s = self.settings
        return [s.tapping_python, str(s.tapping_script), "--video", str(job.video),
                "--target", job.target]  # fmt: skip

    def run(self, job: TappingJob) -> None:
        reason = "The analysis ended without a result."
        result: dict | None = None
        try:
            job.started_file.write_text(json.dumps({"t": time.time()}))
            done = run_stoppable(  # the program's output is not logged
                self.command(job),
                cwd=REPO_ROOT,
                timeout_s=self.settings.tapping_timeout_minutes * 60,
                folder=job.folder,
            )
            result = valid_result(done.stdout.decode("utf-8", errors="replace"))
        except Stopped:
            pass  # the request side has recorded it; the folder is removed below
        except subprocess.TimeoutExpired:
            reason = "The analysis took too long and was stopped."
        except OSError:
            reason = "The analysis could not start."
        finally:
            job.video.unlink(missing_ok=True)  # the uploaded video itself is never kept
        if clean_up_if_stopped(job.folder):
            return
        job.result_file.write_text(json.dumps(result or failed_result(reason, job.target)))
