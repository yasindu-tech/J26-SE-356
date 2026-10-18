"""Runs models/mri/src/predict_scan.py for one analysis, outside the request.

MRI takes minutes to hours (eddy is the slow step), so a request never waits on it
(services/backend/README.md: inference is async). One analysis runs at a time, in a
separate process with the models' Python environment; the request side only reads the
status and result files that process writes.

This is the research-prototype stand-in for the Redis queue + ML worker in Backend-Spec:
same contract (submit, then poll), no extra infrastructure. If the backend restarts while
an analysis runs, that analysis is marked failed when it is next read (see
mri_service.refresh_from_disk).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.core.config import REPO_ROOT, Settings


@dataclass(frozen=True)
class MriJob:
    analysis_id: uuid.UUID
    folder: Path  # data/backend/mri/<id>/ : upload.zip, status.json, result.json
    age: float
    sex: str
    eddy_profile: str = "full"  # "full" | "fast", validated at the API

    @property
    def upload(self) -> Path:
        return self.folder / "upload.zip"

    @property
    def status_file(self) -> Path:
        return self.folder / "status.json"

    @property
    def result_file(self) -> Path:
        return self.folder / "result.json"


class MriRunner(Protocol):
    def submit(self, job: MriJob) -> None: ...


class SubprocessRunner:
    """One predict_scan.py process at a time, in a background thread."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mri")

    def submit(self, job: MriJob) -> None:
        self._pool.submit(self.run, job)

    def command(self, job: MriJob) -> list[str]:
        s = self.settings
        return [
            s.mri_python,
            str(s.mri_script),
            "--input", str(job.upload),
            "--age", str(job.age),
            "--sex", job.sex,
            "--out", str(job.result_file),
            "--status-file", str(job.status_file),
            "--work", str(job.folder / "work"),
            "--eddy-threads", str(s.mri_eddy_threads),
            "--eddy-profile", job.eddy_profile,
        ]  # fmt: skip

    def run(self, job: MriJob) -> None:
        try:
            subprocess.run(
                self.command(job),
                cwd=REPO_ROOT,
                capture_output=True,  # tool output can contain file names: not logged
                timeout=self.settings.mri_timeout_hours * 3600,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            reason = (
                "timed out" if isinstance(exc, subprocess.TimeoutExpired) else "could not start"
            )
            if not job.result_file.exists():
                job.result_file.write_text(
                    json.dumps(
                        {
                            "modality": "mri",
                            "schema_version": "mri-result/1",
                            "status": "failed",
                            "error": f"the MRI analysis {reason}",
                        }
                    )
                )
        finally:
            # The DICOM upload (which may carry identifiers in its headers) and every
            # converted or processed image are removed as soon as the analysis ends.
            job.upload.unlink(missing_ok=True)
            shutil.rmtree(job.folder / "work", ignore_errors=True)
        if not job.result_file.exists():
            job.result_file.write_text(
                json.dumps(
                    {
                        "modality": "mri",
                        "schema_version": "mri-result/1",
                        "status": "failed",
                        "error": "the MRI analysis ended without a result",
                    }
                )
            )
