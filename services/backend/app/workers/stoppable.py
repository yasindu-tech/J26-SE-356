"""Run a model script as a child process that a clinician can stop.

The request side stops an analysis by writing an empty ``stop`` file into the analysis folder
(``request_stop``). The worker thread polls for that file while the script runs and, when it
appears, ends the script together with everything it started (eddy, ANTs, MediaPipe...),
because the script runs in its own process group. A file is used, not a shared object, so a
stop also works on an analysis that is still queued, and the request side needs no handle on
the runner.

POSIX only (process groups); the backend runs on macOS or Linux.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

STOP_FILE = "stop"
POLL_SECONDS = 1.0
GRACE_SECONDS = 10.0
STOPPED_MESSAGE = "Stopped by the clinician before it finished"


class Stopped(Exception):  # noqa: N818 - reads better than "StoppedError" where it is raised
    """The clinician stopped the analysis."""


def request_stop(folder: Path) -> None:
    """Ask the worker to stop this analysis. Harmless if it already ended or never started."""
    if folder.is_dir():
        (folder / STOP_FILE).touch()


def stop_requested(folder: Path) -> bool:
    return (folder / STOP_FILE).exists()


def clean_up_if_stopped(folder: Path) -> bool:
    """After a run: if it was stopped, delete everything it left. The request side has already
    recorded the analysis as stopped, so nothing in the folder will be read again."""
    if not stop_requested(folder):
        return False
    shutil.rmtree(folder, ignore_errors=True)
    return True


def _end(proc: subprocess.Popen[bytes]) -> None:
    """End the script's process group: SIGTERM, then SIGKILL after a grace period."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def run_stoppable(
    cmd: list[str], *, cwd: Path, timeout_s: float, folder: Path
) -> subprocess.CompletedProcess[bytes]:
    """Like ``subprocess.run(cmd, capture_output=True, timeout=..., check=False)``, but ends the
    script and its children when a stop is requested for ``folder``.

    Raises ``Stopped`` when a stop was requested (also before the script starts), and
    ``subprocess.TimeoutExpired`` when the time runs out. Output goes to temporary files, not
    pipes, so a script that prints a lot cannot block.
    """
    if stop_requested(folder):
        raise Stopped
    deadline = time.monotonic() + timeout_s
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=out, stderr=err, start_new_session=True)
        try:
            while True:
                try:
                    code = proc.wait(
                        timeout=min(POLL_SECONDS, max(0.01, deadline - time.monotonic()))
                    )
                    break
                except subprocess.TimeoutExpired:
                    if stop_requested(folder):
                        raise Stopped from None
                    if time.monotonic() >= deadline:
                        raise subprocess.TimeoutExpired(cmd, timeout_s) from None
        finally:
            _end(proc)  # also ends children left behind by a script that has exited
        out.seek(0)
        err.seek(0)
        return subprocess.CompletedProcess(cmd, code, out.read(), err.read())
