"""Stopping an analysis: the process helper, the three runners and the three stop routes.

The helper and runner tests start real child processes (small Python scripts), so they prove
that a stop actually ends the script and what it started, not just that a flag was set.
"""

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.models.clinician import Clinician
from app.workers.gait_runner import GaitJob, SubprocessGaitRunner
from app.workers.mri_runner import MriJob, SubprocessRunner
from app.workers.stoppable import (
    STOPPED_MESSAGE,
    Stopped,
    request_stop,
    run_stoppable,
)
from app.workers.tapping_runner import SubprocessTappingRunner, TappingJob

from .test_gait import auth as gait_auth
from .test_gait import gait_env  # noqa: F401 - fixture
from .test_gait import upload as gait_upload
from .test_mri import auth as mri_auth
from .test_mri import mri_env  # noqa: F401 - fixture
from .test_mri import upload as mri_upload
from .test_tapping import auth as tapping_auth
from .test_tapping import tapping_env  # noqa: F401 - fixture
from .test_tapping import upload as tapping_upload


def _script(tmp_path: Path, body: str, name: str = "fake.py") -> Path:
    path = tmp_path / name
    path.write_text("import json, os, subprocess, sys, time\n" + body)
    return path


def _stop_later(folder: Path, after: float = 0.5) -> threading.Timer:
    timer = threading.Timer(after, request_stop, args=(folder,))
    timer.start()
    return timer


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


# --------------------------------------------------------------------------- the helper


def test_a_script_that_finishes_returns_its_output(tmp_path: Path) -> None:
    done = run_stoppable(
        [sys.executable, str(_script(tmp_path, "print('hello')\n"))],
        cwd=tmp_path,
        timeout_s=30,
        folder=tmp_path,
    )
    assert done.returncode == 0 and done.stdout.strip() == b"hello"


def test_a_stop_requested_before_the_start_never_runs_the_script(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    request_stop(tmp_path)
    with pytest.raises(Stopped):
        run_stoppable(
            [sys.executable, str(_script(tmp_path, f"open({str(marker)!r}, 'w')\n"))],
            cwd=tmp_path,
            timeout_s=30,
            folder=tmp_path,
        )
    assert not marker.exists()


def test_a_stop_ends_the_script_and_the_processes_it_started(tmp_path: Path) -> None:
    # The script starts a long-running child (as predict_scan.py starts eddy) and writes its pid.
    pid_file = tmp_path / "child.pid"
    body = (
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    timer = _stop_later(tmp_path, after=1.0)
    started = time.monotonic()
    with pytest.raises(Stopped):
        run_stoppable(
            [sys.executable, str(_script(tmp_path, body))],
            cwd=tmp_path,
            timeout_s=60,
            folder=tmp_path,
        )
    timer.join()
    assert time.monotonic() - started < 15  # not the script's 60 s
    child = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while _alive(child) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _alive(child)


def test_running_out_of_time_still_raises_a_timeout(tmp_path: Path) -> None:
    with pytest.raises(subprocess.TimeoutExpired):
        run_stoppable(
            [sys.executable, str(_script(tmp_path, "time.sleep(60)\n"))],
            cwd=tmp_path,
            timeout_s=1,
            folder=tmp_path,
        )


# --------------------------------------------------------------------------- the runners


def test_a_stopped_mri_run_leaves_nothing_on_disk(tmp_path: Path) -> None:
    folder = tmp_path / "job"
    folder.mkdir()
    job = MriJob(analysis_id=uuid.uuid4(), folder=folder, age=60.0, sex="F")
    job.upload.write_bytes(b"zip")
    settings = Settings(mri_python=sys.executable, mri_script=_script(tmp_path, "time.sleep(60)\n"))
    timer = _stop_later(folder)
    SubprocessRunner(settings).run(job)
    timer.join()
    assert not folder.exists()  # no upload, no work folder, no result to read


def test_a_stopped_gait_run_leaves_nothing_on_disk(tmp_path: Path) -> None:
    folder = tmp_path / "job"
    folder.mkdir()
    video = folder / "abc.mp4"
    video.write_bytes(b"\x00" * 50)
    job = GaitJob(uuid.uuid4(), folder, video, True)
    settings = Settings(
        gait_python=sys.executable, gait_script=_script(tmp_path, "time.sleep(60)\n")
    )
    timer = _stop_later(folder)
    SubprocessGaitRunner(settings).run(job)
    timer.join()
    assert not folder.exists()


def test_a_stopped_tapping_run_leaves_nothing_on_disk(tmp_path: Path) -> None:
    folder = tmp_path / "job"
    folder.mkdir()
    video = folder / "abc.mp4"
    video.write_bytes(b"\x00" * 50)
    job = TappingJob(uuid.uuid4(), folder, video, "ge2")
    settings = Settings(
        tapping_python=sys.executable, tapping_script=_script(tmp_path, "time.sleep(60)\n")
    )
    timer = _stop_later(folder)
    SubprocessTappingRunner(settings).run(job)
    timer.join()
    assert not folder.exists()


def test_a_run_that_was_not_stopped_still_writes_its_result(tmp_path: Path) -> None:
    folder = tmp_path / "job"
    folder.mkdir()
    video = folder / "abc.mp4"
    video.write_bytes(b"\x00" * 50)
    job = TappingJob(uuid.uuid4(), folder, video, "ge2")
    settings = Settings(tapping_python=sys.executable, tapping_script=_script(tmp_path, "pass\n"))
    SubprocessTappingRunner(settings).run(job)
    assert json.loads(job.result_file.read_text())["status"] == "failed"  # printed nothing valid


# --------------------------------------------------------------------------- the routes


def test_stopping_a_running_mri_analysis(
    client: TestClient,
    seeded_clinician: Clinician,
    mri_env,  # noqa: F811
) -> None:
    settings, runner = mri_env
    runner.finish_with = None  # still working
    h = mri_auth(client)
    aid = mri_upload(client, h).json()["id"]
    r = client.post(f"/api/v1/mri/analyses/{aid}/stop", headers=h)
    assert r.status_code == 200
    assert r.json()["status"] == "failed" and r.json()["error"] == STOPPED_MESSAGE
    assert (settings.mri_work_dir / aid / "stop").exists()  # the worker is asked to end it
    # The progress screen asks for the result once the status is final: it must be there.
    result = client.get(f"/api/v1/mri/analyses/{aid}/result", headers=h)
    assert result.status_code == 200 and result.json()["status"] == "failed"
    again = client.get(f"/api/v1/mri/analyses/{aid}", headers=h).json()
    assert again["status"] == "failed" and again["error"] == STOPPED_MESSAGE


def test_stopping_a_finished_mri_analysis_changes_nothing(
    client: TestClient,
    seeded_clinician: Clinician,
    mri_env,  # noqa: F811
) -> None:
    h = mri_auth(client)
    aid = mri_upload(client, h).json()["id"]  # the fake runner finishes it as scored
    assert client.get(f"/api/v1/mri/analyses/{aid}", headers=h).json()["status"] == "scored"
    r = client.post(f"/api/v1/mri/analyses/{aid}/stop", headers=h)
    assert r.status_code == 200 and r.json()["status"] == "scored" and r.json()["error"] is None


def test_stopping_an_unknown_mri_analysis_is_not_found(
    client: TestClient,
    seeded_clinician: Clinician,
    mri_env,  # noqa: F811
) -> None:
    r = client.post(f"/api/v1/mri/analyses/{uuid.uuid4()}/stop", headers=mri_auth(client))
    assert r.status_code == 404


def test_stopping_a_running_gait_analysis(
    client: TestClient,
    seeded_clinician: Clinician,
    gait_env,  # noqa: F811
) -> None:
    settings, runner = gait_env
    runner.finish_with = None
    h = gait_auth(client)
    aid = gait_upload(client, h).json()["id"]
    r = client.post(f"/api/v1/gait/analyses/{aid}/stop", headers=h)
    assert r.status_code == 200
    assert r.json()["status"] == "failed" and r.json()["error"] == STOPPED_MESSAGE
    assert (settings.gait_work_dir / aid / "stop").exists()


def test_stopping_a_running_tapping_analysis(
    client: TestClient,
    seeded_clinician: Clinician,
    tapping_env,  # noqa: F811
) -> None:
    settings, runner = tapping_env
    runner.finish_with = None
    h = tapping_auth(client)
    aid = tapping_upload(client, h).json()["id"]
    r = client.post(f"/api/v1/tapping/analyses/{aid}/stop", headers=h)
    assert r.status_code == 200
    assert r.json()["status"] == "failed" and r.json()["error"] == STOPPED_MESSAGE
    assert (settings.tapping_work_dir / aid / "stop").exists()
