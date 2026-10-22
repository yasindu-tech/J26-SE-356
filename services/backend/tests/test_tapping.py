"""Finger-tapping analysis endpoints. predict_clip.py is replaced by a fake runner that writes
the files the real one writes, so these tests need no video, MediaPipe or model."""

import json
import shutil
import sys
import time
import uuid
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from sqlalchemy.orm import Session

from app.api.deps import get_tapping_runner
from app.core.config import Settings, get_settings
from app.core.security import hash_password
from app.main import app
from app.models.audit_log import AuditLogEntry
from app.models.clinician import Clinician
from app.workers.tapping_runner import (
    SubprocessTappingRunner,
    TappingJob,
    failed_result,
    valid_result,
)

SCORED = {
    "schema_version": "tapping-result/1",
    "status": "scored",
    "clip_ref": "f0e37b01",
    "target": "ge2",
    "recording": {"duration_s": 18.2, "fps": 59.9, "hand_found_pct": 100.0, "n_taps": 29},
    "quality": {"flags": [{"name": "low_range", "text": "the range is small"}], "notes": ["n"]},
    "score": {
        "value": 0.09,
        "band_low": 0.02,
        "band_high": 0.1,
        "band_basis": "5-95% of 25 cross-validation models",
        "threshold": 0.5,
        "above_threshold": False,
        "cv_auroc": 0.87,
        "cv_ci_low": 0.81,
        "cv_ci_high": 0.92,
    },
    "contributions": [
        {
            "feature_label": "Tapping rate",
            "value_text": "1.73 taps/s",
            "percentile_text": "70th percentile of the research clips",
            "shap": -0.5,
            "sentence": "Tapping rate pushed the score down by 0.50 (log-odds).",
        }
    ],
    "curve": {
        "times": [1.0, 2.0],
        "amps": [1.1, None],
        "line_x": [1.0, 2.0],
        "line_y": [1.1, 1.2],
        "slope_sentence": "rises",
        "count_sentence": "Taps shown: 2 of 2.",
        "note": "note",
        "usable": True,
    },
    "signal": {"t": [0.0, 1.0], "smoothed": [0.5, 0.6], "peak_t": [0.5], "peak_v": [0.6]},
    "preview_frames": [{"t": 0.5, "jpeg_base64": "/9j/4AAQ"}],
    "model": {"demo_model_sha8": "1c623df9", "folds_sha8": "ae519116"},
    "pdf_base64": "JVBERi0xLjQK",
}
UNREADABLE = {
    "schema_version": "tapping-result/1",
    "status": "failed",
    "reason": "Hand not found in enough frames; keep the hand in view.",
    "clip_ref": "f0e37b01",
    "target": "ge2",
}


class FakeRunner:
    """Writes what the real runner and predict_clip.py would: started.json, then result.json."""

    def __init__(self, finish_with: dict | None) -> None:
        self.finish_with = finish_with
        self.jobs: list[TappingJob] = []

    def submit(self, job: TappingJob) -> None:
        self.jobs.append(job)
        assert job.video.exists() and job.video.parent == job.folder  # saved before the job
        job.started_file.write_text(json.dumps({"t": time.time()}))
        if self.finish_with is not None:
            job.video.unlink()
            job.result_file.write_text(json.dumps(self.finish_with))


@pytest.fixture
def tapping_env(tmp_path: Path) -> Generator[tuple[Settings, FakeRunner], None, None]:
    settings = Settings(tapping_work_dir=tmp_path / "tapping", tapping_max_upload_mb=1)
    runner = FakeRunner(SCORED)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_tapping_runner] = lambda: runner
    yield settings, runner
    app.dependency_overrides.pop(get_settings, None)
    app.dependency_overrides.pop(get_tapping_runner, None)


def auth(client: TestClient, username: str = "j.fernando") -> dict[str, str]:
    token = client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": "correct-horse-battery-staple"},
    ).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def upload(
    client: TestClient,
    headers: dict,
    data: bytes = b"\x00" * 300,
    name: str = "Patient Name 2026-10-18.MOV",
    **form: str,
) -> Response:
    files = {"file": (name, data, "video/quicktime")}
    return client.post("/api/v1/tapping/analyses", files=files, data=form, headers=headers)


# --------------------------------------------------------------------------- the flow
def test_upload_runs_and_the_result_is_the_tapping_record(
    client: TestClient, seeded_clinician: Clinician, tapping_env, db_session: Session
) -> None:
    settings, runner = tapping_env
    h = auth(client)
    created = upload(client, h)
    assert created.status_code == 202 and created.json()["status"] == "queued"
    assert created.json()["target"] == "ge2"
    aid = created.json()["id"]
    job = runner.jobs[0]
    assert job.video.name == f"{aid}.mov" and job.target == "ge2"  # named by the analysis

    status = client.get(f"/api/v1/tapping/analyses/{aid}", headers=h).json()
    assert status["status"] == "scored" and status["abstain_reason"] is None
    assert status["elapsed_s"] >= 0 and status["target"] == "ge2"
    body = client.get(f"/api/v1/tapping/analyses/{aid}/result", headers=h).json()
    assert body["status"] == "scored" and body["schema_version"] == "tapping-result/1"
    assert body["score"]["value"] == 0.09 and body["clip_ref"] == "f0e37b01"
    assert body["curve"]["amps"] == [1.1, None]  # a missing amplitude stays null, never 0
    assert body["pdf_base64"] == "JVBERi0xLjQK"
    assert not (settings.tapping_work_dir / aid).exists()  # nothing of the upload left on disk
    actions = {a.action for a in db_session.query(AuditLogEntry).all()}
    assert {"tapping_upload", "tapping_result_view"} <= actions


def test_the_result_is_returned_exactly_as_the_program_wrote_it(
    client: TestClient, seeded_clinician: Clinician, tapping_env
) -> None:
    # The desktop app checks the exact set of keys, so none may be added (as null) or dropped.
    _, runner = tapping_env
    h = auth(client)
    for written in (SCORED, UNREADABLE):
        runner.finish_with = written
        aid = upload(client, h).json()["id"]
        assert client.get(f"/api/v1/tapping/analyses/{aid}/result", headers=h).json() == written


def test_the_result_is_stored_in_the_database_not_on_disk(
    client: TestClient, seeded_clinician: Clinician, tapping_env, db_session: Session
) -> None:
    from app.models.tapping_analysis import TappingAnalysis

    h = auth(client)
    aid = upload(client, h).json()["id"]
    client.get(f"/api/v1/tapping/analyses/{aid}", headers=h)
    row = db_session.get(TappingAnalysis, uuid.UUID(aid))
    assert row is not None and row.status == "scored"
    assert json.loads(row.result_json or "")["clip_ref"] == "f0e37b01"


def test_the_target_can_be_the_secondary_one(
    client: TestClient, seeded_clinician: Clinician, tapping_env
) -> None:
    _, runner = tapping_env
    h = auth(client)
    created = upload(client, h, target="ge1")
    assert created.json()["target"] == "ge1" and runner.jobs[0].target == "ge1"
    assert upload(client, h, target="ge3").status_code == 422


def test_the_upload_name_is_stored_nowhere(
    client: TestClient, seeded_clinician: Clinician, tapping_env, db_session: Session
) -> None:
    h = auth(client)
    aid = upload(client, h).json()["id"]
    client.get(f"/api/v1/tapping/analyses/{aid}/result", headers=h)
    stored = " ".join(
        [str(a.detail) for a in db_session.query(AuditLogEntry).all()]
        + [client.get("/api/v1/tapping/analyses", headers=h).text]
    )
    assert "Patient" not in stored


def test_a_video_that_cannot_be_scored_is_abstained_with_its_reason(
    client: TestClient, seeded_clinician: Clinician, tapping_env
) -> None:
    _, runner = tapping_env
    runner.finish_with = UNREADABLE
    h = auth(client)
    aid = upload(client, h).json()["id"]
    status = client.get(f"/api/v1/tapping/analyses/{aid}", headers=h).json()
    assert status["status"] == "abstained" and status["error"] is None
    assert status["abstain_reason"] == UNREADABLE["reason"]
    body = client.get(f"/api/v1/tapping/analyses/{aid}/result", headers=h).json()
    assert body["status"] == "failed" and "score" not in body
    assert body["reason"] == UNREADABLE["reason"]


def test_status_is_running_while_the_video_is_analysed(
    client: TestClient, seeded_clinician: Clinician, tapping_env
) -> None:
    _, runner = tapping_env
    runner.finish_with = None
    h = auth(client)
    aid = upload(client, h).json()["id"]
    status = client.get(f"/api/v1/tapping/analyses/{aid}", headers=h).json()
    assert status["status"] == "running" and status["finished_at"] is None
    early = client.get(f"/api/v1/tapping/analyses/{aid}/result", headers=h)
    assert early.status_code == 409 and early.json()["error"]["code"] == "TAPPING_RESULT_NOT_READY"


def test_a_failure_of_the_runner_itself_is_failed_not_abstained(
    client: TestClient, seeded_clinician: Clinician, tapping_env
) -> None:
    _, runner = tapping_env
    runner.finish_with = failed_result("The analysis took too long and was stopped.", "ge2")
    h = auth(client)
    aid = upload(client, h).json()["id"]
    status = client.get(f"/api/v1/tapping/analyses/{aid}", headers=h).json()
    assert status["status"] == "failed"
    assert status["error"] == "The analysis took too long and was stopped."


def test_a_lost_job_is_marked_failed(
    client: TestClient, seeded_clinician: Clinician, tapping_env
) -> None:
    settings, runner = tapping_env
    runner.finish_with = None
    h = auth(client)
    aid = upload(client, h).json()["id"]
    shutil.rmtree(settings.tapping_work_dir / aid)  # e.g. the backend restarted and lost it
    status = client.get(f"/api/v1/tapping/analyses/{aid}", headers=h).json()
    assert status["status"] == "failed" and "interrupted" in status["error"]


def test_a_job_running_past_the_time_limit_is_marked_failed(
    client: TestClient, seeded_clinician: Clinician, tapping_env
) -> None:
    settings, runner = tapping_env
    runner.finish_with = None
    h = auth(client)
    aid = upload(client, h).json()["id"]
    started = settings.tapping_work_dir / aid / "started.json"
    started.write_text(json.dumps({"t": time.time() - settings.tapping_timeout_minutes * 60 - 120}))
    status = client.get(f"/api/v1/tapping/analyses/{aid}", headers=h).json()
    assert status["status"] == "failed" and "stopped" in status["error"]
    assert not (settings.tapping_work_dir / aid).exists()


# --------------------------------------------------------------------------- refused uploads
def test_a_file_that_is_not_an_accepted_video_type_is_refused(
    client: TestClient, seeded_clinician: Clinician, tapping_env
) -> None:
    settings, runner = tapping_env
    r = upload(client, auth(client), name="notes.txt")
    assert r.status_code == 400 and r.json()["error"]["code"] == "INVALID_UPLOAD"
    assert upload(client, auth(client), name="walk.avi").status_code == 400  # gait only
    assert runner.jobs == [] and not settings.tapping_work_dir.exists()


@pytest.mark.parametrize("name", ["a.mp4", "a.MOV", "a.m4v"])
def test_every_accepted_video_type_is_taken(
    client: TestClient, seeded_clinician: Clinician, tapping_env, name: str
) -> None:
    assert upload(client, auth(client), name=name).status_code == 202


def test_an_empty_or_too_large_upload_is_refused(
    client: TestClient, seeded_clinician: Clinician, tapping_env
) -> None:
    settings, runner = tapping_env
    h = auth(client)
    assert upload(client, h, data=b"").status_code == 400
    big = upload(client, h, data=b"\x00" * (1024 * 1024 + 1))
    assert big.status_code == 413 and big.json()["error"]["code"] == "UPLOAD_TOO_LARGE"
    assert runner.jobs == [] and not list(settings.tapping_work_dir.glob("*"))


def test_every_tapping_route_needs_sign_in(client: TestClient, tapping_env) -> None:
    aid = uuid.uuid4()
    assert upload(client, {}).status_code == 401
    for path in ("", f"/{aid}", f"/{aid}/result"):
        assert client.get(f"/api/v1/tapping/analyses{path}").status_code == 401


def test_another_clinicians_analysis_is_not_found(
    client: TestClient, seeded_clinician: Clinician, tapping_env, db_session: Session
) -> None:
    other = Clinician(
        username="s.perera",
        hashed_password=hash_password("correct-horse-battery-staple"),
        full_name="S. Perera",
    )
    db_session.add(other)
    db_session.commit()
    aid = upload(client, auth(client)).json()["id"]
    theirs = auth(client, "s.perera")
    for path in (f"/{aid}", f"/{aid}/result"):
        r = client.get(f"/api/v1/tapping/analyses{path}", headers=theirs)
        assert r.status_code == 404 and r.json()["error"]["code"] == "TAPPING_ANALYSIS_NOT_FOUND"
    assert client.get("/api/v1/tapping/analyses", headers=theirs).json()["total"] == 0


def test_list_shows_own_analyses_newest_first_and_the_label(
    client: TestClient, seeded_clinician: Clinician, tapping_env
) -> None:
    h = auth(client)
    first = upload(client, h).json()["id"]
    second = upload(client, h, subject_label="p-09").json()
    assert second["subject_label"] == "p-09"
    body = client.get("/api/v1/tapping/analyses?limit=1", headers=h).json()
    assert body["total"] == 2 and [i["id"] for i in body["items"]] == [second["id"]]
    assert first != second["id"]


def test_subject_label_that_looks_like_a_name_is_rejected(
    client: TestClient, seeded_clinician: Clinician, tapping_env
) -> None:
    assert upload(client, auth(client), subject_label="Jane Doe").status_code == 422


# --------------------------------------------------------------------------- the runner
def job_in(tmp_path: Path) -> TappingJob:
    folder = tmp_path / "job"
    folder.mkdir()
    video = folder / "abc.mp4"
    video.write_bytes(b"\x00" * 50)
    return TappingJob(uuid.uuid4(), folder, video, "ge1")


def script(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "fake_predict_clip.py"
    path.write_text("import json, sys, time\n" + body)
    return path


def runner_for(tmp_path: Path, body: str, minutes: float = 1.0) -> SubprocessTappingRunner:
    return SubprocessTappingRunner(
        Settings(
            tapping_python=sys.executable,
            tapping_script=script(tmp_path, body),
            tapping_timeout_minutes=minutes,
        )
    )


def test_runner_command_passes_the_video_and_target(tmp_path: Path) -> None:
    job = job_in(tmp_path)
    cmd = SubprocessTappingRunner(Settings()).command(job)
    assert cmd[2:] == ["--video", str(job.video), "--target", "ge1"]


def test_runner_keeps_the_programs_result_and_deletes_the_video(tmp_path: Path) -> None:
    job = job_in(tmp_path)
    runner_for(tmp_path, f"print(json.dumps({SCORED!r}))\n").run(job)
    assert not job.video.exists() and job.started_file.exists()
    assert json.loads(job.result_file.read_text()) == SCORED


def test_runner_records_a_program_that_prints_nothing_valid(tmp_path: Path) -> None:
    for body in (
        "print('not json')\n",
        "print(json.dumps({'status': 'scored'}))\n",
        "sys.exit(1)\n",
    ):
        job = job_in(tmp_path)
        runner_for(tmp_path, body).run(job)
        result = json.loads(job.result_file.read_text())
        assert result["status"] == "failed" and result["clip_ref"] is None
        assert not job.video.exists()
        shutil.rmtree(job.folder)


def test_runner_stops_a_program_that_takes_too_long(tmp_path: Path) -> None:
    job = job_in(tmp_path)
    runner_for(tmp_path, "time.sleep(30)\n", minutes=0.02).run(job)
    result = json.loads(job.result_file.read_text())
    assert result["status"] == "failed" and "too long" in result["reason"]
    assert not job.video.exists()


def test_runner_reports_a_program_that_cannot_start(tmp_path: Path) -> None:
    job = job_in(tmp_path)
    SubprocessTappingRunner(
        Settings(tapping_python=str(tmp_path / "missing-python"), tapping_script=tmp_path / "x.py")
    ).run(job)
    assert "could not start" in json.loads(job.result_file.read_text())["reason"]


def test_valid_result_checks_the_schema_and_size() -> None:
    assert valid_result(json.dumps(SCORED)) == SCORED
    assert valid_result(json.dumps(UNREADABLE)) == UNREADABLE
    for text in (
        "",
        "x",
        "[]",
        json.dumps({**SCORED, "schema_version": "tapping-result/2"}),
        json.dumps({**SCORED, "status": "abstained"}),
        "{" + " " * (21 * 1024 * 1024),
    ):
        assert valid_result(text) is None
