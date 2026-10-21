"""Gait analysis endpoints. predict_video.py is replaced by a fake runner that writes the
files the real one writes, so these tests need no video, MediaPipe or model."""

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

from app.api.deps import get_gait_runner
from app.core.config import Settings, get_settings
from app.core.security import hash_password
from app.main import app
from app.models.audit_log import AuditLogEntry
from app.models.clinician import Clinician
from app.workers.gait_runner import GaitJob, SubprocessGaitRunner

CONFIRMED = (
    "You confirmed one person is walking. The video itself was not checked for other people."
)
GATE = {
    "passed": True,
    "reasons": [],
    "warnings": ["One person in view is assumed: the video was not checked for other people."],
    "checks": {"frame_rate": {"value": 50.0, "limit": 20.0, "ok": True}},
}
SCORED = {
    "modality": "gait",
    "schema_version": "gait-result/1",
    "available": True,
    "video": "w-1a2b3c4d",
    "reasons": [],
    "warnings": [CONFIRMED],
    "gate": GATE,
    "walk": {"seconds": 19.0, "start_s": 1.2, "end_s": 20.2, "fps": 50.0},
    "predicted_class": 2,
    "predicted_label": "class 2-3",
    "class_labels": ["class 0", "class 1", "class 2-3"],
    "probabilities": [0.01, 0.13, 0.86],
    "probabilities_calibrated": False,
    "expected_severity": 1.85,
    "expected_severity_note": "A summary of the three model scores on a 0 to 2 scale.",
    "top_features": [
        {
            "name": "hip_flexion_range_deg",
            "value": 18.6,
            "shap": 1.56,
            "direction": "towards",
            "strength": "strongly",
            "sentence": "The model used hip swing range (19°), lower than about 9 in 10 of ...",
        }
    ],
    "feature_vector": {"hip_flexion_range_deg": 18.6},
    "model_version": "gait-xgb-ed1c5ad4",
    "one_person_confirmed": True,
}
NOT_SCORED = SCORED | {
    "available": False,
    "reasons": ["The legs were clearly visible in only 41% of the walk (at least 80% is needed)."],
    "gate": GATE | {"passed": False},
    "predicted_class": None,
    "predicted_label": None,
    "probabilities": None,
    "expected_severity": None,
    "top_features": [],
    "feature_vector": None,
}


class FakeRunner:
    """Writes what the real runner and predict_video.py would: started.json, then result.json."""

    def __init__(self, finish_with: dict | None) -> None:
        self.finish_with = finish_with
        self.jobs: list[GaitJob] = []

    def submit(self, job: GaitJob) -> None:
        self.jobs.append(job)
        assert job.video.exists() and job.video.parent == job.folder  # saved before the job
        job.started_file.write_text(json.dumps({"t": time.time()}))
        if self.finish_with is not None:
            job.video.unlink()
            job.result_file.write_text(json.dumps(self.finish_with))


@pytest.fixture
def gait_env(tmp_path: Path) -> Generator[tuple[Settings, FakeRunner], None, None]:
    settings = Settings(gait_work_dir=tmp_path / "gait", gait_max_upload_mb=1)
    runner = FakeRunner(SCORED)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_gait_runner] = lambda: runner
    yield settings, runner
    app.dependency_overrides.pop(get_settings, None)
    app.dependency_overrides.pop(get_gait_runner, None)


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
    return client.post(
        "/api/v1/gait/analyses",
        files=files,
        data={"one_person_confirmed": "true"} | form,
        headers=headers,
    )


# --------------------------------------------------------------------------- the flow
def test_upload_runs_and_the_result_is_the_gait_record(
    client: TestClient, seeded_clinician: Clinician, gait_env, db_session: Session
) -> None:
    settings, runner = gait_env
    h = auth(client)
    created = upload(client, h)
    assert created.status_code == 202 and created.json()["status"] == "queued"
    aid = created.json()["id"]
    job = runner.jobs[0]
    assert job.one_person_confirmed is True
    assert job.video.name == f"{aid}.mov"  # named by the analysis, never by the upload

    status = client.get(f"/api/v1/gait/analyses/{aid}", headers=h).json()
    assert status["status"] == "scored" and status["abstain_reason"] is None
    assert status["one_person_confirmed"] is True and status["elapsed_s"] >= 0
    body = client.get(f"/api/v1/gait/analyses/{aid}/result", headers=h).json()
    assert body["status"] == "scored" and body["schema_version"] == "gait-result/1"
    assert body["probabilities"] == [0.01, 0.13, 0.86]
    assert body["probabilities_calibrated"] is False
    assert body["top_features"][0]["strength"] == "strongly"
    assert body["warnings"] == [CONFIRMED]
    assert not (settings.gait_work_dir / aid).exists()  # nothing of the upload left on disk
    actions = {a.action for a in db_session.query(AuditLogEntry).all()}
    assert {"gait_upload", "gait_result_view"} <= actions


def test_the_upload_name_is_stored_nowhere(
    client: TestClient, seeded_clinician: Clinician, gait_env, db_session: Session
) -> None:
    h = auth(client)
    aid = upload(client, h).json()["id"]
    client.get(f"/api/v1/gait/analyses/{aid}/result", headers=h)
    stored = " ".join(
        [str(a.detail) for a in db_session.query(AuditLogEntry).all()]
        + [client.get("/api/v1/gait/analyses", headers=h).text]
    )
    assert "Patient" not in stored


def test_a_video_that_cannot_be_scored_is_abstained_with_its_reasons(
    client: TestClient, seeded_clinician: Clinician, gait_env
) -> None:
    _, runner = gait_env
    runner.finish_with = NOT_SCORED
    h = auth(client)
    aid = upload(client, h).json()["id"]
    status = client.get(f"/api/v1/gait/analyses/{aid}", headers=h).json()
    assert status["status"] == "abstained" and status["error"] is None
    assert status["abstain_reason"].startswith("The legs were clearly visible in only 41%")
    body = client.get(f"/api/v1/gait/analyses/{aid}/result", headers=h).json()
    assert body["status"] == "abstained" and body["probabilities"] is None
    assert body["reasons"] == NOT_SCORED["reasons"]


def test_status_is_running_while_the_video_is_analysed(
    client: TestClient, seeded_clinician: Clinician, gait_env
) -> None:
    _, runner = gait_env
    runner.finish_with = None
    h = auth(client)
    aid = upload(client, h).json()["id"]
    status = client.get(f"/api/v1/gait/analyses/{aid}", headers=h).json()
    assert status["status"] == "running" and status["finished_at"] is None
    early = client.get(f"/api/v1/gait/analyses/{aid}/result", headers=h)
    assert early.status_code == 409 and early.json()["error"]["code"] == "GAIT_RESULT_NOT_READY"


def test_a_failure_the_runner_wrote_is_reported(
    client: TestClient, seeded_clinician: Clinician, gait_env
) -> None:
    _, runner = gait_env
    runner.finish_with = {
        "modality": "gait",
        "schema_version": "gait-result/1",
        "status": "failed",
        "error": "the gait analysis timed out",
    }
    h = auth(client)
    aid = upload(client, h).json()["id"]
    status = client.get(f"/api/v1/gait/analyses/{aid}", headers=h).json()
    assert status["status"] == "failed" and status["error"] == "the gait analysis timed out"
    body = client.get(f"/api/v1/gait/analyses/{aid}/result", headers=h).json()
    assert body["status"] == "failed" and body["available"] is False


def test_a_lost_job_is_marked_failed(
    client: TestClient, seeded_clinician: Clinician, gait_env
) -> None:
    settings, runner = gait_env
    runner.finish_with = None
    h = auth(client)
    aid = upload(client, h).json()["id"]
    shutil.rmtree(settings.gait_work_dir / aid)  # e.g. the backend restarted and lost it
    status = client.get(f"/api/v1/gait/analyses/{aid}", headers=h).json()
    assert status["status"] == "failed" and "interrupted" in status["error"]


def test_a_job_running_past_the_time_limit_is_marked_failed(
    client: TestClient, seeded_clinician: Clinician, gait_env
) -> None:
    settings, runner = gait_env
    runner.finish_with = None
    h = auth(client)
    aid = upload(client, h).json()["id"]
    started = settings.gait_work_dir / aid / "started.json"
    started.write_text(json.dumps({"t": time.time() - settings.gait_timeout_minutes * 60 - 120}))
    status = client.get(f"/api/v1/gait/analyses/{aid}", headers=h).json()
    assert status["status"] == "failed" and "stopped" in status["error"]
    assert not (settings.gait_work_dir / aid).exists()


# --------------------------------------------------------------------------- refused uploads
@pytest.mark.parametrize("answer", ["false", "0"])
def test_an_upload_without_the_one_person_confirmation_is_refused(
    client: TestClient, seeded_clinician: Clinician, gait_env, answer: str
) -> None:
    settings, runner = gait_env
    r = upload(client, auth(client), one_person_confirmed=answer)
    assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION_ERROR"
    assert "one person walking" in r.json()["error"]["message"]
    assert runner.jobs == [] and not settings.gait_work_dir.exists()


def test_an_upload_missing_the_confirmation_field_is_a_validation_error(
    client: TestClient, seeded_clinician: Clinician, gait_env
) -> None:
    r = client.post(
        "/api/v1/gait/analyses",
        files={"file": ("walk.mp4", b"\x00" * 10, "video/mp4")},
        headers=auth(client),
    )
    assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.parametrize("name", ["walk.zip", "walk", "walk.mp4.exe", "notes.txt"])
def test_a_file_that_is_not_an_accepted_video_type_is_refused(
    client: TestClient, seeded_clinician: Clinician, gait_env, name: str
) -> None:
    _, runner = gait_env
    r = upload(client, auth(client), name=name)
    assert r.status_code == 400 and r.json()["error"]["code"] == "INVALID_UPLOAD"
    assert ".mp4" in r.json()["error"]["message"] and runner.jobs == []


@pytest.mark.parametrize("name", ["walk.MP4", "walk.mov", "walk.m4v", "walk.avi"])
def test_every_accepted_video_type_is_taken(
    client: TestClient, seeded_clinician: Clinician, gait_env, name: str
) -> None:
    assert upload(client, auth(client), name=name).status_code == 202


def test_an_empty_or_too_large_upload_is_refused(
    client: TestClient, seeded_clinician: Clinician, gait_env
) -> None:
    settings, runner = gait_env
    h = auth(client)
    empty = upload(client, h, data=b"")
    assert empty.status_code == 400 and empty.json()["error"]["code"] == "INVALID_UPLOAD"
    big = upload(client, h, data=b"\x00" * (2 * 1024 * 1024))  # the limit is 1 MB here
    assert big.status_code == 413 and big.json()["error"]["code"] == "UPLOAD_TOO_LARGE"
    assert runner.jobs == [] and list(settings.gait_work_dir.glob("*")) == []


# --------------------------------------------------------------------------- access
def test_every_gait_route_needs_sign_in(client: TestClient, gait_env) -> None:
    some = uuid.uuid4()
    responses = [
        upload(client, {}),
        client.get("/api/v1/gait/analyses"),
        client.get(f"/api/v1/gait/analyses/{some}"),
        client.get(f"/api/v1/gait/analyses/{some}/result"),
        client.get("/api/v1/gait/analyses", headers={"Authorization": "Bearer not-a-token"}),
    ]
    assert all(r.status_code in (401, 403) and "error" in r.json() for r in responses)
    assert responses[-1].status_code == 401


def test_another_clinicians_analysis_is_not_found(
    client: TestClient, seeded_clinician: Clinician, gait_env, db_session: Session
) -> None:
    db_session.add(
        Clinician(
            username="other",
            hashed_password=hash_password("correct-horse-battery-staple"),
            full_name="O",
        )
    )
    db_session.commit()
    aid = upload(client, auth(client)).json()["id"]
    h2 = auth(client, "other")
    for path in (f"/api/v1/gait/analyses/{aid}", f"/api/v1/gait/analyses/{aid}/result"):
        r = client.get(path, headers=h2)
        assert r.status_code == 404 and r.json()["error"]["code"] == "GAIT_ANALYSIS_NOT_FOUND"
    assert client.get("/api/v1/gait/analyses", headers=h2).json()["total"] == 0


def test_list_shows_own_analyses_newest_first(
    client: TestClient, seeded_clinician: Clinician, gait_env
) -> None:
    h = auth(client)
    first = upload(client, h).json()["id"]
    time.sleep(0.01)
    second = upload(client, h).json()["id"]
    body = client.get("/api/v1/gait/analyses", headers=h).json()
    assert body["total"] == 2 and [i["id"] for i in body["items"]] == [second, first]
    assert client.get("/api/v1/gait/analyses?limit=1", headers=h).json()["total"] == 2


def test_subject_label_is_optional_and_remembered(
    client: TestClient, seeded_clinician: Clinician, gait_env
) -> None:
    h = auth(client)
    assert upload(client, h).json()["subject_label"] is None
    created = upload(client, h, subject_label="P-8F31A2")
    assert created.status_code == 202 and created.json()["subject_label"] == "P-8F31A2"
    status = client.get(f"/api/v1/gait/analyses/{created.json()['id']}", headers=h).json()
    assert status["subject_label"] == "P-8F31A2"
    listed = client.get("/api/v1/gait/analyses", headers=h).json()["items"]
    assert [i["subject_label"] for i in listed] == ["P-8F31A2", None]


def test_subject_label_that_looks_like_a_name_is_rejected(
    client: TestClient, seeded_clinician: Clinician, gait_env
) -> None:
    settings, runner = gait_env
    for bad in ["Jane Doe", "a" * 33, "j@ne", ""]:
        assert upload(client, auth(client), subject_label=bad).status_code == 422, bad
    assert runner.jobs == [] and not settings.gait_work_dir.exists()  # nothing was started


# --------------------------------------------------------------------------- the real runner
def job_in(tmp_path: Path, confirmed: bool = True) -> GaitJob:
    folder = tmp_path / "job"
    folder.mkdir()
    video = folder / "abc.mp4"
    video.write_bytes(b"\x00" * 50)
    return GaitJob(uuid.uuid4(), folder, video, confirmed)


def script(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "fake_predict.py"
    path.write_text("import json, sys, time\n" + body)
    return path


def test_runner_command_passes_the_video_bundle_output_and_confirmation(tmp_path: Path) -> None:
    job = job_in(tmp_path)
    cmd = SubprocessGaitRunner(Settings(gait_bundle_dir=tmp_path / "b")).command(job)
    assert cmd[2] == str(job.video) and cmd[cmd.index("--bundle") + 1] == str(tmp_path / "b")
    assert cmd[cmd.index("--out") + 1] == str(job.result_file)
    assert cmd[-1] == "--one-person-confirmed"
    unconfirmed = GaitJob(job.analysis_id, job.folder, job.video, False)
    assert "--one-person-confirmed" not in SubprocessGaitRunner(Settings()).command(unconfirmed)


def test_runner_keeps_the_scripts_result_and_deletes_the_video(tmp_path: Path) -> None:
    body = "out = sys.argv[sys.argv.index('--out') + 1]\n"
    body += "open(out, 'w').write(json.dumps({'available': False, 'reasons': ['x']}))\n"
    body += "sys.exit(1)\n"
    job = job_in(tmp_path)
    settings = Settings(gait_python=sys.executable, gait_script=script(tmp_path, body))
    SubprocessGaitRunner(settings).run(job)
    assert not job.video.exists() and job.started_file.exists()
    assert json.loads(job.result_file.read_text()) == {"available": False, "reasons": ["x"]}


@pytest.mark.parametrize(
    ("body", "python", "minutes", "words"),
    [
        ("sys.exit(2)\n", None, 15.0, "could not start: the model files"),
        ("sys.exit(0)\n", None, 15.0, "ended without a result"),
        ("time.sleep(5)\n", None, 0.002, "timed out"),
        ("", "/no/such/python", 15.0, "could not start"),
    ],
)
def test_runner_writes_a_failure_and_deletes_the_video_when_the_script_gives_no_result(
    tmp_path: Path, body: str, python: str | None, minutes: float, words: str
) -> None:
    job = job_in(tmp_path)
    settings = Settings(
        gait_python=python or sys.executable,
        gait_script=script(tmp_path, body),
        gait_timeout_minutes=minutes,
    )
    SubprocessGaitRunner(settings).run(job)
    assert not job.video.exists()
    result = json.loads(job.result_file.read_text())
    assert (
        result["status"] == "failed" and result["modality"] == "gait" and words in result["error"]
    )
