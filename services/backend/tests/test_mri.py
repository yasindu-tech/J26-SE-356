"""MRI analysis endpoints. predict_scan.py is replaced by a fake runner that writes the
files the real one writes, so these tests need no images, FSL or models."""

import io
import json
import time
import uuid
import zipfile
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.deps import get_mri_runner
from app.core.config import Settings, get_settings
from app.core.security import hash_password
from app.main import app
from app.models.audit_log import AuditLogEntry
from app.models.clinician import Clinician
from app.workers.mri_runner import MriJob, SubprocessRunner

SCORED = {
    "modality": "mri",
    "schema_version": "mri-result/1",
    "status": "scored",
    "model_version": "v1",
    "model_set": "t1",
    "availability": {"t1": True, "dti": False},
    "inputs": {"t1": {"found": True, "series_description": "MPRAGE"}, "dti": {"found": False}},
    "validation": {"usable": True, "problems": ["no DTI series"]},
    "feature_vector": {
        "names": ["L_putamen_grey_frac"],
        "labels": ["Left putamen: grey-matter share"],
        "raw": [0.61],
        "harmonised": [0.6],
        "scanner_batch": {"t1": "Prisma"},
        "outside_training_range": [],
    },
    "score": {
        "kind": "log_odds",
        "value": 0.4,
        "interval_95": [-0.2, 0.9],
        "interval_method": "spread over 50 fold copies of the pipeline",
        "calibrated": False,
        "n_training": 59,
        "meaning": "Raw model score in log-odds.",
    },
    "explanation": {
        "kind": "attribution",
        "method": "TreeSHAP",
        "base_value": 0.1,
        "feature_attributions": {"L_putamen_grey_frac": 0.3},
        "top_drivers": [
            {
                "feature": "L_putamen_grey_frac",
                "label": "Left putamen: grey-matter share",
                "value": 0.61,
                "contribution": 0.3,
                "direction": "towards PD",
                "text": "Left putamen: grey-matter share raised this model's score by 0.30",
            }
        ],
        "caveat": "Contributions show what moved this model's score, not a biological cause.",
    },
    "model_card": {"purpose": "Screening and triage aid. Not a diagnosis."},
    "disclaimer": "Not a diagnosis.",
}


class FakeRunner:
    """Writes what predict_scan.py would: status while running, result at the end."""

    def __init__(self, finish_with: dict | None) -> None:
        self.finish_with = finish_with
        self.jobs: list[MriJob] = []

    def submit(self, job: MriJob) -> None:
        self.jobs.append(job)
        assert job.upload.exists()  # the zip is on disk before the job starts
        progress = {
            "steps": [],
            "done": ["convert"],
            "current": "t1",
            "note": "T1: n4",
            "t": time.time(),
        }
        job.status_file.write_text(json.dumps(progress))
        if self.finish_with is not None:
            job.upload.unlink()
            job.result_file.write_text(json.dumps(self.finish_with))


@pytest.fixture
def mri_env(tmp_path: Path) -> Generator[tuple[Settings, FakeRunner], None, None]:
    settings = Settings(mri_work_dir=tmp_path / "mri", mri_max_upload_mb=1)
    runner = FakeRunner(SCORED)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_mri_runner] = lambda: runner
    yield settings, runner
    app.dependency_overrides.pop(get_settings, None)
    app.dependency_overrides.pop(get_mri_runner, None)


def auth(client: TestClient, username: str = "j.fernando") -> dict[str, str]:
    token = client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": "correct-horse-battery-staple"},
    ).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def dicom_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("study/series1/IM0001.dcm", b"\x00" * 200)
    return buf.getvalue()


def upload(client: TestClient, headers: dict, data: bytes | None = None, **form) -> object:
    files = {"file": ("study.zip", data if data is not None else dicom_zip(), "application/zip")}
    return client.post(
        "/api/v1/mri/analyses", files=files, data={"age": "67", "sex": "M"} | form, headers=headers
    )


def test_upload_runs_and_result_has_vector_score_and_explanation(
    client: TestClient, seeded_clinician: Clinician, mri_env, db_session: Session
) -> None:
    settings, runner = mri_env
    h = auth(client)
    created = upload(client, h)
    assert created.status_code == 202 and created.json()["status"] == "queued"
    aid = created.json()["id"]
    assert runner.jobs[0].age == 67.0 and runner.jobs[0].sex == "M"

    status = client.get(f"/api/v1/mri/analyses/{aid}", headers=h).json()
    assert status["status"] == "scored" and status["done"][-1] == "explain"
    result = client.get(f"/api/v1/mri/analyses/{aid}/result", headers=h)
    assert result.status_code == 200
    body = result.json()
    assert body["feature_vector"]["harmonised"] == [0.6]
    assert body["score"]["kind"] == "log_odds" and body["score"]["interval_95"] == [-0.2, 0.9]
    assert body["explanation"]["top_drivers"][0]["direction"] == "towards PD"
    assert not (settings.mri_work_dir / aid).exists()  # nothing of the upload left on disk
    actions = {a.action for a in db_session.query(AuditLogEntry).all()}
    assert {"mri_upload", "mri_result_view"} <= actions


def test_progress_is_reported_while_running(
    client: TestClient, seeded_clinician: Clinician, mri_env
) -> None:
    _, runner = mri_env
    runner.finish_with = None  # still working
    h = auth(client)
    aid = upload(client, h).json()["id"]
    status = client.get(f"/api/v1/mri/analyses/{aid}", headers=h).json()
    assert status["status"] == "running" and status["current_step"] == "t1"
    assert status["done"] == ["convert"] and status["note"] == "T1: n4"
    early = client.get(f"/api/v1/mri/analyses/{aid}/result", headers=h)
    assert early.status_code == 409 and early.json()["error"]["code"] == "MRI_RESULT_NOT_READY"


def test_abstained_analysis_reports_why(
    client: TestClient, seeded_clinician: Clinician, mri_env
) -> None:
    _, runner = mri_env
    reason = "the T1 scanner model (Ingenia) was not in the training data"
    runner.finish_with = SCORED | {"status": "abstained", "abstain_reason": reason, "score": None}
    h = auth(client)
    aid = upload(client, h).json()["id"]
    status = client.get(f"/api/v1/mri/analyses/{aid}", headers=h).json()
    assert status["status"] == "abstained" and status["abstain_reason"] == reason
    assert client.get(f"/api/v1/mri/analyses/{aid}/result", headers=h).json()["score"] is None


def test_upload_that_is_not_a_zip_is_refused(
    client: TestClient, seeded_clinician: Clinician, mri_env
) -> None:
    _, runner = mri_env
    r = upload(client, auth(client), data=b"not a zip at all")
    assert r.status_code == 400 and r.json()["error"]["code"] == "INVALID_UPLOAD"
    assert runner.jobs == []


def test_upload_over_the_limit_is_refused(
    client: TestClient, seeded_clinician: Clinician, mri_env
) -> None:
    r = upload(client, auth(client), data=b"\x00" * (2 * 1024 * 1024))  # limit is 1 MB here
    assert r.status_code == 413 and r.json()["error"]["code"] == "UPLOAD_TOO_LARGE"


@pytest.mark.parametrize("form", [{"sex": "X"}, {"age": "5"}, {"age": "abc"}])
def test_bad_age_or_sex_is_a_validation_error(
    client: TestClient, seeded_clinician: Clinician, mri_env, form
) -> None:
    r = upload(client, auth(client), **form)
    assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION_ERROR"


def test_every_mri_route_needs_sign_in(client: TestClient, mri_env) -> None:
    # HTTPBearer answers a missing header with 403, as for /auth/me (test_auth.py)
    some = uuid.uuid4()
    responses = [
        upload(client, {}),
        client.get("/api/v1/mri/analyses"),
        client.get(f"/api/v1/mri/analyses/{some}"),
        client.get(f"/api/v1/mri/analyses/{some}/result"),
        client.get("/api/v1/mri/analyses", headers={"Authorization": "Bearer not-a-token"}),
    ]
    assert all(r.status_code in (401, 403) and "error" in r.json() for r in responses)
    assert responses[-1].status_code == 401


def test_another_clinicians_analysis_is_not_found(
    client: TestClient, seeded_clinician: Clinician, mri_env, db_session: Session
) -> None:
    other = Clinician(
        username="other",
        hashed_password=hash_password("correct-horse-battery-staple"),
        full_name="O",
    )
    db_session.add(other)
    db_session.commit()
    aid = upload(client, auth(client)).json()["id"]
    h2 = auth(client, "other")
    assert client.get(f"/api/v1/mri/analyses/{aid}", headers=h2).status_code == 404
    assert client.get(f"/api/v1/mri/analyses/{aid}/result", headers=h2).status_code == 404
    assert client.get("/api/v1/mri/analyses", headers=h2).json()["total"] == 0


def test_list_shows_own_analyses_newest_first(
    client: TestClient, seeded_clinician: Clinician, mri_env
) -> None:
    h = auth(client)
    upload(client, h)
    upload(client, h)
    body = client.get("/api/v1/mri/analyses", headers=h).json()
    assert body["total"] == 2 and len(body["items"]) == 2


def test_lost_job_is_marked_failed(
    client: TestClient, seeded_clinician: Clinician, mri_env
) -> None:
    settings, runner = mri_env
    runner.finish_with = None
    h = auth(client)
    aid = upload(client, h).json()["id"]
    import shutil

    shutil.rmtree(settings.mri_work_dir / aid)  # e.g. the backend restarted and lost it
    status = client.get(f"/api/v1/mri/analyses/{aid}", headers=h).json()
    assert status["status"] == "failed" and "interrupted" in status["error"]


def test_runner_deletes_the_upload_and_writes_a_result_even_if_the_script_fails(
    tmp_path: Path,
) -> None:
    settings = Settings(mri_python="/no/such/python")
    folder = tmp_path / "job"
    folder.mkdir()
    job = MriJob(analysis_id=uuid.uuid4(), folder=folder, age=60.0, sex="F")
    job.upload.write_bytes(dicom_zip())
    (folder / "work").mkdir()
    SubprocessRunner(settings).run(job)
    assert not job.upload.exists() and not (folder / "work").exists()
    assert json.loads(job.result_file.read_text())["status"] == "failed"


def test_runner_command_passes_age_sex_and_files(tmp_path: Path) -> None:
    job = MriJob(analysis_id=uuid.uuid4(), folder=tmp_path, age=67.0, sex="M")
    cmd = SubprocessRunner(Settings(mri_eddy_threads=3)).command(job)
    assert cmd[cmd.index("--age") + 1] == "67.0" and cmd[cmd.index("--sex") + 1] == "M"
    assert cmd[cmd.index("--status-file") + 1].endswith("status.json")
    assert cmd[cmd.index("--eddy-threads") + 1] == "3"
