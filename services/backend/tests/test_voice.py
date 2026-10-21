"""Voice analysis endpoint. analyze_audio.py is replaced by a fake runner, so these tests need no
audio libraries or trained model; SubprocessVoiceRunner is tested with a stand-in script."""

import json
import sys
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from sqlalchemy.orm import Session

from app.api.deps import get_voice_runner
from app.core.config import Settings, get_settings
from app.core.errors import VoiceAnalysisFailedError
from app.main import app
from app.models.audit_log import AuditLogEntry
from app.models.clinician import Clinician
from app.workers.voice_runner import SubprocessVoiceRunner, VoiceJob

EXPLANATIONS = {
    "summary": "Risk score 0.28 (95% range 0.05-0.34).",
    "top_reasons": [],
    "uncertainty": ["The model was trained on only 61 people (figshare)."],
    "quality": {"passed": True, "problems": [], "measured": {"source_format": "m4a"}},
    "model_overview": {"top_groups": [], "sentence": ""},
}
SCORED = {
    "person": "upload-59f4139f",
    "reference_label": "-",
    "modality": "voice",
    "risk_score": 0.28,
    "ci_lower": 0.05,
    "ci_upper": 0.34,
    "triage_priority": "not_prioritised",
    "availability_mask": {"voice": True},
    "shap_top5": [{"feature": "mfcc9_sd", "shap_log_odds": -0.4, "direction": "x"}],
    "sex_fairness_note": "Exploratory.",
    "pipeline_version": "voice-audio-0.1.0+l1_logistic",
    "quality_gate_passed": True,
    "abstained": False,
    "problems": [],
    "n_training": 61,
    "note": "Voice screening score for triage only.",
    "explanations": EXPLANATIONS,
}
ABSTAINED = SCORED | {
    "risk_score": None,
    "ci_lower": None,
    "ci_upper": None,
    "triage_priority": None,
    "availability_mask": {"voice": False},
    "shap_top5": [],
    "quality_gate_passed": False,
    "abstained": True,
    "problems": ["The steady 'aaah' is only 0.4 s long; at least 1 s is needed."],
}


class FakeRunner:
    """Checks the recording is on disk under its ID, then answers like analyze_audio.py."""

    def __init__(self, answer: dict | Exception) -> None:
        self.answer = answer
        self.jobs: list[VoiceJob] = []

    def run(self, job: VoiceJob) -> dict:
        self.jobs.append(job)
        assert job.audio.exists() and job.audio.parent == job.folder
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


@pytest.fixture
def voice_env(tmp_path: Path) -> Generator[tuple[Settings, FakeRunner], None, None]:
    settings = Settings(voice_work_dir=tmp_path / "voice", voice_max_upload_mb=1)
    runner = FakeRunner(SCORED)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_voice_runner] = lambda: runner
    yield settings, runner
    app.dependency_overrides.pop(get_settings, None)
    app.dependency_overrides.pop(get_voice_runner, None)


def auth(client: TestClient) -> dict[str, str]:
    token = client.post(
        "/api/v1/auth/login",
        json={"username": "j.fernando", "password": "correct-horse-battery-staple"},
    ).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def upload(
    client: TestClient,
    headers: dict,
    data: bytes = b"\x00" * 300,
    name: str = "Patient Name 2026-10-20.m4a",
) -> Response:
    return client.post(
        "/api/v1/voice/analyses", files={"file": (name, data, "audio/mp4")}, headers=headers
    )


def test_upload_returns_the_explained_result_and_keeps_nothing(
    client: TestClient, seeded_clinician: Clinician, voice_env, db_session: Session
) -> None:
    settings, runner = voice_env
    response = upload(client, auth(client))
    assert response.status_code == 200
    body = response.json()
    assert body["triage_priority"] == "not_prioritised" and body["risk_score"] == 0.28
    assert body["explanations"]["uncertainty"] == EXPLANATIONS["uncertainty"]
    assert body["shap_top5"][0]["feature"] == "mfcc9_sd"  # extra fields pass through
    job = runner.jobs[0]
    assert job.audio.suffix == ".m4a" and job.audio.stem == job.folder.name
    assert not job.folder.exists()  # the recording is deleted
    assert {a.action for a in db_session.query(AuditLogEntry).all()} >= {"voice_analysis"}


def test_the_file_name_is_stored_nowhere(
    client: TestClient, seeded_clinician: Clinician, voice_env, db_session: Session
) -> None:
    _, runner = voice_env
    upload(client, auth(client))
    assert "Patient" not in str(runner.jobs[0].audio)
    assert "Patient" not in " ".join(str(a.detail) for a in db_session.query(AuditLogEntry).all())


def test_an_unusable_recording_comes_back_abstained_with_reasons(
    client: TestClient, seeded_clinician: Clinician, voice_env
) -> None:
    _, runner = voice_env
    runner.answer = ABSTAINED
    body = upload(client, auth(client)).json()
    assert body["abstained"] is True and body["risk_score"] is None
    assert body["triage_priority"] is None
    assert body["problems"] == ABSTAINED["problems"]


@pytest.mark.parametrize("name", ["voice.txt", "scan.zip", "no-extension"])
def test_a_file_that_is_not_audio_is_refused(
    client: TestClient, seeded_clinician: Clinician, voice_env, name: str
) -> None:
    _, runner = voice_env
    response = upload(client, auth(client), name=name)
    assert response.status_code == 400 and response.json()["error"]["code"] == "INVALID_UPLOAD"
    assert runner.jobs == []


def test_an_empty_file_is_refused_and_removed(
    client: TestClient, seeded_clinician: Clinician, voice_env
) -> None:
    settings, runner = voice_env
    response = upload(client, auth(client), data=b"")
    assert response.status_code == 400 and response.json()["error"]["code"] == "INVALID_UPLOAD"
    assert runner.jobs == [] and not any(settings.voice_work_dir.glob("*"))


def test_a_too_large_file_is_refused_and_removed(
    client: TestClient, seeded_clinician: Clinician, voice_env
) -> None:
    settings, runner = voice_env
    response = upload(client, auth(client), data=b"\x00" * (1024 * 1024 + 1))
    assert response.status_code == 413 and response.json()["error"]["code"] == "UPLOAD_TOO_LARGE"
    assert runner.jobs == [] and not any(settings.voice_work_dir.glob("*"))


def test_a_runner_failure_is_503_and_the_recording_is_still_deleted(
    client: TestClient, seeded_clinician: Clinician, voice_env
) -> None:
    _, runner = voice_env
    runner.answer = VoiceAnalysisFailedError("The voice analysis timed out")
    response = upload(client, auth(client))
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "VOICE_ANALYSIS_FAILED"
    assert not runner.jobs[0].folder.exists()


def test_a_result_with_a_diagnosis_word_as_triage_is_rejected(
    client: TestClient, seeded_clinician: Clinician, voice_env
) -> None:
    _, runner = voice_env
    runner.answer = SCORED | {"triage_priority": "positive"}
    with pytest.raises(ValueError):  # response validation fails rather than pass it on
        upload(client, auth(client))


def test_upload_needs_a_signed_in_clinician(client: TestClient, voice_env) -> None:
    _, runner = voice_env
    assert upload(client, {}).status_code in (401, 403)  # no token: HTTPBearer answers 403
    assert upload(client, {"Authorization": "Bearer not-a-token"}).status_code == 401
    assert runner.jobs == []


# ------------------------------------------------------------------ the real runner
def stand_in(tmp_path: Path, body: str) -> Settings:
    """A script that takes analyze_audio.py's arguments and does ``body``."""
    script = tmp_path / "analyze_audio.py"
    script.write_text("import json, sys\nout = sys.argv[sys.argv.index('--out') + 1]\n" + body)
    return Settings(voice_python=sys.executable, voice_script=script, voice_timeout_seconds=10)


def job(tmp_path: Path) -> VoiceJob:
    folder = tmp_path / "job"
    folder.mkdir()
    audio = folder / "x.wav"
    audio.write_bytes(b"\x00")
    return VoiceJob(folder, audio)


def test_runner_command_passes_the_model_and_output(tmp_path: Path) -> None:
    settings = Settings(voice_model="uci")
    command = SubprocessVoiceRunner(settings).command(job(tmp_path))
    assert command[2].endswith("x.wav")
    assert command[command.index("--model") + 1] == "uci"
    assert command[command.index("--out") + 1].endswith("result.json")


def test_runner_returns_the_single_result(tmp_path: Path) -> None:
    payload = json.dumps([SCORED])
    settings = stand_in(tmp_path, f"open(out, 'w').write({payload!r})\n")
    assert SubprocessVoiceRunner(settings).run(job(tmp_path))["risk_score"] == 0.28


@pytest.mark.parametrize(
    "body",
    [
        "sys.exit(1)\n",  # e.g. no trained model
        "pass\n",  # exits 0 but writes nothing
        "open(out, 'w').write('not json')\n",
        "json.dump([], open(out, 'w'))\n",
    ],
)
def test_runner_failures_raise_instead_of_returning_something(tmp_path: Path, body: str) -> None:
    with pytest.raises(VoiceAnalysisFailedError):
        SubprocessVoiceRunner(stand_in(tmp_path, body)).run(job(tmp_path))
