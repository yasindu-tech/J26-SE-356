"""Tests for models/mri/src/pipeline_monitor.py (dashboard server)."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pipeline_monitor as m
import pytest
from progress import RunRecorder


@pytest.fixture
def server(tmp_path: Path):
    srv = ThreadingHTTPServer(
        ("127.0.0.1", 0), m.make_handler(tmp_path, token="s3cret", interval=0.05)
    )
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", tmp_path
    srv.shutdown()


def get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def post(url: str, body: dict) -> int:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def test_pages_render(server) -> None:
    base, _ = server
    for path in ("/", "/t1", "/dti"):
        code, body = get(base + path)
        assert code == 200 and b"<html>" in body


def test_state_follows_latest_run(server) -> None:
    base, runs = server
    rec = RunRecorder("t1", ["1", "2"], ["n4"], runs_dir=runs)
    rec.subject_start("1")
    state = json.loads(get(base + "/api/state?pipeline=t1")[1])
    assert (state["total"], state["running"], state["run"]) == (2, 1, rec.run_id)
    assert state["step_info"]["n4"]["label"] == "Even out brightness"
    assert json.loads(get(base + "/api/runs?pipeline=t1")[1]) == [rec.run_id]


def test_bad_pipeline_and_run_ids_rejected(server) -> None:
    base, _ = server
    assert get(base + "/api/state?pipeline=x")[0] == 400
    assert get(base + "/api/state?pipeline=t1&run=../../etc")[0] == 400
    assert get(base + "/api/state?pipeline=t1&run=dti_20260101_000000")[0] == 400


def test_remote_events_need_token_and_valid_shape(server) -> None:
    base, runs = server
    event = {
        "t": 1.0,
        "run": "dti_20261010_120000",
        "pipeline": "dti",
        "type": "run_start",
        "subjects": ["abcd1234"],
        "steps": ["eddy"],
    }
    assert post(base + "/api/events", event) == 403
    assert post(base + "/api/events?token=wrong", event) == 403
    assert post(base + "/api/events?token=s3cret", {**event, "run": "../escape"}) == 400
    assert post(base + "/api/events?token=s3cret", event) == 200
    assert (runs / "dti_20261010_120000.jsonl").exists()


def test_stream_sends_state(server) -> None:
    base, runs = server
    RunRecorder("dti", ["1"], ["eddy"], runs_dir=runs)
    with urllib.request.urlopen(base + "/api/stream?pipeline=dti", timeout=5) as r:
        assert r.headers["Content-Type"] == "text/event-stream"
        line = r.readline().decode()
    assert line.startswith("data: ") and json.loads(line[6:])["total"] == 1


@pytest.mark.parametrize(
    ("event", "ok"),
    [
        ({"t": 1, "run": "t1_20260101_000000", "pipeline": "t1", "type": "run_end"}, True),
        ({"t": 1, "run": "t1_x/../../a", "pipeline": "t1", "type": "run_end"}, False),
        ({"t": 1, "run": "t1_20260101_000000", "pipeline": "mri", "type": "run_end"}, False),
        ({"t": "now", "run": "t1_20260101_000000", "pipeline": "t1", "type": "run_end"}, False),
        (["not", "a", "dict"], False),
    ],
)
def test_valid_event(event: object, ok: bool) -> None:
    assert m.valid_event(event) is ok
