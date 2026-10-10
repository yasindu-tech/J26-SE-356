"""Tests for models/mri/src/progress.py: events written, and replayed into dashboard state."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from progress import NullRecorder, RunRecorder, fold_events, latest_run, pseudonym, read_events


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make_run(tmp_path: Path) -> tuple[RunRecorder, Clock]:
    clock = Clock()
    rec = RunRecorder("t1", ["111", "222", "333"], ["n4", "skull"], runs_dir=tmp_path, clock=clock)
    return rec, clock


def test_events_never_contain_raw_subject_ids(tmp_path: Path) -> None:
    rec, clock = make_run(tmp_path)
    rec.subject_start("111")
    with rec.step("111", "n4"):
        clock.now += 5
    rec.subject_end("111", "ok", metrics={"brain_mask_ml": 1500.0, "note": "text is dropped"})
    text = rec.path.read_text()
    assert "111" not in text.replace(pseudonym("111"), "")
    last = json.loads(text.splitlines()[-1])
    assert last["metrics"] == {"brain_mask_ml": 1500.0}  # only numbers leave the pipeline


def test_step_timing_and_failure_are_recorded(tmp_path: Path) -> None:
    rec, clock = make_run(tmp_path)
    with pytest.raises(RuntimeError), rec.step("111", "skull"):
        clock.now += 3
        raise RuntimeError("SynthStrip failed")
    end = read_events(rec.path)[-1]
    assert (end["type"], end["ok"], end["seconds"]) == ("step_end", False, 3.0)


def test_fold_events_live_state(tmp_path: Path) -> None:
    rec, clock = make_run(tmp_path)
    for sid, secs in (("111", (100, 200)),):
        rec.subject_start(sid)
        for step, s in zip(["n4", "skull"], secs, strict=True):
            with rec.step(sid, step):
                clock.now += s
        rec.subject_end(sid, "ok")
    rec.subject_start("222")
    with pytest.raises(ValueError), rec.step("222", "n4"):
        clock.now += 4
        raise ValueError("bad")
    rec.subject_end("222", "failed", "bad input")
    rec.subject_start("333")
    with rec.step("333", "n4"):
        clock.now += 6
    rec._emit("step_start", subject=pseudonym("333"), step="skull")
    clock.now += 7

    st = fold_events(read_events(rec.path), now=clock.now)
    assert (st["total"], st["done"], st["failed"], st["running"], st["waiting"]) == (3, 1, 1, 1, 0)
    assert st["status"] == "running"
    assert st["avg_seconds_per_subject"] == 300.0
    assert st["eta_seconds"] == 300.0  # one person left (the running one) x 300 s
    assert st["current"]["step"] == "skull" and st["current"]["step_seconds"] == 7.0
    by_step = {s["step"]: s for s in st["steps"]}
    assert by_step["n4"] == {"step": "n4", "done": 2, "failed": 1, "avg_seconds": 53.0}
    failed = next(s for s in st["subjects"] if s["state"] == "failed")
    assert (failed["failed_step"], failed["reason"]) == ("n4", "bad input")


def test_run_end_leaves_nobody_running(tmp_path: Path) -> None:
    rec, _ = make_run(tmp_path)
    rec.subject_start("111")
    rec.run_end()  # e.g. the run was stopped with Ctrl+C after this
    st = fold_events(read_events(rec.path), now=2000)
    assert st["status"] == "finished" and st["running"] == 0
    assert next(s for s in st["subjects"] if s["id"] == pseudonym("111"))["state"] == "failed"


def test_half_written_line_is_ignored(tmp_path: Path) -> None:
    rec, _ = make_run(tmp_path)
    with rec.path.open("a") as f:
        f.write('{"type": "subject_st')  # the pipeline is mid-write
    assert len(read_events(rec.path)) == 1


def test_latest_run_and_empty_state(tmp_path: Path) -> None:
    assert latest_run("t1", tmp_path) is None
    st = fold_events([], now=1)
    assert (st["status"], st["total"], st["current"]) == ("no run", 0, None)
    rec, _ = make_run(tmp_path)
    assert latest_run("t1", tmp_path) == rec.path


def test_null_recorder_does_nothing() -> None:
    rec = NullRecorder()
    with rec.step("1", "n4"):
        pass
    rec.subject_end("1", "ok")


def test_unreachable_dashboard_does_not_stop_pipeline(tmp_path: Path) -> None:
    rec = RunRecorder(
        "dti", ["1"], ["eddy"], runs_dir=tmp_path, events_url="http://127.0.0.1:9/api/events"
    )
    rec.subject_start("1")  # POST fails, file still written
    assert len(read_events(rec.path)) == 2


def test_parallel_jobs_eta_and_running_list(tmp_path: Path) -> None:
    clock = Clock()
    rec = RunRecorder(
        "dti", ["a", "b", "c", "d", "e"], ["eddy"], runs_dir=tmp_path, clock=clock, jobs=2
    )
    rec.subject_start("a")
    with rec.step("a", "eddy"):
        clock.now += 100
    rec.subject_end("a", "ok")
    rec.subject_start("b")
    rec.subject_start("c")  # two people at once
    st = fold_events(read_events(rec.path), now=clock.now)
    assert st["jobs"] == 2 and st["running"] == 2 and len(st["running_now"]) == 2
    assert st["eta_seconds"] == 100 * 4 / 2  # 4 left (2 running + 2 waiting), 2 at a time


def test_expected_times_ignore_reused_steps_and_give_percent(tmp_path: Path) -> None:
    from progress import expected_step_seconds

    clock = Clock()
    rec = RunRecorder("dti", ["a", "b", "c"], ["eddy"], runs_dir=tmp_path, clock=clock)
    with rec.step("a", "eddy"):
        clock.now += 0.1  # reused from an earlier run: must not count
    rec.subject_end("a", "ok")
    with rec.step("b", "eddy"):
        clock.now += 1200
    rec.subject_end("b", "ok")
    rec.subject_start("c")
    rec._emit("step_start", subject=pseudonym("c"), step="eddy")
    clock.now += 600
    events = read_events(rec.path)
    expected = expected_step_seconds([events])
    assert expected == {"eddy": 1200.0}
    st = fold_events(events, now=clock.now, expected=expected)
    assert st["current"]["pct"] == 50
    assert st["steps"][0]["avg_seconds"] == 1200.0  # the 0.1 s reuse is left out
    clock.now += 5000  # running far past the usual time: shows 99, never 100+
    assert fold_events(events, now=clock.now, expected=expected)["current"]["pct"] == 99


def test_no_estimate_before_any_step_finished(tmp_path: Path) -> None:
    rec, clock = make_run(tmp_path)
    rec.subject_start("111")
    rec._emit("step_start", subject=pseudonym("111"), step="n4")
    st = fold_events(read_events(rec.path), now=clock.now + 10)
    assert st["current"]["pct"] is None


def test_reused_person_does_not_drive_time_left(tmp_path: Path) -> None:
    clock = Clock()
    rec = RunRecorder("dti", ["a", "b"], ["eddy"], runs_dir=tmp_path, clock=clock)
    rec.subject_start("a")
    clock.now += 6  # outputs already existed: "done" in 6 s
    rec.subject_end("a", "ok")
    rec.subject_start("b")
    st = fold_events(read_events(rec.path), now=clock.now + 100)
    assert st["avg_seconds_per_subject"] is None and st["eta_seconds"] is None


def test_notes_are_throttled_path_free_and_shown_while_running(tmp_path: Path) -> None:
    clock = Clock()
    rec = RunRecorder("dti", ["a"], ["eddy"], runs_dir=tmp_path, clock=clock)
    rec.subject_start("a")
    rec._emit("step_start", subject=pseudonym("a"), step="eddy")
    rec.note("a", "eddy", "Running iteration 1")
    rec.note("a", "eddy", "Running iteration 2")  # within 10 s: dropped
    clock.now += 11
    rec.note("a", "eddy", "--imain=/data/123_dwi.nii.gz")  # has a path: dropped
    rec.note("a", "eddy", "Running iteration 3")
    notes = [e["text"] for e in read_events(rec.path) if e["type"] == "step_note"]
    assert notes == ["Running iteration 1", "Running iteration 3"]
    st = fold_events(read_events(rec.path), now=clock.now)
    assert st["current"]["note"] == "Running iteration 3"


def test_reused_steps_still_count_as_done(tmp_path: Path) -> None:
    clock = Clock()
    rec = RunRecorder("dti", ["a"], ["eddy"], runs_dir=tmp_path, clock=clock)
    with rec.step("a", "eddy"):
        clock.now += 0.1
    st = fold_events(read_events(rec.path), now=clock.now)
    assert st["steps"][0]["done"] == 1 and st["steps"][0]["avg_seconds"] is None
