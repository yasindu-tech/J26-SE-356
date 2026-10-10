"""Run events for the MRI pipelines, and the code that turns them into a live status.

A pipeline creates one ``RunRecorder`` per run and reports as it goes:

    rec = RunRecorder("t1", subjects, steps=["n4", "skull", ...])
    rec.subject_start(sid)
    with rec.step(sid, "n4"):
        n4_correct(...)
    rec.subject_end(sid, "ok", metrics={"brain_mask_ml": 1524.0})
    rec.run_end()

Each report is one line of JSON appended to ``data/interim/runs/<pipeline>_<time>.jsonl``.
If an ``events_url`` is given, the same line is also POSTed there (for a dashboard
running on another machine). A failed POST never stops the pipeline.

Privacy: events hold only hashed subject IDs, step names, timings, numeric check
values and failure reasons. Never images, file paths or PPMI subject IDs.

``fold_events`` replays a run's events into the state the dashboard shows. It is a
pure function, so it is tested without running any pipeline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = REPO_ROOT / "data" / "interim" / "runs"

log = logging.getLogger("progress")


def pseudonym(subject_id: str) -> str:
    """The only form of a subject ID that ever goes into an event."""
    return hashlib.sha256(subject_id.encode()).hexdigest()[:8]


class RunRecorder:
    """Writes the events of one pipeline run. Safe to use when nothing is listening."""

    def __init__(
        self,
        pipeline: str,
        subjects: list[str],
        steps: list[str],
        runs_dir: Path = RUNS_DIR,
        events_url: str | None = None,
        clock: Any = time.time,
        jobs: int = 1,
    ) -> None:
        self.pipeline = pipeline
        self.clock = clock
        self.events_url = events_url
        self.run_id = f"{pipeline}_{time.strftime('%Y%m%d_%H%M%S', time.localtime(clock()))}"
        runs_dir.mkdir(parents=True, exist_ok=True)
        self.path = runs_dir / f"{self.run_id}.jsonl"
        self._lock = threading.Lock()  # several people may be processed at once (--jobs)
        self._last_note: dict[tuple[str, str], float] = {}
        self._emit(
            "run_start",
            subjects=[pseudonym(s) for s in subjects],
            steps=steps,
            jobs=jobs,
        )

    def _emit(self, kind: str, **fields: Any) -> None:
        event = {
            "t": round(self.clock(), 3),
            "run": self.run_id,
            "pipeline": self.pipeline,
            "type": kind,
            **fields,
        }
        line = json.dumps(event)
        with self._lock, self.path.open("a") as f:
            f.write(line + "\n")
        if self.events_url:
            try:
                req = urllib.request.Request(
                    self.events_url,
                    data=line.encode(),
                    headers={"Content-Type": "application/json"},
                )
                urllib.request.urlopen(req, timeout=2).close()
            except OSError as exc:  # dashboard down or unreachable: keep processing
                log.warning("could not send event to %s: %s", self.events_url, exc)

    def subject_start(self, subject_id: str) -> None:
        self._emit("subject_start", subject=pseudonym(subject_id))

    @contextmanager
    def step(self, subject_id: str, step: str) -> Iterator[None]:
        """Time one step. Records a failure, then re-raises, if the step raises."""
        sid = pseudonym(subject_id)
        self._emit("step_start", subject=sid, step=step)
        start = self.clock()
        try:
            yield
        except BaseException:
            self._emit(
                "step_end", subject=sid, step=step, ok=False, seconds=round(self.clock() - start, 2)
            )
            raise
        self._emit(
            "step_end", subject=sid, step=step, ok=True, seconds=round(self.clock() - start, 2)
        )

    def note(self, subject_id: str, step: str, text: str, every_s: float = 10.0) -> None:
        """Latest status line from a long-running tool (e.g. eddy --verbose).

        Throttled to one event per ``every_s`` seconds per person. Lines containing a
        path separator are dropped, because tool output can echo file names, and file
        names carry PPMI subject IDs.
        """
        text = " ".join(text.split())[:120]
        if not text or "/" in text or "\\" in text:
            return
        key = (pseudonym(subject_id), step)
        now = self.clock()
        if now - self._last_note.get(key, -1e9) < every_s:
            return
        self._last_note[key] = now
        self._emit("step_note", subject=key[0], step=step, text=text)

    def subject_end(
        self,
        subject_id: str,
        status: str,
        reason: str = "",
        metrics: dict[str, float | None] | None = None,
    ) -> None:
        clean = {k: v for k, v in (metrics or {}).items() if isinstance(v, int | float)}
        self._emit(
            "subject_end",
            subject=pseudonym(subject_id),
            status=status,
            reason=reason,
            metrics=clean,
        )

    def run_end(self) -> None:
        self._emit("run_end")


class NullRecorder:
    """Used when a pipeline function is called without a recorder (tests, single calls)."""

    def subject_start(self, subject_id: str) -> None:  # noqa: D102
        return

    @contextmanager
    def step(self, subject_id: str, step: str) -> Iterator[None]:  # noqa: D102
        yield

    def note(self, *args: Any, **kwargs: Any) -> None:  # noqa: D102
        return

    def subject_end(self, *args: Any, **kwargs: Any) -> None:  # noqa: D102
        return

    def run_end(self) -> None:  # noqa: D102
        return


# ---------------------------------------------------------------- reading runs


def read_events(path: Path) -> list[dict]:
    """All events in a run file. A half-written last line (run in progress) is skipped."""
    events = []
    for line in path.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def latest_run(pipeline: str, runs_dir: Path = RUNS_DIR) -> Path | None:
    files = sorted(runs_dir.glob(f"{pipeline}_*.jsonl"))
    return files[-1] if files else None


# A step that took less than this was reused from an earlier run (its outputs already
# existed), so it says nothing about how long the step really takes.
CACHED_BELOW_S = 1.0
REUSED_SUBJECT_BELOW_S = 60.0


def step_durations(events: list[dict]) -> dict[str, list[float]]:
    """Real (not reused) durations of every successful step in one run."""
    out: dict[str, list[float]] = {}
    for e in events:
        if e.get("type") == "step_end" and e.get("ok") and e.get("seconds", 0) >= CACHED_BELOW_S:
            out.setdefault(e["step"], []).append(float(e["seconds"]))
    return out


def expected_step_seconds(runs: list[list[dict]]) -> dict[str, float]:
    """Typical (median) time per step, learned from finished steps in these runs."""
    pooled: dict[str, list[float]] = {}
    for events in runs:
        for step, secs in step_durations(events).items():
            pooled.setdefault(step, []).extend(secs)
    return {k: sorted(v)[len(v) // 2] for k, v in pooled.items() if v}


def _progress(elapsed: float | None, expected: float | None) -> int | None:
    """Estimated percent of a running step. Capped at 99 until it really finishes."""
    if elapsed is None or not expected:
        return None
    return min(99, int(100 * elapsed / expected))


def fold_events(events: list[dict], now: float, expected: dict[str, float] | None = None) -> dict:
    """Replay events into the dashboard state for one run.

    ``expected`` (step -> typical seconds, from earlier finished steps) turns the
    time spent on a running step into an estimated percentage. FSL eddy reports
    no progress itself, so this estimate is the best available.
    """
    expected = expected or {}
    state: dict[str, Any] = {
        "run": None,
        "pipeline": None,
        "started": None,
        "finished": None,
        "steps": [],
        "subjects": {},
        "order": [],
    }
    for e in events:
        kind = e.get("type")
        if kind == "run_start":
            state.update(
                run=e["run"],
                pipeline=e["pipeline"],
                started=e["t"],
                steps=e["steps"],
                jobs=max(1, int(e.get("jobs", 1))),
            )
            for s in e["subjects"]:
                state["subjects"][s] = {
                    "id": s,
                    "state": "waiting",
                    "step": None,
                    "step_started": None,
                    "started": None,
                    "ended": None,
                    "durations": {},
                    "failed_step": None,
                    "reason": "",
                    "metrics": {},
                    "note": "",
                }
                state["order"].append(s)
            continue
        sub = state["subjects"].get(e.get("subject", ""))
        if kind == "run_end":
            state["finished"] = e["t"]
        elif sub is None:
            continue
        elif kind == "subject_start":
            sub.update(state="running", started=e["t"])
        elif kind == "step_start":
            sub.update(state="running", step=e["step"], step_started=e["t"], note="")
        elif kind == "step_note":
            if sub["step"] == e["step"]:
                sub["note"] = e["text"]
        elif kind == "step_end":
            sub["durations"][e["step"]] = e["seconds"]
            sub["step"], sub["step_started"] = None, None
            if not e["ok"]:
                sub["failed_step"] = e["step"]
        elif kind == "subject_end":
            sub.update(
                state="done" if e["status"] == "ok" else "failed",
                ended=e["t"],
                reason=e.get("reason", ""),
                metrics=e.get("metrics", {}),
                step=None,
            )

    subs = [state["subjects"][s] for s in state["order"]]
    if state["finished"]:
        for s in subs:  # a run that ended leaves nobody "running"
            if s["state"] == "running":
                s.update(
                    state="failed", reason=s["reason"] or "run ended before this person finished"
                )

    done = [s for s in subs if s["state"] == "done"]
    # A person whose whole run took under a minute was reused from earlier outputs, not
    # processed; counting them would make the average and time-left far too optimistic.
    per_subject = [
        s["ended"] - s["started"]
        for s in done
        if s["started"] and s["ended"] and s["ended"] - s["started"] >= REUSED_SUBJECT_BELOW_S
    ]
    avg = sum(per_subject) / len(per_subject) if per_subject else None
    remaining = sum(s["state"] in ("waiting", "running") for s in subs)
    running_now = [s for s in subs if s["state"] == "running"]
    current = running_now[0] if running_now else None
    jobs = state.get("jobs", 1)

    step_stats = []
    for st in state["steps"]:
        finished = [
            s["durations"][st] for s in subs if st in s["durations"] and s["failed_step"] != st
        ]
        times = [t for t in finished if t >= CACHED_BELOW_S]  # reused steps say nothing about speed
        step_stats.append(
            {
                "step": st,
                "done": len(finished),
                "failed": sum(s["failed_step"] == st for s in subs),
                "avg_seconds": round(sum(times) / len(times), 1) if times else None,
            }
        )

    end = state["finished"] or now
    return {
        "run": state["run"],
        "pipeline": state["pipeline"],
        "status": "finished" if state["finished"] else ("running" if state["run"] else "no run"),
        "started": state["started"],
        "elapsed_seconds": round(end - state["started"], 1) if state["started"] else 0,
        "total": len(subs),
        "done": len(done),
        "failed": sum(s["state"] == "failed" for s in subs),
        "running": sum(s["state"] == "running" for s in subs),
        "waiting": sum(s["state"] == "waiting" for s in subs),
        "avg_seconds_per_subject": round(avg, 1) if avg else None,
        # with --jobs N, N people are processed at once, so the queue empties N times faster
        "eta_seconds": round(avg * remaining / jobs, 0)
        if avg and remaining and not state["finished"]
        else None,
        "jobs": jobs,
        "running_now": [
            {
                "id": s["id"],
                "step": s["step"],
                "step_seconds": round(now - s["step_started"], 1) if s["step_started"] else None,
                "note": s.get("note", ""),
                "expected_seconds": expected.get(s["step"] or ""),
                "pct": _progress(
                    now - s["step_started"] if s["step_started"] else None,
                    expected.get(s["step"] or ""),
                ),
            }
            for s in running_now
        ],
        "current": None
        if current is None
        else {
            "id": current["id"],
            "step": current["step"],
            "step_seconds": round(now - current["step_started"], 1)
            if current["step_started"]
            else None,
            "subject_seconds": round(now - current["started"], 1) if current["started"] else None,
            "note": current.get("note", ""),
            "expected_seconds": expected.get(current["step"] or ""),
            "pct": _progress(
                now - current["step_started"] if current["step_started"] else None,
                expected.get(current["step"] or ""),
            ),
        },
        "steps": step_stats,
        "subjects": subs,
    }
