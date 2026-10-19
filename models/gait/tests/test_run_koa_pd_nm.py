"""Tests for the KOA-PD-NM run (GAIT-32). The predictor is faked: no video or model."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
import run_koa_pd_nm as rk

NAMES = [
    "NM/001_NM_01.MOV",
    "NM/001_NM_02.MOV",
    "NM/002_NM_01.MOV",
    "PD/PD_SV/001_PD_01_SV.MOV",
    "PD/PD_ML/003_PD_01_ML.MOV",
    "KOA/KOA_EL/009_KOA_01_EL.MOV",
]


def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "KOA-PD-NM"
    for name in [*NAMES, "NM/notes.MOV", "NM/readme.txt"]:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(b"x")
    return root


def scored(path: Path) -> dict[str, Any]:
    return {
        "available": True,
        "reasons": [],
        "warnings": ["One person in view is assumed: the video was not checked for other people."],
        "walk": {"seconds": 8.0},
        "gate": {"checks": {"frame_rate": {"value": 30.0}, "strides": {"value": 9.0}}},
        "predicted_class": 1,
        "probabilities": [0.2, 0.5, 0.3],
        "expected_severity": 1.1,
        "feature_vector": {"cadence": 100.0},
        "model_version": "gait-xgb-test",
    }


class FakePredictor:
    def __init__(self, fail_on: str = "") -> None:
        self.seen: list[Path] = []
        self.fail_on = fail_on

    def predict(self, path: Path) -> dict[str, Any]:
        self.seen.append(path)
        if self.fail_on and self.fail_on in path.name:
            raise RuntimeError("MediaPipe fell over")
        if "002_NM" in path.name:
            return scored(path) | {
                "available": False,
                "reasons": ["the video could not be opened"],
                "predicted_class": None,
            }
        return scored(path)


def rows(out: Path) -> list[dict[str, str]]:
    with (out / "results.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def test_clip_names_give_group_level_person_and_trial(tmp_path: Path) -> None:
    root = dataset(tmp_path)
    clips, odd = rk.find_clips(root)
    assert len(clips) == 6 and odd == 1  # notes.MOV has another name; the .txt is ignored
    sv = next(c for c in clips if c.group == "PD" and c.level == "SV")
    assert sv.trial == "01" and sv.known_two_people is True
    nm = [c for c in clips if c.group == "NM"]
    assert {c.level for c in nm} == {""}
    assert nm[0].person == nm[1].person != nm[2].person  # one person, two clips
    assert rk.person_key("NM", "001") != rk.person_key("PD", "001")  # IDs repeat across groups
    assert all(c.clip.startswith("w-") and c.person.startswith("p-") for c in clips)


def test_a_run_writes_one_row_per_clip_and_never_a_file_name(tmp_path: Path) -> None:
    root, out = dataset(tmp_path), tmp_path / "out"
    clips, _ = rk.find_clips(root)
    counts = rk.run(FakePredictor(), clips, out, set())
    got = rows(out)
    assert len(got) == 6 and list(got[0]) == rk.COLUMNS
    assert counts["NM"] == {"scored": 2, "not_scored": 1, "failed": 0}
    bad = next(r for r in got if r["status"] == "not_scored")
    assert bad["reasons"] == "the video could not be opened" and bad["p0"] == ""
    good = next(r for r in got if r["status"] == "scored")
    assert good["p1"] == "0.5" and good["cadence"] == "100.0" and good["strides"] == "9.0"
    text = (out / "results.csv").read_text() + (out / "records.jsonl").read_text()
    assert ".MOV" not in text and "001_NM" not in text
    assert len((out / "records.jsonl").read_text().splitlines()) == 6


def test_one_crashing_clip_is_recorded_and_the_run_carries_on(tmp_path: Path) -> None:
    root, out = dataset(tmp_path), tmp_path / "out"
    clips, _ = rk.find_clips(root)
    counts = rk.run(FakePredictor(fail_on="003_PD"), clips, out, set())
    failed = [r for r in rows(out) if r["status"] == "failed"]
    assert len(failed) == 1 and failed[0]["reasons"] == "RuntimeError: MediaPipe fell over"
    assert counts["PD"]["failed"] == 1 and len(rows(out)) == 6


def test_a_rerun_skips_the_clips_already_done(tmp_path: Path) -> None:
    root, out = dataset(tmp_path), tmp_path / "out"
    clips, _ = rk.find_clips(root)
    rk.run(FakePredictor(), clips[:2], out, set())
    again = FakePredictor()
    rk.run(again, clips, out, rk.done_clips(out / "results.csv"))
    assert len(again.seen) == 4 and len(rows(out)) == 6
    assert len({r["clip"] for r in rows(out)}) == 6  # no clip twice
    assert rows(out)[0].keys() == rows(out)[-1].keys()  # one header only


def test_cli_dry_run_and_limit_list_without_running(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = dataset(tmp_path)

    def no_model(*_a: object) -> None:
        raise AssertionError("the model must not load on a dry run")

    monkeypatch.setattr(rk, "GaitPredictor", no_model)
    assert rk.main(["--data-dir", str(root), "--out-dir", str(tmp_path / "o"), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "NM     3 clips,   2 people" in out and "6 clips, 0 already done, 6 to run" in out
    assert "1 video file(s) with another name skipped" in out
    args = ["--data-dir", str(root), "--out-dir", str(tmp_path / "o"), "--dry-run", "--limit", "3"]
    assert rk.main(args) == 0
    out = capsys.readouterr().out
    assert "3 clips, 0 already done, 3 to run" in out
    assert "NM     1 clips" in out and "PD     1 clips" in out and "KOA    1 clips" in out


def test_a_limit_is_shared_between_the_groups(tmp_path: Path) -> None:
    clips, _ = rk.find_clips(dataset(tmp_path))
    picked = rk.spread(clips, 5)
    assert [c.group for c in picked] == ["KOA", "NM", "NM", "PD", "PD"]  # 2, 2 and 1
    assert {c.group for c in rk.spread(clips, 3)} == {"NM", "PD", "KOA"}
    assert len(rk.spread(clips, 30)) == 6  # never more than there are


def test_cli_runs_with_the_predictor_and_reports_by_group(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, out = dataset(tmp_path), tmp_path / "o"
    made: list[tuple[Any, ...]] = []

    def fake(*a: Any) -> FakePredictor:
        made.append(a)
        return FakePredictor()

    monkeypatch.setattr(rk, "GaitPredictor", fake)
    assert rk.main(["--data-dir", str(root), "--out-dir", str(out), "--max-seconds", "12"]) == 0
    assert made[0][2] == 12.0 and len(rows(out)) == 6
    printed = capsys.readouterr().out
    assert "NM  scored 2, not_scored 1, failed 0" in printed and "min left" in printed
    assert rk.main(["--data-dir", str(root), "--out-dir", str(out)]) == 0  # all done: no load
    assert len(made) == 1
    record = json.loads((out / "records.jsonl").read_text().splitlines()[0])
    assert record["clip"].startswith("w-")


def test_cli_stops_cleanly_when_the_model_cannot_be_used(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = dataset(tmp_path)
    args = ["--data-dir", str(root), "--out-dir", str(tmp_path / "o")]
    assert rk.main([*args, "--bundle", str(tmp_path / "none")]) == 2
    assert "Cannot start" in capsys.readouterr().out


def test_a_row_cut_short_by_a_stopped_run_is_run_again(tmp_path: Path) -> None:
    root, out = dataset(tmp_path), tmp_path / "out"
    clips, _ = rk.find_clips(root)
    rk.run(FakePredictor(), clips[:2], out, set())
    with (out / "results.csv").open("a") as f:
        f.write(f"{clips[2].clip},NM,,p-x")  # the process stopped mid-row
    assert rk.done_clips(out / "results.csv") == {clips[0].clip, clips[1].clip}
