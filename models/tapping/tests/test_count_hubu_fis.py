"""Tests for scripts/count_hubu_fis.py (task FT-06).

No real HUBU-FIS data: tiny synthetic CSVs and videos are written to a temp
folder. Each rule is proved to FAIL on bad input as well as pass on good input.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[3] / "scripts" / "count_hubu_fis.py"
spec = importlib.util.spec_from_file_location("count_hubu_fis", SRC)
assert spec and spec.loader
count_hubu_fis = importlib.util.module_from_spec(spec)
sys.modules["count_hubu_fis"] = count_hubu_fis
spec.loader.exec_module(count_hubu_fis)

# 3 controls (CONTROL9003 left hand only), 3 patients, 11 clips.
ROWS = [
    ("CONTROL9001_DCHA", 0),
    ("CONTROL9001_IZDA", 1),
    ("CONTROL9002_DCHA", 0),
    ("CONTROL9002_IZDA", 0),
    ("CONTROL9003_IZDA", 0),
    ("ID9101_DCHA", 2),
    ("ID9101_IZDA", 0),
    ("ID9102_DCHA", 1),
    ("ID9102_IZDA", 1),
    ("ID9103_DCHA", 3),
    ("ID9103_IZDA", 1),
]
RAW_IDS = sorted({clip.split("_")[0] for clip, _ in ROWS} | {clip for clip, _ in ROWS})


def make_inputs(tmp_path: Path, rows: list[tuple[str, int]] = ROWS) -> tuple[Path, Path]:
    csv = tmp_path / "fis_diagnostic.csv"
    csv.write_text("ID,UPDRS\n" + "".join(f"{i},{u}\n" for i, u in rows))
    videos = tmp_path / "videos"
    videos.mkdir()
    for clip, _ in rows:
        (videos / f"{clip}.mp4").write_bytes(b"")
    return csv, videos


def run_main(tmp_path: Path, csv: Path, videos: Path, *extra: str) -> int:
    return count_hubu_fis.main(
        [
            "--csv",
            str(csv),
            "--video-dir",
            str(videos),
            "--out",
            str(tmp_path / "out"),
            "--tables-dir",
            str(tmp_path / "tables"),
            *extra,
        ]
    )


def audit_of(tmp_path: Path) -> dict[str, Any]:
    csv, videos = make_inputs(tmp_path)
    assert run_main(tmp_path, csv, videos) == 0
    result: dict[str, Any] = json.loads((tmp_path / "out" / "hubu_fis_counts.json").read_text())
    return result


# --------------------------------------------------------------------------- counts


def test_counts_on_synthetic_cohort(tmp_path: Path) -> None:
    r = audit_of(tmp_path)
    p, c, t = r["participants"], r["clips"], r["targets"]
    assert p["total"] == 6 and p["by_group"] == {"CONTROL": 3, "ID": 3}
    assert p["with_both_hands"] == 5 and p["one_hand_only"] == {"left": 1, "right": 0}
    assert c["by_updrs"] == {"0": 5, "1": 4, "2": 1, "3": 1}
    assert c["updrs_never_used"] == [4]
    assert c["by_hand"] == {"left": 6, "right": 5}
    assert c["control_clips_rated_ge1"] == 1 and c["patient_clips_rated_0"] == 1
    assert t["group_control_vs_id"]["clip_ratio"] == 1.2  # 6 / 5
    assert t["group_control_vs_id"]["participant_ratio"] == 1.0
    assert t["updrs_ge1_vs_0"]["clips"] == {"0": 5, ">=1": 6}
    # Highest hand per person: CONTROL9001 1, CONTROL9002 0, CONTROL9003 0, ID9101 2, ID9102 1, ID9103 3
    assert t["updrs_ge1_vs_0"]["participants_max_hand"] == {"0": 2, ">=1": 4}


def test_asymmetry_uses_only_people_with_both_hands(tmp_path: Path) -> None:
    a = audit_of(tmp_path)["asymmetry"]
    # CONTROL9001 |0-1| = 1, CONTROL9002 0; CONTROL9003 has one hand and is left out.
    assert a["CONTROL"]["participants_with_both_hands"] == 2
    assert a["CONTROL"]["asymmetric"] == 1 and a["CONTROL"]["mean_abs_diff"] == 0.5
    # ID9101 2, ID9102 0, ID9103 2
    assert a["ID"]["asymmetric"] == 2 and a["ID"]["mean_abs_diff"] == 1.33


def test_outputs_contain_no_ids_hashed_or_raw(tmp_path: Path) -> None:
    audit_of(tmp_path)
    text = (tmp_path / "out" / "hubu_fis_counts.json").read_text()
    text += (tmp_path / "out" / "hubu_fis_counts.md").read_text()
    for raw in RAW_IDS:
        assert raw not in text
        assert count_hubu_fis.pseudonym(raw) not in text


def test_terminal_shows_hashed_one_hand_participant_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    audit_of(tmp_path)
    printed = capsys.readouterr().out
    assert count_hubu_fis.pseudonym("CONTROL9003") in printed
    for raw in RAW_IDS:
        assert raw not in printed


def test_id_tables_go_to_tables_dir_not_out(tmp_path: Path) -> None:
    audit_of(tmp_path)
    assert (tmp_path / "tables" / "hubu_fis_participants.csv").exists()
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "hubu_fis_counts.json",
        "hubu_fis_counts.md",
    ]


def test_output_is_deterministic(tmp_path: Path) -> None:
    csv, videos = make_inputs(tmp_path)
    run_main(tmp_path, csv, videos)
    first = (tmp_path / "out" / "hubu_fis_counts.json").read_bytes()
    first_md = (tmp_path / "out" / "hubu_fis_counts.md").read_bytes()
    run_main(tmp_path, csv, videos)
    assert (tmp_path / "out" / "hubu_fis_counts.json").read_bytes() == first
    assert (tmp_path / "out" / "hubu_fis_counts.md").read_bytes() == first_md


def test_report_avoids_diagnosis_wording(tmp_path: Path) -> None:
    audit_of(tmp_path)
    text = (tmp_path / "out" / "hubu_fis_counts.md").read_text().lower()
    # The dataset's own label file is called fis_diagnostic.csv; only that name may appear.
    assert "diagnos" not in text.replace("fis_diagnostic.csv", "")


# --------------------------------------------------------------------------- exit codes


def test_missing_inputs_exit_2(tmp_path: Path) -> None:
    csv, videos = make_inputs(tmp_path)
    assert run_main(tmp_path, tmp_path / "nope.csv", videos) == 2
    assert run_main(tmp_path, csv, tmp_path / "no_videos") == 2


@pytest.mark.parametrize(
    "breakage",
    ["duplicate_row", "video_removed", "extra_video", "bad_id", "blank_updrs", "two_files"],
)
def test_integrity_failure_exits_1_and_writes_nothing(tmp_path: Path, breakage: str) -> None:
    csv, videos = make_inputs(tmp_path)
    if breakage == "duplicate_row":
        csv.write_text(csv.read_text() + "ID9101_DCHA,2\n")
    elif breakage == "video_removed":
        (videos / "ID9101_DCHA.mp4").unlink()
    elif breakage == "extra_video":
        (videos / "ID98765_DCHA.mp4").write_bytes(b"")
    elif breakage == "bad_id":
        csv.write_text(csv.read_text().replace("ID9102_DCHA", "ID9102-DCHA"))
    elif breakage == "blank_updrs":
        csv.write_text(csv.read_text().replace("ID9103_DCHA,3", "ID9103_DCHA,"))
    elif breakage == "two_files":
        (videos / "ID9101_DCHA.MOV").write_bytes(b"")
    assert run_main(tmp_path, csv, videos) == 1
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "tables").exists()


# --------------------------------------------------------------------------- video probe


def write_clip(path: Path, width: int, height: int, fps: float, frames: int = 10) -> None:
    cv2 = pytest.importorskip("cv2")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened()
    for _ in range(frames):
        writer.write(np.zeros((height, width, 3), dtype=np.uint8))
    writer.release()


def test_probe_reads_size_fps_and_orientation(tmp_path: Path) -> None:
    clip = tmp_path / "ID9001_DCHA.MOV"  # upper-case suffix on purpose
    write_clip(clip, width=48, height=64, fps=30.0)
    info = count_hubu_fis.probe_clip(clip)
    assert (info["display_width"], info["display_height"]) == (48, 64)
    assert info["orientation_display"] == "portrait"
    assert info["fps_rounded"] == 30 and info["frames"] == 10
    assert info["container"] == "mov" and info["rotation_consistent"]


def test_probe_finds_group_exclusive_settings(tmp_path: Path) -> None:
    csv, videos = make_inputs(tmp_path)
    for clip, _ in ROWS:
        # Every patient clip is landscape at 60 fps: a perfect device shortcut.
        is_patient = clip.startswith("ID")
        size = (64, 48) if is_patient else (48, 64)
        write_clip(videos / f"{clip}.mp4", *size, fps=60.0 if is_patient else 30.0)
    assert run_main(tmp_path, csv, videos, "--probe-video") == 0
    probe = json.loads((tmp_path / "out" / "hubu_fis_counts.json").read_text())["video_probe"]
    exclusive = {(e["property"], e["value"], e["group"]) for e in probe["group_exclusive_values"]}
    assert ("fps", "60", "ID") in exclusive and ("fps", "30", "CONTROL") in exclusive
    assert ("orientation", "landscape", "ID") in exclusive
    assert probe["clips_with_any_group_exclusive_value"] == {"CONTROL": 5, "ID": 6}
    assert (tmp_path / "tables" / "hubu_fis_clip_qc.csv").exists()


def test_unreadable_clip_fails_loudly(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    csv, videos = make_inputs(tmp_path)  # empty files: nothing decodes
    assert run_main(tmp_path, csv, videos, "--probe-video") == 1
    assert not (tmp_path / "out").exists()


def test_unreadable_clip_does_not_leak_ids_to_stderr(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """OpenCV prints the full file path (the raw clip ID) on a failed open, from C code.

    capfd sees file descriptor 2, so this fails if the probe ever opens a clip without muting
    it. Both an empty file and a garbage file are tried; the names are made up.
    """
    pytest.importorskip("cv2")
    csv, videos = make_inputs(tmp_path)  # empty files: nothing decodes
    garbage = videos / "CONTROL9001_DCHA.mp4"
    garbage.write_bytes(np.random.default_rng(0).bytes(4096))
    assert run_main(tmp_path, csv, videos, "--probe-video") == 1
    captured = capfd.readouterr()
    for clip_id, _ in ROWS:
        assert clip_id not in captured.out + captured.err
        assert clip_id.split("_")[0] not in captured.out + captured.err
    assert "clips could not be read" in captured.err  # the hashed-ID message is still shown


# --------------------------------------------------------------------------- probe guard


def seed_counts(tmp_path: Path, text: str) -> Path:
    """An existing counts JSON in --out, as left by an earlier run."""
    (tmp_path / "out").mkdir()
    path = tmp_path / "out" / "hubu_fis_counts.json"
    path.write_text(text)
    return path


def test_refuses_to_drop_probe_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv, videos = make_inputs(tmp_path)
    existing = seed_counts(tmp_path, '{"video_probe": {"tool": "x"}}\n')
    assert run_main(tmp_path, csv, videos) == 1
    assert existing.read_text() == '{"video_probe": {"tool": "x"}}\n'
    assert not (tmp_path / "out" / "hubu_fis_counts.md").exists()
    assert not (tmp_path / "tables").exists()
    err = capsys.readouterr().err
    assert "video_probe" in err and "--allow-drop-probe" in err


def test_allow_drop_probe_overrides(tmp_path: Path) -> None:
    csv, videos = make_inputs(tmp_path)
    existing = seed_counts(tmp_path, '{"video_probe": {"tool": "x"}}\n')
    assert run_main(tmp_path, csv, videos, "--allow-drop-probe") == 0
    assert "video_probe" not in json.loads(existing.read_text())
    assert (tmp_path / "tables" / "hubu_fis_participants.csv").exists()


def test_existing_file_without_probe_is_overwritten(tmp_path: Path) -> None:
    csv, videos = make_inputs(tmp_path)
    existing = seed_counts(tmp_path, '{"files": {}}\n')
    assert run_main(tmp_path, csv, videos) == 0
    assert json.loads(existing.read_text())["participants"]["total"] == 6


def test_no_existing_file_writes_normally(tmp_path: Path) -> None:
    csv, videos = make_inputs(tmp_path)
    assert not (tmp_path / "out").exists()
    assert run_main(tmp_path, csv, videos) == 0
    assert (tmp_path / "out" / "hubu_fis_counts.json").exists()


def test_probe_run_overwrites_probe_file(tmp_path: Path) -> None:
    csv, videos = make_inputs(tmp_path)
    for clip, _ in ROWS:
        write_clip(videos / f"{clip}.mp4", 48, 64, fps=30.0)
    existing = seed_counts(tmp_path, '{"video_probe": {"tool": "old"}}\n')
    assert run_main(tmp_path, csv, videos, "--probe-video") == 0
    assert json.loads(existing.read_text())["video_probe"]["tool"] != "old"


@pytest.mark.parametrize("text", ["{not json", "", "[1, 2]", '"video_probe"'])
def test_unreadable_existing_json_fails_and_is_kept(
    tmp_path: Path, text: str, capsys: pytest.CaptureFixture[str]
) -> None:
    csv, videos = make_inputs(tmp_path)
    existing = seed_counts(tmp_path, text)
    assert run_main(tmp_path, csv, videos) == 1
    assert existing.read_text() == text
    assert not (tmp_path / "out" / "hubu_fis_counts.md").exists()
    assert not (tmp_path / "tables").exists()
    assert "--allow-drop-probe" in capsys.readouterr().err
