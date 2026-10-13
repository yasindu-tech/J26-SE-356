"""Tests for models/tapping/src/hubu_fis.py (task FT-06).

Synthetic data only: tiny CSVs and empty "video" files under a temp folder.
Nothing from data/ is read. Every integrity rule is proved to FAIL on input
that breaks it, not only to pass on clean input.

No conftest.py on purpose: voice and gait each have one, and pytest run from
the repo root already confuses them. ``src`` is put on ``sys.path`` here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hubu_fis import (  # noqa: E402
    HubuFisError,
    check_match,
    list_videos,
    load_labels,
    participant_table,
    pseudonym,
)

# Clean cohort: two controls with both hands, one patient with both hands,
# one patient with the left hand only.
ROWS = [
    ("CONTROL9001_DCHA", "0"),
    ("CONTROL9001_IZDA", "1"),
    ("CONTROL9002_DCHA", "0"),
    ("CONTROL9002_IZDA", "0"),
    ("ID9101_DCHA", "2"),
    ("ID9101_IZDA", "0"),
    ("ID9102_IZDA", "3"),
]


def write_csv(tmp_path: Path, rows: list[tuple[str, str]], header: str = "ID,UPDRS") -> Path:
    path = tmp_path / "fis_diagnostic.csv"
    path.write_text(header + "\n" + "".join(f"{i},{u}\n" for i, u in rows))
    return path


def write_videos(tmp_path: Path, names: list[str]) -> Path:
    folder = tmp_path / "videos"
    folder.mkdir(exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"")
    return folder


# --------------------------------------------------------------------------- labels


def test_clean_labels_load(tmp_path: Path) -> None:
    labels = load_labels(write_csv(tmp_path, ROWS))
    assert len(labels) == 7
    assert labels["UPDRS"].tolist() == [0, 1, 0, 0, 2, 0, 3]
    assert labels["line"].tolist() == [2, 3, 4, 5, 6, 7, 8]
    assert set(labels["group"]) == {"CONTROL", "ID"}


def test_windows_line_endings_load(tmp_path: Path) -> None:
    """The real HUBU-FIS CSV uses CRLF; the carriage return must not reach ID or UPDRS."""
    path = tmp_path / "fis_diagnostic.csv"
    path.write_bytes(("ID,UPDRS\r\n" + "".join(f"{i},{u}\r\n" for i, u in ROWS)).encode())
    labels = load_labels(path)
    assert labels["ID"].tolist() == [i for i, _ in ROWS]
    assert labels["UPDRS"].tolist() == [0, 1, 0, 0, 2, 0, 3]


@pytest.mark.parametrize("bad", ["", " ", "5", "-1", "1.0", "1.5", "one", " 1"])
def test_blank_or_invalid_updrs_fails(tmp_path: Path, bad: str) -> None:
    """A blank rating is an error, never read as 0; out-of-range values are errors too."""
    with pytest.raises(HubuFisError, match="UPDRS.*CSV lines: 9"):
        load_labels(write_csv(tmp_path, [*ROWS, ("ID9103_DCHA", bad)]))


@pytest.mark.parametrize("bad", ["ID9103_RIGHT", "ID9103_DCHA ", " ID9103_DCHA", "id103_dcha", ""])
def test_unparseable_id_fails_with_line_not_id(tmp_path: Path, bad: str) -> None:
    with pytest.raises(HubuFisError, match="CSV lines: 9") as err:
        load_labels(write_csv(tmp_path, [*ROWS, (bad, "1")]))
    assert "ID9103" not in str(err.value)


def test_duplicate_id_fails(tmp_path: Path) -> None:
    with pytest.raises(HubuFisError, match="share an ID.*CSV lines: 2, 9"):
        load_labels(write_csv(tmp_path, [*ROWS, ("CONTROL9001_DCHA", "0")]))


def test_duplicate_id_with_different_rating_fails(tmp_path: Path) -> None:
    with pytest.raises(HubuFisError, match="share an ID"):
        load_labels(write_csv(tmp_path, [*ROWS, ("CONTROL9001_DCHA", "3")]))


def test_same_person_spelled_two_ways_fails(tmp_path: Path) -> None:
    """CONTROL09001 and CONTROL9001 would become two participant keys for one person."""
    with pytest.raises(HubuFisError, match="more than one way") as err:
        load_labels(write_csv(tmp_path, [*ROWS, ("CONTROL09001_DCHA", "0")]))
    assert pseudonym("CONTROL9001") in str(err.value)
    assert "CONTROL9001" not in str(err.value).replace(pseudonym("CONTROL9001"), "")


@pytest.mark.parametrize(
    "text",
    [
        "ID,UPDRS,extra\nID9101_DCHA,1,x\n",  # extra column
        "ID\nID9101_DCHA\n",  # UPDRS missing
        "id,UPDRS\nID9101_DCHA,1\n",  # wrong case
        "UPDRS,ID\n1,ID9101_DCHA\n",  # wrong order
        "ID ,UPDRS\nID9101_DCHA,1\n",  # whitespace in header
    ],
)
def test_wrong_columns_fail(tmp_path: Path, text: str) -> None:
    path = tmp_path / "fis_diagnostic.csv"
    path.write_text(text)
    with pytest.raises(HubuFisError, match="columns"):
        load_labels(path)


def test_empty_csv_fails(tmp_path: Path) -> None:
    with pytest.raises(HubuFisError, match="no rows"):
        load_labels(write_csv(tmp_path, []))


# --------------------------------------------------------------------------- videos


def test_suffix_matching_ignores_case(tmp_path: Path) -> None:
    folder = write_videos(tmp_path, ["A.mp4", "B.MOV", "C.Mp4", "D.mOv", "notes.txt", ".DS_Store"])
    videos, ignored = list_videos(folder)
    assert sorted(videos) == ["A", "B", "C", "D"]
    assert videos["B"].suffix == ".MOV"
    assert sorted(p.name for p in ignored) == [".DS_Store", "notes.txt"]


def test_subfolder_is_ignored_not_a_video(tmp_path: Path) -> None:
    folder = write_videos(tmp_path, ["A.mp4"])
    (folder / "nested.mp4").mkdir()
    videos, ignored = list_videos(folder)
    assert list(videos) == ["A"] and [p.name for p in ignored] == ["nested.mp4"]


def test_two_files_for_one_clip_fail(tmp_path: Path) -> None:
    folder = write_videos(tmp_path, ["ID9101_DCHA.mp4", "ID9101_DCHA.MOV"])
    with pytest.raises(HubuFisError, match="more than one video file") as err:
        list_videos(folder)
    assert pseudonym("ID9101_DCHA") in str(err.value)
    assert "ID9101_DCHA" not in str(err.value).replace(pseudonym("ID9101_DCHA"), "")


def _clean_videos(tmp_path: Path) -> dict[str, Path]:
    names = [f"{i}.mp4" for i, _ in ROWS[:-1]] + [f"{ROWS[-1][0]}.MOV"]
    return list_videos(write_videos(tmp_path, names))[0]


def test_clean_labels_and_videos_match(tmp_path: Path) -> None:
    check_match(load_labels(write_csv(tmp_path, ROWS)), _clean_videos(tmp_path))


def test_label_without_video_fails(tmp_path: Path) -> None:
    videos = _clean_videos(tmp_path)
    videos.pop("ID9101_IZDA")
    with pytest.raises(HubuFisError, match=r"labelled but no video \(1\)") as err:
        check_match(load_labels(write_csv(tmp_path, ROWS)), videos)
    assert pseudonym("ID9101_IZDA") in str(err.value)
    assert "ID9101_IZDA" not in str(err.value).replace(pseudonym("ID9101_IZDA"), "")


def test_video_without_label_fails(tmp_path: Path) -> None:
    with pytest.raises(HubuFisError, match=r"video but no label    \(1\)"):
        check_match(load_labels(write_csv(tmp_path, ROWS[:-1])), _clean_videos(tmp_path))


def test_mac_resource_fork_file_is_not_silently_matched(tmp_path: Path) -> None:
    """A copied-from-Mac "._X.mp4" has a video suffix but no label, so it must fail."""
    _clean_videos(tmp_path)
    write_videos(tmp_path, ["._ID9101_DCHA.mp4"])
    videos = list_videos(tmp_path / "videos")[0]
    with pytest.raises(HubuFisError, match="video but no label"):
        check_match(load_labels(write_csv(tmp_path, ROWS)), videos)


# --------------------------------------------------------------------------- participants


def test_participant_table(tmp_path: Path) -> None:
    table = participant_table(load_labels(write_csv(tmp_path, ROWS)))
    assert table.index.tolist() == ["CONTROL9001", "CONTROL9002", "ID9101", "ID9102"]
    assert table.loc["CONTROL9001", "abs_diff_rl"] == 1
    assert table.loc["ID9101", "abs_diff_rl"] == 2
    assert table.loc["ID9101", "updrs_max"] == 2


def test_one_hand_gives_na_not_zero(tmp_path: Path) -> None:
    """A missing hand must not look like a rating of 0 or a symmetric person."""
    row = participant_table(load_labels(write_csv(tmp_path, ROWS))).loc["ID9102"]
    assert bool(row["has_left"]) and not bool(row["has_right"])
    assert pd.isna(row["updrs_right"]) and pd.isna(row["abs_diff_rl"])
    assert row["updrs_max"] == 3


def test_control_and_patient_with_same_number_stay_apart(tmp_path: Path) -> None:
    rows = [("CONTROL9101_DCHA", "0"), ("ID9101_DCHA", "2")]
    table = participant_table(load_labels(write_csv(tmp_path, rows)))
    assert table.index.tolist() == ["CONTROL9101", "ID9101"]


def test_same_hand_twice_fails() -> None:
    labels = pd.DataFrame(
        {"participant": ["ID9001", "ID9001"], "group": ["ID", "ID"], "hand": ["left", "left"]}
    ).assign(UPDRS=[1, 2])
    with pytest.raises(HubuFisError, match="same hand"):
        participant_table(labels)


def test_hands_with_different_groups_fail() -> None:
    labels = pd.DataFrame(
        {"participant": ["P1", "P1"], "group": ["CONTROL", "ID"], "hand": ["left", "right"]}
    ).assign(UPDRS=[0, 2])
    with pytest.raises(HubuFisError, match="different groups"):
        participant_table(labels)
