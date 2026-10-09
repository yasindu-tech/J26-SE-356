"""Tests for scripts/count_uci470.py.

No real UCI-470 data is used: tiny synthetic CSVs are written to a temp folder.
Each rule is proved to FAIL on bad input as well as pass on good input.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[3] / "scripts" / "count_uci470.py"
spec = importlib.util.spec_from_file_location("count_uci470", SRC)
assert spec and spec.loader
count_uci470 = importlib.util.module_from_spec(spec)
sys.modules["count_uci470"] = count_uci470
spec.loader.exec_module(count_uci470)

GROUP_ROW = ",,Baseline Features,,MFCC,\n"
NAME_ROW = "id,gender,PPE,DFA,mean_MFCC_0th_coef,class\n"


def rows(people: list[tuple[int, int, int]], per_person: int = 3) -> str:
    """(id, gender, class) -> CSV body with `per_person` rows each."""
    return "".join(
        f"{pid},{g},0.1,0.2,0.3,{c}\n" for pid, g, c in people for _ in range(per_person)
    )


PEOPLE = [(101, 1, 1), (102, 0, 1), (103, 0, 0)]


def run(tmp_path: Path, text: str) -> dict[str, object]:
    path = tmp_path / "pd_speech_features.csv"
    path.write_text(text)
    table, groups = count_uci470.read_table(path)
    return count_uci470.audit(table, groups)


def test_two_header_rows_detected_and_grouped(tmp_path: Path) -> None:
    result = run(tmp_path, GROUP_ROW + NAME_ROW + rows(PEOPLE))
    assert (result["people"], result["rows"], result["feature_columns"]) == (3, 9, 3)
    assert (result["people_pd"], result["people_healthy"]) == (2, 1)
    assert result["group_counts"].to_dict() == {"Baseline Features": 2, "MFCC": 1}


def test_single_header_row_detected(tmp_path: Path) -> None:
    result = run(tmp_path, NAME_ROW + rows(PEOPLE))
    assert (result["people"], result["feature_columns"]) == (3, 3)


def test_blank_class_is_missing_not_healthy(tmp_path: Path) -> None:
    body = rows(PEOPLE) + rows([(104, 1, 0)]).replace(",0\n", ",\n")
    result = run(tmp_path, GROUP_ROW + NAME_ROW + body)
    assert result["people_healthy"] == 1  # 104 must not be counted as healthy
    assert result["people_label_missing"] == 1
    assert result["rows_label_missing_or_invalid"] == 3


def test_missing_class_column_fails(tmp_path: Path) -> None:
    text = "id,gender,PPE\n1,0,0.1\n"
    with pytest.raises(ValueError, match="class"):
        run(tmp_path, text)


def test_conflicting_labels_flagged_and_hashed(tmp_path: Path) -> None:
    body = rows(PEOPLE) + "105,1,0.1,0.2,0.3,1\n105,1,0.1,0.2,0.3,0\n"
    result = run(tmp_path, GROUP_ROW + NAME_ROW + body)
    assert result["inconsistent_people"] == [count_uci470.pseudonym(105)]
    assert result["people_pd"] + result["people_healthy"] == 3  # 105 excluded


def test_report_never_prints_raw_ids(tmp_path: Path) -> None:
    body = rows(PEOPLE) + "987654,1,0.1,0.2,0.3,1\n987654,1,0.1,0.2,0.3,0\n"
    text = count_uci470.report(run(tmp_path, GROUP_ROW + NAME_ROW + body))
    assert "987654" not in text
    assert count_uci470.pseudonym(987654) in text


def test_main_exits_nonzero_on_problems(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(NAME_ROW + rows(PEOPLE) + "106,1,0.1,0.2,0.3,\n")
    assert count_uci470.main(["--csv", str(path)]) == 1
    path.write_text(NAME_ROW + rows(PEOPLE))
    assert count_uci470.main(["--csv", str(path)]) == 0
