"""Tests for models/voice/src/data.py. Each check is proved to fail on bad input."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from conftest import write_two_header_csv

import data

N_ROWS = 180  # 60 synthetic people x 3 rows


def test_loads_two_header_file(synthetic_csv: Path) -> None:
    X, y, groups, gender = data.load_uci470(synthetic_csv, expected_rows=N_ROWS)
    assert X.shape == (N_ROWS, 30)
    assert len(y) == len(groups) == len(gender) == N_ROWS
    assert set(y) <= {0, 1}


def test_loads_single_header_file(synthetic_table: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "one_header.csv"
    synthetic_table.to_csv(path, index=False)
    assert data.load_uci470(path, expected_rows=N_ROWS).X.shape == (N_ROWS, 30)


def test_id_gender_class_never_in_features(synthetic_csv: Path) -> None:
    X = data.load_uci470(synthetic_csv, expected_rows=N_ROWS).X
    assert not set(data.NON_FEATURES) & set(X.columns)


def test_wrong_row_count_fails(synthetic_csv: Path) -> None:
    with pytest.raises(ValueError, match="expected 756 rows"):
        data.load_uci470(synthetic_csv)  # default expects the real file's 756


@pytest.mark.parametrize("column", ["id", "class"])
def test_missing_id_or_class_column_fails(
    synthetic_table: pd.DataFrame, tmp_path: Path, column: str
) -> None:
    path = write_two_header_csv(synthetic_table.drop(columns=column), tmp_path / "x.csv")
    with pytest.raises(ValueError, match=column):
        data.load_uci470(path, expected_rows=N_ROWS)


def test_blank_class_fails_not_read_as_healthy(
    synthetic_table: pd.DataFrame, tmp_path: Path
) -> None:
    table = synthetic_table.astype({"class": "object"})
    table.loc[0, "class"] = None
    path = write_two_header_csv(table, tmp_path / "x.csv")
    with pytest.raises(ValueError, match="blank or invalid class"):
        data.load_uci470(path, expected_rows=N_ROWS)


def test_label_changing_within_a_person_fails(
    synthetic_table: pd.DataFrame, tmp_path: Path
) -> None:
    table = synthetic_table.copy()
    table.loc[0, "class"] = 1 - table.loc[0, "class"]  # one row of person 1000 flips
    path = write_two_header_csv(table, tmp_path / "x.csv")
    with pytest.raises(ValueError, match="differs across one person") as err:
        data.load_uci470(path, expected_rows=N_ROWS)
    assert "1000" not in str(err.value)  # id is hashed in the message
    assert data.pseudonym(1000) in str(err.value)


def test_blank_feature_fails(synthetic_table: pd.DataFrame, tmp_path: Path) -> None:
    table = synthetic_table.copy()
    table.loc[5, "feat_3"] = None
    path = write_two_header_csv(table, tmp_path / "x.csv")
    with pytest.raises(ValueError, match="blank feature"):
        data.load_uci470(path, expected_rows=N_ROWS)
