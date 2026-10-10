"""Shared fixtures for the voice tests.

The repo has no package ``__init__`` files, so ``../src`` is put on ``sys.path``
and tests import modules by plain name (``import splitter``).

All data here is synthetic. No real UCI-470 rows are used in any test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

N_PEOPLE = 60
ROWS_PER_PERSON = 3
N_FEATURES = 30


@pytest.fixture
def synthetic_table() -> pd.DataFrame:
    """60 people x 3 rows x 30 noise features, about 75% PD, in the UCI column layout."""
    rng = np.random.default_rng(42)
    person_ids = np.arange(1000, 1000 + N_PEOPLE)
    labels = (rng.random(N_PEOPLE) < 0.75).astype(int)
    genders = rng.integers(0, 2, N_PEOPLE)

    ids = np.repeat(person_ids, ROWS_PER_PERSON)
    features = rng.normal(size=(len(ids), N_FEATURES))
    table = pd.DataFrame(features, columns=[f"feat_{i}" for i in range(N_FEATURES)])
    table.insert(0, "gender", np.repeat(genders, ROWS_PER_PERSON))
    table.insert(0, "id", ids)
    table["class"] = np.repeat(labels, ROWS_PER_PERSON)
    return table


def write_two_header_csv(table: pd.DataFrame, path: Path) -> Path:
    """Write ``table`` the way the UCI file is laid out: a group row, then column names."""
    group_row = ",".join(["", "", "Baseline Features"] + [""] * (table.shape[1] - 3))
    path.write_text(group_row + "\n" + table.to_csv(index=False))
    return path


@pytest.fixture
def synthetic_csv(synthetic_table: pd.DataFrame, tmp_path: Path) -> Path:
    return write_two_header_csv(synthetic_table, tmp_path / "pd_speech_features.csv")
