"""Load the UCI-470 voice feature table (task VOICE-10).

The file has one row per recording and three recordings per person. This loader
returns the feature matrix, the label, the person id and gender as separate
objects, so nothing downstream can accidentally use ``id``, ``class`` or
``gender`` as a model input.

Checks (each raises ValueError, never warns):
- ``id``, ``gender`` and ``class`` columns exist (one or two header rows accepted);
- the row count is what we expect (756 for the real file);
- every label is 0 or 1 — a blank is an error, never read as healthy;
- class and gender are constant within each person;
- no blank feature values.

Raw person ids stay in memory only. Use ``pseudonym()`` for anything printed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV = REPO_ROOT / "data" / "raw" / "UCI-470" / "pd_speech_features.csv"

ID_COL = "id"
GENDER_COL = "gender"
CLASS_COL = "class"
NON_FEATURES = (ID_COL, GENDER_COL, CLASS_COL)
EXPECTED_ROWS = 756


class VoiceData(NamedTuple):
    X: pd.DataFrame  # features only, one row per recording
    y: np.ndarray  # 1 = PD, 0 = healthy, per recording
    groups: np.ndarray  # person id per recording (never print raw)
    gender: np.ndarray  # gender code per recording (not a model input, D7)


def pseudonym(person_id: object) -> str:
    """Short hash for printed lines. Never print the raw person id."""
    return hashlib.sha256(str(person_id).encode()).hexdigest()[:8]


def _read_csv(path: Path) -> pd.DataFrame:
    """Read with one or two header rows, detected from where ``id`` and ``class`` are."""
    head = pd.read_csv(path, header=None, nrows=2, dtype=str, keep_default_na=False)
    for header_row in (0, 1):
        names = [c.strip() for c in head.iloc[header_row]]
        if ID_COL in names and CLASS_COL in names:
            table = pd.read_csv(path, header=header_row)
            table.columns = [c.strip() for c in table.columns]
            return table
    raise ValueError(f"could not find '{ID_COL}' and '{CLASS_COL}' in the first two rows")


def load_uci470(path: Path = DEFAULT_CSV, expected_rows: int | None = EXPECTED_ROWS) -> VoiceData:
    """Load and validate the table. Pass ``expected_rows=None`` to skip the row-count check."""
    table = _read_csv(Path(path))

    missing = [c for c in NON_FEATURES if c not in table.columns]
    if missing:
        raise ValueError(f"required columns missing: {missing}")
    if expected_rows is not None and len(table) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, found {len(table)}")

    label = pd.to_numeric(table[CLASS_COL], errors="coerce")
    if not label.isin([0, 1]).all():
        n_bad = int((~label.isin([0, 1])).sum())
        raise ValueError(f"{n_bad} rows have a blank or invalid class (must be 0 or 1)")
    if table[GENDER_COL].isna().any():
        raise ValueError("blank gender values found")

    per_person = table.groupby(ID_COL)[[CLASS_COL, GENDER_COL]].nunique()
    conflicting = per_person[(per_person > 1).any(axis=1)].index
    if len(conflicting):
        hashed = ", ".join(pseudonym(i) for i in conflicting)
        raise ValueError(f"class or gender differs across one person's rows: {hashed}")

    X = table.drop(columns=list(NON_FEATURES))
    if X.isna().any().any():
        raise ValueError(f"{int(X.isna().sum().sum())} blank feature values found")

    return VoiceData(
        X=X,
        y=label.to_numpy(dtype=int),
        groups=table[ID_COL].to_numpy(),
        gender=table[GENDER_COL].to_numpy(),
    )
