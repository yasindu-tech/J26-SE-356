"""Cohort audit for the UCI-470 voice table (task VOICE-09).

Reads ``pd_speech_features.csv`` and prints the counts the voice module quotes:
people, PD and healthy counts, rows per person, class ratio, gender by class,
how gender is coded, whether an age column exists, and columns per feature group.

Every count in models/voice/README.md must be reproducible from this script
(CLAUDE.md section 3.6). Rules it follows:

- The file may have one header row or two (a feature-group row, then the column
  names). Both are detected; the group row is used for the per-group counts.
- A blank or unexpected value is reported as missing. It is never counted as
  healthy (class 0) or as a gender.
- Subject IDs are never printed. Any person flagged in the output is shown only
  as a short hash.

Usage (from the repo root):
    python scripts/count_uci470.py
    python scripts/count_uci470.py --csv path/to/pd_speech_features.csv
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "data" / "raw" / "UCI-470" / "pd_speech_features.csv"

ID_COL = "id"
CLASS_COL = "class"
GENDER_COL = "gender"
NON_FEATURES = (ID_COL, GENDER_COL, CLASS_COL)
CLASS_LABELS = {0: "healthy", 1: "PD"}
UNGROUPED = "(no group)"


def pseudonym(subject_id: object) -> str:
    """Short hash for printed lines. Never print the raw subject ID."""
    return hashlib.sha256(str(subject_id).encode()).hexdigest()[:8]


def read_table(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load the table and map each column to its feature group.

    Detects a two-row header by checking whether the first row already holds
    the ``id`` and ``class`` column names. Returns (table, column -> group).
    """
    first_two = pd.read_csv(path, header=None, nrows=2, dtype=str, keep_default_na=False)
    first_row = [c.strip() for c in first_two.iloc[0]]
    if ID_COL in first_row and CLASS_COL in first_row:
        table = pd.read_csv(path, header=0)
        return table, {c: UNGROUPED for c in table.columns}

    second_row = [c.strip() for c in first_two.iloc[1]]
    if ID_COL not in second_row or CLASS_COL not in second_row:
        raise ValueError(f"could not find '{ID_COL}' and '{CLASS_COL}' in the first two rows")
    table = pd.read_csv(path, header=1)
    table.columns = [c.strip() for c in table.columns]

    groups: dict[str, str] = {}
    current = UNGROUPED
    for group, column in zip(first_row, table.columns, strict=True):
        if group:
            current = group
        groups[column] = current
    return table, groups


def audit(table: pd.DataFrame, groups: dict[str, str]) -> dict[str, object]:
    """Compute the cohort counts. Raises ValueError if a required column is missing."""
    missing_cols = [c for c in NON_FEATURES if c not in table.columns]
    if missing_cols:
        raise ValueError(f"required columns missing: {missing_cols}")

    # Coerce, so a blank or text value becomes NaN and is counted as missing,
    # never silently read as 0 (healthy).
    label = pd.to_numeric(table[CLASS_COL], errors="coerce")
    valid_label = label.isin(list(CLASS_LABELS))
    gender = pd.to_numeric(table[GENDER_COL], errors="coerce")

    rows_per_person = table.groupby(ID_COL).size()
    per_person = pd.DataFrame(
        {
            "label_values": label.groupby(table[ID_COL]).nunique(dropna=False),
            "gender_values": gender.groupby(table[ID_COL]).nunique(dropna=False),
            "label": label.groupby(table[ID_COL]).first(),
            "gender": gender.groupby(table[ID_COL]).first(),
        }
    )
    inconsistent = per_person[(per_person.label_values > 1) | (per_person.gender_values > 1)]
    people = per_person.drop(index=inconsistent.index)

    n_pd = int((people.label == 1).sum())
    n_hc = int((people.label == 0).sum())
    gender_by_class = pd.crosstab(
        people.gender.fillna(-1).astype(int).rename("gender code (-1 = missing)"),
        people.label.map(CLASS_LABELS).fillna("missing").rename("class"),
    )

    feature_cols = [c for c in table.columns if c not in NON_FEATURES]
    group_counts = pd.Series([groups.get(c, UNGROUPED) for c in feature_cols]).value_counts(
        sort=False
    )

    return {
        "rows": len(table),
        "columns": table.shape[1],
        "feature_columns": len(feature_cols),
        "people": int(rows_per_person.size),
        "rows_per_person_min": int(rows_per_person.min()),
        "rows_per_person_max": int(rows_per_person.max()),
        "people_pd": n_pd,
        "people_healthy": n_hc,
        "people_label_missing": len(people) - n_pd - n_hc,
        "rows_label_missing_or_invalid": int((~valid_label).sum()),
        "pd_to_healthy_ratio": round(n_pd / n_hc, 2) if n_hc else float("nan"),
        "gender_codes": sorted(int(v) for v in gender.dropna().unique()),
        "rows_gender_missing": int(gender.isna().sum()),
        "gender_by_class": gender_by_class,
        "age_columns": [c for c in table.columns if "age" in c.lower()],
        "blank_cells": int(table.isna().sum().sum()),
        "group_counts": group_counts,
        "inconsistent_people": [pseudonym(i) for i in inconsistent.index],
    }


def report(result: dict[str, object]) -> str:
    lines = [
        f"rows: {result['rows']}   columns: {result['columns']}   "
        f"feature columns (excluding {', '.join(NON_FEATURES)}): {result['feature_columns']}",
        f"people: {result['people']}   rows per person: min {result['rows_per_person_min']}, "
        f"max {result['rows_per_person_max']}",
        f"PD: {result['people_pd']}   healthy: {result['people_healthy']}   "
        f"label missing: {result['people_label_missing']}   "
        f"PD:healthy ratio: {result['pd_to_healthy_ratio']}",
        f"rows with missing/invalid class: {result['rows_label_missing_or_invalid']}",
        f"blank cells anywhere: {result['blank_cells']}",
        f"gender codes present: {result['gender_codes']}   "
        f"rows with missing gender: {result['rows_gender_missing']}",
        "people by gender code and class:",
        str(result["gender_by_class"]),
        f"age columns: {result['age_columns'] or 'none'}",
        "feature columns per group (from the file's group header row):",
        str(result["group_counts"]),
    ]
    if result["inconsistent_people"]:
        lines.append(
            "PEOPLE WITH CONFLICTING class/gender ACROSS ROWS (hashed, excluded from counts): "
            + ", ".join(result["inconsistent_people"])  # type: ignore[arg-type]
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = ap.parse_args(argv)

    if not args.csv.exists():
        print(f"error: {args.csv} not found. See models/voice/README.md for where to put it.")
        return 2
    table, groups = read_table(args.csv)
    result = audit(table, groups)
    print(report(result))
    problems = result["inconsistent_people"] or result["rows_label_missing_or_invalid"]
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
