"""Assemble the MRI training data: features, labels and scanner groups, per person.

Inputs (all local, under data/, never committed):

    data/processed/features/t1_features.csv    from t1_features.py  (pseudonymised)
    data/processed/features/dti_features.csv   from dti_features.py (pseudonymised)
    data/interim/pp1_mri_subset.csv            subject list with Research Group, Age, Sex

The label file holds raw PPMI IDs; they are hashed on read with the same pseudonym as the
feature tables and never written anywhere.

What comes out is a ``ModelData`` for one of three sets:

    "t1"      everyone with T1 features                    (T1 model, about 59 people)
    "dti"     everyone with DTI features                   (DTI model)
    "paired"  people with both, T1 and DTI columns side by side

A person missing a modality is left out of that set; nothing is filled in. The label (PD = 1,
control = 0) is kept apart from the features, and age and sex are kept apart too: they are
only used by the age + sex baseline and as ComBat covariates.

Scanner groups (the ComBat batch). ComBat needs at least two people per scanner in every
training fold and cannot correct a scanner absent from training. With 9 scanner models and
groups as small as 2, the batch is built by one fixed rule, before any split and without
looking at labels:

    1. scanner model, with Prisma and Prisma_fit treated as one model (same Siemens family);
    2. any model with fewer than ``min_batch`` people (default 5) joins the others of its
       manufacturer; if that group is still too small, it joins a single "other" group.

The rule is applied per modality (the DTI scanner can differ from the T1 one), and per set.

Command line (prints counts only, no IDs):

    python models/mri/src/training_data.py
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
FEATURES = REPO / "data" / "processed" / "features"
DEFAULT_LABELS = REPO / "data" / "interim" / "pp1_mri_subset.csv"
META_COLUMNS = ("subject", "scanner_model", "manufacturer", "site_key")
LABELS = {"PD": 1, "Control": 0}
SEX = {"M": 1.0, "F": 0.0}
SAME_MODEL = {"Prisma_fit": "Prisma"}  # one Siemens family, two names in the headers
MIN_BATCH = 5
SETS = ("t1", "dti", "paired")


class DataError(ValueError):
    """The inputs cannot be assembled into a valid training table."""


def pseudonym(subject_id: str) -> str:
    """Same hash as the feature tables and progress events."""
    return hashlib.sha256(subject_id.encode()).hexdigest()[:8]


# ------------------------------------------------------------------ labels


def load_labels(path: Path) -> pd.DataFrame:
    """Pseudonym -> y (PD 1, control 0), age, sex (male 1). Raw IDs never leave this function."""
    df = pd.read_csv(path, dtype=str)
    need = {"Subject ID", "Research Group", "Age", "Sex"}
    if not need <= set(df.columns):
        raise DataError(f"label file is missing columns: {sorted(need - set(df.columns))}")
    unknown = sorted(set(df["Research Group"].dropna()) - set(LABELS))
    if unknown:
        raise DataError(f"unexpected Research Group values: {unknown} (only PD and Control)")
    if df["Research Group"].isna().any():
        raise DataError("a person has no Research Group; blanks are never read as control")
    out = pd.DataFrame(
        {
            "subject": df["Subject ID"].map(pseudonym),
            "y": df["Research Group"].map(LABELS).astype(int),
            "age": pd.to_numeric(df["Age"], errors="coerce"),
            "sex": df["Sex"].map(SEX),
        }
    )
    if out["subject"].duplicated().any():
        raise DataError("a person appears twice in the label file")
    return out.set_index("subject")


# ------------------------------------------------------------------ scanner groups


def scanner_groups(models: pd.Series, manufacturers: pd.Series, min_batch: int) -> pd.Series:
    """The fixed batch rule above. Same index as the inputs."""
    model = models.fillna("").replace(SAME_MODEL)
    if (model == "").any():
        raise DataError("a person has no scanner model; the batch cannot be guessed")
    maker = manufacturers.fillna("").str.upper().str.split().str[0].replace({"": "UNKNOWN"})
    group = model.copy()
    small = group.map(group.value_counts()) < min_batch
    group[small] = maker[small] + "_other"
    still_small = group.map(group.value_counts()) < min_batch
    group[still_small] = "other"
    return group


# ------------------------------------------------------------------ model data


@dataclass
class ModelData:
    name: str
    subjects: np.ndarray  # pseudonyms
    y: np.ndarray  # 1 = PD, 0 = control
    age: np.ndarray
    sex: np.ndarray
    features: pd.DataFrame  # feature columns only, same row order
    groups: dict[str, list[str]]  # "t1" / "dti" -> its feature columns
    batches: dict[str, np.ndarray]  # "t1" / "dti" -> scanner group per person

    @property
    def n(self) -> int:
        return len(self.y)

    def covariates(self) -> np.ndarray:
        return np.column_stack([self.age, self.sex])


def _features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DataError(f"feature table not found: {path.name}")
    df = pd.read_csv(path, dtype={c: str for c in META_COLUMNS}).set_index("subject")
    if df.index.duplicated().any():
        raise DataError(f"{path.name} has a person twice")
    return df


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLUMNS]


def build(
    which: str,
    labels: pd.DataFrame,
    t1: pd.DataFrame | None = None,
    dti: pd.DataFrame | None = None,
    min_batch: int = MIN_BATCH,
) -> ModelData:
    """One training set. People without a label, age or sex are left out, never filled."""
    if which not in SETS:
        raise DataError(f"unknown set {which!r}; use one of {SETS}")
    parts = {"t1": t1, "dti": dti}
    wanted = ["t1", "dti"] if which == "paired" else [which]
    for m in wanted:
        if parts[m] is None:
            raise DataError(f"set {which!r} needs the {m} feature table")
    people = set(labels.index)
    for m in wanted:
        people &= set(parts[m].index)
    complete = labels.loc[sorted(people)].dropna(subset=["age", "sex"])
    order = complete.index.to_numpy()
    if len(order) == 0:
        raise DataError(f"set {which!r} has nobody with features and a label")

    cols, groups, batches = [], {}, {}
    for m in wanted:
        tab = parts[m].loc[order]
        fcols = _feature_columns(tab)
        if tab[fcols].isna().any().any():
            raise DataError(f"{m} features have missing values")
        cols.append(tab[fcols])
        groups[m] = fcols
        batches[m] = scanner_groups(tab["scanner_model"], tab["manufacturer"], min_batch).to_numpy()
    features = pd.concat(cols, axis=1)
    if features.columns.duplicated().any():
        raise DataError("T1 and DTI tables share a column name")
    return ModelData(
        name=which,
        subjects=order,
        y=complete["y"].to_numpy(),
        age=complete["age"].to_numpy(dtype=float),
        sex=complete["sex"].to_numpy(dtype=float),
        features=features.reset_index(drop=True),
        groups=groups,
        batches=batches,
    )


def load_all(
    labels_path: Path = DEFAULT_LABELS, features_dir: Path = FEATURES, min_batch: int = MIN_BATCH
) -> dict[str, ModelData]:
    """Every set whose inputs exist."""
    labels = load_labels(labels_path)
    t1_path, dti_path = features_dir / "t1_features.csv", features_dir / "dti_features.csv"
    t1 = _features(t1_path) if t1_path.exists() else None
    dti = _features(dti_path) if dti_path.exists() else None
    out = {}
    if t1 is not None:
        out["t1"] = build("t1", labels, t1=t1, min_batch=min_batch)
    if dti is not None:
        out["dti"] = build("dti", labels, dti=dti, min_batch=min_batch)
    if t1 is not None and dti is not None:
        out["paired"] = build("paired", labels, t1=t1, dti=dti, min_batch=min_batch)
    return out


def describe(data: ModelData) -> str:
    """Counts only: people, PD/control, features, scanner groups."""
    lines = [
        f"{data.name}: {data.n} people ({int(data.y.sum())} PD, {int((data.y == 0).sum())} control),"
        f" {data.features.shape[1]} features"
    ]
    for m, b in data.batches.items():
        counts = pd.Series(b).value_counts().to_dict()
        lines.append(f"  {m} scanner groups: {counts}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    try:
        sets = load_all()
    except DataError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for data in sets.values():
        print(describe(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
