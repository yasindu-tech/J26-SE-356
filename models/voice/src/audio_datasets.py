"""Raw-audio voice datasets: which file belongs to which person and label (VOICE-21).

Each loader returns one ``Recording`` per sustained-vowel audio file. Labels
come from the dataset's own metadata and are cross-checked against its folder
layout; a blank, unknown or conflicting label raises, never becomes "healthy"
(CLAUDE.md section 3.6).

Age and sex are kept for baselines and fairness checks only. They are never
model inputs.

Datasets (all under the gitignored data/raw/):
- ``figshare``: Voice Samples for PD and Healthy Controls (CC BY 4.0), telephone
  "aaah" at 8 kHz, one recording per person. Healthy people are much younger
  than PD people in this set, so it always needs an age-only baseline.

NeuroVoz and Italian PVS loaders are added once their files are on disk; their
folder layouts are not guessed in advance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW = REPO_ROOT / "data" / "raw"
FIGSHARE_DIR = RAW / "Figshare"

FIGSHARE_LABELS = {"PwPD": 1, "HC": 0}
FIGSHARE_FOLDERS = {"PD_AH": 1, "HC_AH": 0}


@dataclass(frozen=True)
class Recording:
    path: Path
    person: str  # dataset-scoped id, e.g. "figshare:<sample id>"; never print raw
    label: int  # 1 = PD, 0 = healthy
    dataset: str
    age: float | None
    sex: str | None  # "F" / "M" as given by the dataset


def load_figshare(root: Path = FIGSHARE_DIR) -> list[Recording]:
    """One recording per person, labelled from the metadata and checked against the folder."""
    meta_path = root / "Demographics_age_sex.xlsx"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Figshare metadata not found at {meta_path}")
    meta = pd.read_excel(meta_path)
    missing_cols = {"Sample ID", "Label", "Age", "Sex"} - set(meta.columns)
    if missing_cols:
        raise ValueError(f"Figshare metadata is missing columns: {sorted(missing_cols)}")
    if meta["Sample ID"].duplicated().any():
        raise ValueError("Figshare metadata repeats a Sample ID")

    unknown = sorted(set(meta["Label"].dropna()) - set(FIGSHARE_LABELS))
    if meta["Label"].isna().any() or unknown:
        raise ValueError(f"Figshare metadata has blank or unknown labels: {unknown}")

    wavs = {p.stem: p for folder in FIGSHARE_FOLDERS for p in (root / folder).glob("*.wav")}
    recordings = []
    for row in meta.to_dict("records"):
        sample_id = str(row["Sample ID"])
        path = wavs.get(sample_id)
        if path is None:
            raise ValueError("a Figshare metadata row has no matching audio file")
        label = FIGSHARE_LABELS[row["Label"]]
        if FIGSHARE_FOLDERS[path.parent.name] != label:
            raise ValueError("a Figshare file's folder disagrees with its metadata label")
        recordings.append(
            Recording(
                path=path,
                person=f"figshare:{sample_id}",
                label=label,
                dataset="figshare",
                age=None if pd.isna(row["Age"]) else float(row["Age"]),
                sex=None if pd.isna(row["Sex"]) else str(row["Sex"]),
            )
        )
    if len(recordings) != len(wavs):
        raise ValueError("some Figshare audio files have no metadata row")
    return recordings


LOADERS = {"figshare": load_figshare}


def load_all(datasets: list[str] | None = None) -> list[Recording]:
    """Recordings from every requested dataset whose files are on disk."""
    names = datasets or list(LOADERS)
    return [rec for name in names for rec in LOADERS[name]()]
