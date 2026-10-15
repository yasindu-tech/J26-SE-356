"""Tests for the raw-audio dataset loaders (VOICE-21). Synthetic files only."""

from __future__ import annotations

from pathlib import Path

import audio_datasets
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

ROWS = [("AH_001", "PwPD", 70.0, "M"), ("AH_002", "HC", 50.0, "F")]


def make_figshare(root: Path, rows: list[tuple[str, object, float, str]]) -> Path:
    """A tiny Figshare-shaped folder: metadata sheet plus one silent WAV per row."""
    for folder in audio_datasets.FIGSHARE_FOLDERS:
        (root / folder).mkdir(parents=True, exist_ok=True)
    for sample_id, label, _, _ in rows:
        folder = "PD_AH" if label == "PwPD" else "HC_AH"
        sf.write(root / folder / f"{sample_id}.wav", np.zeros(800, dtype=np.float32), 8000)
    pd.DataFrame(rows, columns=["Sample ID", "Label", "Age", "Sex"]).to_excel(
        root / "Demographics_age_sex.xlsx", index=False
    )
    return root


def test_figshare_loader_reads_labels_age_and_sex(tmp_path: Path) -> None:
    recordings = audio_datasets.load_figshare(make_figshare(tmp_path, ROWS))
    assert [(r.label, r.age, r.sex) for r in recordings] == [(1, 70.0, "M"), (0, 50.0, "F")]
    assert all(r.person.startswith("figshare:") for r in recordings)


def test_blank_label_raises_instead_of_counting_as_healthy(tmp_path: Path) -> None:
    root = make_figshare(tmp_path, ROWS)
    meta = pd.read_excel(root / "Demographics_age_sex.xlsx")
    meta.loc[1, "Label"] = None
    meta.to_excel(root / "Demographics_age_sex.xlsx", index=False)
    with pytest.raises(ValueError, match="blank or unknown labels"):
        audio_datasets.load_figshare(root)


def test_folder_that_disagrees_with_the_label_raises(tmp_path: Path) -> None:
    root = make_figshare(tmp_path, ROWS)
    (root / "PD_AH" / "AH_001.wav").rename(root / "HC_AH" / "AH_001.wav")
    with pytest.raises(ValueError, match="disagrees"):
        audio_datasets.load_figshare(root)


def test_audio_file_without_metadata_raises(tmp_path: Path) -> None:
    root = make_figshare(tmp_path, ROWS)
    sf.write(root / "HC_AH" / "AH_999.wav", np.zeros(800, dtype=np.float32), 8000)
    with pytest.raises(ValueError, match="no metadata"):
        audio_datasets.load_figshare(root)


def test_missing_dataset_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        audio_datasets.load_figshare(tmp_path / "absent")
