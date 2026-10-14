"""Tests for models/mri/src/training_data.py on made-up tables (no real data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import training_data as td


def labels_csv(tmp: Path, n: int = 12, **override) -> Path:
    df = pd.DataFrame(
        {
            "Subject ID": [f"P{i:03d}" for i in range(n)],
            "Research Group": ["PD" if i % 2 else "Control" for i in range(n)],
            "Age": [str(50 + i) for i in range(n)],
            "Sex": ["M" if i % 3 else "F" for i in range(n)],
        }
    )
    for k, v in override.items():
        df[k] = v
    path = tmp / "labels.csv"
    df.to_csv(path, index=False)
    return path


def features(ids: list[str], prefix: str, models: list[str], k: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(len(ids), k)), columns=[f"{prefix}{j}" for j in range(k)])
    df.insert(0, "subject", [td.pseudonym(i) for i in ids])
    df.insert(1, "scanner_model", models)
    df.insert(
        2, "manufacturer", ["SIEMENS" if m != "SIGNA" else "GE MEDICAL SYSTEMS" for m in models]
    )
    df.insert(3, "site_key", "")
    return df.set_index("subject")


IDS = [f"P{i:03d}" for i in range(12)]


def test_labels_are_hashed_and_encoded(tmp_path: Path) -> None:
    lab = td.load_labels(labels_csv(tmp_path))
    assert td.pseudonym("P001") in lab.index and "P001" not in lab.index
    assert lab.loc[td.pseudonym("P001"), "y"] == 1 and lab.loc[td.pseudonym("P000"), "y"] == 0
    assert set(lab["sex"]) == {0.0, 1.0}


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"Research Group": ["Prodromal"] * 12}, "unexpected Research Group"),
        ({"Research Group": [None] + ["PD"] * 11}, "no Research Group"),
        ({"Subject ID": ["P000"] * 12}, "twice"),
    ],
)
def test_bad_label_files_fail(tmp_path: Path, override, match) -> None:
    with pytest.raises(td.DataError, match=match):
        td.load_labels(labels_csv(tmp_path, **override))


def test_batch_rule_merges_small_groups_without_labels() -> None:
    models = pd.Series(
        ["TrioTim"] * 6
        + ["Prisma"] * 3
        + ["Prisma_fit"] * 3
        + ["Skyra"] * 2
        + ["SIGNA"] * 5
        + ["DISCOVERY"] * 1
        + ["Ingenia"] * 2
    )
    makers = pd.Series(["SIEMENS"] * 14 + ["GE MEDICAL SYSTEMS"] * 6 + ["Philips"] * 2)
    g = td.scanner_groups(models, makers, min_batch=5)
    counts = g.value_counts().to_dict()
    assert counts["TrioTim"] == 6 and counts["Prisma"] == 6  # Prisma_fit joins Prisma
    assert counts["SIGNA"] == 5
    assert counts["other"] == 2 + 1 + 2  # Skyra, DISCOVERY and Ingenia are each too small
    assert min(counts.values()) >= 5


def test_missing_scanner_model_fails() -> None:
    with pytest.raises(td.DataError, match="scanner model"):
        td.scanner_groups(pd.Series(["Prisma", None]), pd.Series(["S", "S"]), 1)


def test_paired_set_keeps_only_people_with_both(tmp_path: Path) -> None:
    lab = td.load_labels(labels_csv(tmp_path))
    t1 = features(IDS, "t1_", ["Prisma"] * 12)
    dti = features(IDS[:8], "dti_", ["Prisma"] * 8, seed=1)
    data = td.build("paired", lab, t1=t1, dti=dti, min_batch=2)
    assert data.n == 8 and list(data.groups) == ["t1", "dti"]
    assert data.features.shape == (8, 6)
    assert set(data.subjects) == {td.pseudonym(i) for i in IDS[:8]}
    # rows line up: the first row's T1 values belong to that person
    first = data.subjects[0]
    assert np.allclose(
        data.features.loc[0, ["t1_0", "t1_1", "t1_2"]],
        t1.loc[first, ["t1_0", "t1_1", "t1_2"]].astype(float),
    )


def test_label_and_demographics_are_not_feature_columns(tmp_path: Path) -> None:
    lab = td.load_labels(labels_csv(tmp_path))
    data = td.build("t1", lab, t1=features(IDS, "t1_", ["Prisma"] * 12), min_batch=2)
    assert not {"y", "age", "sex", "subject", "scanner_model"} & set(data.features.columns)


def test_person_without_age_is_left_out_not_filled(tmp_path: Path) -> None:
    ages = [str(50 + i) for i in range(12)]
    ages[3] = ""
    lab = td.load_labels(labels_csv(tmp_path, Age=ages))
    data = td.build("t1", lab, t1=features(IDS, "t1_", ["Prisma"] * 12), min_batch=2)
    assert data.n == 11 and td.pseudonym("P003") not in data.subjects


def test_missing_feature_values_fail(tmp_path: Path) -> None:
    lab = td.load_labels(labels_csv(tmp_path))
    t1 = features(IDS, "t1_", ["Prisma"] * 12)
    t1.iloc[0, 5] = np.nan
    with pytest.raises(td.DataError, match="missing values"):
        td.build("t1", lab, t1=t1, min_batch=2)


def test_paired_needs_both_tables(tmp_path: Path) -> None:
    lab = td.load_labels(labels_csv(tmp_path))
    with pytest.raises(td.DataError, match="dti feature table"):
        td.build("paired", lab, t1=features(IDS, "t1_", ["Prisma"] * 12))


def test_describe_has_counts_but_no_ids(tmp_path: Path) -> None:
    lab = td.load_labels(labels_csv(tmp_path))
    data = td.build("t1", lab, t1=features(IDS, "t1_", ["Prisma"] * 12), min_batch=2)
    text = td.describe(data)
    assert "12 people (6 PD, 6 control)" in text
    assert not any(td.pseudonym(i) in text or i in text for i in IDS)


def test_load_all_builds_only_what_exists(tmp_path: Path) -> None:
    fdir = tmp_path / "features"
    fdir.mkdir()
    features(IDS, "t1_", ["Prisma"] * 12).reset_index().to_csv(
        fdir / "t1_features.csv", index=False
    )
    sets = td.load_all(labels_csv(tmp_path), fdir, min_batch=2)
    assert list(sets) == ["t1"]
