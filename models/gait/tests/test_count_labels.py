"""Tests for the label counts (GAIT-08, Check B). Synthetic data only: nothing from data/."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from carepd_format import FormatConfig, load_config
from count_labels import (
    MIN_SUBJECTS_TOP_CLASS,
    ClassCount,
    choose_classes,
    count_cohort,
    main,
    pool,
    usable_walks,
    write_csv,
)

GAIT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = GAIT_DIR / "configs" / "carepd_format.toml"
REAL_DATA = GAIT_DIR.parents[1] / "data" / "raw" / "CARE-PD"

Labels = dict[Any, dict[str, dict[str, Any]]]


@pytest.fixture(scope="module")
def cfg() -> FormatConfig:
    return load_config(CONFIG_PATH)


def labels_of(spec: dict[str, dict[str, int | None]]) -> Labels:
    """``{subject: {walk: score}}`` -> the structure of a cohort .pkl (medication is a decoy)."""
    return {
        s: {w: {"UPDRS_GAIT": score, "medication": "ON"} for w, score in walks.items()}
        for s, walks in spec.items()
    }


def all_usable(labels: Labels) -> set[tuple[str, str]]:
    return {(str(s), w) for s, walks in labels.items() for w in walks}


def test_counts_subjects_and_walks_per_score(cfg: FormatConfig) -> None:
    labels = labels_of({"a": {"w1": 0, "w2": 0, "w3": 1}, "b": {"w1": 1, "w2": 3}})
    c = count_cohort("X", labels, all_usable(labels), cfg)
    assert c.raw[0] == ClassCount(subjects=1, walks=2)
    assert c.raw[1] == ClassCount(subjects=2, walks=2)  # two different subjects
    assert c.raw[3] == ClassCount(subjects=1, walks=1)
    assert c.n_usable_walks == 5
    assert c.mixed_subjects == 2  # both subjects have walks in more than one score


def test_scores_2_and_3_merge_and_a_subject_counts_once(cfg: FormatConfig) -> None:
    # Subject "a" has a score-2 walk and a score-3 walk: one subject in the merged class.
    labels = labels_of({"a": {"w1": 2, "w2": 3}, "b": {"w1": 3}})
    c = count_cohort("X", labels, all_usable(labels), cfg)
    assert c.merged[2] == ClassCount(subjects=2, walks=3)
    assert set(c.merged) == {2}


def test_down_copies_count_once() -> None:
    keys = [f"s1__walk1_down{k}" for k in range(5)] + ["s1__walk2_down0", "s2__walk1"]
    assert usable_walks(keys) == {("s1", "walk1"), ("s1", "walk2"), ("s2", "walk1")}


def test_walks_without_a_skeleton_are_not_counted(cfg: FormatConfig) -> None:
    labels = labels_of({"a": {"w1": 1, "w2": 1}})
    c = count_cohort("X", labels, {("a", "w1")}, cfg)  # w2 was dropped by CARE-PD (too short)
    assert c.n_labelled_walks == 2
    assert c.n_usable_walks == 1
    assert c.raw[1].walks == 1


def test_missing_label_is_skipped_not_counted_as_zero(cfg: FormatConfig) -> None:
    labels = labels_of({"a": {"w1": None, "w2": 1}})
    c = count_cohort("X", labels, all_usable(labels), cfg)
    assert 0 not in c.raw
    assert c.n_labelled_walks == 1


def test_integer_subject_ids_match_string_keys(cfg: FormatConfig) -> None:
    # 3DGait subject keys are ints in the .pkl but strings in the h36m keys.
    labels: Labels = {7: {"w1": {"UPDRS_GAIT": 1}}}
    c = count_cohort("3DGait", labels, usable_walks(["7__w1"]), cfg)
    assert c.n_usable_walks == 1


def test_unknown_score_raises(cfg: FormatConfig) -> None:
    labels = labels_of({"a": {"w1": 4}})
    with pytest.raises(ValueError, match="not in the class map"):
        count_cohort("X", labels, all_usable(labels), cfg)


def test_pooling_adds_cohorts(cfg: FormatConfig) -> None:
    a = labels_of({"a": {"w1": 2}})
    b = labels_of({"a": {"w1": 3, "w2": 3}})
    ca = count_cohort("A", a, all_usable(a), cfg)
    cb = count_cohort("B", b, all_usable(b), cfg)
    assert pool([ca, cb], "merged")[2] == ClassCount(subjects=2, walks=3)
    assert pool([ca, cb], "raw")[3] == ClassCount(subjects=1, walks=2)


def test_three_classes_kept_when_top_class_is_big_enough() -> None:
    pooled = {
        0: ClassCount(30, 100),
        1: ClassCount(30, 100),
        2: ClassCount(MIN_SUBJECTS_TOP_CLASS, 50),
    }
    assert choose_classes(pooled) == "three-class"


def test_falls_back_to_binary_when_top_class_is_too_small() -> None:
    pooled = {
        0: ClassCount(30, 100),
        1: ClassCount(30, 100),
        2: ClassCount(MIN_SUBJECTS_TOP_CLASS - 1, 50),
    }
    assert choose_classes(pooled) == "binary-0-vs-1plus"
    assert choose_classes({0: ClassCount(5, 5)}) == "binary-0-vs-1plus"  # no top class at all


def test_csv_has_one_row_per_cohort_scheme_class(cfg: FormatConfig, tmp_path: Path) -> None:
    labels = labels_of({"a": {"w1": 0, "w2": 3}})
    c = count_cohort("X", labels, all_usable(labels), cfg)
    out = tmp_path / "sub" / "counts.csv"
    write_csv(out, [c])
    rows = list(csv.DictReader(out.open()))
    assert {"cohort", "scheme", "class", "subjects", "walks"} == set(rows[0])
    assert {(r["cohort"], r["scheme"], r["class"]) for r in rows} >= {
        ("X", "raw", "3"),
        ("X", "merged", "2-3"),
        ("ALL", "merged", "0"),
    }


def test_missing_files_exit_with_code_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--data-dir", str(tmp_path)]) == 2
    assert "MISSING" in capsys.readouterr().out


def test_cli_end_to_end_on_a_tiny_dataset(
    cfg: FormatConfig, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import pickle

    for name, cohort in cfg.cohorts.items():
        (tmp_path / "h36m" / name).mkdir(parents=True)
        np.savez(tmp_path / "h36m" / name / cohort.world_file, **{"s1__w1": np.zeros((30, 17, 3))})
        with (tmp_path / cohort.label_file).open("wb") as f:
            pickle.dump(labels_of({"s1": {"w1": 1}}), f)
    assert main(["--data-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Decision: binary-0-vs-1plus" in out  # no class 2-3 subjects in this toy data
    assert "s1" not in out.replace("subjects", "")  # no subject IDs printed


@pytest.mark.skipif(not REAL_DATA.is_dir(), reason="CARE-PD files not downloaded")
def test_real_data_keeps_three_classes(capsys: pytest.CaptureFixture[str]) -> None:
    if not (REAL_DATA / "h36m" / "3DGait").is_dir():
        pytest.skip("h36m files not present")
    assert main([]) == 0
    assert "Decision: three-class" in capsys.readouterr().out
