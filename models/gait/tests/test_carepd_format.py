"""Tests for the CARE-PD format check (GAIT-07). Synthetic data only: nothing from data/."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pytest
from carepd_format import (
    HEAD,
    L_ANKLE,
    L_HIP,
    L_KNEE,
    R_ANKLE,
    R_HIP,
    R_KNEE,
    FormatConfig,
    cohort_violations,
    label_coverage,
    load_config,
    load_label_pickle,
    sample_keys,
    split_key,
    summarise_cohort,
    walk_violations,
)
from pydantic import ValidationError

GAIT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = GAIT_DIR / "configs" / "carepd_format.toml"
REAL_DATA = GAIT_DIR.parents[1] / "data" / "raw" / "CARE-PD"


@pytest.fixture(scope="module")
def cfg() -> FormatConfig:
    return load_config(CONFIG_PATH)


def make_walk(n_frames: int = 90, forward_m: float = 3.0) -> np.ndarray:
    """A standing h36m skeleton (metres, Y up, +x = left) walking towards +z."""
    pose = np.zeros((17, 3))
    pose[0] = (0, 0.92, 0)  # pelvis
    pose[R_HIP] = (-0.14, 0.92, 0)
    pose[R_KNEE] = (-0.14, 0.50, 0.02)
    pose[R_ANKLE] = (-0.14, 0.0, 0)
    pose[L_HIP] = (0.14, 0.92, 0)
    pose[L_KNEE] = (0.14, 0.50, 0.02)
    pose[L_ANKLE] = (0.14, 0.0, 0)
    pose[7] = (0, 1.12, 0)
    pose[8] = (0, 1.30, 0)
    pose[9] = (0, 1.40, 0)
    pose[HEAD] = (0, 1.50, 0)
    pose[11:14] = ((0.18, 1.30, 0), (0.20, 1.05, 0), (0.20, 0.85, 0))
    pose[14:17] = ((-0.18, 1.30, 0), (-0.20, 1.05, 0), (-0.20, 0.85, 0))
    walk = np.repeat(pose[None], n_frames, axis=0)
    walk[:, :, 2] += np.linspace(0, forward_m, n_frames)[:, None]
    return walk


# ----------------------------------------------------------------------------- config
def test_config_joint_order_matches_index_constants(cfg: FormatConfig) -> None:
    order = cfg.skeleton.joint_order
    assert len(order) == 17
    assert order[R_HIP] == "right_hip" and order[L_HIP] == "left_hip"
    assert order[R_KNEE] == "right_knee" and order[L_KNEE] == "left_knee"
    assert order[R_ANKLE] == "right_ankle" and order[L_ANKLE] == "left_ankle"
    assert order[HEAD] == "head"


@pytest.mark.parametrize(
    ("cohort", "fps"), [("3DGait", 30), ("T-SDU-PD", 30), ("BMCLab", 30), ("PD-GaM", 25)]
)
def test_cohort_fps(cfg: FormatConfig, cohort: str, fps: int) -> None:
    assert cfg.cohorts[cohort].fps == fps


def test_config_rejects_wrong_joint_count(tmp_path: Path) -> None:
    text = CONFIG_PATH.read_text().replace('"right_wrist",', "", 1)
    bad = tmp_path / "bad.toml"
    bad.write_text(text)
    with pytest.raises(ValidationError):
        load_config(bad)


# ----------------------------------------------------------------------------- one walk
def test_good_walk_passes(cfg: FormatConfig) -> None:
    assert walk_violations(make_walk(), cfg) == []


def _swap_left_right(w: np.ndarray) -> np.ndarray:
    out = w.copy()
    for a, b in ((1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16)):
        out[:, [a, b]] = out[:, [b, a]]
    return out


def _nan_walk(w: np.ndarray) -> np.ndarray:
    out = w.copy()
    out[3, 5, 1] = np.nan
    return out


BROKEN_WALKS = {
    "millimetres": lambda w: w * 1000,
    "centimetres": lambda w: w * 100,
    "upside_down": lambda w: w * np.array([1, -1, 1]),
    "left_right_swapped": _swap_left_right,
    "floating_above_floor": lambda w: w + np.array([0, 0.5, 0]),
    "has_nan": _nan_walk,
    "wrong_joint_count": lambda w: w[:, :16],
    "too_short": lambda w: w[:20],
}


@pytest.mark.parametrize("name", list(BROKEN_WALKS))
def test_broken_walk_is_rejected(cfg: FormatConfig, name: str) -> None:
    assert walk_violations(BROKEN_WALKS[name](make_walk()), cfg) != []


# ----------------------------------------------------------------------------- keys
@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("9__vid0091_0088", ("9", "vid0091_0088")),
        ("P002__P002_day000_001", ("P002", "P002_day000_001")),
        ("SUB01__SUB01_off_walk_1_down3", ("SUB01", "SUB01_off_walk_1")),
        ("007__007-13-000661_wid00_2", ("007", "007-13-000661_wid00_2")),
    ],
)
def test_split_key(key: str, expected: tuple[str, str]) -> None:
    assert split_key(key) == expected


@pytest.mark.parametrize("key", ["no_separator", "__walk", "subject__"])
def test_split_key_rejects_bad_keys(key: str) -> None:
    with pytest.raises(ValueError, match="subject__walk"):
        split_key(key)


def test_sample_keys_is_deterministic_and_spread_out() -> None:
    keys = [f"s{i:03d}__w" for i in range(100)]
    first = sample_keys(keys, 5)
    assert first == sample_keys(list(reversed(keys)), 5)
    assert first[0] == "s000__w" and first[-1] == "s099__w"
    assert len(set(first)) == 5


# ----------------------------------------------------------------------------- cohort
def test_good_cohort_passes(cfg: FormatConfig) -> None:
    walks = {f"s{i}__w{i}": make_walk() for i in range(20)}
    summary = summarise_cohort(walks, cfg)
    assert summary.n_subjects == 20
    assert cohort_violations(summary, cfg) == []


def test_cohort_walking_backwards_is_rejected(cfg: FormatConfig) -> None:
    walks = {f"s{i}__w{i}": make_walk(forward_m=-3.0) for i in range(20)}
    problems = cohort_violations(summarise_cohort(walks, cfg), cfg)
    assert any("+z" in p for p in problems)


def test_cohort_counts_down_copies_once(cfg: FormatConfig) -> None:
    walks = {f"SUB01__walk1_down{k}": make_walk() for k in range(5)}
    summary = summarise_cohort(walks, cfg)
    assert (summary.n_entries, summary.n_walks) == (5, 1)


# ----------------------------------------------------------------------------- labels
def _labels() -> dict:
    return {
        0: {"vid1": {"UPDRS_GAIT": 0}, "vid2": {"UPDRS_GAIT": 3}},
        1: {"vid3": {"UPDRS_GAIT": 2}, "vid4": {"UPDRS_GAIT": None}},
    }


def test_label_coverage_matches_int_subjects_and_merges_2_and_3(cfg: FormatConfig) -> None:
    cov = label_coverage(["0__vid1", "0__vid2", "1__vid3"], _labels(), cfg)
    assert cov.h36m_without_label == 0
    assert cov.labels_without_h36m == 1
    assert cov.class_counts == {"0": 1, "2": 2, "unlabelled": 1}


def test_label_coverage_flags_h36m_walk_without_label(cfg: FormatConfig) -> None:
    cov = label_coverage(["0__vid1", "0__not_in_labels"], _labels(), cfg)
    assert cov.h36m_without_label == 1


def test_label_loader_reads_numpy_and_plain_data(tmp_path: Path) -> None:
    data = {"S1": {"w1": {"pose": np.zeros((3, 72)), "UPDRS_GAIT": 1}}}
    path = tmp_path / "ok.pkl"
    path.write_bytes(pickle.dumps(data))
    assert load_label_pickle(path)["S1"]["w1"]["UPDRS_GAIT"] == 1


class _Evil:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple:
        return (os.system, (f"touch {self.marker}",))


def test_label_loader_refuses_pickles_that_run_code(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    path = tmp_path / "evil.pkl"
    path.write_bytes(pickle.dumps(_Evil(marker)))
    with pytest.raises(pickle.UnpicklingError, match="blocked"):
        load_label_pickle(path)
    assert not marker.exists()


# ----------------------------------------------------------------------------- real files
@pytest.mark.skipif(not REAL_DATA.exists(), reason="CARE-PD data is local only")
@pytest.mark.parametrize("cohort", ["3DGait", "T-SDU-PD", "BMCLab", "PD-GaM"])
def test_real_cohort_file_passes_if_present(cfg: FormatConfig, cohort: str) -> None:
    path = REAL_DATA / "h36m" / cohort / cfg.cohorts[cohort].world_file
    if not path.is_file():
        pytest.skip(f"{path.name} not downloaded")
    with np.load(path, allow_pickle=False) as npz:
        walks = {k: npz[k] for k in npz.files}
    assert cohort_violations(summarise_cohort(walks, cfg), cfg) == []
