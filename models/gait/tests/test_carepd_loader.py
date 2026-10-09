"""Tests for the CARE-PD loader (GAIT-11). Synthetic data only: nothing from data/ is committed."""

from __future__ import annotations

import dataclasses
import pickle
from pathlib import Path
from typing import Any

import carepd_loader as loader
import numpy as np
import pytest
from carepd_format import FormatConfig, load_config
from carepd_loader import (
    Walk,
    class_counts,
    copy_index,
    load_all,
    load_cohort,
    main,
    pick_walk_for_plot,
    plot_walk,
)

GAIT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = GAIT_DIR / "configs" / "carepd_format.toml"
REAL_DATA = GAIT_DIR.parents[1] / "data" / "raw" / "CARE-PD"

Spec = dict[Any, dict[str, int | None]]  # {subject: {walk: score}}


@pytest.fixture(scope="module")
def cfg() -> FormatConfig:
    return load_config(CONFIG_PATH)


def skeleton(n: int = 40, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, 17, 3))


def write_cohort(
    root: Path,
    cfg: FormatConfig,
    name: str,
    spec: Spec,
    *,
    copies: int = 0,
    fps: int | None = None,
    skeleton_for: Any = None,
) -> dict[str, np.ndarray]:
    """Write a world file and a label file for one cohort. Returns {npz key: array}."""
    cohort = cfg.cohorts[name]
    arrays: dict[str, np.ndarray] = {}
    labels: dict[Any, dict[str, dict[str, Any]]] = {}
    seed = 0
    for subject, walks in spec.items():
        labels[subject] = {}
        for walk, score in walks.items():
            labels[subject][walk] = {
                "UPDRS_GAIT": score,
                "medication": "ON",
                "other": "decoy",
                "fps": fps if fps is not None else (cohort.label_fps or cohort.fps),
            }
            if skeleton_for is not None and (str(subject), walk) not in skeleton_for:
                continue
            names = [f"{walk}_down{i}" for i in range(copies)] if copies else [walk]
            for n in names:
                seed += 1
                arrays[f"{subject}__{n}"] = skeleton(40 + seed, seed)
    (root / "h36m" / name).mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = dict(arrays)
    np.savez(root / "h36m" / name / cohort.world_file, **payload)
    with (root / cohort.label_file).open("wb") as f:
        pickle.dump(labels, f)
    return arrays


# --------------------------------------------------------------------------- the join
def test_walks_are_joined_to_cohort_subject_label_and_fps(
    tmp_path: Path, cfg: FormatConfig
) -> None:
    arrays = write_cohort(tmp_path, cfg, "T-SDU-PD", {"a": {"w1": 0, "w2": 3}, "b": {"w1": 1}})
    walks, report = load_cohort("T-SDU-PD", tmp_path, cfg)
    by = {(w.subject, w.walk): w for w in walks}
    assert set(by) == {("a", "w1"), ("a", "w2"), ("b", "w1")}
    w = by[("a", "w2")]
    assert (w.cohort, w.fps, w.score, w.severity, w.copy) == ("T-SDU-PD", 30, 3, 2, None)
    np.testing.assert_array_equal(w.joints, arrays["a__w2"])
    assert w.group == "T-SDU-PD:a"
    assert report.n_walks == 3 and report.n_subjects == 2
    assert report.matches_gait08 is True


def test_pd_gam_uses_25_fps(tmp_path: Path, cfg: FormatConfig) -> None:
    write_cohort(tmp_path, cfg, "PD-GaM", {"a": {"w1": 1}})
    walks, _ = load_cohort("PD-GaM", tmp_path, cfg)
    assert walks[0].fps == 25


def test_integer_subject_keys_join_to_the_world_file(tmp_path: Path, cfg: FormatConfig) -> None:
    write_cohort(tmp_path, cfg, "3DGait", {12: {"w1": 2}, 7: {"w1": 0}})  # 3DGait uses int IDs
    walks, report = load_cohort("3DGait", tmp_path, cfg)
    assert sorted(w.subject for w in walks) == ["12", "7"]
    assert report.matches_gait08 is True


# --------------------------------------------------------------------------- BMCLab copies
def test_only_the_first_copy_is_loaded_by_default(tmp_path: Path, cfg: FormatConfig) -> None:
    arrays = write_cohort(tmp_path, cfg, "BMCLab", {"a": {"w1": 1, "w2": 2}}, copies=5)
    walks, report = load_cohort("BMCLab", tmp_path, cfg)
    assert len(walks) == 2 and all(w.copy == 0 for w in walks)
    np.testing.assert_array_equal(walks[0].joints, arrays["a__w1_down0"])
    assert report.copies_dropped == 8  # 2 walks x 4 extra copies
    assert report.matches_gait08 is True


def test_all_copies_loads_five_per_walk(tmp_path: Path, cfg: FormatConfig) -> None:
    write_cohort(tmp_path, cfg, "BMCLab", {"a": {"w1": 1}}, copies=5)
    walks, report = load_cohort("BMCLab", tmp_path, cfg, all_copies=True)
    assert sorted(w.copy for w in walks if w.copy is not None) == [0, 1, 2, 3, 4]
    assert {w.group for w in walks} == {"BMCLab:a"}  # still one subject
    assert report.copies_dropped == 0 and report.matches_gait08 is None


def test_copy_index() -> None:
    assert copy_index("s__w_down3") == 3
    assert copy_index("s__w") is None


# --------------------------------------------------------------------------- what is skipped
def test_label_without_skeleton_and_missing_score_are_counted(
    tmp_path: Path, cfg: FormatConfig
) -> None:
    spec: Spec = {"a": {"w1": 1, "short": 2, "noscore": None}}
    write_cohort(tmp_path, cfg, "3DGait", spec, skeleton_for={("a", "w1"), ("a", "noscore")})
    walks, report = load_cohort("3DGait", tmp_path, cfg)
    assert [w.walk for w in walks] == ["w1"]
    assert report.labelled_without_skeleton == 1  # "short" has a label but no skeleton
    assert report.unlabelled == 1  # "noscore" has a skeleton but no score
    assert report.matches_gait08 is True


def test_unknown_score_raises(tmp_path: Path, cfg: FormatConfig) -> None:
    write_cohort(tmp_path, cfg, "3DGait", {"a": {"w1": 7}})
    with pytest.raises(ValueError, match="class map"):
        load_cohort("3DGait", tmp_path, cfg)


def test_bad_skeleton_shape_raises(tmp_path: Path, cfg: FormatConfig) -> None:
    cohort = cfg.cohorts["3DGait"]
    (tmp_path / "h36m" / "3DGait").mkdir(parents=True)
    np.savez(tmp_path / "h36m" / "3DGait" / cohort.world_file, a__w1=np.zeros((40, 25, 3)))
    with (tmp_path / cohort.label_file).open("wb") as f:
        pickle.dump({"a": {"w1": {"UPDRS_GAIT": 1, "fps": 30}}}, f)
    with pytest.raises(ValueError, match="17, 3"):
        load_cohort("3DGait", tmp_path, cfg)


def test_missing_file_raises(tmp_path: Path, cfg: FormatConfig) -> None:
    with pytest.raises(FileNotFoundError):
        load_cohort("3DGait", tmp_path, cfg)


def test_fps_that_differs_from_the_config_is_counted_not_trusted(
    tmp_path: Path, cfg: FormatConfig
) -> None:
    write_cohort(tmp_path, cfg, "3DGait", {"a": {"w1": 1}}, fps=60)
    walks, report = load_cohort("3DGait", tmp_path, cfg)
    assert report.fps_mismatch == 1 and walks[0].fps == 30  # the config decides


def test_bmclab_label_fps_150_is_expected_not_a_mismatch(tmp_path: Path, cfg: FormatConfig) -> None:
    assert cfg.cohorts["BMCLab"].label_fps == 150 and cfg.cohorts["BMCLab"].fps == 30
    write_cohort(tmp_path, cfg, "BMCLab", {"a": {"w1": 1}}, copies=5)  # label file says 150
    walks, report = load_cohort("BMCLab", tmp_path, cfg)
    assert report.fps_mismatch == 0 and walks[0].fps == 30  # time uses the skeleton's 30 fps


def test_bmclab_label_fps_other_than_150_is_flagged(tmp_path: Path, cfg: FormatConfig) -> None:
    write_cohort(tmp_path, cfg, "BMCLab", {"a": {"w1": 1}}, copies=5, fps=30)
    _, report = load_cohort("BMCLab", tmp_path, cfg)
    assert report.fps_mismatch == 1


# --------------------------------------------------------------------------- medication stays out
class Recorder(dict[str, Any]):
    """A label-fields dict that records every way the loader reads it."""

    def __init__(self, data: dict[str, Any], seen: set[str]) -> None:
        super().__init__(data)
        self._seen = seen

    def get(self, key: str, default: Any = None) -> Any:
        self._seen.add(key)
        return super().get(key, default)

    def __getitem__(self, key: str) -> Any:
        self._seen.add(key)
        return super().__getitem__(key)

    def items(self) -> Any:
        self._seen.add("*items*")
        return super().items()

    def keys(self) -> Any:
        self._seen.add("*keys*")
        return super().keys()

    def values(self) -> Any:
        self._seen.add("*values*")
        return super().values()

    def __iter__(self) -> Any:
        self._seen.add("*iter*")
        return super().__iter__()


class FakeCounts:
    """Stands in for the GAIT-08 cross-check, which reads label fields on its own."""

    raw: dict[int, Any] = {}  # noqa: RUF012
    merged: dict[int, Any] = {}  # noqa: RUF012


def test_medication_and_other_fields_are_never_read(
    tmp_path: Path, cfg: FormatConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_cohort(tmp_path, cfg, "BMCLab", {"a": {"w1": 1}, "b": {"w1": 2}}, copies=5)
    seen: set[str] = set()
    real = loader.load_label_pickle

    def wrapped(path: Path) -> Any:
        data = real(path)
        return {s: {w: Recorder(f, seen) for w, f in ws.items()} for s, ws in data.items()}

    monkeypatch.setattr(loader, "load_label_pickle", wrapped)
    monkeypatch.setattr(loader, "count_cohort", lambda *a, **k: FakeCounts())
    walks, _ = load_cohort("BMCLab", tmp_path, cfg)
    assert len(walks) == 2
    assert seen == {"UPDRS_GAIT", "fps"}  # medication and other were never touched


def test_walk_has_no_medication_field() -> None:
    names = {f.name for f in dataclasses.fields(Walk)}
    assert not any("medic" in n or "other" in n or "dose" in n for n in names)
    assert {"score", "severity"} <= names  # the label is carried as a label


# --------------------------------------------------------------------------- counts and plot
def test_class_counts_match_scores(tmp_path: Path, cfg: FormatConfig) -> None:
    write_cohort(tmp_path, cfg, "3DGait", {"a": {"w1": 0, "w2": 2}, "b": {"w1": 3}})
    walks, _ = load_cohort("3DGait", tmp_path, cfg)
    merged = class_counts(walks)
    assert (merged[2].subjects, merged[2].walks) == (2, 2)  # scores 2 and 3 merged
    assert class_counts(walks, raw=True)[3].walks == 1


def test_plot_picks_a_median_length_walk_and_writes_a_png(
    tmp_path: Path, cfg: FormatConfig
) -> None:
    write_cohort(tmp_path, cfg, "3DGait", {"a": {"w1": 0, "w2": 1, "w3": 2}})
    walks, _ = load_cohort("3DGait", tmp_path, cfg)
    pick = pick_walk_for_plot(walks)
    assert pick.n_frames == sorted(w.n_frames for w in walks)[1]
    out = tmp_path / "plots" / "walk.png"
    plot_walk(pick, cfg, out)
    assert out.is_file() and out.stat().st_size > 1000


# --------------------------------------------------------------------------- command line
def write_all_cohorts(root: Path, cfg: FormatConfig) -> None:
    for name in cfg.cohorts:
        copies = 5 if cfg.cohorts[name].has_down_copies else 0
        write_cohort(root, cfg, name, {"secretsubj": {"w1": 1, "w2": 0}}, copies=copies)


def test_cli_end_to_end(
    tmp_path: Path, cfg: FormatConfig, capsys: pytest.CaptureFixture[str]
) -> None:
    write_all_cohorts(tmp_path, cfg)
    out_dir = tmp_path / "out"
    rc = main(["--data-dir", str(tmp_path), "--out-dir", str(out_dir)])
    text = capsys.readouterr().out
    assert rc == 0 and "Counts match GAIT-08 for every cohort" in text
    assert "secretsubj" not in text  # subject IDs are never printed
    assert "Not loaded: the label files' medication" in text
    assert (out_dir / "example_walk.png").is_file()


def test_cli_all_copies_skips_the_gait08_comparison(
    tmp_path: Path, cfg: FormatConfig, capsys: pytest.CaptureFixture[str]
) -> None:
    write_all_cohorts(tmp_path, cfg)
    rc = main(["--data-dir", str(tmp_path), "--all-copies", "--no-plot"])
    assert rc == 0 and "not compared with GAIT-08" in capsys.readouterr().out


def test_cli_missing_files_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--data-dir", str(tmp_path)]) == 2
    assert "MISSING" in capsys.readouterr().out


# --------------------------------------------------------------------------- real data
@pytest.mark.skipif(not REAL_DATA.is_dir(), reason="CARE-PD data not downloaded")
def test_real_data_reproduces_the_gait08_counts(cfg: FormatConfig) -> None:
    walks, reports = load_all(REAL_DATA, cfg)
    assert all(r.matches_gait08 for r in reports)
    assert len(walks) == 2950  # usable walks counted in GAIT-08
    merged = class_counts(walks)
    assert [merged[c].walks for c in (0, 1, 2)] == [1244, 1071, 635]
    assert all(w.joints.shape[1:] == (17, 3) for w in walks)
