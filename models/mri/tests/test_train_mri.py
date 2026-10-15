"""Tests for models/mri/src/train_mri.py on synthetic people (no real data).

The synthetic cohort has a real group difference in two T1 features, a scanner shift on
every feature, and age and sex that carry no signal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("sklearn")

import harmonise  # noqa: E402
import train_mri as tm  # noqa: E402
from mri_contract import MriContractError  # noqa: E402
from training_data import ModelData  # noqa: E402

T1_COLS = [
    "L_putamen_grey_frac",
    "R_putamen_grey_frac",
    "L_caudate_t1_mean",
    "R_caudate_t1_mean",
    "brain_volume_ml",
    "affine_det",
]


def cohort(n: int = 40, signal: float = 1.5, seed: int = 0, cols=T1_COLS) -> ModelData:
    rng = np.random.default_rng(seed)
    y = np.tile([0, 1], n // 2)
    batch = np.array(["A", "B"])[np.arange(n) % 4 // 2]  # 2 scanners, balanced with y
    x = rng.normal(size=(n, len(cols)))
    x[:, :2] += signal * y[:, None]  # the group difference
    x += np.where(batch == "B", 3.0, 0.0)[:, None]  # the scanner shift
    return ModelData(
        name="t1",
        subjects=np.array([f"s{i:03d}" for i in range(n)]),
        y=y,
        age=rng.normal(65, 8, n),
        sex=rng.integers(0, 2, n).astype(float),
        features=pd.DataFrame(x, columns=cols),
        groups={"t1": list(cols)},
        batches={"t1": batch},
    )


def test_ladder_models_for_each_set() -> None:
    assert list(tm.models_for(cohort())) == ["age_sex", "best_single", "lgbm_t1"]
    paired = cohort()
    paired.groups = {"t1": T1_COLS[:3], "dti": T1_COLS[3:]}
    assert list(tm.models_for(paired))[-1] == "lgbm_t1dti"


def test_best_single_feature_finds_the_signal() -> None:
    d = cohort(n=200, signal=2.0)
    assert tm.best_single_feature(d.features, d.y) in T1_COLS[:2]


def test_every_person_scored_by_models_that_never_saw_them() -> None:
    d = cohort()
    probs = tm.run_cv(d, tm.models_for(d), n_splits=4, n_repeats=2)
    for p in probs.values():
        assert p.shape == (2, d.n) and not np.isnan(p).any()


def test_signal_beats_age_and_sex() -> None:
    d = cohort(n=60, signal=2.0)
    probs = tm.run_cv(d, tm.models_for(d), n_splits=5, n_repeats=2)
    reports = {r.model: r for r in tm.summarise_models(d.y, probs, n_boot=300, seed=0)}
    assert reports["lgbm_t1"].auc_mean > 0.8
    assert reports["age_sex"].auc_mean < 0.7
    assert reports["lgbm_t1"].beats_age_sex


def test_combat_is_fitted_on_training_people_only(monkeypatch) -> None:
    d = cohort()
    sizes = []
    real_fit = harmonise.ComBat.fit

    def spy(self, x, batch, covariates=None):
        sizes.append(len(x))
        return real_fit(self, x, batch, covariates)

    monkeypatch.setattr(tm.ComBat, "fit", spy)
    folds = tm.subject_folds(d.subjects, d.y, 4, 1, 0)
    for f in folds:
        sizes.clear()
        tm.fit_predict_fold(d, f.train_idx, f.test_idx, tm.models_for(d), 0)
        assert sizes == [len(f.train_idx)]  # one ComBat fit, on exactly the training rows


def test_overlapping_fold_is_refused() -> None:
    d = cohort()
    from folds import SubjectLeakError

    with pytest.raises(SubjectLeakError):
        tm.fit_predict_fold(d, np.arange(0, 30), np.arange(25, 40), tm.models_for(d), 0)


def test_forbidden_column_stops_training() -> None:
    d = cohort(cols=[*T1_COLS[:5], "updrs3_total"])
    with pytest.raises(MriContractError):
        tm.run_cv(d, tm.models_for(d), n_splits=4, n_repeats=1)


def test_unknown_column_stops_training() -> None:
    d = cohort(cols=[*T1_COLS[:5], "moca"])
    with pytest.raises(MriContractError):
        tm.run_cv(d, tm.models_for(d), n_splits=4, n_repeats=1)


def test_full_set_run_and_controls() -> None:
    d = cohort(n=40, signal=2.0)
    report, _ = tm.run_set(
        d,
        n_splits=4,
        n_repeats=2,
        # 20 shuffles: with 5, the mean of the null scores is too noisy to compare with
        # the 0.55 limit (its spread on 40 people is about 0.08 per shuffle)
        n_permutations=20,
        permutation_repeats=1,
        gate_permutations=20,
        n_boot=200,
    )
    gate = report.site_gate["t1"]
    assert gate["before_harmonisation"] > 0.9  # the shift makes the scanner obvious
    assert gate["passed"]  # and ComBat removes it
    assert report.permutation["passed"] and report.permutation["model"] == "lgbm_t1"
    assert report.valid


def test_progress_and_summary_reach_the_dashboard(tmp_path: Path) -> None:
    from progress import RunRecorder, fold_events, read_events

    d = cohort(n=40, signal=2.0)
    rec = RunRecorder("train", ["t1"], tm.TRAIN_STEPS, runs_dir=tmp_path, hash_ids=False)
    rec.subject_start("t1")
    report, _ = tm.run_set(
        d,
        n_splits=4,
        n_repeats=2,
        n_permutations=3,
        permutation_repeats=1,
        gate_permutations=10,
        n_boot=100,
        recorder=rec,
    )
    rec.subject_end("t1", "ok")
    state = fold_events(read_events(rec.path), 1e12)
    item = state["subjects"][0]
    assert set(item["durations"]) == set(tm.TRAIN_STEPS)
    summary = item["report"]
    assert summary["main_model"] == "lgbm_t1" and summary["n"] == 40
    sp = summary["split"]
    assert (sp["folds"], sp["repeats"], sp["fits_per_model"]) == (4, 2, 8)
    assert sp["train_people"] == [30, 30] and sp["test_people"] == [10, 10]
    assert {m["model"] for m in summary["models"]} == {"age_sex", "best_single", "lgbm_t1"}
    # only aggregates leave the run: no pseudonyms, no per-person scores
    text = rec.path.read_text()
    assert not any(s in text for s in d.subjects)


def test_failed_control_marks_report_invalid_and_raises() -> None:
    r = tm.SetReport("t1", 10, 5, 3, {}, [], {"t1": {"passed": False}}, {"passed": True})
    assert not r.valid
    with pytest.raises(tm.ResultsInvalid):
        tm.enforce_controls([r])


def test_outputs_have_no_raw_ids_and_mark_validity(tmp_path: Path) -> None:
    d = cohort(n=40, signal=2.0)
    report, probs = tm.run_set(
        d,
        n_splits=4,
        n_repeats=1,
        n_permutations=3,
        permutation_repeats=1,
        gate_permutations=10,
        n_boot=100,
    )
    settings = {"time": "t", "commit": "c", "folds": 4, "repeats": 1}
    tm.write_outputs(tmp_path, [report], {"t1": probs}, {"t1": d}, settings)
    md = (tmp_path / "report.md").read_text()
    status = "VALID" if report.valid else "INVALID"
    assert f"Set: t1 ({status}" in md and "AUC vs age+sex" in md
    preds = pd.read_csv(tmp_path / "predictions_t1.csv")
    assert list(preds.columns[:2]) == ["subject", "y"] and len(preds) == 40


def test_small_scanner_group_always_keeps_two_in_training() -> None:
    # The real failure: a scanner group of 3 landed 2 people in one test fold, leaving 1 in
    # training, and ComBat refused to fit. Dealt folds keep at most 1 of 3 per test fold.
    d = cohort(n=36)
    d.batches["t1"] = np.array(["A"] * 12 + ["B"] * 9 + ["C"] * 6 + ["D"] * 6 + ["E"] * 3)
    for seed in range(20):
        for f in tm.make_folds(d, d.y, 5, 2, seed):
            train_counts = pd.Series(d.batches["t1"][f.train_idx]).value_counts()
            assert len(train_counts) == 5 and train_counts.min() >= 2
    probs = tm.run_cv(d, tm.models_for(d), n_splits=5, n_repeats=2)
    assert not np.isnan(probs["lgbm_t1"]).any()


def test_fold_groups_join_every_modality() -> None:
    d = cohort(n=8)
    d.batches = {"t1": np.array(["A", "B"] * 4), "dti": np.array(["X"] * 4 + ["Y"] * 4)}
    assert list(tm.fold_groups(d)[:2]) == ["A|X", "B|X"]
