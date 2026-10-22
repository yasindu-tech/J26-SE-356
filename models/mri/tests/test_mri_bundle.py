"""Tests for models/mri/src/mri_bundle.py on synthetic people (no real data)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

import mri_bundle as mb  # noqa: E402
from mri_contract import MriContractError  # noqa: E402
from test_train_mri import T1_COLS, cohort  # noqa: E402


def scanners(d) -> dict[str, pd.DataFrame]:
    names = np.where(d.batches["t1"] == "A", "Prisma_fit", "TrioTim")
    return {"t1": pd.DataFrame({"scanner_model": names}, index=d.subjects)}


def cv_payload(name: str = "t1", valid: bool = True) -> dict:
    model = {
        "model": "lgbm_t1",
        "auc_mean": 0.7,
        "auc_ci": [0.6, 0.8],
        "bal_acc_mean": 0.65,
        "bal_acc_ci": [0.55, 0.75],
        "sensitivity": 0.6,
        "specificity": 0.7,
        "auc_vs_age_sex": [0.2, 0.05, 0.35],
    }
    return {"name": name, "valid": valid, "models": [model], "split": {"folds": 4, "repeats": 1}}


def small_set(tmp_path: Path):
    d = cohort(n=40, signal=2.0)
    return d, mb.build_set(d, scanners(d), cv_payload() | {"run": "r1"}, n_splits=4, n_repeats=1)


def person(d, i: int):
    return (
        d.features.iloc[i].to_dict(),
        {"t1": str(d.batches["t1"][i])},
        float(d.age[i]),
        float(d.sex[i]),
    )


def test_set_has_final_model_fold_copies_map_and_card(tmp_path: Path) -> None:
    d, s = small_set(tmp_path)
    assert len(s.members) == 4  # 4 folds x 1 repeat
    assert s.scanner_map == {"t1": {"Prisma": "A", "TrioTim": "B"}}  # Prisma_fit -> Prisma
    assert s.card["training_people"] == {"total": 40, "pd": 20, "control": 20}
    assert s.card["cross_validated"]["auc_ci"] == [0.6, 0.8]
    assert "Not a diagnosis" in s.card["purpose"]
    assert set(s.feature_ranges) == set(T1_COLS)


def test_track_record_counts_how_often_each_lean_was_right(tmp_path: Path) -> None:
    run = tmp_path / "r1"
    run.mkdir()
    pd.DataFrame(
        {"subject": list("abcdef"), "y": [1, 1, 0, 0, 0, 1], "lgbm_t1": [0.9, 0.6, 0.7, 0.2, 0.4, 0.3]}
    ).to_csv(run / "predictions_t1.csv", index=False)
    tr = mb.track_record("t1", "r1", tmp_path)
    assert tr["leans_pd"] == {"n": 3, "correct": 2}  # 0.9 PD, 0.6 PD, 0.7 control
    assert tr["leans_control"] == {"n": 3, "correct": 2}  # 0.2, 0.4 control; 0.3 PD
    assert mb.track_record("t1", "missing", tmp_path) is None


def test_explanation_checks_summary_comes_from_the_newest_run_of_the_set(tmp_path: Path) -> None:
    def write(name: str, set_name: str, passed: bool) -> None:
        d = tmp_path / name
        d.mkdir()
        payload = {
            "set": set_name, "people": 37,
            "deletion": {"passed": passed, "mean_top_minus_random": 0.2, "ci_95": [0.1, 0.3]},
            "stability": {"mean_spearman": 0.6, "spearman_ci_95": [0.5, 0.7], "meets_target": False,
                          "mean_top5_overlap": 0.4, "top5_overlap_by_chance": 0.05},
        }
        (d / "explain_checks.json").write_text(json.dumps(payload))

    write("explain_20260101_000000", "paired", False)
    write("explain_20260102_000000", "paired", True)
    write("explain_20260103_000000", "t1", False)
    s = mb.explanation_checks("paired", tmp_path)
    assert s["source"] == "explain_20260102_000000" and s["deletion_passed"] is True
    assert mb.explanation_checks("dti", tmp_path) is None


def test_saved_bundle_scores_exactly_like_the_fitted_one(tmp_path: Path) -> None:
    d, s = small_set(tmp_path)
    out = mb.save_bundle(mb.Bundle("v1", {"t1": s}, "now", "abc"), tmp_path)
    assert (tmp_path / "LATEST").read_text().strip() == "v1" and out.name == "v1"
    loaded = mb.load_bundle(root=tmp_path).sets["t1"]
    raw, batch, age, sex = person(d, 3)
    before = s.final.score(s.final.harmonise(raw, batch, age, sex))
    after = loaded.final.score(loaded.final.harmonise(raw, batch, age, sex))
    assert after == pytest.approx(before)
    contrib, base = loaded.final.contributions(loaded.final.harmonise(raw, batch, age, sex))
    assert sum(contrib.values()) + base == pytest.approx(after)  # SHAP adds up to the score


def test_bundle_is_plain_json_and_tampering_is_refused(tmp_path: Path) -> None:
    _, s = small_set(tmp_path)
    out = mb.save_bundle(mb.Bundle("v1", {"t1": s}), tmp_path)
    json.loads((out / "bundle.json").read_text())  # no pickle anywhere
    (out / "bundle.json").write_text((out / "bundle.json").read_text().replace("Prisma", "Prizma"))
    with pytest.raises(mb.BundleError, match="manifest"):
        mb.load_bundle(out)


def test_no_bundle_yet_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(mb.BundleError, match="no MRI model bundle"):
        mb.load_bundle(root=tmp_path)


def test_only_valid_cross_validated_sets_can_be_bundled(tmp_path: Path) -> None:
    run = tmp_path / "20260101_000000"
    run.mkdir()
    (run / "results.json").write_text(json.dumps({"sets": [cv_payload(valid=False)]}))
    with pytest.raises(mb.BundleError, match="VALID"):
        mb.cv_result("t1", tmp_path)
    newer = tmp_path / "20260102_000000"
    newer.mkdir()
    (newer / "results.json").write_text(json.dumps({"sets": [cv_payload()]}))
    assert mb.cv_result("t1", tmp_path)["run"] == "20260102_000000"


def test_forbidden_feature_cannot_reach_the_deployed_model() -> None:
    d = cohort(n=40, cols=[*T1_COLS[:5], "updrs3_total"])
    with pytest.raises(MriContractError):
        mb.fit_member(d, np.arange(d.n), 0)


def test_scanner_model_in_two_batches_is_refused() -> None:
    d = cohort(n=40)
    one_name = {"t1": pd.DataFrame({"scanner_model": ["Prisma"] * 40}, index=d.subjects)}
    with pytest.raises(mb.BundleError, match="two ComBat batches"):
        mb.scanner_map(d, one_name)


def test_choose_uses_everything_that_was_uploaded(tmp_path: Path) -> None:
    _, s = small_set(tmp_path)
    b = mb.Bundle("v", {"t1": s, "paired": s, "dti": s})
    assert b.choose({"t1", "dti"}) is b.sets["paired"]
    assert b.choose({"t1"}) is b.sets["t1"]
    assert b.choose({"dti"}) is b.sets["dti"]
    assert b.choose(set()) is None
    assert mb.Bundle("v", {"t1": s}).choose({"t1", "dti"}) is s  # no paired model: T1 alone
