"""GAIT-22: the saved model reloads with identical predictions and refuses a changed bundle."""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import model_bundle as mb
import numpy as np
import pytest
import train_xgboost as tx
from gait_extract import FEATURE_NAMES
from test_evaluate_cohorts import FAST, cohort_table


@pytest.fixture(scope="module")
def fitted() -> tuple[tx.XGBClassifier, np.ndarray]:
    table = cohort_table()
    model = tx.fit_final_model(table, params=FAST)
    return model, table[list(FEATURE_NAMES)].to_numpy(float)


@pytest.fixture
def saved(fitted: tuple[tx.XGBClassifier, np.ndarray], tmp_path: Path) -> Path:
    mb.save_bundle(fitted[0], tmp_path / "b", {"trained_on": "test"})
    return tmp_path / "b"


def edit_info(folder: Path, **changes: object) -> None:
    path = folder / mb.INFO_FILE
    info = json.loads(path.read_text())
    for key, value in changes.items():
        if value is None:
            info.pop(key)
        else:
            info[key] = value
    path.write_text(json.dumps(info))


def test_reload_predicts_exactly_the_same(
    fitted: tuple[tx.XGBClassifier, np.ndarray], saved: Path
) -> None:
    model, x = fitted
    assert np.array_equal(mb.load_bundle(saved).predict_proba(x), model.predict_proba(x))


def test_version_follows_the_model_not_the_time(
    fitted: tuple[tx.XGBClassifier, np.ndarray], saved: Path, tmp_path: Path
) -> None:
    again = mb.save_bundle(fitted[0], tmp_path / "again", {})
    first = mb.load_bundle(saved)
    assert first.version == again.version
    assert (
        first.version
        == "gait-xgb-" + hashlib.sha1((saved / mb.MODEL_FILE).read_bytes()).hexdigest()[:8]
    )
    other = tx.fit_final_model(cohort_table(), seed=1, params={**FAST, "subsample": 0.5})
    assert mb.save_bundle(other, tmp_path / "other", {}).version != first.version


def test_bundle_records_inputs_classes_and_the_caveat(saved: Path) -> None:
    info = json.loads((saved / mb.INFO_FILE).read_text())
    assert info["features"] == list(FEATURE_NAMES)
    assert info["classes"] == {"0": "class 0", "1": "class 1", "2": "class 2-3"}
    assert info["probabilities"] == "uncalibrated" and "never a result" in info["note"]
    assert info["trained_on"] == "test"


def test_changed_model_file_is_refused(saved: Path) -> None:
    with (saved / mb.MODEL_FILE).open("a") as f:
        f.write(" ")
    with pytest.raises(mb.BundleError, match="hash"):
        mb.load_bundle(saved)


def test_feature_order_change_is_refused(saved: Path) -> None:
    edit_info(saved, features=list(reversed(FEATURE_NAMES)))
    with pytest.raises(mb.BundleError, match="differ"):
        mb.load_bundle(saved)


@pytest.mark.parametrize("field", mb.REQUIRED)
def test_missing_field_is_refused(saved: Path, field: str) -> None:
    edit_info(saved, **{field: None})
    with pytest.raises(mb.BundleError):
        mb.load_bundle(saved)


def test_missing_files_are_refused(saved: Path) -> None:
    (saved / mb.INFO_FILE).unlink()
    with pytest.raises(mb.BundleError, match="missing"):
        mb.load_bundle(saved)


def test_other_xgboost_version_warns_but_loads(saved: Path) -> None:
    edit_info(saved, xgboost_version="0.0.1")
    with pytest.warns(UserWarning, match="0.0.1"):
        mb.load_bundle(saved)


def test_same_xgboost_version_does_not_warn(saved: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mb.load_bundle(saved)


def test_wrong_classes_are_refused_on_save(tmp_path: Path) -> None:
    table = cohort_table()
    two = table[table["severity"] < 2]
    model = tx.XGBClassifier(n_estimators=3).fit(
        two[list(FEATURE_NAMES)].to_numpy(float), two["severity"].to_numpy(int)
    )
    with pytest.raises(mb.BundleError, match="classes"):
        mb.save_bundle(model, tmp_path / "x", {})


def test_predict_proba_checks_its_input(saved: Path) -> None:
    bundle = mb.load_bundle(saved)
    one = np.ones(len(FEATURE_NAMES))
    assert bundle.predict_proba(one).shape == (1, 3)
    assert bundle.predict_proba(np.tile(one, (4, 1))).shape == (4, 3)
    assert np.allclose(bundle.predict_proba(one).sum(axis=1), 1.0)
    with pytest.raises(mb.BundleError, match="expected 8"):
        bundle.predict_proba(one[:5])
    bad = one.copy()
    bad[2] = np.nan
    with pytest.raises(mb.BundleError, match="missing"):
        bundle.predict_proba(bad)
