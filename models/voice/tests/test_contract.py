"""Tests for models/voice/src/contract.py. Each banned input is proved to fail."""

from __future__ import annotations

import contract
import pytest

ACOUSTIC = ["PPE", "locPctJitter", "apq3Shimmer", "mean_MFCC_3rd_coef", "tqwt_energy_dec_12"]


def test_normal_acoustic_features_pass() -> None:
    assert contract.validate_features(ACOUSTIC) == ACOUSTIC


@pytest.mark.parametrize(
    "banned",
    [
        "UPDRS_part3",
        "mds_updrs_total",
        "hoehn_yahr",
        "levodopa_dose",
        "LEDD",
        "medication_status",
        "ON_OFF",
        "on/off",
        "DaTscan_SBR",
        "qsm_nigra",
        "clinician_impression",
        "diagnosis_date",
        "id",
        "class",
        "Class",
    ],
)
def test_banned_column_fails(banned: str) -> None:
    with pytest.raises(contract.FeatureContractError, match=banned):
        contract.validate_features([*ACOUSTIC, banned])


def test_all_problems_reported_at_once() -> None:
    with pytest.raises(contract.FeatureContractError) as err:
        contract.validate_features(["UPDRS_part3", "PPE", "id"])
    assert "UPDRS_part3" in str(err.value) and "'id'" in str(err.value)


def test_empty_feature_list_fails() -> None:
    with pytest.raises(contract.FeatureContractError, match="no feature"):
        contract.validate_features([])


def test_gender_is_not_banned() -> None:
    """Gender is allowed by the contract; D7 keeps it out of the main model in data.py."""
    assert contract.validate_features(["gender"]) == ["gender"]
