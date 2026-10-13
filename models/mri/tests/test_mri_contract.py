"""Tests for models/mri/src/mri_contract.py. The validator must FAIL on banned inputs."""

from __future__ import annotations

import pytest
from mri_contract import (
    R_DAT_QSM,
    R_HY,
    R_LABEL,
    R_MEDICATION,
    R_UNKNOWN,
    R_UPDRS,
    MriContractError,
    check_mri_features,
)

GOOD = [
    "L_putamen_grey_frac",
    "R_caudate_white_frac",
    "L_thalamus_t1_entropy",
    "R_pallidum_t1_kurtosis",
    "brain_volume_ml",
    "affine_det",
]


def rule_for(name: str, **kw) -> str:
    with pytest.raises(MriContractError) as e:
        check_mri_features([*GOOD, name], **kw)
    (v,) = e.value.violations
    assert v.name == name
    return v.rule


def test_real_feature_names_pass() -> None:
    check_mri_features(GOOD)


@pytest.mark.parametrize(
    ("name", "rule"),
    [
        ("NP3TOT_updrs", R_UPDRS),
        ("mdsUpdrsPart3", R_UPDRS),
        ("hoehn_yahr", R_HY),
        ("H&Y", R_HY),
        ("hy_stage", R_HY),
        ("levodopa_equiv", R_MEDICATION),
        ("LEDD", R_MEDICATION),
        ("on_off_state", R_MEDICATION),
        ("medication_status", R_MEDICATION),
        ("DaTscan_SBR", R_DAT_QSM),
        ("dat_spect_putamen", R_DAT_QSM),
        ("qsm_nigra", R_DAT_QSM),
        ("label", R_LABEL),
        ("diagnosis", R_LABEL),
    ],
)
def test_specialist_inputs_are_rejected(name: str, rule: str) -> None:
    assert rule_for(name) == rule


@pytest.mark.parametrize("name", ["np3_total", "moca", "L_putamen_volume", "site_key"])
def test_unknown_names_are_rejected(name: str) -> None:
    # A clinical score with an innocent name, or a feature we decided not to use.
    assert rule_for(name) == R_UNKNOWN


def test_demographics_only_when_asked() -> None:
    assert rule_for("age") == R_UNKNOWN
    check_mri_features(["age", "sex"], allow_demographics=True)


def test_every_violation_is_listed() -> None:
    with pytest.raises(MriContractError) as e:
        check_mri_features(["updrs3", "moca", *GOOD])
    assert [v.name for v in e.value.violations] == ["updrs3", "moca"]


def test_duplicates_and_empty_fail() -> None:
    with pytest.raises(MriContractError, match="twice"):
        check_mri_features(["affine_det", "affine_det"])
    with pytest.raises(MriContractError, match="at least one"):
        check_mri_features([])


@pytest.mark.parametrize(
    "name",
    [
        "dti_CC_genu_fa_mean",
        "dti_L_plic_md_mean",
        "dti_R_cerebral_peduncle_fa_mean",
        "dti_L_putamen_md_mean",
        "dti_wm_fa_median",
    ],
)
def test_dti_feature_names_pass(name: str) -> None:
    check_mri_features([*GOOD, name])


@pytest.mark.parametrize(
    "name",
    ["dti_L_putamen_fa_mean", "dti_CC_genu_fa_max", "dti_substantia_nigra_fa_mean", "FA_putamen"],
)
def test_dti_names_outside_the_scheme_fail(name: str) -> None:
    assert rule_for(name) == R_UNKNOWN


def test_dti_datscan_name_still_banned() -> None:
    assert rule_for("dti_datscan_putamen") == R_DAT_QSM
