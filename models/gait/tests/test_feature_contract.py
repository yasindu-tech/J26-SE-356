"""Tests for the feature contract check (GAIT-10, Check D), including the failing cases."""

from __future__ import annotations

import pytest
from feature_contract import (
    R_DAT_QSM,
    R_HY,
    R_LABEL,
    R_MEDICATION,
    R_UPDRS,
    FeatureContractError,
    check_feature_contract,
    find_violations,
    main,
    split_words,
)

GOOD = ["stride_time_cv", "step_width_mean", "cadence", "gait_speed", "knee_angle_range"]


# --------------------------------------------------------------------------- failing cases
@pytest.mark.parametrize(
    ("name", "rule"),
    [
        ("UPDRS_GAIT", R_UPDRS),
        ("updrs-gait", R_UPDRS),
        ("UPDRSGait", R_UPDRS),
        ("mds_updrs_part3", R_UPDRS),
        ("MDSUPDRS", R_UPDRS),
        ("hoehn_yahr", R_HY),
        ("Hoehn & Yahr", R_HY),
        ("H&Y_stage", R_HY),
        ("hy_stage", R_HY),
        ("medication", R_MEDICATION),
        ("Medication_State", R_MEDICATION),
        ("levodopa_dose", R_MEDICATION),
        ("LEDD", R_MEDICATION),
        ("on_off", R_MEDICATION),
        ("ON/OFF", R_MEDICATION),
        ("onOff", R_MEDICATION),
        ("clinician_rating", "clinician impression or rating"),
        ("DaTscan_sbr", R_DAT_QSM),
        ("dat_spect_ratio", R_DAT_QSM),
        ("qsm_putamen", R_DAT_QSM),
    ],
)
def test_banned_input_fails(name: str, rule: str) -> None:
    with pytest.raises(FeatureContractError) as err:
        check_feature_contract([*GOOD, name])
    assert [(v.name, v.rule) for v in err.value.violations] == [(name, rule)]
    assert name in str(err.value)


def test_every_violation_is_reported_at_once() -> None:
    with pytest.raises(FeatureContractError) as err:
        check_feature_contract(["cadence", "UPDRS_GAIT", "medication", "H&Y"])
    assert [v.name for v in err.value.violations] == ["UPDRS_GAIT", "medication", "H&Y"]


def test_label_column_as_input_fails() -> None:
    with pytest.raises(FeatureContractError) as err:
        check_feature_contract([*GOOD, "Severity-Class"], label_name="severity_class")
    assert err.value.violations[0].rule == R_LABEL


def test_updrs_label_passed_as_input_fails_even_without_naming_the_label() -> None:
    with pytest.raises(FeatureContractError):
        check_feature_contract([*GOOD, "UPDRS_GAIT"])


# --------------------------------------------------------------------------- passing cases
def test_clean_inputs_pass() -> None:
    check_feature_contract(GOOD, label_name="UPDRS_GAIT")  # the label is allowed as a label


@pytest.mark.parametrize(
    "name", ["median_filter", "data_quality", "height_ratio", "on_floor_time", "step_rate", "hip_y"]
)
def test_harmless_names_are_not_flagged(name: str) -> None:
    check_feature_contract([name])


def test_duplicates_are_reported_once() -> None:
    assert len(find_violations(["medication", "medication"])) == 1


def test_accepts_any_iterable_of_names() -> None:
    check_feature_contract(iter(GOOD))
    check_feature_contract(tuple(GOOD))
    with pytest.raises(FeatureContractError):
        check_feature_contract(n for n in ["cadence", "medication"])


# --------------------------------------------------------------------------- bad arguments
def test_empty_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="no input features"):
        check_feature_contract([])


def test_non_string_names_are_rejected() -> None:
    with pytest.raises(TypeError):
        check_feature_contract(["cadence", 7])  # type: ignore[list-item]


def test_split_words() -> None:
    assert split_words("UPDRSGait") == ["updrs", "gait"]
    assert split_words("H&Y_stage") == ["h", "and", "y", "stage"]
    assert split_words("on-off") == ["on", "off"]


# --------------------------------------------------------------------------- command line
def test_cli_exit_codes(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([*GOOD, "--label", "UPDRS_GAIT"]) == 0
    assert "OK" in capsys.readouterr().out
    assert main([*GOOD, "UPDRS_GAIT"]) == 1
    assert "UPDRS_GAIT" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main([])
