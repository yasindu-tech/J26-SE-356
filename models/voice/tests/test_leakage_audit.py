"""Tests for models/voice/src/leakage_audit.py. Synthetic data only."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import leakage_audit
import numpy as np
import pandas as pd
import pytest

import data

FAST = dict(score="anova_f", n_boot=50, k_grid=(5, 10), c_grid=(0.1,))


@pytest.fixture(scope="module")
def audit() -> dict:
    """One small audit run on synthetic data, shared by the tests below."""
    rng = np.random.default_rng(42)
    n_people = 60
    groups = np.repeat(np.arange(5000, 5000 + n_people), 3)
    y = np.repeat((rng.random(n_people) < 0.75).astype(int), 3)
    X = pd.DataFrame(rng.normal(size=(len(groups), 30)), columns=[f"f{i}" for i in range(30)])
    X["f0"] += y * 1.5  # one real signal, so the pipeline has something to find
    return leakage_audit.run_audit(X, y, groups, **FAST)


def test_all_four_variants_with_cis(audit: dict) -> None:
    assert set(audit["variants"]) == {"A_full", "A_sel", "A_split", "B"}
    for v in audit["variants"].values():
        for metric in ("auc", "balanced_accuracy"):
            est = v[metric]
            assert est["lower"] <= est["value"] <= est["upper"]
    leakage_audit.validate_audit(audit)  # passes on a complete result


def test_headline_is_paired_difference_against_b(audit: dict) -> None:
    assert audit["headline"] == "A_full_minus_B"
    assert set(audit["auc_differences"]) == {"A_full_minus_B", "A_sel_minus_B", "A_split_minus_B"}
    expected = audit["variants"]["A_full"]["auc"]["value"] - audit["variants"]["B"]["auc"]["value"]
    assert audit["auc_differences"]["A_full_minus_B"]["value"] == pytest.approx(expected, abs=2e-4)


def test_missing_variant_fails_validation(audit: dict) -> None:
    broken = copy.deepcopy(audit)
    del broken["variants"]["A_split"]
    with pytest.raises(ValueError, match="missing variants"):
        leakage_audit.validate_audit(broken)


def test_missing_ci_fails_validation(audit: dict) -> None:
    broken = copy.deepcopy(audit)
    del broken["variants"]["B"]["auc"]["lower"]
    with pytest.raises(ValueError, match="no confidence interval"):
        leakage_audit.validate_audit(broken)


def test_missing_headline_fails_validation(audit: dict) -> None:
    broken = copy.deepcopy(audit)
    broken["auc_differences"].pop("A_full_minus_B")
    with pytest.raises(ValueError, match="headline"):
        leakage_audit.validate_audit(broken)


def test_output_holds_no_person_ids(audit: dict) -> None:
    text = json.dumps(audit)
    assert not any(str(pid) in text for pid in range(5000, 5060))


def test_charts_and_results_table_are_written(tmp_path: Path, synthetic_csv: Path) -> None:
    """main() itself expects the real 756-row file, so this runs the same steps it does."""
    out = tmp_path / "artifacts"
    X, y, groups, _ = data.load_uci470(synthetic_csv, expected_rows=180)
    result = leakage_audit.run_audit(X, y, groups, **FAST)
    result["noise_check"] = leakage_audit.noise_check(
        scores=("anova_f",), n_repeats=2, n_features=40
    )
    leakage_audit.validate_audit(result)
    out.mkdir()
    leakage_audit.plot_audit(result, out / "leakage_audit.png")
    leakage_audit.plot_noise(result["noise_check"], out / "noise_check.png")
    assert (out / "leakage_audit.png").stat().st_size > 0
    assert (out / "noise_check.png").stat().st_size > 0
    table = leakage_audit.results_markdown(result)
    assert "A_full_minus_B" in table and "Pure-noise check" in table
