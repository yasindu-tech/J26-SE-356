"""Tests for the fast-vs-full eddy comparison and the eddy profile switch."""

from __future__ import annotations

import json
from pathlib import Path

import compare_dti_eddy as c
import numpy as np
import pandas as pd
import preprocess_dti as d
import pytest


def table(values: np.ndarray, names: list[str] | None = None) -> pd.DataFrame:
    n, k = values.shape
    cols = names or [f"dti_f{i}_fa_mean" for i in range(k)]
    df = pd.DataFrame(values, columns=cols)
    df.insert(0, "subject", [f"p{i:02d}" for i in range(n)])
    return df


def people(n: int, k: int = 6, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.5, 0.1, size=(n, k))


# ---------------------------------------------------------------- the profile switch


def test_full_profile_adds_no_options() -> None:
    p = Path("x")
    cmd = d.eddy_arguments(p, p, p, p, p, p, p, d.DTIConfig())
    assert not any(a.startswith(("--niter", "--fwhm")) for a in cmd)


def test_fast_profile_sets_iterations_and_smoothing() -> None:
    p = Path("x")
    cmd = d.eddy_arguments(p, p, p, p, p, p, p, d.DTIConfig(eddy_profile="fast"))
    assert "--niter=2" in cmd and "--fwhm=10,0" in cmd
    assert "--repol" in cmd  # outlier replacement stays on


def test_unknown_profile_is_rejected() -> None:
    p = Path("x")
    with pytest.raises(d.StepError, match="unknown eddy profile"):
        d.eddy_arguments(p, p, p, p, p, p, p, d.DTIConfig(eddy_profile="turbo"))


def test_fast_profile_cannot_write_into_the_full_results_folder() -> None:
    with pytest.raises(SystemExit):
        d.main(["--eddy-profile", "fast", "--out", str(d.DEFAULT_OUT)])


def test_fast_profile_defaults_to_its_own_folders() -> None:
    assert d.DEFAULT_OUT_FAST != d.DEFAULT_OUT
    assert d.DEFAULT_TABLE_OUT_FAST.parent != d.DEFAULT_TABLE_OUT.parent


# ---------------------------------------------------------------------------- ICC


def test_icc_is_one_for_identical_measurements() -> None:
    x = np.array([1.0, 2.0, 3.5, 4.0, 5.2, 6.1])
    assert c.icc_2_1(x, x) == pytest.approx(1.0)


def test_icc_drops_for_a_constant_shift() -> None:
    x = np.array([1.0, 2.0, 3.5, 4.0, 5.2, 6.1])
    assert c.icc_2_1(x, x + 3.0) < 0.8  # correlation is 1, agreement is not


def test_icc_is_low_for_unrelated_measurements() -> None:
    rng = np.random.default_rng(1)
    assert c.icc_2_1(rng.normal(size=40), rng.normal(size=40)) < 0.3


def test_icc_with_no_variation_is_nan() -> None:
    assert np.isnan(c.icc_2_1(np.ones(6), np.ones(6)))


# ---------------------------------------------------------------------- agreement


def test_identical_tables_agree() -> None:
    v = people(8)
    t = c.agreement(table(v), table(v))
    assert t["agrees"].all() and (t["mean_rel_diff_pct"] == 0).all()
    assert c.verdict(t).startswith("AGREES")


def test_small_noise_still_agrees() -> None:
    v = people(20)
    noise = np.random.default_rng(2).normal(0, 0.0005, size=v.shape)  # ~0.1% of the value
    assert c.verdict(c.agreement(table(v), table(v + noise))).startswith("AGREES")


def test_a_biased_fast_profile_is_flagged() -> None:
    v = people(20)
    t = c.agreement(table(v), table(v * 1.08))  # fast reads 8% higher everywhere
    assert not t["agrees"].any()
    assert c.verdict(t).startswith("DIFFERS")
    assert (t["mean_diff"] > 0).all()


def test_two_people_can_never_pass() -> None:
    v = people(2)
    t = c.agreement(table(v), table(v))  # perfect agreement, but only 2 people
    assert not t["agrees"].any()
    assert c.verdict(t).startswith("TOO FEW PEOPLE")


def test_only_shared_people_and_dti_columns_are_compared() -> None:
    a, b = people(6), people(6)
    full = table(a, ["dti_a", "dti_b", "t1_x", "dti_only_full", "dti_c", "dti_d"])
    fast = table(b, ["dti_a", "dti_b", "t1_x", "dti_only_fast", "dti_c", "dti_d"]).iloc[1:]
    out = c.agreement(full, fast)
    assert set(out["feature"]) == {"dti_a", "dti_b", "dti_c", "dti_d"}
    assert (out["n"] == 5).all()


def test_no_shared_people_is_an_error() -> None:
    full, fast = table(people(5)), table(people(5))
    fast["subject"] = [f"q{i}" for i in range(5)]
    with pytest.raises(c.CompareError, match="share no people"):
        c.agreement(full, fast)


def test_duplicate_person_is_an_error() -> None:
    full = table(people(5))
    full.loc[1, "subject"] = full.loc[0, "subject"]
    with pytest.raises(c.CompareError, match="twice"):
        c.agreement(full, table(people(5)))


# ------------------------------------------------------------- result file pairs


def fake_result(
    values: dict[str, float], score: float, profile: str, seconds: float, interval=(-1.0, 1.0)
) -> dict:
    names = list(values)
    return {
        "modality": "mri",
        "status": "scored",
        "feature_vector": {"names": ["t1_x", *names], "raw": [0.1, *values.values()]},
        "score": {"value": score, "interval_95": list(interval)},
        "inputs": {"dti": {"eddy": {"profile": profile, "seconds": seconds}}},
    }


def test_pairs_use_only_dti_features_and_report_scores() -> None:
    feats = {"dti_a": 0.5, "dti_b": 0.6}
    pair = c.ScanPair(
        "3104",
        fake_result(feats, 0.20, "full", 5000.0),
        fake_result({"dti_a": 0.51, "dti_b": 0.6}, 0.30, "fast", 2000.0),
    )
    full, fast = c.pairs_to_tables([pair])
    assert list(full.columns) == ["subject", "dti_a", "dti_b"]
    s = c.score_rows([pair])
    assert s["score_diff"].iloc[0] == pytest.approx(0.10)
    assert bool(s["fast_inside_full_interval"].iloc[0])
    assert "2.50x faster" in c.timing_line(s)


def test_fast_score_outside_the_full_interval_is_shown() -> None:
    feats = {"dti_a": 0.5}
    pair = c.ScanPair(
        "p", fake_result(feats, 0.0, "full", 1.0), fake_result(feats, 3.0, "fast", 1.0)
    )
    assert not bool(c.score_rows([pair])["fast_inside_full_interval"].iloc[0])


def test_mislabelled_files_are_refused() -> None:
    feats = {"dti_a": 0.5}
    pair = c.ScanPair(
        "p", fake_result(feats, 0.0, "fast", 1.0), fake_result(feats, 0.0, "fast", 1.0)
    )
    with pytest.raises(c.CompareError, match="made with the 'fast' profile"):
        c.check_profiles([pair])


def test_result_without_features_is_an_error() -> None:
    with pytest.raises(c.CompareError, match="no feature vector"):
        c.result_features({"modality": "mri", "status": "abstained", "feature_vector": None})


def test_cli_writes_the_report(tmp_path: Path) -> None:
    v = people(8)
    table(v).to_csv(tmp_path / "full.csv", index=False)
    table(v * 1.08).to_csv(tmp_path / "fast.csv", index=False)
    out = tmp_path / "out"
    code = c.main(
        ["--csv", str(tmp_path / "full.csv"), str(tmp_path / "fast.csv"), "--out", str(out)]
    )
    assert code == 0
    text = (out / "comparison.md").read_text()
    assert "DIFFERS" in text and "People compared: **8**" in text
    assert (out / "feature_agreement.csv").exists()


def test_cli_reports_bad_input(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"modality": "gait"}))
    assert c.main(["--pair", "x", str(bad), str(bad), "--out", str(tmp_path / "o")]) == 2
