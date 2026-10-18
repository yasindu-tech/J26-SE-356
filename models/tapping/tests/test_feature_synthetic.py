"""End-to-end tests of Set A + Set B on synthetic signals with a known decrement (task FT-29).

Each signal goes through the shipped chain (smoothing, detector with taps.PARAMS, tap table,
features_a, features_b). Expected values are closed forms derived here, not the modules' own
functions applied to the truth, except where a test says so. Noise level: 3.26% of the real taps
(240 of 7371, 233 retained clips) carry raw_disagrees; white Gaussian noise before smoothing
(30 fps, P = 2/3 s, A0 = 1.2, 29 taps) gives the same share at sigma = 0.0625 (0.0324; 0.065
gives 0.041), so SIGMA = 0.0625. Tolerances for the noisy tests come from a 200-seed Monte Carlo
(d = 0.02, 30 and 60 fps, jitter off/on, 800 runs, 797 of which found all 29 taps). The measured
worst-cell |bias| + 3 SD at sigma 0.0625 is 0.077 / 0.086 / 0.116 / 0.180 for amp_slope /
amp_late_early / amp_min_initial / velocity slope; TOL is those values rounded up (0.08 / 0.09 /
0.12 / 0.19). None of the 797 qualifying runs exceeds them. The 20 fixed seeds here (0..19) were
fixed before looking at these tests' results.
No test reads data/.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import theilslopes

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import features_a as fa  # noqa: E402
import features_b as fb  # noqa: E402
import smoothing  # noqa: E402
import taps  # noqa: E402
from synthetic_tapping import chain, crop, gen  # noqa: E402

P = 2 / 3  # s per tap, a whole number of half-frames at 30 and 60 fps
A0 = 1.2
N = 29
LEAD = 0.5
PK = LEAD + np.arange(N) * P + P / 2  # true peak times of a constant-period clip
SIGMA = 0.0625
SEEDS = range(20)
TOL = {"amp_slope": 0.08, "amp_late_early": 0.09, "amp_min_initial": 0.12, "vel_slope": 0.19}
HERE = Path(__file__).resolve().parent


def ts_norm(t: np.ndarray, y: np.ndarray) -> float:
    """Theil-Sen slope / mean * 10, written out here (the definition of the slope features)."""
    return float(theilslopes(y, t)[0]) / float(np.mean(y)) * 10


def decline(d: float) -> np.ndarray:
    return A0 * (1 - d * PK)


def run(amps: Any = None, fps: float = 30.0, seed: int = 0, **kw: Any) -> Any:
    t, raw, truth = gen(np.random.default_rng(seed), amps=amps, fps=fps, **kw)
    return (*chain(t, raw, fps), truth)


# ----------------------------------------------------------------------- 1-4 amplitude


def test_linear_decline_returns_the_slope_put_in() -> None:
    d = 0.02
    amps = decline(d)  # A_k = A0 (1 - d t_k): Theil-Sen of A on t is exactly -A0 d, the mean of
    # A is A0 (1 - d tbar), so amp_slope = -10 d / (1 - d tbar) (about -0.2510)
    _, table, _, b, truth = run(amps)
    expected = -10 * d / (1 - d * PK.mean())
    assert len(table) == N and truth["peak_t"] == pytest.approx(PK)
    assert b["amp_slope"] == pytest.approx(expected, abs=0.005) and b["amp_slope"] < 0
    assert b["decrement_present"] is True
    # A falls monotonically, so the rolling median is A itself and the initial level is the median
    # of the first three taps, A_1: the onset is the first tap below 0.8 A_1 (all later ones are too)
    # late/early thirds: k = 29 // 3 = 9, so mean(A[-9:]) / mean(A[:9]) = (1 - d mean(t[-9:])) /
    # (1 - d mean(t[:9])); min/initial: the smallest interior 3-tap median of a falling series is
    # the second-to-last amplitude, over A_1 (the median of the first three)
    le = (1 - d * PK[-9:].mean()) / (1 - d * PK[:9].mean())
    assert b["amp_late_early"] == pytest.approx(le, abs=0.005)
    assert b["amp_min_initial"] == pytest.approx(amps[-2] / amps[1], abs=0.005)
    first_low = next(k for k in range(N) if amps[k] < 0.8 * amps[1])
    assert b["decrement_onset_frac"] == pytest.approx(first_low / N, abs=0.001)
    present, (onset, _) = fb.decrement(amps)  # the frozen function on the true amplitudes agrees
    assert present and b["decrement_onset_frac"] == pytest.approx(onset, abs=0.001)


def test_no_decline_is_flat() -> None:
    _, _, _, b, _ = run()
    assert b["amp_slope"] == pytest.approx(0.0, abs=0.005)
    assert b["amp_late_early"] == pytest.approx(1.0, abs=0.005)
    assert b["amp_min_initial"] == pytest.approx(1.0, abs=0.005)
    assert b["decrement_present"] is False and np.isnan(b["decrement_onset_frac"])
    assert "decrement_onset_frac:no_decrement" in b["nan_reasons"].split(";")


def test_increasing_amplitude_has_a_positive_slope_and_no_decrement() -> None:
    g = 0.02  # A_k = A0 (1 + g t_k): slope = 10 g / (1 + g tbar), about +0.166
    _, _, _, b, _ = run(A0 * (1 + g * PK))
    assert b["amp_slope"] == pytest.approx(10 * g / (1 + g * PK.mean()), abs=0.005)
    assert b["amp_slope"] > 0 and b["amp_late_early"] > 1 and b["decrement_present"] is False


def test_step_to_half_amplitude_at_tap_15() -> None:
    # taps 0..14 at A0, taps 15..28 at A0/2: the rolling median first drops below 0.8 x initial
    # at tap 15, so onset = 15 / 29; the last and first thirds (9 taps) are 0.5 and 1 of A0
    _, _, _, b, _ = run(np.where(np.arange(N) < 15, A0, A0 / 2))
    assert b["decrement_present"] is True
    assert b["decrement_onset_frac"] == pytest.approx(15 / 29, abs=0.001)
    assert b["amp_late_early"] == pytest.approx(0.5, abs=0.001)
    assert b["amp_min_initial"] == pytest.approx(0.5, abs=0.001)


def test_step_to_ninety_percent_is_not_a_decrement() -> None:
    _, _, _, b, _ = run(np.where(np.arange(N) < 15, A0, 0.9 * A0))
    assert b["decrement_present"] is False and np.isnan(b["decrement_onset_frac"])
    assert b["amp_late_early"] == pytest.approx(0.9, abs=0.001)


# ----------------------------------------------------------------------- 5 speed


def test_speed_decline_with_constant_amplitude() -> None:
    per = P * (1 + 0.03 * np.arange(N))  # each cycle 3% longer than the one before
    _, _, _, b, truth = run(periods=per, allow_ties=True)
    pk = truth["peak_t"]
    expected = ts_norm(pk, A0 * np.pi / per)  # the maximum slope of a raised cosine is A pi / P
    assert expected < -0.2
    assert b["open_vel_slope"] == pytest.approx(expected, abs=0.015)
    assert b["close_vel_slope"] == pytest.approx(expected, abs=0.015)
    assert abs(b["amp_slope"]) < 0.01
    assert b["iti_slope"] == pytest.approx(ts_norm(pk[1:], np.diff(pk)), abs=0.01)


# ----------------------------------------------------------------------- 6 pauses


@pytest.mark.parametrize("fps", [30.0, 60.0])
@pytest.mark.parametrize(
    ("pauses", "hes", "halts", "ratio", "longest"),
    [
        ({10: 0.8}, 1, 0, 2.2, 1.4667),  # (2/3 + 0.8) / (2/3) = 2.2: above 1.8, up to 3
        ({20: 2.0}, 0, 1, 4.0, 2.6667),  # (2/3 + 2.0) / (2/3) = 4.0: above 3
        ({10: 0.8, 20: 2.0}, 1, 1, 4.0, 2.6667),
        ({10: 0.4}, 0, 0, 1.6, 1.0667),  # 1.6: below 1.8, neither
    ],
)
def test_inserted_pauses(
    fps: float, pauses: dict[int, float], hes: int, halts: int, ratio: float, longest: float
) -> None:
    _, table, _, b, _ = run(fps=fps, pauses=pauses)
    assert len(table) == N
    assert (b["n_hesitations"], b["n_halts"]) == (hes, halts)
    assert b["longest_pause_ratio"] == pytest.approx(ratio, abs=1e-6)
    assert b["longest_pause_s"] == pytest.approx(longest, abs=1e-4)


# ----------------------------------------------------------------------- 7 rate, 8 fps


@pytest.mark.parametrize("fps", [30.0, 60.0])
def test_rate_and_cv_of_a_constant_period(fps: float) -> None:
    _, _, a, _, _ = run(fps=fps)
    assert a["tap_rate_hz"] == pytest.approx(1 / P, abs=1e-9)  # (n - 1) / (t_last - t_first)
    assert a["iti_cv"] < 1e-9 and a["iti_mean_s"] == pytest.approx(P, abs=1e-9)
    assert a["n_over_duration_hz"] != pytest.approx(a["tap_rate_hz"], abs=0.01)  # n / duration


def test_fps_changes_the_normalised_slopes_not_at_all_and_peak_slope_a_little() -> None:
    amps = decline(0.02)
    _, _, _, b30, _ = run(amps, fps=30.0)
    _, _, _, b60, _ = run(amps, fps=60.0)
    assert b30["amp_slope"] == pytest.approx(b60["amp_slope"], abs=1e-4)
    assert b30["open_vel_slope"] == pytest.approx(b60["open_vel_slope"], abs=1e-4)
    # the synthetic peak-slope fps effect is at most about 4% (here +1.2% at 1.5 Hz) and goes the
    # opposite way to the two real 60 fps clips, so it does not explain them
    _, _, a30, _, _ = run(fps=30.0)
    _, _, a60, _, _ = run(fps=60.0)
    assert 0.98 <= a60["open_vel_peak"] / a30["open_vel_peak"] <= 1.05
    assert 0.98 <= a60["close_vel_peak"] / a30["close_vel_peak"] <= 1.05


# ----------------------------------------------------------------------- 9 noise


def test_noisy_decline_is_recovered_within_the_monte_carlo_tolerances() -> None:
    amps = decline(0.02)
    expected = {
        "amp_slope": ts_norm(PK, amps),
        "amp_late_early": fb.late_early(amps)[0],
        "amp_min_initial": fb.min_initial(amps)[0],
        "vel_slope": ts_norm(PK, amps * np.pi / P),
    }
    found = detected = 0
    for seed in SEEDS:
        t, raw, _ = gen(np.random.default_rng(seed), amps=amps, noise=SIGMA)
        result, table, _, b = chain(t, raw, 30.0)
        value, _, _ = smoothing.smooth_series(t, raw)
        assert not np.allclose(value, raw)  # smoothing is applied
        assert not np.allclose(result.table["amplitude"], result.table["amplitude_raw"])
        if len(table) != N:
            continue  # evaluate only the runs that find all 29 taps
        detected += 1
        assert abs(b["amp_slope"] - expected["amp_slope"]) <= TOL["amp_slope"], seed
        assert abs(b["amp_late_early"] - expected["amp_late_early"]) <= TOL["amp_late_early"]
        assert abs(b["amp_min_initial"] - expected["amp_min_initial"]) <= TOL["amp_min_initial"]
        assert abs(b["open_vel_slope"] - expected["vel_slope"]) <= TOL["vel_slope"], seed
        assert abs(b["close_vel_slope"] - expected["vel_slope"]) <= TOL["vel_slope"], seed
        found += b["decrement_present"] is True
    assert detected >= 19
    assert found >= detected - 1


def test_noisy_controls_show_no_decrement() -> None:
    for seed in SEEDS:
        t, raw, _ = gen(np.random.default_rng(seed), noise=SIGMA)
        _, table, _, b = chain(t, raw, 30.0)
        assert b["decrement_present"] is False, seed
        assert abs(b["amp_slope"]) <= 0.05, seed


# ----------------------------------------------------------------------- 10-12 edge cases


def test_nan_gap_of_0_4_s_at_a_trough() -> None:
    c = LEAD + 11 * P  # the trough between taps 10 and 11
    t, raw, _ = gen(np.random.default_rng(0), nan_runs=((c - 0.2, c + 0.2),))
    assert np.isnan(raw).sum() == 12  # 12 frames
    _, table, a, b = chain(t, raw, 30.0)
    assert len(table) == N and int(table["gap_interval"].sum()) == 1
    assert a["n_iti_excluded"] == 1 and a["iti_mean_s"] == pytest.approx(P, abs=1e-6)
    assert b["n_hesitations"] == 0 and b["n_halts"] == 0


def test_first_tap_cut_mid_rise_is_one_sided() -> None:
    # tap 0 deliberately carries another amplitude (3.0 against 1.2), so any use of it shows:
    # excluded, the series is flat (slope 0, late/early 1, min/initial 1)
    t, raw, _ = gen(np.random.default_rng(0), amps=[3.0] + [A0] * (N - 1))
    t, raw = crop(t, raw, LEAD + 0.12)  # the first rise has begun when the signal starts
    _, table, a, b = chain(t, raw, 30.0)
    assert len(table) == N and bool(table["one_sided"].iloc[0])
    assert b["n_valid_taps"] == 28 and a["n_one_sided_excluded"] == 1
    assert b["amp_slope"] == pytest.approx(0.0, abs=0.005)
    assert b["amp_late_early"] == pytest.approx(1.0, abs=0.005)
    assert b["amp_min_initial"] == pytest.approx(1.0, abs=0.005)
    assert (
        b["amp_late_early_sens"] < 0.9
    )  # the sensitivity set keeps tap 0: (3 + 8 x 1.2) / 9 first


def test_seven_taps_give_nan_and_reasons_never_zero() -> None:
    _, table, _, b, _ = run(n=7)
    assert len(table) == 7
    for name in fb.NUMERIC:
        assert np.isnan(b[name]), name
    assert b["decrement_present"] is None  # unknown, not False
    reasons = b["nan_reasons"].split(";")
    assert "amp_slope:n_valid<8" in reasons and "decrement_present:n_valid<8" in reasons


# ----------------------------------------------------------------------- 13 CLI


def test_the_cli_path_matches_the_in_memory_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    amps = decline(0.02)
    t, raw, _ = gen(np.random.default_rng(0), amps=amps)
    result, _, row_a, row_b = chain(t, raw, 30.0)
    interim, docs = tmp_path / "interim", tmp_path / "docs"
    (interim / "taps").mkdir(parents=True)
    docs.mkdir()
    meta = {"tool": taps.TOOL_VERSION, "params": taps.PARAMS.as_metadata(), "status": result.status}
    taps.write_taps(result, interim / "taps" / "aaaa1111.parquet", meta)
    pd.DataFrame([taps.summary_row(result.stats, "retained", "")]).to_csv(
        interim / "taps" / "taps_summary.csv", index=False
    )
    pd.DataFrame({"ref": ["aaaa1111"], "verdict": ["ok"], "note": [""]}).to_csv(
        docs / "qc_retained.csv", index=False
    )
    pd.DataFrame({"ref": ["bbbb2222"], "reason": ["x"]}).to_csv(
        docs / "qc_rejections.csv", index=False
    )
    for module in (fa, fb):
        monkeypatch.setattr(module, "INTERIM_ROOT", interim)
        monkeypatch.setattr(module, "DOCS_ROOT", docs)
    args = ["--taps-dir", str(interim / "taps"), "--out-dir", str(interim / "features"),
            "--docs-dir", str(docs)]  # fmt: skip
    assert fa.main(args) == 0 and fb.main(args) == 0
    for name, row in (("features_a.csv", row_a), ("features_b.csv", row_b)):
        csv = pd.read_csv(interim / "features" / name).iloc[0]
        assert list(csv.index) == list(row)
        for key, want in row.items():
            got = csv[key]
            if isinstance(want, float | int | np.floating) and not isinstance(want, bool):
                assert got == pytest.approx(want, rel=1e-9, abs=1e-9, nan_ok=True), key
            elif want is None:
                assert pd.isna(got), key
            elif isinstance(want, bool):
                assert bool(got) is want, key
            else:
                assert (got == want) or (pd.isna(got) and want == ""), key


# ----------------------------------------------------------------------- 14 leakage


def test_the_synthetic_files_name_no_label_or_identity_module() -> None:
    # the words are split here so that this file does not contain them itself
    parts = ["fis_diag" + "nostic", "UPD" + "RS", "hubu_" + "fis", "clip_" + "id", "pseudo" + "nym"]
    pattern = r"\b(" + "|".join([*parts, "CONT" + "ROL"]) + r")\b"
    for name in ("synthetic_tapping.py", "test_feature_synthetic.py"):
        assert not re.search(pattern, (HERE / name).read_text(), re.IGNORECASE), name
