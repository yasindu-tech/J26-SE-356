"""Tests for models/tapping/src/features_b.py (task FT-28, Set B). Hand-computed small cases on
synthetic tables; the last test checks the real frozen taps and skips when they are absent."""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from scipy.stats import theilslopes

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import features_a as fa  # noqa: E402
import features_b as fb  # noqa: E402
import tap_table as tt  # noqa: E402
from test_features_a import PARAMS, REFS, STALE, world, write_taps  # noqa: E402, F401
from test_tap_table import make_taps  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src"
PINNED_DEFINITIONS_SHA256 = "65bce75bf4201b40749c6d3d4224e1a22c949e403ba90ea961b1753f7d2e4f5b"


def steady(n: int = 12, **kw: Any) -> pd.DataFrame:
    return tt.build_tap_table(make_taps([0.5 * (k + 1) for k in range(n)], **kw))


# ----------------------------------------------------------------------- slopes


def test_slope_of_a_linear_decline_by_hand() -> None:
    t = np.array([0, 1, 2, 4, 5, 7, 8, 9, 12, 15], float)
    y = 10 - 0.5 * t  # sum 68.5, mean 6.85, slope -0.5 per s
    value, reason = fb.slope_per_10s(t, y)
    assert reason is None
    assert value == pytest.approx(-0.5 / 6.85 * 10)  # -0.729927
    assert value == pytest.approx(-0.72992701)


def test_slope_uses_time_not_tap_index() -> None:
    t = np.array([0, 1, 2, 4, 5, 7, 8, 9, 12, 15], float)
    y = 10 - 0.5 * t
    by_index = float(theilslopes(y, np.arange(10))[0]) / y.mean() * 10
    assert fb.slope_per_10s(t, y)[0] != pytest.approx(by_index)


def test_slope_is_scale_free() -> None:
    t = np.arange(10.0)
    y = 10 - 0.4 * t
    assert fb.slope_per_10s(t, 7 * y)[0] == pytest.approx(fb.slope_per_10s(t, y)[0])


def test_theil_sen_survives_one_bad_tap_where_ols_does_not() -> None:
    t = np.arange(10.0)
    y = 10 - 0.5 * t
    y[4] = 100.0
    value, _ = fb.slope_per_10s(t, y)
    assert value * y.mean() / 10 == pytest.approx(-0.5)  # the raw Theil-Sen slope is intact
    ols = np.polyfit(t, y, 1)[0]
    assert abs(ols + 0.5) > 0.5  # OLS is pulled off by more than the true slope itself


def test_constant_amplitude_is_flat() -> None:
    y, t = np.full(12, 2.0), np.arange(12.0)
    assert fb.slope_per_10s(t, y) == (0.0, None)
    assert fb.late_early(y) == (1.0, None)
    assert fb.min_initial(y) == (1.0, None)
    assert fb.decrement(y) == (False, (pytest.approx(np.nan, nan_ok=True), "no_decrement"))


def test_slope_minimum_and_degenerate_mean() -> None:
    t = np.arange(8.0)
    assert fb.slope_per_10s(t, 1.0 + t)[1] is None
    value, reason = fb.slope_per_10s(t[:7], np.ones(7))
    assert np.isnan(value) and reason == "n_valid<8"
    assert fb.slope_per_10s(t, np.zeros(8))[1] == "mean<=0"


# ----------------------------------------------------------------------- late/early, min/initial


@pytest.mark.parametrize(("n", "expected"), [(9, 4.0), (10, 4.5), (11, 5.0), (12, 4.2)])
def test_late_early_uses_k_equal_n_over_3(n: int, expected: float) -> None:
    value, reason = fb.late_early(np.arange(1.0, n + 1))
    assert reason is None and value == pytest.approx(expected)
    # n=9: first 3 mean 2, last 3 mean 8; n=10: 2 and 9; n=11: 2 and 10; n=12: k=4, 2.5 and 10.5


def test_late_early_minimum_is_nine() -> None:
    value, reason = fb.late_early(np.arange(1.0, 9))
    assert np.isnan(value) and reason == "n_valid<9"
    assert fb.late_early(np.array([0.0] * 3 + [1.0] * 6))[1] == "initial<=0"


def test_min_initial_ignores_one_outlier_tap_and_the_edges() -> None:
    mid = np.array([10, 10, 10, 1, 10, 10, 10, 10], float)
    assert fb.min_initial(mid)[0] == pytest.approx(1.0)  # raw min would give 0.1
    edge = np.array([10, 10, 10, 10, 10, 10, 10, 1], float)
    assert fb.min_initial(edge)[0] == pytest.approx(1.0)  # interior points only (min_periods 3)
    drop = np.array([10, 10, 10, 8, 4, 4, 4, 4], float)
    assert fb.min_initial(drop)[0] == pytest.approx(0.4)  # medians of 3: ..., 4, 4, 4
    value, reason = fb.min_initial(drop[:7])
    assert np.isnan(value) and reason == "n_valid<8"


# ----------------------------------------------------------------------- decrement


def test_step_decline_has_a_known_onset() -> None:
    y = np.array([10] * 6 + [5] * 6, float)
    present, (onset, reason) = fb.decrement(y)
    assert present is True and reason is None
    assert onset == pytest.approx(0.5)  # index 6 of 12


def test_decrement_needs_three_consecutive_low_taps() -> None:
    dip = np.array([10] * 4 + [5, 5] + [10] * 6, float)
    assert fb.decrement(dip)[0] is False
    # the two highs between the lows survive the rolling median, so the first run of three
    # starts at index 10; the mutant that tests only one tap would start at index 6
    y = np.array([10] * 6 + [5, 5, 10, 10] + [5] * 12, float)
    present, (onset, _) = fb.decrement(y)
    assert present is True and onset == pytest.approx(10 / 22)


def test_decrement_needs_eighty_percent_of_the_rest_to_stay_low() -> None:
    recovery = np.array([10] * 4 + [5] * 4 + [10] * 4, float)  # 4 low taps, then recovery
    assert fb.decrement(recovery)[0] is False
    edge_case = np.array([10] * 4 + [5] * 8 + [10] * 2, float)  # 8 of the last 10 = exactly 0.8
    present, (onset, _) = fb.decrement(edge_case)
    assert present is True and onset == pytest.approx(4 / 14)


def test_decrement_threshold_is_eighty_percent_of_the_initial_level() -> None:
    assert fb.decrement(np.array([10] * 6 + [7.5] * 6, float))[0] is True  # 7.5 < 8
    assert fb.decrement(np.array([10] * 6 + [8.5] * 6, float))[0] is False


def test_decrement_rolling_median_keeps_the_edge_points() -> None:
    y = np.array([10] * 9 + [1, 1, 1], float)  # min_periods 1: the last point still counts
    present, (onset, _) = fb.decrement(y)
    assert present is True and onset == pytest.approx(9 / 12)


def test_no_decrement_is_nan_with_a_reason_and_below_minimum_is_unknown() -> None:
    present, (onset, reason) = fb.decrement(np.full(10, 3.0))
    assert present is False and np.isnan(onset) and reason == "no_decrement"
    present, (onset, reason) = fb.decrement(np.full(7, 3.0))
    assert present is None and np.isnan(onset) and reason == "n_valid<8"


# ----------------------------------------------------------------------- intervals


def intervals(values: list[float]) -> dict[str, tuple[float, str | None]]:
    return fb.interval_features(np.cumsum(values), np.array(values, float))


def test_ratio_boundaries() -> None:
    base = [1.0] * 6
    on = intervals([*base, 1.8, 3.0])  # median 1.0
    assert on["n_hesitations"][0] == 1 and on["n_halts"][0] == 0  # 1.8 is not, 3.0 is a hesitation
    over = intervals([*base, 1.8001, 3.0001])
    assert over["n_hesitations"][0] == 1 and over["n_halts"][0] == 1
    assert over["longest_pause_ratio"][0] == pytest.approx(3.0001)
    assert over["longest_pause_s"][0] == pytest.approx(3.0001)


def test_ratio_is_to_the_median_not_the_mean() -> None:
    f = intervals([1.0] * 7 + [10.0])  # mean 2.125, median 1.0
    assert f["longest_pause_ratio"][0] == pytest.approx(10.0)
    assert f["n_halts"][0] == 1


def test_iti_slope_by_hand() -> None:
    iv = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
    t = np.cumsum(iv)
    f = fb.interval_features(t, np.array(iv))
    expected = float(theilslopes(iv, t)[0]) / np.mean(iv) * 10
    assert f["iti_slope"][0] == pytest.approx(expected) and expected > 0


def test_too_few_intervals_are_nan_with_a_reason() -> None:
    f = intervals([1.0] * 7)
    assert all(np.isnan(v) and r == "n_intervals<8" for v, r in f.values())
    assert set(f) == set(fb.NUMERIC[6:])


# ----------------------------------------------------------------------- one clip


def row_of(table: pd.DataFrame) -> dict[str, Any]:
    return fb.clip_row(table, "aaaa1111", "retained")


def test_gap_intervals_are_excluded_and_long_intervals_kept() -> None:
    times = [0.5 * k for k in range(6)] + [6.5 + 0.5 * k for k in range(6)]
    gap = tt.build_tap_table(make_taps(times, flags={6: "gap_in_interval;long_interval"}))
    row = row_of(gap)
    assert row["n_halts"] == 0 and row["longest_pause_ratio"] == pytest.approx(1.0)
    long_only = row_of(tt.build_tap_table(make_taps(times, flags={6: "long_interval"})))
    assert long_only["n_halts"] == 1 and long_only["longest_pause_ratio"] == pytest.approx(8.0)
    assert long_only["longest_pause_s"] == pytest.approx(4.0)


def test_one_sided_taps_leave_the_primary_set_and_stay_in_the_sensitivity_set() -> None:
    amp = [50.0] + [1.0] * 11
    table = steady(12, amp=amp, flags={0: "one_sided"})
    row = row_of(table)
    assert row["n_valid_taps"] == 11 and row["amp_slope"] == 0.0
    assert row["amp_late_early"] == 1.0
    assert row["amp_late_early_sens"] < 1.0 and row["n_valid_taps"] == 11


def test_short_intervals_are_dropped_only_in_the_sensitivity_set() -> None:
    times = [0.5 * (k + 1) for k in range(12)]
    times.insert(1, times[0] + 0.1)  # the second tap follows the first by 0.1 s
    amp = [1.0] * 13
    amp[1] = 50.0
    table = tt.build_tap_table(make_taps(times, amp=amp))
    row = row_of(table)
    assert row["n_valid_taps"] == 13 and table["short_iti"].sum() == 1
    assert row["amp_late_early"] < 1.0  # first four taps 1, 50, 1, 1 against four 1s
    assert row["amp_late_early_sens"] == pytest.approx(1.0)  # the short-interval tap is dropped
    kept = table[~table["short_iti"]]
    t, y = kept["t_peak_s"].to_numpy(), kept["amplitude"].to_numpy()
    assert row["amp_slope_sens"] == pytest.approx(fb.slope_per_10s(t, y)[0])


def test_below_minimum_is_nan_with_reasons_never_zero() -> None:
    row = row_of(steady(7))
    for name in fb.NUMERIC:
        assert np.isnan(row[name]), name
    assert row["decrement_present"] is None  # unknown, not "absent"
    reasons = row["nan_reasons"].split(";")
    assert "amp_slope:n_valid<8" in reasons and "decrement_present:n_valid<8" in reasons
    assert "n_halts:n_intervals<8" in reasons and "amp_late_early_sens:n_valid<9" in reasons
    eight = row_of(steady(8))  # 8 taps: amplitude features exist, 7 intervals are too few
    assert not np.isnan(eight["amp_slope"]) and np.isnan(eight["amp_late_early"])
    assert np.isnan(eight["n_hesitations"]) and eight["decrement_present"] is False
    assert not np.isnan(row_of(steady(9))["n_halts"])


def test_columns_are_fixed_unique_and_paired() -> None:
    cols = list(fb.COLUMNS)
    assert len(cols) == len(set(cols)) and list(row_of(steady())) == cols
    sens = [c for c in cols if c.endswith("_sens")]
    assert sens == [f + "_sens" for f in fb.NUMERIC[:5]] + ["decrement_present_sens"] + [
        f + "_sens" for f in fb.NUMERIC[5:]
    ]
    assert not any(c.endswith("_s_s") for c in cols)


def test_definitions_are_pinned() -> None:
    text = json.dumps(fb.DEFINITIONS, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(text.encode()).hexdigest() == PINNED_DEFINITIONS_SHA256
    assert fb.definitions_sha8() == PINNED_DEFINITIONS_SHA256[:8]
    assert fb.DEFINITIONS_VERSION == "features-b-1"
    assert fb.DEFINITIONS["iti_slope"]["status"] == "exploratory"
    assert all(d["status"] in ("primary", "exploratory") for d in fb.DEFINITIONS.values())
    assert (fb.MIN_TAPS, fb.MIN_THIRDS, fb.MIN_INTERVALS) == (8, 9, 8)
    assert (fb.DECREMENT_FRACTION, fb.DECREMENT_RUN, fb.DECREMENT_SHARE) == (0.8, 3, 0.8)
    assert (fb.HESITATION_RATIO, fb.HALT_RATIO, fb.PER_SECONDS) == (1.8, 3.0, 10.0)


# ----------------------------------------------------------------------- a made-up taps folder


@pytest.fixture
def bworld(world: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:  # noqa: F811
    monkeypatch.setattr(fb, "INTERIM_ROOT", world["interim"])
    monkeypatch.setattr(fb, "DOCS_ROOT", world["docs"])
    return world


def run_b(w: dict[str, Any]) -> int:
    return fb.main(["--taps-dir", str(w["taps"]), "--out-dir", str(w["out"]),
                    "--docs-dir", str(w["docs"])])  # fmt: skip


def test_end_to_end_rows_stale_file_and_below_minimum(
    bworld: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_b(bworld) == 0
    assert "ignored 1 taps files that are not in the QC retained list" in capsys.readouterr().out
    df = pd.read_csv(bworld["out"] / fb.OUT_FEATURES)
    assert df["ref"].tolist() == list(REFS) and list(df.columns) == list(fb.COLUMNS)
    assert df["qc_status"].tolist() == ["retained", "retained_flag", "retained"]
    assert df["n_valid_taps"].tolist() == [7, 11, 15]  # the first tap of each clip is one-sided
    assert df["amp_slope"].isna().tolist() == [True, False, False]
    assert df["decrement_present"].isna().tolist() == [True, False, False]
    assert "amp_slope:n_valid<8" in df.loc[0, "nan_reasons"]
    assert df.loc[1, "decrement_present"] in (False, "False")


def test_metadata_hashes(bworld: dict[str, Any]) -> None:
    assert run_b(bworld) == 0
    meta = json.loads((bworld["out"] / fb.OUT_META).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:8]  # noqa: E731
    canonical = json.dumps(PARAMS, sort_keys=True, separators=(",", ":"))
    assert meta["definitions_sha8"] == PINNED_DEFINITIONS_SHA256[:8]
    assert meta["definitions_version"] == "features-b-1" and meta["taps_folder"] == "taps"
    assert meta["tap_params_sha8"] == hashlib.sha256(canonical.encode()).hexdigest()[:8]
    assert meta["taps_summary_sha8"] == sha(bworld["taps"] / "taps_summary.csv")
    assert meta["qc_list_sha8"] == sha(bworld["docs"] / "qc_retained.csv")
    assert meta["qc_list_sha8"] != sha(bworld["docs"] / "qc_rejections.csv")
    assert meta["features_tool"] == fb.TOOL_VERSION and meta["n_clips"] == 3


def test_second_run_is_byte_identical_and_features_a_is_untouched(bworld: dict[str, Any]) -> None:
    assert fa.main(["--taps-dir", str(bworld["taps"]), "--out-dir", str(bworld["out"]),
                    "--docs-dir", str(bworld["docs"])]) == 0  # fmt: skip
    a_before = {p.name: p.read_bytes() for p in bworld["out"].glob("*") if "features_a" in p.name}
    table_before = (bworld["out"] / fa.OUT_TABLE).read_bytes()
    assert run_b(bworld) == 0
    first = [(bworld["out"] / n).read_bytes() for n in (fb.OUT_FEATURES, fb.OUT_META)]
    assert run_b(bworld) == 0
    assert [(bworld["out"] / n).read_bytes() for n in (fb.OUT_FEATURES, fb.OUT_META)] == first
    assert {
        p.name: p.read_bytes() for p in bworld["out"].glob("*") if "features_a" in p.name
    } == a_before
    assert (bworld["out"] / fa.OUT_TABLE).read_bytes() == table_before


def test_exit_codes(bworld: dict[str, Any], tmp_path: Path) -> None:
    path = bworld["taps"] / f"{REFS[1]}.parquet"
    df, _ = tt.read_taps_file(path)
    write_taps(path, df, {**PARAMS, "prominence_frac": 0.25})
    assert run_b(bworld) == 1
    write_taps(path, df, PARAMS, tool="taps-0.2.0")
    assert run_b(bworld) == 1
    write_taps(path, df, PARAMS)
    summary = pd.read_csv(bworld["taps"] / "taps_summary.csv")
    summary.loc[0, "n_taps"] += 1
    summary.to_csv(bworld["taps"] / "taps_summary.csv", index=False)
    assert run_b(bworld) == 1
    summary.loc[0, "n_taps"] -= 1
    summary.to_csv(bworld["taps"] / "taps_summary.csv", index=False)
    (bworld["taps"] / f"{REFS[2]}.parquet").unlink()
    assert run_b(bworld) == 2
    assert fb.main(["--out-dir", str(tmp_path / "elsewhere")]) == 2
    assert fb.main(["--docs-dir", str(tmp_path / "elsewhere")]) == 2
    assert not (tmp_path / "elsewhere").exists()


# ----------------------------------------------------------------------- leakage


def test_the_source_names_no_label_or_identity_module() -> None:
    words = r"\b(fis_diagnostic|UPDRS|hubu_fis|clip_id|pseudonym|CONTROL)\b"
    assert not re.search(words, (SRC / "features_b.py").read_text(), re.IGNORECASE)


def test_the_pipeline_reads_nothing_but_its_own_inputs(
    bworld: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    decoy = bworld["root"] / "raw" / "fis_diagnostic.csv"
    decoy.parent.mkdir()
    decoy.write_text("ID,UPDRS\nCONTROL01_DCHA,0\n")
    allowed = [bworld[k].resolve() for k in ("taps", "docs", "out")]
    seen: list[str] = []

    def guard(path: Any) -> None:
        if isinstance(path, str | Path) and str(path):
            p = Path(path).resolve()
            seen.append(p.name)
            if p.suffix not in (".py", ".pyc", ".so") and not any(
                p.is_relative_to(a) for a in allowed
            ):
                raise AssertionError(f"read outside the allowed inputs: {p.name}")

    def wrap(fn: Any) -> Any:
        def inner(path: Any, *a: Any, **k: Any) -> Any:
            guard(path)
            return fn(path, *a, **k)

        return inner

    for owner, name in ((builtins, "open"), (io, "open"), (pd, "read_csv"),
                        (pd, "read_parquet"), (pq, "read_table"), (pq, "read_schema")):  # fmt: skip
        monkeypatch.setattr(owner, name, wrap(getattr(owner, name)))
    assert run_b(bworld) == 0
    assert seen and "fis_diagnostic.csv" not in seen
    with pytest.raises(AssertionError):
        pd.read_csv(decoy)


# ----------------------------------------------------------------------- real frozen taps


def test_real_taps_valid_counts_and_summary_agree(capsys: pytest.CaptureFixture[str]) -> None:
    taps_dir = fb.INTERIM_ROOT / "taps"
    qc = fb.DOCS_ROOT / "qc_retained.csv"
    if not ((taps_dir / "taps_summary.csv").is_file() and qc.is_file()):
        pytest.skip("real data not available")
    summary = pd.read_csv(taps_dir / "taps_summary.csv", dtype={"ref": str}).set_index("ref")
    kept = pd.read_csv(qc, dtype=str)
    kept = kept[kept["verdict"].isin(["ok", "ok_flag"])]["ref"].tolist()
    assert len(kept) == 233
    for ref in kept:
        raw, _ = tt.read_taps_file(taps_dir / f"{ref}.parquet")
        table = tt.build_tap_table(raw)
        row = fb.clip_row(table, ref, "x")
        assert row["n_valid_taps"] == summary.at[ref, "n_taps"] - summary.at[ref, "n_one_sided"]
        assert row["amp_slope"] == row["amp_slope"]  # 8 or more valid taps in every clip
    with capsys.disabled():
        print(f"\nreal taps: {len(kept)} clips, n_valid_taps = n_taps - n_one_sided on all")
