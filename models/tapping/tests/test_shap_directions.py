"""Tests for models/tapping/src/shap_directions.py (task FT-46, code part). Hand-computed rows
first, then synthetic xai worlds (no real data) for the files and guards."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv_splits as cvs  # noqa: E402
import shap_directions as sd  # noqa: E402
import shap_figures as sf  # noqa: E402
import treeshap_heldout as tsh  # noqa: E402
from test_shap_figures import COLS, make_world, values_table  # noqa: E402

LABEL_WORDS = re.compile(r"rating|updrs|label|group|diagnos|detect|control", re.I)
N12 = 12


@pytest.fixture(autouse=True)
def configs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sd, "CONFIGS", (("ge2", "A"),))


def table12() -> pd.DataFrame:
    """12 clips, 5 repeats; value rises with the clip index, SHAP follows it with a sign per
    feature (one feature has an unrelated pattern); one NaN value in 'amp_mean'."""
    signs = {"tap_rate_hz": 1.0, "iti_cv": -1.0, "open_vel_peak": 1.0, "close_vel_peak": -1.0}
    rows = []
    for repeat in range(5):
        for i in range(N12):
            row: dict[str, Any] = {"repeat": repeat, "outer_fold": (i + repeat) % 5,
                                   "clip_ref": f"{0x30 + i:08x}", "participant_ref": f"{0x80 + i // 2:08x}",
                                   "base_value": -1.0 + 0.01 * repeat}  # fmt: skip
            total = 0.0
            for j, c in enumerate(COLS):
                shap_v = (
                    signs[c] * (i - 5.5) / 5.5 + 0.01 * repeat
                    if c in signs
                    else float(np.sin(2.1 * i + 0.7))
                )
                row[f"shap_{c}"] = shap_v
                total += shap_v
                row[f"value_{c}"] = (
                    np.nan if (c == "amp_mean" and i == 3) else (i + 1) * (j + 1) * 0.1
                )
            row["raw_margin"] = row["base_value"] + total
            row["proba"] = float(1 / (1 + np.exp(-row["raw_margin"])))
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def w(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    return make_world(tmp_path, monkeypatch, table12())


def go(w: dict[str, Path], *extra: str) -> int:
    return sd.main(["--cv-root", str(w["cv"]), "--xai-root", str(w["xai"]), *extra])


def out(w: dict[str, Path]) -> Path:
    return w["xai"] / "directions" / "ge2" / "A"


def read(w: dict[str, Path]) -> pd.DataFrame:
    return pd.read_csv(out(w) / sd.OUT_FILE, keep_default_na=True)


def row_of(value: Any, shap: Any) -> dict[str, Any]:
    return sd.direction_row("f", 1, 0.5, np.array(value, dtype=float), np.array(shap, dtype=float))


# --------------------------------------------------------------------------- constants


def test_preregistered_constants() -> None:
    source = Path(sd.__file__).read_text()
    assert 'CONFIGS = (("ge2", "A"), ("ge1", "A"), ("ge2", "B"))' in source
    assert sd.RHO_CUT == 0.30 and sd.MIN_TERTILE_CLIPS == 9
    assert (
        "pyplot" not in source and "features.csv" not in source and "fis_diagnostic" not in source
    )


# --------------------------------------------------------------------------- rho and direction


def test_rho_by_hand_including_ties() -> None:
    r = row_of([1.0, 2.0, 2.0, 4.0], [1.0, 3.0, 2.0, 4.0])
    assert r["rho_value_vs_shap"] == pytest.approx(4.5 / np.sqrt(4.5 * 5.0))
    assert r["rho_value_vs_shap"] == pytest.approx(
        float(spearmanr([1, 2, 2, 4], [1, 3, 2, 4]).statistic)
    )
    assert r["direction"] == "up"


def test_spearman_not_pearson() -> None:
    value, shap = [1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 100.0]
    assert row_of(value, shap)["rho_value_vs_shap"] == pytest.approx(1.0)
    assert np.corrcoef(value, shap)[0, 1] < 0.99


def test_the_cut_counts_at_exactly_0_30_in_both_directions() -> None:
    assert sd.direction_of(0.30) == "up" and sd.direction_of(-0.30) == "down"
    assert sd.direction_of(0.2999999) == "unclear" and sd.direction_of(-0.2999999) == "unclear"
    assert sd.direction_of(0.3000001) == "up" and sd.direction_of(-0.3000001) == "down"
    assert sd.direction_of(0.0) == "unclear" and sd.direction_of(sd.RHO_CUT) == "up"
    assert sd.direction_of(float("nan")) == ""


def test_a_constant_value_gives_an_empty_rho_and_direction_never_zero() -> None:
    r = row_of([2.0] * 10, np.arange(10.0))
    assert np.isnan(r["rho_value_vs_shap"]) and r["direction"] == ""
    constant_shap = row_of(np.arange(10.0), [1.0] * 10)
    assert np.isnan(constant_shap["rho_value_vs_shap"]) and constant_shap["direction"] == ""
    assert np.isnan(row_of([1.0], [1.0])["rho_value_vs_shap"])


def test_nan_values_are_excluded_from_rho_and_reported_separately() -> None:
    value = [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 3.0]
    shap = [0.1, 0.2, -0.5, 0.4, 0.5, -0.3, 0.3]
    r = row_of(value, shap)
    keep = ~np.isnan(np.array(value))
    assert r["rho_value_vs_shap"] == pytest.approx(
        float(spearmanr(np.array(value)[keep], np.array(shap)[keep]).statistic)
    )
    assert r["n_clips_value"] == 5 and r["n_clips_nan"] == 2
    assert r["mean_shap_nan"] == pytest.approx(-0.4) and r["mean_shap_non_nan"] == pytest.approx(
        0.3
    )
    assert r["rho_value_vs_shap"] != pytest.approx(
        float(spearmanr(np.nan_to_num(value), shap).statistic)
    )  # NaN is not 0


def test_no_nan_clips_leave_the_nan_means_empty() -> None:
    r = row_of([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
    assert (
        np.isnan(r["mean_shap_nan"]) and np.isnan(r["mean_shap_non_nan"]) and r["n_clips_nan"] == 0
    )


def test_share_positive_by_hand() -> None:
    assert row_of([1, 2, 3, 4], [0.5, -0.5, 0.0, 2.0])["share_positive_shap"] == pytest.approx(
        0.5
    )  # 0.0 is not positive
    assert row_of([1, 2], [-1.0, -2.0])["share_positive_shap"] == 0.0


# --------------------------------------------------------------------------- tertiles


def test_tertile_means_by_hand() -> None:
    value = np.arange(1.0, 10.0)
    shap = np.array([10.0, 20, 30, 40, 50, 60, 70, 80, 90])
    low, high = sd.tertile_means(value, shap)
    q1, q2 = np.quantile(value, [1 / 3, 2 / 3])  # 3.667 and 6.333
    assert (low, high) == pytest.approx((np.mean([10, 20, 30]), np.mean([70, 80, 90])))
    assert q1 == pytest.approx(3.6667, abs=1e-3) and q2 == pytest.approx(6.3333, abs=1e-3)


def test_a_tie_at_the_cut_goes_to_the_lower_group() -> None:
    value = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    shap = np.array([10.0, 20, 30, 40, 50, 60, 70, 80, 90])
    low, high = sd.tertile_means(value, shap)
    assert np.quantile(value, 1 / 3) == 1.0  # the low cut sits on the tied value
    assert low == pytest.approx(30.0)  # all five tied clips are in the low group
    assert high == pytest.approx(
        80.0
    )  # values above the high cut (2.33): 3, 4, 5 -> SHAP 70, 80, 90
    assert high != pytest.approx(30.0)


def test_low_and_high_are_not_swapped() -> None:
    value = np.arange(1.0, 13.0)
    shap = -value  # SHAP falls as the value rises
    low, high = sd.tertile_means(value, shap)
    assert low > high


def test_fewer_than_nine_values_leave_the_tertile_cells_empty() -> None:
    low, high = sd.tertile_means(np.arange(8.0), np.arange(8.0))
    assert np.isnan(low) and np.isnan(high)
    low, high = sd.tertile_means(np.arange(9.0), np.arange(9.0))
    assert not np.isnan(low) and not np.isnan(high)
    r = row_of([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], np.arange(10.0))  # 9 non-NaN
    assert not np.isnan(r["mean_shap_value_low_tertile"])
    r8 = row_of(
        [1.0, 2.0, np.nan, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], np.arange(10.0)
    )  # 8 non-NaN
    assert np.isnan(r8["mean_shap_value_low_tertile"]) and np.isnan(
        r8["mean_shap_value_high_tertile"]
    )


def test_tertiles_use_only_non_nan_values() -> None:
    value = np.array([np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    shap = np.array([1000.0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    r = row_of(value, shap)
    assert r["mean_shap_value_low_tertile"] == pytest.approx(
        20.0
    )  # the NaN clip's 1000 is not in it
    assert r["mean_shap_value_high_tertile"] == pytest.approx(80.0)


# --------------------------------------------------------------------------- the table


def test_rows_follow_the_global_rank_order() -> None:
    glob = pd.DataFrame({"feature": ["x", "y", "z"], "rank": [2, 3, 1],
                         "mean_abs_shap_mean_over_repeats": [0.5, 0.1, 0.9]})  # fmt: skip
    clips = pd.DataFrame({f"{k}_{f}": np.arange(10.0) for f in "xyz" for k in ("value", "shap")})
    table = sd.direction_table(glob, clips)
    assert list(table["feature"]) == ["z", "x", "y"]  # rank order, not alphabetical
    assert list(table["global_rank"]) == [1, 2, 3]
    assert list(table["mean_abs_shap"]) == [0.9, 0.5, 0.1] and list(table.columns) == sd.COLUMNS


def test_world_file_matches_an_independent_recomputation(w: dict[str, Path]) -> None:
    assert go(w) == 0
    table = read(w)
    data = sf.read_inputs(w["in"], COLS)
    order = sf.feature_order(data["glob"])
    assert list(table["feature"]) == order and list(table.columns) == sd.COLUMNS
    clips = sf.per_clip_means(data["values"], COLS)
    assert len(clips) == N12
    for r in table.itertuples(index=False):
        value, shap = clips[f"value_{r.feature}"].to_numpy(), clips[f"shap_{r.feature}"].to_numpy()
        keep = ~np.isnan(value)
        rho = float(spearmanr(value[keep], shap[keep]).statistic) if np.ptp(value[keep]) else np.nan
        assert (
            np.isnan(r.rho_value_vs_shap) and np.isnan(rho)
        ) or r.rho_value_vs_shap == pytest.approx(rho)
        assert r.n_clips_value == keep.sum() and r.n_clips_nan == (~keep).sum()
        assert r.share_positive_shap == pytest.approx((shap > 0).mean())
    by_feature = table.set_index("feature")
    assert (
        by_feature.loc["tap_rate_hz", "direction"] == "up"
        and by_feature.loc["iti_cv", "direction"] == "down"
    )
    assert (
        by_feature.loc["open_vel_peak", "direction"] == "up"
        and by_feature.loc["close_vel_peak", "direction"] == "down"
    )
    assert by_feature.loc["amp_mean", "n_clips_nan"] == 1 and not np.isnan(
        by_feature.loc["amp_mean", "mean_shap_nan"]
    )
    assert np.isnan(by_feature.loc["tap_rate_hz", "mean_shap_nan"])  # no NaN clips: empty
    assert (
        by_feature.loc["tap_rate_hz", "mean_shap_value_high_tertile"]
        > by_feature.loc["tap_rate_hz", "mean_shap_value_low_tertile"]
    )


def test_a_world_with_fewer_than_nine_clips_leaves_tertiles_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    w2 = make_world(tmp_path, monkeypatch, values_table())  # 8 clips
    assert go(w2) == 0
    table = read(w2)
    assert (
        table["mean_shap_value_low_tertile"].isna().all()
        and table["mean_shap_value_high_tertile"].isna().all()
    )
    assert table["rho_value_vs_shap"].notna().all()


def test_meta_and_console(w: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    assert go(w) == 0
    console = capsys.readouterr().out
    meta = json.loads((out(w) / sd.META_FILE).read_text())
    assert meta["constants"]["rho_cut"] == 0.3 and meta["n_clips"] == N12
    assert set(meta["inputs_sha8"]) == set(sf.SHAP_INPUTS) and set(meta["versions"]) == {
        "numpy",
        "scipy",
    }
    assert meta["files_sha8"][sd.OUT_FILE] == cvs.sha8_of((out(w) / sd.OUT_FILE).read_bytes())
    assert "tap_rate_hz" in console and "not cause" in console
    assert not any(
        ref in console
        for ref in sf.per_clip_means(sf.read_inputs(w["in"], COLS)["values"], COLS)["clip_ref"]
    )


# --------------------------------------------------------------------------- files


def test_deterministic_private_and_sources_untouched(
    w: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    def snap() -> dict[str, str]:
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(w["proc"].rglob("*")) if p.is_file() and "directions" not in p.parts}  # fmt: skip

    before = snap()
    assert go(w) == 0
    first = {n: (out(w) / n).read_bytes() for n in (sd.OUT_FILE, sd.META_FILE)}
    console = capsys.readouterr().out
    assert go(w) == 0
    assert {
        n: (out(w) / n).read_bytes() for n in first
    } == first and capsys.readouterr().out == console
    assert snap() == before
    text = console + "".join(v.decode() for v in first.values())
    assert not LABEL_WORDS.search(text) and not re.search(
        r"CONTROL|ID\d|DCHA|IZDA|parquet|/Users", text
    )
    assert not re.search(r"20\d\d-\d\d-\d\dT|\d\d:\d\d:\d\d", text)
    assert sorted(p.name for p in out(w).iterdir()) == sorted([sd.OUT_FILE, sd.META_FILE])


def test_failed_atomic_write_keeps_the_old_file(
    w: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert go(w) == 0
    (out(w) / sd.OUT_FILE).write_bytes(b"old")

    def boom(a: Any, b: Any) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(cvs.os, "replace", boom)
    with pytest.raises(OSError):
        go(w)
    assert (out(w) / sd.OUT_FILE).read_bytes() == b"old"
    assert not [p for p in out(w).iterdir() if p.name.startswith(".")]


# --------------------------------------------------------------------------- guards


def edit_json(path: Path, fn: Any) -> None:
    data = json.loads(path.read_text())
    fn(data)
    path.write_text(json.dumps(data))


def test_oof_not_reproduced_exits_1(w: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    edit_json(w["in"] / tsh.META_FILE, lambda m: m.update(oof_reproduced=False))
    assert go(w) == 1 and "did not reproduce" in capsys.readouterr().err and not out(w).exists()


@pytest.mark.parametrize("key", ["features_sha8", "grid_sha8", "splits_sha8"])
def test_hash_differing_from_the_official_meta_exits_1(w: dict[str, Path], key: str) -> None:
    edit_json(w["in"] / tsh.META_FILE, lambda m: m.update({key: "deadbeef"}))
    assert go(w) == 1 and not out(w).exists()


def test_row_count_mismatch_and_wrong_tool_exit_1(w: dict[str, Path]) -> None:
    edit_json(w["in"] / tsh.META_FILE, lambda m: m["row_counts"].update({tsh.SHAP_FILE: 1}))
    assert go(w) == 1
    edit_json(w["in"] / tsh.META_FILE, lambda m: m.update(row_counts={}, tool="other"))
    assert go(w) == 1


def test_a_clip_missing_a_repeat_exits_1(w: dict[str, Path]) -> None:
    path = w["in"] / tsh.SHAP_FILE
    table = pd.read_csv(path).iloc[1:]
    table.to_csv(path, index=False, float_format="%.17g")
    edit_json(
        w["in"] / tsh.META_FILE, lambda m: m["row_counts"].update({tsh.SHAP_FILE: len(table)})
    )
    assert go(w) == 1


def test_missing_inputs_exit_2(w: dict[str, Path]) -> None:
    (w["in"] / tsh.GLOBAL_FILE).unlink()
    assert go(w) == 2
    (w["in"] / tsh.GLOBAL_FILE).write_text("x")
    (w["cv"] / "ge2" / "meta_lgbm_A.json").unlink()
    assert go(w) == 2


def test_xai_root_outside_the_xai_folder_exits_2(w: dict[str, Path], tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    shutil.copytree(w["xai"], elsewhere)  # complete inputs: only the path guard can stop the run
    assert go(w, "--xai-root", str(elsewhere)) == 2
    assert not (elsewhere / "directions").exists()
