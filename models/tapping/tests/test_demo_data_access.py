"""Tests for models/tapping/demo/data_access.py (task FT-47). Synthetic xai worlds only (the
world builder of test_shap_figures); hand-computed means and the guards."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "src"))
sys.path.insert(0, str(HERE.parents[0] / "demo"))
sys.path.insert(0, str(HERE))

import data_access as da  # noqa: E402
import shap_directions as sd  # noqa: E402
import treeshap_heldout as tsh  # noqa: E402
from cv_splits import GuardFailure  # noqa: E402
from test_shap_figures import COLS, REFS, make_world, values_table  # noqa: E402

HEX8 = re.compile(r"^[0-9a-f]{8}$")


@pytest.fixture
def w(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    return make_world(tmp_path, monkeypatch)


def test_clip_list_is_sorted_and_hashed(w: dict[str, Path]) -> None:
    refs = da.available_clips("ge2", w["proc"])
    assert refs == sorted(REFS) and all(HEX8.match(r) for r in refs)


def test_clip_summary_means_over_all_five_repeats(w: dict[str, Path]) -> None:
    table = values_table()
    ref = REFS[2]
    sel = table[table["clip_ref"] == ref]
    got = da.clip_summary("ge2", ref, w["proc"])
    assert got["proba"] == pytest.approx(sel["proba"].sum() / 5, abs=1e-15)
    assert got["proba"] != pytest.approx(sel["proba"].iloc[0], abs=1e-6)  # not repeat 0
    assert got["base_value"] == pytest.approx(-1.0 + 0.05 * 2)
    for c in COLS:
        assert got["shap"][c] == pytest.approx(sel[f"shap_{c}"].sum() / 5, abs=1e-15)
        assert got["value"][c] == pytest.approx(sel[f"value_{c}"].iloc[0], abs=1e-15)


def test_base_plus_shap_equals_mean_margin(w: dict[str, Path]) -> None:
    for ref in REFS:
        s = da.clip_summary("ge2", ref, w["proc"])
        assert s["base_value"] + sum(s["shap"].values()) == pytest.approx(s["raw_margin"], abs=1e-9)


def test_missing_value_stays_nan_not_zero(w: dict[str, Path]) -> None:
    s = da.clip_summary("ge2", REFS[1], w["proc"])  # iti_cv is NaN for clips 1 and 4
    assert pd.isna(s["value"]["iti_cv"]) and s["value"]["tap_rate_hz"] == s["value"]["tap_rate_hz"]


def test_guard_oof_not_reproduced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    w = make_world(tmp_path, monkeypatch, oof_reproduced=False)
    with pytest.raises(GuardFailure, match="reproduce"):
        da.available_clips("ge2", w["proc"])


def test_guard_sha_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    w = make_world(tmp_path, monkeypatch, grid_sha8="deadbeef")
    with pytest.raises(GuardFailure, match="grid_sha8"):
        da.clip_summary("ge2", REFS[0], w["proc"])


def test_guard_row_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    w = make_world(tmp_path, monkeypatch)
    meta = json.loads((w["in"] / tsh.META_FILE).read_text())
    meta["row_counts"][tsh.SHAP_FILE] += 1
    (w["in"] / tsh.META_FILE).write_text(json.dumps(meta))
    with pytest.raises(GuardFailure, match="row count"):
        da.available_clips("ge2", w["proc"])


def test_guard_unhashed_clip_ref(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    table = values_table()
    table["clip_ref"] = table["clip_ref"].map(lambda r: "CONTROL" + r[-2:])
    w = make_world(tmp_path, monkeypatch, table=table)
    with pytest.raises(GuardFailure, match="8-hex"):
        da.available_clips("ge2", w["proc"])


def test_missing_file_is_a_friendly_error(w: dict[str, Path]) -> None:
    (w["in"] / tsh.GLOBAL_FILE).unlink()
    with pytest.raises(da.DemoDataError, match="missing input"):
        da.available_clips("ge2", w["proc"])
    with pytest.raises(da.DemoDataError, match="missing input"):
        da.cache_key("ge2", w["proc"])


def test_unknown_target_clip_and_absent_target(w: dict[str, Path]) -> None:
    with pytest.raises(da.DemoDataError, match="unknown target"):
        da.available_clips("ge3", w["proc"])
    with pytest.raises(da.DemoDataError, match="unknown clip"):
        da.clip_summary("ge2", "12345678", w["proc"])
    with pytest.raises(da.DemoDataError, match="missing input"):
        da.available_clips("ge1", w["proc"])  # the synthetic world only has ge2


def test_label_columns_never_returned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    table = values_table()
    table["updrs_rating"] = 3
    table["group"] = "ID"
    w = make_world(tmp_path, monkeypatch, table=table)
    s = da.clip_summary("ge2", REFS[0], w["proc"])
    flat = json.dumps(s, default=str).lower()
    assert set(s) == {"clip_ref", "proba", "base_value", "raw_margin", "shap", "value"}
    assert set(s["shap"]) == set(COLS) == set(s["value"])
    assert not re.search(r"rating|updrs|group|participant", flat)
    assert not re.search(
        r"rating|updrs|group|participant", json.dumps(da.global_importance("ge2", w["proc"]))
    )


def test_global_importance_follows_rank(w: dict[str, Path]) -> None:
    rows = da.global_importance("ge2", w["proc"])
    assert [r["rank"] for r in rows] == sorted(r["rank"] for r in rows)
    assert {r["feature"] for r in rows} == set(COLS)


def test_cache_key_contains_sha8_and_changes_with_the_file(w: dict[str, Path]) -> None:
    before = da.cache_key("ge2", w["proc"])
    assert all(HEX8.match(sha) for _, sha in before)
    path = w["in"] / tsh.GLOBAL_FILE
    path.write_bytes(path.read_bytes() + b" ")
    after = da.cache_key("ge2", w["proc"])
    assert after != before
    assert {n for (n, a), (_, b) in zip(after, before, strict=True) if a != b} == {tsh.GLOBAL_FILE}


def test_directions_read_and_guard(w: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(da.DemoDataError, match="directions"):
        da.directions("ge2", w["proc"])
    monkeypatch.setattr(sd, "CONFIGS", (("ge2", "A"),))
    sd.run_config(w["cv"], w["xai"], "ge2", "A")
    rows = da.directions("ge2", w["proc"])
    assert {r["feature"] for r in rows} == set(COLS)
    assert all(r["direction"] in {"up", "down", "unclear"} for r in rows)
    shap_file = w["in"] / tsh.SHAP_FILE  # SHAP inputs change after the directions were made
    shap_file.write_bytes(shap_file.read_bytes() + b"\n")
    with pytest.raises(GuardFailure):
        da.directions("ge2", w["proc"])


def test_module_has_no_streamlit_import() -> None:
    source = (HERE.parent / "demo" / "data_access.py").read_text()
    assert not re.search(r"^\s*(import|from)\s+streamlit", source, re.M)


def test_guard_additivity_broken(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    table = values_table()
    table["raw_margin"] += 0.01  # base + SHAP no longer adds up to the model margin
    w = make_world(tmp_path, monkeypatch, table=table)
    with pytest.raises(GuardFailure, match="base value plus SHAP"):
        da.clip_summary("ge2", REFS[0], w["proc"])
