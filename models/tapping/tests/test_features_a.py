"""Tests for models/tapping/src/features_a.py (task FT-27, Set A). Synthetic data; the last
test checks the real frozen taps and skips when they are absent."""

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
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import features_a as fa  # noqa: E402
import tap_table as tt  # noqa: E402
from taps import META_KEY  # noqa: E402
from test_tap_table import make_taps  # noqa: E402

PARAMS = {"prominence_frac": 0.2, "min_spacing_s": 0.15}
SRC = Path(__file__).resolve().parents[1] / "src"


def feats(taps: pd.DataFrame, duration: float = 20.0) -> dict[str, Any]:
    summary = pd.Series({"duration_s": duration, "fps": 30.0, "qc_status": "retained"})
    return fa.clip_features(tt.build_tap_table(taps), "aaaa1111", summary)


def two_hz(n: int = 20) -> list[float]:
    return [0.5 + 0.5 * k for k in range(n)]


# ----------------------------------------------------------------------- definitions


def test_two_hertz_impulses() -> None:
    f = feats(make_taps(two_hz()))
    assert f["tap_rate_hz"] == pytest.approx(2.0)
    assert f["iti_mean_s"] == pytest.approx(0.5)
    assert f["iti_cv"] == pytest.approx(0.0, abs=1e-12)
    assert f["n_taps"] == 20 and f["nan_reasons"] == ""
    assert f["n_over_duration_hz"] == pytest.approx(1.0)  # 20 / 20 s: edge idle time shows up


def test_cv_uses_ddof_1_by_hand() -> None:
    times = list(np.cumsum([1.0, 0.4, 0.6, 0.5, 0.5]))
    f = feats(make_taps(times))
    # intervals 0.4, 0.6, 0.5, 0.5: mean 0.5, sum of squares 0.02, variance 0.02 / 3
    assert f["iti_mean_s"] == pytest.approx(0.5)
    assert f["iti_cv"] == pytest.approx(np.sqrt(0.02 / 3) / 0.5)
    assert f["tap_rate_hz"] == pytest.approx(4 / 2.0)  # (n-1) / (t_last - t_first)


def test_trapezoid_amplitude_and_velocities_are_exact() -> None:
    f = feats(make_taps(two_hz(6), amp=1.5, rise=0.25, fall=0.125))
    assert f["amp_mean"] == f["amp_max"] == 1.5
    assert f["open_vel_mean"] == pytest.approx(6.0) and f["close_vel_mean"] == pytest.approx(12.0)
    assert f["open_vel_peak"] == 9.0 and f["close_vel_peak"] == 11.0  # the parquet's slopes
    assert f["open_vel_peak_median"] == 9.0


def test_mean_and_median_differ_on_skewed_amplitudes() -> None:
    f = feats(make_taps(two_hz(5), amp=[1.0, 1.0, 1.0, 1.0, 6.0]))
    assert f["amp_mean"] == pytest.approx(2.0)
    assert f["open_vel_mean_median"] == pytest.approx(5.0)  # 1.0 / 0.2
    assert f["open_vel_mean"] == pytest.approx(10.0)  # (4 * 5 + 30) / 5
    assert f["amp_max"] == 6.0


def test_below_minimum_is_nan_with_a_reason_never_zero() -> None:
    f = feats(make_taps([0.5, 1.0]))
    for name in ("tap_rate_hz", "iti_cv", "iti_mean_s", "amp_mean", "open_vel_peak"):
        assert np.isnan(f[name]), name
    assert f["amp_max"] == 1.0  # needs one tap only
    for reason in ("tap_rate_hz:2<3", "iti_mean_s:1<2", "iti_cv:1<3", "amp_mean:2<3"):
        assert reason in f["nan_reasons"].split(";")
    one = feats(make_taps([0.5]))
    assert np.isnan(one["iti_mean_s"]) and one["amp_max"] == 1.0
    none = feats(make_taps([]))
    assert (
        np.isnan(none["amp_max"]) and none["n_taps"] == 0 and "amp_max:0<1" in none["nan_reasons"]
    )
    assert not any(f[c] == 0 for c in fa.FEATURES if np.isfinite(f[c]))


def test_minimum_boundaries() -> None:
    f3 = feats(make_taps([0.5, 1.0, 1.5]))
    assert f3["tap_rate_hz"] == pytest.approx(2.0) and f3["amp_mean"] == 1.0
    assert np.isnan(f3["iti_cv"]) and not np.isnan(f3["iti_mean_s"])  # 2 intervals only
    assert not np.isnan(feats(make_taps([0.5, 1.0, 1.5, 2.0]))["iti_cv"])


def test_one_sided_taps_leave_amplitude_and_velocity_but_stay_in_the_iti() -> None:
    amp = [100.0, 1.0, 1.0, 1.0, 1.0, 100.0]
    f = feats(make_taps(two_hz(6), amp=amp, flags={0: "one_sided", 5: "one_sided"}))
    assert f["amp_mean"] == 1.0 and f["amp_max"] == 1.0
    assert f["open_vel_mean"] == pytest.approx(5.0)
    assert f["n_one_sided_excluded"] == 2
    assert f["iti_mean_s"] == pytest.approx(0.5) and f["tap_rate_hz"] == pytest.approx(2.0)
    assert f["amp_max_sens"] == 100.0 and f["amp_mean_sens"] == pytest.approx(34.0)  # sensitivity


def test_a_gap_interval_is_excluded_from_the_iti_only() -> None:
    times = [0.5, 1.0, 1.5, 5.5, 6.0, 6.5]
    f = feats(make_taps(times, flags={3: "gap_in_interval;long_interval"}))
    assert f["iti_mean_s"] == pytest.approx(0.5) and f["iti_cv"] == pytest.approx(0.0, abs=1e-12)
    assert f["n_iti_excluded"] == 1
    assert f["tap_rate_hz"] == pytest.approx(5 / 6.0)  # the rate still spans the whole clip
    long_kept = feats(make_taps(times, flags={3: "long_interval"}))
    assert long_kept["iti_mean_s"] == pytest.approx((0.5 * 4 + 4.0) / 5)  # long intervals stay
    assert long_kept["n_iti_excluded"] == 0


def test_short_intervals_are_counted_and_dropped_only_in_the_sensitivity_set() -> None:
    f = feats(make_taps([0.5, 1.0, 1.1, 1.6, 2.1]))
    assert f["n_short_iti"] == 1
    assert f["iti_mean_s"] == pytest.approx(1.6 / 4)
    assert f["iti_mean_s_sens"] == pytest.approx(0.5) and f["tap_rate_hz_sens"] == pytest.approx(
        2.0
    )
    assert f["iti_cv_sens"] == pytest.approx(0.0, abs=1e-12)


def test_columns_are_fixed() -> None:
    f = feats(make_taps(two_hz()))
    assert list(f) == list(fa.COLUMNS)


def test_sensitivity_columns_are_the_primary_name_plus_sens() -> None:
    cols = list(fa.COLUMNS)
    assert len(cols) == len(set(cols))
    assert not any(c.endswith("_s_s") for c in cols)
    sens = [c for c in cols if c.endswith("_sens")]
    assert sens == [f + "_sens" for f in fa.FEATURES]
    assert all(f in cols for f in fa.FEATURES)
    assert not any(c.endswith("_s") and c[:-2] in fa.FEATURES for c in cols)


# ----------------------------------------------------------------------- a made-up taps folder

REFS = ("aaaa1111", "bbbb2222", "cccc3333")
STALE = "dddd4444"


def write_taps(path: Path, df: pd.DataFrame, params: dict[str, Any] | None = None,
               tool: str = "taps-0.1.0") -> None:  # fmt: skip
    table = pa.Table.from_pandas(df, preserve_index=False)
    meta = {"params": params or PARAMS, "tool": tool, "status": "ok"}
    table = table.replace_schema_metadata({META_KEY: json.dumps(meta).encode()})
    pq.write_table(table, path)


def clip_taps(i: int, ref: str) -> pd.DataFrame:
    return make_taps(two_hz(8 + 4 * i), ref=ref, amp=1.0 + i, flags={0: "one_sided"})


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    interim, docs = tmp_path / "interim", tmp_path / "docs"
    taps, out = interim / "taps", interim / "features"
    taps.mkdir(parents=True)
    docs.mkdir()
    rows = []
    for i, ref in enumerate(REFS):
        df = clip_taps(i, ref)
        write_taps(taps / f"{ref}.parquet", df)
        rows.append({"ref": ref, "duration_s": 20.0 + i, "fps": 30.0 + i, "n_taps": len(df),
                     "qc_status": "retained" if i != 1 else "retained_flag"})  # fmt: skip
    pd.DataFrame(rows).to_csv(taps / "taps_summary.csv", index=False)
    (taps / f"{STALE}.parquet").write_bytes(b"not a parquet file")  # must never be read
    pd.DataFrame(
        {"ref": REFS, "verdict": ["ok", "ok_flag", "ok"], "note": ["", "spiky", ""]}
    ).to_csv(docs / "qc_retained.csv", index=False)
    pd.DataFrame({"ref": [STALE], "reason": ["tiny_signal"]}).to_csv(
        docs / "qc_rejections.csv", index=False
    )
    monkeypatch.setattr(fa, "INTERIM_ROOT", interim)
    monkeypatch.setattr(fa, "DOCS_ROOT", docs)
    return {"interim": interim, "docs": docs, "taps": taps, "out": out, "root": tmp_path}


def run(w: dict[str, Any], *extra: str) -> int:
    return fa.main(["--taps-dir", str(w["taps"]), "--out-dir", str(w["out"]),
                    "--docs-dir", str(w["docs"]), *extra])  # fmt: skip


def outputs(w: dict[str, Any]) -> dict[str, bytes]:
    return {n: (w["out"] / n).read_bytes() for n in (fa.OUT_TABLE, fa.OUT_FEATURES, fa.OUT_META)}


def test_end_to_end_counts_carried_columns_and_the_stale_file(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(world) == 0
    assert "ignored 1 taps files that are not in the QC retained list" in capsys.readouterr().out
    df = pd.read_csv(world["out"] / fa.OUT_FEATURES)
    assert df["ref"].tolist() == list(REFS)  # sorted by ref, the stale clip absent
    assert df["n_taps"].tolist() == [8, 12, 16]
    assert df["duration_s"].tolist() == [20.0, 21.0, 22.0]  # each row from its own summary row
    assert df["fps"].tolist() == [30.0, 31.0, 32.0]
    assert df["qc_status"].tolist() == ["retained", "retained_flag", "retained"]
    assert df["amp_mean"].tolist() == [1.0, 2.0, 3.0]
    table = pd.read_parquet(world["out"] / fa.OUT_TABLE)
    assert len(table) == 36 and set(table["ref"]) == set(REFS)
    assert list(df.columns) == list(fa.COLUMNS)


def test_metadata(world: dict[str, Any]) -> None:
    assert run(world) == 0
    meta = json.loads((world["out"] / fa.OUT_META).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:8]  # noqa: E731
    canonical = json.dumps(PARAMS, sort_keys=True, separators=(",", ":"))
    assert meta["tap_params_sha8"] == hashlib.sha256(canonical.encode()).hexdigest()[:8]
    assert meta["taps_summary_sha8"] == sha(world["taps"] / "taps_summary.csv")
    assert meta["qc_list_sha8"] == sha(world["docs"] / "qc_retained.csv")
    assert meta["qc_list_sha8"] != sha(world["docs"] / "qc_rejections.csv")
    assert meta["taps_folder"] == "taps" and meta["taps_tool"] == "taps-0.1.0"
    assert meta["definitions_version"] == "features-a-1" == fa.DEFINITIONS_VERSION
    assert (
        meta["features_tool"] == fa.TOOL_VERSION and meta["n_clips"] == 3 and meta["n_taps"] == 36
    )


def test_second_run_is_byte_identical_and_independent_of_file_order(world: dict[str, Any]) -> None:
    assert run(world) == 0
    first = outputs(world)
    assert run(world) == 0
    assert outputs(world) == first
    for ref in reversed(REFS):  # rewrite the taps files in the opposite order
        path = world["taps"] / f"{ref}.parquet"
        df, meta = tt.read_taps_file(path)
        path.unlink()
        write_taps(path, df, meta["params"])
    assert run(world) == 0
    assert outputs(world) == first


def test_parameter_or_tool_mismatch_is_exit_1(world: dict[str, Any]) -> None:
    path = world["taps"] / f"{REFS[1]}.parquet"
    df, _ = tt.read_taps_file(path)
    write_taps(path, df, {**PARAMS, "prominence_frac": 0.25})
    assert run(world) == 1
    write_taps(path, df, PARAMS, tool="taps-0.2.0")
    assert run(world) == 1
    write_taps(path, df, PARAMS)
    assert run(world) == 0


def test_summary_count_mismatch_is_exit_1(world: dict[str, Any]) -> None:
    s = pd.read_csv(world["taps"] / "taps_summary.csv")
    s.loc[0, "n_taps"] += 1
    s.to_csv(world["taps"] / "taps_summary.csv", index=False)
    assert run(world) == 1


def test_missing_input_and_bad_paths_are_exit_2(world: dict[str, Any], tmp_path: Path) -> None:
    (world["taps"] / f"{REFS[2]}.parquet").unlink()
    assert run(world) == 2
    assert fa.main(["--out-dir", str(tmp_path / "elsewhere")]) == 2
    assert fa.main(["--taps-dir", str(tmp_path / "elsewhere")]) == 2
    assert fa.main(["--docs-dir", str(tmp_path / "elsewhere")]) == 2
    assert not (tmp_path / "elsewhere").exists()


def test_no_qc_list_is_exit_2(world: dict[str, Any]) -> None:
    (world["docs"] / "qc_retained.csv").unlink()
    assert run(world) == 2


# ----------------------------------------------------------------------- leakage


def test_the_sources_name_no_label_or_identity_module() -> None:
    words = r"\b(fis_diagnostic|UPDRS|hubu_fis|clip_id|pseudonym|CONTROL)\b"
    for name in ("tap_table.py", "features_a.py"):
        assert not re.search(words, (SRC / name).read_text(), re.IGNORECASE), name


def test_the_pipeline_reads_nothing_but_its_own_inputs(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    decoy = world["root"] / "raw" / "fis_diagnostic.csv"
    decoy.parent.mkdir()
    decoy.write_text("ID,UPDRS\nCONTROL01_DCHA,0\n")
    allowed = [world["taps"].resolve(), world["docs"].resolve(), world["out"].resolve()]
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

    monkeypatch.setattr(builtins, "open", wrap(builtins.open))
    monkeypatch.setattr(io, "open", wrap(io.open))
    monkeypatch.setattr(pd, "read_csv", wrap(pd.read_csv))
    monkeypatch.setattr(pd, "read_parquet", wrap(pd.read_parquet))
    monkeypatch.setattr(pq, "read_table", wrap(pq.read_table))
    monkeypatch.setattr(pq, "read_schema", wrap(pq.read_schema))
    assert run(world) == 0
    assert seen and "fis_diagnostic.csv" not in seen
    # the guard itself works: a read of the decoy fails
    with pytest.raises(AssertionError):
        pd.read_csv(decoy)


# ----------------------------------------------------------------------- real frozen taps


def test_real_taps_agree_with_the_summary_where_they_must(
    capsys: pytest.CaptureFixture[str],
) -> None:
    taps_dir = fa.INTERIM_ROOT / "taps"
    qc = fa.DOCS_ROOT / "qc_retained.csv"
    if not ((taps_dir / "taps_summary.csv").is_file() and qc.is_file()):
        pytest.skip("real data not available")
    summary = pd.read_csv(taps_dir / "taps_summary.csv", dtype={"ref": str}).set_index("ref")
    kept = pd.read_csv(qc, dtype=str)
    kept = kept[kept["verdict"].isin(["ok", "ok_flag"])]["ref"].tolist()
    assert len(kept) == 233
    n_checked = 0
    for ref in kept:
        raw, _ = tt.read_taps_file(taps_dir / f"{ref}.parquet")
        table = tt.build_tap_table(raw)
        row = fa.clip_features(table, ref, summary.loc[ref])
        s = summary.loc[ref]
        assert row["n_taps"] == s["n_taps"] == len(raw)
        assert row["n_over_duration_hz"] == pytest.approx(
            s["tap_rate_hz"]
        )  # summary rate = n / duration
        assert row["tap_rate_hz"] == pytest.approx(
            (len(raw) - 1) / (raw["t_peak_s"].iloc[-1] - raw["t_peak_s"].iloc[0])
        )
        assert (table["open_vel_peak"] > 0).all() and (table["close_vel_peak"] > 0).all()
        np.testing.assert_allclose(table["open_vel_mean"], table["amplitude"] / table["rise_s"])
        if table["valid_iti"].iloc[1:].all():  # no excluded interval: the relations are exact
            m = len(table) - 1
            assert row["tap_rate_hz"] == pytest.approx(1 / row["iti_mean_s"])
            # the summary CV is population (ddof=0): mine (ddof=1) is larger by sqrt(m / (m - 1))
            assert row["iti_cv"] * np.sqrt((m - 1) / m) == pytest.approx(s["interval_cv"])
            n_checked += 1
    assert n_checked > 200
    with capsys.disabled():
        print(f"\nreal taps: {len(kept)} clips, {n_checked} with every interval valid")
