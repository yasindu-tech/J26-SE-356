"""Tests for models/tapping/src/reference_features.py (task FT-40). Known-answer checks of every
reference feature on synthetic landmarks, label-blindness, the registered feature list and the table
builder on synthetic parquet files named like raw clip IDs. No real data is used."""

from __future__ import annotations

import builtins
import inspect
import io
import json
import math
import re
import statistics
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv_splits as cvs  # noqa: E402
import reference_features as rf  # noqa: E402
from hubu_fis import pseudonym  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src"
RAW_IDS = ["CONTROL01_DCHA", "CONTROL01_IZDA", "ID1001_DCHA", "ID1001_IZDA", "ID1002_DCHA"]
EXPECTED_COLUMNS = [
    "ref_smoothed_amplitude_mean", "ref_smoothed_amplitude_std", "ref_smoothed_amplitude_max",
    "ref_velocity_mean", "ref_velocity_max", "ref_velocity_std",
    "ref_acceleration_mean", "ref_acceleration_max", "ref_acceleration_std",
    "ref_distance_ang_mean", "ref_distance_ang_std",
    "ref_frequency_mean", "ref_frequency_std", "ref_frequency_count",
    "ref_interval_std", "ref_interval_cv", "ref_amplitude_slope", "ref_frequency_drop",
    "ref_acceleration_increase", "ref_skew_amplitude", "ref_kurtosis_amplitude",
    "ref_skew_velocity", "ref_kurtosis_velocity", "ref_fft_dominant_freq_amplitude",
    "ref_fft_power_amplitude", "ref_fft_dominant_freq_velocity", "ref_fft_power_velocity",
]  # fmt: skip


# --------------------------------------------------------------------------- synthetic landmarks


def landmarks(
    amplitude: np.ndarray, detected: np.ndarray | None = None, scale: float = 1.0
) -> pd.DataFrame:
    """Wrist at the origin, index tip at (1, 0, 0), thumb tip so that |thumb - index| = amplitude."""
    n = len(amplitude)
    names = ["detected"] + [
        f"{a}{k:02d}" for k in (0, 4, 8) for a in "xyz"
    ]  # literal ids, not the module's
    decoys = [f"{a}{k:02d}" for k in (1, 9) for a in "xyz"]  # other landmarks must never be read
    frame = pd.DataFrame(0.0, index=range(n), columns=[*names, *decoys])
    frame[decoys] = 7.0
    frame["x08"] = 1.0 * scale
    frame["x04"] = (
        1.0 - amplitude
    ) * scale  # on the line wrist-index: distance to the index is the amplitude
    frame["y04"] = 0.0
    frame["detected"] = True if detected is None else detected
    if detected is not None:
        frame.loc[
            ~np.asarray(detected, dtype=bool), [c for c in frame.columns if c != "detected"]
        ] = np.nan
    return frame


def pulses(spacings: list[int], first: int = 5, n: int = 300, base: float = 0.2) -> np.ndarray:
    """A baseline with a triangular pulse at each peak frame (strict maximum after smoothing)."""
    amp = np.full(n, base)
    frames = [first]
    for gap in spacings:
        frames.append(frames[-1] + gap)
    assert frames[-1] + 2 < n
    for s in frames:
        for offset, height in ((-2, 0.0), (-1, 0.4), (0, 0.8), (1, 0.4), (2, 0.0)):
            amp[s + offset] += height
    return amp


def ts_of(amp: np.ndarray, fps: float = 30.0) -> pd.DataFrame:
    return rf.reference_time_series(landmarks(amp), fps)


def peak_frames(ts: pd.DataFrame, fps: float = 30.0) -> np.ndarray:
    return rf.find_peaks(ts["SMOOTHED_AMPLITUDE"], distance=fps // 3)[0]


# --------------------------------------------------------------------------- the series


def test_the_amplitude_is_normalised_by_the_hand_size_and_uses_all_three_coordinates() -> None:
    amp = np.array([0.5, 0.6, 0.7, 0.6, 0.5, 0.4])
    ts = rf.reference_time_series(landmarks(amp), 30.0)
    expected = pd.Series(amp).rolling(3, center=True).mean().bfill().ffill()
    assert ts["SMOOTHED_AMPLITUDE"].to_numpy() == pytest.approx(expected.to_numpy())
    big = rf.reference_time_series(landmarks(amp, scale=3.0), 30.0)  # a larger hand changes nothing
    assert big["SMOOTHED_AMPLITUDE"].to_numpy() == pytest.approx(
        ts["SMOOTHED_AMPLITUDE"].to_numpy()
    )
    z = landmarks(
        np.zeros(5)
    )  # thumb above the index tip along z only: amplitude = 0.5 / hand size 1
    z["x04"], z["z04"] = 1.0, 0.5
    assert rf.reference_time_series(z, 30.0)["SMOOTHED_AMPLITUDE"].to_numpy() == pytest.approx(
        np.full(5, 0.5)
    )


def test_the_angle_feature_is_the_wrist_angle_over_ninety_degrees() -> None:
    right = landmarks(np.zeros(4))
    right["x04"], right["y04"] = 0.0, 1.0  # W-I = (-1,0,0), W-T = (0,-1,0): 90 degrees
    assert rf.reference_time_series(right, 30.0)["DISTANCE_ANG"].to_numpy() == pytest.approx(
        np.ones(4)
    )
    sixty = landmarks(np.zeros(4))
    sixty["x04"], sixty["y04"] = 0.5, math.sqrt(3) / 2  # 60 degrees
    assert rf.reference_time_series(sixty, 30.0)["DISTANCE_ANG"].to_numpy() == pytest.approx(
        np.full(4, 60 / 90)
    )
    along_z = landmarks(np.zeros(4))
    along_z["x04"], along_z["z04"] = 0.0, 1.0  # a third coordinate counts
    assert rf.reference_time_series(along_z, 30.0)["DISTANCE_ANG"].to_numpy() == pytest.approx(
        np.ones(4)
    )
    assert rf.angle_degrees(np.array([[1.0, 0, 0]]), np.array([[1.0, 0, 0]]))[0] == pytest.approx(
        0, abs=1e-6
    )


def test_frames_without_a_hand_are_dropped_not_filled_and_time_stays_one_over_fps() -> None:
    amp = np.linspace(0.2, 0.8, 12)
    detected = np.ones(12, dtype=bool)
    detected[[3, 4, 9]] = False
    ts = rf.reference_time_series(landmarks(amp, detected), 30.0)
    assert (
        len(ts) == 9 and ts.notna().all().drop("FREQUENCY").all()
    )  # no NaN rows, nothing interpolated
    kept = pd.Series(amp[detected]).rolling(3, center=True).mean().bfill().ffill()
    assert ts["VELOCITY"].iloc[1:].to_numpy() == pytest.approx(
        (kept.diff() * 30.0).iloc[1:].to_numpy()
    )  # 1/fps between kept rows
    assert (
        ts["VELOCITY"].iloc[0] == 0 and ts["ACCELERATION"].iloc[0] == 0
    )  # NaN is set to 0, as in the reference


def test_velocity_and_acceleration_are_differences_over_one_over_fps() -> None:
    amp = np.array([0.2, 0.3, 0.5, 0.8, 0.5, 0.3, 0.2, 0.2])
    for fps in (30.0, 60.0):
        ts = rf.reference_time_series(landmarks(amp), fps)
        smoothed = ts["SMOOTHED_AMPLITUDE"].to_numpy()
        velocity = np.concatenate([[0.0], np.diff(smoothed) * fps])
        assert ts["VELOCITY"].to_numpy() == pytest.approx(velocity)
        assert ts["ACCELERATION"].to_numpy() == pytest.approx(
            np.concatenate([[0.0], np.diff(velocity) * fps])
        )


def test_frequency_is_one_over_the_peak_interval_and_set_from_the_second_peak() -> None:
    ts = ts_of(pulses([15] * 18))
    peaks = peak_frames(ts)
    assert peaks.tolist() == [5 + 15 * k for k in range(19)]
    freq = ts["FREQUENCY"]
    assert freq.iloc[peaks[0]] != freq.iloc[peaks[0]]  # the first peak has no interval: NaN
    assert freq.iloc[peaks[1:]].to_numpy() == pytest.approx(np.full(18, 2.0))
    assert int(freq.notna().sum()) == 18 and freq.drop(index=peaks[1:]).isna().all()


def test_find_peaks_uses_a_minimum_distance_of_fps_floor_three() -> None:
    amp = pulses([12] * 5, first=5, n=120)  # peaks 12 frames apart
    assert len(peak_frames(ts_of(amp, 30.0), 30.0)) == 6  # distance 10: all kept
    assert (
        len(peak_frames(ts_of(amp, 45.0), 45.0)) < 6
    )  # distance 15: neighbours closer than 15 frames are dropped


# --------------------------------------------------------------------------- the 27 features


def feature_frame(spacings: list[int], fps: float = 30.0, **kw: Any) -> dict[str, float]:
    ts = ts_of(pulses(spacings, **kw), fps)
    return rf.reference_features(ts, fps)


def test_regular_taps_give_zero_interval_spread_and_exact_frequency() -> None:
    f = feature_frame([15] * 18)
    assert f["ref_interval_std"] == pytest.approx(0, abs=1e-12) and f[
        "ref_interval_cv"
    ] == pytest.approx(0, abs=1e-12)
    assert f["ref_frequency_mean"] == pytest.approx(2.0) and f[
        "ref_frequency_std"
    ] == pytest.approx(0, abs=1e-9)
    assert f["ref_frequency_count"] == 18


def test_alternating_intervals_give_population_std_and_cv() -> None:
    f = feature_frame([12, 18] * 8)  # periods 0.4 s and 0.6 s, 16 intervals
    assert f["ref_interval_std"] == pytest.approx(0.1)  # population std (ddof=0), not 0.1033
    assert f["ref_interval_cv"] == pytest.approx(0.1 / 0.5)
    assert f["ref_frequency_mean"] == pytest.approx((2.5 + 1 / 0.6) / 2)
    assert f["ref_frequency_std"] == pytest.approx(
        statistics.stdev([2.5, 1 / 0.6] * 8)
    )  # sample std for FREQUENCY


def test_frequency_drop_is_first_half_minus_second_half() -> None:
    f = feature_frame([10] * 14 + [20] * 7)  # 3 Hz in the first half, 1.5 Hz in the second
    assert f["ref_frequency_drop"] == pytest.approx(
        1.5
    )  # positive for a slowing hand: the reference's sign
    slow_first = feature_frame([20] * 7 + [10] * 14)
    assert slow_first["ref_frequency_drop"] == pytest.approx(-1.5)


def test_acceleration_increase_is_first_half_minus_second_half_of_acceleration() -> None:
    ts = ts_of(pulses([15] * 18))
    a = ts["ACCELERATION"].to_numpy()
    half = len(a) // 2
    assert rf.reference_features(ts, 30.0)["ref_acceleration_increase"] == pytest.approx(
        a[:half].mean() - a[-half:].mean()
    )
    ts_loud_start = ts.copy()
    ts_loud_start.loc[:20, "ACCELERATION"] *= 5.0
    assert rf.reference_features(ts_loud_start, 30.0)["ref_acceleration_increase"] != pytest.approx(
        rf.reference_features(ts, 30.0)["ref_acceleration_increase"]
    )


def test_amplitude_slope_is_per_kept_frame_not_per_second() -> None:
    ramp = 0.2 + 0.001 * np.arange(200)
    for fps in (30.0, 60.0):
        assert rf.reference_features(ts_of(ramp, fps), fps)["ref_amplitude_slope"] == pytest.approx(
            0.001, rel=0.02
        )
    assert rf.reference_features(ts_of(ramp[::-1].copy()), 30.0)["ref_amplitude_slope"] < 0


def test_aggregates_match_an_independent_computation() -> None:
    ts = ts_of(pulses([13, 17, 14, 16] * 4))
    f = rf.reference_features(ts, 30.0)
    for series, stats in rf.AGGREGATES:
        values = ts[series.upper()].dropna().tolist()
        for stat in stats:
            expected = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values),
                "max": max(values),
                "count": len(values),
            }[stat]
            assert f[f"ref_{series}_{stat}"] == pytest.approx(expected, rel=1e-9)


def test_skew_and_kurtosis_use_population_moments_and_fisher_kurtosis() -> None:
    ts = ts_of(pulses([14, 16] * 8))
    f = rf.reference_features(ts, 30.0)
    for name, values in (("amplitude", ts["SMOOTHED_AMPLITUDE"]), ("velocity", ts["VELOCITY"])):
        x = values.to_numpy()
        m2, m3, m4 = (np.mean((x - x.mean()) ** k) for k in (2, 3, 4))
        assert f[f"ref_skew_{name}"] == pytest.approx(m3 / m2**1.5, rel=1e-9)
        assert f[f"ref_kurtosis_{name}"] == pytest.approx(m4 / m2**2 - 3, rel=1e-9)


def test_fft_features_keep_the_first_half_of_the_bins_including_dc() -> None:
    n = 100
    wave = np.cos(2 * np.pi * 5 * np.arange(n) / n)
    freq, power = rf.fft_features(wave)
    assert freq == pytest.approx(5 / n) and power == pytest.approx(
        50.0**2
    )  # bin 95 is outside the first n//2 bins
    freq_dc, power_dc = rf.fft_features(wave + 10.0)  # a large offset: the DC bin wins, frequency 0
    assert freq_dc == 0.0 and power_dc == pytest.approx((n * 10.0) ** 2 + 50.0**2)
    odd_freq, _ = rf.fft_features(np.cos(2 * np.pi * 3 * np.arange(99) / 99))
    assert odd_freq == pytest.approx(3 / 99)


def test_every_feature_has_a_known_value_in_one_designed_clip() -> None:
    f = feature_frame([15] * 18)
    assert list(f) == EXPECTED_COLUMNS and all(isinstance(v, float) for v in f.values())
    assert (
        f["ref_fft_dominant_freq_amplitude"] == 0.0
    )  # the baseline is non-zero, so the DC bin dominates
    assert f["ref_amplitude_slope"] == pytest.approx(0, abs=2e-4)
    assert f["ref_distance_ang_mean"] == pytest.approx(
        0, abs=1e-6
    )  # thumb on the wrist-index line: angle 0


def test_undefined_values_are_nan_never_zero() -> None:
    two_peaks = feature_frame([40])  # one interval: the interval statistics need two
    assert math.isnan(two_peaks["ref_interval_std"]) and math.isnan(two_peaks["ref_interval_cv"])
    flat = ts_of(np.full(60, 0.4))
    flat["SMOOTHED_AMPLITUDE"] = 0.4
    flat["VELOCITY"] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # scipy's precision-loss warning must not leak out
        values = rf.reference_features(flat, 30.0)
    assert (
        math.isnan(values["ref_frequency_mean"]) and values["ref_frequency_count"] == 0
    )  # no peaks
    assert math.isnan(values["ref_interval_std"]) and math.isnan(values["ref_frequency_drop"])
    # scipy returns a finite number for a constant series (float rounding); the reference gets the same
    assert math.isfinite(values["ref_skew_amplitude"])
    assert math.isnan(
        rf.half_difference(pd.Series([np.nan, np.nan, 1.0, 2.0]))
    )  # a half without a value
    assert math.isnan(rf.half_difference(pd.Series([1.0])))  # n // 2 = 0


def test_a_clip_with_too_few_detected_frames_is_refused() -> None:
    with pytest.raises(cvs.GuardFailure):
        rf.reference_time_series(
            landmarks(np.ones(5), np.array([True, True, False, False, False])), 30.0
        )


# --------------------------------------------------------------------------- the registered list


def test_the_feature_list_and_its_hashes_are_pinned() -> None:
    assert rf.REFERENCE_COLUMNS == EXPECTED_COLUMNS and len(set(rf.REFERENCE_COLUMNS)) == 27
    assert rf.LANDMARK_IDS == {"wrist": 0, "thumb": 4, "index": 8}  # wrist, thumb tip, index tip
    assert ["detected"] + [f"{a}{k:02d}" for k in (0, 4, 8) for a in "xyz"] == rf.LANDMARK_COLUMNS
    assert rf.roles_sha8() == rf.PINNED_ROLES_SHA8 == "3373dae7"
    assert cvs.sha8_of(json.dumps(rf.REFERENCE_COLUMNS)) == rf.PINNED_FEATURES_SHA8 == "17fb7bbf"
    roles = rf.roles_table()
    assert roles["role"].tolist() == ["id", "group_key"] + ["reference_feature"] * 27
    assert not set(roles["column"]) & {"rating", "hand", "UPDRS", "group"}
    assert roles["column"].is_unique and (roles["note"].str.len() > 0).all()
    assert set(rf.REFERENCE_COLUMNS).isdisjoint(cvs.MODEL_COLUMNS)  # no clash with sets A and B


def test_the_source_records_why_two_reference_sets_are_not_used() -> None:
    assert rf.SOURCE["licence"] == "Apache-2.0" and "classical" in rf.SOURCE["used"]
    assert "labels" in rf.SOURCE["not_used"] and "tsfresh" in rf.SOURCE["not_used"]


# --------------------------------------------------------------------------- label blindness


def test_no_feature_function_can_see_a_label() -> None:
    for fn in (
        rf.reference_time_series,
        rf.reference_features,
        rf.fft_features,
        rf.half_difference,
        rf.clip_inputs,
        rf.build_table,
    ):
        assert not [
            p
            for p in inspect.signature(fn).parameters
            if re.search(r"label|rating|updrs|group|diagnos", p, re.I)
        ]
    source = (SRC / "reference_features.py").read_text()
    assert "fis_diagnostic" not in source and "load_labels" not in source
    imports = re.findall(r"^\s*from\s+hubu_fis\s+import\s+(.+)$", source, flags=re.M)
    assert imports == ["pseudonym"]
    assert not {"cv_harness", "feature_matrix", "gait"} & {
        i.split(".")[0] for i in re.findall(r"^\s*(?:from|import)\s+([\w.]+)", source, flags=re.M)
    }


# --------------------------------------------------------------------------- the table builder


def write_clip(
    path: Path,
    amp: np.ndarray,
    fps: float = 30.0,
    detected: np.ndarray | None = None,
    metadata: bool = True,
) -> None:
    table = pa.Table.from_pandas(landmarks(amp, detected), preserve_index=False)
    table = table.replace_schema_metadata(
        {"tapping.hand_landmarks": json.dumps({"fps": fps})} if metadata else {}
    )
    pq.write_table(table, path)


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    landmarks_dir = tmp_path / "data" / "interim" / "landmarks"
    landmarks_dir.mkdir(parents=True)
    out = tmp_path / "data" / "processed" / "reference"
    monkeypatch.setattr(rf, "INTERIM_ROOT", tmp_path / "data" / "interim")
    monkeypatch.setattr(rf, "REFERENCE_ROOT", out)
    rng = np.random.default_rng(1)
    refs = [pseudonym(i) for i in RAW_IDS[:4]]
    for k, raw in enumerate(RAW_IDS):
        spacings = [int(g) for g in rng.integers(11, 20, 15)]
        detected = np.ones(300, dtype=bool)
        if k == 2:
            detected[::4] = (
                False  # 75% detected: below the reference's own 90% rule, but still scored
            )
        write_clip(landmarks_dir / f"{raw}.parquet", pulses(spacings), detected=detected)
    features = tmp_path / "data" / "processed" / "features.csv"
    features.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "ref": refs,
            "participant": [pseudonym("P1"), pseudonym("P1"), pseudonym("P2"), pseudonym("P2")],
            "rating": [0, 1, 2, 3],
        }
    ).to_csv(features, index=False)
    return {
        "features": features,
        "landmarks": landmarks_dir,
        "out": out,
        "refs": refs,
        "tmp": tmp_path,
    }


def build(w: dict[str, Any], *extra: str, out: Path | None = None) -> int:
    return rf.main(
        [
            "--features",
            str(w["features"]),
            "--landmarks",
            str(w["landmarks"]),
            "--out-dir",
            str(out or w["out"]),
            *extra,
        ]
    )


def test_the_table_has_the_registered_columns_and_nothing_else(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert build(world) == 0
    table = pd.read_csv(
        world["out"] / "reference_features.csv", dtype={"ref": str, "participant": str}
    )
    assert list(table.columns) == ["ref", "participant", *EXPECTED_COLUMNS] and len(table) == 4
    assert table["ref"].tolist() == sorted(world["refs"]) and "rating" not in table.columns
    roles = pd.read_csv(world["out"] / "reference_roles.csv", dtype=str)
    assert cvs.roles_sha8_from_file(roles) == rf.PINNED_ROLES_SHA8
    meta = json.loads((world["out"] / "reference_meta.json").read_text())
    assert (
        meta["labels_used"] == []
        and meta["group_exported"] is False
        and meta["n_clips"] == 4
        and meta["n_participants"] == 2
    )
    assert (
        meta["reference_roles_sha8"] == "3373dae7"
        and meta["features_sha8"] == "17fb7bbf"
        and meta["columns"] == EXPECTED_COLUMNS
    )
    assert meta["n_landmark_files_not_used"] == 1 and meta["source"]["licence"] == "Apache-2.0"
    assert (
        meta["n_clips_at_or_below_reference_detection_rule"] == 1
    )  # counted, but that clip is kept
    assert sorted(p.name for p in world["out"].iterdir()) == [
        "reference_features.csv",
        "reference_meta.json",
        "reference_roles.csv",
    ]
    assert "reference_roles_sha8 3373dae7" in capsys.readouterr().out


def test_the_table_values_equal_the_function_values(world: dict[str, Any]) -> None:
    assert build(world) == 0
    table = pd.read_csv(
        world["out"] / "reference_features.csv", dtype={"ref": str}, float_precision="round_trip"
    ).set_index("ref")
    raw = RAW_IDS[0]
    frame, fps = rf.clip_inputs(world["landmarks"] / f"{raw}.parquet")
    expected = rf.reference_features(rf.reference_time_series(frame, fps), fps)
    for column, value in expected.items():
        assert table.loc[pseudonym(raw), column] == pytest.approx(
            value, rel=1e-12, abs=1e-12, nan_ok=True
        )


def test_labels_do_not_change_anything_and_the_label_file_is_never_opened(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = world["tmp"] / "data" / "raw" / "HUBU-FIS"
    raw.mkdir(parents=True)
    (raw / "fis_diagnostic.csv").write_text("ID,UPDRS\nCONTROL01_DCHA,3\nID1001_DCHA,0\n")
    assert build(world) == 0
    first = (world["out"] / "reference_features.csv").read_bytes()
    opened: list[str] = []
    real_open = builtins.open

    def spy(file: Any, *a: Any, **k: Any) -> Any:
        opened.append(str(file))
        return real_open(file, *a, **k)

    (raw / "fis_diagnostic.csv").write_text(
        "ID,UPDRS\nCONTROL01_DCHA,0\nID1001_DCHA,3\n"
    )  # shuffled labels
    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(io, "open", spy)
    pd.read_csv(world["features"]).assign(rating=[3, 2, 1, 0]).to_csv(
        world["features"], index=False
    )  # shuffled ratings
    assert build(world) == 0
    assert (world["out"] / "reference_features.csv").read_bytes() == first
    assert not [p for p in opened if "fis_diagnostic" in p or "HUBU-FIS" in p]


def test_rerun_is_byte_identical_and_atomic(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert build(world) == 0
    first = {p.name: p.read_bytes() for p in world["out"].iterdir()}
    assert build(world) == 0 and {p.name: p.read_bytes() for p in world["out"].iterdir()} == first
    assert not [p for p in world["out"].iterdir() if p.name.startswith(".")]
    write_clip(
        world["landmarks"] / f"{RAW_IDS[0]}.parquet", pulses([13] * 18)
    )  # new content for the next run
    real_replace = rf.os.replace

    def broken(src: Any, dst: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(rf.os, "replace", broken)
    with pytest.raises(OSError):
        build(world)
    after = {p.name: p.read_bytes() for p in world["out"].iterdir()}
    assert after == first  # no file was partly rewritten
    monkeypatch.setattr(rf.os, "replace", real_replace)
    assert build(world) == 0
    assert (world["out"] / "reference_features.csv").read_bytes() != first["reference_features.csv"]


def test_no_raw_id_appears_in_any_output(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert build(world) == 0
    blob = capsys.readouterr().out + "".join(p.read_text() for p in world["out"].iterdir())
    for needle in (*RAW_IDS, "CONTROL", "DCHA", "IZDA", "ID1001"):
        assert needle not in blob
    assert not re.search(r"20\d\d-\d\d-\d\d|T\d\d:\d\d", blob)


def test_nan_counts_are_reported_per_feature_and_nothing_is_imputed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    landmarks_dir = tmp_path / "data" / "interim" / "landmarks"
    landmarks_dir.mkdir(parents=True)
    monkeypatch.setattr(rf, "INTERIM_ROOT", tmp_path / "data" / "interim")
    monkeypatch.setattr(rf, "REFERENCE_ROOT", tmp_path / "data" / "processed" / "reference")
    write_clip(landmarks_dir / "ID1001_DCHA.parquet", pulses([15] * 18))
    write_clip(
        landmarks_dir / "ID1001_IZDA.parquet", np.full(300, 0.4)
    )  # a flat clip: no peaks, NaN skew
    features = tmp_path / "f.csv"
    pd.DataFrame(
        {
            "ref": [pseudonym("ID1001_DCHA"), pseudonym("ID1001_IZDA")],
            "participant": [pseudonym("P"), pseudonym("P")],
        }
    ).to_csv(features, index=False)
    assert (
        rf.main(
            [
                "--features",
                str(features),
                "--landmarks",
                str(landmarks_dir),
                "--out-dir",
                str(tmp_path / "data" / "processed" / "reference"),
            ]
        )
        == 0
    )
    meta = json.loads(
        (tmp_path / "data" / "processed" / "reference" / "reference_meta.json").read_text()
    )
    assert (
        meta["nan_counts"]["ref_skew_amplitude"] == 1
        and meta["nan_counts"]["ref_frequency_mean"] == 1
        and meta["nan_counts"]["ref_smoothed_amplitude_mean"] == 0
    )
    assert meta["n_features_with_nan"] == sum(v > 0 for v in meta["nan_counts"].values()) > 0
    table = pd.read_csv(
        tmp_path / "data" / "processed" / "reference" / "reference_features.csv", dtype={"ref": str}
    )
    assert table["ref_skew_amplitude"].isna().sum() == 1  # empty cell, not 0


def test_refusals_and_exit_codes(
    world: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (world["landmarks"] / f"{RAW_IDS[1]}.parquet").unlink()
    assert (
        build(world) == 1 and "1 retained clip(s) have no landmark file" in capsys.readouterr().err
    )
    assert not world["out"].exists()
    write_clip(world["landmarks"] / f"{RAW_IDS[1]}.parquet", pulses([15] * 18), metadata=False)
    assert build(world) == 1 and "no usable nominal fps" in capsys.readouterr().err
    write_clip(
        world["landmarks"] / f"{RAW_IDS[1]}.parquet", pulses([15] * 18), detected=np.arange(300) < 2
    )
    assert build(world) == 1 and "fewer than 3 frames" in capsys.readouterr().err
    pd.DataFrame({"ref": ["aaaaaaaa", "aaaaaaaa"], "participant": ["p", "p"]}).to_csv(
        world["features"], index=False
    )
    assert build(world) == 1 and "ref must be unique" in capsys.readouterr().err
    world["features"].unlink()
    assert build(world) == 2 and "missing input: features file" in capsys.readouterr().err
    assert (
        build(world, out=tmp_path / "elsewhere") == 2
        and "--out-dir must be inside processed/reference" in capsys.readouterr().err
    )
    assert not (tmp_path / "elsewhere").exists()
    assert (
        rf.main(
            [
                "--features",
                str(world["features"]),
                "--landmarks",
                str(tmp_path),
                "--out-dir",
                str(world["out"]),
            ]
        )
        == 2
    )
    assert "--landmarks must be inside data/interim" in capsys.readouterr().err
