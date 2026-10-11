"""Tests for models/tapping/src/hand_scale.py (task FT-13).

Synthetic data only: small made-up landmark tables, no real clip. Clip names such as
``CONTROL9001_DCHA`` are made up; the numbers are not real participants.

No conftest.py on purpose (voice and gait each have one); ``src`` goes on ``sys.path`` here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from matplotlib.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hand_landmarks as hl  # noqa: E402
import hand_scale as hs  # noqa: E402
from hubu_fis import pseudonym  # noqa: E402

STEM = "CONTROL9001_DCHA"
REF = pseudonym(STEM)
WRIST_XY = (400.0, 900.0)
# Knuckle offsets from the wrist (pixels): lengths 100, 100, 100, 100; width 5-17 is sqrt(20000).
OFFSETS = {5: (-60.0, -80.0), 9: (0.0, -100.0), 13: (60.0, -80.0), 17: (80.0, -60.0)}
FPS = 30.0


def make_table(
    distances: Any,
    *,
    zoom: Any = 1.0,
    pts: Any = None,
    detected: Any = None,
) -> pd.DataFrame:
    """A landmarks table with a known palm and a thumb-index distance of ``distances``.

    ``zoom`` scales every pixel coordinate about the wrist, per frame (a hand moving toward or
    away from the camera). World landmarks are the same geometry in metres (1 px = 1 mm).
    """
    d = np.asarray(distances, dtype=float)
    n = len(d)
    s = np.broadcast_to(np.asarray(zoom, dtype=float), (n,))
    df = pd.DataFrame(np.nan, index=range(n), columns=list(hl.COLUMNS))
    df["frame"] = np.arange(n)
    df["timestamp_ms"] = np.round(np.arange(n) * 1000 / FPS).astype(np.int64)
    df["pts_ms"] = df["timestamp_ms"].astype(float) if pts is None else np.asarray(pts, float)
    live = np.isfinite(d)
    df["detected"] = live if detected is None else np.asarray(detected, bool)

    def put(index: int, dx: Any, dy: Any) -> None:
        df.loc[live, f"x_px{index:02d}"] = WRIST_XY[0] + np.asarray(dx)[live] * s[live]
        df.loc[live, f"y_px{index:02d}"] = WRIST_XY[1] + np.asarray(dy)[live] * s[live]
        df.loc[live, f"wx{index:02d}"] = np.asarray(dx)[live] * s[live] / 1000.0
        df.loc[live, f"wy{index:02d}"] = np.asarray(dy)[live] * s[live] / 1000.0
        df.loc[live, f"wz{index:02d}"] = 0.0

    zero = np.zeros(n)
    put(0, zero, zero)
    for index, (ox, oy) in OFFSETS.items():
        put(index, np.full(n, ox), np.full(n, oy))
    put(4, np.full(n, 100.0), np.full(n, -300.0))  # thumb tip
    put(8, 100.0 + 0.6 * np.nan_to_num(d), -300.0 + 0.8 * np.nan_to_num(d))  # 3-4-5 offset
    # the 3-4-5 offset is in unzoomed pixels and the zoom multiplies it, as a real approach does
    df.attrs["hand_landmarks"] = {"fps": FPS, "width": 1080, "height": 1920}
    return df


def open_close(n: int = 600, low: float = 15.0, high: float = 80.0) -> np.ndarray:
    """A clean 1 Hz open/close signal at 30 fps."""
    return np.asarray(low + (high - low) * (0.5 + 0.5 * np.sin(2 * np.pi * np.arange(n) / FPS)))


def text_of(fig: Any) -> str:
    return " ".join(t.get_text() for t in fig.findobj(Text))


# --------------------------------------------------------------------------- known geometry


def test_known_lengths_for_every_candidate() -> None:
    df = make_table([50.0])
    expected = {
        "w9": 100.0,
        "w5": 100.0,
        "w17": 100.0,
        "w5_17_mean": 100.0,
        "palm_width": float(np.sqrt(20000.0)),
        "median4": 100.0,
    }
    assert set(expected) == set(hs.SCALES)
    for name, value in expected.items():
        assert hs.scale_length(df, name)[0] == pytest.approx(value), name


def test_median4_ignores_one_odd_knuckle_but_the_mean_does_not() -> None:
    df = make_table([50.0])
    df.loc[0, "x_px13"], df.loc[0, "y_px13"] = WRIST_XY[0] + 400.0, WRIST_XY[1]  # length 400
    assert hs.scale_length(df, "median4")[0] == pytest.approx(100.0)
    df.loc[0, "x_px17"], df.loc[0, "y_px17"] = WRIST_XY[0] + 300.0, WRIST_XY[1]  # length 300
    assert hs.scale_length(df, "median4")[0] == pytest.approx(200.0)  # median of 100,100,400,300


def test_per_frame_normalisation_gives_distance_over_length() -> None:
    result = hs.normalised_distance(make_table([50.0, 25.0, 100.0]), "w9", "per_frame")
    assert result.values == pytest.approx([0.5, 0.25, 1.0])


def test_the_scale_name_changes_the_answer_as_expected() -> None:
    df = make_table([100.0])
    assert hs.normalised_distance(df, "w9", "per_frame").values[0] == pytest.approx(1.0)
    width = hs.normalised_distance(df, "palm_width", "per_frame").values[0]
    assert width == pytest.approx(100.0 / np.sqrt(20000.0))


def test_world_scale_is_in_metres_and_the_reference_distance_in_centimetres() -> None:
    df = make_table([50.0])
    assert hs.scale_length(df, "w9", "world")[0] == pytest.approx(0.1)  # 100 mm
    assert hs.thumb_index_distance_world_cm(df)[0] == pytest.approx(5.0)  # 50 mm


# --------------------------------------------------------------------------- zoom invariance


def test_zoom_invariance_per_frame_is_exact() -> None:
    d = open_close()
    ramp = np.linspace(0.7, 1.4, len(d))
    flat = hs.normalised_distance(make_table(d), "w9", "per_frame").values
    zoomed = hs.normalised_distance(make_table(d * 1.0, zoom=ramp), "w9", "per_frame").values
    # the raw pixel distance changed with the zoom, the normalised one did not
    raw = hs.thumb_index_distance_px(make_table(d, zoom=ramp))
    assert raw[-1] / raw[0] == pytest.approx(ramp[-1] / ramp[0] * d[-1] / d[0])
    assert zoomed == pytest.approx(flat, rel=1e-9)


def test_zoom_invariance_rolling_median_is_close_and_clip_median_is_not() -> None:
    d = open_close()
    ramp = np.linspace(0.7, 1.4, len(d))
    flat = hs.normalised_distance(make_table(d), "w9", "per_frame").values
    zoomed_df = make_table(d, zoom=ramp)
    rolling = hs.normalised_distance(zoomed_df, "w9", "rolling_median", window_s=3.0).values
    assert np.nanmax(np.abs(rolling / flat - 1)) < 0.05
    assert np.nanmedian(np.abs(rolling / flat - 1)) < 0.01
    clip_median = hs.normalised_distance(zoomed_df, "w9", "clip_median").values
    assert np.nanmax(np.abs(clip_median / flat - 1)) > 0.25  # cannot follow an approach


def test_zoom_changes_the_raw_signal_but_not_the_normalised_open_level() -> None:
    d = open_close()
    ramp = np.linspace(0.7, 1.4, len(d))
    df = make_table(d, zoom=ramp)
    raw = hs.thumb_index_distance_px(df)
    norm = hs.normalised_distance(df, "w9", "rolling_median").values
    first, last = slice(15, 75), slice(-75, -15)
    assert np.nanmax(raw[last]) / np.nanmax(raw[first]) > 1.7  # raw open level doubled
    assert np.nanmax(norm[last]) / np.nanmax(norm[first]) == pytest.approx(1.0, abs=0.06)


# --------------------------------------------------------------------------- NaN and gaps


def test_nan_in_gives_nan_out_never_zero() -> None:
    d = open_close(40)
    d[[5, 6, 20]] = np.nan
    for mode in hs.MODES:
        values = hs.normalised_distance(make_table(d), "w9", mode).values
        assert np.isnan(values[[5, 6, 20]]).all(), mode
        assert not (values == 0).any(), mode
        assert np.isfinite(values[[0, 10, 30]]).all(), mode


def test_a_frame_marked_not_detected_is_nan_even_with_coordinates() -> None:
    detected = np.ones(40, bool)
    detected[10] = False
    values = hs.normalised_distance(make_table(open_close(40), detected=detected)).values
    assert np.isnan(values[10]) and np.isfinite(values[11])
    assert np.isnan(hs.scale_length(make_table(open_close(40), detected=detected))[10])


def test_nan_in_the_scale_landmark_gives_nan_not_one() -> None:
    df = make_table(open_close(40))
    df.loc[7, "x_px09"] = np.nan
    assert np.isnan(hs.normalised_distance(df, "w9", "per_frame").values[7])
    # the other candidates need their own landmarks only
    assert np.isfinite(hs.normalised_distance(df, "w5", "per_frame").values[7])
    assert np.isnan(hs.normalised_distance(df, "median4", "per_frame").values[7])


def test_a_clip_with_no_hand_at_all_is_all_nan() -> None:
    df = make_table(np.full(30, np.nan))
    for mode in hs.MODES:
        assert np.isnan(hs.normalised_distance(df, "w9", mode).values).all()


# --------------------------------------------------------------------------- bad landmark


def bad_frame_table() -> pd.DataFrame:
    df = make_table(np.full(300, 50.0))
    df.loc[150, "y_px09"] = WRIST_XY[1] - 400.0  # knuckle 9 jumps: length 400 instead of 100
    return df


def test_one_bad_landmark_shows_in_per_frame_but_not_in_the_robust_modes() -> None:
    df = bad_frame_table()
    per_frame = hs.normalised_distance(df, "w9", "per_frame").values
    assert per_frame[150] == pytest.approx(0.125)  # 50 / 400: that frame is wrecked
    assert per_frame[149] == pytest.approx(0.5)
    for mode in ("rolling_median", "clip_median"):
        values = hs.normalised_distance(df, "w9", mode).values
        assert values == pytest.approx(np.full(300, 0.5)), mode


def test_a_collapsed_scale_is_rejected_and_counted() -> None:
    df = make_table(np.full(300, 50.0))
    df.loc[40, ["x_px09", "y_px09"]] = list(WRIST_XY)  # length exactly 0
    df.loc[41, ["x_px09", "y_px09"]] = [WRIST_XY[0] + 3.0, WRIST_XY[1]]  # length 3: absurdly small
    result = hs.normalised_distance(df, "w9", "per_frame")
    assert np.isnan(result.values[[40, 41]]).all()
    assert np.isfinite(result.values[[39, 42]]).all()
    assert result.scale_rejected_frames == 2


def test_guard_rules_on_a_plain_series() -> None:
    scale, rejected = hs._guard(np.array([100.0, -5.0, 0.0, np.nan, 24.0, 26.0, 100.0, 100.0]))
    assert np.isnan(scale[[1, 2, 3, 4]]).all()  # negative, zero, NaN, below 25% of the median
    assert scale[[0, 5, 6, 7]].tolist() == [100.0, 26.0, 100.0, 100.0]
    assert rejected == 3  # NaN input is missing, not rejected
    all_bad, count = hs._guard(np.array([0.0, -1.0, np.nan]))
    assert np.isnan(all_bad).all() and count == 2


def test_a_zero_scale_everywhere_never_gives_inf_or_zero() -> None:
    df = make_table(open_close(40))
    df["x_px09"], df["y_px09"] = df["x_px00"], df["y_px00"]
    for mode in hs.MODES:
        values = hs.normalised_distance(df, "w9", mode).values
        assert np.isnan(values).all(), mode


# --------------------------------------------------------------------------- the scale cannot follow the tap


def test_the_scale_never_uses_the_thumb_or_index_tip() -> None:
    used = {i for pairs in hs._PAIRS.values() for pair in pairs for i in pair}
    assert not used & set(hs.FORBIDDEN_LANDMARKS) and used == {0, 5, 9, 13, 17}
    still, moving = make_table(np.full(60, 20.0)), make_table(open_close(60))
    for name in hs.SCALES:
        assert hs.scale_length(still, name) == pytest.approx(hs.scale_length(moving, name)), name
    broken = make_table(open_close(60))
    for index in hs.FORBIDDEN_LANDMARKS:
        broken[f"x_px{index:02d}"] = np.nan
    assert np.isfinite(hs.scale_length(broken, "median4")).all()


# --------------------------------------------------------------------------- time base and arguments


def test_rolling_window_is_in_seconds_of_pts_not_in_frames() -> None:
    d = np.full(40, 50.0)
    sparse = make_table(d, pts=np.arange(40) * 1000.0)  # one frame per second
    dense = make_table(d)  # 30 fps
    assert np.isnan(hs.estimate_scale(sparse, "w9", "rolling_median", window_s=3.0)[0]).all()
    assert np.isfinite(hs.estimate_scale(dense, "w9", "rolling_median", window_s=3.0)[0][20])
    assert np.isfinite(hs.estimate_scale(sparse, "w9", "rolling_median", window_s=9.0)[0][20])


def test_frames_without_pts_get_no_rolling_scale() -> None:
    pts = np.arange(40) * 33.3
    pts[3] = np.nan
    scale, _ = hs.estimate_scale(make_table(np.full(40, 50.0), pts=pts), "w9", "rolling_median")
    assert np.isnan(scale[3]) and np.isfinite(scale[4])


def test_metadata_records_scale_mode_and_window() -> None:
    df = make_table(open_close(60))
    meta = hs.normalised_distance(df, "median4", "rolling_median", window_s=2.0).metadata()
    assert meta["scale"] == "median4" and meta["mode"] == "rolling_median"
    assert meta["window_s"] == 2.0 and meta["scale_median_px"] == pytest.approx(100.0)
    assert hs.normalised_distance(df, "w9", "per_frame").metadata()["window_s"] is None
    default = hs.normalised_distance(df)
    assert (default.scale, default.mode) == (hs.DEFAULT_SCALE, hs.DEFAULT_MODE)


@pytest.mark.parametrize(
    "kwargs",
    [{"scale": "w4"}, {"mode": "mean"}, {"window_s": 0.0}, {"window_s": float("nan")}],
)
def test_bad_arguments_are_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        hs.normalised_distance(make_table([1.0]), **kwargs)
    with pytest.raises(ValueError, match="space"):
        hs.scale_length(make_table([1.0]), "w9", "camera")


# --------------------------------------------------------------------------- privacy: figure and command line


def test_figure_has_the_hash_and_no_raw_id() -> None:
    fig = hs.plot_normalisation(make_table(open_close(90)), REF)
    text = text_of(fig)
    assert REF in text and "CONTROL" not in text and "9001" not in text


def write_landmarks(tmp_path: Path, stem: str = STEM) -> Path:
    path = tmp_path / "landmarks" / f"{stem}.parquet"
    hl.write_landmarks(make_table(open_close(90)), path)
    return path


def test_cli_prints_and_names_files_by_hash_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_landmarks(tmp_path)
    args = ["--clip", REF, "--landmarks-dir", str(tmp_path / "landmarks")]
    code = hs.main([*args, "--out-dir", str(tmp_path / "plots")])
    captured = capsys.readouterr()
    assert code == 0
    files = sorted(p.name for p in (tmp_path / "plots").iterdir())
    assert files == [f"{REF}_normalised_w9_rolling_median.png"]
    assert "CONTROL" not in captured.out + captured.err + " ".join(files)
    assert (tmp_path / "plots" / files[0]).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_cli_missing_clip_exits_2_and_corrupt_file_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_landmarks(tmp_path)
    folder = str(tmp_path / "landmarks")
    assert hs.main(["--clip", "deadbeef", "--landmarks-dir", folder]) == 2
    bad = tmp_path / "bad" / "CONTROL9002_IZDA.parquet"
    bad.parent.mkdir()
    bad.write_bytes(b"not a parquet file")
    code = hs.main(["--clip", pseudonym("CONTROL9002_IZDA"), "--landmarks-dir", str(bad.parent)])
    err = capsys.readouterr().err
    assert code == 1 and "FAILED" in err and "CONTROL9002" not in err


def test_cli_missing_columns_exit_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stem = "CONTROL9003_DCHA"
    df = make_table(open_close(60)).drop(columns=["x_px13"])
    hl.write_landmarks(df, tmp_path / f"{stem}.parquet")
    code = hs.main(
        ["--clip", pseudonym(stem), "--landmarks-dir", str(tmp_path), "--scale", "median4"]
    )
    assert code == 1 and "CONTROL9003" not in capsys.readouterr().err
