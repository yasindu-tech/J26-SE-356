"""Tests for models/tapping/src/plot_distance.py (task FT-12).

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
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hand_landmarks as hl  # noqa: E402
import plot_distance as pdist  # noqa: E402
from hubu_fis import pseudonym  # noqa: E402

STEM = "CONTROL9001_DCHA"
REF = pseudonym(STEM)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def make_table(
    distances: Any,
    *,
    pts: Any = None,
    timestamps: Any = None,
    detected: Any = None,
    hand: float = 200.0,
) -> pd.DataFrame:
    """A landmarks table whose thumb-index distance is ``distances`` (NaN = no hand).

    Thumb tip at (500, 600); index tip at a 3-4-5 offset from it, so the distance is exact.
    Landmark 12 is placed far away on purpose: a function that used it would give a wrong
    answer. Wrist at (400, 900), knuckle 9 ``hand`` pixels above it.
    """
    d = np.asarray(distances, dtype=float)
    n = len(d)
    df = pd.DataFrame(np.nan, index=range(n), columns=list(hl.COLUMNS))
    df["frame"] = np.arange(n)
    df["timestamp_ms"] = (
        np.round(np.arange(n) * 1000 / 30).astype(np.int64) if timestamps is None else timestamps
    )
    df["pts_ms"] = df["timestamp_ms"].astype(float) if pts is None else np.asarray(pts, float)
    ok = np.isfinite(d) if detected is None else np.asarray(detected, bool)
    df["detected"] = ok
    live = np.isfinite(d)
    df.loc[live, "x_px04"], df.loc[live, "y_px04"] = 500.0, 600.0
    df.loc[live, "x_px08"] = 500.0 + 0.6 * d[live]
    df.loc[live, "y_px08"] = 600.0 + 0.8 * d[live]
    df.loc[live, "x_px12"], df.loc[live, "y_px12"] = 2000.0, 100.0
    df.loc[live, "x_px00"], df.loc[live, "y_px00"] = 400.0, 900.0
    df.loc[live, "x_px09"], df.loc[live, "y_px09"] = 400.0, 900.0 - hand
    df.attrs["hand_landmarks"] = {"fps": 30.0, "width": 1080, "height": 1920}
    return df


def line_y(line: Any) -> np.ndarray:
    return np.asarray(line.get_ydata(), dtype=float)


def line_x(line: Any) -> np.ndarray:
    return np.asarray(line.get_xdata(), dtype=float)


def text_of(fig: Any) -> str:
    return " ".join(t.get_text() for t in fig.findobj(Text))


# --------------------------------------------------------------------------- the distance


def test_distance_on_a_known_geometry() -> None:
    d = pdist.thumb_index_distance_px(make_table([5.0, 50.0, 123.5]))
    assert d == pytest.approx([5.0, 50.0, 123.5])


def test_distance_uses_landmarks_4_and_8_not_others() -> None:
    df = make_table([30.0])
    assert pdist.thumb_index_distance_px(df)[0] == pytest.approx(30.0)
    assert pdist.thumb_index_distance_px(df, thumb=4, index=12)[0] > 1000  # landmark 12 is far
    assert (pdist.THUMB_TIP, pdist.INDEX_TIP) == (4, 8)


def test_nan_in_gives_nan_out_never_zero() -> None:
    df = make_table([10.0, np.nan, np.nan, 40.0, np.nan])
    d = pdist.thumb_index_distance_px(df)
    assert np.isnan(d[[1, 2, 4]]).all()
    assert not (d == 0).any()
    assert d[0] == pytest.approx(10.0) and d[3] == pytest.approx(40.0)


def test_a_frame_marked_not_detected_is_nan_even_with_coordinates() -> None:
    df = make_table([10.0, 20.0, 30.0], detected=[True, False, True])
    d = pdist.thumb_index_distance_px(df)
    assert np.isnan(d[1]) and d[0] == pytest.approx(10.0) and d[2] == pytest.approx(30.0)


@pytest.mark.parametrize("n", [1, 7, 100])
def test_output_length_equals_the_number_of_frames(n: int) -> None:
    assert len(pdist.thumb_index_distance_px(make_table(np.linspace(1, 50, n)))) == n


def test_hand_size_is_the_median_wrist_to_knuckle_distance() -> None:
    df = make_table([10.0, np.nan, 20.0], hand=210.0)
    assert pdist.hand_size_px(df) == pytest.approx(210.0)
    assert np.isnan(pdist.hand_size_px(make_table([np.nan, np.nan])))


# --------------------------------------------------------------------------- the time axis


def test_time_axis_uses_pts_ms_not_timestamp_ms() -> None:
    df = make_table([1.0, 2.0, 3.0], pts=[0.0, 40.0, 91.0], timestamps=[0, 33, 67])
    t, label = pdist.time_axis_s(df)
    assert t.tolist() == pytest.approx([0.0, 0.040, 0.091])
    assert t.tolist() != pytest.approx([0.0, 0.033, 0.067])
    assert "pts_ms" in label and "missing" not in label
    t_index, label_index = pdist.time_axis_s(df, "index")
    assert t_index.tolist() == pytest.approx([0.0, 0.033, 0.067]) and "timestamp_ms" in label_index


def test_missing_pts_everywhere_falls_back_and_says_so() -> None:
    df = make_table([1.0, 2.0], pts=[np.nan, np.nan], timestamps=[0, 33])
    t, label = pdist.time_axis_s(df)
    assert t.tolist() == pytest.approx([0.0, 0.033])
    assert "pts_ms was missing" in label


def test_missing_pts_for_some_frames_leaves_those_frames_without_time() -> None:
    df = make_table([1.0, 2.0, 3.0], pts=[0.0, np.nan, 66.0])
    t, label = pdist.time_axis_s(df)
    assert np.isnan(t[1]) and t[2] == pytest.approx(0.066) and "missing" not in label


def test_unknown_clock_is_rejected() -> None:
    with pytest.raises(ValueError, match="clock"):
        pdist.time_axis_s(make_table([1.0]), "wall")


# --------------------------------------------------------------------------- exploratory numbers


def test_exploratory_numbers_on_a_small_signal() -> None:
    df = make_table([10.0, 20.0, 30.0, np.nan, 40.0], pts=[0, 100, 200, 300, 400], hand=200.0)
    s = pdist.exploratory_numbers(df)
    assert (s["min_px"], s["median_px"], s["max_px"]) == (10.0, 25.0, 40.0)
    assert (s["min_hand"], s["median_hand"], s["max_hand"]) == (0.05, 0.125, 0.2)
    assert s["nan_frames"] == 1 and s["pts_missing"] == 0
    assert s["duration_s"] == pytest.approx(0.4)
    assert s["hand_size_px"] == pytest.approx(200.0)
    assert (s["pts_gap_ms_min"], s["pts_gap_ms_max"]) == (100.0, 100.0)


def sine(
    cycles_hz: float,
    seconds: float,
    fps: float = 60.0,
    centre: float = 100.0,
    amplitude: float = 100.0,
) -> np.ndarray:
    t = np.arange(int(seconds * fps)) / fps
    return centre + amplitude * np.sin(2 * np.pi * cycles_hz * t)


def test_cycle_count_on_a_clean_oscillation() -> None:
    assert pdist.count_open_close_cycles(sine(2.0, 10.0)) in (19, 20)


def test_cycle_count_is_zero_for_a_flat_or_empty_signal() -> None:
    assert pdist.count_open_close_cycles(np.full(100, 50.0)) == 0
    assert pdist.count_open_close_cycles(np.full(100, np.nan)) == 0
    assert pdist.count_open_close_cycles(np.array([5.0])) == 0


def test_cycle_count_skips_gaps_without_filling_them() -> None:
    signal = sine(2.0, 10.0)
    gapped = signal.copy()
    gapped[200:260] = np.nan
    assert abs(pdist.count_open_close_cycles(gapped) - pdist.count_open_close_cycles(signal)) <= 3


def test_cycle_count_keeps_taps_whose_peaks_differ_in_height() -> None:
    """Twelve taps with peaks alternating between 230 and 160 px, starting from the floor.

    A 75% open level would count only the six tall ones; the midpoint level counts all twelve.
    """
    bump = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(40) / 40)  # 0 -> 1 -> 0 over one tap
    signal = np.concatenate([20 + (peak - 20) * bump for peak in [230, 160] * 6])
    assert pdist.count_open_close_cycles(signal) == 12


def test_cycle_count_misses_taps_that_stay_small_a_documented_limit() -> None:
    """Ten full taps, then five small ones that never reach the 25% / 75% levels."""
    signal = np.concatenate([sine(2.0, 5.0), sine(2.0, 2.5, amplitude=30.0)])
    assert 9 <= pdist.count_open_close_cycles(signal) <= 10


# --------------------------------------------------------------------------- the figure


def test_main_line_is_the_raw_signal_with_gaps_as_breaks() -> None:
    df = make_table([10.0, 40.0, np.nan, np.nan, 25.0, 60.0])
    fig = pdist.plot_distance(df, REF)
    ydata = line_y(fig.axes[0].lines[0])
    assert np.allclose(ydata, pdist.thumb_index_distance_px(df), equal_nan=True)
    assert np.isnan(ydata[[2, 3]]).all()
    assert (ydata[np.isfinite(ydata)] > 0).all()  # no zero was drawn for a gap
    assert len(fig.axes[0].collections) >= 1  # the red ticks for the undetected frames


def test_hand_size_is_a_reference_line_and_does_not_replace_the_signal() -> None:
    df = make_table([10.0, 40.0, 25.0], hand=200.0)
    fig = pdist.plot_distance(df, REF)
    ax = fig.axes[0]
    assert np.allclose(line_y(ax.lines[0]), [10.0, 40.0, 25.0])  # still pixels
    reference = [ln for ln in ax.lines[1:] if "reference only" in str(ln.get_label())]
    assert len(reference) == 1 and line_y(reference[0]).tolist() == [200.0, 200.0]
    assert "reference scale only" in text_of(fig)


def test_figure_is_drawn_on_the_time_axis_from_pts_ms() -> None:
    df = make_table([10.0, 40.0, 25.0], pts=[0.0, 50.0, 120.0], timestamps=[0, 33, 67])
    xdata = line_x(pdist.plot_distance(df, REF).axes[0].lines[0])
    assert np.allclose(xdata, [0.0, 0.05, 0.12])


def test_figure_text_has_the_hash_and_never_the_raw_id() -> None:
    df = make_table([10.0, 40.0, np.nan, 25.0])
    for fig in (pdist.plot_distance(df, REF), pdist.plot_clock_comparison(df, REF)):
        text = text_of(fig)
        assert REF in text and "CONTROL9001" not in text and STEM not in text


def test_figure_uses_an_agg_canvas_so_no_display_is_needed() -> None:
    assert isinstance(pdist.plot_distance(make_table([1.0, 2.0]), REF).canvas, FigureCanvasAgg)


def test_clock_comparison_has_two_panels_with_both_clocks() -> None:
    n = 180  # 6 s at 30 fps, with the clocks drifting apart by up to 20 ms
    df = make_table(sine(2.0, 6.0, fps=30.0), pts=np.arange(n) * 1000 / 30 + np.linspace(0, 20, n))
    fig = pdist.plot_clock_comparison(df, REF)
    assert len(fig.axes) == 2 and all(len(ax.lines) == 2 for ax in fig.axes)
    assert "largest difference" in text_of(fig)
    low, high = fig.axes[1].get_xlim()
    assert high - low == pytest.approx(pdist.ZOOM_SECONDS)


def test_clock_comparison_zoom_stays_inside_the_clip() -> None:
    """The largest difference is at the very start here: the window must not run to negative time."""
    n = 180
    pts = np.arange(n) * 1000 / 30
    pts[1:] += 25.0  # one long first gap: the clocks differ most right at the start
    fig = pdist.plot_clock_comparison(make_table(sine(2.0, 6.0, fps=30.0), pts=pts), REF)
    low, high = fig.axes[1].get_xlim()
    assert low >= 0.0 and high <= 6.0 and high - low == pytest.approx(pdist.ZOOM_SECONDS)


def test_clock_comparison_zoom_is_shortened_for_a_short_clip() -> None:
    fig = pdist.plot_clock_comparison(make_table(sine(2.0, 3.0, fps=30.0)), REF)
    low, high = fig.axes[1].get_xlim()
    assert low >= 0.0 and high - low <= 3.0 + 1e-9


def test_a_clip_with_no_detections_still_plots() -> None:
    fig = pdist.plot_distance(make_table([np.nan, np.nan, np.nan]), REF)
    assert np.isnan(line_y(fig.axes[0].lines[0])).all()


# --------------------------------------------------------------------------- command line


def write_landmarks(tmp_path: Path, df: pd.DataFrame | None = None, stem: str = STEM) -> Path:
    df = make_table(sine(2.0, 3.0, fps=30.0)) if df is None else df
    path = tmp_path / "landmarks" / f"{stem}.parquet"
    hl.write_landmarks(df, path)
    return path


def test_cli_by_hash_saves_a_png_named_by_the_hash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_landmarks(tmp_path)
    code = pdist.main(
        [
            "--clip",
            REF,
            "--landmarks-dir",
            str(tmp_path / "landmarks"),
            "--out-dir",
            str(tmp_path / "plots"),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    files = sorted(p.name for p in (tmp_path / "plots").iterdir())
    assert files == [f"{REF}_distance.png"]
    assert (tmp_path / "plots" / files[0]).read_bytes()[:8] == PNG_SIGNATURE
    assert "CONTROL9001" not in " ".join(files) + captured.out + captured.err
    assert REF in captured.out and "not the tap detector" in captured.out


def test_cli_by_parquet_path_and_compare_clocks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_landmarks(tmp_path)
    code = pdist.main(
        ["--parquet", str(path), "--out-dir", str(tmp_path / "plots"), "--compare-clocks"]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert sorted(p.name for p in (tmp_path / "plots").iterdir()) == [
        f"{REF}_distance.png",
        f"{REF}_distance_clocks.png",
    ]
    assert STEM not in captured.out + captured.err


def test_cli_missing_clip_hash_exits_2(tmp_path: Path) -> None:
    write_landmarks(tmp_path)
    code = pdist.main(
        [
            "--clip",
            "deadbeef",
            "--landmarks-dir",
            str(tmp_path / "landmarks"),
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert code == 2


def test_cli_missing_parquet_exits_2_with_the_hash_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = pdist.main(["--parquet", str(tmp_path / f"{STEM}.parquet"), "--out-dir", str(tmp_path)])
    err = capsys.readouterr().err
    assert code == 2 and REF in err and "CONTROL9001" not in err
    assert not list(tmp_path.glob("*.png"))


def test_cli_corrupt_file_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / f"{STEM}.parquet"
    bad.write_bytes(np.random.default_rng(0).bytes(2048))
    assert pdist.main(["--parquet", str(bad), "--out-dir", str(tmp_path / "plots")]) == 1
    assert "FAILED" in capsys.readouterr().err and not (tmp_path / "plots").exists()


def test_cli_file_without_our_metadata_exits_1(tmp_path: Path) -> None:
    other = tmp_path / f"{STEM}.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(other)
    assert pdist.main(["--parquet", str(other), "--out-dir", str(tmp_path / "plots")]) == 1


def test_cli_missing_columns_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_landmarks(tmp_path, make_table([1.0, 2.0]).drop(columns=["pts_ms"]))
    assert pdist.main(["--parquet", str(path), "--out-dir", str(tmp_path / "plots")]) == 1
    assert "lacks columns" in capsys.readouterr().err


def test_cli_needs_exactly_one_source(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        pdist.main(["--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        pdist.main(["--clip", REF, "--parquet", str(tmp_path / "x.parquet")])


def test_find_parquet_matches_by_hash(tmp_path: Path) -> None:
    path = write_landmarks(tmp_path)
    assert pdist.find_parquet(REF, tmp_path / "landmarks") == path
    assert pdist.find_parquet("00000000", tmp_path / "landmarks") is None
