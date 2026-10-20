"""Tests for models/tapping/demo/tap_curve.py and the amplitude panel (task FT-50). The slope is
compared with the official features_b routine on the same clip; synthetic taps with known
trends; the lookup of research-clip tap tables never lists a folder."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent / "demo"
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(DEMO))
sys.path.insert(0, str(HERE))

import data_access as da  # noqa: E402
import features_b as fb  # noqa: E402
import streamlit as st  # noqa: E402
import tap_curve as tc  # noqa: E402
import tap_table  # noqa: E402
import taps  # noqa: E402
import upload_pipeline as up  # noqa: E402
from cv_splits import GuardFailure  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from test_demo_resources import add_model  # noqa: E402
from test_demo_score_app import (  # noqa: E402
    FORBIDDEN,
    fixed_sentences,
    start,
    strip_fixed,
    texts,
    upload,
)
from test_demo_upload import DATA, Factory, fake_pipeline  # noqa: E402
from test_shap_figures import REFS, make_world, text_problems  # noqa: E402

REF = "abcd1234"
SIGN_AFTER_BY = re.compile(r"by [-+−]")


@pytest.fixture(autouse=True)
def fresh_caches() -> None:
    st.cache_resource.clear()
    st.cache_data.clear()


@pytest.fixture
def upload_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    fake_pipeline(monkeypatch)
    video = tmp_path / "upload.mp4"
    video.write_bytes(DATA)
    return up.process_video(video, Factory())


def line(
    n: int, start_amp: float, slope: float, step: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(n) * step
    return t, start_amp + slope * t


# --------------------------------------------------------------------------- the slope


def test_slope_equals_the_official_amp_slope_on_a_clip(upload_result: Any) -> None:
    table = tap_table.build_tap_table(upload_result.taps_table)
    official = fb.clip_row(table, "x", "")["amp_slope"]
    curve = tc.curve_for_upload(upload_result)
    assert curve.usable and curve.slope_per10s_frac == official  # exact: the same routine
    assert curve.n_taps == int(table["valid_amp"].sum()) and curve.n_nan == 0
    valid = fb.valid_taps(table)
    assert np.array_equal(curve.amps, valid["amplitude"].to_numpy(float))
    assert np.array_equal(curve.times, valid["t_peak_s"].to_numpy(float))


def test_the_official_helpers_are_called(
    monkeypatch: pytest.MonkeyPatch, upload_result: Any
) -> None:
    calls: list[str] = []
    for name in ("slope_per_10s", "theil_sen_line", "valid_taps"):
        real = getattr(fb, name)

        def spy(*a: Any, _n: str = name, _r: Any = real, **k: Any) -> Any:
            calls.append(_n)
            return _r(*a, **k)

        monkeypatch.setattr(fb, name, spy)
    tc.curve_for_upload(upload_result)
    assert {"slope_per_10s", "theil_sen_line", "valid_taps"} <= set(calls)


def test_line_is_the_theil_sen_line_not_least_squares() -> None:
    t = np.arange(12) * 0.5
    a = 2.0 - 0.1 * t
    a[-1] = 6.0  # one bad tap: Theil-Sen ignores it, least squares does not
    curve = tc.tap_curve(t, a)
    slope, intercept = fb.theil_sen_line(t, a)
    assert curve.line_y[0] == pytest.approx(intercept + slope * t[0], abs=1e-12)
    assert curve.line_y[1] == pytest.approx(intercept + slope * t[-1], abs=1e-12)
    ols = np.polyfit(t, a, 1)[0]
    assert abs(slope - (-0.1)) < 1e-9 and abs(ols - (-0.1)) > 0.05


# --------------------------------------------------------------------------- the words


def test_falling_rising_and_flat_by_hand() -> None:
    t, a = line(20, 2.0, -0.05)
    fall = tc.tap_curve(t, a)
    mean = float(np.mean(a))
    assert fall.slope_per10s_frac == pytest.approx(-0.05 / mean * 10, abs=1e-12)
    assert tc.slope_sentence(fall) == (
        "Fitted line (Theil-Sen): amplitude changes by 28.4% of its mean per 10 s (falls)."
    )
    t, a = line(20, 1.0, 0.05)
    assert tc.slope_sentence(tc.tap_curve(t, a)) == (
        "Fitted line (Theil-Sen): amplitude changes by 40.4% of its mean per 10 s (rises)."
    )
    t, a = line(20, 1.5, 0.0)
    assert tc.slope_sentence(tc.tap_curve(t, a)) == (
        "Fitted line (Theil-Sen): amplitude changes by 0.0% of its mean per 10 s (is flat)."
    )


@pytest.mark.parametrize(
    ("slope", "word"),
    [(-0.0201, "falls"), (-0.02, "is flat"), (-0.0199, "is flat"), (0.0, "is flat"),
     (0.0199, "is flat"), (0.02, "is flat"), (0.0201, "rises")],
)  # fmt: skip
def test_display_thresholds(slope: float, word: str) -> None:
    c = tc.TapCurve(np.zeros(1), np.zeros(1), slope, np.zeros(2), np.zeros(2), 1, 1, 0, True, "")
    assert tc.slope_sentence(c).endswith(f"({word}).")


def test_no_sign_character_follows_by() -> None:
    for slope in (-0.9, -0.05, 0.0, 0.05, 0.9):
        t, a = line(20, 2.0 if slope <= 0 else 1.0, slope / 10 * 2.0)
        text = tc.slope_sentence(tc.tap_curve(t, a)) + " " + tc.count_sentence(tc.tap_curve(t, a))
        assert not SIGN_AFTER_BY.search(text), text
        assert re.search(r"by \d+\.\d% of its mean per 10 s", tc.slope_sentence(tc.tap_curve(t, a)))


def test_note_and_sentences_use_no_forbidden_words() -> None:
    assert not FORBIDDEN.search(tc.NOTE) and not FORBIDDEN.search(" ".join(tc.REASONS.values()))
    assert tc.NOTE == (
        "The score above uses Set A, which has no amplitude-trend feature. "
        "This curve explains the signal; it did not change the score."
    )


# --------------------------------------------------------------------------- minimum and NaN


def test_minimum_number_of_taps_is_the_official_one() -> None:
    assert tc.MIN_TAPS == fb.MIN_TAPS == 8
    t, a = line(fb.MIN_TAPS - 1, 2.0, -0.05)
    few = tc.tap_curve(t, a)
    assert not few.usable and len(few.line_x) == 0 and np.isnan(few.slope_per10s_frac)
    assert few.reason == "Fewer than 8 taps with a valid amplitude, so no line is fitted."
    assert tc.slope_sentence(few) == few.reason and few.n_taps == 7  # the points still show
    t, a = line(fb.MIN_TAPS, 2.0, -0.05)
    assert tc.tap_curve(t, a).usable


def test_non_positive_mean_gives_no_line() -> None:
    t, a = line(12, 0.0, 0.0)
    c = tc.tap_curve(t, a)
    assert not c.usable and "not positive" in c.reason


def test_a_missing_amplitude_is_left_out_not_zero() -> None:
    t, a = line(20, 2.0, -0.05)
    a_nan = a.copy()
    a_nan[7] = np.nan
    c = tc.tap_curve(t, a_nan)
    assert c.n_taps == 19 and c.n_found == 20 and c.n_nan == 1
    assert len(c.times) == len(c.amps) == 19 and 0.0 not in c.amps and not np.isnan(c.amps).any()
    assert t[7] not in c.times
    keep = np.arange(20) != 7
    assert c.slope_per10s_frac == fb.slope_per_10s(t[keep], a[keep])[0]  # the official one
    assert tc.count_sentence(c) == "Taps shown: 19 of 20. 1 without a valid amplitude left out."
    assert tc.count_sentence(tc.tap_curve(t, a)) == "Taps shown: 20 of 20."


def test_one_sided_taps_are_counted_as_left_out(upload_result: Any) -> None:
    table = tap_table.build_tap_table(upload_result.taps_table)
    table["valid_amp"] = True  # the synthetic clip has an edge tap of its own: start clean
    table.loc[[0, 5], "valid_amp"] = False  # two one-sided taps, as the research code flags them
    curve = tc.curve_for_table(table)
    assert curve.n_taps == len(table) - 2 and curve.n_found == len(table) and curve.n_edge == 2
    assert tc.count_sentence(curve) == (
        f"Taps shown: {len(table) - 2} of {len(table)}. "
        "2 taps at the clip edges left out (one-sided amplitude)."
    )
    assert curve.slope_per10s_frac == fb.clip_row(table, "x", "")["amp_slope"]


def test_a_single_edge_tap_reads_in_the_singular(upload_result: Any) -> None:
    table = tap_table.build_tap_table(upload_result.taps_table)
    table["valid_amp"] = True
    table.loc[3, "valid_amp"] = False
    assert tc.count_sentence(tc.curve_for_table(table)).endswith(
        "1 tap at the clip edges left out (one-sided amplitude)."
    )


def test_the_display_offset_moves_the_times_not_the_slope() -> None:
    t, a = line(20, 2.0, -0.05)
    base, shifted = tc.tap_curve(t, a), tc.tap_curve(t, a, time_offset_s=1.25)
    assert shifted.slope_per10s_frac == base.slope_per10s_frac
    assert np.array_equal(shifted.times, t + 1.25) and np.array_equal(
        shifted.line_x, base.line_x + 1.25
    )


# --------------------------------------------------------------------------- the figure


def test_figure_axes_legend_title_and_line() -> None:
    t, a = line(20, 2.0, -0.05, step=0.6)
    c = tc.tap_curve(t, a)
    fig = tc.curve_figure(c, REF)
    fig.canvas.draw()
    ax = fig.axes[0]
    assert np.array_equal(c.line_x, [t[0], t[-1]])  # spans the first to the last tap time
    xs = [np.asarray(ln.get_xdata()) for ln in ax.get_lines()]
    assert all(np.array_equal(x, t) for x in xs[:2])  # seconds, not tap index 0..19
    assert not np.array_equal(xs[0], np.arange(20)) and np.array_equal(xs[2], c.line_x)
    labels = [t_.get_text() for t_ in fig.findobj(Text) if t_.get_text()]
    assert "time (s)" in labels and "tap amplitude (hand sizes)" in labels
    titles = [x for x in labels if x.startswith("Per-tap amplitude")]
    assert titles == [f"Per-tap amplitude (clip {REF})"]
    assert fig.legends and not ax.get_legend()
    legend, box = fig.legends[0].get_window_extent(), ax.get_window_extent()
    assert legend.y0 >= box.y1 - 1 or legend.x0 >= box.x1 - 1  # outside the data area
    assert "fitted line (Theil-Sen)" in labels and "tap amplitude" in labels
    assert not text_problems(fig)
    assert all(float(x.get_fontsize()) >= 11 for x in fig.findobj(Text) if x.get_text())


def test_unusable_curve_draws_points_without_a_line() -> None:
    t, a = line(5, 2.0, -0.05)
    fig = tc.curve_figure(tc.tap_curve(t, a), REF)
    assert len(fig.axes[0].get_lines()) == 2
    assert "fitted line (Theil-Sen)" not in [x.get_text() for x in fig.findobj(Text)]


def test_figure_bytes_are_deterministic() -> None:
    t, a = line(20, 2.0, -0.05)
    c = tc.tap_curve(t, a)
    assert up.figure_png(tc.curve_figure(c, REF)) == up.figure_png(tc.curve_figure(c, REF))


# --------------------------------------------------------------------------- research-clip lookup


def write_taps_file(proc: Path, ref: str, table: pd.DataFrame, name_ref: str | None = None) -> Path:
    path = proc.parent / "interim" / "taps" / f"{name_ref or ref}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    taps.write_taps(taps.TapResult(table.assign(ref=ref), {}, "ok"), path, {})
    return path


def test_heldout_table_is_found_by_hashed_ref_without_listing_a_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, upload_result: Any
) -> None:
    w = make_world(tmp_path, monkeypatch)
    write_taps_file(w["proc"], REFS[0], upload_result.taps_table)
    # a second file that must never be touched
    write_taps_file(w["proc"], REFS[1], upload_result.taps_table)

    def forbidden(*a: Any, **k: Any) -> Any:
        raise AssertionError("a folder was listed")

    for owner, name in ((Path, "iterdir"), (Path, "glob"), (Path, "rglob"), (os, "listdir"),
                        (os, "scandir"), (os, "walk")):  # fmt: skip
        monkeypatch.setattr(owner, name, forbidden)
    table = da.heldout_tap_table(REFS[0], w["proc"])
    assert set(table["ref"]) == {REFS[0]} and len(table) == len(upload_result.taps_table)


def test_heldout_table_errors_are_friendly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, upload_result: Any
) -> None:
    w = make_world(tmp_path, monkeypatch)
    with pytest.raises(da.DemoDataError, match="not available for this clip"):
        da.heldout_tap_table(REFS[0], w["proc"])
    with pytest.raises(da.DemoDataError, match="unknown clip ref"):
        da.heldout_tap_table("CONTROL01_DCHA", w["proc"])  # a raw id is never looked up
    write_taps_file(w["proc"], REFS[2], upload_result.taps_table, name_ref=REFS[0])
    with pytest.raises(GuardFailure, match="another clip"):
        da.heldout_tap_table(REFS[0], w["proc"])
    bad = w["proc"].parent / "interim" / "taps" / f"{REFS[3]}.parquet"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"not parquet")
    with pytest.raises(da.DemoDataError, match="not available for this clip"):
        da.heldout_tap_table(REFS[3], w["proc"])


def test_new_code_never_lists_folders_or_names_the_landmark_folder() -> None:
    for name in ("tap_curve.py", "data_access.py", "app.py", "scoring.py", "upload_pipeline.py"):
        src = (DEMO / name).read_text()
        assert not re.search(r"\.(iterdir|glob|rglob)\(|listdir|scandir|os\.walk", src), name
        assert not re.search(r"[\"']landmarks[\"']|data/raw|fis_diagnostic", src), name


# --------------------------------------------------------------------------- the app


def test_upload_shows_the_amplitude_curve_with_words_and_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = upload(start(tmp_path, monkeypatch))
    assert not app.exception
    assert [h.value for h in app.header][-1] == "Per-tap amplitude"
    markdown = [m.value for m in app.markdown]
    assert any(re.match(r"Taps shown: \d+ of \d+\.", m) for m in markdown)
    words = [m for m in markdown if m.startswith("Fitted line (Theil-Sen)")]
    assert len(words) == 1 and re.search(r"\((falls|rises|is flat)\)\.$", words[0])
    assert not SIGN_AFTER_BY.search(words[0])
    assert tc.NOTE in [c.value for c in app.caption]
    assert len(app.image) == 9  # 6 frames, signal, SHAP and the amplitude curve


def test_example_mode_shows_the_curve_when_the_clip_has_a_tap_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, upload_result: Any
) -> None:
    w = make_world(tmp_path, monkeypatch)
    add_model(w["proc"])
    monkeypatch.setenv(da.ENV_ROOT, str(w["proc"]))
    write_taps_file(w["proc"], REFS[0], upload_result.taps_table)
    app = AppTest.from_file(str(DEMO / "app.py"), default_timeout=60).run()
    assert not app.exception and len(app.image) == 2  # SHAP figure and the amplitude curve
    assert tc.NOTE in [c.value for c in app.caption]
    assert any(m.value.startswith("Fitted line (Theil-Sen)") for m in app.markdown)


def test_example_mode_says_so_when_the_tap_table_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    w = make_world(tmp_path, monkeypatch)
    add_model(w["proc"])
    monkeypatch.setenv(da.ENV_ROOT, str(w["proc"]))
    app = AppTest.from_file(str(DEMO / "app.py"), default_timeout=60).run()
    assert not app.exception and len(app.image) == 1  # the SHAP figure only
    assert "The per-tap curve is not available for this clip." in [i.value for i in app.info]
    assert [h.value for h in app.header][-1] == "Per-tap amplitude"


def test_no_raw_id_or_file_name_and_no_forbidden_words_on_the_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = start(tmp_path, monkeypatch)
    pages = {"example": texts(app)}
    app.sidebar.radio[0].set_value("Upload video").run()
    pages["upload"] = texts(
        app.file_uploader[0].set_value(("PATIENT_12345_visit.mp4", DATA, "video/mp4")).run()
    )
    for name, items in pages.items():
        whole = "\n".join(items)
        assert not re.search(r"PATIENT|12345|visit|\.mp4|CONTROL\d|ID\d{4}_", whole), name
        assert not FORBIDDEN.search(strip_fixed(whole, "ge2")), name
    assert all(s in "\n".join(pages["upload"]) for s in fixed_sentences("ge2")[:1])
