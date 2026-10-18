"""Tests for models/tapping/src/qc_sheet.py (task FT-21).

Synthetic data only: made-up signals under a temporary ``data/interim`` (``INTERIM_ROOT`` is
patched, nothing touches the real one). The manifest in these tests carries poison values in the
columns the tool must never read (raw ID, group, hand, participant). No conftest.py on purpose.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import qc_sheet as qc  # noqa: E402
import signal_processing as sp  # noqa: E402
import smoothing as sm  # noqa: E402

POISON = ("CONTROL9999_DCHA", "GROUP_POISON", "HAND_POISON", "PARTICIPANT_POISON", "UPDRS")


def refs_of(n: int) -> list[str]:
    return [f"{i:08x}" for i in range(n)]


@pytest.fixture
def interim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "interim"
    (root / "smoothed").mkdir(parents=True)
    monkeypatch.setattr(qc, "INTERIM_ROOT", root)
    return root


def tap_signal(n: int = 150, peak: float = 1.5) -> np.ndarray:
    t = np.arange(n) / 30.0
    return 0.12 + (peak - 0.12) * (0.5 - 0.5 * np.cos(2 * np.pi * 1.5 * t))


def write_clip(
    interim: Path, ref: str, value: np.ndarray, filled: np.ndarray | None = None
) -> None:
    n = len(value)
    fl = np.zeros(n, dtype=bool) if filled is None else np.asarray(filled, dtype=bool)
    table = pd.DataFrame(
        {
            "frame": np.arange(n, dtype=np.int64),
            "t_s": np.arange(n) / 30.0,
            "value": value,
            "measured": np.where(fl, np.nan, value),
            "filled": fl,
            "scale_px": np.full(n, 100.0),
        },
        columns=list(sp.TABLE_COLUMNS),
    )
    meta = {"clip_ref": ref, "trimmed_start_s": 0.25, "trimmed_end_s": 0.5}
    smoothed = sm.smooth_signal(sp.Signal(ref, table, meta))
    sm.write_smoothed(smoothed, interim / "smoothed" / f"{ref}.parquet", "x")


def write_manifest(interim: Path, refs: list[str], flags: dict[str, str] | None = None) -> Path:
    flags = flags or {}
    df = pd.DataFrame(
        {
            "clip_id": [POISON[0]] * len(refs),
            "ref": refs,
            "participant_ref": [POISON[3]] * len(refs),
            "group": [POISON[1]] * len(refs),
            "hand": [POISON[2]] * len(refs),
            "duration_s": [5.0] * len(refs),
            "fps": [30.0] * len(refs),
            "detection_rate": [0.97] * len(refs),
            "longest_missing_run": [3] * len(refs),
            "flags": [flags.get(r, "") for r in refs],
            "scale_tap_abs_r": [0.1] * len(refs),
        }
    )
    path = interim / "tracking_manifest.csv"
    df.to_csv(path, index=False)
    return path


def make_world(interim: Path, n: int, **kwargs: Any) -> list[str]:
    refs = refs_of(n)
    for r in refs:
        write_clip(interim, r, tap_signal())
    write_manifest(interim, refs, **kwargs)
    return refs


def panels_for(interim: Path, refs: list[str]) -> list[qc.Panel]:
    manifest = qc.read_manifest(interim / "tracking_manifest.csv")
    return [qc.load_panel(r, interim / "smoothed", manifest) for r in refs]


# --------------------------------------------------------------------------- selection


def test_same_seed_same_24_different_seed_different() -> None:
    refs = refs_of(234)
    first = qc.sample_refs(refs, 24, 21)
    assert first == qc.sample_refs(refs, 24, 21)
    assert first == qc.sample_refs(list(reversed(refs)), 24, 21)  # input order does not matter
    assert first != qc.sample_refs(refs, 24, 22)
    assert len(first) == len(set(first)) == 24 and first == sorted(first)
    assert set(first) <= set(refs)


def test_sample_size_is_validated() -> None:
    with pytest.raises(ValueError):
        qc.sample_refs(refs_of(234), 25, 21)
    with pytest.raises(ValueError):
        qc.sample_refs(refs_of(234), 0, 21)
    with pytest.raises(ValueError):
        qc.sample_refs(refs_of(10), 11, 21)


def test_234_clips_make_10_pages_the_last_with_18() -> None:
    refs = refs_of(234)
    assert qc.n_pages(234) == 10 and qc.n_pages(24) == 1 and qc.n_pages(25) == 2
    pages = [qc.page_refs(refs, k) for k in range(1, 11)]
    assert [len(p) for p in pages] == [24] * 9 + [18]
    flat = [r for p in pages for r in p]
    assert flat == sorted(refs) and len(set(flat)) == 234  # every clip once, in hash order
    shuffled = list(reversed(refs))
    assert qc.page_refs(shuffled, 10) == pages[-1]  # sorted before paging
    for bad in (0, 11):
        with pytest.raises(ValueError):
            qc.page_refs(refs, bad)


# --------------------------------------------------------------------------- numbers and hints


def test_qc_numbers_on_known_signals() -> None:
    ramp = np.linspace(0.0, 1.0, 101)
    n = qc.qc_numbers(ramp)
    assert n["range"] == pytest.approx(0.9) and n["floor"] == pytest.approx(0.05)
    assert n["max"] == pytest.approx(1.0)
    with_nan = np.concatenate([ramp, [np.nan] * 20])
    assert qc.qc_numbers(with_nan) == n  # NaN is ignored, not counted as 0
    empty = qc.qc_numbers(np.full(5, np.nan))
    assert all(np.isnan(v) for v in empty.values()) and qc.hints_for(empty) == []


def test_hints_fire_exactly_at_their_thresholds() -> None:
    ok = {"range": 1.0, "floor": 0.1, "max": 1.5}
    assert qc.hints_for(ok) == []
    assert qc.hints_for({**ok, "range": 0.59}) == ["LOW RANGE"]
    assert qc.hints_for({**ok, "range": 0.6}) == []
    assert qc.hints_for({**ok, "floor": 0.31}) == ["FLOOR HIGH"]
    assert qc.hints_for({**ok, "floor": 0.3}) == []
    assert qc.hints_for({**ok, "max": 2.41}) == ["ABOVE AXIS"]
    assert qc.hints_for({**ok, "max": 2.4}) == []
    assert qc.hints_for({"range": 0.2, "floor": 0.5, "max": 3.0}) == [
        "LOW RANGE",
        "FLOOR HIGH",
        "ABOVE AXIS",
    ]


# --------------------------------------------------------------------------- drawing


def test_nan_stays_nan_in_the_plotted_line_and_a_gap_span_is_drawn(interim: Path) -> None:
    value = tap_signal(120)
    value[40:55] = np.nan
    value[80:90] = np.nan
    write_clip(interim, "0000000a", value)
    write_manifest(interim, ["0000000a"])
    panel = panels_for(interim, ["0000000a"])[0]
    fig = qc.make_figure([panel], "t")
    ax = fig.axes[0]
    smooth_line = np.asarray(ax.lines[1].get_ydata(), dtype=float)
    assert np.isnan(smooth_line[40:55]).all() and np.isnan(smooth_line[80:90]).all()
    assert not (smooth_line == 0).any()  # never drawn as zero
    assert len(ax.patches) == 2  # one shaded span per NaN run


def test_filled_samples_are_drawn_as_red_dots(interim: Path) -> None:
    filled = np.zeros(150, dtype=bool)
    filled[30:33] = True
    write_clip(interim, "0000000a", tap_signal(), filled)
    write_manifest(interim, ["0000000a"])
    ax = qc.make_figure(panels_for(interim, ["0000000a"]), "t").axes[0]
    assert len(ax.collections) == 1 and np.asarray(ax.collections[0].get_offsets()).shape[0] == 3


def test_x_span_is_the_clips_own_duration(interim: Path) -> None:
    write_clip(interim, "0000000a", tap_signal(60))
    write_clip(interim, "0000000b", tap_signal(240))
    write_manifest(interim, ["0000000a", "0000000b"])
    fig = qc.make_figure(panels_for(interim, ["0000000a", "0000000b"]), "t")
    assert fig.axes[0].get_xlim() == pytest.approx((0.0, 59 / 30.0))
    assert fig.axes[1].get_xlim() == pytest.approx((0.0, 239 / 30.0))


def test_fixed_axis_marker_and_autoscale(interim: Path) -> None:
    write_clip(interim, "0000000a", tap_signal(peak=3.0))
    write_clip(interim, "0000000b", tap_signal(peak=1.5))
    write_manifest(interim, ["0000000a", "0000000b"])
    panels = panels_for(interim, ["0000000a", "0000000b"])
    fixed = qc.make_figure(panels, "t")
    assert fixed.axes[0].get_ylim() == pytest.approx((qc.Y_MIN, qc.Y_MAX))
    assert any("max 3.0" in t.get_text() and "↑" in t.get_text() for t in fixed.axes[0].texts)
    assert not any("max" in t.get_text() and "↑" in t.get_text() for t in fixed.axes[1].texts)
    auto = qc.make_figure(panels, "t", yscale="auto")
    assert auto.axes[0].get_ylim()[1] > 2.4  # not forced to the fixed range
    assert not any("↑" in t.get_text() for t in auto.axes[0].texts)


def test_a_short_page_hides_the_extra_panels_and_keeps_positions(interim: Path) -> None:
    refs = make_world(interim, 30)
    chosen = qc.page_refs(refs, 2)
    panels = panels_for(interim, chosen)
    fig = qc.make_figure(panels, "t")
    assert [ax.get_visible() for ax in fig.axes] == [True] * 6 + [False] * 18
    for k, ref in enumerate(chosen):
        assert fig.axes[k].get_title().startswith(ref)
    table = qc.csv_rows(panels, "s", None, 2)
    assert table["ref"].tolist() == chosen and len(table) == 6
    assert table["panel"].tolist() == [1, 2, 3, 4, 5, 6]
    assert table["row"].tolist() == [1] * 6 and table["col"].tolist() == [1, 2, 3, 4, 5, 6]


def test_csv_positions_follow_the_draw_order() -> None:
    panels = [qc.Panel(r, qc.STATUS_OK) for r in refs_of(24)]
    table = qc.csv_rows(panels, "s", 21, None)
    for k in range(24):
        assert (table["row"][k], table["col"][k], table["panel"][k]) == (
            k // 6 + 1,
            k % 6 + 1,
            k + 1,
        )
    assert table["seed"].tolist() == [21] * 24 and (table["page"] == "").all()


def test_titles_show_numbers_and_skip_info_flags(interim: Path) -> None:
    write_clip(interim, "0000000a", tap_signal())
    write_clip(interim, "0000000b", tap_signal())
    write_manifest(
        interim,
        ["0000000a", "0000000b"],
        flags={
            "0000000a": "info_irregular_timing;scale_tap_coupling",
            "0000000b": "info_irregular_timing",
        },
    )
    fig = qc.make_figure(panels_for(interim, ["0000000a", "0000000b"]), "t")
    first, second = fig.axes[0].get_title(), fig.axes[1].get_title()
    assert "0000000a" in first and "5s" in first and "30 fps" in first and "det 0.97" in first
    assert "miss 3" in first and "scale_tap_coupling" in first and "fill 0" in first
    assert "trim 0.8s" in first and "gap 0" in first
    assert "info_" not in first and "info_" not in second


def agg_renderer(fig: Any) -> Any:
    canvas = fig.canvas
    assert isinstance(canvas, FigureCanvasAgg)  # make_figure attaches an Agg canvas
    canvas.draw()
    return canvas.get_renderer()


def visible_samples_under(ax: Any, bbox: Any) -> int:
    """How many DRAWN samples of the smoothed line lie under a text box (display coordinates)."""
    line = [ln for ln in ax.lines if ln.get_color() == "tab:blue"][0]
    xy = np.column_stack(
        [np.asarray(line.get_xdata(), dtype=float), np.asarray(line.get_ydata(), dtype=float)]
    )
    pts = ax.transData.transform(xy)
    axes_box = ax.get_window_extent()
    drawn = (
        np.isfinite(pts).all(axis=1)
        & (pts[:, 0] >= axes_box.x0)
        & (pts[:, 0] <= axes_box.x1)
        & (pts[:, 1] >= axes_box.y0)
        & (pts[:, 1] <= axes_box.y1)
    )
    inside = (
        (pts[:, 0] >= bbox.x0)
        & (pts[:, 0] <= bbox.x1)
        & (pts[:, 1] >= bbox.y0)
        & (pts[:, 1] <= bbox.y1)
    )
    return int((drawn & inside).sum())


def hinted_world(interim: Path) -> list[str]:
    """Clips that trigger every hint, one clip with a wide title, and ordinary clips."""
    refs = refs_of(24)
    values = {
        refs[0]: tap_signal(peak=3.0),  # ABOVE AXIS
        refs[1]: 0.12 + 0.2 * (0.5 - 0.5 * np.cos(np.linspace(0, 40, 150))),  # LOW RANGE
        refs[2]: 0.5
        + 0.4 * (0.5 - 0.5 * np.cos(np.linspace(0, 40, 150))),  # FLOOR HIGH + LOW RANGE
    }
    for r in refs:
        write_clip(interim, r, values.get(r, tap_signal()))
    flags = {refs[3]: "low_detection;gaps_not_fillable;landmarks_out_of_frame"}
    write_manifest(interim, refs, flags=flags)
    return refs


def test_no_text_sits_inside_the_plotting_area_or_covers_the_signal(interim: Path) -> None:
    refs = hinted_world(interim)
    panels = panels_for(interim, refs)
    fig = qc.make_figure(panels, "t")
    renderer = agg_renderer(fig)
    for ax, panel in zip(fig.axes, panels, strict=True):
        axes_box = ax.get_window_extent(renderer)
        title_box = ax.title.get_window_extent(renderer)
        assert title_box.y0 >= axes_box.y1 - 0.5  # the footer line lives in the title strip above
        assert visible_samples_under(ax, title_box) == 0
        if ax.get_xlabel():
            label_box = ax.xaxis.label.get_window_extent(renderer)
            assert label_box.y1 <= axes_box.y0 + 0.5  # the hints live below the x axis
            assert visible_samples_under(ax, label_box) == 0
        inside = [t.get_text() for t in ax.texts]
        assert all("\u2191" in t for t in inside)  # only the red max marker may be inside
        footer = ax.get_title().splitlines()[-1]
        assert footer.startswith("range ") and "floor " in footer and "max " in footer
        assert ax.get_xlabel() == "  ".join(panel.hints)
    assert fig.axes[0].get_xlabel() == "ABOVE AXIS"
    assert fig.axes[1].get_xlabel() == "LOW RANGE"
    assert fig.axes[2].get_xlabel() == "LOW RANGE  FLOOR HIGH"
    assert fig.axes[5].get_xlabel() == ""


def test_titles_and_labels_do_not_overlap_other_panels(interim: Path) -> None:
    refs = hinted_world(interim)
    fig = qc.make_figure(panels_for(interim, refs), "t")
    renderer = agg_renderer(fig)
    boxes = [
        [ax.title.get_window_extent(renderer), ax.xaxis.label.get_window_extent(renderer)
         if ax.get_xlabel() else None, ax.get_window_extent(renderer)]
        for ax in fig.axes
    ]  # fmt: skip

    def overlap(a: Any, b: Any) -> bool:
        return (
            a is not None
            and b is not None
            and a.x0 < b.x1
            and b.x0 < a.x1
            and a.y0 < b.y1
            and b.y0 < a.y1
        )

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ti, li, ai = boxes[i]
            tj, lj, aj = boxes[j]
            assert not any(overlap(p, q) for p in (ti, li) for q in (tj, lj, aj))
            assert not any(overlap(p, q) for p in (tj, lj) for q in (ti, li, ai))
    width, height = fig.bbox.width, fig.bbox.height
    for ti, li, _ in boxes:
        for b in (ti, li):
            if b is not None:
                assert b.x0 >= 0 and b.x1 <= width and b.y0 >= 0 and b.y1 <= height


# --------------------------------------------------------------------------- errors


def test_empty_and_missing_and_unreadable_clips(
    interim: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_clip(interim, "0000000a", np.full(60, np.nan))  # no valid sample
    write_clip(interim, "0000000b", tap_signal())
    (interim / "smoothed" / "0000000d.parquet").write_bytes(b"not parquet")
    write_manifest(interim, ["0000000a", "0000000b", "0000000c", "0000000d"])  # 0c has no file
    panels = panels_for(interim, ["0000000a", "0000000b", "0000000c", "0000000d"])
    assert [p.status for p in panels] == [
        qc.STATUS_EMPTY,
        qc.STATUS_OK,
        qc.STATUS_MISSING,
        qc.STATUS_UNREADABLE,
    ]
    fig = qc.make_figure(panels, "t")
    texts = [[t.get_text() for t in ax.texts] for ax in fig.axes[:4]]
    assert "no valid samples" in texts[0] and "missing smoothed file" in texts[2]
    assert "smoothed file unreadable" in texts[3]
    code = qc.main(["--page", "1"])
    out = capsys.readouterr().out
    assert code == 1 and "2 clips are missing or unreadable" in out
    table = pd.read_csv(interim / "qc" / "sheet_page01of01.csv")
    assert table["status"].tolist() == ["no_valid_samples", "ok", "missing", "unreadable"]
    assert table["range"].isna().tolist() == [True, False, True, True]


def test_an_empty_clip_alone_exits_zero(interim: Path) -> None:
    write_clip(interim, "0000000a", np.full(60, np.nan))
    write_manifest(interim, ["0000000a"])
    assert qc.main(["--page", "1"]) == 0


def test_a_missing_file_on_another_page_still_exits_1_with_the_note(
    interim: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    refs = make_world(interim, 30)
    (interim / "smoothed" / f"{refs[-1]}.parquet").unlink()  # the last clip: page 2, not page 1
    assert qc.main(["--page", "1"]) == 1
    out = capsys.readouterr().out
    assert (
        "note: 29 smoothed files, 30 manifest rows; 0 only in the folder, 1 only in the manifest"
        in out
    )
    assert "disagree" in out and "exit code 1" in out
    table = pd.read_csv(interim / "qc" / "sheet_page01of02.csv", dtype=str)
    assert table["status"].tolist() == ["ok"] * 24  # page 1 itself is fine
    assert refs[-1] not in table["ref"].tolist()
    capsys.readouterr()
    assert qc.main(["--page", "2"]) == 1  # the page that has it: error panel and exit 1
    assert "1 clips are missing or unreadable" in capsys.readouterr().out


def test_a_file_missing_from_the_manifest_also_exits_1_and_agreement_exits_0(
    interim: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    refs = make_world(interim, 5)
    assert qc.main(["--page", "1"]) == 0
    assert "note:" not in capsys.readouterr().out  # folder and manifest agree: silent
    write_clip(interim, "000000ff", tap_signal())  # in the folder, not in the manifest
    assert qc.main(["--page", "1"]) == 1
    out = capsys.readouterr().out
    assert "6 smoothed files, 5 manifest rows; 1 only in the folder, 0 only in the manifest" in out
    assert refs and "disagree" in out


def test_folder_and_manifest_mismatch_is_reported(
    interim: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_clip(interim, "0000000a", tap_signal())
    write_clip(interim, "0000000b", tap_signal())
    write_manifest(interim, ["0000000a", "0000000c"])
    assert qc.main(["--page", "1"]) == 1  # 0c is only in the manifest: missing
    out = capsys.readouterr().out
    assert "2 smoothed files, 2 manifest rows; 1 only in the folder, 1 only in the manifest" in out


# --------------------------------------------------------------------------- files and CLI


def test_sample_run_writes_named_files_with_seed_in_csv_and_title(interim: Path) -> None:
    make_world(interim, 30)
    assert qc.main(["--sample", "24"]) == 0
    out = interim / "qc"
    assert sorted(p.name for p in out.iterdir()) == ["sheet_seed021.csv", "sheet_seed021.png"]
    table = pd.read_csv(out / "sheet_seed021.csv", dtype=str, keep_default_na=False)
    assert list(table.columns) == list(qc.CSV_COLUMNS) and "verdict" not in table.columns
    assert table["seed"].tolist() == ["21"] * 24 and len(set(table["ref"])) == 24
    assert qc.main(["--sample", "24", "--seed", "5"]) == 0
    assert (out / "sheet_seed005.png").is_file()
    assert qc.build_parser().parse_args(["--sample", "3"]).seed == 21


def test_all_pages_cover_every_clip_exactly_once(interim: Path) -> None:
    refs = make_world(interim, 53)
    assert qc.main(["--all-pages"]) == 0
    out = interim / "qc"
    names = sorted(p.name for p in out.glob("*.csv"))
    assert names == ["sheet_page01of03.csv", "sheet_page02of03.csv", "sheet_page03of03.csv"]
    listed = [r for n in names for r in pd.read_csv(out / n, dtype=str)["ref"]]
    assert listed == sorted(refs) and len(set(listed)) == 53
    last = pd.read_csv(out / names[-1], dtype=str)
    assert len(last) == 5 and last["page"].tolist() == ["3"] * 5


def test_no_group_column_is_read_and_nothing_poisonous_is_written(
    interim: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_world(interim, 4)
    seen: list[Any] = []
    real = pd.read_csv

    def spy(*args: Any, **kwargs: Any) -> pd.DataFrame:
        seen.append(kwargs.get("usecols"))
        return real(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", spy)
    assert qc.main(["--page", "1"]) == 0
    # a literal list, not the module's own constant: adding a column there must fail this test
    expected = ["ref", "duration_s", "fps", "detection_rate", "longest_missing_run", "flags"]
    assert seen == [expected]
    assert list(qc.MANIFEST_COLUMNS) == expected
    out = interim / "qc"
    blob = "".join(p.read_text() for p in out.glob("*.csv")) + "".join(
        p.name for p in out.iterdir()
    )
    blob += qc.make_figure(panels_for(interim, refs_of(4)), "t").axes[0].get_title()
    figure = qc.make_figure(panels_for(interim, refs_of(4)), "t")
    blob += "".join(t.get_text() for ax in figure.axes for t in ax.texts + [ax.title])
    for bad in POISON:
        assert bad not in blob
    assert b"GROUP_POISON" not in (out / "sheet_page01of01.png").read_bytes()


def test_outputs_only_under_qc_and_always_regenerated(interim: Path) -> None:
    make_world(interim, 5)
    before = sorted(p.relative_to(interim).as_posix() for p in interim.rglob("*"))
    assert qc.main(["--page", "1"]) == 0
    after = sorted(p.relative_to(interim).as_posix() for p in interim.rglob("*"))
    assert [p for p in after if p not in before] == [
        "qc",
        "qc/sheet_page01of01.csv",
        "qc/sheet_page01of01.png",
    ]
    png = interim / "qc" / "sheet_page01of01.png"
    png.write_bytes(b"garbage")
    assert qc.main(["--page", "1"]) == 0
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"  # regenerated, not skipped
    assert not list((interim / "qc").glob(f".*{qc.TMP_MARKER}*"))


def test_exit_code_2_cases(interim: Path, tmp_path: Path) -> None:
    make_world(interim, 5)
    assert qc.main(["--page", "1", "--out-dir", str(tmp_path / "elsewhere")]) == 2
    assert qc.main(["--page", "2"]) == 2  # only one page exists
    assert qc.main(["--page", "0"]) == 2
    assert qc.main(["--sample", "25"]) == 2
    assert qc.main(["--sample", "6"]) == 2  # more than there are clips
    assert qc.main(["--page", "1", "--manifest", str(interim / "absent.csv")]) == 2
    empty = tmp_path / "interim2"
    empty.mkdir()
    (empty / "tracking_manifest.csv").write_text(",".join(qc.MANIFEST_COLUMNS) + "\n")
    assert (
        qc.main(
            [
                "--page",
                "1",
                "--smoothed-dir",
                str(empty),
                "--manifest",
                str(empty / "tracking_manifest.csv"),
            ]
        )
        == 2
    )
    bad = interim / "bad_manifest.csv"
    bad.write_text("ref,duration_s\n0000000a,5\n")
    assert qc.main(["--page", "1", "--manifest", str(bad)]) == 2  # a needed column is missing
    with pytest.raises(SystemExit):
        qc.main(["--page", "1", "--sample", "3"])  # modes are exclusive
