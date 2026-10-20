"""Tests for models/tapping/demo/summary_pdf.py and the PDF button (task FT-51). No PDF library
is installed, so the page is checked through the texts it draws (build_page) and the PDF bytes
through their structure (header, page objects, info dictionary)."""

from __future__ import annotations

import builtins
import hashlib
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent / "demo"
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(DEMO))
sys.path.insert(0, str(HERE))

import data_access as da  # noqa: E402
import scoring as sc  # noqa: E402
import streamlit as st  # noqa: E402
import summary_pdf as pdf  # noqa: E402
import tap_curve as tc  # noqa: E402
import texts as tx  # noqa: E402
import upload_pipeline as up  # noqa: E402
from test_demo_resources import add_reference  # noqa: E402
from test_demo_score_app import FORBIDDEN, LABEL_WORDS, start, upload  # noqa: E402
from test_demo_upload import DATA, Factory, fake_pipeline  # noqa: E402
from test_shap_figures import REFS  # noqa: E402

REF = "abcd1234"
FIXED = [tx.DISCLAIMER, tx.SHAP_CAPTION]
AUROC = tx.SCORE_CAPTION.format(est=0.87, lo=0.81, hi=0.92, target="ge2")
SHAP_VALUES = dict(zip(sc.SET_A, (0.31, -0.67, -0.13, 0.04, -0.4), strict=True))


@pytest.fixture(autouse=True)
def fresh_caches() -> None:
    st.cache_resource.clear()
    st.cache_data.clear()


@pytest.fixture
def result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    fake_pipeline(monkeypatch)
    video = tmp_path / "upload.mp4"
    video.write_bytes(DATA)
    return up.process_video(video, Factory())


def full(result: Any, **edit: Any) -> pdf.SummaryData:
    values = dict(zip(sc.SET_A, (1.55, 0.18, 7.95, 11.62, 1.146), strict=True))
    ctx = {c: {"percentile": 40.0 + i, "in_range": True} for i, c in enumerate(sc.SET_A)}
    rows = sc.contributions(values, SHAP_VALUES, ctx)
    curve = tc.curve_for_upload(result)
    flags = [("landmarks_out_of_frame", "part of the hand left the picture in many frames")]
    notes = pdf.quality_notes(
        mode="upload", target="ge2", fps=60.0, tracked_share=0.8, qc_flags=flags,
        outside=["Mean amplitude"], taps_note="Taps shown: 28 of 30. 2 taps at the clip edges left "
        "out (one-sided amplitude).", curve_reason=None,
    )  # fmt: skip
    data = pdf.SummaryData(
        ref=REF, mode="upload", target="ge2", date_text="2026-10-20",
        recording={"duration_s": 18.2, "fps": 60.0, "tracked_share": 0.8, "n_found": 30,
                   "n_shown": 28},  # fmt: skip
        qc_flags=flags, notes=notes, score=0.09, band_text="(range 0.02-0.10) " + tx.BAND_UPLOAD,
        threshold_text="Fixed study threshold 0.5: below", auroc_text=AUROC,
        source_text=tx.TRAINED_ALL, sentences=sc.wording(rows),
        shap_png=up.figure_png(sc.shap_figure(rows, f"Contributions (clip {REF})", compact=True)),
        signal_png=up.figure_png(up.signal_figure(result, compact=True)),
        curve_png=up.figure_png(tc.curve_figure(curve, REF, compact=True)),
        curve_count=tc.count_sentence(curve), curve_slope=tc.slope_sentence(curve),
        model_sha8="1c623df9", input_sha8=["ae519116", "184171e8"],
    )  # fmt: skip
    for key, value in edit.items():
        setattr(data, key, value)
    return data


def page_text(d: pdf.SummaryData) -> str:
    return " ".join(t["text"] for t in pdf.build_page(d).texts)


def n_pages(blob: bytes) -> int:
    return len(re.findall(rb"/Type /Page(?![s\w])", blob))


# --------------------------------------------------------------------------- content


def test_pdf_is_one_page_and_says_everything_the_screen_says(result: Any) -> None:
    d = full(result)
    blob = pdf.build_pdf(d)
    assert blob.startswith(b"%PDF") and n_pages(blob) == 1 and b"/Count 1" in blob
    text = page_text(d)
    for needle in (
        pdf.TITLE, REF, tx.DISCLAIMER, "2026-10-20", "Model score 0.09", d.band_text,
        d.threshold_text, d.auroc_text, d.source_text, tx.SHAP_CAPTION, tc.NOTE,
        d.curve_count, d.curve_slope, "Model: 1c623df9 (demo model, not used for metrics)",
        "ae519116, 184171e8",
    ):  # fmt: skip
        assert str(needle) in text, needle
    for sentence in d.sentences:
        assert sentence in text and "pushed the score" in sentence
    assert len(d.sentences) == 5 and any("down by 0.67 (log-odds)" in s for s in d.sentences)
    for note in d.notes:
        assert note in text, note
    assert "Duration: 18.2 s" in text and "Frame rate: 60 fps" in text
    assert "Hand found in: 80% of frames" in text and "Taps found: 30" in text


def test_data_quality_notes_by_hand() -> None:
    flags = [("low_detection", "the hand was found in fewer frames than expected")]
    notes = pdf.quality_notes(
        mode="upload", target="ge1", fps=59.9, tracked_share=0.85, qc_flags=flags,
        outside=["Tapping rate", "Mean amplitude"], taps_note="Taps shown: 9 of 10.",
        curve_reason="The mean amplitude is not positive, so no line is fitted.",
    )  # fmt: skip
    assert notes == [
        "Automatic check low_detection: the hand was found in fewer frames than expected.",
        tx.FPS_NOTE.format(fps=59.9),
        "The hand was found in fewer than 90% of frames.",
        "Taps shown: 9 of 10.",
        tx.RANGE_NOTE.format(names="Tapping rate, Mean amplitude are"),
        "Scored with the demo model (trained on all research clips).",
        tx.SECONDARY_NOTE,
        "The mean amplitude is not positive, so no line is fitted.",
    ]
    quiet = pdf.quality_notes(
        mode="example", target="ge2", fps=30.0, tracked_share=0.95, qc_flags=[], outside=[],
        taps_note=None, curve_reason=None,
    )  # fmt: skip
    assert quiet == ["Automatic checks: no flags.", tx.HELD_OUT]
    unknown = pdf.quality_notes(mode="example", target="ge2", fps=None, tracked_share=None,
                                qc_flags=None, outside=[], taps_note=None, curve_reason=None)  # fmt: skip
    assert unknown[0] == "Automatic checks are not available for example clips."


# --------------------------------------------------------------------------- privacy and words


def test_no_names_ids_labels_or_forbidden_words_outside_the_fixed_sentences(result: Any) -> None:
    d = full(result)
    text = page_text(d)
    rest = text.replace(tx.DISCLAIMER, "").replace(AUROC, "").replace(tx.SHAP_CAPTION, "")
    assert FORBIDDEN.search(text) and not FORBIDDEN.search(rest)  # only in the fixed sentences
    assert not LABEL_WORDS.search(rest)
    assert not re.search(r"\.mp4|\.mov|CONTROL\d|ID\d{4}_|PATIENT|/Users/", text)
    blob = pdf.build_pdf(d)
    for forbidden in (b"/Author", b"/Users/", b"naduliweerasinghe", b".mp4", b"rating", b"PATIENT",
                      b"updrs", b"Parkinson", b"/CreationDate", b"/ModDate", b"fis_diagnostic"):  # fmt: skip
        assert forbidden.lower() not in blob.lower() or forbidden == b"Parkinson", forbidden
    assert b"Parkinson" not in blob  # the disclaimer text is inside the page content stream only


def test_metadata_is_fixed(result: Any) -> None:
    blob = pdf.build_pdf(full(result))
    info = blob[blob.index(b"/Title") - 200 : blob.index(b"/Title") + 400]
    assert b"/Title (Finger-tapping research prototype - one-page summary)" in blob
    assert b"/Creator (Finger-tapping research prototype \\(demo\\))" in blob or (
        b"/Creator (Finger-tapping research prototype (demo))" in blob
    ), info
    assert b"/Author" not in blob and b"/CreationDate" not in blob and b"/Keywords" not in blob


def test_two_builds_are_identical_and_inputs_change_the_bytes(result: Any) -> None:
    d = full(result)
    a, b = pdf.build_pdf(d), pdf.build_pdf(full(result))
    assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()
    other = pdf.build_pdf(full(result, ref="eeeeeeee"))
    assert (
        other != a and d.sha8() == full(result).sha8() and d.sha8() != full(result, ref="x").sha8()
    )


def test_nothing_is_written_to_disk(monkeypatch: pytest.MonkeyPatch, result: Any) -> None:
    d = full(result)
    pdf.build_pdf(d)  # warm the font cache, which reads files
    real_open = builtins.open

    def guarded(file: Any, mode: str = "r", *a: Any, **k: Any) -> Any:
        if any(c in str(mode) for c in "wax+"):
            raise AssertionError("a file was opened for writing")
        return real_open(file, mode, *a, **k)

    def refuse(*a: Any, **k: Any) -> Any:
        raise AssertionError("a file was written")

    monkeypatch.setattr(builtins, "open", guarded)
    monkeypatch.setattr(Path, "write_bytes", refuse)
    monkeypatch.setattr(Path, "write_text", refuse)
    monkeypatch.setattr(Path, "open", refuse)
    assert pdf.build_pdf(d).startswith(b"%PDF")


# --------------------------------------------------------------------------- not available, fit


def test_missing_numbers_read_not_available_never_zero(result: Any) -> None:
    d = full(result, recording={"duration_s": float("nan"), "fps": None, "tracked_share": float("nan"),
                                "n_found": None, "n_shown": float("nan")})  # fmt: skip
    lines = " ".join(pdf.recording_lines(d.recording))
    assert "Duration: not available" in lines and "Frame rate: not available" in lines
    assert "Hand found in: not available" in lines and "Taps found: not available" in lines
    assert "Taps shown in the amplitude curve: not available" in lines
    assert "0.0 s" not in lines and "0 fps" not in lines and "nan" not in lines.lower()
    assert "nan" not in page_text(d).lower() and pdf.build_pdf(d).startswith(b"%PDF")


def test_a_nan_feature_is_not_available_in_the_sentence_not_zero() -> None:
    values = dict(zip(sc.SET_A, (float("nan"), 0.18, 7.95, 11.62, 1.146), strict=True))
    ctx = sc.context(values, __import__("pandas").DataFrame({c: [1.0, 2.0, 3.0] for c in sc.SET_A}))
    assert np.isnan(ctx["tap_rate_hz"]["percentile"]) and ctx["tap_rate_hz"]["in_range"]
    line = next(s for s in sc.wording(sc.contributions(values, SHAP_VALUES, ctx)) if "Tapping" in s)
    assert line.startswith("Tapping rate (not available, percentile not available)")
    assert "0.00" not in line and "nan" not in line.lower()


@pytest.mark.parametrize(
    "missing",
    [["shap_png", "sentences"], ["signal_png"], ["curve_png", "curve_count", "curve_slope"],
     ["score", "band_text", "threshold_text", "auroc_text", "source_text"],
     ["qc_flags", "model_sha8", "input_sha8"]],
)  # fmt: skip
def test_a_missing_section_still_builds_a_one_page_pdf(result: Any, missing: list[str]) -> None:
    empty: dict[str, Any] = {}
    for name in missing:
        empty[name] = [] if name in ("sentences", "input_sha8") else None
    d = full(result, **empty)
    blob = pdf.build_pdf(d)
    assert blob.startswith(b"%PDF") and n_pages(blob) == 1
    text = page_text(d)
    assert "not available" in text and "nan" not in text.lower()
    if "score" in missing:
        assert "Model score: not available" in text and "0.00" not in text


def test_a_lot_of_notes_still_fit_on_one_page(result: Any) -> None:
    d = full(
        result, notes=[f"Automatic check flag{i}: a plain sentence about it." for i in range(40)]
    )
    page = pdf.build_page(d)
    text = " ".join(t["text"] for t in page.texts)
    assert pdf.MORE in text and "flag39" not in text and "flag0:" in text
    assert n_pages(pdf.build_pdf(d)) == 1


def test_layout_stays_inside_the_page_with_readable_sizes(result: Any) -> None:
    for d in (full(result), full(result, notes=["x " * 300])):
        page = pdf.build_page(d)
        assert page.y <= pdf.H - pdf.MARGIN
        for t in page.texts:
            assert t["size"] >= (pdf.SMALL if t["kind"] == "footer" else 9.0), t
            assert t["x"] >= pdf.MARGIN
            assert t["x"] + page.width(t["text"], t["size"]) <= pdf.W - pdf.MARGIN + 1, t
            assert t["y"] + t["size"] <= pdf.H - pdf.MARGIN + 2, t


def test_compact_figures_are_readable() -> None:
    from matplotlib.text import Text
    from test_shap_figures import text_problems

    rows = sc.contributions(dict.fromkeys(sc.SET_A, 1.0), SHAP_VALUES,
                            {c: {"percentile": 50.0, "in_range": True} for c in sc.SET_A})  # fmt: skip
    c = tc.tap_curve(np.arange(12) * 0.6, 1.2 - 0.01 * np.arange(12))
    for fig in (
        sc.shap_figure(rows, "t (clip abcd1234)", compact=True),
        tc.curve_figure(c, REF, True),
    ):
        assert not text_problems(fig)
        assert all(float(t.get_fontsize()) >= 8.5 for t in fig.findobj(Text) if t.get_text())
    fig = tc.curve_figure(c, REF, True)
    fig.canvas.draw()
    legend, box = fig.legends[0].get_window_extent(), fig.axes[0].get_window_extent()
    assert legend.y1 <= box.y0 + 1  # below the data area, never over the title or the data


# --------------------------------------------------------------------------- the app button


def capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    got: dict[str, Any] = {"data": []}
    real_btn = st.download_button
    real_build = pdf.build_pdf

    def btn(label: str, data: Any = None, file_name: str | None = None, mime: str | None = None,
            **kw: Any) -> Any:  # fmt: skip
        got.update(label=label, bytes=data, file_name=file_name, mime=mime)
        return real_btn(label, data=data, file_name=file_name, mime=mime, **kw)

    def build(d: pdf.SummaryData) -> bytes:
        got["data"].append(d)
        return real_build(d)

    monkeypatch.setattr(st, "download_button", btn)
    monkeypatch.setattr(pdf, "build_pdf", build)
    return got


def test_upload_mode_offers_a_pdf_with_what_is_on_screen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    got = capture(monkeypatch)
    app = upload(start(tmp_path, monkeypatch))
    assert not app.exception and got["label"] == "Download one-page summary (PDF)"
    assert got["mime"] == "application/pdf" and re.fullmatch(
        r"summary_[0-9a-f]{8}\.pdf", got["file_name"]
    )
    blob = got["bytes"]
    assert blob.startswith(b"%PDF") and n_pages(blob) == 1
    d = got["data"][-1]
    assert f"Model score {float(app.metric[0].value):.2f}" in page_text(d)
    on_screen = next(m.value for m in app.markdown if m.value.startswith("- "))
    assert d.sentences == [line[2:] for line in on_screen.split("\n")]
    assert d.band_text in [m.value for m in app.markdown] and d.mode == "upload"
    assert re.fullmatch(r"[0-9a-f]{8}", d.model_sha8) and d.input_sha8 and d.qc_flags is not None
    assert d.shap_png and d.signal_png and d.curve_png and d.recording["fps"] > 0


def test_example_mode_offers_a_pdf_of_the_held_out_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    got = capture(monkeypatch)
    app = start(tmp_path, monkeypatch)
    assert not app.exception and got["bytes"].startswith(b"%PDF")
    d = got["data"][-1]
    assert d.mode == "example" and d.ref == REFS[0] and d.qc_flags is None
    assert d.source_text == tx.HELD_OUT and d.signal_png is None
    assert pdf.NA in page_text(d) and tx.HELD_OUT in d.notes
    proc = da.processed_root()
    summary = da.clip_summary("ge2", REFS[0], proc)
    ctx = sc.context(summary["value"], sc.load_reference(proc))
    rows = sc.contributions(summary["value"], summary["shap"], ctx)
    expected = up.figure_png(
        sc.shap_figure(rows, f"Contributions to the model score (clip {REFS[0]})", compact=True)
    )
    assert d.shap_png == expected  # the figure of this clip, not of another one
    assert d.sentences == sc.wording(rows) and d.score == pytest.approx(summary["proba"])
    assert got["file_name"] == f"summary_{REFS[0]}.pdf"


def test_the_pdf_is_cached_on_its_content_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    got = capture(monkeypatch)
    app = start(tmp_path, monkeypatch)
    app.run().run()
    assert len(got["data"]) == 1  # three runs, one build
    add_reference(da.processed_root())  # unrelated rewrite of the reference: same PDF key
    assert (
        pdf.SummaryData(ref="a", mode="x", target="ge2", date_text="d").sha8()
        != pdf.SummaryData(ref="b", mode="x", target="ge2", date_text="d").sha8()
    )
