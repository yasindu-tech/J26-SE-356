"""Tests for models/tapping/demo/upload_pipeline.py and the upload view of app.py (task FT-48).
Synthetic landmarks of a known 1.5 Hz tapping and a tiny cv2-written video; no real video."""

from __future__ import annotations

import hashlib
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent / "demo"))
sys.path.insert(0, str(HERE))

import data_access as da  # noqa: E402
import hand_landmarks as hl  # noqa: E402
import resources as res  # noqa: E402
import streamlit as st  # noqa: E402
import taps  # noqa: E402
import upload_pipeline as up  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from signal_processing import build_signal  # noqa: E402
from smoothing import smooth_signal  # noqa: E402
from test_demo_resources import add_model  # noqa: E402
from test_shap_figures import make_world, text_problems  # noqa: E402

APP = HERE.parent / "demo" / "app.py"
W = H = 1000
RATE_HZ, SECONDS, FPS = 1.5, 20.0, 30.0
DATA = b"synthetic video bytes"
REF = hashlib.sha256(DATA).hexdigest()[:8]


def synthetic_df(detected_share: float = 1.0, seconds: float = SECONDS, fps: float = FPS) -> Any:
    n = int(seconds * fps)
    frame = np.arange(n)
    t = frame / fps
    d = 0.3 * (0.9 + 0.6 * np.sin(2 * np.pi * RATE_HZ * t))  # value = d * W / 300 px = 0.9..
    cols: dict[str, Any] = {c: np.full(n, 0.5) for c in hl.COLUMNS}
    cols.update(frame=frame, timestamp_ms=np.round(t * 1000).astype(np.int64), pts_ms=t * 1000)
    cols["handedness_score"] = np.full(n, 0.9)
    for k in range(hl.N_LANDMARKS):
        for p in ("wx", "wy", "wz"):
            cols[hl.landmark_column(p, k)] = np.zeros(n)

    def put(k: int, x: Any, y: Any) -> None:
        cols[hl.landmark_column("x", k)], cols[hl.landmark_column("y", k)] = x, y
        cols[hl.landmark_column("x_px", k)], cols[hl.landmark_column("y_px", k)] = x * W, y * H

    put(0, np.full(n, 0.5), np.full(n, 0.8))
    put(9, np.full(n, 0.5), np.full(n, 0.5))  # hand size = 0.3 * H px
    put(4, 0.5 - d / 2, np.full(n, 0.3))
    put(8, 0.5 + d / 2, np.full(n, 0.3))
    cols["detected"] = frame < int(round(detected_share * n))
    cols["handedness_label"] = np.array(["Right"] * n, dtype=object)
    df = pd.DataFrame(cols, columns=list(hl.COLUMNS))
    off = ~df["detected"].to_numpy()
    df.loc[off, list(hl.LANDMARK_COLUMNS)] = np.nan
    df.loc[off, "handedness_label"] = None
    df.loc[off, "handedness_score"] = np.nan
    df.attrs["hand_landmarks"] = {"fps": fps, "frames_container": n, "width": W, "height": H,
                                  "orientation_meta": 0}  # fmt: skip
    return df


class Factory:
    model_path = Path("model.task")

    def __call__(self, *a: Any, **k: Any) -> Any:  # never used: track_clip is faked
        raise AssertionError("the landmarker must not be created in these tests")


@pytest.fixture
def video(tmp_path: Path) -> Path:
    path = tmp_path / "upload.mp4"
    path.write_bytes(DATA)
    return path


def fake_pipeline(
    monkeypatch: pytest.MonkeyPatch, df: Any = None, probe: tuple[float, int] | None = None
) -> None:
    df = synthetic_df() if df is None else df
    monkeypatch.setattr(up.hl, "track_clip", lambda *a, **k: _with_attrs(df))
    monkeypatch.setattr(up, "probe_video", lambda p: probe or (FPS, len(df)))


def _with_attrs(df: Any) -> Any:
    out = df.copy()
    out.attrs = dict(df.attrs)
    return out


def direct_chain(df: Any) -> tuple[Any, Any]:
    sig = build_signal(df, REF)
    tb = smooth_signal(sig).table
    result = taps.extract_taps(
        REF, tb["t_s"].to_numpy(float), tb["value"].to_numpy(float),
        tb["value_raw"].to_numpy(float), tb["filled"].to_numpy(bool),
        tb["frame"].to_numpy(np.int64), taps.PARAMS,
    )  # fmt: skip
    return tb, result


# --------------------------------------------------------------------------- the pipeline


def test_signal_and_taps_equal_the_official_chain(
    monkeypatch: pytest.MonkeyPatch, video: Path
) -> None:
    fake_pipeline(monkeypatch)
    r = up.process_video(video, Factory())
    tb, official = direct_chain(synthetic_df())
    assert np.array_equal(r.smoothed, tb["value"].to_numpy(float), equal_nan=True)
    assert np.array_equal(r.raw, tb["value_raw"].to_numpy(float), equal_nan=True)
    assert np.array_equal(r.t, tb["t_s"].to_numpy(float) - tb["t_s"].iloc[0] + 0.0)
    assert r.n_taps == len(official.table) and np.array_equal(
        r.peak_t, official.table["t_peak_s"].to_numpy(float)
    )
    assert abs(r.n_taps - RATE_HZ * SECONDS) <= 1  # the synthetic tapping is recovered
    assert r.upload_ref == REF and r.fps == FPS and r.n_frames == int(SECONDS * FPS)
    assert r.tracked_share == 1.0 and r.duration_s == pytest.approx(r.t[-1])


def test_the_official_functions_are_really_called(
    monkeypatch: pytest.MonkeyPatch, video: Path
) -> None:
    fake_pipeline(monkeypatch)
    calls: list[str] = []
    for name in ("build_signal", "smooth_signal", "extract_taps"):
        real = getattr(up, name)

        def spy(*a: Any, _n: str = name, _r: Callable[..., Any] = real, **k: Any) -> Any:
            calls.append(_n)
            if _n == "extract_taps":
                assert a[0] == REF and a[-1] is taps.PARAMS  # TapParams v1, hashed ref
            return _r(*a, **k)

        monkeypatch.setattr(up, name, spy)
    up.process_video(video, Factory())
    assert calls == ["build_signal", "smooth_signal", "extract_taps"]


def test_constants_are_the_stated_limits() -> None:
    assert (up.MAX_BYTES, up.MAX_DURATION_S, up.MIN_FPS, up.MIN_TRACKED_SHARE, up.N_PREVIEW) == (
        200 * 1024 * 1024, 60.0, 15.0, 0.5, 6,
    )  # fmt: skip


def test_guard_size(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_pipeline(monkeypatch)
    big = tmp_path / "big.mp4"
    with big.open("wb") as f:
        f.truncate(up.MAX_BYTES)  # sparse file exactly at the limit: accepted
    assert up.process_video(big, Factory()).n_taps > 0
    with big.open("ab") as f:
        f.write(b"x")
    with pytest.raises(up.UploadError, match="larger than 200 MB"):
        up.process_video(big, Factory())
    with pytest.raises(up.UploadError, match="larger than 200 MB"):
        up.analyse_upload(b"x" * (up.MAX_BYTES + 1), ".mp4", Factory())


@pytest.mark.parametrize(
    ("fps", "frames", "message"),
    [(14.9, 300, "below 15 fps"), (30.0, 1801, "longer than 60 s"), (0.0, 300, "below 15 fps")],
)
def test_guard_probe_values(
    monkeypatch: pytest.MonkeyPatch, video: Path, fps: float, frames: int, message: str
) -> None:
    fake_pipeline(monkeypatch, probe=(fps, frames))
    with pytest.raises(up.UploadError, match=message):
        up.process_video(video, Factory())


def test_probe_edges_are_accepted(monkeypatch: pytest.MonkeyPatch, video: Path) -> None:
    fake_pipeline(monkeypatch, probe=(15.0, 900))  # 15 fps and 60 s exactly
    assert up.process_video(video, Factory()).n_taps > 0


def test_guard_duration_after_tracking(monkeypatch: pytest.MonkeyPatch, video: Path) -> None:
    fake_pipeline(monkeypatch, df=synthetic_df(seconds=61), probe=(FPS, 300))
    with pytest.raises(up.UploadError, match="longer than 60 s"):
        up.process_video(video, Factory())


def test_guard_tracked_share(monkeypatch: pytest.MonkeyPatch, video: Path) -> None:
    fake_pipeline(monkeypatch, df=synthetic_df(detected_share=0.3))
    with pytest.raises(up.UploadError, match="Hand not found"):
        up.process_video(video, Factory())
    fake_pipeline(monkeypatch, df=synthetic_df(detected_share=0.5))  # exactly at the limit
    assert up.process_video(video, Factory()).tracked_share == 0.5


def test_unreadable_video_and_tracking_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, video: Path
) -> None:
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"this is not a video")
    with pytest.raises(up.UploadError, match="could not be read"):
        up.process_video(junk, Factory())

    def boom(*a: Any, **k: Any) -> Any:
        raise hl.HandLandmarksError("clip abcd1234: cannot be opened as a video")

    monkeypatch.setattr(up, "probe_video", lambda p: (FPS, 600))
    monkeypatch.setattr(up.hl, "track_clip", boom)
    with pytest.raises(up.UploadError, match="could not be tracked") as err:
        up.process_video(video, Factory())
    assert "abcd1234" not in str(err.value)


def test_upload_ref_is_the_hash_of_the_bytes_and_the_name_is_not_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pipeline(monkeypatch)
    seen: list[str] = []
    monkeypatch.setattr(up, "preview_frames", lambda *a, **k: [np.zeros((4, 4, 3), np.uint8)])
    real = up.process_video

    def watch(p: Any, f: Any) -> Any:
        seen.append(Path(p).name)
        return real(p, f)

    monkeypatch.setattr(up, "process_video", watch)
    a = up.analyse_upload(DATA, ".mp4", Factory())
    assert a.result.upload_ref == REF and seen == ["upload.mp4"]
    texts = " ".join(t.get_text() for t in up.signal_figure(a.result).findobj(Text))
    assert REF in texts and "patient" not in (texts + repr(a.result.qc_flags)).lower()
    with pytest.raises(up.UploadError, match="mp4, mov or m4v"):
        up.analyse_upload(DATA, ".avi", Factory())


class Recorder(up.tempfile.TemporaryDirectory):  # type: ignore[type-arg]
    made: list[str] = []
    alive: list[Any] = []  # keep the objects: their finalizer must not do the cleanup for us
    existed_during: list[bool] = []

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        Recorder.made.append(self.name)
        Recorder.alive.append(self)


@pytest.mark.parametrize("fails", [False, True])
def test_temp_directory_is_removed_also_on_error(
    monkeypatch: pytest.MonkeyPatch, fails: bool
) -> None:
    fake_pipeline(monkeypatch)
    Recorder.made, Recorder.existed_during, Recorder.alive = [], [], []
    monkeypatch.setattr(up.tempfile, "TemporaryDirectory", Recorder)
    real = up.process_video

    def process(p: Any, f: Any) -> Any:
        Recorder.existed_during.append(Path(p).is_file())
        if fails:
            raise up.UploadError("forced failure")
        return real(p, f)

    monkeypatch.setattr(up, "process_video", process)
    if fails:
        with pytest.raises(up.UploadError, match="forced"):
            up.analyse_upload(DATA, ".mp4", Factory())
    else:
        monkeypatch.setattr(up, "preview_frames", lambda *a, **k: [])
        up.analyse_upload(DATA, ".mp4", Factory())
    assert len(Recorder.made) == 1 and Recorder.existed_during == [True]
    assert not Path(Recorder.made[0]).exists()


# --------------------------------------------------------------------------- preview frames


def test_preview_only_uses_frames_with_a_hand() -> None:
    detected = np.zeros(300, dtype=bool)
    detected[120:290] = True
    picked = up.pick_preview(detected)
    assert len(picked) == 6 and detected[picked].all() and picked[0] == 120 and picked[-1] == 289
    assert list(up.pick_preview(np.array([False, True, True, False]))) == [1, 2]  # fewer than 6


def test_preview_frames_are_rgb_with_marks_at_the_landmarks(tmp_path: Path) -> None:
    path = tmp_path / "tiny.mp4"
    w, h, n = 320, 240, 12
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 30.0, (w, h))
    for i in range(n):
        bgr = np.empty((h, w, 3), dtype=np.uint8)
        bgr[..., 0], bgr[..., 1], bgr[..., 2] = 200, 40 + 10 * i, 50  # blue-ish in BGR order
        writer.write(bgr)
    writer.release()
    pts = np.zeros((21, 2))
    pts[0], pts[4], pts[8], pts[9] = (160, 200), (100, 80), (200, 60), (160, 120)
    frames = up.preview_frames(path, [7, 2, 5], {2: pts, 5: pts, 7: pts})
    assert len(frames) == 3 and all(f.shape == (h, w, 3) and f.dtype == np.uint8 for f in frames)
    for f in frames:  # marked pixels sit exactly at the landmark positions, in RGB order
        assert tuple(f[80, 100]) == up.THUMB_RGB and tuple(f[60, 200]) == up.INDEX_RGB
        assert tuple(f[200, 160]) == up.SCALE_RGB and tuple(f[120, 160]) == up.SCALE_RGB
        assert tuple(f[70, 150]) == up.LINE_RGB or f[70, 150].max() > 200  # the line between
    assert frames[0][5, 5, 1] > frames[1][5, 5, 1]  # frame 7 has more green than frame 2
    assert all(
        int(f[5, 5, 2]) > int(f[5, 5, 0]) + 80 for f in frames
    )  # blue stays blue: RGB, not BGR
    with pytest.raises(up.UploadError, match="preview frame"):
        up.preview_frames(path, [40], {40: pts})


def test_probe_reads_a_real_video(tmp_path: Path) -> None:
    path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 25.0, (64, 48))
    for _ in range(10):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    fps, frames = up.probe_video(path)
    assert fps == pytest.approx(25.0) and frames == 10


# --------------------------------------------------------------------------- the figure


def test_signal_figure(monkeypatch: pytest.MonkeyPatch, video: Path) -> None:
    fake_pipeline(monkeypatch)
    r = up.process_video(video, Factory())
    fig = up.signal_figure(r)
    texts = [t.get_text() for t in fig.findobj(Text) if t.get_text()]
    assert f"Upload {REF}: {r.n_taps} taps detected" in texts
    assert "time (s)" in texts and "thumb-index distance (hand sizes)" in texts
    assert up.FOOTER in texts and "raw" in texts and "smoothed" in texts and "tap peaks" in texts
    lines = fig.axes[0].get_lines()
    assert len(np.asarray(lines[-1].get_xdata())) == r.n_taps  # one marker per tap
    assert not text_problems(fig)
    assert all(float(t.get_fontsize()) >= 11 for t in fig.findobj(Text) if t.get_text())
    assert up.figure_png(up.signal_figure(r)) == up.figure_png(up.signal_figure(r))


# --------------------------------------------------------------------------- the app


@pytest.fixture(autouse=True)
def fresh_caches() -> None:
    st.cache_resource.clear()
    st.cache_data.clear()


def upload_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, analyse: Callable[..., Any] | None = None
) -> AppTest:
    w = make_world(tmp_path, monkeypatch)
    add_model(w["proc"])
    monkeypatch.setenv(da.ENV_ROOT, str(w["proc"]))
    monkeypatch.setattr(res, "make_landmarker", lambda: Factory())
    fake_pipeline(monkeypatch)
    monkeypatch.setattr(
        up, "preview_frames", lambda p, idx, lm: [np.zeros((30, 20, 3), np.uint8)] * len(idx)
    )
    if analyse is not None:
        monkeypatch.setattr(up, "analyse_upload", analyse)
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.sidebar.radio[0].set_value("Upload video").run()
    return app


def page(app: AppTest) -> str:
    parts: list[str] = []
    for kind in ("title", "header", "markdown", "caption", "info", "warning", "error"):
        parts += [str(getattr(el, "value", "")) for el in getattr(app, kind)]
    return "\n".join(parts)


def test_upload_shows_summary_frames_figure_and_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = upload_app(tmp_path, monkeypatch)
    assert not app.exception and not app.image
    assert any(
        c.value == "Processed on this computer; the video is not stored." for c in app.caption
    )
    app.file_uploader[0].set_value(("PATIENT_12345_visit.mp4", DATA, "video/mp4")).run()
    assert not app.exception
    summary = [m.value for m in app.markdown if m.value.startswith(f"**Clip {REF}**")]
    assert len(summary) == 1 and re.fullmatch(
        rf"\*\*Clip {REF}\*\*: 20\.0 s, 30 fps, hand found in 100% of frames, \d+ taps detected",
        summary[0],
    )
    assert len(app.image) == 7  # 6 preview frames and the signal figure
    captions = [c for i in app.image for c in i.captions]
    assert captions[-1] == "Thumb-index distance over time"
    assert all(re.fullmatch(r"\d+\.\d s", c) for c in captions[:6])
    text = page(app)
    assert "PATIENT" not in text and "12345" not in text and "visit" not in text
    assert any("Automatic checks" in m.value for m in app.markdown)


def test_upload_error_is_a_warning_not_an_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*a: Any, **k: Any) -> Any:
        raise up.UploadError("Hand not found in enough frames; keep the hand in view.")

    app = upload_app(tmp_path, monkeypatch, analyse=refuse)
    app.file_uploader[0].set_value(("x.mp4", DATA, "video/mp4")).run()
    assert not app.exception and not app.image
    assert [w.value for w in app.warning][
        -1
    ] == "Hand not found in enough frames; keep the hand in view."
    assert "Traceback" not in page(app)


def test_cache_is_keyed_on_the_bytes_not_the_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real = up.analyse_upload

    def counted(d: bytes, s: str, f: Any) -> Any:
        calls.append(len(d))
        return real(d, s, f)

    app = upload_app(tmp_path, monkeypatch, analyse=counted)
    app.file_uploader[0].set_value(("same.mp4", DATA, "video/mp4")).run()
    app.run()
    app.file_uploader[0].set_value(("other-name.mp4", DATA, "video/mp4")).run()
    assert calls == [len(DATA)]  # same bytes under another name: cached
    app.file_uploader[0].set_value(("same.mp4", DATA + b"!", "video/mp4")).run()
    assert calls == [len(DATA), len(DATA) + 1]  # other bytes under the same name: processed
    assert not app.exception


def test_example_mode_is_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = upload_app(tmp_path, monkeypatch)
    app.sidebar.radio[0].set_value("Example clip (held-out)").run()
    assert not app.exception
    assert [h.value for h in app.header] == [
        "Clip",
        "Score",
        "Why this score (SHAP)",
        "Per-tap amplitude",
    ]
    assert not app.file_uploader
