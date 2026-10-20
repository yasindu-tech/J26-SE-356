"""Finger-tapping explainer, research demo (tasks FT-47, FT-48): cached loaders and the upload view.
Run from the repo root: streamlit run models/tapping/demo/app.py
Reads data/processed/xai outputs through data_access; the demo model and the landmarker handle
are cached with st.cache_resource (resources.py). Upload mode runs upload_pipeline.py in memory;
the file name is never read, shown or stored. Score and SHAP panels follow in FT-49, FT-50.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_access as da  # noqa: E402
import resources as res  # noqa: E402
import streamlit as st  # noqa: E402
import upload_pipeline as up  # noqa: E402

TITLE = "Finger-tapping explainer (research demo)"
DISCLAIMER = (
    "Research demo. The score is a model output on a research dataset; it is not a diagnosis "
    "and the model does not detect Parkinson's."
)
MODES = ("Example clip (held-out)", "Upload video")
TARGET_LABELS = {"ge2 (primary)": "ge2", "ge1": "ge1"}
NOT_STORED = "Processed on this computer; the video is not stored."


@st.cache_data
def cached_clips(target: str, root: str, key: tuple[tuple[str, str], ...]) -> list[str]:
    return da.available_clips(target, Path(root))  # key (file sha8s) invalidates on change


@st.cache_data
def cached_summary(
    target: str, ref: str, root: str, key: tuple[tuple[str, str], ...]
) -> da.ClipSummary:
    return da.clip_summary(target, ref, Path(root))


@st.cache_resource
def get_model(target: str, root: str, model_sha8: str) -> object:
    return res.load_demo_model(target, Path(root))  # model_sha8 in the key: a new file reloads


@st.cache_resource
def get_landmarker() -> res.LandmarkerHandle:
    return res.make_landmarker()


@st.cache_data(show_spinner=False, max_entries=3)
def cached_analysis(
    sha256: str, suffix: str, _data: bytes, _factory: res.LandmarkerHandle
) -> up.UploadAnalysis:
    return up.analyse_upload(_data, suffix, _factory)  # key = sha256 of the bytes + extension


def render_upload() -> None:
    factory = None
    try:
        factory = get_landmarker()  # created lazily: only in this mode
    except da.DemoDataError as err:
        st.warning(f"The hand landmarker is not available: {err}")
    uploaded = st.file_uploader("Video (mp4, mov or m4v)", type=["mp4", "mov", "m4v"])
    st.caption(NOT_STORED)
    if uploaded is None or factory is None:
        return
    data = uploaded.getvalue()
    suffix = Path(uploaded.name).suffix.lower()  # only the extension; the name is never kept
    try:
        with st.spinner("Tracking the hand and detecting taps..."):
            analysis = cached_analysis(hashlib.sha256(data).hexdigest(), suffix, data, factory)
    except up.UploadError as err:
        st.warning(str(err))
        return
    r = analysis.result
    st.markdown(
        f"**Clip {r.upload_ref}**: {r.duration_s:.1f} s, {r.fps:.0f} fps, hand found in "
        f"{r.tracked_share:.0%} of frames, {r.n_taps} taps detected"
    )
    for col, frame, t in zip(
        st.columns(len(analysis.frames)), analysis.frames, r.preview_times, strict=True
    ):
        col.image(frame, caption=f"{t:.1f} s", width="stretch")
    st.image(
        up.figure_png(up.signal_figure(r)),
        caption="Thumb-index distance over time",
        width="stretch",
    )
    if r.qc_flags:
        st.markdown("**Automatic checks**\n" + "\n".join(f"- {n}: {why}" for n, why in r.qc_flags))
    else:
        st.markdown("**Automatic checks**: no flags")


def main() -> None:
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    st.warning(DISCLAIMER)
    mode = st.sidebar.radio("Mode", MODES)
    target = TARGET_LABELS[st.sidebar.selectbox("Target", list(TARGET_LABELS))]
    st.sidebar.text_input("Feature set", "Set A", disabled=True)
    root = da.processed_root()
    key: tuple[tuple[str, str], ...] = ()
    model_caption = "model: not available"
    try:
        model_sha8 = res.model_sha8(target, root)
        get_model(target, str(root), model_sha8)
        model_caption = f"model: {model_sha8} (demo model, not used for metrics)"
    except da.DemoDataError as err:  # example mode does not need the model yet
        st.info(f"The demo model is not available: {err}")
    if mode == MODES[0]:
        try:
            key = da.cache_key(target, root)
            refs = cached_clips(target, str(root), key)
            ref = st.selectbox("Clip (hashed ref)", refs)
            cached_summary(target, ref, str(root), key)  # loaded and guarded; panels follow
        except (da.DemoDataError, da.GuardFailure) as err:
            st.error(f"The demo inputs are not available: {err}")
            st.stop()
        for header in ("Clip", "Score", "Why this score (SHAP)", "Per-tap amplitude"):
            st.header(header)
    else:
        render_upload()
    st.caption(model_caption)
    st.caption("inputs: " + (", ".join(sha for _, sha in key) if key else "none loaded"))


main()
