"""Finger-tapping explainer, research demo (task FT-47): skeleton with cached loaders.
Run from the repo root: streamlit run models/tapping/demo/app.py
Reads data/processed/xai outputs through data_access; the demo model and the landmarker handle
are cached with st.cache_resource (resources.py). Panels are added in FT-48..50.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_access as da  # noqa: E402
import resources as res  # noqa: E402
import streamlit as st  # noqa: E402

TITLE = "Finger-tapping explainer (research demo)"
DISCLAIMER = (
    "Research demo. The score is a model output on a research dataset; it is not a diagnosis "
    "and the model does not detect Parkinson's."
)
MODES = ("Example clip (held-out)", "Upload video")
TARGET_LABELS = {"ge2 (primary)": "ge2", "ge1": "ge1"}


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
def get_landmarker() -> object:
    return res.make_landmarker()


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
        st.file_uploader("Video", type=["mp4", "mov"])
        st.info("Available in a later step.")
        try:
            get_landmarker()  # created lazily, only in this mode
        except da.DemoDataError as err:
            st.info(f"The hand landmarker is not available: {err}")
    st.caption(model_caption)
    st.caption("inputs: " + (", ".join(sha for _, sha in key) if key else "none loaded"))


main()
