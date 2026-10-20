"""Finger-tapping explainer, research demo (tasks FT-47..49): cached loaders, upload view, score and
SHAP panels (fixed text templates; contributions to the model score, never causes).
Run from the repo root: streamlit run models/tapping/demo/app.py
Reads data/processed/xai outputs through data_access; the demo model and the landmarker handle
are cached with st.cache_resource (resources.py). Upload mode runs upload_pipeline.py in memory;
the file name is never read, shown or stored. Score and SHAP panels follow in FT-49, FT-50.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_access as da  # noqa: E402
import resources as res  # noqa: E402
import scoring as sc  # noqa: E402
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
SCORE_CAPTION = (
    "Model score from 0 to 1. Not a calibrated probability and not a diagnosis. "
    "In cross-validation this model reached AUROC {est:.2f} [{lo:.2f}, {hi:.2f}] for {target}."
)
HELD_OUT = "Held-out: this clip was not used to train the models that scored it."
TRAINED_ALL = "Demo model trained on all research clips."
BAND_EXAMPLE = "held-out scores over 5 repeats"
BAND_UPLOAD = "5-95% of 25 cross-validation models"
SHAP_CAPTION = "Contributions explain the model's score, not the cause of any condition."
FPS_NOTE = (
    "This video is about {fps:.0f} fps. Speed features are measured per second, so a high "
    "frame rate can shift them and the score may be slightly off."
)
RANGE_NOTE = "{names} outside the range of the research clips; treat the score with care."
SECONDARY_NOTE = "ge1 is the secondary study target; ge2 is the primary target."


@st.cache_data
def cached_clips(target: str, root: str, key: tuple[tuple[str, str], ...]) -> list[str]:
    return da.available_clips(target, Path(root))  # key (file sha8s) invalidates on change


@st.cache_data
def cached_summary(
    target: str, ref: str, root: str, key: tuple[tuple[str, str], ...]
) -> da.ClipSummary:
    return da.clip_summary(target, ref, Path(root))


@st.cache_resource
def get_model(target: str, root: str, model_sha8: str) -> Any:
    return res.load_demo_model(target, Path(root))  # model_sha8 in the key: a new file reloads


@st.cache_resource
def get_fold_models(target: str, root: str, folds_sha8: str) -> Any:
    return res.load_fold_models(target, Path(root))  # folds_sha8 in the key: new files reload


@st.cache_data
def cached_reference(root: str, features_sha8: str) -> Any:
    return sc.load_reference(Path(root))  # Set A columns only


@st.cache_resource
def get_landmarker() -> res.LandmarkerHandle:
    return res.make_landmarker()


@st.cache_data(show_spinner=False, max_entries=3)
def cached_analysis(
    sha256: str, suffix: str, _data: bytes, _factory: res.LandmarkerHandle
) -> up.UploadAnalysis:
    return up.analyse_upload(_data, suffix, _factory)  # key = sha256 of the bytes + extension


def render_score(
    *, target: str, root: Path, score: float, low: float, high: float, held_out: bool,
    values: dict[str, float], shap_values: dict[str, float], title: str, fps: float | None = None,
) -> None:  # fmt: skip
    """The Score and 'Why this score (SHAP)' sections (shared by both modes)."""
    ref_sha = da.cvs.file_sha256(root / "features.csv")[: da.cvs.HASH_LEN]
    ctx = sc.context(values, cached_reference(str(root), ref_sha))
    rows = sc.contributions(values, shap_values, ctx)
    auroc = sc.cv_auroc(target, root)
    st.header("Score")
    left, right = st.columns([1, 3])
    left.metric("Model score", f"{score:.2f}")
    band = BAND_EXAMPLE if held_out else BAND_UPLOAD
    right.markdown(f"(range {low:.2f}-{high:.2f}) {band}")
    side = "above" if score >= sc.THRESHOLD else "below"
    right.markdown(f"Fixed study threshold {sc.THRESHOLD}: {side}")
    st.caption(
        SCORE_CAPTION.format(est=auroc["estimate"], lo=auroc["ci_low"], hi=auroc["ci_high"],
                             target=target)
    )  # fmt: skip
    st.caption(HELD_OUT if held_out else TRAINED_ALL)
    if target == "ge1":
        st.info(SECONDARY_NOTE)
    if fps is not None and fps > sc.FPS_NOTE_ABOVE:
        st.info(FPS_NOTE.format(fps=fps))
    outside = [sc.PLAIN[r["feature"]][0] for r in rows if not r["in_range"]]
    if outside:
        verb = "is" if len(outside) == 1 else "are"
        st.info(RANGE_NOTE.format(names=", ".join(outside) + f" {verb}"))
    st.header("Why this score (SHAP)")
    st.image(up.figure_png(sc.shap_figure(rows, title)), width="stretch")
    st.markdown("\n".join(f"- {line}" for line in sc.wording(rows)))
    st.caption(SHAP_CAPTION)


def render_upload(target: str, root: Path, model: Any) -> None:
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
        with st.spinner("Tracking the hand and finding taps..."):
            analysis = cached_analysis(hashlib.sha256(data).hexdigest(), suffix, data, factory)
    except up.UploadError as err:
        st.warning(str(err))
        return
    r = analysis.result
    st.markdown(
        f"**Clip {r.upload_ref}**: {r.duration_s:.1f} s, {r.fps:.0f} fps, hand found in "
        f"{r.tracked_share:.0%} of frames, {r.n_taps} taps found"
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
    if model is None:
        st.info("The score is not available until the demo models are trained.")
        return
    try:
        folds = get_fold_models(target, str(root), res.fold_sha8(target, root))
        features = sc.upload_features(r)
        result = sc.score(features, model, folds)
        explained = sc.explain(features, model)
        render_score(
            target=target, root=root, score=result["score"], low=result["band_low"],
            high=result["band_high"], held_out=False, values=features,
            shap_values=explained["shap"], title=f"Contributions to the model score (clip {r.upload_ref})",
            fps=r.fps,
        )  # fmt: skip
    except (da.DemoDataError, da.GuardFailure) as err:
        st.info(f"No score for this clip: {err}")


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
    model = None
    try:
        model_sha8 = res.model_sha8(target, root)
        model = get_model(target, str(root), model_sha8)
        model_caption = f"model: {model_sha8} (demo model, not used for metrics)"
    except da.DemoDataError as err:  # example mode does not need the model yet
        st.info(f"The demo model is not available: {err}")
    if mode == MODES[0]:
        try:
            key = da.cache_key(target, root)
            refs = cached_clips(target, str(root), key)
            ref = st.selectbox("Clip (hashed ref)", refs)
            summary = cached_summary(target, ref, str(root), key)  # loaded and guarded
            probas = da.heldout_probas(target, ref, root)
            st.header("Clip")
            st.markdown(f"Research clip {ref} (hashed reference)")
            render_score(
                target=target, root=root, score=summary["proba"], low=min(probas),
                high=max(probas), held_out=True, values=summary["value"],
                shap_values=summary["shap"], title=f"Contributions to the model score (clip {ref})",
            )  # fmt: skip
            st.header("Per-tap amplitude")
        except (da.DemoDataError, da.GuardFailure) as err:
            st.error(f"The demo inputs are not available: {err}")
            st.stop()
    else:
        render_upload(target, root, model)
    st.caption(model_caption)
    st.caption("inputs: " + (", ".join(sha for _, sha in key) if key else "none loaded"))


main()
