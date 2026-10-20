"""Finger-tapping explainer, research demo (tasks FT-47..49): cached loaders, upload view, score and
SHAP panels (fixed text templates; contributions to the model score, never causes).
Run from the repo root: streamlit run models/tapping/demo/app.py
Reads data/processed/xai outputs through data_access; the demo model and the landmarker handle
are cached with st.cache_resource (resources.py). Upload mode runs upload_pipeline.py in memory;
the file name is never read, shown or stored. Score and SHAP panels follow in FT-49, FT-50.
"""

from __future__ import annotations

import datetime
import hashlib
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_access as da  # noqa: E402
import numpy as np  # noqa: E402
import resources as res  # noqa: E402
import scoring as sc  # noqa: E402
import streamlit as st  # noqa: E402
import summary_pdf as pdf  # noqa: E402
import tap_curve as tc  # noqa: E402
import upload_pipeline as up  # noqa: E402
from texts import (  # noqa: E402,F401  (re-exported: the tests read them from the app)
    BAND_EXAMPLE,
    BAND_UPLOAD,
    DISCLAIMER,
    FPS_NOTE,
    HELD_OUT,
    MODES,
    NOT_STORED,
    RANGE_NOTE,
    SCORE_CAPTION,
    SECONDARY_NOTE,
    SHAP_CAPTION,
    TARGET_LABELS,
    TITLE,
    TRAINED_ALL,
)


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
) -> dict[str, Any]:  # fmt: skip
    """The Score and 'Why this score (SHAP)' sections (shared by both modes); returns what was
    shown, so the PDF repeats exactly the same texts and figure."""
    ref_sha = da.cvs.file_sha256(root / "features.csv")[: da.cvs.HASH_LEN]
    ctx = sc.context(values, cached_reference(str(root), ref_sha))
    rows = sc.contributions(values, shap_values, ctx)
    auroc = sc.cv_auroc(target, root)
    st.header("Score")
    left, right = st.columns([1, 3])
    left.metric("Model score", f"{score:.2f}")
    band = BAND_EXAMPLE if held_out else BAND_UPLOAD
    band_text = f"(range {low:.2f}-{high:.2f}) {band}"
    right.markdown(band_text)
    side = "above" if score >= sc.THRESHOLD else "below"
    threshold_text = f"Fixed study threshold {sc.THRESHOLD}: {side}"
    right.markdown(threshold_text)
    auroc_text = SCORE_CAPTION.format(
        est=auroc["estimate"], lo=auroc["ci_low"], hi=auroc["ci_high"], target=target
    )
    st.caption(auroc_text)
    source_text = HELD_OUT if held_out else TRAINED_ALL
    st.caption(source_text)
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
    shap_png = up.figure_png(
        sc.shap_figure(rows, title, compact=True)
    )  # page-size copy for the PDF
    sentences = sc.wording(rows)
    st.markdown("\n".join(f"- {line}" for line in sentences))
    st.caption(SHAP_CAPTION)
    return {"score": score, "band_text": band_text, "threshold_text": threshold_text,
            "auroc_text": auroc_text, "source_text": source_text, "outside": outside,
            "sentences": sentences, "shap_png": shap_png, "ref_sha": ref_sha}  # fmt: skip


@st.cache_data(show_spinner=False)
def cached_curve_png(key: str, _curve: tc.TapCurve, ref: str, compact: bool = False) -> bytes:
    return up.figure_png(tc.curve_figure(_curve, ref, compact))  # key: sha8 of the curve data


def render_curve(curve: tc.TapCurve, ref: str) -> dict[str, Any]:
    """The per-tap amplitude figure with the official fitted line, its words and the note."""
    data = np.concatenate([curve.times, curve.amps, curve.line_y]).tobytes()
    key = hashlib.sha256(data + ref.encode()).hexdigest()[:8]
    st.image(cached_curve_png(key, curve, ref), width="stretch")
    png = cached_curve_png(key, curve, ref, True)  # page-size copy for the PDF
    count, slope = tc.count_sentence(curve), tc.slope_sentence(curve)
    st.markdown(count)
    st.markdown(slope)
    st.caption(tc.NOTE)
    return {"png": png, "count": count, "slope": slope, "curve": curve,
            "reason": None if curve.usable else curve.reason}  # fmt: skip


@st.cache_data(show_spinner=False)
def cached_pdf(key: str, _data: pdf.SummaryData) -> bytes:
    return pdf.build_pdf(_data)  # key: sha8 of everything the page shows


def render_pdf_button(data: pdf.SummaryData) -> None:
    st.download_button(
        "Download one-page summary (PDF)",
        data=cached_pdf(data.sha8(), data),
        file_name=f"summary_{data.ref}.pdf",  # the hashed ref only
        mime="application/pdf",
    )


def summary_data(
    *, ref: str, mode: str, target: str, shown: dict[str, Any] | None,
    curve_shown: dict[str, Any] | None, recording: dict[str, Any], qc_flags: Any,
    signal_png: bytes | None, model_sha8: str | None, inputs: list[str],
) -> pdf.SummaryData:  # fmt: skip
    """What the PDF repeats from the screen; a missing section stays None (fixed reason text)."""
    curve = curve_shown["curve"] if curve_shown else None
    count_note = None
    if curve is not None and (curve.n_edge or curve.n_nan):
        count_note = tc.count_sentence(curve)
    notes = pdf.quality_notes(
        mode=mode, target=target, fps=recording.get("fps"),
        tracked_share=recording.get("tracked_share"), qc_flags=qc_flags,
        outside=shown["outside"] if shown else [], taps_note=count_note,
        curve_reason=curve_shown["reason"] if curve_shown else tc.REASONS["missing"],
    )  # fmt: skip
    return pdf.SummaryData(
        ref=ref, mode=mode, target=target, date_text=datetime.date.today().isoformat(),
        recording={**recording, "n_found": curve.n_found if curve else None,
                   "n_shown": curve.n_taps if curve else None},
        qc_flags=qc_flags, notes=notes,
        score=shown["score"] if shown else None,
        band_text=shown["band_text"] if shown else None,
        threshold_text=shown["threshold_text"] if shown else None,
        auroc_text=shown["auroc_text"] if shown else None,
        source_text=shown["source_text"] if shown else None,
        sentences=shown["sentences"] if shown else [],
        shap_png=shown["shap_png"] if shown else None, signal_png=signal_png,
        curve_png=curve_shown["png"] if curve_shown else None,
        curve_count=curve_shown["count"] if curve_shown else None,
        curve_slope=curve_shown["slope"] if curve_shown else None,
        model_sha8=model_sha8, input_sha8=inputs,
    )  # fmt: skip


def render_upload(target: str, root: Path, model: Any, model_sha8: str | None) -> None:
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
    signal_png = up.figure_png(up.signal_figure(r, compact=True))  # page-size copy for the PDF
    if r.qc_flags:
        st.markdown("**Automatic checks**\n" + "\n".join(f"- {n}: {why}" for n, why in r.qc_flags))
    else:
        st.markdown("**Automatic checks**: no flags")
    shown: dict[str, Any] | None = None
    folds_sha8: str | None = None
    if model is None:
        st.info("The score is not available until the demo models are trained.")
    else:
        try:
            folds_sha8 = res.fold_sha8(target, root)
            folds = get_fold_models(target, str(root), folds_sha8)
            features = sc.upload_features(r)
            result = sc.score(features, model, folds)
            explained = sc.explain(features, model)
            shown = render_score(
                target=target, root=root, score=result["score"], low=result["band_low"],
                high=result["band_high"], held_out=False, values=features,
                shap_values=explained["shap"], title=f"Contributions to the model score (clip {r.upload_ref})",
                fps=r.fps,
            )  # fmt: skip
        except (da.DemoDataError, da.GuardFailure) as err:
            st.info(f"No score for this clip: {err}")
    st.header("Per-tap amplitude")
    curve_shown = render_curve(tc.curve_for_upload(r), r.upload_ref)
    render_pdf_button(
        summary_data(
            ref=r.upload_ref,
            mode="upload",
            target=target,
            shown=shown,
            curve_shown=curve_shown,
            recording={"duration_s": r.duration_s, "fps": r.fps, "tracked_share": r.tracked_share},
            qc_flags=list(r.qc_flags),
            signal_png=signal_png,
            model_sha8=model_sha8,
            inputs=[x for x in (folds_sha8, shown["ref_sha"] if shown else None) if x],
        )
    )


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
    model_sha8: str | None = None
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
            shown = render_score(
                target=target, root=root, score=summary["proba"], low=min(probas),
                high=max(probas), held_out=True, values=summary["value"],
                shap_values=summary["shap"], title=f"Contributions to the model score (clip {ref})",
            )  # fmt: skip
            st.header("Per-tap amplitude")
            curve_shown = None
            try:
                curve_shown = render_curve(tc.curve_for_table(da.heldout_tap_table(ref, root)), ref)
            except da.DemoDataError as err:  # the curve is optional; the score above stands
                st.info(str(err))
            render_pdf_button(
                summary_data(
                    ref=ref,
                    mode="example",
                    target=target,
                    shown=shown,
                    curve_shown=curve_shown,
                    recording={},
                    qc_flags=None,
                    signal_png=None,
                    model_sha8=model_sha8,
                    inputs=[sha for _, sha in key],
                )
            )
        except (da.DemoDataError, da.GuardFailure) as err:
            st.error(f"The demo inputs are not available: {err}")
            st.stop()
    else:
        render_upload(target, root, model, model_sha8)
    st.caption(model_caption)
    st.caption("inputs: " + (", ".join(sha for _, sha in key) if key else "none loaded"))


main()
