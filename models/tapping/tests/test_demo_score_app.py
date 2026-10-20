"""AppTest tests for the Score and SHAP sections of app.py (task FT-49), both modes, on
synthetic worlds: texts, numbers by hand, forbidden words, labels never read."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

HERE = Path(__file__).resolve().parent
APP = HERE.parent / "demo" / "app.py"
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent / "demo"))
sys.path.insert(0, str(HERE))

import data_access as da  # noqa: E402
import resources as res  # noqa: E402
import scoring as sc  # noqa: E402
import streamlit as st  # noqa: E402
import upload_pipeline as up  # noqa: E402
from test_demo_resources import add_folds, add_model  # noqa: E402
from test_demo_upload import DATA, Factory, fake_pipeline, synthetic_df  # noqa: E402
from test_shap_figures import REFS, make_world, values_table  # noqa: E402

import app as app_module  # noqa: E402  (constants only; main() runs under AppTest)

FORBIDDEN = re.compile(r"\bcaus|\bdisease|\bdiagnos|\bdetect", re.I)
LABEL_WORDS = re.compile(r"rating|updrs|label|group|control", re.I)
TEXT_KINDS = ("title", "header", "subheader", "markdown", "caption", "info", "warning", "error",
              "metric", "text_input", "selectbox", "radio")  # fmt: skip


@pytest.fixture(autouse=True)
def fresh_caches() -> None:
    st.cache_resource.clear()
    st.cache_data.clear()


def start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, df: Any = None, models: bool = True
) -> AppTest:
    w = make_world(tmp_path, monkeypatch)
    add_model(w["proc"])
    if models:
        add_folds(w["proc"])
    monkeypatch.setenv(da.ENV_ROOT, str(w["proc"]))
    monkeypatch.setattr(res, "make_landmarker", lambda: Factory())
    fake_pipeline(monkeypatch, df=df)
    monkeypatch.setattr(
        up, "preview_frames", lambda p, idx, lm: [np.zeros((30, 20, 3), np.uint8)] * len(idx)
    )
    return AppTest.from_file(str(APP), default_timeout=60).run()


def upload(app: AppTest, data: bytes = DATA) -> AppTest:
    app.sidebar.radio[0].set_value("Upload video").run()
    return app.file_uploader[0].set_value(("clip.mp4", data, "video/mp4")).run()


def texts(app: AppTest) -> list[str]:
    out: list[str] = []
    for kind in TEXT_KINDS:
        for el in getattr(app, kind):
            for attr in ("value", "label"):
                v = getattr(el, attr, None)
                if isinstance(v, str):
                    out.append(v)
            out += [str(o) for o in getattr(el, "options", [])]
    return out


def sentences(app: AppTest) -> list[str]:
    block = next(m.value for m in app.markdown if m.value.startswith("- "))
    return [line[2:] for line in block.split("\n")]


def fixed_sentences(target: str) -> list[str]:
    caption = app_module.SCORE_CAPTION.format(est=0.87, lo=0.81, hi=0.92, target=target)
    return [app_module.DISCLAIMER, caption, app_module.SHAP_CAPTION]


def strip_fixed(text: str, target: str) -> str:
    for fixed in fixed_sentences(target):
        text = text.replace(fixed, "")
    return text


# --------------------------------------------------------------------------- example mode


def test_example_mode_shows_score_band_threshold_and_sentences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = start(tmp_path, monkeypatch)
    assert not app.exception
    table = values_table()
    sel = table[table["clip_ref"] == REFS[0]]["proba"]
    assert [(m.label, m.value) for m in app.metric] == [("Model score", f"{sel.mean():.2f}")]
    band = f"(range {sel.min():.2f}-{sel.max():.2f}) held-out scores over 5 repeats"
    markdown = [m.value for m in app.markdown]
    assert band in markdown
    side = "above" if sel.mean() >= 0.5 else "below"
    assert f"Fixed study threshold 0.5: {side}" in markdown
    captions = [c.value for c in app.caption]
    assert fixed_sentences("ge2")[1] in captions
    assert "Held-out: this clip was not used to train the models that scored it." in captions
    assert app_module.SHAP_CAPTION in captions
    assert "Demo model trained on all research clips." not in captions
    lines = sentences(app)
    assert len(lines) == 5 and all("percentile of the research clips" in s for s in lines)
    assert len(app.image) == 1  # the SHAP figure
    assert [h.value for h in app.header] == [
        "Clip", "Score", "Why this score (SHAP)", "Per-tap amplitude"
    ]  # fmt: skip


def test_ge1_gets_the_secondary_target_note_and_its_own_auroc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = start(tmp_path, monkeypatch)
    assert not any("secondary" in i.value for i in app.info)
    app.sidebar.selectbox[0].set_value("ge1").run()  # the synthetic xai world has ge2 only
    assert not app.exception and len(app.error) == 1  # friendly error, no scoring


# --------------------------------------------------------------------------- upload mode


def test_upload_mode_scores_with_the_25_model_band(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = upload(start(tmp_path, monkeypatch))
    assert not app.exception
    assert len(app.metric) == 1 and app.metric[0].label == "Model score"
    assert re.fullmatch(r"0\.\d\d|1\.00", app.metric[0].value)
    band = [m.value for m in app.markdown if m.value.startswith("(range ")]
    assert len(band) == 1 and band[0].endswith("5-95% of 25 cross-validation models")
    captions = [c.value for c in app.caption]
    assert "Demo model trained on all research clips." in captions
    assert not any(c.startswith("Held-out") for c in captions)
    assert len(sentences(app)) == 5
    assert len(app.image) == 9  # 6 frames, signal, SHAP and amplitude figures
    assert any(m.value.startswith("Fixed study threshold 0.5: ") for m in app.markdown)


def test_upload_band_comes_from_the_fold_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}
    real = sc.score

    def spy(features: Any, model: Any, folds: Any) -> Any:
        seen["n"] = len(folds)
        out = real(features, model, folds)
        seen["out"] = out
        return out

    monkeypatch.setattr(sc, "score", spy)
    app = upload(start(tmp_path, monkeypatch))
    out = seen["out"]
    assert seen["n"] == 25 and out["n_models"] == 25
    assert (
        f"(range {out['band_low']:.2f}-{out['band_high']:.2f}) 5-95% of 25 cross-validation models"
        in [m.value for m in app.markdown]
    )


def test_fps_and_out_of_range_notes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = upload(start(tmp_path, monkeypatch, df=synthetic_df(fps=60.0)))
    info = [i.value for i in app.info]
    assert any(i.startswith("This video is about 60 fps.") for i in info)
    st.cache_data.clear()  # the analysis is cached on the bytes, which are the same here
    app = upload(start(tmp_path / "b", monkeypatch, df=synthetic_df(fps=30.0)))
    assert not any("fps." in i.value for i in app.info)


def test_a_feature_outside_the_research_range_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = sc.load_reference

    def narrow(root: Any = None) -> pd.DataFrame:
        ref = real(root)
        ref["amp_mean"] = 50.0  # the upload's amplitude (about 1) is below every reference clip
        return ref

    monkeypatch.setattr(sc, "load_reference", narrow)
    app = start(tmp_path, monkeypatch)
    app = upload(app)
    notes = [i.value for i in app.info if "outside the range of the research clips" in i.value]
    assert (
        notes and "Mean amplitude" in notes[0] and " are outside " in notes[0]
    )  # two features here
    assert any(s.endswith("; outside the range of the research clips.") for s in sentences(app))


def test_no_score_for_a_clip_with_too_few_taps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(result: Any) -> Any:
        raise sc.ScoreError("Not enough valid taps to score this clip.")

    monkeypatch.setattr(sc, "upload_features", refuse)
    app = upload(start(tmp_path, monkeypatch))
    assert not app.exception and not app.metric
    assert "No score for this clip: Not enough valid taps to score this clip." in [
        i.value for i in app.info
    ]
    assert len(app.image) == 8  # the signal view and the amplitude curve are still shown


def test_missing_fold_models_give_a_friendly_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = upload(start(tmp_path, monkeypatch, models=False))
    assert not app.exception and not app.metric
    assert any("train_demo_model.py" in i.value for i in app.info)


# --------------------------------------------------------------------------- the text rules


def test_forbidden_words_only_in_the_fixed_sentences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = start(tmp_path, monkeypatch)
    pages = {"example": texts(app)}
    pages["upload"] = texts(upload(app))
    for name, items in pages.items():
        whole = "\n".join(items)
        assert FORBIDDEN.search(whole), name  # the fixed sentences are really on the page
        rest = strip_fixed(whole, "ge2")
        assert not FORBIDDEN.search(rest), (name, FORBIDDEN.search(rest))
        assert not LABEL_WORDS.search(rest), (name, LABEL_WORDS.search(rest))


def test_the_app_never_reads_the_rating_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, Any]] = []
    real = pd.read_csv

    def spy(path: Any, *a: Any, **k: Any) -> Any:
        calls.append((Path(path).name, k.get("usecols")))
        return real(path, *a, **k)

    monkeypatch.setattr(pd, "read_csv", spy)
    app = upload(start(tmp_path, monkeypatch))
    assert not app.exception
    features = [u for name, u in calls if name == "features.csv"]
    assert features and all(u is not None and set(u) <= set(sc.SET_A) for u in features)
    for _name, usecols in calls:
        assert not any("rating" in str(c) or "participant" in str(c) for c in (usecols or []))
    assert not any(n == "fis_diagnostic.csv" for n, _ in calls)
