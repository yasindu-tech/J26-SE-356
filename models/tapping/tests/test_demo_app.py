"""Smoke tests for models/tapping/demo/app.py (task FT-47) with streamlit AppTest, pointed at a
synthetic data/processed through TAPPING_DEMO_PROCESSED. No real data."""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

HERE = Path(__file__).resolve().parent
APP = HERE.parent / "demo" / "app.py"
sys.path.insert(0, str(HERE.parents[0] / "src"))
sys.path.insert(0, str(HERE.parents[0] / "demo"))
sys.path.insert(0, str(HERE))

import data_access as da  # noqa: E402
import resources as res  # noqa: E402
import streamlit as st  # noqa: E402
from test_demo_resources import add_model  # noqa: E402
from test_shap_figures import REFS, make_world  # noqa: E402

LABEL_WORDS = re.compile(r"rating|updrs|label|group|control", re.I)


def spy(log: list[Any], fn: Callable[..., Any]) -> Callable[..., Any]:
    def inner(*args: Any, **kwargs: Any) -> Any:
        log.append(args[0] if args else 1)
        return fn(*args, **kwargs)

    return inner


def page_text(at: AppTest) -> str:
    parts: list[str] = []
    for kind in ("title", "header", "markdown", "caption", "info", "error", "text_input"):
        parts += [str(getattr(el, "value", "")) for el in getattr(at, kind)]
    return "\n".join(parts)


@pytest.fixture(autouse=True)
def fresh_resource_cache() -> None:
    st.cache_resource.clear()  # the cache is process-wide; each test starts empty


def new_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_model: bool = True) -> AppTest:
    w = make_world(tmp_path, monkeypatch)
    if with_model:
        add_model(w["proc"])
    monkeypatch.setenv(da.ENV_ROOT, str(w["proc"]))
    return AppTest.from_file(str(APP), default_timeout=30)


@pytest.fixture
def at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setattr(res, "make_landmarker", lambda: object())  # hermetic: no model asset
    return new_app(tmp_path, monkeypatch).run()


def test_example_mode_runs_with_disclaimer_and_hashed_refs(at: AppTest) -> None:
    assert not at.exception
    assert [w.value for w in at.warning] == [
        "Research demo. The score is a model output on a research dataset; it is not a "
        "diagnosis and the model does not detect Parkinson's."
    ]
    clip_box = next(s for s in at.selectbox if s.label.startswith("Clip"))
    assert list(clip_box.options) == sorted(REFS)
    assert all(re.fullmatch(r"[0-9a-f]{8}", o) for o in clip_box.options)
    assert [h.value for h in at.header] == [
        "Clip", "Score", "Why this score (SHAP)", "Per-tap amplitude"
    ]  # fmt: skip
    assert at.caption[-1].value.startswith("inputs: ")
    assert at.caption[-2].value.endswith("(demo model, not used for metrics)")
    assert re.fullmatch(r"model: [0-9a-f]{8} \(.*\)", at.caption[-2].value)
    assert not LABEL_WORDS.search(page_text(at))


def test_upload_mode_runs_and_is_a_placeholder(at: AppTest) -> None:
    at.sidebar.radio[0].set_value("Upload video").run()
    assert not at.exception
    assert len(at.file_uploader) == 1 and not at.info
    assert any(
        c.value == "Processed on this computer; the video is not stored." for c in at.caption
    )
    assert not LABEL_WORDS.search(page_text(at))


def test_missing_target_shows_a_friendly_error_not_a_traceback(at: AppTest) -> None:
    at.sidebar.selectbox[0].set_value("ge1").run()  # the synthetic world only has ge2
    assert not at.exception
    assert len(at.error) == 1 and "not available" in at.error[0].value
    assert "Traceback" not in page_text(at)


def test_model_and_landmarker_are_cached_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loads: list[str] = []
    marks: list[int] = []
    real_load = res.load_demo_model
    monkeypatch.setattr(res, "load_demo_model", spy(loads, real_load))
    monkeypatch.setattr(res, "make_landmarker", spy(marks, lambda: object()))
    app = new_app(tmp_path, monkeypatch).run()
    app.run().run()  # two more reruns
    assert loads == ["ge2"] and marks == []  # one load in total; no landmarker in example mode
    app.sidebar.radio[0].set_value("Upload video").run()
    app.run().run()
    assert not app.exception and loads == ["ge2"] and marks == [1]  # created once, lazily


def test_missing_model_gives_a_friendly_message_and_example_mode_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = new_app(tmp_path, monkeypatch, with_model=False).run()
    assert not app.exception
    assert any("train_demo_model.py" in i.value for i in app.info)
    assert app.caption[-2].value == "model: not available"
    assert [h.value for h in app.header][0] == "Clip"  # example mode still renders
    assert "Traceback" not in page_text(app)
    assert not LABEL_WORDS.search(page_text(app))


def test_model_file_change_reloads_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loads: list[str] = []
    real_load = res.load_demo_model
    monkeypatch.setattr(res, "load_demo_model", spy(loads, real_load))
    app = new_app(tmp_path, monkeypatch).run()
    first = app.caption[-2].value
    root = da.processed_root()
    other = add_model(root, "ge2")  # same bytes: the cache is hit
    app.run()
    assert loads == ["ge2"] and app.caption[-2].value == first
    (other / "model_ge2_A.txt").write_bytes((other / "model_ge2_A.txt").read_bytes() + b"\n")
    app.run()  # a changed file has a new sha8, so it is refused by the meta check
    assert any("model_sha8" in i.value for i in app.info)
