"""Tests for models/tapping/demo/scoring.py (task FT-49). Hand-computed bands, percentiles and
sentences; synthetic worlds only."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent / "demo"))
sys.path.insert(0, str(HERE))

import cv_harness as ch  # noqa: E402
import cv_splits as cvs  # noqa: E402
import data_access as da  # noqa: E402
import features_a  # noqa: E402
import scoring as sc  # noqa: E402
import shap  # noqa: E402
import tap_table  # noqa: E402
import upload_pipeline as up  # noqa: E402
from cv_splits import GuardFailure  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from test_demo_resources import PARAMS, add_model, add_reference, frame  # noqa: E402
from test_demo_upload import Factory, fake_pipeline  # noqa: E402
from test_shap_figures import REFS, make_world, text_problems, values_table  # noqa: E402

COLS = sc.SET_A


class Const:
    """A stand-in model that always predicts one probability."""

    def __init__(self, p: float) -> None:
        self.p = p

    def predict(self, x: np.ndarray, raw_score: bool = False) -> np.ndarray:
        return np.full(len(x), self.p)


def features() -> dict[str, float]:
    return dict(zip(COLS, (1.5, 0.2, 7.0, 8.0, 1.1), strict=True))


# --------------------------------------------------------------------------- the score band


def test_band_is_the_5th_and_95th_percentile_of_the_25_fold_scores() -> None:
    p = [0.02, 0.9, 0.5, 0.11, 0.3, 0.97, 0.45, 0.6, 0.05, 0.33, 0.71, 0.18, 0.84, 0.26, 0.39,
         0.52, 0.64, 0.08, 0.77, 0.93, 0.15, 0.58, 0.22, 0.48, 0.69]  # fmt: skip
    got = sc.score(features(), Const(0.42), [Const(v) for v in p])
    v = sorted(p)  # linear interpolation: position 0.05 * 24 = 1.2 and 0.95 * 24 = 22.8
    assert got["band_low"] == pytest.approx(v[1] + 0.2 * (v[2] - v[1]), abs=1e-12)
    assert got["band_high"] == pytest.approx(v[22] + 0.8 * (v[23] - v[22]), abs=1e-12)
    assert got["score"] == 0.42 and got["n_models"] == 25
    assert got["band_low"] > min(p) and got["band_high"] < max(p)  # not the min/max


def test_the_band_uses_the_fold_models_not_the_demo_model() -> None:
    got = sc.score(features(), Const(0.99), [Const(0.1)] * 25)
    assert got["band_low"] == got["band_high"] == pytest.approx(0.1) and got["score"] == 0.99


def test_example_band_is_min_and_max_of_the_5_heldout_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    w = make_world(tmp_path, monkeypatch)
    table = values_table()
    for ref in (REFS[0], REFS[5]):
        probas = sorted(table[table["clip_ref"] == ref]["proba"])
        panel = sc.example_panel("ge2", ref, w["proc"])
        assert panel["band_low"] == pytest.approx(probas[0], abs=1e-15)
        assert panel["band_high"] == pytest.approx(probas[-1], abs=1e-15)
        assert panel["score"] == pytest.approx(sum(probas) / 5, abs=1e-15)
        assert panel["n_models"] == 5 and set(panel["shap"]) == set(COLS)
    with pytest.raises(da.DemoDataError):
        sc.example_panel("ge2", "12345678", w["proc"])


# --------------------------------------------------------------------------- SHAP


def small_model() -> Any:
    fr = frame()
    x, _ = ch.design_matrix(fr, "A")
    return ch.make_estimator("lgbm", PARAMS).fit(x, cvs.target_labels(fr, "ge2")).booster_


def test_shap_is_additive_and_uses_the_registered_explainer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = small_model()
    seen: dict[str, Any] = {}
    real = shap.TreeExplainer

    def spy(m: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return real(m, **kwargs)

    monkeypatch.setattr(sc.shap, "TreeExplainer", spy)
    out = sc.explain(features(), model)
    x = np.array([[features()[c] for c in COLS]])
    assert seen == {"feature_perturbation": "tree_path_dependent", "model_output": "raw"}
    raw = float(model.predict(x, raw_score=True)[0])
    assert out["raw_margin"] == raw
    assert out["base_value"] + sum(out["shap"].values()) == pytest.approx(raw, abs=1e-6)
    assert list(out["shap"]) == COLS


def test_explain_refuses_another_output_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    model = small_model()
    monkeypatch.setattr(sc, "EXPLAINER", {"feature_perturbation": "tree_path_dependent",
                                          "model_output": "probability"})  # fmt: skip
    with pytest.raises((GuardFailure, ValueError)):
        sc.explain(features(), model)


def test_explain_refuses_when_additivity_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    model = small_model()

    class Broken:  # values and base that cannot add up to the margin
        def __init__(self, m: Any, **kwargs: Any) -> None:
            self.expected_value = np.array([123.0])

        def shap_values(self, x: np.ndarray, check_additivity: bool = False) -> np.ndarray:
            return np.zeros(x.shape)

    monkeypatch.setattr(sc.shap, "TreeExplainer", Broken)
    with pytest.raises(GuardFailure, match="base value plus SHAP"):
        sc.explain(features(), model)


# --------------------------------------------------------------------------- upload features


@pytest.fixture
def upload_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    fake_pipeline(monkeypatch)
    video = tmp_path / "upload.mp4"
    video.write_bytes(b"synthetic video bytes")
    return up.process_video(video, Factory())


def test_upload_features_equal_the_official_features_a_call(
    monkeypatch: pytest.MonkeyPatch, upload_result: Any
) -> None:
    calls: list[str] = []
    real = features_a.clip_features

    def spy(table: Any, ref: str, summary: Any) -> Any:
        calls.append(ref)
        return real(table, ref, summary)

    monkeypatch.setattr(features_a, "clip_features", spy)
    got = sc.upload_features(upload_result)
    assert calls == [upload_result.upload_ref]
    summary = pd.Series(
        {
            "duration_s": upload_result.stats["duration_s"],
            "fps": upload_result.stats["fps"],
            "qc_status": "",
        }  # fmt: skip
    )
    direct = real(tap_table.build_tap_table(upload_result.taps_table), "x", summary)
    assert got == {c: float(direct[c]) for c in COLS}  # exact equality
    assert got["tap_rate_hz"] == pytest.approx(1.5, abs=0.05)  # the synthetic tapping rate


def test_too_few_taps_is_a_friendly_score_error(upload_result: Any) -> None:
    upload_result.taps_table = upload_result.taps_table.head(2)
    with pytest.raises(sc.ScoreError, match="Not enough valid taps"):
        sc.upload_features(upload_result)
    assert issubclass(sc.ScoreError, da.DemoDataError)


# --------------------------------------------------------------------------- context


def reference(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({c: values for c in COLS})


def test_percentile_and_range_by_hand() -> None:
    ref = reference([1.0, 2.0, 3.0, 4.0, 5.0])
    f = dict.fromkeys(COLS, 3.0)
    assert sc.context(f, ref)["amp_mean"] == {"percentile": 50.0, "in_range": True}
    f = {**f, "iti_cv": 6.0, "tap_rate_hz": 0.5, "open_vel_peak": 5.0, "close_vel_peak": 1.0}
    ctx = sc.context(f, ref)
    assert ctx["iti_cv"] == {"percentile": 100.0, "in_range": False}  # above the maximum
    assert ctx["tap_rate_hz"] == {"percentile": 0.0, "in_range": False}  # below the minimum
    assert ctx["open_vel_peak"]["in_range"] and ctx["close_vel_peak"]["in_range"]  # the edges
    assert ctx["open_vel_peak"]["percentile"] == 90.0  # (4 below + half of 1 equal) / 5


def test_percentile_ignores_missing_reference_values() -> None:
    ref = reference([1.0, 2.0, np.nan, 3.0])
    assert sc.context(dict.fromkeys(COLS, 2.0), ref)["amp_mean"]["percentile"] == 50.0


def test_load_reference_reads_only_the_set_a_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_reference(tmp_path)
    calls: list[dict[str, Any]] = []
    real = pd.read_csv

    def spy(path: Any, *a: Any, **k: Any) -> Any:
        calls.append({"name": Path(path).name, **k})
        return real(path, *a, **k)

    monkeypatch.setattr(pd, "read_csv", spy)
    ref = sc.load_reference(tmp_path)
    assert list(ref.columns) == COLS and len(ref) == 30
    assert calls == [{"name": "features.csv", "usecols": COLS}]
    with pytest.raises(da.DemoDataError, match="features file"):
        sc.load_reference(tmp_path / "nowhere")


# --------------------------------------------------------------------------- AUROC


def test_cv_auroc_is_the_results_table_row(tmp_path: Path) -> None:
    add_reference(tmp_path)
    assert sc.cv_auroc("ge2", tmp_path) == {"estimate": 0.87, "ci_low": 0.81, "ci_high": 0.92}
    assert sc.cv_auroc("ge1", tmp_path) == {"estimate": 0.895, "ci_low": 0.844, "ci_high": 0.942}
    path = tmp_path / "cv" / "results" / "results_table.csv"
    path.write_text(path.read_text().replace("lgbm A", "lgbm B"))
    with pytest.raises(GuardFailure, match="expected one lgbm A AUROC row"):
        sc.cv_auroc("ge2", tmp_path)
    path.unlink()
    with pytest.raises(da.DemoDataError, match="results table"):
        sc.cv_auroc("ge2", tmp_path)


# --------------------------------------------------------------------------- wording


def rows_for(shap_values: list[float], in_range: bool = True) -> Any:
    values = dict(zip(COLS, (1.55, 0.18, 7.95, 11.62, 1.146), strict=True))
    ctx = {c: {"percentile": p, "in_range": in_range}
           for c, p in zip(COLS, (71.0, 14.0, 31.0, 51.0, 3.0), strict=True)}  # fmt: skip
    return sc.contributions(values, dict(zip(COLS, shap_values, strict=True)), ctx)


def test_sentences_come_from_the_fixed_templates() -> None:
    rows = rows_for([0.31, -0.67, -0.13, 0.002, -0.4])
    assert sc.wording(rows) == [
        "Rhythm irregularity (0.18 CV, 14th percentile of the research clips) pushed the score "
        "down by 0.67 (log-odds).",
        "Mean amplitude (1.15 hand sizes, 3rd percentile of the research clips) pushed the score "
        "down by 0.40 (log-odds).",
        "Tapping rate (1.55 taps/s, 71st percentile of the research clips) pushed the score up "
        "by 0.31 (log-odds).",
        "Opening speed (7.95 hand sizes/s, 31st percentile of the research clips) pushed the "
        "score down by 0.13 (log-odds).",
        "Closing speed (11.62 hand sizes/s, 51st percentile of the research clips) had almost "
        "no effect on the score (+0.00 log-odds).",
    ]
    assert [r["feature"] for r in rows] == ["iti_cv", "amp_mean", "tap_rate_hz", "open_vel_peak",
                                            "close_vel_peak"]  # fmt: skip


def test_a_negative_contribution_reads_down_by_its_absolute_value() -> None:
    line = sc.wording(rows_for([0, -0.456, 0, 0, 0]))[0]
    assert line.endswith("pushed the score down by 0.46 (log-odds).")


def test_a_positive_contribution_reads_up_by_its_value() -> None:
    line = sc.wording(rows_for([0.456, 0, 0, 0, 0]))[0]
    assert line.endswith("pushed the score up by 0.46 (log-odds).")


def test_no_sign_character_follows_by_in_any_sentence() -> None:
    lines = sc.wording(rows_for([0.9, -0.7, 0.3, -0.2, 0.1]))
    assert all(" by " in line for line in lines) and len(lines) == 5
    for line in lines:
        assert not re.search(r"by [-+\u2212]", line), line
        assert re.search(r"by \d\.\d\d \(log-odds\)", line)


def test_out_of_range_is_stated_and_ordinals_are_right() -> None:
    line = sc.wording(rows_for([0.3, 0.1, 0.1, 0.1, 0.1], in_range=False))[0]
    assert line.endswith("; outside the range of the research clips.")
    assert [sc.ordinal(n) for n in (0, 1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 100, 101)] == [
        "0th", "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "23rd",
        "100th", "101st",
    ]  # fmt: skip


def test_no_sentence_names_a_cause_or_a_diagnosis() -> None:
    text = " ".join(sc.wording(rows_for([0.3, -0.2, 0.1, -0.1, 0.0])))
    assert not re.search(r"\bcaus|\bdisease|\bdiagnos|\bdetect", text, re.I)
    assert "pushed the score" in text and "caused" not in text


# --------------------------------------------------------------------------- the figure


def test_shap_figure_sorted_coloured_and_readable() -> None:
    rows = rows_for([0.31, -0.67, -0.13, 0.04, -0.4])
    fig = sc.shap_figure(rows, "Contributions to the model score (clip abcd1234)")
    ax = fig.axes[0]
    widths = [float(p.get_width()) for p in ax.containers[0]]  # type: ignore[attr-defined]
    assert len(widths) == 5 and widths == [-0.67, -0.4, 0.31, -0.13, 0.04]  # by |value|, top first
    assert [p.get_facecolor()[:3] == pytest.approx(
        tuple(int(c[i : i + 2], 16) / 255 for i in (1, 3, 5)), abs=1e-6)
        for p, c in zip(ax.patches, [sc.DOWN, sc.DOWN, sc.UP, sc.DOWN, sc.UP], strict=True)
    ] == [True] * 5  # fmt: skip
    texts = [t.get_text() for t in fig.findobj(Text) if t.get_text()]
    assert sc.X_LABEL in texts and "Rhythm irregularity (0.18 CV)" in texts
    assert not text_problems(fig)
    assert all(float(t.get_fontsize()) >= 11 for t in fig.findobj(Text) if t.get_text())


def test_signal_figure_legend_is_off_the_plot_area(upload_result: Any) -> None:
    fig = up.signal_figure(upload_result)
    fig.canvas.draw()
    assert fig.legends and not fig.axes[0].get_legend()
    legend, axes = fig.legends[0].get_window_extent(), fig.axes[0].get_window_extent()
    assert legend.y0 >= axes.y1 - 1 or legend.x0 >= axes.x1 - 1  # clear of the data area
    assert not text_problems(fig)


# --------------------------------------------------------------------------- models together


def test_score_and_explain_on_real_boosters(tmp_path: Path) -> None:
    add_model(tmp_path)
    from test_demo_resources import add_folds

    add_folds(tmp_path)
    import resources as res

    model = res.load_demo_model("ge2", tmp_path)
    out = sc.score(features(), model, res.load_fold_models("ge2", tmp_path))
    assert 0.0 <= out["band_low"] <= out["band_high"] <= 1.0 and 0.0 <= out["score"] <= 1.0
    e = sc.explain(features(), model)
    assert 1 / (1 + np.exp(-e["raw_margin"])) == pytest.approx(out["score"], abs=1e-12)
