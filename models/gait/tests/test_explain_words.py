"""GAIT-24: the plain-words sentences, the reference behind them and the explanation's faithfulness."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import explain_shap as es
import explain_words as ew
import model_bundle as mb
import numpy as np
import pytest
import train_xgboost as tx
from gait_extract import FEATURE_NAMES
from test_evaluate_cohorts import FAST, cohort_table

BANNED = re.compile(
    r"cause|because|due to|diagnos|parkinson|indicat|chance|disease|patient|\bpd\b", re.I
)
MARKS = {"p10": 10.0, "p25": 20.0, "p75": 40.0, "p90": 50.0}


def reference(cuts: tuple[float, float] = (0.2, 0.4), steady: bool = True) -> dict[str, Any]:
    return {
        "walks": 100,
        "percentiles": {f: dict(MARKS) for f in FEATURE_NAMES},
        "strength_cuts": list(cuts),
        "steady": {f: {"0": steady, "1": steady, "2": steady} for f in FEATURE_NAMES},
    }


def fake(feature: str, value: float, shap: float, klass: int = 2) -> es.Explanation:
    top = (es.Contribution(feature, value, shap),) * 3
    return es.Explanation(klass, (0.1, 0.2, 0.7), top)


# --------------------------------------------------------------------------- the pieces
@pytest.mark.parametrize(
    ("value", "words"),
    [
        (5.0, "lower than about 9 in 10 of the training walks"),
        (10.0, "lower than about 9 in 10 of the training walks"),
        (10.1, "on the low side of the training walks"),
        (20.0, "on the low side of the training walks"),
        (20.1, "in the middle range of the training walks"),
        (39.9, "in the middle range of the training walks"),
        (40.0, "on the high side of the training walks"),
        (49.9, "on the high side of the training walks"),
        (50.0, "higher than about 9 in 10 of the training walks"),
    ],
)
def test_where_a_value_sits(value: float, words: str) -> None:
    assert ew.where_it_sits(value, MARKS) == words


@pytest.mark.parametrize(
    ("size", "word"),
    [(0.0, "slightly"), (0.199, "slightly"), (0.2, "moderately"), (0.399, "moderately"),
     (0.4, "strongly"), (3.0, "strongly")],
)  # fmt: skip
def test_strength_word(size: float, word: str) -> None:
    assert ew.strength_word(size, [0.2, 0.4]) == word


def test_sentence_is_exactly_this_for_a_known_entry() -> None:
    words = ew.describe(fake("hip_flexion_range_deg", 9.0, 1.43), reference())
    assert words.sentences[0] == (
        "The model used hip swing range (9°), lower than about 9 in 10 of the training walks, "
        "which strongly pushed its score towards a moderate-to-severe impairment pattern."
    )
    assert words.entries[0].as_record()["name"] == "hip_flexion_range_deg"


def test_away_wording_and_class_words() -> None:
    away = ew.describe(fake("knee_flexion_range_deg", 37.0, -0.27, klass=0), reference())
    assert away.sentences[0].endswith(
        "moderately pushed its score away from no gait impairment pattern."
    )
    mild = ew.describe(fake("cadence", 120.0, 0.1, klass=1), reference())
    assert "towards a mild impairment pattern." in mild.sentences[0]


def test_varies_note_only_when_the_push_is_not_steady() -> None:
    steady = ew.describe(fake("cadence", 30.0, 0.3), reference(steady=True))
    unsteady = ew.describe(fake("cadence", 30.0, 0.3), reference(steady=False))
    assert ew.VARIES not in steady.sentences[0] and steady.sentences[0].endswith("pattern.")
    assert unsteady.sentences[0].endswith(f"pattern{ew.VARIES}.") and unsteady.entries[0].varies


def test_the_note_follows_the_predicted_class() -> None:
    ref = reference(steady=True)
    ref["steady"]["cadence"]["0"] = False
    assert ew.describe(fake("cadence", 30.0, 0.3, klass=0), ref).entries[0].varies
    assert not ew.describe(fake("cadence", 30.0, 0.3, klass=2), ref).entries[0].varies


def test_every_input_has_a_name_and_a_readable_value() -> None:
    assert set(ew.NAMES) == set(ew.SHOWN) == set(FEATURE_NAMES)
    assert ew.SHOWN["step_length_asymmetry"](0.104) == "10%"
    assert ew.SHOWN["knee_flexion_range_deg"](61.9) == "62°"
    assert ew.SHOWN["cadence"](110.2) == "110 steps/min"


# --------------------------------------------------------------------------- the reference
def values_and_inputs(
    steady_input: int, zigzag_input: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(size=(n, len(FEATURE_NAMES)))
    values = np.zeros((n, len(FEATURE_NAMES), 3))
    values[:, steady_input, 2] = 0.5 * x[:, steady_input]  # rises steadily with the value
    band = np.argsort(np.argsort(x[:, zigzag_input])) * 5 // n
    values[:, zigzag_input, 2] = np.array([0.5, -0.5, 0.5, -0.5, 0.5])[band]  # up and down
    values[:, 0, 0] = 0.001 * x[:, 0]  # a tiny effect is flat, not unsteady
    return x, values, np.full(n, 2)


def test_reference_percentiles_strength_and_steadiness() -> None:
    x, values, predicted = values_and_inputs(steady_input=3, zigzag_input=5)
    ref = ew.build_reference(x, values, predicted)
    j = 3
    assert ref["percentiles"][FEATURE_NAMES[j]]["p10"] == pytest.approx(np.percentile(x[:, j], 10))
    assert ref["percentiles"][FEATURE_NAMES[j]]["p90"] == pytest.approx(np.percentile(x[:, j], 90))
    assert ref["steady"][FEATURE_NAMES[3]]["2"] is True
    assert ref["steady"][FEATURE_NAMES[5]]["2"] is False
    assert ref["steady"][FEATURE_NAMES[0]]["0"] is True
    sizes = -np.sort(-np.abs(values[:, :, 2]), axis=1)[:, :3]
    assert ref["strength_cuts"] == pytest.approx(list(np.percentile(sizes, [100 / 3, 200 / 3])))
    assert ref["walks"] == 500


# --------------------------------------------------------------------------- with a real model
@pytest.fixture(scope="module")
def fitted() -> tuple[es.Explainer, np.ndarray]:
    import shap

    table = cohort_table()
    model = tx.fit_final_model(table, params=FAST)
    x = table[list(FEATURE_NAMES)].to_numpy(float)
    ref = ew.build_reference(
        x,
        tx.shap_array(shap.TreeExplainer(model).shap_values(x)),
        model.predict_proba(x).argmax(axis=1),
    )
    import tempfile

    folder = Path(tempfile.mkdtemp())
    bundle = mb.save_bundle(model, folder, {"explanation_reference": ref})
    return es.Explainer(bundle), x


def test_sentences_of_every_walk_are_safe_and_match_the_numbers(
    fitted: tuple[es.Explainer, np.ndarray],
) -> None:
    explainer, x = fitted
    assert explainer.bundle.reference["walks"] == len(x)
    for i in range(0, len(x), 3):
        words = explainer.in_words(x[i])
        assert len(words.entries) == 3 and words.predicted_class == int(
            words.probabilities.index(max(words.probabilities))
        )
        for e in words.entries:
            s = e.sentence
            assert s.startswith("The model used ") and s.endswith(".") and "nan" not in s.lower()
            assert not BANNED.search(s), s
            assert ew.NAMES[e.feature] in s and ew.CLASS_WORDS[words.predicted_class] in s
            assert (" pushed its score towards " in s) == (e.shap > 0)
            assert (" pushed its score away from " in s) == (e.shap <= 0)
            assert e.strength in s and (ew.VARIES in s) == e.varies


def test_words_never_change_the_numbers(fitted: tuple[es.Explainer, np.ndarray]) -> None:
    explainer, x = fitted
    plain, words = explainer.explain(x[4]), explainer.in_words(x[4])
    assert [(c.feature, c.value, c.shap, c.direction) for c in plain.top] == [
        (e.feature, e.value, e.shap, e.direction) for e in words.entries
    ]


def test_in_words_uses_the_bundles_own_reference(fitted: tuple[es.Explainer, np.ndarray]) -> None:
    explainer, x = fitted
    ref = explainer.bundle.reference
    for i in (0, 50, 120):
        words = explainer.in_words(x[i])
        assert words == ew.describe(explainer.explain(x[i]), ref)
        for e in words.entries:
            assert e.strength == ew.strength_word(abs(e.shap), ref["strength_cuts"])
            assert e.varies == (not ref["steady"][e.feature][str(words.predicted_class)])


def test_the_top_three_changes_the_score_more_than_random_inputs(
    fitted: tuple[es.Explainer, np.ndarray],
) -> None:
    explainer, x = fitted
    model = explainer.bundle.model
    proba = explainer.bundle.predict_proba(x)
    pick = proba.argmax(axis=1)
    median = np.median(x, axis=0)
    values = explainer.values(x)[np.arange(len(x)), :, pick]
    rng = np.random.default_rng(0)

    def drop(columns: list[np.ndarray]) -> float:
        changed = x.copy()
        for i, cols in enumerate(columns):
            changed[i, cols] = median[cols]
        after = np.asarray(model.predict_proba(changed))
        return float((proba[np.arange(len(x)), pick] - after[np.arange(len(x)), pick]).mean())

    top = list(np.argsort(-np.abs(values), axis=1)[:, :3])
    bottom = list(np.argsort(np.abs(values), axis=1)[:, :3])
    random = [rng.choice(len(FEATURE_NAMES), 3, replace=False) for _ in range(len(x))]
    assert drop(top) > 1.3 * drop(random) and drop(top) > 2 * drop(bottom)


def test_bundle_without_the_reference_is_refused_by_the_words(tmp_path: Path) -> None:
    table = cohort_table()
    model = tx.fit_final_model(table, params=FAST)
    bundle = mb.save_bundle(model, tmp_path / "b", {})
    with pytest.raises(mb.BundleError, match="rerun train_xgboost"):
        es.Explainer(bundle).in_words(table[list(FEATURE_NAMES)].to_numpy(float)[0])
