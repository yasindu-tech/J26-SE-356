"""Tests for models/tapping/src/explanation_stability.py (task FT-45). Hand-computed statistics
first, then synthetic xai worlds (no real data) for the files, figure and guards."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv_splits as cvs  # noqa: E402
import explanation_stability as es  # noqa: E402
import shap_figures as sf  # noqa: E402
import treeshap_heldout as tsh  # noqa: E402
from test_shap_figures import COLS, make_world, text_problems  # noqa: E402

FILES = (es.PAIRS_FILE, es.CLIPS_FILE, es.SUMMARY_FILE, es.FIG_FILE, es.META_FILE)
LABEL_WORDS = re.compile(r"rating|updrs|label|group|diagnos|detect|control", re.I)
N_CLIPS = 20


def table25(seed: int = 0, signal: float = 1.0) -> pd.DataFrame:
    """5 repeats x 20 clips, five folds per repeat (four clips each), a stable feature scale."""
    rng = np.random.default_rng(seed)
    scale = np.array([2.0, 1.2, 0.8, 0.5, 0.2]) * signal
    rows = []
    for repeat in range(5):
        for i in range(N_CLIPS):
            shap_v = rng.normal(size=len(COLS)) * scale + 0.05 * rng.normal(size=len(COLS))
            base = -1.0 + 0.02 * repeat
            raw = base + float(shap_v.sum())
            row = {"repeat": repeat, "outer_fold": (i + repeat) % 5, "clip_ref": f"{0x20 + i:08x}",
                   "participant_ref": f"{0x80 + i // 2:08x}", "base_value": base, "raw_margin": raw,
                   "proba": float(1 / (1 + np.exp(-raw)))}  # fmt: skip
            for j, c in enumerate(COLS):
                row[f"shap_{c}"] = shap_v[j]
            for j, c in enumerate(COLS):
                row[f"value_{c}"] = 0.1 * (i + 1) * (j + 1)
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def configs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(es, "CONFIGS", (("ge2", "A"),))


@pytest.fixture
def w(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    return make_world(tmp_path, monkeypatch, table25())


def go(w: dict[str, Path], *extra: str) -> int:
    return es.main(["--cv-root", str(w["cv"]), "--xai-root", str(w["xai"]), *extra])


def out(w: dict[str, Path]) -> Path:
    return w["xai"] / "stability" / "ge2" / "A"


def inputs_of(w: dict[str, Path]) -> dict[str, Any]:
    return sf.read_inputs(w["in"], COLS)


# --------------------------------------------------------------------------- constants


def test_preregistered_constants() -> None:
    source = Path(es.__file__).read_text()
    assert 'CONFIGS = (("ge2", "A"), ("ge1", "A"), ("ge2", "B"))' in source
    assert es.TARGET == 0.70 and es.N_FOLDS == 25
    assert "pyplot" not in source and "FigureCanvasAgg" in source
    assert "features.csv" not in source and "fis_diagnostic" not in source
    assert es.FOOTER == (
        "Rank of mean |SHAP| per held-out fold. Stability of the explanation, not of the disease "
        "signal; folds share most training data."
    )


# --------------------------------------------------------------------------- spearman and pairs


def test_spearman_by_hand_with_ties() -> None:
    a, b = np.array([1.0, 2.0, 2.0, 4.0]), np.array([1.0, 3.0, 2.0, 4.0])
    assert es.spearman(a, b) == pytest.approx(
        4.5 / np.sqrt(4.5 * 5.0)
    )  # ranks 1,2.5,2.5,4 vs 1,3,2,4
    assert es.spearman(a, b) == pytest.approx(float(spearmanr(a, b).statistic))
    assert es.spearman(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])) == pytest.approx(-1.0)
    assert es.spearman(
        np.array([1.0, 2.0, 3.0, 4.0]), np.array([10.0, 20.0, 30.0, 400.0])
    ) == pytest.approx(1.0)


def test_spearman_is_not_pearson() -> None:
    a, b = np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    assert es.spearman(a, b) == pytest.approx(1.0)
    assert np.corrcoef(a, b)[0, 1] < 0.99  # Pearson would say less


def test_ties_use_average_ranks_not_order() -> None:
    a, b = np.array([5.0, 5.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0, 4.0])
    ra, rb = rankdata(a), rankdata(b)  # average ranks: 3.5, 3.5, 1, 2
    expected = float(np.corrcoef(ra, rb)[0, 1])
    assert es.spearman(a, b) == pytest.approx(expected)
    assert es.spearman(a[::-1], b[::-1]) == pytest.approx(expected)  # not order-dependent


def test_constant_vector_gives_nan_never_zero_or_one() -> None:
    assert np.isnan(es.spearman(np.ones(5), np.arange(5.0)))
    assert np.isnan(es.spearman(np.arange(5.0), np.ones(5)))


def test_pair_table_has_300_pairs_split_50_within_and_250_between() -> None:
    ids = [(r, f) for r in range(5) for f in range(5)]
    rng = np.random.default_rng(1)
    pairs = es.pair_table(ids, rng.normal(size=(25, 6)))
    assert len(pairs) == 300 == 25 * 24 // 2
    assert int(pairs["same_repeat"].sum()) == 50 and int((~pairs["same_repeat"]).sum()) == 250
    assert list(pairs.columns) == ["fold_a", "fold_b", "same_repeat", "rho"]
    assert pairs.iloc[0]["fold_a"] == "r0f0" and pairs.iloc[0]["fold_b"] == "r0f1"
    assert not pairs.duplicated(["fold_a", "fold_b"]).any()


def test_pair_rho_equals_the_hand_computed_value() -> None:
    ids = [(0, 0), (0, 1), (1, 0)]
    m = np.array([[1.0, 2.0, 2.0, 4.0], [1.0, 3.0, 2.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    pairs = es.pair_table(ids, m)
    assert pairs.loc[0, "rho"] == pytest.approx(4.5 / np.sqrt(4.5 * 5.0))
    assert (
        pairs.loc[1, "rho"] == pytest.approx(-0.9486832980505138)
        and not pairs.loc[1, "same_repeat"]
    )
    assert pairs.loc[2, "rho"] == pytest.approx(-0.8)


def test_a_constant_fold_vector_is_refused() -> None:
    ids = [(0, 0), (0, 1)]
    with pytest.raises(cvs.GuardFailure, match="constant"):
        es.pair_table(ids, np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]]))


def summary_of(matrix: np.ndarray, names: list[str], glob: pd.DataFrame) -> dict[str, Any]:
    ids = [(r, f) for r in range(5) for f in range(5)]
    clips = pd.DataFrame(
        {"clip_ref": ["a"], "mean_rho_over_repeats": [0.5], "n_pairs_defined": [10]}
    )
    return es.summarise(es.pair_table(ids, matrix), matrix, names, glob, clips)


GLOB = pd.DataFrame({"feature": ["a", "b", "c", "d"], "rank": [1, 2, 3, 4],
                     "mean_abs_shap_mean_over_repeats": [4.0, 3.0, 2.0, 1.0]})  # fmt: skip


def test_identical_vectors_give_rho_1_w_1_and_full_agreement() -> None:
    s = summary_of(np.tile([4.0, 3.0, 2.0, 1.0], (25, 1)), ["a", "b", "c", "d"], GLOB)
    assert s["mean_rho"] == pytest.approx(1.0) and s["kendall_w"] == pytest.approx(1.0)
    assert s["top1_agreement"] == 1.0 and s["top2_agreement"] == 1.0
    assert (
        s["share_ge_target"] == 1.0
        and s["meets_target"] is True
        and s["min_rho"] == pytest.approx(1.0)
    )
    assert s["mean_rho_within_repeat"] == pytest.approx(1.0) and s[
        "mean_rho_between_repeat"
    ] == pytest.approx(1.0)


def test_reversed_halves_give_low_rho_and_low_w() -> None:
    m = np.array([[4.0, 3.0, 2.0, 1.0]] * 12 + [[1.0, 2.0, 3.0, 4.0]] * 13)
    s = summary_of(m, ["a", "b", "c", "d"], GLOB)
    assert s["mean_rho"] < 0 and s["meets_target"] is False
    assert s["min_rho"] == pytest.approx(-1.0) and s["kendall_w"] < 0.1
    assert s["top1_agreement"] == pytest.approx(12 / 25)


def test_meets_target_boundary_exactly_at_0_70_counts() -> None:
    assert es.meets_target(0.70) is True
    assert es.meets_target(0.7000001) is True
    assert es.meets_target(0.6999999) is False
    assert es.meets_target(es.TARGET) is True


def test_within_and_between_repeat_means_are_split_correctly() -> None:
    rng = np.random.default_rng(3)
    ids = [(r, f) for r in range(5) for f in range(5)]
    m = rng.normal(size=(25, 7))
    pairs = es.pair_table(ids, m)
    within = np.mean(
        [
            spearmanr(m[i], m[j]).statistic
            for i, j in itertools.combinations(range(25), 2)
            if ids[i][0] == ids[j][0]
        ]
    )
    between = np.mean(
        [
            spearmanr(m[i], m[j]).statistic
            for i, j in itertools.combinations(range(25), 2)
            if ids[i][0] != ids[j][0]
        ]
    )
    names = [f"f{i}" for i in range(7)]
    glob = pd.DataFrame(
        {"feature": names, "rank": range(1, 8), "mean_abs_shap_mean_over_repeats": 1.0}
    )
    clips = pd.DataFrame(
        {"clip_ref": ["a"], "mean_rho_over_repeats": [0.5], "n_pairs_defined": [10]}
    )
    s = es.summarise(pairs, m, names, glob, clips)
    assert s["mean_rho_within_repeat"] == pytest.approx(within) and s[
        "mean_rho_between_repeat"
    ] == pytest.approx(between)
    assert s["mean_rho"] == pytest.approx(pairs["rho"].mean()) and s["median_rho"] == pytest.approx(
        pairs["rho"].median()
    )
    assert s["p05_rho"] == pytest.approx(np.percentile(pairs["rho"], 5))
    assert s["share_ge_target"] == pytest.approx((pairs["rho"] >= 0.7).mean())


# --------------------------------------------------------------------------- Kendall W


def test_kendall_w_by_hand_without_ties() -> None:
    v = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 4.0, 3.0]])
    assert es.kendall_w(v) == pytest.approx(12 * 41 / (9 * 60))  # S = 41, m = 3, n = 4
    assert es.kendall_w(v) == pytest.approx(0.9111111111)


def test_kendall_w_by_hand_with_ties_is_corrected() -> None:
    v = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 2.0]])
    assert es.kendall_w(v) == pytest.approx(
        78 / 84
    )  # 12 S / (m^2 (n^3 - n) - m T) = 78 / (96 - 12)
    assert es.kendall_w(v) != pytest.approx(78 / 96)  # the uncorrected value


def test_kendall_w_extremes() -> None:
    same = np.tile([3.0, 1.0, 2.0], (4, 1))
    assert es.kendall_w(same) == pytest.approx(1.0)
    opposed = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    assert es.kendall_w(opposed) == pytest.approx(0.0)
    with pytest.raises(cvs.GuardFailure, match="undefined"):
        es.kendall_w(np.ones((3, 4)))


# --------------------------------------------------------------------------- top agreement


def test_top1_and_top2_agreement_by_hand_with_the_global_rank_one() -> None:
    names = ["a", "b", "c"]
    glob = pd.DataFrame(
        {"feature": names, "rank": [2, 1, 3], "mean_abs_shap_mean_over_repeats": [1, 2, 0]}
    )
    m = np.array([[1.0, 5.0, 2.0], [4.0, 5.0, 1.0], [9.0, 1.0, 2.0]])
    assert es.top_agreement(m, names, glob) == pytest.approx((2 / 3, 1 / 3))
    other = glob.assign(rank=[1, 2, 3])  # the global rank-1 is now "a": fold 0 would say "b"
    assert es.top_agreement(m, names, other) == pytest.approx((1 / 3, 1 / 3))


def test_top_agreement_ties_go_to_the_smaller_feature_name() -> None:
    names = ["a", "b", "c"]
    glob = pd.DataFrame(
        {"feature": names, "rank": [1, 2, 3], "mean_abs_shap_mean_over_repeats": [3, 2, 1]}
    )
    m = np.array([[2.0, 2.0, 1.0]])
    assert es.top_agreement(m, names, glob) == (1.0, 1.0)


# --------------------------------------------------------------------------- per clip


def clip_values(vectors: np.ndarray, ref: str = "0000000a") -> pd.DataFrame:
    rows = [
        {"repeat": r, "clip_ref": ref, **{f"shap_{c}": vectors[r][j] for j, c in enumerate(COLS)}}
        for r in range(5)
    ]
    return pd.DataFrame(rows)


def test_per_clip_hand_examples() -> None:
    same = np.tile(np.arange(5.0), (5, 1))
    assert es.clip_table(clip_values(same), COLS).iloc[0]["mean_rho_over_repeats"] == pytest.approx(
        1.0
    )
    alternating = np.array(
        [np.arange(5.0) if r % 2 == 0 else np.arange(5.0)[::-1] for r in range(5)]
    )
    row = es.clip_table(clip_values(alternating), COLS).iloc[0]
    assert (
        row["mean_rho_over_repeats"] == pytest.approx((4 - 6) / 10) and row["n_pairs_defined"] == 10
    )


def test_a_constant_clip_vector_is_excluded_counted_and_never_0_or_1() -> None:
    vectors = np.tile(np.arange(5.0), (5, 1))
    vectors[2] = 0.0  # one repeat's SHAP vector is constant
    row = es.clip_table(clip_values(vectors), COLS).iloc[0]
    assert np.isnan(row["mean_rho_over_repeats"]) and row["n_pairs_defined"] == 6  # C(4, 2)
    clips = pd.concat([es.clip_table(clip_values(vectors, "0000000a"), COLS),
                       es.clip_table(clip_values(np.tile(np.arange(5.0), (5, 1)), "0000000b"), COLS)])  # fmt: skip
    assert int(clips["mean_rho_over_repeats"].isna().sum()) == 1
    assert clips["mean_rho_over_repeats"].dropna().tolist() == [
        pytest.approx(1.0)
    ]  # the other clip


def test_per_clip_uses_all_ten_repeat_pairs() -> None:
    rng = np.random.default_rng(5)
    vectors = rng.normal(size=(5, 5))
    row = es.clip_table(clip_values(vectors), COLS).iloc[0]
    expected = np.mean(
        [
            spearmanr(vectors[i], vectors[j]).statistic
            for i, j in itertools.combinations(range(5), 2)
        ]
    )
    assert row["n_pairs_defined"] == 10 and row["mean_rho_over_repeats"] == pytest.approx(expected)


def test_a_clip_missing_a_repeat_is_refused() -> None:
    values = clip_values(np.tile(np.arange(5.0), (5, 1))).iloc[1:]
    with pytest.raises(cvs.GuardFailure, match="every repeat"):
        es.clip_table(values, COLS)


def test_summary_counts_excluded_clips_and_summarises_the_rest() -> None:
    clips = pd.DataFrame({"clip_ref": list("abcd"), "mean_rho_over_repeats": [0.2, np.nan, 0.6, 1.0],
                          "n_pairs_defined": [10, 6, 10, 10]})  # fmt: skip
    ids = [(r, f) for r in range(5) for f in range(5)]
    m = np.tile([4.0, 3.0, 2.0, 1.0], (25, 1))
    s = es.summarise(es.pair_table(ids, m), m, ["a", "b", "c", "d"], GLOB, clips)
    assert s["n_clips_excluded"] == 1
    assert s["clip_rho_median"] == pytest.approx(0.6)
    assert s["clip_rho_p25"] == pytest.approx(np.percentile([0.2, 0.6, 1.0], 25))
    assert s["clip_rho_p75"] == pytest.approx(np.percentile([0.2, 0.6, 1.0], 75))


# --------------------------------------------------------------------------- files


def test_outputs_match_an_independent_recomputation(w: dict[str, Path]) -> None:
    assert go(w) == 0
    data = inputs_of(w)
    by_fold, glob = data["by_fold"], data["glob"]
    wide = by_fold.pivot(index=["repeat", "outer_fold"], columns="feature", values="mean_abs_shap")
    assert wide.shape == (25, 5)
    rhos = [
        spearmanr(wide.iloc[i], wide.iloc[j]).statistic
        for i, j in itertools.combinations(range(25), 2)
    ]
    pairs = pd.read_csv(out(w) / es.PAIRS_FILE)
    assert len(pairs) == 300 and np.allclose(pairs["rho"], rhos)
    summary = pd.read_csv(out(w) / es.SUMMARY_FILE)
    assert list(summary.columns) == es.SUMMARY_COLUMNS and len(summary) == 1
    s = summary.iloc[0]
    assert (
        s["mean_rho"] == pytest.approx(np.mean(rhos))
        and s["n_folds"] == 25
        and s["n_features"] == 5
    )
    assert s["target"] == pytest.approx(0.7) and bool(s["meets_target"]) == (np.mean(rhos) >= 0.7)
    assert s["kendall_w"] == pytest.approx(es.kendall_w(wide.to_numpy()))
    top1 = glob.sort_values("rank")["feature"].iloc[0]
    expected_top1 = np.mean([wide.columns[int(np.argmax(row))] == top1 for row in wide.to_numpy()])
    assert s["top1_agreement"] == pytest.approx(expected_top1)
    clips = pd.read_csv(out(w) / es.CLIPS_FILE, dtype={"clip_ref": str})
    assert len(clips) == N_CLIPS and list(clips.columns) == [
        "clip_ref",
        "mean_rho_over_repeats",
        "n_pairs_defined",
    ]
    assert (clips["n_pairs_defined"] == 10).all() and s["n_clips_excluded"] == 0


def test_a_stable_world_meets_the_target_and_a_noisy_one_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_world(tmp_path, monkeypatch, table25())
    stable = es.run_config(
        tmp_path / "data" / "processed" / "cv", tmp_path / "data" / "processed" / "xai", "ge2", "A"
    )
    assert stable["summary"]["mean_rho"] > 0.7 and stable["summary"]["meets_target"] is True
    noisy = table25(seed=1)
    rng = np.random.default_rng(9)
    for c in COLS:
        noisy[f"shap_{c}"] = rng.normal(size=len(noisy))  # no stable feature scale at all
    root = tmp_path / "other"
    root.mkdir()
    with pytest.MonkeyPatch.context() as mp:
        make_world(root, mp, noisy)
        res = es.run_config(
            root / "data" / "processed" / "cv", root / "data" / "processed" / "xai", "ge2", "A"
        )
    assert res["summary"]["mean_rho"] < 0.7 and res["summary"]["meets_target"] is False


# --------------------------------------------------------------------------- figure


def figure_for(w: dict[str, Path]) -> tuple[Any, np.ndarray, list[str]]:
    data = inputs_of(w)
    order = sf.feature_order(data["glob"])
    ids, matrix = es.fold_matrix(data["by_fold"], order)
    pairs = es.pair_table(ids, matrix)
    clips = es.clip_table(data["values"], COLS)
    summary = es.summarise(pairs, matrix, order, data["glob"], clips)
    return es.fig_ranks("ge2", "A", ids, order, matrix, summary), matrix, order


def test_png_magic_bytes_and_text_inside_the_canvas(w: dict[str, Path]) -> None:
    assert go(w) == 0
    assert (out(w) / es.FIG_FILE).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    fig, _, _ = figure_for(w)
    assert text_problems(fig) == []


def test_rank_numbers_in_the_cells_equal_the_computed_ranks(w: dict[str, Path]) -> None:
    fig, matrix, order = figure_for(w)
    ax = fig.axes[0]
    ranks = es.fold_ranks(matrix).T  # rows = features in global-rank order, columns = folds
    assert ranks.shape == (5, 25)
    shown = np.array(ax.images[0].get_array())
    assert np.array_equal(shown, ranks)  # not transposed
    cells = [t.get_text() for t in ax.texts]
    assert cells == [es.rank_text(float(ranks[i, j])) for i in range(5) for j in range(25)]
    assert [t.get_text() for t in ax.get_yticklabels()] == order
    assert set(np.unique(ranks)) <= {1.0, 2.0, 3.0, 4.0, 5.0}
    assert (ranks.sum(axis=0) == 15).all()  # every fold ranks the five features 1..5


def test_figure_texts_separators_fonts_and_footer(w: dict[str, Path]) -> None:
    fig, _, _ = figure_for(w)
    texts = sf.figure_texts(fig)
    assert es.FOOTER in texts and "ge2, Set A, LightGBM: feature rank by outer fold" in texts
    assert any(
        t.startswith("mean Spearman rho") and "Kendall W" in t and "fold pairs" in t for t in texts
    )
    assert (
        any("cell = rank, 1 = largest mean |SHAP|" in t for t in texts)
        and "repeat 0" in texts
        and "repeat 4" in texts
    )
    assert not [t for t in texts if LABEL_WORDS.search(t)]
    for artist in fig.findobj(sf.Text):
        if artist.get_text():
            assert float(artist.get_fontsize()) >= 11
    assert tuple(fig.get_size_inches()) == sf.FIGSIZE and fig.dpi == 200
    lines = [ln.get_xdata()[0] for ln in fig.axes[0].lines if len(set(ln.get_xdata())) == 1]
    assert sorted(lines) == [4.5, 9.5, 14.5, 19.5]  # four separators between five repeats


def test_names_meta_determinism_and_privacy(
    w: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert go(w) == 0
    console = capsys.readouterr().out
    assert sorted(p.name for p in out(w).iterdir()) == sorted(FILES)
    first = {n: (out(w) / n).read_bytes() for n in FILES}
    assert go(w) == 0
    assert {
        n: (out(w) / n).read_bytes() for n in FILES
    } == first and capsys.readouterr().out == console
    meta = json.loads((out(w) / es.META_FILE).read_text())
    assert meta["constants"]["target_mean_rho"] == 0.7 and meta["constants"]["n_pairs"] == 300
    assert set(meta["versions"]) == {"matplotlib", "numpy", "scipy"} and set(
        meta["inputs_sha8"]
    ) == set(sf.SHAP_INPUTS)
    for n, h in meta["files_sha8"].items():
        assert cvs.sha8_of((out(w) / n).read_bytes()) == h
    clips = pd.read_csv(out(w) / es.CLIPS_FILE, dtype={"clip_ref": str})
    assert clips["clip_ref"].map(lambda v: bool(re.fullmatch(r"[0-9a-f]{8}", v))).all()
    text = console + "".join(
        first[n].decode(errors="ignore") for n in FILES if not n.endswith(".png")
    )
    assert not LABEL_WORDS.search(text) and not re.search(
        r"CONTROL|ID\d|DCHA|IZDA|parquet|/Users", text
    )
    assert not re.search(r"20\d\d-\d\d-\d\dT|\d\d:\d\d:\d\d", text)
    assert all(len(line) < 160 for line in console.splitlines())


def test_sources_are_never_modified(w: dict[str, Path]) -> None:
    def snap() -> dict[str, str]:
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(w["proc"].rglob("*")) if p.is_file() and "stability" not in p.parts}  # fmt: skip

    before = snap()
    assert go(w) == 0
    assert snap() == before


def test_failed_atomic_write_keeps_the_old_file(
    w: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert go(w) == 0
    (out(w) / es.PAIRS_FILE).write_bytes(b"old")
    real = cvs.os.replace

    def fail_csv(src: Any, dst: Any) -> None:
        if str(dst).endswith(".csv"):
            raise OSError("fail")
        real(src, dst)

    monkeypatch.setattr(cvs.os, "replace", fail_csv)
    with pytest.raises(OSError):
        go(w)
    assert (out(w) / es.PAIRS_FILE).read_bytes() == b"old"
    assert not [p for p in out(w).iterdir() if p.name.startswith(".")]


# --------------------------------------------------------------------------- guards


def edit_json(path: Path, fn: Any) -> None:
    data = json.loads(path.read_text())
    fn(data)
    path.write_text(json.dumps(data))


def test_oof_not_reproduced_exits_1(w: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    edit_json(w["in"] / tsh.META_FILE, lambda m: m.update(oof_reproduced=False))
    assert go(w) == 1
    assert "did not reproduce" in capsys.readouterr().err and not out(w).exists()


@pytest.mark.parametrize("key", ["features_sha8", "grid_sha8", "splits_sha8"])
def test_hash_differing_from_the_official_meta_exits_1(w: dict[str, Path], key: str) -> None:
    edit_json(w["in"] / tsh.META_FILE, lambda m: m.update({key: "deadbeef"}))
    assert go(w) == 1 and not out(w).exists()


def test_row_count_mismatch_exits_1(w: dict[str, Path]) -> None:
    edit_json(w["in"] / tsh.META_FILE, lambda m: m["row_counts"].update({tsh.FOLD_FILE: 1}))
    assert go(w) == 1


def test_wrong_tool_or_feature_set_exits_1(w: dict[str, Path]) -> None:
    edit_json(w["in"] / tsh.META_FILE, lambda m: m.update(tool="other"))
    assert go(w) == 1


def test_a_missing_fold_or_duplicate_row_exits_1(w: dict[str, Path]) -> None:
    path = w["in"] / tsh.FOLD_FILE
    table = pd.read_csv(path)
    keep = table[~((table["repeat"] == 4) & (table["outer_fold"] == 4))]
    keep.to_csv(path, index=False, float_format="%.17g")
    edit_json(w["in"] / tsh.META_FILE, lambda m: m["row_counts"].update({tsh.FOLD_FILE: len(keep)}))
    assert go(w) == 1  # 24 folds
    dup = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    dup.to_csv(path, index=False, float_format="%.17g")
    edit_json(w["in"] / tsh.META_FILE, lambda m: m["row_counts"].update({tsh.FOLD_FILE: len(dup)}))
    assert go(w) == 1


def test_a_constant_fold_vector_in_the_file_exits_1(w: dict[str, Path]) -> None:
    path = w["in"] / tsh.FOLD_FILE
    table = pd.read_csv(path)
    table.loc[(table["repeat"] == 0) & (table["outer_fold"] == 0), "mean_abs_shap"] = 0.5
    table.to_csv(path, index=False, float_format="%.17g")
    assert go(w) == 1


def test_missing_inputs_exit_2(w: dict[str, Path]) -> None:
    (w["in"] / tsh.GLOBAL_FILE).unlink()
    assert go(w) == 2
    (w["in"] / tsh.GLOBAL_FILE).write_text("x")
    (w["cv"] / "ge2" / "meta_lgbm_A.json").unlink()
    assert go(w) == 2


def test_xai_root_outside_the_xai_folder_exits_2(w: dict[str, Path], tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    shutil.copytree(w["xai"], elsewhere)  # complete inputs: only the path guard can stop the run
    assert go(w, "--xai-root", str(elsewhere)) == 2
    assert not (elsewhere / "stability").exists()


def test_rank_one_is_the_largest_mean_abs_shap_and_the_top_row_is_the_global_top(
    w: dict[str, Path],
) -> None:
    assert es.fold_ranks(np.array([[3.0, 1.0, 2.0]])).tolist() == [[1.0, 3.0, 2.0]]
    assert es.fold_ranks(np.array([[2.0, 2.0, 1.0]])).tolist() == [[1.5, 1.5, 3.0]]  # ties averaged
    fig, matrix, order = figure_for(w)
    shown = np.array(fig.axes[0].images[0].get_array())
    top = inputs_of(w)["glob"].sort_values("rank")["feature"].iloc[0]
    assert order[0] == top and shown[0].mean() < shown[-1].mean()  # the global top ranks near 1
    assert (
        matrix[np.arange(25), shown.argmin(axis=0)] == matrix.max(axis=1)
    ).all()  # rank 1 = largest


def test_share_at_or_above_target_counts_a_pair_exactly_at_0_70() -> None:
    pairs = pd.DataFrame({"fold_a": ["r0f0"] * 3, "fold_b": ["r0f1", "r0f2", "r0f3"],
                          "same_repeat": [True, False, False], "rho": [0.7, 0.69, 0.71]})  # fmt: skip
    m = np.tile([4.0, 3.0, 2.0, 1.0], (25, 1))
    clips = pd.DataFrame(
        {"clip_ref": ["a"], "mean_rho_over_repeats": [0.5], "n_pairs_defined": [10]}
    )
    s = es.summarise(pairs, m, ["a", "b", "c", "d"], GLOB, clips)
    assert s["share_ge_target"] == pytest.approx(2 / 3)  # 0.70 and 0.71 meet the target


def test_a_duplicated_row_with_the_right_row_count_exits_1(w: dict[str, Path]) -> None:
    path = w["in"] / tsh.FOLD_FILE
    table = pd.read_csv(path)
    table.iloc[1] = table.iloc[0]  # same (repeat, fold, feature) twice; another one is missing
    table.to_csv(path, index=False, float_format="%.17g")
    assert go(w) == 1


def test_rank_text_is_compact_for_ties() -> None:
    assert es.rank_text(3.0) == "3" and es.rank_text(13.0) == "13"
    assert es.rank_text(12.5) == "12\u00bd" and es.rank_text(1.5) == "1\u00bd"
    assert es.rank_text(2.0 + 1 / 3) == "2.3"  # a three-way tie keeps one decimal
    assert all(len(es.rank_text(r)) <= 3 for r in (1.0, 9.0, 12.0, 12.5, 13.0))


def test_a_figure_with_tied_ranks_has_no_overlapping_or_clipped_text() -> None:
    rng = np.random.default_rng(2)
    base = rng.normal(size=(25, 13))
    base[:, -3:] = 0.0  # three tied features in every fold: ranks 11, 11, 11 averaged
    base[::2, -5:-3] = 0.5  # and a pair tied in half of the folds
    order = [f"feature_number_{i}" for i in range(13)]
    ids = [(r, f) for r in range(5) for f in range(5)]
    summary = {"mean_rho": 0.9, "kendall_w": 0.9, "meets_target": True}
    fig = es.fig_ranks("ge2", "B", ids, order, np.abs(base), summary)
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert any("\u00bd" in t for t in texts)
    assert text_problems(fig) == []


def test_neighbouring_cell_numbers_keep_a_visible_gap_even_with_ties() -> None:
    rng = np.random.default_rng(4)
    base = np.abs(rng.normal(size=(25, 13)))
    base[:, -2:] = 0.0  # the two smallest features tie in every fold: rank 12.5 for both
    order = [f"feature_number_{i}" for i in range(13)]
    ids = [(r, f) for r in range(5) for f in range(5)]
    summary = {"mean_rho": 0.9, "kendall_w": 0.9, "meets_target": True}
    fig = es.fig_ranks("ge2", "B", ids, order, base, summary)
    fig.canvas.draw()
    texts = fig.axes[0].texts
    for row in range(13):
        boxes = [texts[row * 25 + j].get_window_extent() for j in range(25)]
        gaps = [boxes[j + 1].x0 - boxes[j].x1 for j in range(24)]
        assert min(gaps) >= 3, (row, min(gaps))  # at least 3 px (dpi 200) between neighbours
