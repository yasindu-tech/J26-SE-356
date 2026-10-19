"""Tests for models/tapping/src/deletion_curve.py (task FT-44). A faithfulness check of the
explanation, not evidence of cause. Hand-computed pieces first, then a synthetic world built from
the real harness (official lgbm runs and SHAP outputs); no real data is used."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv_harness as ch  # noqa: E402
import cv_splits as cvs  # noqa: E402
import deletion_curve as dc  # noqa: E402
import shap_figures as sf  # noqa: E402
import treeshap_heldout as tsh  # noqa: E402
from test_cv_harness import SMALL_LGBM, build_world, make_data  # noqa: E402
from test_shap_figures import text_problems  # noqa: E402

N_SMALL = 12
FILES = (dc.CURVES_FILE, dc.AREAS_FILE, dc.SUMMARY_FILE, dc.FIG_FILE, dc.META_FILE)
LABEL_WORDS = re.compile(r"rating|updrs|label|group|diagnos|detect|control", re.I)


@pytest.fixture(autouse=True)
def fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(ch.GRIDS, "lgbm", [dict(p) for p in SMALL_LGBM])
    monkeypatch.setattr(tsh, "CONFIGS", (("ge2", "A"), ("ge2", "B")))
    monkeypatch.setattr(dc, "CONFIGS", (("ge2", "A"), ("ge2", "B")))
    monkeypatch.setattr(dc, "N_RANDOM", N_SMALL)


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Official lgbm A and B runs and their SHAP outputs on a synthetic matrix, built once."""
    tmp = tmp_path_factory.mktemp("official")
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(ch.GRIDS, "lgbm", [dict(p) for p in SMALL_LGBM])
        mp.setattr(tsh, "CONFIGS", (("ge2", "A"), ("ge2", "B")))
        wd = build_world(tmp, mp, make_data(), repeats=5)
        for fs in ("A", "B"):
            args = ["--features", str(wd["proc"] / "features.csv"),
                    "--roles", str(wd["proc"] / "feature_roles.csv"),
                    "--meta", str(wd["proc"] / "features_meta.json")]  # fmt: skip
            assert ch.main(["run", "--model", "lgbm", "--feature-set", fs, "--repeats", "5", *args,
                            "--splits-dir", str(wd["cv"]), "--out-dir", str(wd["cv"])]) == 0  # fmt: skip
        mp.setattr(cvs, "PROCESSED_ROOT", wd["proc"])
        assert tsh.main(["--cv-root", str(wd["cv"]), *args]) == 0
    return {"tmp": tmp}


@pytest.fixture
def w(base: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    shutil.copytree(base["tmp"] / "data", tmp_path / "data")
    proc = tmp_path / "data" / "processed"
    monkeypatch.setattr(cvs, "CV_ROOT", proc / "cv")
    monkeypatch.setattr(cvs, "PROCESSED_ROOT", proc)
    return {"proc": proc, "cv": proc / "cv", "xai": proc / "xai", "out": proc / "xai" / "deletion"}


def go(w: dict[str, Path], *extra: str) -> int:
    return dc.main(["--cv-root", str(w["cv"]), "--features", str(w["proc"] / "features.csv"),
                    "--roles", str(w["proc"] / "feature_roles.csv"),
                    "--meta", str(w["proc"] / "features_meta.json"), *extra])  # fmt: skip


def out(w: dict[str, Path], fs: str = "A") -> Path:
    return w["out"] / "ge2" / fs


def pieces(w: dict[str, Path], fs: str = "A") -> dict[str, Any]:
    frame = cvs.canonical(
        pd.read_csv(w["proc"] / "features.csv", dtype={"ref": str, "participant": str})
    )
    splits = cvs.load_splits(w["cv"] / "ge2" / "splits.csv", frame)
    cols = list(ch.FEATURE_SETS[fs])
    values = sf.read_inputs(w["xai"] / "ge2" / fs, cols)["values"]
    runs = pd.read_csv(w["cv"] / "ge2" / f"runs_lgbm_{fs}.csv").sort_values("repeat")
    return {
        "frame": frame, "splits": splits, "cols": cols, "values": values,
        "oof": ch.read_oof(w["cv"] / "ge2" / f"oof_lgbm_{fs}.csv"),
        "best": tsh.read_best_params(w["cv"] / "ge2" / f"best_params_lgbm_{fs}.csv", ch.grid_for("lgbm")),
        "runs_auroc": runs["auroc"].to_numpy(dtype=float),
        "abs_shap": dc.shap_abs_by_repeat(values, frame, cols, splits),
    }  # fmt: skip


def compute(p: dict[str, Any], fs: str = "A", n_random: int = N_SMALL) -> dict[str, Any]:
    return dc.compute_curves(p["frame"], p["splits"], p["oof"], p["best"], p["abs_shap"], "ge2", fs,
                             0, p["runs_auroc"], n_random)  # fmt: skip


# --------------------------------------------------------------------------- constants


def test_preregistered_constants() -> None:
    source = Path(dc.__file__).read_text()
    assert 'CONFIGS = (("ge2", "A"), ("ge1", "A"), ("ge2", "B"))' in source
    assert (
        "N_RANDOM = 100" in source and "RANDOM_SEED = 2026" in source and "ALPHA = 0.05" in source
    )
    assert dc.OOF_TOL == 1e-9 and dc.K0_TOL == 1e-9
    assert "pyplot" not in source and "FigureCanvasAgg" in source
    assert dc.FOOTER == (
        "SHAP order = per-clip |SHAP| descending; deleted value = training-fold median. "
        "Descriptive; not a cause."
    )


# --------------------------------------------------------------------------- orders


def test_shap_order_descending_by_abs_with_name_tie_break() -> None:
    abs_shap = np.array([[0.1, 0.5, 0.5], [0.3, 0.3, 0.3], [0.9, 0.0, 0.2]])
    order = dc.shap_order(abs_shap, ["b", "a", "c"])
    assert order.tolist() == [[1, 2, 0], [1, 0, 2], [0, 2, 1]]  # ties: a before b before c


def test_reverse_is_the_exact_reverse() -> None:
    order = dc.shap_order(np.array([[0.1, 0.5, 0.5, 0.2]]), ["d", "a", "c", "b"])
    assert dc.reverse_order(order).tolist() == order[:, ::-1].tolist()
    assert sorted(dc.reverse_order(order)[0].tolist()) == [0, 1, 2, 3]


def test_random_orders_are_reproducible_valid_and_differ_across_draws() -> None:
    a, b = dc.random_orders(1, 3, 5, 7, 6), dc.random_orders(1, 3, 5, 7, 6)
    assert a.shape == (5, 7, 6) and np.array_equal(a, b)
    assert (np.sort(a, axis=2) == np.arange(6)).all()  # a permutation per (repeat, clip)
    assert not np.array_equal(a, dc.random_orders(1, 4, 5, 7, 6))  # another draw
    assert not np.array_equal(a, dc.random_orders(2, 3, 5, 7, 6))  # another config
    assert len({tuple(r) for r in a.reshape(-1, 6)}) > 5  # not one permutation repeated
    rng = np.random.default_rng([2026, 1, 3])
    assert np.array_equal(a, np.argsort(rng.random((5, 7, 6)), axis=2))  # seed [2026, config, draw]


# --------------------------------------------------------------------------- deleting


def test_training_median_is_nan_aware_and_never_the_test_statistic() -> None:
    x_train = np.array([[1.0, 10.0], [2.0, np.nan], [9.0, 30.0], [100.0, 50.0]])
    assert dc.train_medians(x_train).tolist() == [5.5, 30.0]
    x_test = np.array([[1000.0, -5.0], [2000.0, -6.0]])
    assert dc.train_medians(x_train).tolist() != np.nanmedian(x_test, axis=0).tolist()
    with pytest.raises(cvs.GuardFailure, match="no median"):
        dc.train_medians(np.array([[1.0, np.nan], [2.0, np.nan]]))


def test_masked_matrices_k0_unmasked_kF_all_medians_and_exactly_k_replaced() -> None:
    x = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0]])
    med = np.array([-1.0, -2.0, -3.0])
    order = np.array([[2, 0, 1], [1, 2, 0]])
    m = dc.masked(x, med, order)
    assert m.shape == (4, 2, 3)
    assert np.array_equal(m[0], x, equal_nan=True)  # k = 0: the unmasked rows
    assert (m[3] == med).all()  # k = F: every feature replaced, NaN included
    assert m[1].tolist() == [[1.0, 2.0, -3.0], [4.0, -2.0, 6.0]]  # first of each row's order
    assert m[2].tolist() == [[-1.0, 2.0, -3.0], [4.0, -2.0, -3.0]]
    for k in range(4):
        changed = ~((m[k] == x) | (np.isnan(m[k]) & np.isnan(x)))
        assert (changed.sum(axis=1) <= k).all()  # never more than k replaced
    assert not np.isnan(m[3]).any() and (m[3] != 0).all()  # NaN deleted to the median, not 0


def test_delete_masks_are_nested_in_k() -> None:
    order = np.array([[2, 0, 1, 3]])
    masks = dc.delete_masks(order, 4)[0]
    assert masks.sum(axis=1).tolist() == [0, 1, 2, 3, 4]
    assert all((masks[j] <= masks[j + 1]).all() for j in range(4))


# --------------------------------------------------------------------------- area and p


def test_area_by_hand() -> None:
    assert dc.curve_area(np.array([1.0, 0.5, 0.5])) == pytest.approx(0.625)  # (0.75 + 0.5) / 2
    assert dc.curve_area(np.array([0.8, 0.8, 0.8, 0.8])) == pytest.approx(0.8)
    assert dc.curve_area(np.array([1.0, 0.0])) == pytest.approx(0.5)
    assert dc.curve_area(np.array([0.9, 0.7, 0.5, 0.5, 0.5])) == pytest.approx(
        (0.8 + 0.6 + 0.5 + 0.5) / 4
    )


def test_p_value_formula_by_hand_with_ties() -> None:
    areas = np.array([1.0, 2.0, 3.0, 4.0])
    assert dc.empirical_p(2.0, areas) == pytest.approx((1 + 2) / 5)  # ties count: <=
    assert dc.empirical_p(0.5, areas) == pytest.approx(1 / 5)  # the +1: never zero
    assert dc.empirical_p(9.0, areas) == pytest.approx(5 / 5)
    assert dc.empirical_p(-1.0, np.arange(100.0)) == pytest.approx(1 / 101)
    assert dc.empirical_p(2.0, areas) != pytest.approx(1 / 5)  # strict < would give 1 / 5


def fake_result(
    shap_curve: np.ndarray, spread: float, n_random: int = 100, f: int = 4
) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    base = np.linspace(0.85, 0.5, f + 1)
    random = base[None, None, :] + rng.normal(0, spread, (n_random, 5, f + 1))
    random[..., 0] = base[0]
    one = np.tile(shap_curve, (5, 1))
    return {"auc": {"shap": one, "reverse": np.tile(base[::-1], (5, 1)), "random": random},
            "change": {"shap": one * 0, "reverse": one * 0, "random": random * 0}, "n_features": f}  # fmt: skip


def test_falls_faster_true_on_a_signal_case_and_false_on_noise() -> None:
    signal = dc.summarise(fake_result(np.array([0.85, 0.6, 0.5, 0.5, 0.5]), 0.01))
    assert signal["summary"]["falls_faster"] is True
    assert signal["summary"]["p_value"] == pytest.approx(1 / 101)
    noise = dc.summarise(fake_result(np.linspace(0.85, 0.5, 5), 0.01))  # same as the random mean
    assert noise["summary"]["falls_faster"] is False and noise["summary"]["p_value"] > 0.05
    slower = dc.summarise(fake_result(np.array([0.85, 0.84, 0.83, 0.8, 0.5]), 0.01))
    assert slower["summary"]["falls_faster"] is False  # a curve above the random band


def test_summary_areas_band_and_curves_by_hand() -> None:
    res = fake_result(np.array([0.85, 0.6, 0.5, 0.5, 0.5]), 0.02, n_random=40)
    sm = dc.summarise(res)
    curves = res["auc"]["random"].mean(axis=1)
    areas = np.array([dc.curve_area(c) for c in curves])
    assert np.allclose(sm["areas"], areas)
    s = sm["summary"]
    assert s["area_random_mean"] == pytest.approx(areas.mean())
    assert s["area_random_p05"] == pytest.approx(np.percentile(areas, 5))
    assert s["area_random_p95"] == pytest.approx(np.percentile(areas, 95))
    assert np.allclose(sm["curves"]["band_low"], np.percentile(curves, 5, axis=0))
    assert np.allclose(sm["curves"]["band_high"], np.percentile(curves, 95, axis=0))
    assert s["n_random"] == 40 and s["area_shap"] == pytest.approx(
        dc.curve_area(np.array([0.85, 0.6, 0.5, 0.5, 0.5]))
    )


# --------------------------------------------------------------------------- labels


def test_orders_and_medians_do_not_take_labels() -> None:
    for fn in (dc.shap_order, dc.train_medians, dc.masked, dc.delete_masks, dc.random_orders):
        assert not [
            n for n in inspect.signature(fn).parameters if re.search(r"label|^y$|rating|target", n)
        ]


def test_flipping_labels_changes_neither_orders_nor_medians(w: dict[str, Path]) -> None:
    p = pieces(w)
    flipped = p["frame"].copy()
    flipped["rating"] = 3 - flipped["rating"]
    abs_flipped = dc.shap_abs_by_repeat(p["values"], flipped, p["cols"], p["splits"])
    assert np.array_equal(abs_flipped, p["abs_shap"])
    for r in range(5):
        assert np.array_equal(
            dc.shap_order(abs_flipped[r], p["cols"]), dc.shap_order(p["abs_shap"][r], p["cols"])
        )
    X, _ = ch.design_matrix(p["frame"], "B")
    Xf, _ = ch.design_matrix(flipped, "B")
    member = ch.membership(p["frame"]["participant"].to_numpy(), p["splits"], 0)
    assert np.array_equal(dc.train_medians(X[member != 0]), dc.train_medians(Xf[member != 0]))


# --------------------------------------------------------------------------- the curves


def test_k0_equals_the_official_auroc_and_every_clip_is_predicted_once(w: dict[str, Path]) -> None:
    p = pieces(w)
    res = compute(p)
    assert dc.check_k0(res["auc"], res["runs_auroc"]) <= 1e-9
    for key in ("shap", "reverse"):
        assert np.allclose(res["auc"][key][:, 0], p["runs_auroc"], atol=1e-9)
    assert np.allclose(res["auc"]["random"][..., 0], p["runs_auroc"], atol=1e-9)
    assert res["max_oof"] <= 1e-9
    assert not np.isnan(res["proba"]["shap"]).any() and not np.isnan(res["proba"]["random"]).any()
    assert res["proba"]["random"].shape == (N_SMALL, 5, 6, len(p["frame"]))


def test_k0_predictions_are_the_official_oof_and_kF_is_constant_per_fold(
    w: dict[str, Path],
) -> None:
    p = pieces(w)
    res = compute(p)
    refs, groups = p["frame"]["ref"].to_numpy(), p["frame"]["participant"].to_numpy()
    for r in range(5):
        official = p["oof"][p["oof"]["repeat"] == r].set_index("ref").loc[refs, "proba"].to_numpy()
        member = ch.membership(groups, p["splits"], r)
        for _key, arr in (("shap", res["proba"]["shap"][r]), ("reverse", res["proba"]["reverse"][r]),
                         ("random", res["proba"]["random"][0, r])):  # fmt: skip
            assert np.allclose(arr[0], official, atol=1e-9, rtol=0)  # k = 0 is the unmasked model
            for fold in range(5):
                assert np.ptp(arr[-1][member == fold]) < 1e-12  # k = F: one constant per fold
            assert np.abs(arr[1] - arr[0]).max() > 0  # deleting one feature changes some scores


def test_the_training_fold_median_is_what_gets_deleted(
    w: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    p = pieces(w, "B")
    seen: list[np.ndarray] = []
    real = dc.train_medians

    def record_medians(x: np.ndarray) -> np.ndarray:
        seen.append(np.array(x))
        return real(x)

    monkeypatch.setattr(dc, "train_medians", record_medians)
    compute(p, "B", n_random=2)
    X, _ = ch.design_matrix(p["frame"], "B")
    groups = p["frame"]["participant"].to_numpy()
    assert len(seen) == 25
    expected = []
    for r in range(5):
        member = ch.membership(groups, p["splits"], r)
        expected += [X[member != f] for f in range(5)]
    for got, want in zip(seen, expected, strict=True):
        assert np.array_equal(got, want, equal_nan=True)  # training rows only, never test rows


def test_nan_test_values_become_the_median_and_never_zero(
    w: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    p = pieces(w, "B")
    batches: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    real = dc.masked

    def spy(x: np.ndarray, med: np.ndarray, order: np.ndarray) -> np.ndarray:
        out_ = real(x, med, order)
        batches.append((x, med, out_))
        return out_

    monkeypatch.setattr(dc, "masked", spy)
    compute(p, "B", n_random=1)
    assert any(np.isnan(x).any() for x, _, _ in batches)
    for x, med, m in batches:
        assert np.array_equal(m[0], x, equal_nan=True)  # k = 0: NaN stays NaN (native handling)
        assert not np.isnan(m[-1]).any() and (m[-1] == med).all()  # k = F: the median, never NaN
        nan_cells = np.isnan(x)
        if nan_cells.any():
            assert (
                np.isnan(med).sum() == 0
                and (m[-1][nan_cells] == np.broadcast_to(med, x.shape)[nan_cells]).all()
            )
            assert not np.isnan(m[-1][nan_cells]).any()


# --------------------------------------------------------------------------- run and outputs


def test_outputs_have_the_documented_shape(w: dict[str, Path]) -> None:
    assert go(w) == 0
    for fs, f in (("A", 5), ("B", 13)):
        curves = pd.read_csv(out(w, fs) / dc.CURVES_FILE)
        assert list(curves.columns) == dc.CURVE_COLUMNS
        assert set(curves["order"]) == {"shap", "reverse", "random_mean"}
        assert len(curves) == 3 * 6 * (f + 1)
        mean = curves[curves["repeat"] == "mean"]
        assert sorted(mean["k"].unique()) == list(range(f + 1))
        band = curves[curves["auroc_p05"].notna()]
        assert set(band["order"]) == {"random_mean"} and set(band["repeat"]) == {"mean"}
        assert (band["auroc_p05"] <= band["auroc_p95"] + 1e-12).all()
        areas = pd.read_csv(out(w, fs) / dc.AREAS_FILE)
        assert list(areas.columns) == ["draw", "area"] and len(areas) == N_SMALL
        summary = pd.read_csv(out(w, fs) / dc.SUMMARY_FILE)
        assert list(summary.columns) == dc.SUMMARY_COLUMNS and len(summary) == 1
        s = summary.iloc[0]
        assert s["n_random"] == N_SMALL
        assert s["p_value"] == pytest.approx(
            dc.empirical_p(s["area_shap"], areas["area"].to_numpy())
        )
        assert bool(s["falls_faster"]) == bool(s["p_value"] <= 0.05)
        shap_mean = mean[mean["order"] == "shap"].sort_values("k")["auroc"].to_numpy()
        assert s["area_shap"] == pytest.approx(dc.curve_area(shap_mean))


def test_k0_rows_of_the_curves_equal_the_official_runs(w: dict[str, Path]) -> None:
    assert go(w) == 0
    curves = pd.read_csv(out(w) / dc.CURVES_FILE)
    runs = pd.read_csv(w["cv"] / "ge2" / "runs_lgbm_A.csv").sort_values("repeat")
    for order in ("shap", "reverse", "random_mean"):
        k0 = curves[(curves["order"] == order) & (curves["k"] == 0) & (curves["repeat"] != "mean")]
        k0 = k0.assign(repeat=k0["repeat"].astype(int)).sort_values("repeat")
        assert np.allclose(k0["auroc"].to_numpy(), runs["auroc"].to_numpy(), atol=1e-9)
        assert (
            curves[(curves["order"] == order) & (curves["k"] == 0)]["mean_abs_margin_change"] == 0
        ).all()


def test_meta_records_guards_constants_versions_and_hashes(w: dict[str, Path]) -> None:
    assert go(w) == 0
    meta = json.loads((out(w) / dc.META_FILE).read_text())
    assert (
        meta["guards"]["k0_max_diff_vs_runs"] <= 1e-9
        and meta["guards"]["max_abs_diff_vs_official_oof"] <= 1e-9
    )
    assert set(meta["guards"]["k_full_auroc_mean"]) == {"shap", "reverse", "random"}
    assert meta["constants"]["n_random"] == N_SMALL and meta["constants"]["random_seed"] == 2026
    assert set(meta["versions"]) == {"lightgbm", "matplotlib", "numpy"}
    official = json.loads((w["cv"] / "ge2" / "meta_lgbm_A.json").read_text())
    assert (
        meta["features_sha8"] == official["features_sha8"]
        and meta["grid_sha8"] == official["grid_sha8"]
    )
    for name, h in meta["files_sha8"].items():
        assert cvs.sha8_of((out(w) / name).read_bytes()) == h


def test_png_magic_text_inside_canvas_font_and_footer(w: dict[str, Path]) -> None:
    assert go(w) == 0
    assert (out(w) / dc.FIG_FILE).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    p = pieces(w)
    sm = dc.summarise(compute(p, n_random=4))
    fig = dc.fig_curve("ge2", "A", sm)
    assert text_problems(fig) == []
    texts = sf.figure_texts(fig)
    assert dc.FOOTER in texts
    assert any(
        t == "ge2, Set A, LightGBM: deleting features in SHAP order vs random order" for t in texts
    )
    assert not [t for t in texts if LABEL_WORDS.search(t)]
    for artist in fig.findobj(sf.Text):
        if artist.get_text():
            assert float(artist.get_fontsize()) >= 11
    assert tuple(fig.get_size_inches()) == sf.FIGSIZE and fig.dpi == 200


def test_outputs_are_deterministic_and_private(
    w: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert go(w) == 0
    first = {(fs, n): (out(w, fs) / n).read_bytes() for fs in ("A", "B") for n in FILES}
    console = capsys.readouterr().out
    assert go(w) == 0
    assert {(fs, n): (out(w, fs) / n).read_bytes() for fs in ("A", "B") for n in FILES} == first
    assert capsys.readouterr().out == console
    frame = pieces(w)["frame"]
    text = console + "".join(
        v.decode(errors="ignore") for (_, n), v in first.items() if not n.endswith(".png")
    )
    assert not LABEL_WORDS.search(text) and not re.search(
        r"CONTROL|ID\d|DCHA|IZDA|parquet|/Users", text
    )
    assert not any(ref in text for ref in frame["ref"]) and not any(
        g in text for g in frame["participant"]
    )
    assert not re.search(r"20\d\d-\d\d-\d\dT|\d\d:\d\d:\d\d", text)
    assert all(len(line) < 140 for line in console.splitlines())


def test_sources_are_never_modified(w: dict[str, Path]) -> None:
    def snap() -> dict[str, str]:
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(w["proc"].rglob("*")) if p.is_file() and "deletion" not in p.parts}  # fmt: skip

    before = snap()
    assert go(w) == 0
    assert snap() == before


def test_failed_atomic_write_keeps_the_old_file(
    w: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert go(w) == 0
    (out(w) / dc.CURVES_FILE).write_bytes(b"old")
    real = cvs.os.replace

    def fail_csv(src: Any, dst: Any) -> None:  # the PNG is written first and must not mask the CSV
        if str(dst).endswith(".csv"):
            raise OSError("fail")
        real(src, dst)

    monkeypatch.setattr(cvs.os, "replace", fail_csv)
    with pytest.raises(OSError):
        go(w)
    assert (out(w) / dc.CURVES_FILE).read_bytes() == b"old"
    assert not [p for p in out(w).iterdir() if p.name.startswith(".")]


# --------------------------------------------------------------------------- guards


def test_k0_auroc_differing_from_the_runs_file_exits_1(
    w: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    path = w["cv"] / "ge2" / "runs_lgbm_A.csv"
    runs = pd.read_csv(path)
    runs.loc[0, "auroc"] += 1e-6
    runs.to_csv(path, index=False, float_format="%.17g")
    assert go(w) == 1
    assert "k=0" in capsys.readouterr().err
    assert not (out(w) / dc.CURVES_FILE).exists()


def test_k0_guard_function_tolerance() -> None:
    auc = {"shap": np.array([[0.8, 0.5]] * 5), "random": np.full((3, 5, 2), 0.8)}
    assert dc.check_k0(auc, [0.8] * 5) == 0.0
    assert dc.check_k0(auc, [0.8 + 5e-10] * 5) == pytest.approx(5e-10)
    with pytest.raises(cvs.GuardFailure, match="k=0"):
        dc.check_k0(auc, [0.8 + 2e-9] * 5)


def test_oof_not_reproduced_by_the_refit_exits_1(
    w: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    path = w["cv"] / "ge2" / "oof_lgbm_A.csv"
    oof = ch.read_oof(path)
    oof.loc[7, "proba"] += 1e-6
    oof.to_csv(path, index=False, float_format="%.17g")
    assert go(w) == 1
    assert "does not reproduce" in capsys.readouterr().err


def test_shap_meta_oof_not_reproduced_exits_1(w: dict[str, Path]) -> None:
    path = w["xai"] / "ge2" / "A" / tsh.META_FILE
    meta = json.loads(path.read_text())
    meta["oof_reproduced"] = False
    path.write_text(json.dumps(meta))
    assert go(w) == 1


@pytest.mark.parametrize("key", ["features_sha8", "grid_sha8", "splits_sha8"])
def test_shap_meta_hash_differing_from_the_official_run_exits_1(
    w: dict[str, Path], key: str
) -> None:
    path = w["xai"] / "ge2" / "A" / tsh.META_FILE
    meta = json.loads(path.read_text())
    meta[key] = "deadbeef"
    path.write_text(json.dumps(meta))
    assert go(w) == 1


def test_official_meta_version_or_grid_mismatch_exits_1(w: dict[str, Path]) -> None:
    path = w["cv"] / "ge2" / "meta_lgbm_A.json"
    meta = json.loads(path.read_text())
    meta["versions"]["lightgbm"] = "0.0.1"
    path.write_text(json.dumps(meta))
    assert go(w) == 1


def test_train_test_overlap_exits_1(
    w: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(*a: Any, **k: Any) -> None:
        raise cvs.LeakageError("overlap")

    monkeypatch.setattr(cvs, "assert_disjoint", boom)
    assert go(w) == 1
    assert "overlap" in capsys.readouterr().err


def test_test_positions_are_passed_to_refit_outer(
    w: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []
    real = ch.refit_outer

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs.get("test_positions"))
        return real(*args, **kwargs)

    monkeypatch.setattr(ch, "refit_outer", spy)
    compute(pieces(w), n_random=2)
    assert len(seen) == 25 and all(s is not None and len(s) > 0 for s in seen)


def test_shap_rows_from_other_folds_are_refused(w: dict[str, Path]) -> None:
    path = w["xai"] / "ge2" / "A" / tsh.SHAP_FILE
    table = pd.read_csv(path)
    table.loc[0, "outer_fold"] = (table.loc[0, "outer_fold"] + 1) % 5
    table.to_csv(path, index=False, float_format="%.17g")
    assert go(w) == 1


def test_a_clip_not_evaluated_exactly_once_is_refused(
    w: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    p = pieces(w)
    groups = p["frame"]["participant"].to_numpy()
    real = ch.membership

    def odd(g: np.ndarray, splits: pd.DataFrame, repeat: int) -> np.ndarray:
        member = real(g, splits, repeat).copy()
        if repeat == 0:
            member[groups == groups[0]] = 5
        return member

    monkeypatch.setattr(ch, "membership", odd)
    oof = p["oof"].copy()
    sel = oof["repeat"] == 0
    oof.loc[sel, "outer_fold"] = oof.loc[sel, "ref"].map(
        dict(zip(p["frame"]["ref"], odd(groups, p["splits"], 0), strict=True))
    )
    with pytest.raises(cvs.GuardFailure, match="exactly once"):
        dc.compute_curves(
            p["frame"],
            p["splits"],
            oof,
            p["best"],
            p["abs_shap"],
            "ge2",
            "A",
            0,
            p["runs_auroc"],
            2,
        )


def test_missing_inputs_exit_2(w: dict[str, Path]) -> None:
    (w["cv"] / "ge2" / "oof_lgbm_B.csv").unlink()
    assert go(w) == 2


def test_missing_shap_outputs_exit_2(w: dict[str, Path]) -> None:
    (w["xai"] / "ge2" / "A" / tsh.GLOBAL_FILE).unlink()
    assert go(w) == 2


def test_xai_root_outside_the_xai_folder_exits_2(w: dict[str, Path], tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    shutil.copytree(w["xai"], elsewhere)  # complete inputs: only the path guard can stop the run
    assert go(w, "--xai-root", str(elsewhere)) == 2
    assert not (elsewhere / "deletion").exists()


def test_the_shap_order_is_built_from_the_saved_abs_shap_of_each_repeat(
    w: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    p = pieces(w)
    calls: list[np.ndarray] = []
    real = dc.shap_order

    def record_order(a: np.ndarray, names: Any) -> np.ndarray:
        calls.append(np.array(a))
        return real(a, names)

    monkeypatch.setattr(dc, "shap_order", record_order)
    compute(p, n_random=1)
    assert len(calls) == 5
    for r, got in enumerate(calls):
        assert np.array_equal(got, p["abs_shap"][r])  # |SHAP| only: no label enters the order


def test_the_decision_rule_boundary_p_equal_to_0_05_counts() -> None:
    res = fake_result(np.array([0.85, 0.3, 0.3, 0.3, 0.3]), 0.01, n_random=19)
    s = dc.summarise(res)["summary"]
    assert s["p_value"] == pytest.approx(1 / 20) and s["falls_faster"] is True  # p <= 0.05
    res = fake_result(np.array([0.85, 0.3, 0.3, 0.3, 0.3]), 0.01, n_random=18)
    assert dc.summarise(res)["summary"]["p_value"] == pytest.approx(1 / 19)


def test_margin_change_is_relative_to_the_unmasked_margin(w: dict[str, Path]) -> None:
    res = compute(pieces(w), n_random=2)
    for key in ("shap", "reverse", "random"):
        assert (res["change"][key][..., 0] == 0).all()  # k = 0: nothing changed
        assert (res["change"][key][..., 1:] > 0).all()  # every deletion moves the scores
