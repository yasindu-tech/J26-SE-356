"""Tests for models/tapping/src/shap_figures.py (task FT-43). Synthetic xai folders only; no real
data. Hand-computed checks for the data preparation, then the drawn figures and the guards."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv_splits as cvs  # noqa: E402
import shap_figures as sf  # noqa: E402
import treeshap_heldout as tsh  # noqa: E402
from matplotlib import colormaps  # noqa: E402

COLS = list(tsh.ch.FEATURE_SETS["A"])
HASHES = {"features_sha8": "aaaaaaaa", "grid_sha8": "bbbbbbbb", "splits_sha8": "cccccccc"}
REFS = [f"{i:08x}" for i in (0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11)]
NAME = re.compile(
    r"^(global_importance|summary_dots|figures_meta|clip_(highest|lowest|nearest_half)_[0-9a-f]{8})\.(png|json)$"
)
LABEL_WORDS = re.compile(r"rating|updrs|label|group|diagnos|detect|control", re.I)


def values_table(seed: int = 0, nan_feature: str | None = "iti_cv") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_features = rng.normal(size=(len(REFS), len(COLS)))
    rows = []
    for repeat in range(5):
        for i, ref in enumerate(REFS):
            shap_v = rng.normal(scale=0.5, size=len(COLS)) + 0.3 * (i - 3.5)
            base = -1.0 + 0.05 * repeat
            raw = base + float(shap_v.sum())
            row = {"repeat": repeat, "outer_fold": i % 2, "clip_ref": ref,
                   "participant_ref": f"{0x80 + i // 2:08x}", "base_value": base, "raw_margin": raw,
                   "proba": float(1 / (1 + np.exp(-raw)))}  # fmt: skip
            for j, c in enumerate(COLS):
                row[f"shap_{c}"] = shap_v[j]
            for j, c in enumerate(COLS):
                v = base_features[i, j]
                row[f"value_{c}"] = np.nan if (c == nan_feature and i in (1, 4)) else v
            rows.append(row)
    return pd.DataFrame(rows)


def make_world(tmp: Path, monkeypatch: pytest.MonkeyPatch, table: pd.DataFrame | None = None,
               **meta_edit: Any) -> dict[str, Path]:  # fmt: skip
    proc = tmp / "data" / "processed"
    monkeypatch.setattr(cvs, "PROCESSED_ROOT", proc)
    monkeypatch.setattr(cvs, "CV_ROOT", proc / "cv")
    monkeypatch.setattr(tsh, "CONFIGS", (("ge2", "A"),))
    table = values_table() if table is None else table
    folder = proc / "xai" / "ge2" / "A"
    folder.mkdir(parents=True)
    by_fold, glob = tsh.importance_tables(table, COLS)
    table.to_csv(folder / tsh.SHAP_FILE, index=False, float_format="%.17g")
    by_fold.to_csv(folder / tsh.FOLD_FILE, index=False, float_format="%.17g")
    glob.to_csv(folder / tsh.GLOBAL_FILE, index=False, float_format="%.17g")
    meta = {"tool": tsh.TOOL, "target": "ge2", "feature_set": "A", "columns": COLS,
            "oof_reproduced": True, **HASHES,
            "row_counts": {tsh.SHAP_FILE: len(table), tsh.FOLD_FILE: len(by_fold),
                           tsh.GLOBAL_FILE: len(glob)}}  # fmt: skip
    meta.update(meta_edit)
    (folder / tsh.META_FILE).write_text(json.dumps(meta))
    (proc / "cv" / "ge2").mkdir(parents=True)
    (proc / "cv" / "ge2" / "meta_lgbm_A.json").write_text(json.dumps(HASHES))
    return {"proc": proc, "cv": proc / "cv", "xai": proc / "xai", "in": folder,
            "out": proc / "xai" / "figures" / "ge2" / "A"}  # fmt: skip


@pytest.fixture
def w(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    return make_world(tmp_path, monkeypatch)


def go(w: dict[str, Path], *extra: str) -> int:
    return sf.main(["--cv-root", str(w["cv"]), "--xai-root", str(w["xai"]), *extra])


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


# --------------------------------------------------------------------------- data preparation


def test_per_clip_mean_equals_a_hand_computed_mean() -> None:
    table = values_table()
    clips = sf.per_clip_means(table, COLS)
    assert list(clips["clip_ref"]) == sorted(REFS) and len(clips) == len(REFS)
    ref = REFS[2]
    sel = table[table["clip_ref"] == ref]
    hand = sum(sel["shap_tap_rate_hz"]) / 5
    got = clips.set_index("clip_ref").loc[ref, "shap_tap_rate_hz"]
    assert got == pytest.approx(hand, abs=1e-15)
    assert got != pytest.approx(sel["shap_tap_rate_hz"].iloc[0], abs=1e-6)  # not repeat 0 alone
    assert clips.set_index("clip_ref").loc[ref, "base_value"] == pytest.approx(-1.0 + 0.05 * 2)


def test_each_clip_has_exactly_one_dot_per_feature() -> None:
    clips = sf.per_clip_means(values_table(), COLS)
    dots = sf.dot_layout(clips, COLS)
    assert set(dots) == set(COLS)
    for c in COLS:
        assert len(dots[c]["x"]) == len(dots[c]["y"]) == len(dots[c]["rgba"]) == len(REFS)


def test_a_clip_missing_from_a_repeat_or_duplicated_is_refused() -> None:
    table = values_table()
    with pytest.raises(cvs.GuardFailure, match="every repeat"):
        sf.per_clip_means(table.iloc[1:], COLS)
    with pytest.raises(cvs.GuardFailure, match="twice"):
        sf.per_clip_means(pd.concat([table, table.iloc[[0]]], ignore_index=True), COLS)


def test_a_feature_value_changing_between_repeats_is_refused() -> None:
    table = values_table()
    table.loc[0, "value_amp_mean"] += 1.0
    with pytest.raises(cvs.GuardFailure, match="differs between the repeats"):
        sf.per_clip_means(table, COLS)


def test_nan_feature_values_stay_nan_in_the_clip_table_and_are_grey() -> None:
    clips = sf.per_clip_means(values_table(), COLS).set_index("clip_ref")
    assert clips.loc[REFS[1], "value_iti_cv"] != clips.loc[REFS[1], "value_iti_cv"]  # NaN
    assert clips["value_iti_cv"].isna().sum() == 2 and clips["value_tap_rate_hz"].isna().sum() == 0
    dots = sf.dot_layout(clips.reset_index(), COLS)["iti_cv"]
    nan = np.isnan(dots["rank"])
    assert nan.sum() == 2
    assert (dots["rgba"][nan] == np.array(sf.NAN_GREY)).all()
    assert not (dots["rank"][nan] == 0).any()  # never a rank of 0
    assert not nan[~nan].any() and len(dots["x"]) == len(REFS)  # nothing dropped


def test_percentile_ranks_by_hand() -> None:
    r = sf.percentile_ranks(np.array([3.0, 1.0, 2.0, np.nan]))
    assert r[:3].tolist() == [1.0, 0.0, 0.5] and np.isnan(r[3])
    assert sf.percentile_ranks(np.array([5.0, 5.0, 5.0, 5.0])).tolist() == [0.5] * 4
    assert sf.percentile_ranks(np.array([1.0, 1.0, 2.0])).tolist() == [0.25, 0.25, 1.0]
    assert sf.percentile_ranks(np.array([7.0])).tolist() == [0.5]
    assert np.isnan(sf.percentile_ranks(np.array([np.nan, np.nan]))).all()
    rng = np.random.default_rng(1).normal(size=50)
    ranks = sf.percentile_ranks(rng)
    assert ranks.min() == 0.0 and ranks.max() == 1.0 and ((ranks >= 0) & (ranks <= 1)).all()


def test_colour_follows_the_rank_low_dark_high_light() -> None:
    rgba = sf.dot_colours(np.array([0.0, 1.0, 0.5, np.nan]))
    cmap = colormaps[sf.CMAP]
    assert np.allclose(rgba[0], cmap(0.0)) and np.allclose(rgba[1], cmap(1.0))
    assert np.allclose(rgba[2], cmap(0.5)) and (rgba[3] == np.array(sf.NAN_GREY)).all()
    assert not np.allclose(rgba[0], rgba[1])


def test_feature_order_equals_the_global_rank_order() -> None:
    glob = pd.DataFrame({"feature": ["b", "a", "c"], "mean_abs_shap_mean_over_repeats": [1, 1, 2],
                         "rank": [2, 3, 1]})  # fmt: skip
    assert sf.feature_order(glob) == ["c", "b", "a"]
    table = values_table()
    _, g = tsh.importance_tables(table, COLS)
    order = sf.feature_order(g)
    assert list(sf.dot_layout(sf.per_clip_means(table, COLS), order)) == order


def test_jitter_is_deterministic_with_the_fixed_seed() -> None:
    assert sf.JITTER_SEED == 2026
    clips = sf.per_clip_means(values_table(), COLS)
    a, b = sf.dot_layout(clips, COLS), sf.dot_layout(clips, COLS)
    rng = np.random.default_rng(2026)
    for row, c in enumerate(COLS):
        jitter = rng.uniform(-0.3, 0.3, len(clips))
        assert np.array_equal(a[c]["y"], b[c]["y"])
        assert np.allclose(a[c]["y"], row + jitter)


def test_dot_x_is_the_mean_shap_over_repeats() -> None:
    table = values_table()
    clips = sf.per_clip_means(table, COLS)
    x = sf.dot_layout(clips, COLS)["amp_mean"]["x"]
    hand = table.groupby("clip_ref")["shap_amp_mean"].mean().sort_index().to_numpy()
    assert np.allclose(x, hand, atol=1e-15)


# --------------------------------------------------------------------------- whiskers


def test_whisker_is_min_max_of_weighted_per_repeat_means() -> None:
    rows = []
    for repeat, (a, b) in enumerate([(1.0, 3.0), (2.0, 2.0), (0.0, 4.0), (4.0, 0.0), (1.0, 1.0)]):
        rows += [{"repeat": repeat, "outer_fold": 0, "feature": "x", "mean_abs_shap": a, "n_test_rows": 10},
                 {"repeat": repeat, "outer_fold": 1, "feature": "x", "mean_abs_shap": b, "n_test_rows": 30}]  # fmt: skip
    wh = sf.whiskers(pd.DataFrame(rows), ["x"]).iloc[0]
    weighted = [(10 * a + 30 * b) / 40 for a, b in [(1, 3), (2, 2), (0, 4), (4, 0), (1, 1)]]
    assert weighted == [2.5, 2.0, 3.0, 1.0, 1.0]
    assert (wh["low"], wh["high"]) == (1.0, 3.0) and wh["mean"] == pytest.approx(1.9)
    unweighted = [(a + b) / 2 for a, b in [(1, 3), (2, 2), (0, 4), (4, 0), (1, 1)]]
    assert (min(unweighted), max(unweighted)) != (1.0, 3.0)  # the weighting matters here


def test_whiskers_from_the_pipeline_contain_the_global_value() -> None:
    table = values_table()
    by_fold, glob = tsh.importance_tables(table, COLS)
    sf.check_whiskers(glob, sf.whiskers(by_fold, COLS))  # no error


def test_inconsistent_global_importance_is_refused() -> None:
    table = values_table()
    by_fold, glob = tsh.importance_tables(table, COLS)
    glob.loc[0, "mean_abs_shap_mean_over_repeats"] += 0.5
    with pytest.raises(cvs.GuardFailure, match="disagrees"):
        sf.check_whiskers(glob, sf.whiskers(by_fold, COLS))


def test_whiskers_need_five_repeats() -> None:
    by_fold, _ = tsh.importance_tables(values_table(), COLS)
    with pytest.raises(cvs.GuardFailure, match="repeats"):
        sf.whiskers(by_fold[by_fold["repeat"] < 4], COLS)


# --------------------------------------------------------------------------- clip selection


def test_selection_by_hand_and_ties_go_to_the_smallest_ref() -> None:
    clips = pd.DataFrame({"clip_ref": ["0000000c", "0000000a", "0000000b", "0000000d", "0000000e"],
                          "proba": [0.9, 0.9, 0.1, 0.1, 0.52]})  # fmt: skip
    assert sf.select_clips(clips) == {"highest": "0000000a", "lowest": "0000000b",
                                      "nearest_half": "0000000e"}  # fmt: skip
    near = pd.DataFrame({"clip_ref": ["0000000b", "0000000a"], "proba": [0.6, 0.4]})
    assert sf.select_clips(near)["nearest_half"] == "0000000a"  # equal distance 0.1


def test_selection_ignores_label_like_columns() -> None:
    clips = sf.per_clip_means(values_table(), COLS)
    base = sf.select_clips(clips)
    noisy = clips.assign(rating=np.arange(len(clips))[::-1], group="g", y=1, label=0)
    assert sf.select_clips(noisy) == base
    mean_p = clips.set_index("clip_ref")["proba"]
    assert base["highest"] == mean_p.idxmax() and base["lowest"] == mean_p.idxmin()
    assert base["nearest_half"] == (mean_p - 0.5).abs().sort_values(kind="stable").index[0]


def test_label_columns_in_the_input_file_never_reach_the_figures(w: dict[str, Path]) -> None:
    assert go(w) == 0
    first = snapshot(w["out"])
    table = pd.read_csv(w["in"] / tsh.SHAP_FILE)
    table["rating"] = np.arange(len(table))[::-1]
    table["y"] = 1
    table.to_csv(w["in"] / tsh.SHAP_FILE, index=False, float_format="%.17g")
    assert go(w) == 0
    second = snapshot(w["out"])
    pngs = [n for n in first if n.endswith(".png")]
    assert pngs and all(first[n] == second[n] for n in pngs)
    assert "rating" not in sf.read_inputs(w["in"], COLS)["values"].columns


def test_chosen_clips_are_recorded_in_the_meta(w: dict[str, Path]) -> None:
    assert go(w) == 0
    meta = json.loads((w["out"] / sf.META_FILE).read_text())
    clips = sf.per_clip_means(sf.read_inputs(w["in"], COLS)["values"], COLS)
    assert meta["chosen_clips"] == sf.select_clips(clips)
    for kind, ref in meta["chosen_clips"].items():
        assert (w["out"] / f"clip_{kind}_{ref}.png").is_file()


# --------------------------------------------------------------------------- waterfall


def test_waterfall_base_plus_bars_equals_the_final_margin() -> None:
    clips = sf.per_clip_means(values_table(), COLS)
    for ref in REFS:
        wf = sf.waterfall(clips, ref, COLS)
        assert wf["base"] + sum(b[1] for b in wf["bars"]) == pytest.approx(wf["final"], abs=1e-9)
        magnitudes = [abs(b[1]) for b in wf["bars"]]
        assert magnitudes == sorted(magnitudes, reverse=True)
    row = clips.set_index("clip_ref").loc[REFS[3]]
    assert sf.waterfall(clips, REFS[3], COLS)["proba"] == row["proba"]


def test_waterfall_refuses_when_the_margin_does_not_add_up() -> None:
    clips = sf.per_clip_means(values_table(), COLS)
    clips.loc[0, "raw_margin"] += 1e-6
    with pytest.raises(cvs.GuardFailure, match="final margin"):
        sf.waterfall(clips, clips.loc[0, "clip_ref"], COLS)


def test_waterfall_without_the_base_value_would_not_add_up() -> None:
    clips = sf.per_clip_means(values_table(), COLS)
    wf = sf.waterfall(clips, REFS[0], COLS)
    assert abs(sum(b[1] for b in wf["bars"]) - wf["final"]) > 1e-3  # the base value is needed


def test_missing_values_are_labelled_missing_in_the_waterfall(w: dict[str, Path]) -> None:
    clips = sf.per_clip_means(values_table(), COLS)
    wf = sf.waterfall(clips, REFS[1], COLS)
    fig = sf.fig_waterfall(wf, "highest", "ge2, Set A, LightGBM")
    assert any(t == "iti_cv = missing" for t in sf.figure_texts(fig))


# --------------------------------------------------------------------------- figures and files


def test_source_uses_the_agg_canvas_and_never_pyplot() -> None:
    source = Path(sf.__file__).read_text()
    assert "pyplot" not in source and "FigureCanvasAgg" in source
    assert "from matplotlib.figure import Figure" in source


def test_pngs_exist_start_with_the_magic_bytes_and_rerun_is_identical(w: dict[str, Path]) -> None:
    assert go(w) == 0
    names = sorted(p.name for p in w["out"].iterdir())
    assert len([n for n in names if n.endswith(".png")]) == 5 and sf.META_FILE in names
    for n in names:
        if n.endswith(".png"):
            assert (w["out"] / n).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    first = snapshot(w["out"])
    assert go(w) == 0
    assert snapshot(w["out"]) == first


def test_png_metadata_has_no_software_or_timestamp(w: dict[str, Path]) -> None:
    assert go(w) == 0
    data = (w["out"] / sf.FIG_GLOBAL).read_bytes()
    assert b"Software" not in data and b"matplotlib" not in data and b"Creation Time" not in data


def test_every_figure_has_the_footer_title_and_stays_inside_the_canvas(w: dict[str, Path]) -> None:
    built = sf.build_config(w["xai"], w["cv"], "ge2", "A")
    assert len(built["figures"]) == 5
    for name, fig in built["figures"].items():
        fig.canvas.draw()
        texts = sf.figure_texts(fig)
        assert sf.FOOTER in texts and any("ge2, Set A, LightGBM" in t for t in texts)
        box = fig.bbox
        hidden = set()
        for ax in fig.axes:
            for axis, (lo, hi) in (
                (ax.xaxis, sorted(ax.get_xlim())),
                (ax.yaxis, sorted(ax.get_ylim())),
            ):
                for tick, label in zip(
                    axis.get_majorticklocs(), axis.get_majorticklabels(), strict=False
                ):
                    if not lo <= tick <= hi:
                        hidden.add(id(label))
        for artist in fig.findobj(match=lambda a: hasattr(a, "get_text")):
            if artist.get_text() and artist.get_visible():
                e = artist.get_window_extent()
                if id(artist) in hidden:  # a tick label beyond the axis limits is not drawn
                    continue
                assert e.x0 >= box.x0 - 1 and e.x1 <= box.x1 + 1, (name, artist.get_text())
                assert e.y0 >= box.y0 - 1 and e.y1 <= box.y1 + 1, (name, artist.get_text())


def test_font_sizes_are_at_least_11_points(w: dict[str, Path]) -> None:
    built = sf.build_config(w["xai"], w["cv"], "ge2", "A")
    for name, fig in built["figures"].items():
        for artist in fig.findobj(match=lambda a: hasattr(a, "get_fontsize")):
            if artist.get_text() if hasattr(artist, "get_text") else False:
                assert artist.get_fontsize() >= 11, (name, artist.get_text())


def test_caption_colourbar_and_grey_legend_text(w: dict[str, Path]) -> None:
    built = sf.build_config(w["xai"], w["cv"], "ge2", "A")
    glob_texts = " ".join(sf.figure_texts(built["figures"][sf.FIG_GLOBAL]))
    assert "range over 5 repeats, not a confidence interval" in glob_texts
    dots = built["figures"][sf.FIG_DOTS]
    dot_texts = sf.figure_texts(dots)
    assert "feature value (low to high)" in dot_texts and "grey = value missing" in dot_texts
    assert any("positive = pushes toward the higher-severity class" in t for t in dot_texts)
    grey = [
        c
        for c in dots.axes[0].collections
        if len(c.get_facecolors()) and np.allclose(c.get_facecolors()[0][:3], sf.NAN_GREY[:3])
    ]
    assert grey  # grey dots are really drawn


def test_no_grey_legend_when_nothing_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    w2 = make_world(tmp_path, monkeypatch, values_table(nan_feature=None))
    built = sf.build_config(w2["xai"], w2["cv"], "ge2", "A")
    assert "grey = value missing" not in sf.figure_texts(built["figures"][sf.FIG_DOTS])


def test_figure_text_has_no_label_wording(w: dict[str, Path]) -> None:
    built = sf.build_config(w["xai"], w["cv"], "ge2", "A")
    for name, fig in built["figures"].items():
        for text in sf.figure_texts(fig):
            assert not LABEL_WORDS.search(text), (name, text)


def test_privacy_file_names_meta_and_console(
    w: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert go(w) == 0
    for p in w["out"].iterdir():
        assert NAME.match(p.name), p.name
    meta_text = (w["out"] / sf.META_FILE).read_text()
    assert not LABEL_WORDS.search(meta_text) and "/Users" not in meta_text
    refs = set(re.findall(r"clip_(?:highest|lowest|nearest_half)_([0-9a-f]{8})", meta_text))
    assert refs <= set(REFS)
    out = capsys.readouterr().out
    assert not re.search(r"\d\.\d", out) and not LABEL_WORDS.search(out)  # names, counts, sha8 only
    assert all(len(line) < 100 for line in out.splitlines())


def test_meta_records_inputs_files_constants_and_version(w: dict[str, Path]) -> None:
    import matplotlib

    assert go(w) == 0
    meta = json.loads((w["out"] / sf.META_FILE).read_text())
    assert meta["matplotlib"] == matplotlib.__version__
    assert meta["constants"] == {"jitter_seed": 2026, "dpi": 200, "figsize": [11.5, 6.4],
                                 "cmap": "viridis", "kinds": ["highest", "lowest", "nearest_half"]}  # fmt: skip
    assert set(meta["inputs_sha8"]) == set(sf.SHAP_INPUTS)
    for n in meta["figures"]:
        assert meta["figures_sha8"][n] == cvs.sha8_of((w["out"] / n).read_bytes())


def test_sources_are_never_modified(w: dict[str, Path]) -> None:
    def snap() -> dict[str, str]:
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(w["proc"].rglob("*")) if p.is_file() and "figures" not in p.parts}  # fmt: skip

    before = snap()
    assert go(w) == 0
    assert snap() == before


def test_failed_atomic_write_keeps_the_old_file(
    w: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert go(w) == 0
    target = w["out"] / sf.FIG_GLOBAL
    target.write_bytes(b"old")

    def boom(a: Any, b: Any) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(sf.os, "replace", boom)
    with pytest.raises(OSError):
        go(w)
    assert target.read_bytes() == b"old"
    assert not [p for p in w["out"].iterdir() if p.name.startswith(".")]


# --------------------------------------------------------------------------- guards


def test_oof_not_reproduced_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    w2 = make_world(tmp_path, monkeypatch, oof_reproduced=False)
    assert go(w2) == 1
    assert "did not reproduce" in capsys.readouterr().err
    assert not w2["out"].exists()


@pytest.mark.parametrize("key", ["features_sha8", "grid_sha8", "splits_sha8"])
def test_a_hash_differing_from_the_official_meta_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    w2 = make_world(tmp_path, monkeypatch, **{key: "deadbeef"})
    assert go(w2) == 1 and not w2["out"].exists()


def test_official_meta_changed_exits_1(w: dict[str, Path]) -> None:
    (w["cv"] / "ge2" / "meta_lgbm_A.json").write_text(
        json.dumps({**HASHES, "grid_sha8": "00000000"})
    )
    assert go(w) == 1


def test_wrong_tool_target_or_row_counts_exit_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    w2 = make_world(tmp_path, monkeypatch, tool="something-else")
    assert go(w2) == 1
    path = w2["in"] / tsh.META_FILE
    meta = json.loads(path.read_text())
    meta.update(tool=tsh.TOOL, row_counts={**meta["row_counts"], tsh.SHAP_FILE: 1})
    path.write_text(json.dumps(meta))
    assert go(w2) == 1


def test_features_differing_from_the_registered_set_exit_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    w2 = make_world(tmp_path, monkeypatch, columns=COLS[::-1])
    assert go(w2) == 1


def test_missing_folder_or_files_exit_2(w: dict[str, Path]) -> None:
    (w["in"] / tsh.FOLD_FILE).unlink()
    assert go(w) == 2
    (w["in"] / tsh.FOLD_FILE).write_text("x")
    (w["cv"] / "ge2" / "meta_lgbm_A.json").unlink()
    assert go(w) == 2


def test_missing_xai_folder_exits_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    w2 = make_world(tmp_path, monkeypatch)
    import shutil

    shutil.rmtree(w2["in"])
    assert go(w2) == 2


def test_xai_root_outside_the_xai_folder_exits_2(w: dict[str, Path], tmp_path: Path) -> None:
    import shutil

    elsewhere = tmp_path / "elsewhere"
    shutil.copytree(w["xai"], elsewhere)  # complete inputs: only the path guard can stop the run
    assert sf.main(["--cv-root", str(w["cv"]), "--xai-root", str(elsewhere)]) == 2
    assert not (elsewhere / "figures").exists()


def test_the_two_configs_with_distinct_feature_sets_have_the_right_rows() -> None:
    assert sf.title_of("ge1", "B") == "ge1, Set B, LightGBM"
    assert sf.KINDS == ("highest", "lowest", "nearest_half")
    assert sf.DPI == 200 and sf.FIGSIZE[1] == 6.4 and sf.FONT_SMALL >= 11


def test_dot_colour_rank_rises_with_the_feature_value() -> None:
    clips = sf.per_clip_means(values_table(), COLS)
    d = sf.dot_layout(clips, COLS)["tap_rate_hz"]
    values = clips["value_tap_rate_hz"].to_numpy()
    assert d["rank"][int(np.argmax(values))] == 1.0 and d["rank"][int(np.argmin(values))] == 0.0
    assert np.allclose(d["rgba"][int(np.argmax(values))], colormaps[sf.CMAP](1.0))


def test_the_fixed_footer_and_caption_texts_are_exact() -> None:
    assert sf.FOOTER == (
        "SHAP = contribution to the model's score, not a cause. "
        "Illustrative clips are chosen by model score, not representative."
    )
    assert sf.KIND_TEXT == {"highest": "highest model score", "lowest": "lowest model score",
                            "nearest_half": "score nearest 0.5"}  # fmt: skip


def test_whisker_low_is_not_above_high_and_order_is_not_reversed() -> None:
    by_fold, _ = tsh.importance_tables(values_table(), COLS)
    wh = sf.whiskers(by_fold, COLS)
    assert (wh["low"] <= wh["mean"] + 1e-12).all() and (wh["mean"] <= wh["high"] + 1e-12).all()
    assert list(wh["feature"]) == COLS


# --------------------------------------------------------------------------- numbers on the figures


def expected_from(xai: Path, cv_root: Path, target: str, fs: str) -> dict[str, Any]:
    """What the drawn numbers must equal, recomputed from the saved inputs."""
    cols = list(tsh.ch.FEATURE_SETS[fs])
    inputs = sf.read_inputs(xai / target / fs, cols)
    order = sf.feature_order(inputs["glob"])
    clips = sf.per_clip_means(inputs["values"], cols)
    return {"cols": cols, "inputs": inputs, "order": order, "clips": clips,
            "wh": sf.whiskers(inputs["by_fold"], order), "chosen": sf.select_clips(clips)}  # fmt: skip


def check_global_figure(fig: Any, exp: dict[str, Any]) -> list[str]:
    ax, bad = fig.axes[0], []
    order, glob = exp["order"], exp["inputs"]["glob"].set_index("feature")
    wh = exp["wh"].set_index("feature")
    values = [float(glob.loc[f, "mean_abs_shap_mean_over_repeats"]) for f in order]
    if [t.get_text() for t in ax.get_yticklabels()] != list(order):
        bad.append("row labels are not in rank order")
    widths = [p.get_width() for p in ax.patches]
    if not np.allclose(widths, values, atol=1e-12):
        bad.append("bar lengths differ from the global importance")
    if [t.get_text() for t in ax.texts] != [f"{v:.3f}" for v in values]:
        bad.append("bar value labels differ from the global importance as rounded")
    segs = [seg for c in ax.collections for seg in c.get_segments()]
    ends = [(float(s[0][0]), float(s[1][0])) for s in segs]
    want = [(float(wh.loc[f, "low"]), float(wh.loc[f, "high"])) for f in order]
    if len(ends) != len(want) or not np.allclose(ends, want, atol=1e-12):
        bad.append("whisker ends differ from the per-repeat min and max")
    return bad


def check_waterfall_figure(fig: Any, wf: dict[str, Any], kind: str) -> list[str]:
    ax, bad = fig.axes[0], []
    texts = [t.get_text() for t in ax.texts]
    running, starts, ends = wf["base"], [0.0], [wf["base"]]
    for _, shap_v, _ in wf["bars"]:
        starts.append(running)
        running += shap_v
        ends.append(running)
    starts.append(0.0)
    ends.append(wf["final"])
    want = [sf.plain_text(wf["base"])] + [sf.delta_text(b[1]) for b in wf["bars"]]
    want.append(sf.plain_text(wf["final"]))
    if texts != want:
        bad.append("printed base, deltas or final margin differ from the waterfall numbers")
    geometry = [(p.get_x(), p.get_x() + p.get_width()) for p in ax.patches]
    expected = [(min(a, b), max(a, b)) for a, b in zip(starts, ends, strict=True)]
    if not np.allclose(geometry, expected, atol=1e-9):
        bad.append("bar geometry differs from the waterfall numbers")
    if abs(wf["base"] + sum(b[1] for b in wf["bars"]) - wf["final"]) > 1e-9:
        bad.append("base plus deltas is not the final margin")
    if (
        ax.get_title(loc="right")
        != f"mean predicted probability {sf.proba_text(wf['proba'])} (mean of 5 repeats)"
    ):
        bad.append("probability text differs")
    if f"clip {wf['ref']} ({sf.KIND_TEXT[kind]})" not in fig._suptitle.get_text():  # type: ignore[union-attr]
        bad.append("title does not carry the chosen clip ref")
    names = [t.get_text() for t in ax.get_yticklabels()]
    if names[0] != "base value" or names[-1] != "final margin" or len(names) != len(wf["bars"]) + 2:
        bad.append("row labels differ")
    return bad


def check_dots_figure(fig: Any, exp: dict[str, Any]) -> list[str]:
    ax, bad = fig.axes[0], []
    if [t.get_text() for t in ax.get_yticklabels()] != list(exp["order"]):
        bad.append("row labels are not in global rank order")
    sets = [c for c in ax.collections if len(c.get_offsets()) > 0]
    if len(sets) != len(exp["order"]):
        return bad + ["one scatter per feature expected"]
    grey_any = False
    for feature, collection in zip(exp["order"], sets, strict=True):
        n_clips = len(exp["clips"])
        if len(collection.get_offsets()) != n_clips:
            bad.append("dots per feature differ from the number of clips")
        n_nan = int(exp["clips"][f"value_{feature}"].isna().sum())
        grey = int(sum(np.allclose(c, sf.NAN_GREY) for c in collection.get_facecolors()))
        grey_any |= n_nan > 0
        if grey != n_nan:
            bad.append("grey dots differ from the number of missing values")
        x = np.sort(np.asarray(collection.get_offsets())[:, 0])
        if not np.allclose(x, np.sort(exp["clips"][f"shap_{feature}"].to_numpy()), atol=1e-12):
            bad.append("dot positions differ from the clip means")
    if ("grey = value missing" in sf.figure_texts(fig)) != grey_any:
        bad.append("grey legend does not match the presence of missing values")
    return bad


def text_problems(fig: Any) -> list[str]:
    """Drawn texts that are clipped by the canvas or overlap each other."""
    fig.canvas.draw()
    hidden: set[int] = set()
    for ax in fig.axes:
        for axis, (lo, hi) in (
            (ax.xaxis, sorted(ax.get_xlim())),
            (ax.yaxis, sorted(ax.get_ylim())),
        ):
            for tick, label in zip(
                axis.get_majorticklocs(), axis.get_majorticklabels(), strict=False
            ):
                if not lo <= tick <= hi:
                    hidden.add(id(label))
    boxes = []
    for t in fig.findobj(sf.Text):
        if t.get_text() and t.get_visible() and id(t) not in hidden:
            boxes.append((t.get_text(), t.get_window_extent()))
    bad = []
    canvas = fig.bbox
    for _, e in boxes:
        if (
            e.x0 < canvas.x0 - 1
            or e.x1 > canvas.x1 + 1
            or e.y0 < canvas.y0 - 1
            or e.y1 > canvas.y1 + 1
        ):
            bad.append("clipped by the canvas")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if a.x0 < b.x1 - 1 and b.x0 < a.x1 - 1 and a.y0 < b.y1 - 1 and b.y0 < a.y1 - 1:
                bad.append("two texts overlap")
    return bad


def named_figures(
    xai: Path, cv_root: Path, target: str, fs: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    exp = expected_from(xai, cv_root, target, fs)
    return sf.build_config(xai, cv_root, target, fs)["figures"], exp


def test_global_figure_numbers_match_the_saved_importance(w: dict[str, Path]) -> None:
    figures, exp = named_figures(w["xai"], w["cv"], "ge2", "A")
    assert check_global_figure(figures[sf.FIG_GLOBAL], exp) == []


def test_waterfall_numbers_match_the_waterfall_data_for_every_chosen_clip(
    w: dict[str, Path],
) -> None:
    figures, exp = named_figures(w["xai"], w["cv"], "ge2", "A")
    for kind, ref in exp["chosen"].items():
        wf = sf.waterfall(exp["clips"], ref, exp["cols"])
        assert check_waterfall_figure(figures[f"clip_{kind}_{ref}.png"], wf, kind) == []


def test_dots_figure_rows_counts_and_grey_match(w: dict[str, Path]) -> None:
    figures, exp = named_figures(w["xai"], w["cv"], "ge2", "A")
    assert check_dots_figure(figures[sf.FIG_DOTS], exp) == []
    assert exp["clips"]["value_iti_cv"].isna().sum() == 2  # grey dots do exist in this world


def test_dots_figure_without_missing_values_has_no_grey(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    w2 = make_world(tmp_path, monkeypatch, values_table(nan_feature=None))
    figures, exp = named_figures(w2["xai"], w2["cv"], "ge2", "A")
    assert check_dots_figure(figures[sf.FIG_DOTS], exp) == []
    assert all(exp["clips"][f"value_{c}"].isna().sum() == 0 for c in COLS)


def test_no_text_is_clipped_or_overlapping_in_any_figure(w: dict[str, Path]) -> None:
    figures, _ = named_figures(w["xai"], w["cv"], "ge2", "A")
    for name, fig in figures.items():
        assert text_problems(fig) == [], name


def test_zero_deltas_carry_no_sign_and_extreme_probabilities_are_not_rounded_away() -> None:
    assert sf.delta_text(-0.001) == "0.00" and sf.delta_text(0.004) == "0.00"
    assert sf.delta_text(0.006) == "+0.01" and sf.delta_text(-0.37) == "-0.37"
    assert sf.plain_text(-0.0001) == "0.00" and sf.plain_text(-1.2) == "-1.20"
    assert sf.proba_text(0.001) == "<0.01" and sf.proba_text(0.999) == ">0.99"
    assert sf.proba_text(0.5) == "0.50" and sf.proba_text(0.005) == "0.01"


def test_the_checks_themselves_can_fail(w: dict[str, Path]) -> None:
    figures, exp = named_figures(w["xai"], w["cv"], "ge2", "A")
    fig = figures[sf.FIG_GLOBAL]
    fig.axes[0].texts[0].set_text("9.999")
    assert check_global_figure(fig, exp)
    ref = exp["chosen"]["highest"]
    wf = sf.waterfall(exp["clips"], ref, exp["cols"])
    wf_fig = figures[f"clip_highest_{ref}.png"]
    wf_fig.axes[0].texts[1].set_text("+9.99")
    assert check_waterfall_figure(wf_fig, wf, "highest")
