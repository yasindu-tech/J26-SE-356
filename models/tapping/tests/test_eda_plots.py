"""Tests for models/tapping/src/eda_plots.py (task FT-31). Synthetic matrix of 60 clips whose ref
and participant values are poison strings; Agg figures only; the label file is a decoy that must
never be opened."""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import eda_plots as eda  # noqa: E402
import feature_matrix as fm  # noqa: E402

BLOCK = ["n_valid_taps", *eda.SET_B, *[f + "_sens" for f in eda.SET_B], "nan_reasons_b"]
COLUMNS = ["ref", "participant", "hand", "rating", "tap_rate_hz", "nan_reasons_a", *BLOCK]
POISON = ("POISONREF", "POISONPART", "CONTROL", "DCHA", "IZDA", "fis_diagnostic")
FILE_NAMES = sorted([*(f + ".png" for f in eda.SET_B), eda.OUT_SUMMARY_PNG, eda.OUT_SUMMARY_CSV,
                     eda.OUT_META])  # fmt: skip
N_CLIPS = 60


def make_frame(seed: int = 3) -> pd.DataFrame:
    """60 clips from 40 participants: 20 with two clips (left and right), 20 with one."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(40):
        hands = ["right", "left"] if pid < 20 else [["right", "left"][pid % 2]]
        level = int(rng.integers(0, 4))
        for hand in hands:
            rows.append((pid, hand, int(np.clip(level + rng.integers(-1, 2), 0, 3))))
    rows = rows[:N_CLIPS]
    n = len(rows)
    ratings = [r[2] for r in rows]
    for need in range(4):  # every rating level present
        if need not in ratings:
            ratings[need] = need
    present = rng.random(n) < 0.4
    cols: dict[str, list[str]] = {
        "ref": [f"POISONREF{i:03d}" for i in range(n)],
        "participant": [f"POISONPART{r[0]:03d}" for r in rows],
        "hand": [r[1] for r in rows],
        "rating": [str(v) for v in ratings],
    }
    num = lambda x: repr(float(x))  # noqa: E731
    for c in COLUMNS[4:]:
        base = c.removesuffix("_sens")
        if base == "decrement_present":
            col = ["True" if p else "False" for p in present]
        elif base == "decrement_onset_frac":
            col = [num(rng.random()) if p else "" for p in present]
        elif base in ("n_hesitations", "n_halts"):
            col = [num(rng.choice([0, 0, 0, 1, 2])) for _ in range(n)]
            col[5] = ""
        elif base in ("longest_pause_ratio", "longest_pause_s"):
            col = [num(1 + rng.random() * 3) for _ in range(n)]
            col[5] = ""
        elif base == "amp_late_early":
            col = [num(0.5 + rng.random()) for _ in range(n)]
            col[2] = col[7] = ""
        elif base == "iti_slope":
            col = [num(rng.normal(0, 0.1)) if i % 3 else num(0.0004) for i in range(n)]
        elif c in ("nan_reasons_a", "nan_reasons_b"):
            col = [""] * n
        else:
            col = [num(rng.normal()) for _ in range(n)]
        cols[c] = col
    return pd.DataFrame({c: cols[c] for c in COLUMNS})


def write_world(root: Path, frame: pd.DataFrame) -> Path:
    """features.csv, feature_roles.csv and features_meta.json under root/processed."""
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    frame.to_csv(processed / "features.csv", index=False)
    roles = pd.DataFrame(
        {
            "column": list(frame.columns),
            "role": [fm.ROLES.get(c, ("audit_count", "x"))[0] for c in frame.columns],
            "note": [fm.ROLES.get(c, ("audit_count", "x"))[1] for c in frame.columns],
        }
    )
    roles.to_csv(processed / "feature_roles.csv", index=False)
    (processed / "features_meta.json").write_text(json.dumps({"roles_sha8": eda.roles_sha8(roles)}))
    return processed


def run(processed: Path, out: Path | None = None, *extra: str) -> int:
    return eda.main(["--features", str(processed / "features.csv"),
                     "--out-dir", str(out or processed / "eda"), "--boot", "60", *extra])  # fmt: skip


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(eda, "PROCESSED_ROOT", tmp_path / "processed")
    return write_world(tmp_path, make_frame())


@pytest.fixture(scope="module")
def ran(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """One real run (twice) into two folders; read-only for the tests that use it."""
    root = tmp_path_factory.mktemp("eda")
    mp = pytest.MonkeyPatch()
    mp.setattr(eda, "PROCESSED_ROOT", root / "processed")
    processed = write_world(root, make_frame())
    assert run(processed, processed / "eda_a") == 0
    assert run(processed, processed / "eda_b") == 0
    yield {"root": root, "processed": processed, "a": processed / "eda_a", "b": processed / "eda_b"}
    mp.undo()


def data_and_roles(frame: pd.DataFrame) -> tuple[eda.Data, dict[str, tuple[str, str]]]:
    eda.validate(frame)
    roles = {c: fm.ROLES.get(c, ("audit_count", "x")) for c in frame.columns}
    return eda.Data(frame), roles


def figures(frame: pd.DataFrame, seed: int = 2026) -> dict[str, Any]:
    d, roles = data_and_roles(frame)
    draws = eda.cluster_draws(d.participant, 30, np.random.default_rng(1))
    out = {}
    for f in eda.SET_B:
        kind = eda.KIND.get(f, "continuous")
        res: dict[str, Any] = {"rho": eda.rho_ci(d.values[f], d.rating, draws)}
        if kind in ("binary", "count"):
            x = d.values[f]
            ind = x if kind == "binary" else np.where(np.isfinite(x), (x > 0).astype(float), np.nan)
            res["props"] = eda.proportion_by_rating(
                ind, d.rating, None if f == "n_halts" else draws
            )
        out[f] = eda.feature_figure(f, d, roles, res, seed)[0]
    return out


def strip_points(fig: Any) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for ax in fig.axes:
        for c in ax.collections:
            if c.get_gid() == "strip":
                pts += [tuple(p) for p in c.get_offsets()]
    return pts


def texts(fig: Any) -> list[str]:
    return [t.get_text() for t in fig.findobj(Text)]


# ----------------------------------------------------------------------- statistics by hand


def test_spearman_with_ties_by_hand() -> None:
    # ranks of x = [1, 2.5, 2.5, 4], of y = [1, 2, 3.5, 3.5]; cov 3.75, both variances 4.5
    assert eda.spearman(np.array([1.0, 2, 2, 3]), np.array([1.0, 2, 3, 3])) == pytest.approx(
        3.75 / 4.5
    )
    assert list(eda.average_ranks(np.array([5.0, 1, 1, 9]))) == [3.0, 1.5, 1.5, 4.0]
    assert eda.spearman(np.array([1.0, 2, 3, 4]), np.array([4.0, 3, 2, 1])) == pytest.approx(-1.0)
    assert np.isnan(eda.spearman(np.array([2.0, 2, 2, 2]), np.array([1.0, 2, 3, 4])))
    x = np.array([1.0, np.nan, 3, 4, 5])  # NaN rows are dropped, never ranked as a value
    assert eda.spearman(x, np.array([1.0, 9, 2, 3, 4])) == pytest.approx(
        eda.spearman(np.array([1.0, 3, 4, 5]), np.array([1.0, 2, 3, 4]))
    )


def test_median_and_quartiles_by_hand() -> None:
    values = np.array([1.0, 2, 3, 4, 10, 5, np.nan, 7])
    rating = np.array([0, 0, 0, 0, 1, 3, 3, 3])
    s = eda.rating_stats(values, rating)
    assert (s[0]["n"], s[0]["median"], s[0]["q1"], s[0]["q3"]) == (4, 2.5, 1.75, 3.25)
    assert (s[1]["n"], s[1]["median"], s[1]["q1"], s[1]["q3"]) == (1, 10.0, 10.0, 10.0)
    assert s[2]["n"] == 0 and np.isnan(s[2]["median"])
    assert (s[3]["n"], s[3]["n_nan"], s[3]["median"]) == (2, 1, 6.0)


def test_proportions_events_and_few_events_flag() -> None:
    ind = np.array([1.0, 0, 1, np.nan, 0, 0, 1, 1, 1, 1, 1, 1])
    rating = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2])
    plain = eda.proportion_by_rating(ind, rating, None)
    assert [(r["events"], r["n"]) for r in plain] == [(2, 3), (0, 2), (6, 6), (0, 0)]
    assert plain[0]["prop"] == pytest.approx(2 / 3) and np.isnan(plain[3]["prop"])
    draws = eda.cluster_draws(np.arange(12).astype(str), 50, np.random.default_rng(0))
    flagged = eda.proportion_by_rating(ind, rating, draws)
    assert flagged[0]["flag"] == "few events" and flagged[2]["flag"] == ""


def test_cluster_bootstrap_resamples_participants_with_all_their_clips() -> None:
    participants = np.array(["a", "a", "b", "c", "c", "c"])
    draws = eda.cluster_draws(participants, 300, np.random.default_rng(0))
    assert len(draws) == 300
    for d in draws:
        counts = np.bincount(d, minlength=6)
        assert counts[0] == counts[1] and counts[3] == counts[4] == counts[5]  # clips stay together
    assert len({len(d) for d in draws}) > 1  # the total varies because participants differ in size
    again = eda.cluster_draws(participants, 300, np.random.default_rng(0))
    assert all(np.array_equal(a, b) for a, b in zip(draws, again, strict=True))


def test_undefined_rho_share_is_reported_not_dropped() -> None:
    rating = np.tile([0.0, 1, 2, 3], 5)
    constant = np.full(20, 2.0)
    draws = eda.cluster_draws(np.arange(20).astype(str), 100, np.random.default_rng(0))
    r = eda.rho_ci(constant, rating, draws)
    assert np.isnan(r["rho"]) and r["undefined_share"] == 1.0 and r["unstable"] is True
    varying = np.arange(20, dtype=float)
    ok = eda.rho_ci(varying, rating, draws)
    assert (
        ok["undefined_share"] == 0.0 and ok["unstable"] is False and ok["ci_low"] <= ok["ci_high"]
    )


# ----------------------------------------------------------------------- fixed choices


def test_limits_scale_and_jitter_are_fixed_from_the_feature_alone() -> None:
    assert eda.axis_limits(np.array([1.0, 3.0, np.nan]), "linear") == pytest.approx((0.9, 3.1))
    f = 100.0**0.05
    assert eda.axis_limits(np.array([1.0, 100.0]), "log") == pytest.approx((1 / f, 100 * f))
    assert eda.choose_scale("longest_pause_s", np.array([0.3, 2.0])) == ("log", "")
    scale, note = eda.choose_scale("longest_pause_s", np.array([0.0, 2.0]))
    assert scale == "linear" and "refused" in note
    assert eda.choose_scale("amp_slope", np.array([-1.0, 2.0]))[0] == "linear"
    digest = hashlib.sha256(b"amp_slope:2026").digest()
    expected = np.random.default_rng(int.from_bytes(digest[:8], "big")).uniform(-0.18, 0.18, 10)
    assert np.array_equal(eda.jitter_for("amp_slope", 2026, 10), expected)
    assert not np.array_equal(eda.jitter_for("amp_slope", 2027, 10), expected)
    assert not np.array_equal(eda.jitter_for("iti_slope", 2026, 10), expected)


def test_axis_limits_do_not_depend_on_the_ratings() -> None:
    frame = make_frame()
    shuffled = frame.copy()
    shuffled["rating"] = np.random.default_rng(9).permutation(frame["rating"].to_numpy())
    a, b = figures(frame), figures(shuffled)
    for f in eda.SET_B:
        assert a[f].axes[0].get_ylim() == b[f].axes[0].get_ylim(), f
        assert a[f].axes[0].get_yscale() == b[f].axes[0].get_yscale()
    assert a["longest_pause_s"].axes[0].get_yscale() == "log"
    assert a["amp_slope"].axes[0].get_yscale() == "linear"


# ----------------------------------------------------------------------- the figures


def test_nan_is_never_plotted_and_n_labels_match_a_hand_count() -> None:
    frame = make_frame()
    figs = figures(frame)
    x = pd.to_numeric(frame["amp_late_early"].replace("", np.nan))
    pts = strip_points(figs["amp_late_early"])
    assert len(pts) == int(x.notna().sum()) == N_CLIPS - 2
    assert sorted(round(p[1], 12) for p in pts) == sorted(round(v, 12) for v in x.dropna())
    rating = frame["rating"].astype(int)
    labels = [t.get_text() for t in figs["amp_late_early"].axes[0].get_xticklabels()]
    expected = [f"{lv}\n(n={int(((rating == lv) & x.notna()).sum())})" for lv in range(4)]
    assert labels == expected
    nan_texts = [t for t in texts(figs["amp_late_early"]) if t.startswith("NaN ")]
    assert sorted(nan_texts) == sorted(
        f"NaN {int(((rating == lv) & x.isna()).sum())}" for lv in range(4)
    )


def test_onset_is_plotted_only_for_clips_with_a_decrement() -> None:
    frame = make_frame()
    fig = figures(frame)["decrement_onset_frac"]
    true = frame["decrement_present"] == "True"
    assert len(strip_points(fig)) == int(true.sum())
    rating = frame["rating"].astype(int)
    nan_texts = sorted(t for t in texts(fig) if t.startswith("NaN "))
    assert nan_texts == sorted(f"NaN {int(((rating == lv) & ~true).sum())}" for lv in range(4))
    assert "among clips with a decrement" in " ".join(texts(fig))


def test_proportion_and_count_panels_and_the_iti_band() -> None:
    figs = figures(make_frame())
    prop = [c for c in figs["decrement_present"].axes[0].collections if c.get_gid() == "proportion"]
    assert len(prop) == 1 and len(prop[0].get_offsets()) == 4
    halts = " ".join(texts(figs["n_halts"]))
    assert "counts only" in halts and "rho" not in halts.replace("counts only, no rho", "")
    assert len(figs["n_hesitations"].axes) == 2
    iti = " ".join(texts(figs["iti_slope"]))
    assert "split-half reliability 0.43" in iti
    assert any(abs(p.get_height()) > 0 for p in figs["iti_slope"].axes[0].patches)


def test_hand_is_a_marker_shape_only(ran: dict[str, Any]) -> None:
    frame = make_frame()
    swapped = frame.copy()
    swapped["hand"] = swapped["hand"].map({"left": "right", "right": "left"})
    fa, fb_ = figures(frame), figures(swapped)
    for f in eda.SET_B:  # the same points, only the marker groups differ
        assert sorted(strip_points(fa[f])) == sorted(strip_points(fb_[f])), f
    assert eda.rating_stats(np.arange(5.0), np.array([0, 1, 2, 3, 3])) == eda.rating_stats(
        np.arange(5.0), np.array([0, 1, 2, 3, 3])
    )


# ----------------------------------------------------------------------- the run


def test_exact_file_set_and_nothing_else_written(ran: dict[str, Any]) -> None:
    assert sorted(p.name for p in ran["a"].iterdir()) == FILE_NAMES and len(FILE_NAMES) == 15
    outside = sorted(p.name for p in ran["processed"].iterdir())
    assert outside == ["eda_a", "eda_b", "feature_roles.csv", "features.csv", "features_meta.json"]
    assert sorted(p.name for p in ran["root"].iterdir()) == ["processed"]


def test_rerun_is_byte_identical(ran: dict[str, Any]) -> None:
    for name in FILE_NAMES:
        assert (ran["a"] / name).read_bytes() == (ran["b"] / name).read_bytes(), name


def test_summary_csv_is_in_column_order_with_expected_columns(ran: dict[str, Any]) -> None:
    table = pd.read_csv(ran["a"] / eda.OUT_SUMMARY_CSV)
    assert len(table) == 48 and table["feature"].drop_duplicates().tolist() == list(eda.SET_B)
    assert table["rating"].tolist() == [0, 1, 2, 3] * 12
    for col in ("n", "n_nan", "median", "q1", "q3", "n_true", "rho", "ci_low", "ci_high",
                "undefined_share", "flags"):  # fmt: skip
        assert col in table.columns
    halts = table[table["feature"] == "n_halts"]
    assert halts["rho"].isna().all() and (halts["flags"] == "counts only").all()
    onset = table[table["feature"] == "decrement_onset_frac"]
    assert (onset["flags"] == "conditional").all()
    frame = make_frame()
    rating = frame["rating"].astype(int)
    amp = pd.to_numeric(frame["amp_slope"])
    row = table[(table["feature"] == "amp_slope") & (table["rating"] == 2)].iloc[0]
    assert row["n"] == int((rating == 2).sum()) and row["median"] == pytest.approx(
        amp[rating == 2].median()
    )
    true = frame["decrement_present"] == "True"
    drow = table[(table["feature"] == "decrement_present") & (table["rating"] == 1)].iloc[0]
    assert drow["n_true"] == int(((rating == 1) & true).sum())


def test_summary_figure_follows_the_column_order_not_the_rho_order() -> None:
    frame = make_frame()
    d, roles = data_and_roles(frame)
    draws = eda.cluster_draws(d.participant, 20, np.random.default_rng(1))
    results = {f: {"rho": eda.rho_ci(d.values[f], d.rating, draws)} for f in eda.SET_B}
    results["n_halts"]["rho"] = {"rho": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                                 "undefined_share": np.nan, "unstable": False}  # fmt: skip
    fig = eda.summary_figure(results, roles)
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert [lbl.split(" ")[0] for lbl in labels] == list(eda.SET_B)
    rhos = [results[f]["rho"]["rho"] for f in eda.SET_B if f != "n_halts"]
    assert rhos != sorted(rhos)
    assert "n_halts (descriptive; counts only)" in labels
    assert eda.SUBTITLE in texts(fig)


def test_meta_contents(ran: dict[str, Any]) -> None:
    meta = json.loads((ran["a"] / eda.OUT_META).read_text())
    sha = lambda p: hashlib.sha256((ran["processed"] / p).read_bytes()).hexdigest()[:8]  # noqa: E731
    assert meta["inputs_sha8"] == {"features_csv": sha("features.csv"),
                                   "feature_roles_csv": sha("feature_roles.csv"),
                                   "features_meta_json": sha("features_meta.json")}  # fmt: skip
    assert meta["seed"] == 2026 and meta["boot"] == 60 and meta["rows"] == N_CLIPS
    assert meta["versions"]["matplotlib"] == matplotlib.__version__
    assert meta["scale"]["longest_pause_s"] == "log" and meta["scale"]["amp_slope"] == "linear"
    assert meta["statement"] == eda.SUBTITLE
    assert not re.search(r"\d{4}-\d{2}-\d{2}|utc|timestamp", json.dumps(meta), re.IGNORECASE)


def test_no_identity_group_word_or_p_value_in_any_output(ran: dict[str, Any]) -> None:
    pattern = re.compile("|".join(POISON) + r"|ID\d{4}", re.IGNORECASE)
    for p in sorted(ran["a"].iterdir()):
        assert not pattern.search(p.name), p.name
        raw = p.read_bytes()
        assert not pattern.search(raw.decode("latin-1")), p.name
        if p.suffix == ".png":
            assert raw[:8] == b"\x89PNG\r\n\x1a\n" and b"Software" not in raw
    p_value = re.compile(r"p[-_ ]?value|\bp\s*[<=>]", re.IGNORECASE)
    for p in (ran["a"] / eda.OUT_SUMMARY_CSV, ran["a"] / eda.OUT_META):
        assert not p_value.search(p.read_text())
    for fig in figures(make_frame()).values():
        for t in texts(fig):
            assert not pattern.search(t) and not p_value.search(t), t
    cell = {"rho": 0.1, "ci_low": 0.0, "ci_high": 0.2, "undefined_share": 0.0, "unstable": False}
    summary = eda.summary_figure(
        {f: {"rho": cell} for f in eda.SET_B},
        {c: fm.ROLES.get(c, ("audit_count", "x")) for c in COLUMNS},
    )
    assert not pattern.search(" ".join(texts(summary)))


def test_the_label_file_is_never_opened(
    world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decoy = tmp_path / "raw" / "fis_diagnostic.csv"
    decoy.parent.mkdir()
    decoy.write_text("ID,UPDRS\nCONTROL01_DCHA,0\n")
    allowed = [world.resolve()]
    free = [Path(matplotlib.get_data_path()).resolve(), Path(matplotlib.get_cachedir()).resolve(),
            Path(matplotlib.get_configdir()).resolve()]  # fmt: skip
    seen: list[str] = []

    def guard(path: Any) -> None:
        if isinstance(path, str | Path) and str(path):
            p = Path(path).resolve()
            seen.append(p.name)
            ok = p.suffix in (".py", ".pyc", ".so", ".ttf", ".afm") or any(
                p.is_relative_to(a) for a in allowed + free
            )
            if not ok:
                raise AssertionError(f"read outside the allowed inputs: {p.name}")

    def wrap(fn: Any) -> Any:
        def inner(path: Any, *a: Any, **k: Any) -> Any:
            guard(path)
            return fn(path, *a, **k)

        return inner

    for owner, name in ((builtins, "open"), (io, "open"), (pd, "read_csv")):
        monkeypatch.setattr(owner, name, wrap(getattr(owner, name)))
    assert run(world, world / "eda") == 0
    assert "features.csv" in seen and "fis_diagnostic.csv" not in seen
    with pytest.raises(AssertionError):
        pd.read_csv(decoy)


# ----------------------------------------------------------------------- refusals


def test_roles_hash_mismatch_is_exit_1(world: Path) -> None:
    meta = json.loads((world / "features_meta.json").read_text())
    meta["roles_sha8"] = "deadbeef"
    (world / "features_meta.json").write_text(json.dumps(meta))
    assert run(world) == 1 and not (world / "eda").exists()
    meta["roles_sha8"] = eda.roles_sha8(
        pd.read_csv(world / "feature_roles.csv", dtype=str, keep_default_na=False)
    )
    (world / "features_meta.json").write_text(json.dumps(meta))
    roles = pd.read_csv(world / "feature_roles.csv", dtype=str, keep_default_na=False)
    roles.loc[roles["column"] == "n_halts", "role"] = "feature_primary"
    roles.to_csv(world / "feature_roles.csv", index=False)
    assert run(world) == 1  # the file no longer matches the pinned hash


@pytest.mark.parametrize("bad", ["", "4", "1.5", "x", "-1"])
def test_blank_or_invalid_rating_is_exit_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    monkeypatch.setattr(eda, "PROCESSED_ROOT", tmp_path / "processed")
    frame = make_frame()
    frame.loc[4, "rating"] = bad
    assert run(write_world(tmp_path, frame)) == 1


def test_inconsistent_onset_is_exit_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eda, "PROCESSED_ROOT", tmp_path / "processed")
    frame = make_frame()
    false_row = frame.index[frame["decrement_present"] == "False"][0]
    frame.loc[false_row, "decrement_onset_frac"] = "0.5"
    assert run(write_world(tmp_path, frame)) == 1


def test_missing_or_extra_set_b_column_is_exit_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(eda, "PROCESSED_ROOT", tmp_path / "processed")
    assert run(write_world(tmp_path, make_frame().drop(columns=["n_halts"]))) == 1
    extra = make_frame()
    extra.insert(extra.columns.get_loc("n_valid_taps") + 1, "brand_new", "1")
    assert run(write_world(tmp_path, extra)) == 1


def test_log_scale_is_refused_for_non_positive_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(eda, "PROCESSED_ROOT", tmp_path / "processed")
    frame = make_frame()
    frame.loc[3, "longest_pause_s"] = "0.0"
    processed = write_world(tmp_path, frame)
    assert run(processed) == 0
    meta = json.loads((processed / "eda" / eda.OUT_META).read_text())
    assert meta["scale"]["longest_pause_s"].startswith("linear (log refused")
    assert meta["scale"]["longest_pause_ratio"] == "log"


def test_missing_inputs_are_exit_2(world: Path) -> None:
    for name in ("feature_roles.csv", "features_meta.json", "features.csv"):
        (world / name).rename(world / (name + ".moved"))
        assert run(world) == 2
        (world / (name + ".moved")).rename(world / name)
    assert run(world) == 0


@pytest.mark.parametrize("where", ["elsewhere", "processed_sibling"])
def test_out_dir_outside_processed_is_exit_2_and_writes_nothing(
    world: Path, tmp_path: Path, where: str
) -> None:
    target = tmp_path / where / "eda"
    assert run(world, target) == 2
    assert not target.exists() and not (tmp_path / where).exists()
    assert not (world / "eda").exists()


def test_boot_must_be_positive(world: Path) -> None:
    assert eda.main(["--features", str(world / "features.csv"), "--boot", "0"]) == 2


def test_percentile_ci_by_hand() -> None:
    assert eda.percentile_ci(np.arange(101.0)) == pytest.approx((2.5, 97.5))
    low, high = eda.percentile_ci(np.array([np.nan, 1.0, 2.0, 3.0]))  # undefined resamples ignored
    assert 1.0 <= low <= high <= 3.0 and all(np.isnan(eda.percentile_ci(np.array([np.nan]))))


def test_hand_changes_no_statistic_and_the_seed_changes_the_bootstrap(
    ran: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(eda, "PROCESSED_ROOT", tmp_path / "processed")
    frame = make_frame()
    frame["hand"] = frame["hand"].map({"left": "right", "right": "left"})
    processed = write_world(tmp_path, frame)
    assert run(processed, processed / "swapped") == 0
    same = (ran["a"] / eda.OUT_SUMMARY_CSV).read_bytes()
    assert (processed / "swapped" / eda.OUT_SUMMARY_CSV).read_bytes() == same
    assert run(processed, processed / "seed5", "--seed", "5") == 0
    other = pd.read_csv(processed / "seed5" / eda.OUT_SUMMARY_CSV)
    base = pd.read_csv(ran["a"] / eda.OUT_SUMMARY_CSV)
    assert np.array_equal(other["rho"], base["rho"], equal_nan=True)  # no seed in the estimate
    assert not np.array_equal(other["ci_low"], base["ci_low"], equal_nan=True)  # but in the CI
    assert json.loads((processed / "seed5" / eda.OUT_META).read_text())["seed"] == 5


def test_every_interval_in_the_csv_comes_from_participant_cluster_draws(
    ran: dict[str, Any],
) -> None:
    """Recompute the CIs from clusters drawn with the run's seed and boot, and compare."""
    d, _ = data_and_roles(make_frame())
    draws = eda.cluster_draws(d.participant, 60, np.random.default_rng(2026))
    rows = np.random.default_rng(2026)  # what a row-wise resample would give instead
    row_draws = [rows.integers(0, d.n, d.n) for _ in range(60)]
    table = pd.read_csv(ran["a"] / eda.OUT_SUMMARY_CSV)
    for f in ("decrement_present", "n_hesitations"):
        x = d.values[f]
        ind = (
            x
            if f == "decrement_present"
            else np.where(np.isfinite(x), (x > 0).astype(float), np.nan)
        )
        part = table[table["feature"] == f]
        for cluster_or_rows, same in ((draws, True), (row_draws, False)):
            props = eda.proportion_by_rating(ind, d.rating, cluster_or_rows)
            low = [p["low"] for p in props]
            high = [p["high"] for p in props]
            got = np.allclose(part["prop_low"], low, equal_nan=True) and np.allclose(
                part["prop_high"], high, equal_nan=True
            )
            assert got is same, f
    for f in ("amp_slope", "iti_slope"):
        part = table[table["feature"] == f]
        for cluster_or_rows, same in ((draws, True), (row_draws, False)):
            r = eda.rho_ci(d.values[f], d.rating, cluster_or_rows)
            got = np.allclose(part["ci_low"], r["ci_low"]) and np.allclose(
                part["ci_high"], r["ci_high"]
            )
            assert got is same, f
