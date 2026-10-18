"""Tests for models/tapping/src/cv_harness.py (task FT-32, part 2). Synthetic matrices only (about
60 participants x 2 clips, planted signal or none, hashed ids, a decoy label file that must never be
opened). No test uses the real export, so no real-label metric is ever computed here."""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv_harness as ch  # noqa: E402
import cv_splits as cvs  # noqa: E402
from test_cv_splits import EXTRA_ROLES, hid, write_world  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src"
SMALL_LGBM = [
    {"num_leaves": 3, "n_estimators": 20, "min_child_samples": 5},
    {"num_leaves": 5, "n_estimators": 20, "min_child_samples": 5},
]
METRIC_WORDS = re.compile(r"auroc|balanced|sensitivity|specificity|mcc|ppv", re.I)


def make_data(
    mode: str = "signal", n_part: int = 60, shift: float = 1.2, seed: int = 0
) -> pd.DataFrame:
    """One singleton participant, the rest with two clips (right, left). Modes: signal, noise,
    identity (features and labels constant within a participant, unrelated to each other)."""
    rng = np.random.default_rng(seed)
    z = np.array([1, 0] * (n_part // 2))
    rng.shuffle(z)
    rows = []
    for i in range(n_part):
        for c in range(1 if i == 0 else 2):
            if mode == "identity":
                rating = 2 if z[i] else 0
            elif z[i]:
                rating = 1 if (c == 1 and i % 10 == 0) else 2 + (i + c) % 2
            else:
                rating = (i + c) % 2
            rows.append(
                (hid("hc", 1000 * i + c), hid("hp", i), "right" if c == 0 else "left", rating, i)
            )
    frame = pd.DataFrame(rows, columns=["ref", "participant", "hand", "rating", "i"])
    zc = z[frame["i"].to_numpy()]
    X = rng.normal(size=(len(frame), len(cvs.MODEL_COLUMNS)))
    if mode == "signal":
        X[:, [0, 5, 8]] += shift * zc[:, None]
    if mode == "identity":
        X = rng.normal(size=(n_part, len(cvs.MODEL_COLUMNS)))[frame["i"].to_numpy()]
    for j, col in enumerate(cvs.MODEL_COLUMNS):
        frame[col] = X[:, j]
    frame["decrement_present"] = frame["decrement_present"] > 0
    frame["decrement_onset_frac"] = np.where(
        frame["decrement_present"], frame["decrement_onset_frac"].abs(), np.nan
    )
    for col in EXTRA_ROLES:
        if col not in frame.columns:
            frame[col] = rng.normal(size=len(frame))
    return frame.drop(columns="i")


def build_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame, repeats: int = 2
) -> dict[str, Any]:
    proc = tmp_path / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    cv_root = proc / "cv"
    monkeypatch.setattr(cvs, "CV_ROOT", cv_root)
    raw = tmp_path / "data" / "raw" / "HUBU-FIS"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "fis_diagnostic.csv").write_text("ID,UPDRS\nCONTROL01_DCHA,0\nID1004_IZDA,3\n")
    w: dict[str, Any] = {"proc": proc, "cv": cv_root, "tmp": tmp_path}
    write_world(w, frame)
    code = cvs.main(
        ["--features", str(proc / "features.csv"), "--roles", str(proc / "feature_roles.csv"),
         "--meta", str(proc / "features_meta.json"), "--target", "ge2", "--repeats", str(repeats),
         "--out-dir", str(cv_root)]
    )  # fmt: skip
    assert code == 0
    w["repeats"] = repeats
    return w


@pytest.fixture(autouse=True)
def small_lgbm_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(ch.GRIDS, "lgbm", [dict(p) for p in SMALL_LGBM])


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    return build_world(tmp_path, monkeypatch, make_data())


def harness(w: dict[str, Any], command: str, *extra: str, out: Path | None = None) -> int:
    return ch.main(
        [command, "--features", str(w["proc"] / "features.csv"), "--roles", str(w["proc"] / "feature_roles.csv"),
         "--meta", str(w["proc"] / "features_meta.json"), "--splits-dir", str(w["cv"]),
         "--out-dir", str(out or w["cv"]), *extra]
    )  # fmt: skip


def run(w: dict[str, Any], model: str, fs: str = "A", *extra: str, out: Path | None = None) -> int:
    return harness(
        w,
        "run",
        "--model",
        model,
        "--feature-set",
        fs,
        "--repeats",
        str(w["repeats"]),
        *extra,
        out=out,
    )


def frame_of(w: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(w["proc"] / "features.csv", dtype={"ref": str, "participant": str})
    return cvs.canonical(frame)


def arrays(w: dict[str, Any], fs: str = "A") -> dict[str, Any]:
    frame = frame_of(w)
    X, cols = ch.design_matrix(frame, fs)
    splits = cvs.load_splits(w["cv"] / "ge2" / "splits.csv", frame)
    groups = frame["participant"].to_numpy()
    return {"frame": frame, "X": X, "cols": cols, "y": cvs.target_labels(frame, "ge2"),
            "groups": groups, "splits": splits, "member": ch.membership(groups, splits, 0)}  # fmt: skip


def out_dir(w: dict[str, Any]) -> Path:
    return w["cv"] / "ge2"


def snapshot(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}  # fmt: skip


# --------------------------------------------------------------------------- constants


def test_constants_and_grids_are_pinned() -> None:
    assert ch.THRESHOLD == 0.5 and ch.LGBM_SEED == 20260 and ch.PERMUTATION_MAX_MEAN_AUROC == 0.55
    assert ch.TARGET_STATUS == "pending_default_ge2" and ch.PPV_PREVALENCES == (0.01, 0.02, 0.05)
    assert ch.MODELS == ("b0", "b1", "lgbm")
    assert ch.LGBM_FIXED == {
        "learning_rate": 0.05, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
        "reg_lambda": 1.0, "class_weight": "balanced", "deterministic": True,
        "force_row_wise": True, "n_jobs": 1, "verbose": -1,
    }  # fmt: skip
    assert ch.B1_FIXED == {"class_weight": "balanced", "solver": "lbfgs", "max_iter": 2000}


def test_real_grids_are_pinned_exactly() -> None:
    real = ch.real_grids()
    assert real["b0"] == [{}]
    assert real["b1"] == [{"C": 0.01}, {"C": 0.1}, {"C": 1.0}, {"C": 10.0}]
    lgbm = real["lgbm"]
    assert len(lgbm) == 27 and len({json.dumps(g, sort_keys=True) for g in lgbm}) == 27
    expected = [
        {"num_leaves": nl, "n_estimators": ne, "min_child_samples": mc}
        for nl in (3, 5, 7) for ne in (100, 200, 300) for mc in (20, 15, 10)
    ]  # fmt: skip
    assert lgbm == expected
    assert lgbm[0] == {"num_leaves": 3, "n_estimators": 100, "min_child_samples": 20}
    assert lgbm[-1] == {"num_leaves": 7, "n_estimators": 300, "min_child_samples": 10}
    assert set(real) == {"b0", "b1", "lgbm"} and set(ch.GRIDS) == set(real)
    assert ch.GRIDS["b1"] == real["b1"] and ch.GRIDS["b0"] == real["b0"]


def test_estimators_carry_the_preregistered_settings() -> None:
    lgbm = ch.make_estimator("lgbm", SMALL_LGBM[0])
    params = lgbm.get_params()
    for key, value in ch.LGBM_FIXED.items():
        assert params[key] == value
    assert params["random_state"] == ch.LGBM_SEED and params["num_leaves"] == 3
    b1 = ch.make_estimator("b1", {"C": 0.1})
    assert list(b1.named_steps) == ["scale", "clf"]
    clf = b1.named_steps["clf"]
    assert (clf.C, clf.class_weight, clf.solver, clf.max_iter) == (0.1, "balanced", "lbfgs", 2000)
    assert isinstance(ch.make_estimator("b0", {}), ch.PrevalenceClassifier)


def test_feature_sets_are_exactly_the_registry_columns() -> None:
    import feature_matrix as fm  # the registry itself; the harness never imports it

    assert ch.FEATURE_SETS["A"] == list(cvs.SET_A_PRIMARY)
    assert ch.FEATURE_SETS["B"] == list(cvs.SET_A_PRIMARY) + list(cvs.SET_B_ADDED)
    assert (
        sorted(ch.FEATURE_SETS["B"]) == sorted(fm.PRIMARY_COLUMNS)
        and len(ch.FEATURE_SETS["B"]) == 13
    )


def test_design_matrix_is_float_keeps_nan_and_only_registry_columns() -> None:
    frame = cvs.canonical(make_data())
    for fs, n in (("A", 5), ("B", 13)):
        X, cols = ch.design_matrix(frame, fs)
        assert X.dtype == np.float64 and X.shape == (len(frame), n) and cols == ch.FEATURE_SETS[fs]
        assert not {"ref", "participant", "hand", "rating"} & set(cols)
    X, cols = ch.design_matrix(frame, "B")
    j = cols.index("decrement_present")
    assert set(np.unique(X[:, j])) <= {0.0, 1.0} and (X[:, j] == frame["decrement_present"]).all()
    k = cols.index("decrement_onset_frac")
    assert (
        np.isnan(X[:, k]).sum() == frame["decrement_onset_frac"].isna().sum() > 0
    )  # NaN stays NaN


@pytest.mark.parametrize(
    "column", ["n_taps", "n_over_duration_hz", "amp_mean_sens", "hand", "rating"]
)
def test_design_matrix_refuses_non_primary_columns(
    column: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(ch.FEATURE_SETS, "A", [*cvs.SET_A_PRIMARY, column])
    with pytest.raises(cvs.ContractError):
        ch.design_matrix(cvs.canonical(make_data()), "A")


# --------------------------------------------------------------------------- metrics


def test_metrics_by_hand() -> None:
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    proba = np.array([0.9, 0.8, 0.3, 0.6, 0.2, 0.1, 0.4, 0.55])
    m = ch.threshold_metrics(y, proba)  # tp 2, fn 1, fp 2, tn 3
    sens, spec = 2 / 3, 3 / 5
    assert m["sensitivity"] == pytest.approx(sens) and m["specificity"] == pytest.approx(spec)
    assert m["balanced_accuracy"] == pytest.approx((sens + spec) / 2)
    assert m["balanced_accuracy"] != pytest.approx(5 / 8)  # not plain accuracy
    assert m["mcc"] == pytest.approx(4 / np.sqrt(4 * 3 * 5 * 4)) and m["n_pred_pos"] == 4
    for pct, p in ((1, 0.01), (2, 0.02), (5, 0.05)):
        assert m[f"ppv_at_{pct}pct"] == pytest.approx(sens * p / (sens * p + (1 - spec) * (1 - p)))
    assert ch.auroc(y, proba) == pytest.approx(12 / 15)


def test_auroc_ties_constant_scores_and_single_class() -> None:
    assert ch.auroc(np.array([1, 0]), np.array([0.5, 0.5])) == 0.5
    assert ch.auroc(np.array([1, 1, 0, 0, 0]), np.full(5, 0.3)) == 0.5
    assert ch.auroc(np.array([1, 1, 0, 0]), np.array([0.7, 0.5, 0.5, 0.2])) == pytest.approx(0.875)
    assert np.isnan(ch.auroc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])))
    rng = np.random.default_rng(0)
    for _ in range(5):
        y = rng.integers(0, 2, 60)
        s = np.round(rng.normal(size=60), 1)  # many ties
        assert ch.auroc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_mcc_zero_denominator_is_zero_and_visible() -> None:
    m = ch.threshold_metrics(np.array([1, 1, 0, 0]), np.array([0.1, 0.2, 0.3, 0.4]))
    assert (
        m["mcc"] == 0.0
        and m["n_pred_pos"] == 0
        and m["sensitivity"] == 0.0
        and m["specificity"] == 1.0
    )
    assert m["balanced_accuracy"] == 0.5


def test_ppv_is_nan_when_its_denominator_is_zero() -> None:
    assert np.isnan(ch.ppv(0.0, 1.0, 0.02)) and np.isnan(ch.ppv(float("nan"), 0.9, 0.02))
    assert ch.ppv(1.0, 1.0, 0.02) == 1.0


def test_pooled_auroc_is_reported_and_differs_from_the_fold_mean() -> None:
    rng = np.random.default_rng(3)
    member = np.repeat(np.arange(5), 20)
    y = rng.integers(0, 2, 100)
    proba = np.clip(
        0.5 * y + rng.normal(0.3, 0.2, 100) + 0.35 * member, 0, 1
    )  # fold-shifted scores
    m = ch.repeat_metrics(y, proba, member)
    assert m["auroc"] == pytest.approx(roc_auc_score(y, proba))
    assert abs(m["auroc"] - m["auroc_fold_mean"]) > 1e-3
    assert m["auroc_fold_mean"] == pytest.approx(
        np.mean([roc_auc_score(y[member == f], proba[member == f]) for f in range(5)])
    )
    assert m["n_clips"] == 100 and m["n_pos"] == y.sum() and m["n_neg"] == (1 - y).sum()


def test_a_single_class_fold_makes_the_fold_statistics_nan_not_zero() -> None:
    member = np.repeat(np.arange(5), 4)
    y = np.tile([0, 1, 0, 1], 5)
    y[member == 2] = 0
    m = ch.repeat_metrics(y, np.linspace(0.05, 0.95, 20), member)
    assert (
        np.isnan(m["auroc_fold_mean"]) and np.isnan(m["auroc_fold_sd"]) and not np.isnan(m["auroc"])
    )


# --------------------------------------------------------------------------- fitting


class Spy:
    """Wraps an estimator and logs which rows it is fitted on and asked about."""

    def __init__(self, est: Any, log: list[Any], lookup: dict[bytes, int]) -> None:
        self.est, self.log, self.lookup = est, log, lookup

    def _pos(self, X: np.ndarray) -> list[int]:
        return [self.lookup[row.tobytes()] for row in X]

    def fit(self, X: np.ndarray, y: np.ndarray, *args: Any, **kwargs: Any) -> Spy:
        self.log.append(("fit", self._pos(X), np.asarray(y).copy(), args, kwargs, X.shape[1]))
        self.est.fit(X, y, *args, **kwargs)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.log.append(("predict", self._pos(X), None, (), {}, X.shape[1]))
        return self.est.predict_proba(X)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.est, name)


def install_spy(monkeypatch: pytest.MonkeyPatch, X: np.ndarray) -> list[Any]:
    lookup = {row.tobytes(): i for i, row in enumerate(X)}
    assert len(lookup) == len(X)
    log: list[Any] = []
    real = ch.make_estimator
    monkeypatch.setattr(
        ch,
        "make_estimator",
        lambda model, params, *a, **k: Spy(real(model, params, *a, **k), log, lookup),
    )
    return log


@pytest.mark.parametrize("fs", ["A", "B"])
def test_fit_never_sees_outer_test_rows_and_only_the_registry_columns(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch, fs: str
) -> None:
    a = arrays(world, fs)
    log = install_spy(monkeypatch, a["X"])
    assert run(world, "lgbm", fs, "--repeats", "1") == 0
    grid, per_fold = len(SMALL_LGBM), 2 * (3 * len(SMALL_LGBM) + 1)
    assert len(log) == 5 * per_fold
    n_cols = len(a["cols"])
    for fold in range(5):
        train = set(np.flatnonzero(a["member"] != fold).tolist())
        test = set(np.flatnonzero(a["member"] == fold).tolist())
        part = log[fold * per_fold : (fold + 1) * per_fold]
        for kind, pos, y_seen, args, kwargs, width in part:
            assert width == n_cols and not args and not kwargs  # no early stopping, no extras
            assert len(pos) == len(set(pos))  # no duplicated rows (no oversampling)
            if kind == "fit":
                assert np.array_equal(y_seen, a["y"][pos])
        tuning, final = part[: 2 * 3 * grid], part[2 * 3 * grid :]
        for (kf, fit_pos, *_), (kp, val_pos, *_) in zip(tuning[0::2], tuning[1::2], strict=True):
            assert kf == "fit" and kp == "predict"
            assert set(fit_pos) <= train and set(val_pos) <= train and not set(val_pos) & test
            assert not set(a["groups"][fit_pos]) & set(a["groups"][val_pos])  # participant-wise
        assert sorted(final[0][1]) == sorted(train) and sorted(final[1][1]) == sorted(test)


def fold_inputs(w: dict[str, Any], fs: str = "A") -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    a = arrays(w, fs)
    return a, np.flatnonzero(a["member"] != 0), np.flatnonzero(a["member"] == 0)


@pytest.mark.parametrize("model", ["b0", "b1", "lgbm"])
def test_nothing_is_learned_from_the_outer_test_rows(world: dict[str, Any], model: str) -> None:
    a, train, test = fold_inputs(world)
    grid = ch.grid_for(model)
    base = ch.fit_outer_fold(model, grid, a["X"], a["y"], a["groups"], train, 0, 0)
    X2, y2 = a["X"].copy(), a["y"].copy()
    X2[test] = X2[test] * 1000 + 50  # corrupt only the held-out rows
    y2[test] = 1  # all held-out labels positive: changes every total, whatever the balance
    other = ch.fit_outer_fold(model, grid, X2, y2, a["groups"], train, 0, 0)
    assert base[0] == other[0]
    assert (np.isnan(base[1]) and np.isnan(other[1])) or base[1] == other[1]
    assert np.array_equal(
        base[2].predict_proba(a["X"][train]), other[2].predict_proba(a["X"][train])
    )


def test_scaler_is_fit_on_the_training_part_only(world: dict[str, Any]) -> None:
    a, train, test = fold_inputs(world)
    est = ch.refit_outer(a["frame"], "b1", {"C": 1.0}, train, "A", "ge2")
    scaler = est.named_steps["scale"]
    assert np.allclose(scaler.mean_, a["X"][train].mean(axis=0))
    assert not np.allclose(scaler.mean_, a["X"].mean(axis=0))


def test_b0_probability_is_the_training_prevalence() -> None:
    est = ch.PrevalenceClassifier().fit(np.zeros((6, 2)), np.array([1, 0, 0, 0, 0, 1]))
    assert np.allclose(est.predict_proba(np.zeros((3, 2))), [[2 / 3, 1 / 3]] * 3)


class Tied:
    """Scores depend on the config only through a weight on a noise column."""

    def __init__(self, params: dict[str, Any]) -> None:
        self.noise = params["noise"]

    def fit(self, X: np.ndarray, y: np.ndarray) -> Tied:
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        s = 1 / (1 + np.exp(-(X[:, 0] + self.noise * X[:, 1])))
        return np.column_stack([1 - s, s])


def test_tie_break_takes_the_first_configuration(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    a, train, _ = fold_inputs(world)
    monkeypatch.setattr(ch, "make_estimator", lambda model, params, *x, **k: Tied(params))

    def pick(grid: list[dict[str, Any]]) -> int:
        chosen = ch.tune("lgbm", grid, a["X"], a["y"], a["groups"], train, 0, 0)[0]
        return int(chosen["id"])

    assert pick([{"noise": 0.0, "id": i} for i in range(3)]) == 0  # a three-way tie
    assert pick([{"noise": 50.0, "id": 0}, {"noise": 0.0, "id": 1}, {"noise": 50.0, "id": 2}]) == 1
    assert pick([{"noise": 0.0, "id": 0}, {"noise": 50.0, "id": 1}, {"noise": 0.0, "id": 2}]) == 0


class Inverted:
    """Scores every row with the opposite of its label, so every AUROC is exactly 0."""

    truth: np.ndarray
    lookup: dict[bytes, int]

    def fit(self, X: np.ndarray, y: np.ndarray) -> Inverted:
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = np.array([1.0 - self.truth[self.lookup[r.tobytes()]] for r in X])
        return np.column_stack([1 - p, p])


def test_a_configuration_scoring_exactly_zero_can_still_win(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    a, train, _ = fold_inputs(world)
    Inverted.truth, Inverted.lookup = a["y"], {r.tobytes(): i for i, r in enumerate(a["X"])}
    monkeypatch.setattr(ch, "make_estimator", lambda model, params, *x, **k: Inverted())
    params, score = ch.tune(
        "lgbm", [{"id": 0}, {"id": 1}], a["X"], a["y"], a["groups"], train, 0, 0
    )
    assert params == {"id": 0} and score == 0.0


def test_tuning_scores_the_pooled_inner_predictions(world: dict[str, Any]) -> None:
    a, train, _ = fold_inputs(world)
    params, score = ch.tune("b1", ch.grid_for("b1"), a["X"], a["y"], a["groups"], train, 0, 0)
    folds = cvs.inner_folds(train, a["y"], a["groups"], 0, 0)
    oof = np.full(len(a["y"]), np.nan)
    for fit, val in folds:
        oof[val] = (
            ch.make_estimator("b1", params)
            .fit(a["X"][fit], a["y"][fit])
            .predict_proba(a["X"][val])[:, 1]
        )
    assert score == pytest.approx(roc_auc_score(a["y"][train], oof[train]))


def test_inner_folds_inside_tuning_come_from_the_outer_training_part(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    a, train, _ = fold_inputs(world)
    seen: list[np.ndarray] = []
    real = ch.inner_folds

    def watch(tr: np.ndarray, *x: Any, **k: Any) -> Any:
        seen.append(np.asarray(tr))
        return real(tr, *x, **k)

    monkeypatch.setattr(ch, "inner_folds", watch)
    ch.tune("b1", ch.grid_for("b1"), a["X"], a["y"], a["groups"], train, 0, 0)
    assert len(seen) == 1 and sorted(seen[0]) == sorted(train)


# --------------------------------------------------------------------------- CLI guards


def test_b1_on_set_b_exits_1(world: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    assert run(world, "b1", "B") == 1
    assert (
        "b1 is pre-registered on Set A" in capsys.readouterr().err
        and not out_dir(world).joinpath("oof_b1_B.csv").exists()
    )


def test_b1_with_nan_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    frame = make_data()
    frame.loc[3, "tap_rate_hz"] = np.nan
    w = build_world(tmp_path, monkeypatch, frame)
    assert run(w, "b1") == 1 and "no NaN" in capsys.readouterr().err
    assert run(w, "lgbm") == 0  # lgbm handles NaN natively


def test_a_twin_column_in_the_feature_set_exits_1(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(ch.FEATURE_SETS, "A", [*cvs.SET_A_PRIMARY, "n_taps"])
    assert run(world, "lgbm") == 1 and not list(out_dir(world).glob("oof_*"))


def test_roles_hash_mismatch_exits_1(world: dict[str, Any]) -> None:
    roles = pd.read_csv(world["proc"] / "feature_roles.csv", dtype=str, keep_default_na=False)
    roles.loc[0, "note"] = "edited"
    roles.to_csv(world["proc"] / "feature_roles.csv", index=False)
    assert run(world, "b1") == 1


def test_tampered_splits_exit_1(world: dict[str, Any]) -> None:
    path = out_dir(world) / "splits.csv"
    lines = path.read_text().splitlines()
    first = lines[1].split(",")
    first[2] = str((int(first[2]) + 1) % 5)
    path.write_text("\n".join([lines[0], ",".join(first), *lines[2:]]) + "\n")
    assert run(world, "b1") == 1 and not list(out_dir(world).glob("oof_*"))


def test_more_repeats_than_the_splits_file_exit_1(world: dict[str, Any]) -> None:
    assert harness(world, "run", "--model", "b1", "--repeats", str(world["repeats"] + 1)) == 1
    assert harness(world, "run", "--model", "b1", "--repeats", "0") == 1


def test_seed_must_match_the_splits_file(world: dict[str, Any]) -> None:
    assert run(world, "b1", "A", "--seed", "1") == 1


def test_unknown_target_and_model_are_rejected_by_the_parser(world: dict[str, Any]) -> None:
    for extra in (["--target", "ge9", "--model", "b1"], ["--model", "svm"]):
        with pytest.raises(SystemExit) as err:
            harness(world, "run", *extra)
        assert err.value.code == 2


def test_a_missing_splits_file_exits_2(world: dict[str, Any]) -> None:
    assert harness(world, "run", "--model", "b1", "--target", "ge1", "--repeats", "1") == 2


def test_out_dir_outside_data_processed_cv_exits_2_and_writes_nothing(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    outside = world["tmp"] / "elsewhere"
    assert run(world, "b1", out=outside) == 2
    assert (
        "--out-dir must be inside data/processed/cv" in capsys.readouterr().err
        and not outside.exists()
    )
    assert (
        harness(world, "permute", "--model", "b1", "--draws", "1", out=outside) == 2
        and not outside.exists()
    )
    assert not list(out_dir(world).glob("oof_*"))


# --------------------------------------------------------------------------- outputs


@pytest.mark.parametrize("model,fs", [("b0", "A"), ("b1", "A"), ("lgbm", "A"), ("lgbm", "B")])
def test_outputs_files_rows_and_byte_identical_rerun(
    world: dict[str, Any], model: str, fs: str
) -> None:
    assert run(world, model, fs) == 0
    tag = f"{model}_{fs}"
    root = out_dir(world)
    assert sorted(p.name for p in root.iterdir()) == sorted(
        [
            "splits.csv",
            "splits_meta.json",
            f"oof_{tag}.csv",
            f"best_params_{tag}.csv",
            f"runs_{tag}.csv",
            f"meta_{tag}.json",
        ]
    )
    n = len(frame_of(world))
    oof = ch.read_oof(root / f"oof_{tag}.csv")
    assert list(oof.columns) == ["ref", "repeat", "outer_fold", "y", "proba"] and len(oof) == n * 2
    runs = pd.read_csv(root / f"runs_{tag}.csv")
    assert (
        list(runs.columns) == ch.RUN_COLUMNS and len(runs) == 2 and (runs["threshold"] == 0.5).all()
    )
    best = pd.read_csv(root / f"best_params_{tag}.csv")
    assert len(best) == 2 * 5 and list(best.columns) == [
        "repeat",
        "outer_fold",
        "params",
        "inner_auroc",
    ]
    first = snapshot(world["cv"])
    assert run(world, model, fs) == 0
    assert snapshot(world["cv"]) == first
    other = world["cv"] / "again"
    assert run(world, model, fs, out=other) == 0
    assert snapshot(other)[f"ge2/oof_{tag}.csv"] == first[f"ge2/oof_{tag}.csv"]
    assert snapshot(other)[f"ge2/meta_{tag}.json"] == first[f"ge2/meta_{tag}.json"]
    assert not list(world["cv"].rglob(".*tmp*"))


def test_a_clip_that_is_never_predicted_is_an_error(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real = ch.membership

    def broken(groups: np.ndarray, splits: pd.DataFrame, repeat: int) -> np.ndarray:
        member = real(groups, splits, repeat)
        idx = np.flatnonzero(member == 4)
        dropped = set(sorted(set(groups[idx]))[::2])  # whole participants, so the split stays valid
        member[[i for i in idx if groups[i] in dropped]] = 7  # in no fold, so never predicted
        return member

    monkeypatch.setattr(ch, "membership", broken)
    assert run(world, "b0") == 1
    assert "not predicted exactly once" in capsys.readouterr().err
    assert not list(out_dir(world).glob("oof_*"))


def test_a_probability_outside_zero_one_is_an_error(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Wild:
        def fit(self, X: np.ndarray, y: np.ndarray) -> Wild:
            return self

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return np.column_stack([np.full(len(X), -1.0), np.full(len(X), 2.0)])

    monkeypatch.setattr(ch, "make_estimator", lambda *a, **k: Wild())
    assert run(world, "b0") == 1
    assert "outside [0, 1]" in capsys.readouterr().err


def test_b0_is_not_tuned_and_fits_once_per_outer_fold(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    log = install_spy(monkeypatch, arrays(world)["X"])
    assert run(world, "b0", "A", "--repeats", "1") == 0
    assert [e[0] for e in log].count("fit") == 5  # no inner fits
    best = pd.read_csv(out_dir(world) / "best_params_b0_A.csv")
    assert best["inner_auroc"].isna().all() and (best["params"] == "{}").all()


def test_b0_ignores_the_feature_set(world: dict[str, Any]) -> None:
    assert run(world, "b0", "B") == 0
    assert (out_dir(world) / "meta_b0_A.json").is_file() and not (
        out_dir(world) / "meta_b0_B.json"
    ).exists()


def test_every_clip_is_predicted_once_per_repeat_with_valid_probabilities(
    world: dict[str, Any],
) -> None:
    assert run(world, "lgbm", "B") == 0
    a = arrays(world, "B")
    oof = ch.read_oof(out_dir(world) / "oof_lgbm_B.csv")
    assert (oof.groupby(["repeat", "ref"]).size() == 1).all() and set(oof["ref"]) == set(
        a["frame"]["ref"]
    )
    assert np.isfinite(oof["proba"]).all() and oof["proba"].between(0, 1).all()
    assert (oof.groupby("repeat").size() == len(a["frame"])).all()
    by_ref = a["frame"].set_index("ref")
    assert (oof["y"].to_numpy() == (by_ref.loc[oof["ref"], "rating"].to_numpy() >= 2)).all()
    for repeat, part in oof.groupby("repeat"):
        expected = ch.membership(a["groups"], a["splits"], int(repeat))
        got = part.set_index("ref")["outer_fold"].reindex(a["frame"]["ref"]).to_numpy()
        assert (got == expected).all()  # the folds come from the saved splits of that repeat
    f0 = oof[oof["repeat"] == 0].set_index("ref")["outer_fold"]
    f1 = oof[oof["repeat"] == 1].set_index("ref")["outer_fold"].reindex(f0.index)
    assert not f0.equals(f1)  # repeats use different folds


def test_models_a_and_b_share_the_folds(world: dict[str, Any]) -> None:
    assert run(world, "lgbm", "A") == 0 and run(world, "lgbm", "B") == 0
    meta_a = json.loads((out_dir(world) / "meta_lgbm_A.json").read_text())
    meta_b = json.loads((out_dir(world) / "meta_lgbm_B.json").read_text())
    assert ch.same_folds(meta_a, meta_b)
    splits_meta = json.loads((out_dir(world) / "splits_meta.json").read_text())
    assert meta_a["splits_sha8"] == splits_meta["splits_sha8"] == meta_b["splits_sha8"]
    assert (
        meta_a["inner_sha8"] == meta_b["inner_sha8"] and meta_a["grid_sha8"] == meta_b["grid_sha8"]
    )
    oof_a = ch.read_oof(out_dir(world) / "oof_lgbm_A.csv")
    oof_b = ch.read_oof(out_dir(world) / "oof_lgbm_B.csv")
    assert oof_a[["ref", "repeat", "outer_fold", "y"]].equals(
        oof_b[["ref", "repeat", "outer_fold", "y"]]
    )
    assert not ch.same_folds(meta_a, {**meta_b, "inner_sha8": "00000000"})
    assert not ch.same_folds(meta_a, {**meta_b, "splits_sha8": "00000000"})
    assert meta_a["features_sha8"] != meta_b["features_sha8"]


def test_meta_contents(world: dict[str, Any]) -> None:
    assert run(world, "lgbm", "B") == 0
    meta = json.loads((out_dir(world) / "meta_lgbm_B.json").read_text())
    roles = json.loads((world["proc"] / "features_meta.json").read_text())["roles_sha8"]
    assert meta["target"] == "ge2" and meta["target_status"] == "pending_default_ge2"
    assert meta["roles_sha8"] == roles and meta["columns"] == ch.FEATURE_SETS["B"]
    assert meta["features_sha8"] == cvs.sha8_of(json.dumps(ch.FEATURE_SETS["B"]))
    assert meta["constants"]["threshold"] == 0.5 and meta["constants"]["lgbm_seed"] == 20260
    assert meta["grid"] == SMALL_LGBM and set(meta["versions"]) == {
        "python",
        "scikit_learn",
        "lightgbm",
        "numpy",
        "pandas",
    }
    assert set(meta["inputs_sha8"]) == {"features", "roles", "features_meta", "splits"}
    assert not re.search(r"20\d\d-\d\d-\d\d|T\d\d:\d\d", json.dumps(meta))


@pytest.mark.parametrize("model,fs", [("b0", "A"), ("b1", "A"), ("lgbm", "B")])
def test_runs_match_an_independent_recomputation_at_the_fixed_threshold(
    world: dict[str, Any], model: str, fs: str
) -> None:
    assert run(world, model, fs) == 0
    oof = ch.read_oof(out_dir(world) / f"oof_{model}_{fs}.csv")
    runs = pd.read_csv(out_dir(world) / f"runs_{model}_{fs}.csv")
    for repeat, part in oof.groupby("repeat"):
        y, p = part["y"].to_numpy(), part["proba"].to_numpy()
        row = runs[runs["repeat"] == repeat].iloc[0]
        pred = p >= 0.5
        tp, fn = (pred & (y == 1)).sum(), (~pred & (y == 1)).sum()
        tn, fp = (~pred & (y == 0)).sum(), (pred & (y == 0)).sum()
        sens, spec = tp / (tp + fn), tn / (tn + fp)
        assert row["auroc"] == pytest.approx(roc_auc_score(y, p))
        folds = [roc_auc_score(g["y"], g["proba"]) for _, g in part.groupby("outer_fold")]
        assert row["auroc_fold_mean"] == pytest.approx(np.mean(folds))
        assert row["auroc_fold_sd"] == pytest.approx(np.std(folds, ddof=1))
        assert row["sensitivity"] == pytest.approx(sens) and row["specificity"] == pytest.approx(
            spec
        )
        assert row["balanced_accuracy"] == pytest.approx((sens + spec) / 2)
        denominator = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        assert row["mcc"] == pytest.approx(
            (tp * tn - fp * fn) / denominator if denominator else 0.0
        )
        assert row["n_pred_pos"] == tp + fp
        for pct, prevalence in ((1, 0.01), (2, 0.02), (5, 0.05)):
            den = sens * prevalence + (1 - spec) * (1 - prevalence)
            expected = sens * prevalence / den if den else np.nan
            assert row[f"ppv_at_{pct}pct"] == pytest.approx(expected) or (
                np.isnan(expected) and np.isnan(row[f"ppv_at_{pct}pct"])
            )
        assert (
            row["n_clips"] == len(y) and row["n_pos"] == y.sum() and row["n_neg"] == (1 - y).sum()
        )


@pytest.mark.parametrize("model,fs", [("b1", "A"), ("lgbm", "B")])
def test_refit_reproduces_the_stored_fold_predictions_bit_for_bit(
    world: dict[str, Any], model: str, fs: str
) -> None:
    assert run(world, model, fs) == 0
    a = arrays(world, fs)
    oof = ch.read_oof(out_dir(world) / f"oof_{model}_{fs}.csv")
    best = pd.read_csv(out_dir(world) / f"best_params_{model}_{fs}.csv")
    for repeat in range(world["repeats"]):
        member = ch.membership(a["groups"], a["splits"], repeat)
        stored = (
            oof[oof["repeat"] == repeat]
            .set_index("ref")["proba"]
            .reindex(a["frame"]["ref"])
            .to_numpy()
        )
        for fold in range(5):
            row = best[(best["repeat"] == repeat) & (best["outer_fold"] == fold)].iloc[0]
            train, test = np.flatnonzero(member != fold), np.flatnonzero(member == fold)
            est = ch.refit_outer(a["frame"], model, json.loads(row["params"]), train, fs, "ge2")
            assert np.array_equal(est.predict_proba(a["X"][test])[:, 1], stored[test])


def test_overwrite_with_different_content_needs_force(world: dict[str, Any]) -> None:
    assert run(world, "b1") == 0
    before = snapshot(world["cv"])
    assert harness(world, "run", "--model", "b1", "--repeats", "1") == 1
    assert snapshot(world["cv"]) == before
    assert harness(world, "run", "--model", "b1", "--repeats", "1", "--force") == 0
    assert snapshot(world["cv"]) != before


def test_no_raw_id_or_label_text_in_any_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    frame = make_data().assign(
        note_raw="CONTROL07_DCHA UPDRS_POISON",
        hand=lambda d: np.where(d["hand"] == "right", "IZDA_POISON", "DCHA_POISON"),
    )
    w = build_world(tmp_path, monkeypatch, frame)
    for model in ("b1", "lgbm"):
        assert run(w, model, "A", "--show-metrics") == 0
    assert harness(w, "permute", "--model", "b0", "--draws", "2") == 0
    blob = capsys.readouterr().out + capsys.readouterr().err
    for p in w["cv"].rglob("*"):
        if p.is_file():
            blob += p.read_text()
    for needle in ("CONTROL", "DCHA", "IZDA", "UPDRS", "POISON", "ID1004", "hand"):
        assert needle not in blob
    assert not re.search(r"20\d\d-\d\d-\d\d|T\d\d:\d\d:\d\d", blob)


def test_the_label_file_is_never_opened(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []
    real_open = builtins.open

    def spy(file: Any, *args: Any, **kwargs: Any) -> Any:
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(io, "open", spy)
    assert run(world, "lgbm") == 0
    assert harness(world, "permute", "--model", "b0", "--draws", "1") == 0
    assert any(p.endswith("features.csv") for p in opened)
    assert not [
        p for p in opened if "fis_diagnostic" in p or "HUBU-FIS" in p or "raw" in Path(p).parts
    ]


def test_source_has_no_label_file_name_or_forbidden_imports() -> None:
    source = (SRC / "cv_harness.py").read_text()
    assert "fis_diagnostic" not in source and "UPDRS" not in source
    imports = re.findall(r"^\s*(?:from|import)\s+([\w.]+)", source, flags=re.M)
    assert not {"feature_matrix", "hubu_fis", "gait"} & {i.split(".")[0] for i in imports}


def test_no_metric_is_printed_without_show_metrics(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(world, "lgbm", "B") == 0
    captured = capsys.readouterr()
    assert not METRIC_WORDS.search(captured.out + captured.err)
    assert (
        "splits_sha8" in captured.out
        and "runs rows 2" in captured.out
        and "oof rows" in captured.out
    )
    value = pd.read_csv(out_dir(world) / "runs_lgbm_B.csv")["auroc"].mean()
    assert f"{value:.4f}" not in captured.out
    assert run(world, "lgbm", "B", "--show-metrics") == 0
    shown = capsys.readouterr().out
    assert "auroc: mean" in shown and "balanced_accuracy: mean" in shown


# --------------------------------------------------------------------------- planted signal and leaks


def pooled_auroc(w: dict[str, Any], model: str, fs: str) -> float:
    assert run(w, model, fs) == 0
    return float(pd.read_csv(out_dir(w) / f"runs_{model}_{fs}.csv")["auroc"].mean())


def test_planted_signal_is_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    w = build_world(tmp_path, monkeypatch, make_data("signal"))
    assert pooled_auroc(w, "lgbm", "B") > 0.75


def test_pure_noise_stays_near_chance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    w = build_world(tmp_path, monkeypatch, make_data("noise", seed=4))
    assert 0.35 < pooled_auroc(w, "lgbm", "B") < 0.65


def test_a_planted_identity_leak_is_not_exploited_and_the_permutation_control_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = make_data("identity", n_part=100, seed=2)
    X = frame[cvs.MODEL_COLUMNS[:5]].to_numpy()
    y = (frame["rating"].to_numpy() >= 2).astype(int)
    row_level = cross_val_predict(
        KNeighborsClassifier(1),
        X,
        y,
        cv=StratifiedKFold(5, shuffle=True, random_state=0),
        method="predict_proba",
    )[:, 1]
    assert roc_auc_score(y, row_level) > 0.8  # the leak is real for a row-level split
    w = build_world(tmp_path, monkeypatch, frame)
    assert pooled_auroc(w, "lgbm", "B") < 0.65  # participant-grouped folds do not exploit it
    assert harness(w, "permute", "--model", "lgbm", "--feature-set", "B", "--draws", "4") == 0


# --------------------------------------------------------------------------- permutation


def vectors(frame: pd.DataFrame, y: np.ndarray) -> list[tuple[int, ...]]:
    """Per-participant label vectors ordered by (hand, ref), sorted as a multiset."""
    out = []
    for _, part in frame.assign(y=y).sort_values(["hand", "ref"]).groupby("participant"):
        out.append(tuple(int(v) for v in part["y"]))
    return sorted(out)


def pattern_frame() -> tuple[pd.DataFrame, np.ndarray]:
    pats = [(0, 1)] * 4 + [(1, 0)] * 4 + [(0, 0)] * 2 + [(1, 1)] * 2 + [(1,)]
    rows = []
    for i, pat in enumerate(pats):
        for c, v in enumerate(pat):
            rows.append((hid("pc", 10 * i + c), hid("pp", i), "left" if c == 0 else "right", v))
    frame = pd.DataFrame(rows, columns=["ref", "participant", "hand", "y"])
    return frame, frame["y"].to_numpy()


def test_permutation_moves_whole_participants_and_keeps_the_structure() -> None:
    frame, y = pattern_frame()
    changed = False
    for d in range(10):
        out = ch.permute_labels(frame, y, np.random.default_rng(d))
        assert vectors(frame, out) == vectors(frame, y)  # same multiset of ordered vectors
        assert out.sum() == y.sum()
        single = frame.index[frame["participant"] == hid("pp", 12)]
        assert (out[single] == y[single]).all()  # the lone singleton stays fixed
        changed |= not np.array_equal(out, y)
    assert changed


def test_participants_of_every_size_class_are_shuffled_within_their_class() -> None:
    sizes = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    rows, y = [], []
    for i, n in enumerate(sizes):
        for c in range(n):
            rows.append((hid("sc", 10 * i + c), hid("sp", i), f"h{c}"))
            y.append(int((i * 7 + c * 3) % 2 == 0 or i % 4 == 0))
    frame = pd.DataFrame(rows, columns=["ref", "participant", "hand"])
    y_arr = np.array(y)
    moved = {1: False, 2: False, 3: False}
    for d in range(12):
        out = ch.permute_labels(frame, y_arr, np.random.default_rng(d))
        assert vectors(frame, out) == vectors(frame, y_arr)
        for size in moved:
            idx = frame.index[
                frame["participant"].isin([hid("sp", i) for i, n in enumerate(sizes) if n == size])
            ]
            assert out[idx].sum() == y_arr[idx].sum()  # nothing crosses between size classes
            moved[size] |= not np.array_equal(out[idx], y_arr[idx])
    assert all(moved.values())


def test_permutation_is_deterministic_and_differs_across_draws() -> None:
    frame, y = pattern_frame()
    a = ch.permute_labels(frame, y, np.random.default_rng(5))
    assert np.array_equal(a, ch.permute_labels(frame, y, np.random.default_rng(5)))
    assert any(
        not np.array_equal(a, ch.permute_labels(frame, y, np.random.default_rng(s)))
        for s in range(6, 12)
    )


def test_permutation_in_the_cli_uses_the_seeded_draws_and_recomputes_the_inner_folds(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    a = arrays(world)
    perms: list[np.ndarray] = []
    real_permute, real_inner = ch.permute_labels, ch.inner_folds
    inner_y: list[np.ndarray] = []

    def spy_permute(f: pd.DataFrame, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        perms.append(real_permute(f, y, rng))
        return perms[-1]

    def spy_inner(tr: np.ndarray, y: np.ndarray, g: np.ndarray, r: int, o: int) -> Any:
        inner_y.append(y.copy())
        return real_inner(tr, y, g, r, o)

    monkeypatch.setattr(ch, "permute_labels", spy_permute)
    monkeypatch.setattr(ch, "inner_folds", spy_inner)
    assert harness(world, "permute", "--model", "b1", "--draws", "3", "--seed", "0") == 0
    assert len(perms) == 3 and not np.array_equal(perms[0], perms[1])
    for d, got in enumerate(perms):
        assert np.array_equal(got, real_permute(a["frame"], a["y"], np.random.default_rng(d)))
        assert vectors(a["frame"], got) == vectors(a["frame"], a["y"])
        assert all(
            np.array_equal(v, got) for v in inner_y[5 * d : 5 * d + 5]
        )  # inner folds see the permuted target
    root = out_dir(world)
    table = pd.read_csv(root / "permutation_b1_A.csv")
    summary = json.loads((root / "permutation_b1_A.json").read_text())
    assert list(table.columns) == ["draw", "null_auroc"] and len(table) == 3
    assert summary["mean"] == pytest.approx(table["null_auroc"].mean()) and summary["decision"] in {
        "pass",
        "fail",
    }
    assert summary["max_allowed_mean"] == 0.55 and summary["min"] == pytest.approx(
        table["null_auroc"].min()
    )
    first = snapshot(root)
    assert harness(world, "permute", "--model", "b1", "--draws", "3", "--seed", "0") == 0
    assert snapshot(root) == first  # byte-identical rerun


class Oracle:
    """A leaking estimator: it reads the (unpermuted) labels of the rows it is asked about."""

    truth: np.ndarray
    lookup: dict[bytes, int]

    def fit(self, X: np.ndarray, y: np.ndarray) -> Oracle:
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = np.array([self.truth[self.lookup[r.tobytes()]] for r in X], dtype=float)
        return np.column_stack([1 - p, p])


def test_the_permutation_decision_fails_the_run_when_the_null_is_above_chance(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    a = arrays(world)
    Oracle.truth, Oracle.lookup = a["y"], {r.tobytes(): i for i, r in enumerate(a["X"])}
    monkeypatch.setattr(ch, "make_estimator", lambda model, params, *x, **k: Oracle())
    monkeypatch.setattr(
        ch, "permute_labels", lambda f, y, rng: y
    )  # a pipeline that leaks the label
    assert harness(world, "permute", "--model", "b1", "--draws", "2") == 1
    captured = capsys.readouterr()
    assert "permutation control failed" in captured.err and "decision fail" in captured.out
    summary = json.loads((out_dir(world) / "permutation_b1_A.json").read_text())
    assert summary["decision"] == "fail" and summary["mean"] > 0.55


def test_the_permutation_decision_passes_on_a_leak_free_pipeline(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert harness(world, "permute", "--model", "b0", "--draws", "2") == 0
    assert "decision pass" in capsys.readouterr().out
    summary = json.loads((out_dir(world) / "permutation_b0_A.json").read_text())
    assert summary["decision"] == "pass" and summary["mean"] <= 0.55
