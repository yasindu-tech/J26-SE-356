"""Tests for models/tapping/demo/train_demo_model.py and resources.py (task FT-47). Synthetic
frames and worlds only; the pre-registered mode rule is checked by hand, including a tie."""

from __future__ import annotations

import json
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
import lightgbm  # noqa: E402
import resources as res  # noqa: E402
import train_demo_model as tdm  # noqa: E402
import treeshap_heldout as tsh  # noqa: E402
from cv_splits import GuardFailure  # noqa: E402

COLS = list(ch.FEATURE_SETS["A"])
PARAMS = {"num_leaves": 3, "n_estimators": 10, "min_child_samples": 5}
GRID = ch.grid_for("lgbm")
N = 40


def frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(N, len(COLS)))
    out = pd.DataFrame(x, columns=COLS)
    out["ref"] = [f"{i:08x}" for i in range(N)]
    out["rating"] = (x[:, 0] + 0.5 * rng.normal(size=N) > 0).astype(int) * 2
    return out


def best_with(counts: dict[int, int]) -> dict[tuple[int, int], dict[str, Any]]:
    flat = [i for i, n in counts.items() for _ in range(n)]
    assert len(flat) == 25
    return {(k // 5, k % 5): GRID[i] for k, i in enumerate(flat)}


def files_for(target: str = "ge2", seed: int = 0) -> dict[str, bytes]:
    fr = frame(seed)
    est = tdm.fit_all(fr, target, PARAMS)
    return tdm.model_files(est, target, 0, [25] + [0] * (len(GRID) - 1), GRID, {"features": "x"}, N)


def add_model(proc: Path, target: str = "ge2", **meta_edit: Any) -> Path:
    """A valid synthetic demo model under proc/demo (features.csv gets arbitrary bytes)."""
    (proc / "features.csv").write_text("synthetic\n")
    inputs = {"features": cvs.file_sha256(proc / "features.csv")[: cvs.HASH_LEN]}
    fr = frame()
    est = tdm.fit_all(fr, target, PARAMS)
    files = tdm.model_files(est, target, 0, [25] + [0] * (len(GRID) - 1), GRID, inputs, N)
    folder = proc / "demo"
    folder.mkdir(parents=True, exist_ok=True)
    meta_name = res.META_FILE.format(target=target)
    for name, data in files.items():
        (folder / name).write_bytes(data)
    if meta_edit:
        meta = json.loads((folder / meta_name).read_text())
        meta.update(meta_edit)
        (folder / meta_name).write_text(json.dumps(meta))
    return folder


# --------------------------------------------------------------------------- the mode rule


def test_mode_rule_by_hand() -> None:
    idx, counts = tdm.choose_params(best_with({4: 9, 7: 8, 20: 8}), GRID)
    assert idx == 4 and counts[4] == 9 and counts[7] == 8 and counts[20] == 8 and sum(counts) == 25


def test_mode_rule_not_the_first_folds_params() -> None:
    best = best_with({6: 3, 11: 12, 2: 10})
    first = GRID.index(best[(0, 0)])
    idx, _ = tdm.choose_params(best, GRID)
    assert idx == 11 and first == 6  # the first fold's configuration is not the mode


def test_tie_goes_to_the_simplest_first_entry() -> None:
    best = best_with({13: 10, 5: 10, 20: 5})  # grid order: 5 is simpler than 13
    assert tdm.choose_params(best, GRID)[0] == 5
    best = best_with({26: 12, 0: 12, 3: 1})
    assert tdm.choose_params(best, GRID)[0] == 0


# --------------------------------------------------------------------------- the model


def test_saved_model_equals_a_fresh_fit_on_all_clips(tmp_path: Path) -> None:
    fr = frame()
    files = tdm.model_files(
        tdm.fit_all(fr, "ge2", PARAMS), "ge2", 0, [25] + [0] * 26, GRID, {"features": "x"}, N
    )
    path = tmp_path / "m.txt"
    path.write_bytes(files[res.MODEL_FILE.format(target="ge2")])
    booster = lightgbm.Booster(model_file=str(path))
    x, _ = ch.design_matrix(fr, "A")
    fresh = ch.make_estimator("lgbm", PARAMS).fit(x, cvs.target_labels(fr, "ge2"))
    assert np.max(np.abs(booster.predict(x) - fresh.predict_proba(x)[:, 1])) < 1e-12
    subset = ch.make_estimator("lgbm", PARAMS).fit(x[:30], cvs.target_labels(fr, "ge2")[:30])
    assert np.max(np.abs(booster.predict(x) - subset.predict_proba(x)[:, 1])) > 1e-6


def test_model_bytes_are_deterministic() -> None:
    assert files_for() == files_for()
    assert files_for("ge2") != files_for("ge1")


def test_meta_content() -> None:
    meta = json.loads(files_for()[res.META_FILE.format(target="ge2")])
    assert meta["note"] == "Demo model; not used for any reported metric"
    assert meta["params"] == GRID[0] and meta["grid_index"] == 0
    assert meta["n_clips"] == N and meta["n_best_params"] == 25
    assert meta["config_counts"] == [{"grid_index": 0, "params": GRID[0], "n_folds": 25}]
    assert meta["lightgbm"] == lightgbm.__version__
    assert len(meta["model_sha8"]) == 8 and meta["grid_sha8"] == ch.grid_sha8("lgbm", GRID)


# --------------------------------------------------------------------------- run_target


def world(tmp_path: Path, best_counts: dict[int, int], target: str = "ge2") -> tuple[Path, Path]:
    cv_root = tmp_path / "cv"
    (cv_root / target).mkdir(parents=True)
    (cv_root / target / "meta_lgbm_A.json").write_text("{}")
    rows = [{"repeat": k[0], "outer_fold": k[1], "params": json.dumps(p)}
            for k, p in best_with(best_counts).items()]  # fmt: skip
    pd.DataFrame(rows).assign(inner_auroc=0.8).to_csv(
        cv_root / target / "best_params_lgbm_A.csv", index=False
    )
    return cv_root, tmp_path / "demo"


def test_run_target_writes_the_mode_model_and_is_repeatable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(tsh, "check_official", lambda *a, **k: calls.append("checked"))
    cv_root, out = world(tmp_path, {0: 8, 1: 8, 14: 9})

    def spy_fit(f: pd.DataFrame, t: str, p: dict[str, Any]) -> Any:
        calls.append(json.dumps(p))
        return _fit(f, t, p)

    monkeypatch.setattr(tdm, "fit_all", spy_fit)
    first = tdm.run_target(cv_root, out, frame(), pd.DataFrame(), {"features": "a"}, "ge2")
    snap = {p.name: p.read_bytes() for p in out.iterdir()}
    second = tdm.run_target(cv_root, out, frame(), pd.DataFrame(), {"features": "a"}, "ge2")
    assert calls[0] == "checked" and json.loads(calls[1]) == GRID[14]
    assert first["sha8"] == second["sha8"] and snap == {
        p.name: p.read_bytes() for p in out.iterdir()
    }
    assert first["counts"] == {0: 8, 1: 8, 14: 9}
    assert sorted(snap) == ["model_ge2_A.txt", "model_ge2_A_meta.json"]


def test_run_target_names_the_files_by_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tsh, "check_official", lambda *a, **k: None)
    monkeypatch.setattr(tdm, "fit_all", lambda f, t, p: _fit(f, t, p))
    cv_root, out = world(tmp_path, {2: 25}, "ge1")
    done = tdm.run_target(cv_root, out, frame(), pd.DataFrame(), {"features": "a"}, "ge1")
    assert sorted(p.name for p in out.iterdir()) == ["model_ge1_A.txt", "model_ge1_A_meta.json"]
    assert json.loads((out / "model_ge1_A_meta.json").read_text())["target"] == "ge1"
    assert done["params"] == GRID[2]


def _fit(f: pd.DataFrame, t: str, p: dict[str, Any]) -> Any:
    x, _ = ch.design_matrix(f, "A")
    return ch.make_estimator("lgbm", PARAMS).fit(x, cvs.target_labels(f, t))


def test_run_target_refuses_when_the_official_guard_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*a: Any, **k: Any) -> None:
        raise GuardFailure("grid_sha8 differs from the official run")

    monkeypatch.setattr(tsh, "check_official", boom)
    cv_root, out = world(tmp_path, {0: 25})
    with pytest.raises(GuardFailure, match="grid_sha8"):
        tdm.run_target(cv_root, out, frame(), pd.DataFrame(), {"features": "a"}, "ge2")
    assert not out.exists()


def test_run_target_refuses_params_outside_the_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tsh, "check_official", lambda *a, **k: None)
    cv_root, out = world(tmp_path, {0: 25})
    path = cv_root / "ge2" / "best_params_lgbm_A.csv"
    path.write_text(path.read_text().replace('""num_leaves"": 3', '""num_leaves"": 4', 1))
    with pytest.raises(GuardFailure, match="outside the grid"):
        tdm.run_target(cv_root, out, frame(), pd.DataFrame(), {"features": "a"}, "ge2")


# --------------------------------------------------------------------------- resources


def test_load_demo_model_ok(tmp_path: Path) -> None:
    add_model(tmp_path)
    booster = res.load_demo_model("ge2", tmp_path)
    assert booster.num_feature() == len(COLS)
    assert len(res.model_sha8("ge2", tmp_path)) == 8


@pytest.mark.parametrize(
    "edit",
    [{"features_sha8": "00000000"}, {"grid_sha8": "00000000"}, {"lightgbm": "0.0.0"},
     {"model_sha8": "00000000"}, {"tool": "other"}, {"inputs_sha8": {"features": "00000000"}}],
)  # fmt: skip
def test_meta_mismatch_is_refused(tmp_path: Path, edit: dict[str, Any]) -> None:
    add_model(tmp_path, **edit)
    with pytest.raises(res.DemoModelError, match="retrain"):
        res.load_demo_model("ge2", tmp_path)


def test_changed_model_file_or_features_file_is_refused(tmp_path: Path) -> None:
    folder = add_model(tmp_path)
    (tmp_path / "features.csv").write_text("changed\n")
    with pytest.raises(res.DemoModelError, match="features file changed"):
        res.load_demo_model("ge2", tmp_path)
    (tmp_path / "features.csv").write_text("synthetic\n")
    model = folder / "model_ge2_A.txt"
    model.write_bytes(model.read_bytes() + b"\n")
    with pytest.raises(res.DemoModelError, match="model_sha8"):
        res.load_demo_model("ge2", tmp_path)


def test_missing_model_is_a_friendly_error(tmp_path: Path) -> None:
    with pytest.raises(res.DemoModelError, match="train_demo_model.py"):
        res.model_sha8("ge2", tmp_path)
    with pytest.raises(da.DemoDataError):
        res.load_demo_model("ge2", tmp_path)


def test_landmarker_handle_checks_the_model_file(tmp_path: Path) -> None:
    with pytest.raises(da.DemoDataError, match="not found"):
        res.make_landmarker(tmp_path / "none.task")
    bad = tmp_path / "bad.task"
    bad.write_bytes(b"not the model")
    with pytest.raises(da.DemoDataError, match="not the expected"):
        res.make_landmarker(bad)
    import hand_landmarks as hl

    if hl.DEFAULT_MODEL.is_file():  # the asset is gitignored; skip where it is absent
        handle = res.make_landmarker()
        assert handle.sha256 == hl.EXPECTED_MODEL_SHA256 and callable(handle)
