"""Tests for models/tapping/src/reference_baseline.py (task FT-40). The official side (splits, lgbm A
and b1 A runs) comes from the real harness on a synthetic matrix; the reference table is synthetic.
No real data is used."""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import re
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import baseline_report as br  # noqa: E402
import bootstrap_ci as bc  # noqa: E402
import cv_harness as ch  # noqa: E402
import cv_splits as cvs  # noqa: E402
import reference_baseline as rb  # noqa: E402
import reference_features as rf  # noqa: E402
from test_baseline_report import edit_json, edit_runs  # noqa: E402
from test_cv_harness import SMALL_LGBM, build_world, harness, make_data  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src"
HASH = re.compile(r"\b[0-9a-f]{8}\b")
TARGETS = ("ge2", "ge1")


# --------------------------------------------------------------------------- the worlds


def write_reference(
    directory: Path, frame: pd.DataFrame, nan_columns: dict[str, int] | None = None, seed: int = 0
) -> None:
    """A synthetic reference table with the registered columns (the real one is built from landmarks)."""
    rng = np.random.default_rng(seed)
    table = pd.DataFrame(rng.normal(size=(len(frame), 27)), columns=rf.REFERENCE_COLUMNS)
    table.insert(0, "participant", frame["participant"].to_numpy())
    table.insert(0, "ref", frame["ref"].to_numpy())
    for column, count in (nan_columns or {}).items():
        table.loc[: count - 1, column] = np.nan
    directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(directory / rf.OUT_FEATURES, index=False, float_format="%.17g")
    rf.roles_table().to_csv(directory / rf.OUT_ROLES, index=False)
    meta = {
        "reference_roles_sha8": rf.PINNED_ROLES_SHA8,
        "columns": rf.REFERENCE_COLUMNS,
        "features_sha8": rf.PINNED_FEATURES_SHA8,
        "source": rf.SOURCE,
    }
    (directory / rf.OUT_META).write_text(json.dumps(meta))


@pytest.fixture(scope="module")
def official(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    patch = pytest.MonkeyPatch()
    patch.setitem(ch.GRIDS, "lgbm", [dict(p) for p in SMALL_LGBM])
    w = build_world(
        tmp_path_factory.mktemp("official"), patch, make_data("signal", seed=5), repeats=5
    )
    proc = w["proc"]
    assert cvs.main(["--features", str(proc / "features.csv"), "--roles", str(proc / "feature_roles.csv"),
                     "--meta", str(proc / "features_meta.json"), "--target", "ge1", "--repeats", "5",
                     "--out-dir", str(w["cv"])]) == 0  # fmt: skip
    for target in TARGETS:
        for model in ("b1", "lgbm"):
            assert harness(w, "run", "--model", model, "--target", target, "--repeats", "5") == 0
    frame = cvs.canonical(
        pd.read_csv(proc / "features.csv", dtype={"ref": str, "participant": str})
    )
    write_reference(proc / "reference", frame)
    w["frame"] = frame
    yield w
    patch.undo()


@pytest.fixture
def root(
    official: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    cv_copy = tmp_path / "data" / "processed" / "cv"
    shutil.copytree(official["cv"], cv_copy)
    ref_copy = tmp_path / "data" / "processed" / "reference"
    shutil.copytree(official["proc"] / "reference", ref_copy)
    monkeypatch.setattr(cvs, "CV_ROOT", cv_copy)
    monkeypatch.setattr(bc, "N_DRAWS", 60)
    return {
        "cv": cv_copy,
        "ref": ref_copy,
        "proc": official["proc"],
        "frame": official["frame"],
        "tmp": tmp_path,
    }


def cli(w: dict[str, Any], command: str, *extra: str) -> int:
    base = [
        "--features",
        str(w["proc"] / "features.csv"),
        "--roles",
        str(w["proc"] / "feature_roles.csv"),
        "--meta",
        str(w["proc"] / "features_meta.json"),
        "--cv-root",
        str(w["cv"]),
    ]
    if command == "report":
        return rb.main(
            [
                command,
                "--features",
                str(w["proc"] / "features.csv"),
                "--cv-root",
                str(w["cv"]),
                "--reference-dir",
                str(w["ref"]),
                *extra,
            ]
        )
    return rb.main([command, *base, "--reference-dir", str(w["ref"]), *extra])


def snapshot(folder: Path, names: list[str] | None = None) -> dict[str, str]:
    files = [
        p for p in sorted(folder.rglob("*")) if p.is_file() and (names is None or p.name in names)
    ]
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def r_files(w: dict[str, Any], target: str = "ge2") -> list[str]:
    return sorted(
        p.name
        for p in (w["cv"] / target).iterdir()
        if p.name.endswith("_R.csv") or p.name.endswith("_R.json")
    )


# --------------------------------------------------------------------------- invariants


def test_the_existing_grids_and_feature_sets_are_untouched() -> None:
    real = ch.real_grids()
    assert (
        ch.grid_sha8("b0", real["b0"]) == "7e846e65"
        and ch.grid_sha8("b1", real["b1"]) == "b7c060fb"
    )
    assert ch.grid_sha8("lgbm", real["lgbm"]) == "a2670eaf" and len(real["lgbm"]) == 27
    assert set(ch.FEATURE_SETS) == {
        "A",
        "B",
    }  # this module does not register a third set in the harness
    assert len(ch.FEATURE_SETS["A"]) == 5 and len(ch.FEATURE_SETS["B"]) == 13
    assert (
        cvs.sha8_of(json.dumps(ch.FEATURE_SETS["A"])) == "fc420d98"
        and cvs.sha8_of(json.dumps(ch.FEATURE_SETS["B"])) == "8219222e"
    )
    assert (
        rb.MODELS == ("b1", "lgbm")
        and rb.FEATURE_SET == "R"
        and rb.N_REPEATS == 5
        and rb.OFFICIAL_SET == "A"
    )


def test_the_harness_refuses_the_reference_columns_so_this_module_is_the_only_route() -> None:
    frame = pd.DataFrame(np.zeros((3, 27)), columns=rf.REFERENCE_COLUMNS)
    with pytest.raises(cvs.ContractError):
        cvs.validate_inputs(
            frame, rf.REFERENCE_COLUMNS
        )  # not feature_primary: the contract validator stays strict


def test_the_run_reproduces_the_harness_bit_for_bit_when_given_the_same_matrix(
    root: dict[str, Any],
) -> None:
    ns = rb.argparse.Namespace(
        cv_root=root["cv"],
        features=root["proc"] / "features.csv",
        roles=root["proc"] / "feature_roles.csv",
        meta=root["proc"] / "features_meta.json",
        reference_dir=root["ref"],
        target="ge2",
        model="lgbm",
        seed=0,
    )
    ctx = rb.prepare(ns)
    frame = ctx.frame
    ctx.X = (
        frame[ch.FEATURE_SETS["A"]].astype("float64").to_numpy()
    )  # the official Set A matrix instead of R
    files = rb.run_files("lgbm", "ge2", 0, ctx)
    for name in ("oof", "best_params"):
        assert (
            files[f"{name}_lgbm_R.csv"] == (root["cv"] / "ge2" / f"{name}_lgbm_A.csv").read_bytes()
        )  # same code path, same bytes
    ours = pd.read_csv(root["cv"] / "ge2" / "runs_lgbm_A.csv")
    mine = pd.read_csv(io.BytesIO(files["runs_lgbm_R.csv"]))
    drop = ["feature_set", "n_features", "features_sha8"]
    pd.testing.assert_frame_equal(ours.drop(columns=drop), mine.drop(columns=drop))


# --------------------------------------------------------------------------- run


@pytest.mark.parametrize("model", ["b1", "lgbm"])
def test_run_writes_only_new_files_with_the_official_hashes(
    root: dict[str, Any], model: str
) -> None:
    before = snapshot(root["cv"])
    assert cli(root, "run", "--model", model, "--target", "ge2") == 0
    after = snapshot(root["cv"])
    new = sorted(set(after) - set(before))
    assert new == sorted(
        f"ge2/{n}_{model}_R.{e}"
        for n, e in (("best_params", "csv"), ("meta", "json"), ("oof", "csv"), ("runs", "csv"))
    )
    assert {k: v for k, v in after.items() if k in before} == before  # nothing existing changed
    meta = json.loads((root["cv"] / "ge2" / f"meta_{model}_R.json").read_text())
    official = json.loads((root["cv"] / "ge2" / "meta_lgbm_A.json").read_text())
    official_model = json.loads((root["cv"] / "ge2" / f"meta_{model}_A.json").read_text())
    for key in (
        "splits_sha8",
        "inner_sha8",
        "roles_sha8",
        "target_status",
        "seed",
        "repeats",
        "constants",
    ):
        assert meta[key] == official[key]
    assert (
        meta["grid_sha8"] == official_model["grid_sha8"] and meta["grid"] == official_model["grid"]
    )
    assert (
        meta["feature_set"] == "R"
        and meta["columns"] == rf.REFERENCE_COLUMNS
        and meta["features_sha8"] == "17fb7bbf"
    )
    assert meta["reference_roles_sha8"] == "3373dae7" and meta["nan_counts"] == {
        c: 0 for c in rf.REFERENCE_COLUMNS
    }
    runs = pd.read_csv(root["cv"] / "ge2" / f"runs_{model}_R.csv")
    assert (
        list(runs.columns) == ch.RUN_COLUMNS
        and len(runs) == 5
        and (runs["feature_set"] == "R").all()
    )
    assert (
        (runs["n_features"] == 27).all()
        and (runs["threshold"] == 0.5).all()
        and (runs["seed"] == 0).all()
    )
    oof = ch.read_oof(root["cv"] / "ge2" / f"oof_{model}_R.csv")
    assert len(oof) == len(root["frame"]) * 5 and oof["proba"].between(0, 1).all()
    assert (oof.groupby(["repeat", "ref"]).size() == 1).all()
    folds = ch.read_oof(root["cv"] / "ge2" / "oof_lgbm_A.csv")[["ref", "repeat", "outer_fold"]]
    assert folds.equals(oof[["ref", "repeat", "outer_fold"]])  # exactly the official folds


def test_the_outputs_load_in_bootstrap_ci_and_match_an_independent_metric_computation(
    root: dict[str, Any],
) -> None:
    assert cli(root, "run", "--model", "lgbm") == 0
    loaded = bc.load_model(root["cv"], "ge2", "lgbm", "R", root["frame"]["ref"])
    assert loaded["probas"].shape == (5, len(root["frame"]))
    runs = pd.read_csv(root["cv"] / "ge2" / "runs_lgbm_R.csv")
    for r in range(5):
        y, p = loaded["y"], loaded["probas"][r]
        assert runs.loc[r, "auroc"] == pytest.approx(ch.auroc(y, p), abs=1e-9)
        assert runs.loc[r, "balanced_accuracy"] == pytest.approx(
            ch.threshold_metrics(y, p)["balanced_accuracy"], abs=1e-9
        )


def test_the_leakage_assertion_runs_on_every_refit_fold(
    root: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    where: list[str] = []
    real = cvs.assert_disjoint

    def spy(train: Any, test: Any, name: str) -> None:
        where.append(name)
        real(train, test, name)

    monkeypatch.setattr(cvs, "assert_disjoint", spy)
    monkeypatch.setattr(ch, "assert_disjoint", spy)
    assert cli(root, "run", "--model", "b1") == 0
    assert where.count("refit_reference") == 25  # one per outer fold of the 5 repeats
    assert (
        sum(w.startswith("outer repeat") for w in where) >= 25
    )  # run_repeat plus the inner-table build


def test_a_participant_on_both_sides_stops_the_run_and_writes_nothing(
    root: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real = ch.membership

    def straddle(groups: np.ndarray, splits: pd.DataFrame, repeat: int) -> np.ndarray:
        member = real(groups, splits, repeat).copy()
        for person in sorted(set(groups)):
            pos = np.flatnonzero(groups == person)
            if len(pos) == 2:
                member[pos[1]] = (member[pos[0]] + 1) % 5
                break
        return member

    monkeypatch.setattr(ch, "membership", straddle)
    assert cli(root, "run", "--model", "lgbm") == 1
    err = capsys.readouterr().err
    assert "group(s) on both sides" in err and not HASH.search(err) and r_files(root) == []


def test_a_refit_that_does_not_reproduce_the_predictions_is_refused(
    root: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real = rb.refit_fold
    monkeypatch.setattr(rb, "refit_fold", lambda *a, **k: real(*a, **k) + 1e-9)
    assert cli(root, "run", "--model", "lgbm") == 1
    assert (
        "does not reproduce the out-of-fold predictions" in capsys.readouterr().err
        and r_files(root) == []
    )


def test_nan_features_skip_b1_keep_every_clip_and_are_never_imputed(
    root: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_reference(
        root["ref"], root["frame"], nan_columns={"ref_skew_amplitude": 3, "ref_interval_cv": 2}
    )
    assert cli(root, "run", "--model", "b1") == 0
    out = capsys.readouterr().out
    assert (
        "skipped b1 on R" in out
        and "ref_skew_amplitude': 3" in out
        and "ref_interval_cv': 2" in out
        and r_files(root) == []
    )
    assert (
        cli(root, "permute", "--model", "b1", "--draws", "2") == 0
        and "skipped the b1 permutation" in capsys.readouterr().out
    )
    seen: list[int] = []
    real = ch.make_estimator

    class Spy:
        def __init__(self, est: Any) -> None:
            self.est = est

        def fit(self, X: np.ndarray, y: np.ndarray) -> Spy:
            seen.append(int(np.isnan(X).sum()))
            self.est.fit(X, y)
            return self

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return np.asarray(self.est.predict_proba(X))

    monkeypatch.setattr(ch, "make_estimator", lambda m, p, *a, **k: Spy(real(m, p, *a, **k)))
    assert cli(root, "run", "--model", "lgbm") == 0
    assert max(seen) > 0  # LightGBM received the NaN values themselves
    meta = json.loads((root["cv"] / "ge2" / "meta_lgbm_R.json").read_text())
    assert (
        meta["nan_counts"]["ref_skew_amplitude"] == 3 and meta["nan_counts"]["ref_interval_cv"] == 2
    )
    assert (
        len(ch.read_oof(root["cv"] / "ge2" / "oof_lgbm_R.csv")) == len(root["frame"]) * 5
    )  # no clip dropped


def test_overwrite_needs_force_and_identical_content_is_fine(
    root: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli(root, "run", "--model", "b1") == 0
    first = snapshot(root["cv"], [n for n in r_files(root)])
    assert (
        cli(root, "run", "--model", "b1") == 0 and snapshot(root["cv"], r_files(root)) == first
    )  # byte-identical rerun
    write_reference(root["ref"], root["frame"], seed=9)  # a different reference table
    capsys.readouterr()
    assert (
        cli(root, "run", "--model", "b1") == 1
        and "refusing to overwrite" in capsys.readouterr().err
    )
    assert snapshot(root["cv"], r_files(root)) == first
    assert (
        cli(root, "run", "--model", "b1", "--force") == 0
        and snapshot(root["cv"], r_files(root)) != first
    )


def refusal_cases() -> list[Any]:
    def table(change: Any) -> Any:
        def damage(w: dict[str, Any]) -> None:
            path = w["ref"] / rf.OUT_FEATURES
            frame = pd.read_csv(
                path, dtype={"ref": str, "participant": str}, float_precision="round_trip"
            )
            change(frame).to_csv(path, index=False, float_format="%.17g")

        return damage

    def meta_edit(**updates: Any) -> Any:
        return lambda w: edit_json(w["ref"] / rf.OUT_META, **updates)

    def official_edit(**updates: Any) -> Any:
        return lambda w: edit_json(w["cv"] / "ge2" / "meta_lgbm_A.json", **updates)

    def roles_edit(w: dict[str, Any]) -> None:
        roles = pd.read_csv(w["ref"] / rf.OUT_ROLES, dtype=str, keep_default_na=False)
        roles.loc[3, "note"] = "edited"
        roles.to_csv(w["ref"] / rf.OUT_ROLES, index=False)

    return [
        (
            "label_column",
            table(lambda f: f.assign(rating=0)),
            "not exactly ref, participant and the 27 registered features",
        ),
        (
            "group_column",
            table(lambda f: f.assign(hand="left")),
            "not exactly ref, participant and the 27 registered features",
        ),
        (
            "set_a_column_added",
            table(lambda f: f.assign(tap_rate_hz=0.0)),
            "not exactly ref, participant and the 27 registered features",
        ),
        (
            "feature_missing",
            table(lambda f: f.drop(columns=["ref_skew_velocity"])),
            "not exactly ref, participant and the 27 registered features",
        ),
        (
            "columns_reordered",
            table(lambda f: f[["ref", "participant", *reversed(rf.REFERENCE_COLUMNS)]]),
            "not exactly ref, participant and the 27 registered features",
        ),
        ("clip_missing", table(lambda f: f.iloc[1:]), "does not cover exactly the clips"),
        (
            "clip_duplicated",
            table(lambda f: pd.concat([f, f.iloc[[0]]], ignore_index=True)),
            "does not cover exactly the clips",
        ),
        (
            "participant_differs",
            table(
                lambda f: f.assign(participant=np.where(f.index == 0, "deadbeef", f["participant"]))
            ),
            "disagrees with features.csv on a participant",
        ),
        (
            "infinite_value",
            table(
                lambda f: f.assign(
                    ref_velocity_max=np.where(f.index == 0, np.inf, f["ref_velocity_max"])
                )
            ),
            "infinite value",
        ),
        ("roles_edited", roles_edit, "do not match the registered roles_sha8"),
        (
            "meta_roles_hash",
            meta_edit(reference_roles_sha8="deadbeef"),
            "do not match the registered roles_sha8",
        ),
        (
            "meta_columns",
            meta_edit(columns=rf.REFERENCE_COLUMNS[:-1]),
            "disagrees with the registered feature list",
        ),
        (
            "meta_features_hash",
            meta_edit(features_sha8="deadbeef"),
            "disagrees with the registered feature list",
        ),
        (
            "splits_hash",
            official_edit(splits_sha8="deadbeef"),
            "splits_sha8 differs from the official runs",
        ),
        (
            "inner_hash",
            official_edit(inner_sha8="deadbeef"),
            "inner_sha8 differs from the official runs",
        ),
    ]


@pytest.mark.parametrize("name,damage,needle", refusal_cases(), ids=[c[0] for c in refusal_cases()])
def test_refusals_exit_1_with_counts_only_and_write_nothing(
    root: dict[str, Any], capsys: pytest.CaptureFixture[str], name: str, damage: Any, needle: str
) -> None:
    damage(root)
    assert cli(root, "run", "--model", "lgbm") == 1
    err = capsys.readouterr().err
    assert needle in err and not HASH.search(err) and r_files(root) == []


def test_a_changed_grid_is_refused(
    root: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(ch.GRIDS, "lgbm", [dict(SMALL_LGBM[0])])
    assert cli(root, "run", "--model", "lgbm") == 1
    assert (
        "grid_sha8 of lgbm differs from the official runs" in capsys.readouterr().err
        and r_files(root) == []
    )
    monkeypatch.setitem(ch.GRIDS, "b1", [{"C": 1.0}])
    assert cli(root, "run", "--model", "b1") == 1


def test_stored_splits_must_be_the_official_ones(
    root: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    folder = root["cv"] / "ge2"
    csv, meta = folder / "splits.csv", folder / "splits_meta.json"
    original_csv, original_meta = csv.read_bytes(), meta.read_bytes()
    lines = original_csv.decode().splitlines()
    first = lines[1].split(",")
    first[2] = str((int(first[2]) + 1) % 5)
    csv.write_text("\n".join([lines[0], ",".join(first), *lines[2:]]) + "\n")
    assert cli(root, "run", "--model", "lgbm") == 1 and r_files(root) == []  # a tampered file
    csv.write_bytes(original_csv)
    for change in ({"seed": 3}, {"target": "ge1"}):
        meta.write_bytes(original_meta)
        edit_json(meta, **change)
        capsys.readouterr()
        assert cli(root, "run", "--model", "lgbm") == 1, change
        assert (
            "stored splits are not the official ones" in capsys.readouterr().err
            and r_files(root) == []
        )
    meta.write_bytes(original_meta)
    proc = root["proc"]
    args = [
        "--features",
        str(proc / "features.csv"),
        "--roles",
        str(proc / "feature_roles.csv"),
        "--meta",
        str(proc / "features_meta.json"),
    ]
    assert (
        cvs.main(
            [*args, "--target", "ge2", "--repeats", "3", "--out-dir", str(root["cv"]), "--force"]
        )
        == 0
    )  # valid, 3 repeats
    capsys.readouterr()
    assert cli(root, "run", "--model", "lgbm") == 1
    assert (
        "stored splits are not the official ones" in capsys.readouterr().err and r_files(root) == []
    )


def test_missing_inputs_and_bad_paths_exit_2(
    root: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    (root["cv"] / "ge2" / "meta_lgbm_A.json").unlink()
    assert (
        cli(root, "run", "--model", "lgbm") == 2 and "official meta file" in capsys.readouterr().err
    )
    shutil.rmtree(root["ref"])
    assert (
        cli(root, "run", "--model", "lgbm", "--target", "ge1") == 2
        and "missing input" in capsys.readouterr().err
    )
    elsewhere = root["tmp"] / "elsewhere"
    shutil.copytree(root["cv"], elsewhere)
    base = [
        "run",
        "--model",
        "lgbm",
        "--cv-root",
        str(elsewhere),
        "--features",
        str(root["proc"] / "features.csv"),
    ]
    assert (
        rb.main(base) == 2
        and "--cv-root must be inside data/processed/cv" in capsys.readouterr().err
    )
    assert rb.main(["permute", *base[1:]]) == 2 and not list(elsewhere.rglob("*_R.*"))
    assert rb.main(["report", "--cv-root", str(elsewhere)]) == 2
    with pytest.raises(SystemExit) as err:
        rb.main(["run", "--model", "b0"])
    assert err.value.code == 2


# --------------------------------------------------------------------------- permutation


def test_the_permutation_control_passes_on_noise_and_is_reproducible(root: dict[str, Any]) -> None:
    write_reference(root["ref"], root["frame"], seed=3)
    assert cli(root, "permute", "--model", "lgbm", "--draws", "4") == 0
    summary = json.loads((root["cv"] / "ge2" / "permutation_lgbm_R.json").read_text())
    table = pd.read_csv(root["cv"] / "ge2" / "permutation_lgbm_R.csv")
    assert (
        summary["decision"] == "pass"
        and summary["draws"] == 4
        and summary["repeats"] == 1
        and summary["max_allowed_mean"] == 0.55
    )
    assert summary["mean"] == pytest.approx(table["null_auroc"].mean()) and list(table.columns) == [
        "draw",
        "null_auroc",
    ]
    first = snapshot(root["cv"], ["permutation_lgbm_R.csv", "permutation_lgbm_R.json"])
    assert cli(root, "permute", "--model", "lgbm", "--draws", "4") == 0
    assert snapshot(root["cv"], ["permutation_lgbm_R.csv", "permutation_lgbm_R.json"]) == first
    assert (
        rb.build_parser().parse_args(["permute", "--model", "lgbm"]).draws == 20
    )  # the official number of draws


def test_each_permutation_draw_uses_whole_participant_permuted_labels_on_repeat_zero(
    root: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[np.ndarray, int, int]] = []
    real = ch.null_auroc

    def spy(
        model: str,
        grid: Any,
        X: np.ndarray,
        y_perm: np.ndarray,
        groups: np.ndarray,
        splits: pd.DataFrame,
        repeats: int,
    ) -> float:
        calls.append((y_perm.copy(), int(splits["repeat"].max()), repeats))
        return real(model, grid, X, y_perm, groups, splits, repeats)

    monkeypatch.setattr(ch, "null_auroc", spy)
    assert cli(root, "permute", "--model", "b1", "--draws", "3", "--seed", "0") in (0, 1)
    y = cvs.target_labels(root["frame"], "ge2")
    assert len(calls) == 3 and all(c[1] == 0 and c[2] == 1 for c in calls)  # repeat 0 only
    for d, (got, _, _) in enumerate(calls):
        assert np.array_equal(
            got, ch.permute_labels(root["frame"], y, np.random.default_rng(d))
        )  # seed + draw
        assert got.sum() == y.sum()
    assert any(not np.array_equal(c[0], y) for c in calls) and not np.array_equal(
        calls[0][0], calls[1][0]
    )


@pytest.mark.parametrize(
    "null_mean,code,decision",
    [(0.54, 0, "pass"), (0.55, 0, "pass"), (0.56, 1, "fail"), (0.70, 1, "fail")],
)
def test_the_permutation_rule_is_exactly_a_mean_above_0_55(
    root: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    null_mean: float,
    code: int,
    decision: str,
) -> None:
    monkeypatch.setattr(ch, "null_auroc", lambda *a, **k: null_mean)
    assert cli(root, "permute", "--model", "lgbm", "--draws", "2") == code
    summary = json.loads((root["cv"] / "ge2" / "permutation_lgbm_R.json").read_text())
    assert summary["decision"] == decision and summary["max_allowed_mean"] == 0.55


def test_a_leaking_pipeline_fails_the_permutation_control(
    root: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    y = cvs.target_labels(root["frame"], "ge2")

    class Oracle:
        lookup = {
            r.tobytes(): i
            for i, r in enumerate(
                pd.read_csv(root["ref"] / rf.OUT_FEATURES, float_precision="round_trip")[
                    rf.REFERENCE_COLUMNS
                ].to_numpy(dtype=float)
            )
        }

        def fit(self, X: np.ndarray, y: np.ndarray) -> Oracle:
            return self

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            p = np.array([y[self.lookup[r.tobytes()]] for r in X], dtype=float)
            return np.column_stack([1 - p, p])

    monkeypatch.setattr(ch, "make_estimator", lambda *a, **k: Oracle())
    monkeypatch.setattr(
        ch, "permute_labels", lambda f, yy, rng: yy
    )  # nothing is permuted: the leak is visible
    assert cli(root, "permute", "--model", "lgbm", "--draws", "2") == 1
    captured = capsys.readouterr()
    assert "permutation control failed" in captured.err and "decision fail" in captured.out
    assert (
        json.loads((root["cv"] / "ge2" / "permutation_lgbm_R.json").read_text())["decision"]
        == "fail"
    )


# --------------------------------------------------------------------------- the report


@pytest.fixture
def with_r(root: dict[str, Any]) -> dict[str, Any]:
    write_reference(root["ref"], root["frame"], seed=3)
    for target in TARGETS:
        for model in rb.MODELS:
            assert cli(root, "run", "--model", model, "--target", target) == 0
            assert cli(root, "permute", "--model", model, "--target", target, "--draws", "4") == 0
    for name in (
        "ci_table.csv",
        "auroc_difference.csv",
    ):  # decoys: the report must never touch them
        (root["cv"] / "reports_ci").mkdir(exist_ok=True)
        (root["cv"] / "reports_ci" / name).write_text(f"decoy {name}\n")
    return root


def explicit_draws(
    w: dict[str, Any], target: str, minuend: tuple[str, str], subtrahend: tuple[str, str] | None
) -> dict[str, np.ndarray]:
    """Explicit bootstrap with the harness metric functions; returns metric -> per-draw statistic."""
    frame = w["frame"]
    counts = bc.bootstrap_draws(frame["participant"].tolist(), bc.N_DRAWS, bc.SEED)
    order = bc.sorted_participants(frame["participant"].tolist())
    pidx = np.array([order.index(q) for q in frame["participant"]])
    oof = {
        k: ch.read_oof(w["cv"] / target / f"oof_{k[0]}_{k[1]}.csv")
        for k in [minuend, *([subtrahend] if subtrahend else [])]
    }
    y = cvs.target_labels(frame, target)
    probas = {
        k: np.array(
            [
                o[o["repeat"] == r].set_index("ref").loc[frame["ref"], "proba"].to_numpy()
                for r in range(5)
            ]
        )
        for k, o in oof.items()
    }
    out: dict[str, list[float]] = {m: [] for m in rb.SHOWN_METRICS}
    for d in range(len(counts)):
        idx = np.repeat(np.arange(len(y)), counts[d][pidx])
        if y[idx].sum() in (0, len(idx)):
            continue
        for m in rb.SHOWN_METRICS:
            vals = []
            for r in range(5):
                a = {
                    **ch.threshold_metrics(y[idx], probas[minuend][r][idx]),
                    "auroc": ch.auroc(y[idx], probas[minuend][r][idx]),
                }[m]
                if subtrahend:
                    b = {
                        **ch.threshold_metrics(y[idx], probas[subtrahend][r][idx]),
                        "auroc": ch.auroc(y[idx], probas[subtrahend][r][idx]),
                    }[m]
                    a = a - b
                vals.append(a)
            out[m].append(float(np.mean(vals)))
    return {m: np.array(v) for m, v in out.items()}


def report_table(w: dict[str, Any]) -> pd.DataFrame:
    text = {
        c: str
        for c in (
            "draws_sha8",
            "splits_sha8",
            "inner_sha8",
            "roles_sha8",
            "reference_roles_sha8",
            "features_sha8",
        )
    }
    return pd.read_csv(
        w["cv"] / "reports_ci" / "reference_table.csv", float_precision="round_trip", dtype=text
    )


def test_the_report_matches_an_explicit_participant_bootstrap(
    with_r: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli(with_r, "report", "--targets", "ge2") == 0
    t = report_table(with_r)
    for model in rb.MODELS:
        explicit = explicit_draws(with_r, "ge2", (model, "R"), None)
        for metric in rb.SHOWN_METRICS:
            r = t[(t["subject"] == f"{model} R") & (t["metric"] == metric)].iloc[0]
            lo, hi = np.percentile(explicit[metric], [2.5, 97.5])
            assert r["ci_low"] == pytest.approx(lo, abs=1e-9) and r["ci_high"] == pytest.approx(
                hi, abs=1e-9
            )
            runs = pd.read_csv(
                with_r["cv"] / "ge2" / f"runs_{model}_R.csv", float_precision="round_trip"
            )
            assert r["estimate"] == pytest.approx(runs[metric].mean(), abs=1e-9)
    for minuend, subtrahend, name in (
        (("lgbm", "A"), ("lgbm", "R"), "lgbm A minus lgbm R"),
        (("lgbm", "R"), ("b1", "R"), "lgbm R minus b1 R"),
    ):
        d = explicit_draws(with_r, "ge2", minuend, subtrahend)["auroc"]
        r = t[(t["subject"] == name) & (t["section"] == "difference")].iloc[0]
        assert r["ci_low"] == pytest.approx(np.percentile(d, 2.5), abs=1e-9) and r[
            "ci_high"
        ] == pytest.approx(np.percentile(d, 97.5), abs=1e-9)
    assert t["draws_sha8"].nunique() == 1 and t["draws_sha8"].iloc[0] == bc.draws_sha8(
        bc.bootstrap_draws(with_r["frame"]["participant"].tolist(), bc.N_DRAWS, bc.SEED)
    )
    out = capsys.readouterr().out
    assert rb.NOTE.format(n=with_r["frame"]["participant"].nunique()) in out.splitlines()


def test_the_report_layout_order_and_the_absence_of_decision_columns(
    with_r: dict[str, Any],
) -> None:
    assert cli(with_r, "report") == 0
    t = report_table(with_r)
    assert list(t.columns) == rb.TABLE_COLUMNS and not [
        c for c in t.columns if re.search(r"^p$|p_value|best|rank|winner|delta", c)
    ]
    expected = [
        (tg, "performance", s, m)
        for tg in ("ge1", "ge2")
        for s in ("b1 R", "lgbm R")
        for m in rb.SHOWN_METRICS
    ]

    def expected_diff(tg: str) -> list[tuple[str, str, str, str]]:
        return [
            (tg, "difference", name, "auroc")
            for name in ("lgbm A minus lgbm R", "lgbm R minus b1 R")
        ]

    order = list(zip(t["target"], t["section"], t["subject"], t["metric"], strict=True))
    assert order == [*expected[:10], *expected_diff("ge1"), *expected[10:], *expected_diff("ge2")]
    assert (
        (t["reference_roles_sha8"] == "3373dae7").all()
        and (t["seed"] == 2026).all()
        and (t["n_excluded"] == 0).all()
    )


def test_identical_models_have_a_zero_paired_difference(with_r: dict[str, Any]) -> None:
    folder = with_r["cv"] / "ge2"
    for kind in ("oof", "runs"):  # make lgbm R an exact copy of lgbm A
        shutil.copy(folder / f"{kind}_lgbm_A.csv", folder / f"{kind}_lgbm_R.csv")
    meta = json.loads((folder / "meta_lgbm_A.json").read_text())
    table_sha8 = cvs.sha8_of(
        (with_r["ref"] / rf.OUT_FEATURES).read_bytes()
    )  # the copy must name the current table
    fake = {
        **meta,
        "feature_set": "R",
        "features_sha8": rf.PINNED_FEATURES_SHA8,
        "inputs_sha8": {"reference_features": table_sha8},
    }
    (folder / "meta_lgbm_R.json").write_text(json.dumps(fake))
    assert cli(with_r, "report", "--targets", "ge2") == 0
    r = report_table(with_r).query("subject == 'lgbm A minus lgbm R'").iloc[0]
    assert (
        r["estimate"] == 0 and r["ci_low"] == 0 and r["ci_high"] == 0
    )  # the same draws for both models


def test_existing_report_files_and_official_outputs_are_never_touched(
    with_r: dict[str, Any],
) -> None:
    before = snapshot(with_r["cv"], ["ci_table.csv", "auroc_difference.csv"])
    official_before = {
        k: v
        for k, v in snapshot(with_r["cv"]).items()
        if "_R." not in k and "reference_table" not in k
    }
    assert cli(with_r, "report") == 0
    assert snapshot(with_r["cv"], ["ci_table.csv", "auroc_difference.csv"]) == before
    assert {
        k: v
        for k, v in snapshot(with_r["cv"]).items()
        if "_R." not in k and "reference_table" not in k
    } == official_before
    assert sorted(p.name for p in (with_r["cv"] / "reports_ci").iterdir()) == [
        "auroc_difference.csv",
        "ci_table.csv",
        "reference_table.csv",
    ]


def test_a_report_without_b1_r_still_reports_lgbm_and_the_a_comparison(
    with_r: dict[str, Any],
) -> None:
    for p in (with_r["cv"] / "ge2").glob("*b1_R*"):
        p.unlink()
    for p in (with_r["cv"] / "ge1").glob("*b1_R*"):
        p.unlink()
    assert cli(with_r, "report") == 0
    t = report_table(with_r)
    assert set(t["subject"]) == {"lgbm R", "lgbm A minus lgbm R"}


def report_refusal_cases() -> list[Any]:
    def folder(w: dict[str, Any]) -> Path:
        return w["cv"] / "ge2"

    return [
        (
            "permutation_missing",
            lambda w: (folder(w) / "permutation_lgbm_R.json").unlink(),
            "permutation control files are missing",
        ),
        (
            "permutation_failed",
            lambda w: edit_json(folder(w) / "permutation_b1_R.json", decision="fail"),
            "did not pass",
        ),
        (
            "permutation_mean",
            lambda w: edit_json(folder(w) / "permutation_lgbm_R.json", mean=0.123),
            "summary mean disagrees",
        ),
        (
            "splits_sha8",
            lambda w: edit_json(folder(w) / "meta_lgbm_R.json", splits_sha8="deadbeef"),
            "differs from the other models in splits_sha8",
        ),
        (
            "inner_sha8",
            lambda w: edit_json(folder(w) / "meta_b1_R.json", inner_sha8="deadbeef"),
            "differs from the other models in inner_sha8",
        ),
        (
            "target_status",
            lambda w: edit_json(folder(w) / "meta_lgbm_R.json", target_status="settled"),
            "differs from the other models in target_status",
        ),
        (
            "runs_auroc",
            lambda w: edit_runs(folder(w) / "runs_lgbm_R.csv", auroc=0.99),
            "lgbm R: oof and runs disagree on auroc",
        ),
        (
            "runs_mcc",
            lambda w: edit_runs(folder(w) / "runs_b1_R.csv", mcc=0.77),
            "b1 R: oof and runs disagree on mcc",
        ),
        (
            "clip_missing",
            lambda w: (
                pd.read_csv(folder(w) / "oof_lgbm_R.csv", dtype={"ref": str})
                .iloc[1:]
                .to_csv(folder(w) / "oof_lgbm_R.csv", index=False)
            ),
            "does not predict every clip exactly once",
        ),
        (
            "repeats",
            lambda w: edit_json(folder(w) / "meta_lgbm_R.json", repeats=4),
            "meta has 4 repeats",
        ),
    ]


@pytest.mark.parametrize(
    "name,damage,needle", report_refusal_cases(), ids=[c[0] for c in report_refusal_cases()]
)
def test_report_refusals_exit_1_and_write_nothing(
    with_r: dict[str, Any], capsys: pytest.CaptureFixture[str], name: str, damage: Any, needle: str
) -> None:
    damage(with_r)
    capsys.readouterr()
    with pytest.raises(cvs.GuardFailure, match=re.escape(needle)):
        rb.build_report(with_r["cv"], with_r["proc"] / "features.csv", ["ge2"], with_r["ref"])
    assert (
        rb.main(
            [
                "report",
                "--features",
                str(with_r["proc"] / "features.csv"),
                "--cv-root",
                str(with_r["cv"]),
                "--reference-dir",
                str(with_r["ref"]),
                "--targets",
                "ge2",
            ]
        )
        == 1
    )
    assert not (with_r["cv"] / "reports_ci" / "reference_table.csv").exists()
    assert not HASH.search(capsys.readouterr().err)


def test_runs_made_from_another_reference_table_are_not_reported(
    with_r: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    write_reference(
        with_r["ref"], with_r["frame"], seed=11
    )  # the table changed after the models were run
    assert cli(with_r, "report", "--targets", "ge2") == 1
    err = capsys.readouterr().err
    assert "b1 R was not produced from the current reference table" in err and not HASH.search(err)
    assert not (with_r["cv"] / "reports_ci" / "reference_table.csv").exists()
    for p in (with_r["cv"] / "ge2").glob("*b1_R*"):  # without b1 R the lgbm R run is still stale
        p.unlink()
    assert cli(with_r, "report", "--targets", "ge2") == 1
    assert "lgbm R was not produced from the current reference table" in capsys.readouterr().err


def test_a_stale_b1_run_is_caught_when_the_table_later_gets_nan(
    with_r: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    write_reference(with_r["ref"], with_r["frame"], seed=3, nan_columns={"ref_velocity_max": 2})
    assert (
        cli(with_r, "run", "--model", "b1", "--target", "ge2") == 0
    )  # skipped: the old b1 files stay on disk
    assert cli(with_r, "run", "--model", "lgbm", "--target", "ge2", "--force") == 0
    capsys.readouterr()
    assert cli(with_r, "report", "--targets", "ge2") == 1
    assert "b1 R was not produced from the current reference table" in capsys.readouterr().err


def test_the_r_grid_must_still_be_the_official_grid_at_report_time(
    with_r: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    edit_json(with_r["cv"] / "ge2" / "meta_lgbm_R.json", grid_sha8="deadbeef")
    assert cli(with_r, "report", "--targets", "ge2") == 1
    assert "grid_sha8 of lgbm R differs from the official runs" in capsys.readouterr().err
    edit_json(with_r["cv"] / "ge1" / "meta_b1_R.json", grid_sha8="deadbeef")
    assert cli(with_r, "report", "--targets", "ge1") == 1
    assert "grid_sha8 of b1 R differs from the official runs" in capsys.readouterr().err
    assert not (with_r["cv"] / "reports_ci" / "reference_table.csv").exists()


def test_a_missing_reference_table_makes_the_report_exit_2(
    with_r: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    (with_r["ref"] / rf.OUT_FEATURES).unlink()
    assert cli(with_r, "report", "--targets", "ge2") == 2
    assert "missing input: reference_features.csv" in capsys.readouterr().err


def test_a_missing_lgbm_r_run_exits_2(
    with_r: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    (with_r["cv"] / "ge2" / "runs_lgbm_R.csv").unlink()
    assert (
        cli(with_r, "report", "--targets", "ge2") == 2
        and "missing input" in capsys.readouterr().err
    )


def test_excluded_draws_follow_the_same_rule_as_the_other_reports(
    with_r: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real = bc.bootstrap_draws
    order = bc.sorted_participants(with_r["frame"]["participant"].tolist())
    y_max = (
        with_r["frame"]
        .assign(y=cvs.target_labels(with_r["frame"], "ge2"))
        .groupby("participant")["y"]
        .max()
    )

    def planted(k: int) -> Any:
        def make(participants: Any, n_draws: int, seed: int) -> np.ndarray:
            counts = real(participants, 100, seed)
            for i in range(k):
                counts[3 + i] = np.array([1 if y_max[p] == 0 else 0 for p in order])
            return counts

        return make

    monkeypatch.setattr(bc, "bootstrap_draws", planted(1))
    assert cli(with_r, "report", "--targets", "ge2") == 0
    assert (report_table(with_r)["n_excluded"] == 1).all() and (
        report_table(with_r)["n_draws_used"] == 99
    ).all()
    t = report_table(with_r)
    for model in (
        rb.MODELS
    ):  # intervals are computed without the excluded draw (a class-less draw would add an MCC of 0)
        explicit = explicit_draws(with_r, "ge2", (model, "R"), None)
        for metric in rb.SHOWN_METRICS:
            r = t[(t["subject"] == f"{model} R") & (t["metric"] == metric)].iloc[0]
            assert len(explicit[metric]) == 99
            assert r["ci_low"] == pytest.approx(np.percentile(explicit[metric], 2.5), abs=1e-9)
            assert r["ci_high"] == pytest.approx(np.percentile(explicit[metric], 97.5), abs=1e-9)
    (with_r["cv"] / "reports_ci" / "reference_table.csv").unlink()
    monkeypatch.setattr(bc, "bootstrap_draws", planted(2))
    capsys.readouterr()
    assert (
        rb.main(
            [
                "report",
                "--features",
                str(with_r["proc"] / "features.csv"),
                "--cv-root",
                str(with_r["cv"]),
                "--reference-dir",
                str(with_r["ref"]),
                "--targets",
                "ge2",
            ]
        )
        == 1
    )
    assert "2 of 100 draws lack a class" in capsys.readouterr().err


def test_report_rerun_is_byte_identical_atomic_and_has_no_ids(
    with_r: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli(with_r, "report") == 0
    path = with_r["cv"] / "reports_ci" / "reference_table.csv"
    first = path.read_bytes()
    assert cli(with_r, "report") == 0 and path.read_bytes() == first
    ids = set(with_r["frame"]["ref"]) | set(with_r["frame"]["participant"])
    blob = capsys.readouterr().out + path.read_text()
    assert not [i for i in ids if i in blob] and not re.search(r"20\d\d-\d\d-\d\d|T\d\d:\d\d", blob)

    def broken(src: Any, dst: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(bc.os, "replace", broken)
    with pytest.raises(OSError):
        cli(with_r, "report")
    assert path.read_bytes() == first


def test_no_label_file_is_opened_and_outputs_carry_no_ids(
    root: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    opened: list[str] = []
    real_open = builtins.open

    def spy(file: Any, *a: Any, **k: Any) -> Any:
        opened.append(str(file))
        return real_open(file, *a, **k)

    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(io, "open", spy)
    assert cli(root, "run", "--model", "lgbm") == 0 and cli(
        root, "permute", "--model", "lgbm", "--draws", "2"
    ) in (0, 1)
    assert not [p for p in opened if "fis_diagnostic" in p or "HUBU-FIS" in p]
    ids = set(root["frame"]["participant"])
    blob = capsys.readouterr().out + "".join(
        p.read_text()
        for p in (root["cv"] / "ge2").iterdir()
        if p.name.endswith(("_R.json", "_R.csv")) and not p.name.startswith("oof_")
    )
    assert not [i for i in ids if i in blob] and not re.search(r"20\d\d-\d\d-\d\d|T\d\d:\d\d", blob)


def test_source_names_no_label_file_and_imports_no_gait_or_feature_matrix() -> None:
    source = (SRC / "reference_baseline.py").read_text()
    assert "fis_diagnostic" not in source and "UPDRS" not in source
    imports = {
        i.split(".")[0] for i in re.findall(r"^\s*(?:from|import)\s+([\w.]+)", source, flags=re.M)
    }
    assert not {"feature_matrix", "hubu_fis", "gait"} & imports
    assert br is not None
