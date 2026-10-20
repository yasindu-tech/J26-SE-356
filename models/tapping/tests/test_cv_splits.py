"""Tests for models/tapping/src/cv_splits.py (task FT-32, part 1). Synthetic frames with hashed
ids and a decoy label file that must never be opened; the last tests use the real export and skip
when the data is absent."""

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
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv_splits as cv  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src"
REAL = Path(__file__).resolve().parents[3] / "data" / "processed"
REAL_PRESENT = (REAL / "features.csv").is_file() and (REAL / "features_meta.json").is_file()

EXTRA_ROLES = {
    "hand": "covariate_candidate",
    "n_taps": "feature_twin",
    "n_over_duration_hz": "feature_secondary",
    "amp_mean_sens": "feature_sensitivity",
    "iti_slope": "feature_exploratory",
    "n_halts": "feature_descriptive",
    "n_valid_taps": "audit_count",
    "qc_status": "audit_count",
}


def hid(prefix: str, i: int) -> str:
    return hashlib.sha256(f"{prefix}{i}".encode()).hexdigest()[:8]


def make_frame(n_pos: int = 30, n_neg: int = 60, mixed: bool = True, seed: int = 0) -> pd.DataFrame:
    """n_pos participants with a clip rated >= 2, n_neg with ratings 0-1; 1-2 clips each."""
    rng = np.random.default_rng(seed)
    rows = []
    for kind, n in (("pos", n_pos), ("neg", n_neg)):
        for i in range(n):
            pid = hid(kind + "p", i)
            clips = 2 if (i % 3 != 0 and (mixed or kind == "pos")) else 1
            if not mixed and kind == "pos":
                clips = 1
            for c in range(clips):
                if kind == "pos":
                    rating = 2 + (i % 2) if (c == 0 or not mixed or i % 4 != 0) else 1
                else:
                    rating = (i + c) % 2
                rows.append((hid(kind + "c", 1000 * i + c), pid, rating))
    frame = pd.DataFrame(rows, columns=["ref", "participant", "rating"])
    frame.insert(2, "hand", np.where(np.arange(len(frame)) % 2 == 0, "right", "left"))
    for col in cv.MODEL_COLUMNS:
        frame[col] = rng.normal(size=len(frame))
    frame["decrement_present"] = frame["decrement_present"] > 0
    frame.loc[~frame["decrement_present"], "decrement_onset_frac"] = np.nan
    for col in EXTRA_ROLES:
        if col not in frame.columns:
            frame[col] = rng.normal(size=len(frame))
    assert frame["ref"].is_unique
    return frame.sample(frac=1, random_state=seed).reset_index(drop=True)  # row order is shuffled


def roles_for(frame: pd.DataFrame) -> pd.DataFrame:
    role = {"ref": "id", "participant": "group_key", "rating": "label", **EXTRA_ROLES}
    role.update({c: "feature_primary" for c in cv.MODEL_COLUMNS})
    cols = [c for c in frame.columns if c in role]
    return pd.DataFrame({"column": cols, "role": [role[c] for c in cols],
                         "note": [f"note, with comma {c}" for c in cols]})  # fmt: skip


def roles_hash(roles: pd.DataFrame) -> str:
    table = {
        c: [r, n] for c, r, n in zip(roles["column"], roles["role"], roles["note"], strict=True)
    }
    text = json.dumps(table, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:8]


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    proc = tmp_path / "data" / "processed"
    proc.mkdir(parents=True)
    cv_root = proc / "cv"
    monkeypatch.setattr(cv, "CV_ROOT", cv_root)
    raw = tmp_path / "data" / "raw" / "HUBU-FIS"
    raw.mkdir(parents=True)
    (raw / "fis_diagnostic.csv").write_text("ID,UPDRS\nCONTROL01_DCHA,0\nID1004_IZDA,3\n")
    w = {"proc": proc, "cv": cv_root, "labels": raw / "fis_diagnostic.csv", "tmp": tmp_path}
    write_world(w, make_frame())
    return w


def write_world(w: dict[str, Any], frame: pd.DataFrame, roles: pd.DataFrame | None = None) -> None:
    roles = roles_for(frame) if roles is None else roles
    frame.to_csv(w["proc"] / "features.csv", index=False)
    roles.to_csv(w["proc"] / "feature_roles.csv", index=False)
    meta = {"rows": len(frame), "participants": int(frame["participant"].nunique()),
            "roles_sha8": roles_hash(roles)}  # fmt: skip
    (w["proc"] / "features_meta.json").write_text(json.dumps(meta))


def run(w: dict[str, Any], *extra: str, out: Path | None = None) -> int:
    return cv.main(
        ["--features", str(w["proc"] / "features.csv"), "--roles", str(w["proc"] / "feature_roles.csv"),
         "--meta", str(w["proc"] / "features_meta.json"), "--out-dir", str(out or w["cv"]), *extra]
    )  # fmt: skip


def snapshot(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}  # fmt: skip


# --------------------------------------------------------------------------- constants


def test_constants_are_pinned() -> None:
    assert cv.TARGETS == {"ge1": 1, "ge2": 2, "ge3": 3} and cv.DEFAULT_TARGET == "ge2"
    assert (cv.N_OUTER, cv.N_INNER, cv.MIN_POS_PARTICIPANTS) == (5, 3, 5)
    assert len(cv.MODEL_COLUMNS) == 13 and len(set(cv.MODEL_COLUMNS)) == 13
    assert set(cv.SET_A_PRIMARY).isdisjoint(cv.SET_B_ADDED)
    assert len(cv.SET_A_PRIMARY) == 5 and len(cv.SET_B_ADDED) == 8


def test_model_columns_equal_the_registry_primary_set() -> None:
    import feature_matrix as fm  # the registry itself; the module under test never imports it

    assert sorted(cv.SET_A_PRIMARY + cv.SET_B_ADDED) == sorted(fm.PRIMARY_COLUMNS)
    assert cv.roles_sha8_from_file(fm.roles_table(list(fm.ROLES))) == fm.roles_sha8()


@pytest.mark.skipif(not REAL_PRESENT, reason="real export absent")
def test_local_roles_hash_equals_the_real_meta_value() -> None:
    roles = pd.read_csv(REAL / "feature_roles.csv", dtype=str, keep_default_na=False)
    meta = json.loads((REAL / "features_meta.json").read_text())
    assert cv.roles_sha8_from_file(roles) == meta["roles_sha8"]


def test_target_thresholds_are_fixed() -> None:
    frame = pd.DataFrame({"rating": [0, 1, 2, 3]})
    assert cv.target_labels(frame, "ge1").tolist() == [0, 1, 1, 1]
    assert cv.target_labels(frame, "ge2").tolist() == [0, 0, 1, 1]
    assert cv.target_labels(frame, "ge3").tolist() == [0, 0, 0, 1]
    with pytest.raises(cv.GuardFailure):
        cv.target_labels(frame, "ge4")


def test_seed_formulas() -> None:
    assert [cv.outer_seed(0, r) for r in range(3)] == [0, 1, 2] and cv.outer_seed(7, 2) == 9
    assert cv.inner_seed(0, 0) == 1 and cv.inner_seed(2, 3) == 2004 and cv.inner_seed(1, 0) == 1001


def test_inner_seed_pinned_by_a_hand_computed_example() -> None:
    frame = cv.canonical(make_frame())
    y = cv.target_labels(frame, "ge2")
    groups = frame["participant"].to_numpy()
    train = np.flatnonzero(np.arange(len(frame)) % 5 != 0)
    got = cv.inner_folds(train, y, groups, repeat=2, outer_fold=3)
    direct = StratifiedGroupKFold(3, shuffle=True, random_state=2004).split(
        np.zeros(len(train)), y[train], groups[train]
    )
    for (fit, val), (dfit, dval) in zip(got, direct, strict=True):
        assert fit.tolist() == train[dfit].tolist() and val.tolist() == train[dval].tolist()


# --------------------------------------------------------------------------- splits


def test_every_participant_is_in_exactly_one_test_fold_per_repeat() -> None:
    frame = make_frame()
    splits = cv.make_splits(frame, "ge2", repeats=3, seed=0)
    assert list(splits.columns) == ["participant", "repeat", "outer_fold"]
    people = set(frame["participant"])
    for _, part in splits.groupby("repeat"):
        assert len(part) == len(people) and set(part["participant"]) == people
        assert part["participant"].is_unique and set(part["outer_fold"]) == set(range(5))
    assert len(splits) == 3 * len(people)


def test_outer_and_inner_splits_are_disjoint_and_complete() -> None:
    frame = cv.canonical(make_frame())
    plan = cv.build_plan(frame, "ge2", repeats=2, seed=0)
    groups = frame["participant"].to_numpy()
    for (repeat, fold), part in plan.inner.groupby(["repeat", "outer_fold"]):
        train_groups = set(groups[part["position"].unique()])
        test = plan.splits[(plan.splits["repeat"] == repeat) & (plan.splits["outer_fold"] == fold)]
        assert train_groups.isdisjoint(set(test["participant"]))
        for _, inner in part.groupby("inner_fold"):
            fit = set(groups[inner.loc[inner["role"] == "fit", "position"]])
            val = inner.loc[inner["role"] == "val", "position"]
            assert fit.isdisjoint(set(groups[val]))
        val_all = part.loc[part["role"] == "val", "position"]
        assert sorted(val_all) == sorted(part["position"].unique())  # each clip validated once


def test_inner_folds_are_disjoint_and_cover_the_training_part() -> None:
    frame = cv.canonical(make_frame())
    y = cv.target_labels(frame, "ge2")
    groups = frame["participant"].to_numpy()
    train = np.arange(len(frame))[::2]
    folds = cv.inner_folds(train, y, groups, 0, 0)
    assert len(folds) == 3
    for fit, val in folds:
        assert set(groups[fit]).isdisjoint(set(groups[val]))
        assert sorted(np.concatenate([fit, val])) == sorted(train)


def test_inner_folds_do_not_depend_on_the_feature_values() -> None:
    frame = make_frame()
    other = frame.copy()
    other[cv.MODEL_COLUMNS[0]] = 0.0
    a = cv.build_plan(frame, "ge2", 1, 0)
    b = cv.build_plan(other, "ge2", 1, 0)
    assert a.inner_sha8 == b.inner_sha8 and a.splits_sha8 == b.splits_sha8


def test_splits_do_not_depend_on_row_order() -> None:
    frame = make_frame()
    shuffled = frame.sample(frac=1, random_state=5).reset_index(drop=True)
    assert (
        cv.build_plan(frame, "ge2", 1, 0).splits_sha8
        == cv.build_plan(shuffled, "ge2", 1, 0).splits_sha8
    )


def test_positions_refer_to_the_ref_sorted_frame() -> None:
    frame = make_frame()
    canon = cv.canonical(frame)
    assert canon["ref"].is_monotonic_increasing and set(canon["ref"]) == set(frame["ref"])
    assert cv.canonical(canon).equals(canon)
    inner = cv.build_plan(frame, "ge2", 1, 0).inner
    assert (canon["ref"].to_numpy()[inner["position"].to_numpy()] == inner["ref"].to_numpy()).all()
    assert inner["position"].max() == len(frame) - 1


def test_same_seed_is_identical_and_different_seeds_differ() -> None:
    frame = make_frame()
    a = cv.make_splits(frame, "ge2", 2, 0)
    assert cv.splits_bytes(a) == cv.splits_bytes(cv.make_splits(frame, "ge2", 2, 0))
    assert cv.splits_bytes(a) != cv.splits_bytes(cv.make_splits(frame, "ge2", 2, 1))
    plan = cv.build_plan(frame, "ge2", 2, 0)
    other = cv.build_plan(frame, "ge2", 2, 1)
    assert plan.splits_sha8 != other.splits_sha8 and plan.inner_sha8 != other.inner_sha8


def test_repeats_use_different_seeds() -> None:
    splits = cv.make_splits(make_frame(), "ge2", 3, 0)
    folds = [part.set_index("participant")["outer_fold"] for _, part in splits.groupby("repeat")]
    assert not folds[0].equals(folds[1].reindex(folds[0].index))
    assert not folds[1].equals(folds[2].reindex(folds[1].index))
    assert not folds[0].equals(folds[2].reindex(folds[0].index))


def test_participants_with_clips_in_both_classes_stay_on_one_side() -> None:
    frame = make_frame()
    both = frame.assign(y=frame["rating"] >= 2).groupby("participant")["y"].agg(["min", "max"])
    mixed = set(both.index[both["min"] != both["max"]])
    assert len(mixed) >= 3
    plan = cv.build_plan(frame, "ge2", 2, 0)
    assert (
        plan.splits[plan.splits["participant"].isin(mixed)]
        .groupby("repeat")
        .participant.nunique()
        .eq(len(mixed))
        .all()
    )
    canon = cv.canonical(frame)
    groups = canon["participant"].to_numpy()
    for _, part in plan.inner.groupby(["repeat", "outer_fold", "inner_fold"]):
        sides: dict[str, set[str]] = {}
        for role, pos in zip(part["role"], part["position"], strict=True):
            sides.setdefault(groups[pos], set()).add(role)
        assert all(len(s) == 1 for s in sides.values())


# --------------------------------------------------------------------------- guards


def test_assert_disjoint_raises_on_overlap_with_counts_only() -> None:
    with pytest.raises(cv.LeakageError) as err:
        cv.assert_disjoint(["SECRET_A", "SECRET_B", "x"], ["SECRET_B", "y"], "fold 3")
    text = str(err.value)
    assert "1 group(s) on both sides" in text and "fold 3" in text
    assert "SECRET" not in text and isinstance(err.value, AssertionError)


def test_assert_disjoint_raises_on_an_empty_side_and_passes_when_disjoint() -> None:
    with pytest.raises(cv.LeakageError, match="empty side"):
        cv.assert_disjoint([], ["a"], "w")
    with pytest.raises(cv.LeakageError, match="empty side"):
        cv.assert_disjoint(["a"], [], "w")
    cv.assert_disjoint(["a", "a"], ["b"], "w")


def test_gate_trips_on_too_few_positive_participants() -> None:
    with pytest.raises(cv.GateFailure, match="gate failed") as err:
        cv.make_splits(make_frame(n_pos=6, n_neg=60), "ge2", 1, 0)
    assert "participants" in str(err.value)


def test_gate_trips_on_too_few_negative_participants() -> None:
    with pytest.raises(cv.GateFailure, match="gate failed"):
        cv.make_splits(make_frame(n_pos=60, n_neg=6, mixed=False), "ge2", 1, 0)


def test_gate_passes_on_a_larger_matrix_and_reports_counts() -> None:
    plan = cv.build_plan(make_frame(n_pos=40, n_neg=60), "ge2", 2, 0)
    gate = plan.gate
    assert gate["outer_test"]["folds"] == 10 and gate["inner_validation"]["folds"] == 30
    assert gate["outer_test"]["pos_participants"]["min"] >= 5
    assert gate["inner_validation"]["pos_participants"]["min"] >= 5
    assert gate["outer_test"]["thin_folds"] == 0


def test_gate_boundary_is_five_positive_participants() -> None:
    ok = cv.build_plan(make_frame(n_pos=25, n_neg=50, mixed=False), "ge2", 1, 0)
    assert ok.gate["outer_test"]["pos_participants"]["min"] == 5
    with pytest.raises(cv.GateFailure):
        cv.make_splits(make_frame(n_pos=24, n_neg=50, mixed=False), "ge2", 1, 0)


def test_gate_applies_to_the_chosen_target() -> None:
    frame = make_frame(n_pos=30, n_neg=60)
    cv.make_splits(frame, "ge1", 1, 0)  # many positives at >= 1
    cv.make_splits(frame, "ge2", 1, 0)
    with pytest.raises(cv.GateFailure):
        cv.make_splits(frame, "ge3", 1, 0)  # only half of the positives are rated 3


def test_check_partition_rejects_duplicates_missing_and_bad_folds() -> None:
    frame = make_frame()
    people = set(frame["participant"])
    splits = cv.make_splits(frame, "ge2", 2, 0)
    cv.check_partition(splits, people, 2)
    with pytest.raises(cv.GuardFailure, match="repeated"):
        cv.check_partition(pd.concat([splits, splits.iloc[[0]]], ignore_index=True), people, 2)
    with pytest.raises(cv.GuardFailure, match="missing or extra"):
        cv.check_partition(splits.iloc[1:], people, 2)
    with pytest.raises(cv.GuardFailure, match="outer_fold"):
        cv.check_partition(splits.assign(outer_fold=splits["outer_fold"] + 5), people, 2)
    with pytest.raises(cv.GuardFailure, match="repeats"):
        cv.check_partition(splits, people, 3)


def test_hashes_depend_on_every_column() -> None:
    splits = cv.make_splits(make_frame(), "ge2", 2, 0)
    base = cv.splits_sha8(splits)
    relabelled = splits.assign(repeat=1 - splits["repeat"])  # same rows, repeats swapped
    assert cv.splits_sha8(relabelled) != base
    moved = splits.copy()
    moved.loc[0, "outer_fold"] = (moved.loc[0, "outer_fold"] + 1) % 5
    assert cv.splits_sha8(moved) != base
    swapped = splits.copy()
    swapped["participant"] = np.roll(swapped["participant"].to_numpy(), 1)
    assert cv.splits_sha8(swapped) != base
    assert cv.splits_sha8(splits.sample(frac=1, random_state=0)) == base  # row order is irrelevant


def test_inner_hash_depends_on_every_column() -> None:
    inner = cv.build_plan(make_frame(), "ge2", 2, 0).inner
    base = cv.inner_sha8(inner)
    for column in ("repeat", "outer_fold", "inner_fold"):
        assert cv.inner_sha8(inner.assign(**{column: inner[column] + 1})) != base
    assert cv.inner_sha8(inner.assign(role=inner["role"].map({"fit": "val", "val": "fit"}))) != base
    assert cv.inner_sha8(inner.assign(ref=np.roll(inner["ref"].to_numpy(), 1))) != base
    assert cv.inner_sha8(inner.sample(frac=1, random_state=0)) == base  # row order is irrelevant


# --------------------------------------------------------------------------- inputs


def test_validate_inputs_accepts_primary_columns() -> None:
    frame = make_frame()
    cv.validate_inputs(frame, cv.MODEL_COLUMNS)
    cv.validate_inputs(frame, cv.SET_A_PRIMARY)


@pytest.mark.parametrize(
    "column",
    ["n_taps", "n_over_duration_hz", "amp_mean_sens", "iti_slope", "n_halts", "n_valid_taps",
     "hand", "ref", "participant", "rating", "qc_status", "not_a_column"],
)  # fmt: skip
def test_validate_inputs_refuses_non_primary_columns(column: str) -> None:
    with pytest.raises(cv.ContractError, match="not feature_primary"):
        cv.validate_inputs(make_frame(), [*cv.SET_A_PRIMARY, column])


def test_validate_inputs_refuses_duplicates_missing_empty_and_text() -> None:
    frame = make_frame()
    with pytest.raises(cv.ContractError, match="duplicate"):
        cv.validate_inputs(frame, ["amp_mean", "amp_mean"])
    with pytest.raises(cv.ContractError, match="missing"):
        cv.validate_inputs(frame.drop(columns=["amp_mean"]), ["amp_mean"])
    with pytest.raises(cv.ContractError, match="no model columns"):
        cv.validate_inputs(frame, [])
    with pytest.raises(cv.ContractError, match="not numeric"):
        cv.validate_inputs(frame.assign(amp_mean="x"), ["amp_mean"])


def test_load_inputs_returns_the_frame_and_keeps_nan(world: dict[str, Any]) -> None:
    frame, cols = cv.load_inputs(world["proc"] / "features.csv", world["proc"] / "feature_roles.csv",
                                 world["proc"] / "features_meta.json")  # fmt: skip
    assert (
        cols["model"] == cv.MODEL_COLUMNS and cols["set_a"] + cols["set_b_added"] == cols["model"]
    )
    assert frame["decrement_onset_frac"].isna().any()  # NaN stays NaN, never 0
    assert (frame.loc[frame["decrement_present"], "decrement_onset_frac"].notna()).all()


def test_roles_hash_mismatch_exits_1(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    roles = pd.read_csv(world["proc"] / "feature_roles.csv", dtype=str, keep_default_na=False)
    roles.loc[0, "note"] = "edited after the meta file was written"
    roles.to_csv(world["proc"] / "feature_roles.csv", index=False)
    assert run(world) == 1
    assert "roles_sha8" in capsys.readouterr().err and not world["cv"].exists()


def test_a_changed_primary_set_exits_1_even_with_a_matching_hash(world: dict[str, Any]) -> None:
    frame = pd.read_csv(world["proc"] / "features.csv", dtype={"ref": str, "participant": str})
    roles = roles_for(frame)
    roles.loc[roles["column"] == "amp_mean", "role"] = "feature_twin"
    write_world(world, frame, roles)
    assert run(world) == 1 and not world["cv"].exists()


@pytest.mark.parametrize(
    "damage",
    ["rating_nan", "rating_four", "rating_float", "dup_ref", "raw_participant", "empty_participant",
     "raw_ref", "missing_column"],
)  # fmt: skip
def test_inconsistent_inputs_exit_1_without_echoing_values(
    world: dict[str, Any], damage: str, capsys: pytest.CaptureFixture[str]
) -> None:
    frame = make_frame()
    if damage == "rating_nan":
        frame["rating"] = frame["rating"].astype(float)
        frame.loc[0, "rating"] = np.nan
    elif damage == "rating_four":
        frame.loc[0, "rating"] = 4
    elif damage == "rating_float":
        frame["rating"] = frame["rating"].astype(float)
    elif damage == "dup_ref":
        frame.loc[1, "ref"] = frame.loc[0, "ref"]
    elif damage == "raw_participant":
        frame.loc[0, "participant"] = "CONTROL01"
    elif damage == "empty_participant":
        frame.loc[0, "participant"] = ""
    elif damage == "raw_ref":
        frame.loc[0, "ref"] = "ID1004_IZDA"
    else:
        frame = frame.drop(columns=["close_vel_peak"])
    write_world(world, frame, roles_for(make_frame()))
    assert run(world) == 1
    err = capsys.readouterr().err
    assert "CONTROL01" not in err and "ID1004" not in err and not world["cv"].exists()


def test_row_count_must_match_the_meta_file(world: dict[str, Any]) -> None:
    meta_path = world["proc"] / "features_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["rows"] += 1
    meta_path.write_text(json.dumps(meta))
    assert run(world) == 1


def test_missing_input_exits_2(world: dict[str, Any]) -> None:
    (world["proc"] / "feature_roles.csv").unlink()
    assert run(world) == 2 and not world["cv"].exists()


# --------------------------------------------------------------------------- CLI and files


def test_cli_writes_splits_and_meta(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(world, "--repeats", "2") == 0
    out = world["cv"] / "ge2"
    assert sorted(p.name for p in out.iterdir()) == ["splits.csv", "splits_meta.json"]
    meta = json.loads((out / "splits_meta.json").read_text())
    splits = pd.read_csv(out / "splits.csv", dtype={"participant": str})
    assert meta["target"] == "ge2" and meta["threshold"] == 2 and meta["repeats"] == 2
    assert meta["outer_seeds"] == [0, 1] and meta["roles_sha8"] == roles_hash(
        pd.read_csv(world["proc"] / "feature_roles.csv", dtype=str, keep_default_na=False)
    )
    assert meta["splits_sha8"] == cv.splits_sha8(splits) and len(meta["inner_sha8"]) == 8
    assert set(meta["inputs_sha8"]) == {"features", "roles", "features_meta"}
    assert set(meta["versions"]) == {"python", "scikit_learn", "numpy", "pandas"}
    assert meta["gate"]["outer_test"]["pos_participants"]["min"] >= 5
    assert not re.search(r"20\d\d-\d\d-\d\d|T\d\d:\d\d", json.dumps(meta))  # no timestamps
    printed = capsys.readouterr().out
    assert "splits_sha8" in printed and "gate ok" in printed


def test_rerun_is_byte_identical(world: dict[str, Any]) -> None:
    assert run(world) == 0
    first = snapshot(world["cv"])
    assert run(world) == 0
    assert snapshot(world["cv"]) == first
    other = world["cv"] / "again"
    assert run(world, out=other) == 0
    assert snapshot(other)["ge2/splits.csv"] == first["ge2/splits.csv"]
    assert snapshot(other)["ge2/splits_meta.json"] == first["ge2/splits_meta.json"]
    assert not list(world["cv"].rglob(".*tmp*"))  # atomic writes leave nothing behind


def test_out_dir_outside_data_processed_cv_exits_2_and_writes_nothing(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    outside = world["tmp"] / "elsewhere"
    assert run(world, out=outside) == 2
    err = capsys.readouterr().err
    assert "--out-dir must be inside data/processed/cv" in err and str(world["tmp"]) not in err
    assert not outside.exists() and not world["cv"].exists()
    sneaky = world["cv"] / ".." / ".." / "escape"
    assert run(world, out=sneaky) == 2 and not (world["proc"].parent / "escape").exists()
    assert not (world["proc"] / "escape").exists()


def test_a_failed_gate_exits_1_and_writes_nothing(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    write_world(world, make_frame(n_pos=6, n_neg=60))
    assert run(world) == 1
    assert "gate failed" in capsys.readouterr().err and not world["cv"].exists()


def test_overwrite_with_different_content_needs_force(world: dict[str, Any]) -> None:
    assert run(world) == 0
    before = snapshot(world["cv"])
    assert run(world, "--seed", "1") == 1
    assert snapshot(world["cv"]) == before  # nothing changed
    assert run(world, "--seed", "1", "--force") == 0
    assert snapshot(world["cv"]) != before
    assert run(world, "--seed", "1") == 0  # identical content is not an overwrite


def test_load_splits_roundtrip_and_tamper_detection(world: dict[str, Any]) -> None:
    assert run(world) == 0
    path = world["cv"] / "ge2" / "splits.csv"
    frame = pd.read_csv(world["proc"] / "features.csv", dtype={"ref": str, "participant": str})
    splits = cv.load_splits(path, frame)
    assert (
        cv.splits_sha8(splits)
        == json.loads(path.with_name("splits_meta.json").read_text())["splits_sha8"]
    )

    good = path.read_bytes()
    frame_small = frame[frame["participant"] != frame["participant"].iloc[0]]
    with pytest.raises(cv.GuardFailure, match="missing or extra"):
        cv.load_splits(path, frame_small)  # participant sets differ

    text = good.decode().splitlines()
    first = text[1].split(",")
    first[2] = str((int(first[2]) + 1) % 5)
    path.write_bytes(("\n".join([text[0], ",".join(first), *text[2:]]) + "\n").encode())
    with pytest.raises(cv.GuardFailure, match="file hash"):
        cv.load_splits(path, frame)

    path.write_bytes(good)
    meta_path = path.with_name("splits_meta.json")
    meta = json.loads(meta_path.read_text())
    meta["splits_sha8"] = "00000000"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(cv.GuardFailure, match="row hash"):
        cv.load_splits(path, frame)

    meta_path.unlink()
    with pytest.raises(cv.InputProblem):
        cv.load_splits(path, frame)


def test_load_splits_runs_the_partition_check_even_when_the_hashes_agree(
    world: dict[str, Any],
) -> None:
    assert run(world) == 0
    path = world["cv"] / "ge2" / "splits.csv"
    meta_path = path.with_name("splits_meta.json")
    frame = pd.read_csv(world["proc"] / "features.csv", dtype={"ref": str, "participant": str})
    splits = pd.read_csv(path, dtype={"participant": str})
    bad = pd.concat([splits, splits.iloc[[0]]], ignore_index=True)  # one participant twice
    path.write_bytes(cv.splits_bytes(bad))
    meta = json.loads(meta_path.read_text())
    meta["splits_file_sha8"] = cv.sha8_of(path.read_bytes())
    meta["splits_sha8"] = cv.splits_sha8(bad)
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(cv.GuardFailure, match="repeated"):
        cv.load_splits(path, frame)


# --------------------------------------------------------------------------- privacy


def test_no_raw_id_or_label_text_reaches_any_output(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    frame = make_frame().assign(note_raw="CONTROL07_DCHA UPDRS_POISON", hand="IZDA_POISON")
    write_world(world, frame, roles_for(make_frame()))
    assert run(world) == 0
    blob = capsys.readouterr().out
    for p in world["cv"].rglob("*"):
        if p.is_file():
            blob += p.read_text()
    for needle in ("CONTROL", "DCHA", "IZDA", "UPDRS", "POISON", "ID1004", "rating"):
        assert needle not in blob
    ids = set(pd.read_csv(world["cv"] / "ge2" / "splits.csv", dtype=str)["participant"])
    assert ids == set(frame["participant"]) and all(re.fullmatch(r"[0-9a-f]{8}", i) for i in ids)
    assert not set(frame["ref"]) & (set(re.findall(r"[0-9a-f]{8}", blob)) - ids)


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
    assert run(world) == 0
    assert any(p.endswith("features.csv") for p in opened)  # the spy sees reads
    assert not [
        p for p in opened if "fis_diagnostic" in p or "HUBU-FIS" in p or "raw" in Path(p).parts
    ]


def test_source_does_not_name_the_label_file_or_import_other_modules() -> None:
    source = (SRC / "cv_splits.py").read_text()
    assert "fis_diagnostic" not in source and "UPDRS" not in source
    imports = re.findall(r"^\s*(?:from|import)\s+([\w.]+)", source, flags=re.M)
    assert not {"feature_matrix", "hubu_fis", "gait"} & {i.split(".")[0] for i in imports}
    assert "models/gait" not in source


# --------------------------------------------------------------------------- real data


@pytest.mark.skipif(not REAL_PRESENT, reason="real export absent")
def test_real_export_gate_by_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "data" / "processed" / "cv"
    monkeypatch.setattr(cv, "CV_ROOT", root)

    def go(target: str) -> int:
        return cv.main(
            ["--features", str(REAL / "features.csv"), "--roles", str(REAL / "feature_roles.csv"),
             "--meta", str(REAL / "features_meta.json"), "--target", target, "--out-dir", str(root)]
        )  # fmt: skip

    assert go("ge2") == 0
    meta = json.loads((root / "ge2" / "splits_meta.json").read_text())
    assert meta["participants"] == 117 and meta["rows"] == 233
    assert meta["gate"]["outer_test"]["pos_participants"]["min"] >= 5
    assert meta["gate"]["inner_validation"]["pos_participants"]["min"] >= 5
    assert go("ge1") == 0
    capsys.readouterr()
    assert go("ge3") == 1
    err = capsys.readouterr().err
    assert "gate failed (ge3)" in err and not (root / "ge3").exists()
