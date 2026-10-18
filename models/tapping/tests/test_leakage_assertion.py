"""FT-33: a participant on both sides of a split fails the run. Every guard in cv_splits.py and
cv_harness.py gets one test that plants an overlap at that place (by monkeypatching the function
that builds the folds) and checks exit 1, a counts-only message and that no result file is written.
Synthetic data only."""

from __future__ import annotations

import json
import re
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
from test_cv_harness import (  # noqa: E402
    arrays,
    build_world,
    frame_of,
    harness,
    make_data,
    out_dir,
    snapshot,
)

RESULT_PREFIXES = ("oof_", "runs_", "best_params_", "meta_", "permutation_")


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    return build_world(tmp_path, monkeypatch, make_data())


def cv_args(w: dict[str, Any], out: Path) -> list[str]:
    return ["--features", str(w["proc"] / "features.csv"), "--roles", str(w["proc"] / "feature_roles.csv"),
            "--meta", str(w["proc"] / "features_meta.json"), "--target", "ge2", "--repeats", "2",
            "--out-dir", str(out)]  # fmt: skip


def assert_refused(
    w: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
    rc: int,
    needle: str,
    root: Path | None = None,
) -> str:
    """Exit 1, the message names the guard with counts only, and no result file exists."""
    captured = capsys.readouterr()
    assert rc == 1 and needle in captured.err
    frame = frame_of(w)
    hashes = set(frame["participant"]) | set(frame["ref"])
    assert not [h for h in hashes if h in captured.out + captured.err]  # counts only
    found = [p for p in (root or out_dir(w)).rglob("*") if p.name.startswith(RESULT_PREFIXES)]
    assert not found
    return captured.err


def straddle(member: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Move one clip of a two-clip participant into the next fold: that participant is then in
    the training part of one fold and in its test part."""
    out = member.copy()
    for person in sorted(set(groups)):
        pos = np.flatnonzero(groups == person)
        if len(pos) == 2:
            out[pos[1]] = (out[pos[0]] + 1) % 5
            return out
    raise AssertionError("no two-clip participant")


@pytest.fixture
def straddled_membership(monkeypatch: pytest.MonkeyPatch) -> None:
    real = ch.membership
    monkeypatch.setattr(ch, "membership", lambda g, s, r: straddle(real(g, s, r), g))


COMMANDS = [
    ("run", "b0"),
    ("run", "b1"),
    ("run", "lgbm"),
    ("permute", "b0"),
    ("permute", "b1"),
]


def call(w: dict[str, Any], command: str, model: str) -> int:
    extra = ["--draws", "2"] if command == "permute" else ["--repeats", str(w["repeats"])]
    return harness(w, command, "--model", model, *extra)


# --------------------------------------------------------------------------- cv_harness guards


@pytest.mark.parametrize("command,model", COMMANDS)
def test_run_repeat_stops_when_a_participant_is_on_both_sides_of_an_outer_split(
    world: dict[str, Any], straddled_membership: None, capsys: pytest.CaptureFixture[str],
    command: str, model: str,
) -> None:  # fmt: skip
    err = assert_refused(world, capsys, call(world, command, model), "group(s) on both sides")
    assert re.search(
        r"outer repeat 0 fold \d: 1 group\(s\) on both sides \(train groups \d+, test groups \d+\)",
        err,
    )


def overlapping_inner(real: Any) -> Any:
    def planted(
        train: np.ndarray, y: np.ndarray, groups: np.ndarray, repeat: int, outer: int
    ) -> Any:
        return [(train, train[:10])] * 3  # inside the training part, but the same participants

    return planted


@pytest.mark.parametrize("command", ["run", "permute"])
def test_tune_stops_when_an_inner_fold_shares_a_participant(
    world: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    monkeypatch.setattr(ch, "inner_folds", overlapping_inner(ch.inner_folds))
    err = assert_refused(world, capsys, call(world, command, "b1"), "group(s) on both sides")
    assert "inner repeat 0 outer 0" in err


@pytest.mark.parametrize("command", ["run", "permute"])
def test_tune_stops_when_an_inner_fold_uses_rows_outside_the_training_part(
    world: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    n = len(frame_of(world))

    def planted(
        train: np.ndarray, y: np.ndarray, groups: np.ndarray, repeat: int, outer: int
    ) -> Any:
        held_out = np.setdiff1d(
            np.arange(n), train
        )  # participants differ, but the rows are test rows
        return [(train, held_out)] * 3

    monkeypatch.setattr(ch, "inner_folds", planted)
    err = assert_refused(world, capsys, call(world, command, "b1"), "rows outside the part")
    assert re.search(r"inner repeat 0 outer 0: rows outside the part", err)


def forge_splits(w: dict[str, Any]) -> None:
    """A splits file with one participant in two outer folds of repeat 0, with a matching meta."""
    path = out_dir(w) / "splits.csv"
    splits = pd.read_csv(path, dtype={"participant": str})
    first = splits.iloc[0]
    extra = pd.DataFrame(
        [
            {
                "participant": first["participant"],
                "repeat": 0,
                "outer_fold": (first["outer_fold"] + 1) % 5,
            }
        ]
    )
    forged = pd.concat([splits, extra], ignore_index=True)
    path.write_bytes(cvs.splits_bytes(forged))
    meta_path = path.with_name("splits_meta.json")
    meta = json.loads(meta_path.read_text())
    meta["splits_file_sha8"] = cvs.sha8_of(path.read_bytes())
    meta["splits_sha8"] = cvs.splits_sha8(forged)
    meta_path.write_text(json.dumps(meta))


@pytest.mark.parametrize("command,model", [("run", "b1"), ("permute", "b0")])
def test_a_splits_file_with_a_participant_in_two_folds_is_refused(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str], command: str, model: str
) -> None:
    forge_splits(world)  # hashes agree, so only the partition check can catch it
    err = assert_refused(world, capsys, call(world, command, model), "participant row(s) repeated")
    assert re.search(r"repeat 0: 1 participant row\(s\) repeated", err)


def test_refit_outer_asserts_when_given_the_test_positions(world: dict[str, Any]) -> None:
    a = arrays(world)
    train, test = np.flatnonzero(a["member"] != 0), np.flatnonzero(a["member"] == 0)
    ch.refit_outer(
        a["frame"], "b1", {"C": 1.0}, train, "A", "ge2", test_positions=test
    )  # clean fold
    planted = np.append(test, train[0])  # one participant now on both sides
    with pytest.raises(cvs.LeakageError, match=r"refit_outer: \d+ group\(s\) on both sides") as err:
        ch.refit_outer(a["frame"], "b1", {"C": 1.0}, train, "A", "ge2", test_positions=planted)
    assert not set(a["frame"]["participant"]) & set(str(err.value).split())


# --------------------------------------------------------------------------- cv_splits guards


def overlapping_split(real: Any, n_splits: int, mode: str) -> Any:
    def planted(k: int, seed: int, y: np.ndarray, groups: np.ndarray, where: str) -> Any:
        pairs = real(k, seed, y, groups, where)
        if k != n_splits:
            return pairs
        (train0, test0), (train1, test1) = pairs[0], pairs[1]
        if mode == "train_has_test_row":  # one held-out row is also in the training part
            pairs[0] = (np.append(train0, test0[0]), test0)
        else:  # a participant is tested in two folds, each fold on its own disjoint
            rows = np.flatnonzero(groups == groups[test0[0]])
            pairs[1] = (np.setdiff1d(train1, rows), np.union1d(test1, rows))
        return pairs

    return planted


@pytest.mark.parametrize(
    "n_splits,mode,needle",
    [
        (cvs.N_OUTER, "train_has_test_row", "outer repeat 0 fold 0: 1 group(s) on both sides"),
        (
            cvs.N_INNER,
            "train_has_test_row",
            "inner repeat 0 outer 0 fold 0: 1 group(s) on both sides",
        ),
        (cvs.N_OUTER, "tested_twice", "participant row(s) repeated"),
    ],
)
def test_cv_splits_cli_refuses_planted_overlaps(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    n_splits: int, mode: str, needle: str,
) -> None:  # fmt: skip
    capsys.readouterr()
    monkeypatch.setattr(cvs, "_split", overlapping_split(cvs._split, n_splits, mode))
    fresh = world["cv"] / "fresh"
    rc = cvs.main(cv_args(world, fresh))
    assert_refused(world, capsys, rc, needle, root=fresh)
    assert not (fresh / "ge2" / "splits.csv").exists() and not (fresh / "ge2").exists()


def test_outer_table_runs_its_own_partition_check(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Called directly, so the later check in inner_table cannot be what raises.
    monkeypatch.setattr(cvs, "_split", overlapping_split(cvs._split, cvs.N_OUTER, "tested_twice"))
    with pytest.raises(cvs.GuardFailure, match=r"repeat 0: 1 participant row\(s\) repeated"):
        cvs.outer_table(frame_of(world), "ge2", 2, 0)


def test_inner_table_asserts_on_the_outer_membership(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real = cvs._outer_membership
    monkeypatch.setattr(cvs, "_outer_membership", lambda g, s, r: straddle(real(g, s, r), g))
    frame = frame_of(world)
    splits = cvs.load_splits(out_dir(world) / "splits.csv", frame)
    with pytest.raises(
        cvs.LeakageError, match=r"outer repeat 0 fold \d: 1 group\(s\) on both sides"
    ):
        cvs.inner_table(frame, splits, "ge2")


def test_inner_table_runs_the_partition_check_on_the_splits_it_is_given(
    world: dict[str, Any],
) -> None:
    frame = frame_of(world)
    splits = cvs.load_splits(out_dir(world) / "splits.csv", frame)
    dup = pd.concat([splits, splits.iloc[[0]]], ignore_index=True)
    with pytest.raises(cvs.GuardFailure, match="repeated"):
        cvs.inner_table(frame, dup, "ge2")


def test_a_clean_run_writes_results_and_no_guard_fires(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert harness(world, "run", "--model", "b1", "--repeats", "2") == 0
    assert {p.name for p in out_dir(world).iterdir() if p.name.startswith(RESULT_PREFIXES)} >= {
        "oof_b1_A.csv",
        "runs_b1_A.csv",
    }
    assert "both sides" not in capsys.readouterr().err
    assert snapshot(world["cv"])  # something was written
