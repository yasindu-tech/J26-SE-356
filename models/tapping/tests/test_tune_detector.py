"""Tests for models/tapping/src/tune_detector.py (task FT-26, part 1: pre-registration).

Synthetic data only (the FT-24 test world: made-up participants, rates and QC flags) under
temporary roots, except one read-only test of the real frame that skips when data is absent.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import manual_counts as mc  # noqa: E402
import tune_detector as td  # noqa: E402
import validate_counts as vc  # noqa: E402
from test_manual_counts import make_world  # noqa: E402

RAW_ID = re.compile(r"(CONTROL|ID)\d+", re.IGNORECASE)


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    w: dict[str, Any] = dict(make_world(tmp_path))
    kept = list(w["refs"])
    for mod in (mc, td):
        monkeypatch.setattr(mod, "INTERIM_ROOT", w["interim"])
        monkeypatch.setattr(mod, "DOCS_ROOT", w["docs"])
        monkeypatch.setattr(mod, "VIDEO_ROOT", w["videos"])
    monkeypatch.setattr(mc, "PRACTICE", tuple(kept[:3]))
    monkeypatch.setattr(mc, "HARD", tuple(kept[3:19]))
    pop = mc.load_population(Path(w["docs"]), Path(w["interim"]) / "taps" / "taps_summary.csv")
    parts = mc.participant_keys(Path(w["videos"]))
    sel = mc.build_selection(pop, parts)
    seen = set(sel.loc[sel["set"] != "frame", "ref"])
    seen_people = {parts[r][0] for r in seen}
    frame = [r for r in pop["ref"] if r not in seen and parts[r][0] not in seen_people]
    monkeypatch.setattr(td, "EXPECTED_FRAME_CLIPS", len(frame))
    monkeypatch.setattr(td, "EXPECTED_FRAME_PARTICIPANTS", len({parts[r][0] for r in frame}))
    w.update(pop=pop, parts=parts, seen=seen, seen_people=seen_people, frame=frame, sel=sel)
    w["out"] = Path(w["interim"]) / "tuning"
    return w


def run(*argv: str) -> int:
    return td.main(list(argv))


# ----------------------------------------------------------------------- frozen protocol


def test_protocol_constants_are_the_pre_registered_ones() -> None:
    assert td.MAE_TARGET == 1.0
    assert td.GATE_STATUS == "not_met"
    assert td.GATE_MAE_MAX == 1.3
    assert td.PROMINENCE_GRID == (0.15, 0.175, 0.20, 0.225, 0.25)
    assert td.SPACING_GRID == (0.10, 0.15, 0.20)
    assert td.FROZEN == (0.20, 0.15)
    assert td.BIG_ERROR == 3
    assert td.BIG_ERROR_WEIGHT == 0.5
    assert td.MIN_GAIN == 0.25
    assert td.BOOT_LOWER_PERCENTILE == 5.0
    assert td.MIN_STABILITY == 0.80
    assert td.TUNING_SETS == ("hard", "practice")
    assert td.FRESH_SEED == 26
    assert td.EXPECTED_FRAME_CLIPS == 133
    assert td.EXPECTED_FRAME_PARTICIPANTS == 67
    assert td.EXPECTED_TUNING_PRIMARY_OVERLAP == 4
    assert td.MAE_TARGET == vc.MAE_TARGET
    assert td.GATE_STATUS == "not_met"


def test_candidate_grid_is_fixed_in_order_and_contains_the_frozen_pair() -> None:
    expected = (
        (0.15, 0.10), (0.15, 0.15), (0.15, 0.20),
        (0.175, 0.10), (0.175, 0.15), (0.175, 0.20),
        (0.20, 0.10), (0.20, 0.15), (0.20, 0.20),
        (0.225, 0.10), (0.225, 0.15), (0.225, 0.20),
        (0.25, 0.10), (0.25, 0.15), (0.25, 0.20),
    )  # fmt: skip
    assert expected == td.CANDIDATES
    assert td.CANDIDATES.index(td.FROZEN) == 7


def test_allocation_is_literal_and_totals_twenty() -> None:
    assert list(td.ALLOCATION) == list(mc.CELL_ORDER)
    assert list(td.ALLOCATION.values()) == [4, 2, 5, 1, 5, 3]
    assert sum(td.ALLOCATION.values()) == 20
    assert td.FRESH_CELL_ORDER[:3] == ("slow/flagged", "mid/flagged", "fast/flagged")


# ----------------------------------------------------------------------- gate


@pytest.mark.parametrize(
    ("status", "mae", "expected"),
    [
        ("not_met", 1.0001, td.GATE_TUNE),
        ("not_met", 1.3, td.GATE_TUNE),
        ("not_met", 1.31, td.GATE_CLOSE_LIMITATIONS),
        ("not_met", 1.4, td.GATE_CLOSE_LIMITATIONS),
        ("not_met", 2.5, td.GATE_CLOSE_LIMITATIONS),
        ("established", 0.5, td.GATE_CLOSE_FROZEN),
        ("met_on_point_estimate", 1.0, td.GATE_CLOSE_FROZEN),
        ("icc_below_target", 0.9, td.GATE_CLOSE_FROZEN),
    ],
)
def test_gate(status: str, mae: float, expected: str) -> None:
    assert td.decide_gate(status, mae) == expected


@pytest.mark.parametrize(
    ("status", "mae"),
    [("not_met", 0.9), ("established", 1.2), ("not_met", float("nan")), ("not_met", float("inf"))],
)
def test_gate_refuses_inconsistent_input(status: str, mae: float) -> None:
    with pytest.raises(ValueError):
        td.decide_gate(status, mae)


def test_gate_agrees_with_the_ft25_decision() -> None:
    for mae, expected in (
        (0.4, td.GATE_CLOSE_FROZEN),
        (1.0, td.GATE_CLOSE_FROZEN),
        (1.1, td.GATE_TUNE),
        (1.3, td.GATE_TUNE),
        (1.6, td.GATE_CLOSE_LIMITATIONS),
    ):
        status = vc.decide(mae, [mae - 0.1, mae + 0.1], 0.95, [0.9, 0.97])["status"]
        assert td.decide_gate(status, mae) == expected


# ----------------------------------------------------------------------- objective and rule


def test_objective_by_hand() -> None:
    # |d| = 0, 1, 2, 5: mean 2.0; excess over 3 is 0, 0, 0, 2 -> mean 0.5 -> 0.5 * 0.5 = 0.25
    assert td.objective([0, 1, -2, 5]) == pytest.approx(2.25)
    assert td.objective([1, -1, 0, 0]) == pytest.approx(0.5)  # no excess, no penalty
    assert td.objective([3, -3]) == pytest.approx(3.0)  # exactly 3 is not penalised
    assert td.objective([-4]) == pytest.approx(4.5)
    with pytest.raises(ValueError):
        td.objective([])


def test_frozen_parameters_win_ties() -> None:
    flat = [1.0] * 15
    assert td.pick_candidate(flat) == 7
    tie = [1.0] * 15
    tie[2] = tie[7]  # candidate 2 exactly equal to the frozen objective
    assert td.pick_candidate(tie) == 7
    near = [1.0] * 15
    near[2] = 1.0 - 1e-13  # within the tie tolerance
    assert td.pick_candidate(near) == 7


def test_a_strictly_better_candidate_beats_frozen_and_first_wins_among_equals() -> None:
    obj = [1.0] * 15
    obj[4] = obj[10] = 0.5
    assert td.pick_candidate(obj) == 4
    obj[7] = 0.5  # frozen equals the best: frozen wins
    assert td.pick_candidate(obj) == 7
    with pytest.raises(ValueError):
        td.pick_candidate([1.0] * 14)


@pytest.mark.parametrize(
    ("gain", "lower", "stability", "expected"),
    [
        (0.25, 0.01, 0.8, True),
        (0.2499, 0.01, 0.8, False),
        (0.25, 0.0, 0.8, False),
        (0.25, -0.1, 0.8, False),
        (0.25, 0.01, 0.7999, False),
        (1.0, 0.5, 1.0, True),
    ],
)
def test_acceptance_rule(gain: float, lower: float, stability: float, expected: bool) -> None:
    assert td.passes_rule(gain, lower, stability) is expected


def test_tuning_facts_count_the_overlap(world: dict[str, Any]) -> None:
    sel, parts = world["sel"], world["parts"]
    facts = td.tuning_facts(sel, parts)
    tuning = sel.loc[sel["set"].isin(["hard", "practice"]), "ref"]
    primary = sel.loc[sel["set"] == "primary", "ref"]
    t = {parts[r][0] for r in tuning}
    p = {parts[r][0] for r in primary}
    assert facts == {
        "tuning_clips": 19,
        "tuning_participants": len(t),
        "tuning_primary_overlap_participants": len(t & p),
    }


# ----------------------------------------------------------------------- the fresh frame


def test_frame_excludes_every_seen_participant(world: dict[str, Any]) -> None:
    frame = td.build_frame(world["pop"], world["parts"])
    parts = world["parts"]
    people = {parts[r][0] for r in frame.index}
    assert not people & world["seen_people"]
    assert not set(frame.index) & world["seen"]
    assert sorted(frame.index) == sorted(world["frame"])
    # the synthetic world keeps both hands, so excluding clips only would leave seen people in
    clip_only = [r for r in world["pop"]["ref"] if r not in world["seen"]]
    assert len(clip_only) > len(frame)


def test_frame_size_is_asserted(world: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(td, "EXPECTED_FRAME_CLIPS", td.EXPECTED_FRAME_CLIPS + 1)
    with pytest.raises(mc.InputError):
        td.build_fresh(world["pop"], world["parts"])


def cells_by_hand(summary: Path) -> dict[str, str]:
    """Independent recomputation: tertile cuts over all retained clips, then the QC flag."""
    df = pd.read_csv(summary, dtype={"ref": str})
    lo, hi = np.quantile(df["tap_rate_hz"].astype(float), [1 / 3, 2 / 3])
    out = {}
    for ref, rate, status in zip(df["ref"], df["tap_rate_hz"], df["qc_status"], strict=True):
        tertile = "slow" if rate <= lo else "mid" if rate <= hi else "fast"
        out[ref] = f"{tertile}/{'flagged' if status == 'retained_flag' else 'unflagged'}"
    return out


def test_cells_come_from_the_frozen_summary_and_the_tertile_cuts(world: dict[str, Any]) -> None:
    summary = Path(world["interim"]) / "taps" / "taps_summary.csv"
    expected = cells_by_hand(summary)
    frame = td.build_frame(world["pop"], world["parts"])
    assert frame.to_dict() == {r: expected[r] for r in frame.index}
    sel = td.build_fresh(world["pop"], world["parts"])
    assert sel.set_index("ref")["cell"].to_dict() == {r: expected[r] for r in sel["ref"]}


# ----------------------------------------------------------------------- the draw


def test_draw_cell_counts_participants_and_weights(world: dict[str, Any]) -> None:
    sel = td.build_fresh(world["pop"], world["parts"])
    parts = world["parts"]
    fresh = sel[sel["set"] == "fresh"]
    assert len(fresh) == 20
    for cell, n in zip(mc.CELL_ORDER, (4, 2, 5, 1, 5, 3), strict=True):
        assert (fresh["cell"] == cell).sum() == n
        frame_n = int((sel["cell"] == cell).sum())  # the frame is the whole table
        assert (fresh.loc[fresh["cell"] == cell, "weight"].astype(float) == frame_n / n).all()
    assert float(fresh["weight"].astype(float).sum()) == pytest.approx(len(sel))
    people = [parts[r][0] for r in fresh["ref"]]
    assert len(set(people)) == 20
    assert not set(people) & world["seen_people"]
    assert sorted(fresh["order"].astype(int)) == list(range(1, 21))
    assert (sel["seed"] == 26).all()
    assert list(sel.columns) == list(td.SELECTION_COLUMNS)


def test_weights_are_hand_checked_on_a_small_case(world: dict[str, Any]) -> None:
    sel = td.build_fresh(world["pop"], world["parts"])
    n = sel["cell"].value_counts()
    fresh = sel[sel["set"] == "fresh"].set_index("ref")["weight"].astype(float)
    cell = sel.set_index("ref")["cell"]
    assert {c: fresh[cell[fresh.index] == c].iloc[0] for c in mc.CELL_ORDER} == {
        "slow/unflagged": n["slow/unflagged"] / 4,
        "slow/flagged": n["slow/flagged"] / 2,
        "mid/unflagged": n["mid/unflagged"] / 5,
        "mid/flagged": n["mid/flagged"] / 1,
        "fast/unflagged": n["fast/unflagged"] / 5,
        "fast/flagged": n["fast/flagged"] / 3,
    }


def test_one_clip_per_participant_for_many_seeds(world: dict[str, Any]) -> None:
    for seed in range(1, 25):
        sel = td.build_fresh(world["pop"], world["parts"], seed=seed)
        fresh = sel[sel["set"] == "fresh"]
        assert fresh["ref"].map(lambda r: world["parts"][r][0]).is_unique


def test_same_seed_identical_and_other_seed_differs(world: dict[str, Any]) -> None:
    a = td.build_fresh(world["pop"], world["parts"])
    b = td.build_fresh(world["pop"], world["parts"])
    c = td.build_fresh(world["pop"], world["parts"], seed=27)
    assert a.equals(b)
    assert set(a.loc[a["set"] == "fresh", "ref"]) != set(c.loc[c["set"] == "fresh", "ref"])


def test_counting_order_is_the_fixed_shuffle(world: dict[str, Any]) -> None:
    import random

    sel = td.build_fresh(world["pop"], world["parts"])
    fresh = sorted(sel.loc[sel["set"] == "fresh", "ref"])
    random.Random("26:order").shuffle(fresh)
    got = sel[sel["set"] == "fresh"].set_index("ref")["order"].astype(int)
    assert [got[r] for r in fresh] == list(range(1, 21))


# ----------------------------------------------------------------------- commands


def test_select_check_and_refuse_overwrite(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    out = Path(world["out"])
    assert run("select") == 0
    assert (out / td.SELECTION_NAME).is_file()
    first = (out / td.SELECTION_NAME).read_bytes()
    capsys.readouterr()
    assert run("select") == 2
    assert "pre-registered" in capsys.readouterr().err
    assert (out / td.SELECTION_NAME).read_bytes() == first
    assert run("check") == 0
    assert run("select", "--force") == 0
    assert (out / td.SELECTION_NAME).read_bytes() == first  # same inputs, same draw


def test_check_finds_a_tampered_file(world: dict[str, Any]) -> None:
    out = Path(world["out"]) / td.SELECTION_NAME
    assert run("select") == 0
    df = pd.read_csv(out, dtype=str, keep_default_na=False)
    row = df.index[df["set"] == "fresh"][0]
    df.loc[row, "cell"] = (
        "fast/flagged" if df.loc[row, "cell"] != "fast/flagged" else "slow/flagged"
    )
    df.to_csv(out, index=False)
    assert run("check") == 1
    df = pd.read_csv(out, dtype=str, keep_default_na=False)
    df.loc[df["set"] == "fresh", "seed"] = "27"
    df.to_csv(out, index=False)
    assert run("check") == 1


def test_check_without_a_file_is_an_input_error(world: dict[str, Any]) -> None:
    assert run("check") == 2


def test_path_guards(world: dict[str, Any], tmp_path: Path) -> None:
    assert run("select", "--out-dir", str(tmp_path / "elsewhere")) == 2
    assert run("select", "--summary", str(tmp_path / "taps_summary.csv")) == 2
    assert run("select", "--docs-dir", str(tmp_path / "other_docs")) == 2
    assert not (tmp_path / "elsewhere").exists()


def test_the_default_summary_is_the_frozen_taps_folder(world: dict[str, Any]) -> None:
    interim = Path(world["interim"])
    args = td.build_parser().parse_args(["select"])
    assert td._paths(args)[2] == interim / "taps" / "taps_summary.csv"
    assert td._paths(args)[0] == interim / "tuning"


def test_an_explicit_summary_moves_the_cells(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    interim = Path(world["interim"])
    other = interim / "taps_candidate" / "taps_summary.csv"
    other.parent.mkdir()
    shifted = pd.read_csv(interim / "taps" / "taps_summary.csv", dtype=str)
    shifted["tap_rate_hz"] = shifted["tap_rate_hz"].astype(float).iloc[::-1].to_numpy().astype(str)
    shifted.to_csv(other, index=False)
    assert run("select") == 0
    pop2 = mc.load_population(Path(world["docs"]), other)
    frame2 = td.build_frame(pop2, world["parts"])
    assert frame2.to_dict() != td.build_frame(world["pop"], world["parts"]).to_dict()
    # the frame depends on the FT-24 draw, which depends on the summary: the size check bites
    assert run("select", "--summary", str(other), "--out-dir", str(interim / "tuning_b")) == 2
    monkeypatch.setattr(td, "EXPECTED_FRAME_CLIPS", len(frame2))
    monkeypatch.setattr(
        td,
        "EXPECTED_FRAME_PARTICIPANTS",
        len({world["parts"][r][0] for r in frame2.index}),
    )
    assert run("select", "--summary", str(other), "--out-dir", str(interim / "tuning_b")) == 0
    for name, summary in (("tuning", interim / "taps" / "taps_summary.csv"), ("tuning_b", other)):
        got = pd.read_csv(interim / name / td.SELECTION_NAME, dtype=str, keep_default_na=False)
        expected = cells_by_hand(summary)
        assert got.set_index("ref")["cell"].to_dict() == {r: expected[r] for r in got["ref"]}


def test_replacement_takes_next_pool_clip_of_the_cell(world: dict[str, Any]) -> None:
    parts = world["parts"]
    sel = td.build_fresh(world["pop"], parts).astype(str)
    bad = sel.loc[sel["set"] == "fresh", "ref"].iloc[0]
    cell = sel.set_index("ref").at[bad, "cell"]
    new, messages = td.replace_unreadable(sel, [bad], parts)
    assert messages == [f"{bad}: replaced by {new.loc[new['replaced_for'] == bad, 'ref'].iloc[0]}"]
    got = new[new["replaced_for"] == bad].iloc[0]
    assert got["set"] == "fresh" and got["cell"] == cell and got["order"] == "21"
    assert new.set_index("ref").at[bad, "set"] == "replaced"
    used = {parts[r][0] for r in new.loc[new["set"] == "fresh", "ref"]}
    assert len(used) == 20
    pool = sel[(sel["set"] == "pool") & (sel["cell"] == cell)]
    ranked = pool.sort_values("cell_rank", key=lambda s: s.astype(int))["ref"].tolist()
    others = {parts[r][0] for r in sel.loc[sel["set"] == "fresh", "ref"] if r != bad}
    assert got["ref"] == next(r for r in ranked if parts[r][0] not in others)
    assert not td.validate_fresh(new, parts, None, None)
    again, _ = td.replace_unreadable(new, [bad], parts)
    assert again.equals(new)


def test_only_fresh_clips_are_replaced(world: dict[str, Any]) -> None:
    parts = world["parts"]
    sel = td.build_fresh(world["pop"], parts).astype(str)
    pool_ref = sel.loc[sel["set"] == "pool", "ref"].iloc[0]
    new, messages = td.replace_unreadable(sel, [pool_ref, "00000000"], parts)
    assert new.equals(sel) and len(messages) == 2


def test_outputs_hold_no_raw_ids_or_labels(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("select") == 0
    assert run("check") == 0
    shown = capsys.readouterr()
    text = shown.out + shown.err + (Path(world["out"]) / td.SELECTION_NAME).read_text()
    assert not RAW_ID.search(text)
    assert "UPDRS" not in text and "DCHA" not in text and "IZDA" not in text
    refs = set(world["pop"]["ref"])
    printed = set(re.findall(r"\b[0-9a-f]{8}\b", shown.out + shown.err))
    assert not printed & refs  # the console shows counts only, never refs


# ----------------------------------------------------------------------- cross-check with FT-24


def ft24_file(world: dict[str, Any]) -> Path:
    assert mc.main(["select"]) == 0
    return Path(world["interim"]) / "manual_counts" / mc.SELECTION_NAME


def tamper_sibling(world: dict[str, Any], path: Path, new_set: str) -> None:
    """Give a plain-frame clip of a fresh participant another set, as a replacement would."""
    parts = world["parts"]
    fresh = td.build_fresh(world["pop"], parts)
    people = {parts[r][0] for r in fresh.loc[fresh["set"] == "fresh", "ref"]}
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    hit = df.index[(df["set"] == "frame") & df["ref"].map(lambda r: parts[r][0] in people)]
    assert len(hit) > 0
    df.loc[hit[0], "set"] = new_set
    df.to_csv(path, index=False)


def test_cross_check_passes_and_reports(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    ft24 = ft24_file(world)
    before = ft24.read_bytes()
    capsys.readouterr()
    assert run("select") == 0
    msg = "cross-check with manual_counts selection: 20 participants checked, 0 shared"
    assert msg in capsys.readouterr().out
    assert run("check") == 0
    assert msg in capsys.readouterr().out
    assert ft24.read_bytes() == before  # read-only, plain frame rows of the same people allowed


def test_cross_check_skips_with_a_note_when_the_file_is_missing(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    note = "note: manual_counts selection.csv not found, cross-check skipped"
    assert run("select") == 0
    assert note in capsys.readouterr().out
    assert run("check") == 0
    assert note in capsys.readouterr().out


@pytest.mark.parametrize("new_set", ["primary", "replaced", "hard", "practice"])
def test_a_shared_participant_fails_check_with_exit_1(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str], new_set: str
) -> None:
    ft24 = ft24_file(world)
    assert run("select") == 0
    tamper_sibling(world, ft24, new_set)
    before = ft24.read_bytes()
    capsys.readouterr()
    assert run("check") == 1
    out = capsys.readouterr().out
    assert "cross-check failed: 1 of 20 fresh participants also belong to the manual_counts" in out
    assert f"{new_set} 1" in out
    assert not set(re.findall(r"\b[0-9a-f]{8}\b", out)) & set(world["pop"]["ref"])
    assert ft24.read_bytes() == before


@pytest.mark.parametrize("new_set", ["primary", "replaced", "hard", "practice"])
def test_a_shared_participant_stops_select_with_exit_2_and_writes_nothing(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str], new_set: str
) -> None:
    ft24 = ft24_file(world)
    tamper_sibling(world, ft24, new_set)
    before = ft24.read_bytes()
    capsys.readouterr()
    assert run("select") == 2
    assert "cross-check failed" in capsys.readouterr().err
    assert not (Path(world["out"]) / td.SELECTION_NAME).exists()
    assert ft24.read_bytes() == before


def test_a_replacement_in_the_manual_counts_selection_is_caught(world: dict[str, Any]) -> None:
    """A primary clip replaced by a clip of a fresh participant: the replacement row is
    "primary" with replaced_for set and the old row is "replaced"."""
    ft24 = ft24_file(world)
    assert run("select") == 0
    parts = world["parts"]
    fresh = td.build_fresh(world["pop"], parts)
    people = {parts[r][0] for r in fresh.loc[fresh["set"] == "fresh", "ref"]}
    df = pd.read_csv(ft24, dtype=str, keep_default_na=False)
    new = df.index[(df["set"] == "frame") & df["ref"].map(lambda r: parts[r][0] in people)][0]
    old = df.index[df["set"] == "primary"][0]
    df.loc[new, ["set", "replaced_for"]] = ["primary", df.at[old, "ref"]]
    df.loc[old, "set"] = "replaced"
    df.to_csv(ft24, index=False)
    assert run("check") == 1


def test_an_unknown_clip_in_the_manual_counts_selection_is_an_input_error(
    world: dict[str, Any],
) -> None:
    ft24 = ft24_file(world)
    df = pd.read_csv(ft24, dtype=str, keep_default_na=False)
    df.loc[0, "ref"] = "ffffffff"
    df.to_csv(ft24, index=False)
    assert run("select") == 2
    assert not (Path(world["out"]) / td.SELECTION_NAME).exists()


def test_replace_also_cross_checks(world: dict[str, Any]) -> None:
    ft24 = ft24_file(world)
    assert run("select") == 0
    fresh_file = Path(world["out"]) / td.SELECTION_NAME
    parts = world["parts"]
    sel = pd.read_csv(fresh_file, dtype=str, keep_default_na=False)
    bad = sel.loc[sel["set"] == "fresh", "ref"].iloc[0]
    (fresh_file.parent / mc.UNREADABLE_NAME).write_text(f"ref,rater,list,utc\n{bad},R1,main,x\n")
    new, _ = td.replace_unreadable(sel, [bad], parts)
    chosen = new.loc[new["replaced_for"] == bad, "ref"].iloc[0]
    df = pd.read_csv(ft24, dtype=str, keep_default_na=False)
    sibling = df.index[
        (df["ref"] != chosen) & df["ref"].map(lambda r: parts[r][0] == parts[chosen][0])
    ][0]
    original = df.copy()
    df.loc[sibling, "set"] = "practice"
    df.to_csv(ft24, index=False)
    before = fresh_file.read_bytes()
    assert run("select", "--replace") == 2
    assert fresh_file.read_bytes() == before  # nothing written
    original.to_csv(ft24, index=False)
    assert run("select", "--replace") == 0
    assert "replaced" in fresh_file.read_text()


def test_manual_dir_must_be_inside_interim(world: dict[str, Any], tmp_path: Path) -> None:
    assert run("select", "--manual-dir", str(tmp_path / "elsewhere")) == 2
    assert not (Path(world["out"]) / td.SELECTION_NAME).exists()


# ----------------------------------------------------------------------- file hash


def test_the_printed_hash_is_the_sha256_of_the_file(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    import hashlib

    path = Path(world["out"]) / td.SELECTION_NAME
    assert run("select") == 0
    out = capsys.readouterr().out
    first = hashlib.sha256(path.read_bytes()).hexdigest()[:8]
    assert f"fresh_selection.csv sha256 {first}" in out
    assert run("check") == 0
    assert f"fresh_selection.csv sha256 {first}" in capsys.readouterr().out
    data = bytearray(path.read_bytes())
    data[data.index(b"26")] = ord("7")  # one byte of the seed column
    path.write_bytes(bytes(data))
    assert run("check") == 1
    second = hashlib.sha256(path.read_bytes()).hexdigest()[:8]
    assert second != first
    assert f"fresh_selection.csv sha256 {second}" in capsys.readouterr().out
    assert td.file_hash8(path) == second


# ----------------------------------------------------------------------- the real frame


def test_real_frame_has_133_clips_from_67_participants(capsys: pytest.CaptureFixture[str]) -> None:
    summary = td.INTERIM_ROOT / "taps" / "taps_summary.csv"
    if not (
        summary.is_file()
        and td.VIDEO_ROOT.is_dir()
        and (td.DOCS_ROOT / "qc_retained.csv").is_file()
    ):
        pytest.skip("real data not available")
    pop = mc.load_population(td.DOCS_ROOT, summary)
    parts = mc.participant_keys(td.VIDEO_ROOT)
    frame = td.build_frame(pop, parts)
    people = {parts[r][0] for r in frame.index}
    facts = td.tuning_facts(mc.build_selection(pop, parts), parts)
    sel = td.build_fresh(pop, parts)
    per_cell = sel[sel["set"] == "fresh"]["cell"].value_counts().reindex(mc.CELL_ORDER).tolist()
    with capsys.disabled():
        print(
            f"\nreal frame: {len(frame)} clips, {len(people)} participants; "
            f"fresh per cell {per_cell}; {facts}"
        )
    assert (len(frame), len(people)) == (133, 67)
    assert per_cell == [4, 2, 5, 1, 5, 3]
    assert facts["tuning_primary_overlap_participants"] == 4
    assert facts["tuning_clips"] == 19
