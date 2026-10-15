"""Tests for models/tapping/src/manual_counts.py (task FT-24 selection and bookkeeping).

Synthetic data only: made-up participants (empty video files with made-up numbers), made-up
rates and QC flags, under temporary roots (INTERIM_ROOT, DOCS_ROOT, VIDEO_ROOT patched).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hubu_fis  # noqa: E402
import manual_counts as mc  # noqa: E402

N_PEOPLE = 118


def make_world(root: Path, rejected: int = 1) -> dict[str, object]:
    """Videos, retained list and taps summary for N_PEOPLE made-up participants (2 hands)."""
    rng = random.Random(7)
    video_dir = root / "raw" / "videos"
    video_dir.mkdir(parents=True)
    (video_dir.parent / "fis_diagnostic.csv").write_text("ID,UPDRS\nCONTROL01_DCHA,0\n")
    stems = [
        f"{g}{n:02d}_{h}"
        for n, g in enumerate(["CONTROL", "ID"] * (N_PEOPLE // 2))
        for h in ("DCHA", "IZDA")
    ]
    for stem in stems:
        (video_dir / f"{stem}.mp4").write_bytes(b"")
    refs = sorted(hubu_fis.pseudonym(s) for s in stems)
    rows = [
        {
            "ref": r,
            "tap_rate_hz": round(rng.uniform(0.4, 3.8), 4),
            "qc_status": "retained_flag" if rng.random() < 0.22 else "retained",
            "fps": "59.96" if i < 2 else "30.04",
            "low_range": str(rng.random() < 0.05),
            "n_one_sided": str(rng.randint(0, 2)),
        }
        for i, r in enumerate(refs)
    ]
    summary = pd.DataFrame(rows[rejected:])  # the first ref is the rejected clip
    interim, docs = root / "interim", root / "docs"
    (interim / "taps").mkdir(parents=True)
    docs.mkdir()
    summary.to_csv(interim / "taps" / "taps_summary.csv", index=False)
    verdict = summary["qc_status"].map({"retained": "ok", "retained_flag": "ok_flag"})
    pd.DataFrame({"ref": summary["ref"], "verdict": verdict, "note": ""}).to_csv(
        docs / "qc_retained.csv", index=False
    )
    kept = summary["ref"].tolist()
    return {
        "interim": interim,
        "docs": docs,
        "videos": video_dir,
        "refs": kept,
        "rejected": refs[0],
    }


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    w = make_world(tmp_path)
    kept = list(w["refs"])  # type: ignore[call-overload]
    monkeypatch.setattr(mc, "INTERIM_ROOT", w["interim"])
    monkeypatch.setattr(mc, "DOCS_ROOT", w["docs"])
    monkeypatch.setattr(mc, "VIDEO_ROOT", w["videos"])
    monkeypatch.setattr(mc, "PRACTICE", tuple(kept[:3]))
    monkeypatch.setattr(mc, "HARD", tuple(kept[3:19]))
    w["out"] = Path(w["interim"]) / "manual_counts"  # type: ignore[arg-type]
    return w


def inputs(w: dict[str, object]) -> tuple[pd.DataFrame, dict[str, tuple[str, str]]]:
    pop = mc.load_population(Path(w["docs"]), Path(w["interim"]) / "taps" / "taps_summary.csv")  # type: ignore[arg-type]
    return pop, mc.participant_keys(Path(w["videos"]))  # type: ignore[arg-type]


def selection(w: dict[str, object]) -> tuple[pd.DataFrame, dict[str, tuple[str, str]]]:
    pop, people = inputs(w)
    return mc.build_selection(pop, people).astype(str), people


# --------------------------------------------------------------------------- the draw


def test_selection_is_reproducible_and_seed_dependent(world: dict[str, object]) -> None:
    pop, people = inputs(world)
    a, b = mc.build_selection(pop, people), mc.build_selection(pop, people)
    pd.testing.assert_frame_equal(a, b)
    other = mc.build_selection(pop, people, seed=99)
    assert set(a.loc[a["set"] == "primary", "ref"]) != set(
        other.loc[other["set"] == "primary", "ref"]
    )


def test_draw_matches_the_written_rule(world: dict[str, object]) -> None:
    """An independent re-implementation of the docstring's draw gives the same primary set."""
    pop, people = inputs(world)
    lo, hi = np.quantile(pop["tap_rate_hz"].to_numpy(dtype=float), [1 / 3, 2 / 3])
    seen = set(mc.PRACTICE) | set(mc.HARD)
    cell = {}
    for ref, rate, status in zip(pop["ref"], pop["tap_rate_hz"], pop["qc_status"], strict=True):
        t = "slow" if rate <= lo else "mid" if rate <= hi else "fast"
        cell[ref] = f"{t}/{'flagged' if status == 'retained_flag' else 'unflagged'}"
    rng = random.Random(24)
    taken: set[str] = set()
    expected: list[str] = []
    for c, n in zip(mc.CELL_ORDER, (9, 4, 9, 4, 9, 5), strict=True):
        pool = sorted(r for r in cell if cell[r] == c and r not in seen)
        rng.shuffle(pool)
        for r in pool:
            if sum(cell[x] == c for x in expected) == n:
                break
            if people[r][0] not in taken:
                expected.append(r)
                taken.add(people[r][0])
    sel = mc.build_selection(pop, people)
    assert set(sel.loc[sel["set"] == "primary", "ref"]) == set(expected)


def test_allocation_weights_and_validation(world: dict[str, object]) -> None:
    sel, people = selection(world)
    primary = sel[sel["set"] == "primary"]
    assert primary["cell"].value_counts().to_dict() == mc.ALLOCATION
    frame = sel[sel["set"].isin(["primary", "frame"])]
    assert len(frame) == len(world["refs"]) - 19  # type: ignore[arg-type]
    assert np.isclose(primary["weight"].astype(float).sum(), len(frame))
    for cell, n in mc.ALLOCATION.items():
        w = primary.loc[primary["cell"] == cell, "weight"].astype(float)
        assert np.allclose(w, (frame["cell"] == cell).sum() / n)
    assert mc.validate_selection(sel, people, sel) == []


def test_exactly_the_fixed_allocation_with_real_numbers() -> None:
    assert list(mc.ALLOCATION.values()) == [9, 4, 9, 4, 9, 5]
    assert mc.CELL_ORDER[0] == "slow/unflagged" and mc.CELL_ORDER[-1] == "fast/flagged"
    assert (mc.SEED, mc.SECOND_SEED) == (24, 25)
    assert len(mc.PRACTICE) == 3 and len(mc.HARD) == 16
    assert not set(mc.PRACTICE) & set(mc.HARD)
    assert set(mc.PRACTICE) == {"001ee42d", "9c76e15b", "6b06ed76"}


def test_one_clip_per_participant(world: dict[str, object]) -> None:
    sel, people = selection(world)
    who = [people[r][0] for r in sel.loc[sel["set"] == "primary", "ref"]]
    assert len(who) == len(set(who)) == 40


def test_participant_rule_actually_skips(world: dict[str, object]) -> None:
    """With only two participants per cell on offer, the rule must leave a cell short."""
    pop, people = inputs(world)
    same = {r: ("one-person", "left") for r in people}
    with pytest.raises(mc.InputError, match="only 1 of 9"):
        mc.build_selection(pop, same)


def test_seen_and_rejected_clips_are_never_drawn(world: dict[str, object]) -> None:
    sel, _ = selection(world)
    seen = set(mc.PRACTICE) | set(mc.HARD)
    assert not seen & set(sel.loc[sel["set"].isin(["primary", "frame"]), "ref"])
    assert set(sel.loc[sel["set"] == "practice", "ref"]) == set(mc.PRACTICE)
    assert set(sel.loc[sel["set"] == "hard", "ref"]) == set(mc.HARD)
    assert world["rejected"] not in set(sel["ref"])


def test_tertile_cuts_use_all_retained_clips(world: dict[str, object]) -> None:
    pop, _ = inputs(world)
    lo, hi = np.quantile(pop["tap_rate_hz"].to_numpy(dtype=float), [1 / 3, 2 / 3])
    got = mc.cells(pop)
    for ref, rate in zip(pop["ref"], pop["tap_rate_hz"], strict=True):
        tertile = "slow" if rate <= lo else "mid" if rate <= hi else "fast"
        assert got[ref].split("/")[0] == tertile


def test_second_rater_subset(world: dict[str, object]) -> None:
    sel, _ = selection(world)
    second = sel[sel["second_rater"] == "1"]
    assert len(second) == 12 and (second["set"] == "primary").all()
    for t in mc.TERTILES:
        assert (second["cell"] == f"{t}/unflagged").sum() == 3
        assert (second["cell"] == f"{t}/flagged").sum() == 1
    rng = random.Random(25)
    expected: list[str] = []
    primary = sel[sel["set"] == "primary"]
    for t in mc.TERTILES:
        for f, n in (("unflagged", 3), ("flagged", 1)):
            pool = sorted(primary.loc[primary["cell"] == f"{t}/{f}", "ref"])
            rng.shuffle(pool)
            expected += pool[:n]
    assert set(second["ref"]) == set(expected)


# --------------------------------------------------------------------------- replacement


def test_replacement_takes_next_unused_clip_of_the_cell(world: dict[str, object]) -> None:
    sel, people = selection(world)
    bad = sel[(sel["set"] == "primary") & (sel["second_rater"] == "1")].iloc[0]
    used = {people[r][0] for r in sel.loc[sel["set"] == "primary", "ref"] if r != bad["ref"]}
    pool = sel[(sel["set"] == "frame") & (sel["cell"] == bad["cell"])]
    pool = pool.sort_values("cell_rank", key=lambda s: s.astype(int))
    expected = next(r for r in pool["ref"] if people[r][0] not in used)
    new, messages = mc.replace_unreadable(sel, [bad["ref"]], people)
    row = new.set_index("ref").loc[expected]
    assert row["set"] == "primary" and row["replaced_for"] == bad["ref"]
    assert row["second_rater"] == "1" and row["weight"] == bad["weight"]
    assert new.set_index("ref").at[bad["ref"], "set"] == "replaced"
    assert messages == [f"{bad['ref']}: replaced by {expected}"]
    assert mc.validate_selection(new, people, sel) == []
    again, more = mc.replace_unreadable(new, [bad["ref"]], people)  # idempotent
    pd.testing.assert_frame_equal(again, new)
    assert more == []


def test_replacement_skips_a_used_participant(world: dict[str, object]) -> None:
    sel, people = selection(world)
    bad = sel[sel["set"] == "primary"].iloc[0]
    pool = sel[(sel["set"] == "frame") & (sel["cell"] == bad["cell"])]
    first = pool.sort_values("cell_rank", key=lambda s: s.astype(int))["ref"].iloc[0]
    other = sel[sel["set"] == "primary"].iloc[1]["ref"]
    people = dict(people)
    people[first] = people[other]  # the next clip belongs to a person already in the sample
    new, _ = mc.replace_unreadable(sel, [bad["ref"]], people)
    assert new.set_index("ref").at[first, "set"] == "frame"


def test_only_primary_clips_are_replaced(world: dict[str, object]) -> None:
    sel, people = selection(world)
    new, messages = mc.replace_unreadable(sel, [mc.HARD[0]], people)
    pd.testing.assert_frame_equal(new, sel)
    assert "not replaced" in messages[0]


def test_select_refuses_to_overwrite(
    world: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    assert mc.main(["select"]) == 0
    capsys.readouterr()
    assert mc.main(["select"]) == 2
    assert "breaks the pre-registration" in capsys.readouterr().err
    assert mc.main(["select", "--force"]) == 0


def test_select_replace_reads_unreadable_list(
    world: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    assert mc.main(["select"]) == 0
    out = Path(world["out"])  # type: ignore[arg-type]
    sel = mc.read_selection(out)
    bad = sel.loc[sel["set"] == "primary", "ref"].iloc[0]
    mc.add_unreadable(out, bad, "R1", "main")
    assert mc.main(["select", "--replace"]) == 0
    assert mc.read_selection(out).set_index("ref").at[bad, "set"] == "replaced"
    assert mc.main(["check"]) == 0


def test_labels_file_is_never_opened(
    world: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_: object, **__: object) -> None:
        raise AssertionError("labels must not be read")

    real_read_csv = pd.read_csv

    def guarded(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
        if "fis_diagnostic" in str(path):
            raise AssertionError("the labels file was opened")
        return real_read_csv(path, *args, **kwargs)  # type: ignore[call-overload,no-any-return]

    monkeypatch.setattr(hubu_fis, "load_labels", refuse)
    monkeypatch.setattr(pd, "read_csv", guarded)
    assert mc.main(["select"]) == 0
    assert mc.main(["check"]) == 0
    assert "load_labels" not in Path(mc.__file__).read_text()
    assert "fis_diagnostic" not in Path(mc.__file__).read_text()


# --------------------------------------------------------------------------- orders and files


def test_rater_orders_are_fixed_and_differ(world: dict[str, object]) -> None:
    sel, _ = selection(world)
    r1 = mc.list_refs(sel, "R1", "main")
    expected = sorted(sel.loc[sel["set"] == "primary", "ref"])
    random.Random("24:R1").shuffle(expected)
    assert r1 == expected and r1 != sorted(r1)
    assert mc.list_refs(sel, "R1", "main") == r1
    r2 = mc.list_refs(sel, "R2", "main")
    assert set(r2) == set(sel.loc[sel["second_rater"] == "1", "ref"])
    assert r2 != [r for r in r1 if r in set(r2)]  # not the R1 order
    with pytest.raises(mc.InputError):
        mc.list_refs(sel, "R2", "practice")
    with pytest.raises(mc.InputError):
        mc.list_refs(sel, "R1", "hard")


def test_replacements_come_last_in_the_order(world: dict[str, object]) -> None:
    sel, people = selection(world)
    before = mc.list_refs(sel, "R1", "main")
    new, _ = mc.replace_unreadable(sel, [before[5]], people)
    after = mc.list_refs(new, "R1", "main")
    assert after[:-1] == [r for r in before if r != before[5]]
    assert new.set_index("ref").at[after[-1], "replaced_for"] == before[5]
    r2_before = mc.list_refs(sel, "R2", "main")
    newer, _ = mc.replace_unreadable(sel, [r2_before[2]], people)
    r2_after = mc.list_refs(newer, "R2", "main")
    assert r2_after[:-1] == [r for r in r2_before if r != r2_before[2]] and len(r2_after) == 12
    assert mc.validate_selection(newer, people, sel) == []


ODD_REFS = ("00012345", "1234e567", "0000e000", "12345678")


def row_for(ref: str, **kw: object) -> dict[str, object]:
    row: dict[str, object] = {
        "rater": "R1", "order": 1, "ref": ref, "n_taps": 2, "tap_frames": "10;40",
        "tap_times_s": "0.333;1.334", "n_edge_partials": 1, "confidence": 3, "note": "",
        "started_utc": "2026-10-16T08:00:00Z", "finished_utc": "2026-10-16T08:03:00Z",
        "tool_version": "t",
    }  # fmt: skip
    row.update(kw)
    return row


def test_csv_round_trip_keeps_refs_as_text(tmp_path: Path) -> None:
    path = tmp_path / "R1.csv"
    for ref in ODD_REFS:
        mc.append_row(path, row_for(ref))
    back = mc.read_rows(path)
    assert back["ref"].tolist() == list(ODD_REFS)
    assert back["tap_times_s"].iloc[0] == "0.333;1.334"
    assert mc.check_rows(back, "R1", list(ODD_REFS), "R1.csv") == []
    sel = pd.DataFrame({c: ["x"] * 4 for c in mc.SELECTION_COLUMNS})
    sel["ref"] = ODD_REFS
    mc.atomic_csv(sel, tmp_path / "s.csv")
    assert mc.read_text(tmp_path / "s.csv")["ref"].tolist() == list(ODD_REFS)


@pytest.mark.parametrize(
    "note", ["control group", "PD", "slow, Parkinson", "CONTROL01_DCHA", "ID1004", "id_1004 blur"]
)
def test_bad_notes_are_rejected(note: str, tmp_path: Path) -> None:
    assert mc.note_problem(note) is not None
    df = pd.DataFrame([row_for("12345678", note=note)]).astype(str)
    errors = mc.check_rows(df, "R1", ["12345678"], "R1.csv")
    assert errors and all(note not in e for e in errors)


@pytest.mark.parametrize("note", ["", "blur at the end", "motor_control_unclear", "fast; hidden"])
def test_good_notes_pass(note: str) -> None:
    assert mc.note_problem(note) is None


@pytest.mark.parametrize(
    ("change", "text"),
    [
        ({"n_taps": 3}, "does not match"),
        ({"tap_frames": "40;10"}, "not increasing"),
        ({"confidence": 4}, "confidence"),
        ({"n_taps": "x"}, "unreadable numbers"),
        ({"rater": "R2"}, "rater must be"),
    ],
)
def test_check_rows_rules(change: dict[str, object], text: str) -> None:
    df = pd.DataFrame([row_for("12345678", **change)]).astype(str)
    assert any(text in e for e in mc.check_rows(df, "R1", ["12345678"], "R1.csv"))


def test_check_rows_rejects_foreign_refs() -> None:
    df = pd.DataFrame([row_for("12345678")]).astype(str)
    assert any("not in this list" in e for e in mc.check_rows(df, "R1", ["87654321"], "R1.csv"))


# --------------------------------------------------------------------------- check and export


def fill(out: Path, rater: str, kind: str, refs: list[str]) -> None:
    """One row per clip; the tap frames differ per clip and rater, so a mix-up shows."""
    for k, ref in enumerate(refs, 1):
        base = 10 + k + (500 if rater == "R2" else 0) + (200 if kind == "hard" else 0)
        frames = [base, base + 25]
        row = row_for(
            ref,
            rater=rater,
            order=k,
            tap_frames=";".join(map(str, frames)),
            tap_times_s=";".join(f"{f / 30:.3f}" for f in frames),
        )
        mc.append_row(mc.rater_path(out, rater, kind), row)


def test_export_refuses_an_incomplete_set(
    world: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    out = Path(world["out"])  # type: ignore[arg-type]
    assert mc.main(["select"]) == 0
    sel = mc.read_selection(out)
    fill(out, "R1", "main", mc.list_refs(sel, "R1", "main")[:-1])
    fill(out, "R2", "main", mc.list_refs(sel, "R2", "main"))
    assert mc.main(["export"]) == 1
    assert "refused" in capsys.readouterr().err
    assert not (Path(world["docs"]) / mc.EXPORT_NAME).exists()  # type: ignore[arg-type]
    fill(out, "R1", "main", mc.list_refs(sel, "R1", "main")[-1:])
    mc.atomic_csv(
        mc.read_rows(mc.rater_path(out, "R2", "main")).iloc[:-1], mc.rater_path(out, "R2", "main")
    )
    assert mc.main(["export"]) == 1  # the second rater is incomplete now


def test_export_writes_the_committed_table(world: dict[str, object]) -> None:
    out = Path(world["out"])  # type: ignore[arg-type]
    assert mc.main(["select"]) == 0
    sel = mc.read_selection(out)
    fill(out, "R1", "main", mc.list_refs(sel, "R1", "main"))
    fill(out, "R2", "main", mc.list_refs(sel, "R2", "main"))
    fill(out, "R2", "hard", mc.list_refs(sel, "R2", "hard")[:4])
    fill(out, "R1", "practice", mc.list_refs(sel, "R1", "practice"))
    assert mc.main(["check"]) == 0
    assert mc.main(["export"]) == 0
    table = mc.read_text(Path(world["docs"]) / mc.EXPORT_NAME)  # type: ignore[arg-type]
    assert list(table.columns) == list(mc.EXPORT_COLUMNS)
    assert list(table.columns) == [
        "ref", "rater", "set", "n_taps", "tap_frames", "tap_times_s", "n_edge_partials",
        "confidence", "note", "cell", "weight",
    ]  # fmt: skip
    assert table["set"].value_counts().to_dict() == {"primary": 40, "second": 12, "hard": 4}
    sources = {
        ("R1", "main"): "primary",
        ("R2", "main"): "second",
        ("R2", "hard"): "hard",
    }
    for (rater, kind), label in sources.items():
        rows = mc.read_rows(mc.rater_path(out, rater, kind)).set_index("ref")
        exported = table[table["set"] == label].set_index("ref")
        assert set(exported.index) <= set(rows.index) and len(exported) > 0
        for ref in exported.index:
            assert exported.at[ref, "tap_frames"] == rows.at[ref, "tap_frames"] != ""
            assert exported.at[ref, "tap_times_s"] == rows.at[ref, "tap_times_s"]
            assert exported.at[ref, "n_taps"] == rows.at[ref, "n_taps"]
    assert table["tap_frames"].nunique() > 20  # not one repeated value
    assert set(table["rater"]) == {"R1", "R2"}
    assert (table.loc[table["set"] != "primary", "weight"] == "").all()
    assert (table.loc[table["set"] == "primary", "weight"] != "").all()
    assert not set(table["ref"]) & set(mc.PRACTICE)


def test_check_finds_a_tampered_selection(
    world: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    out = Path(world["out"])  # type: ignore[arg-type]
    assert mc.main(["select"]) == 0
    sel = mc.read_selection(out)
    good = sel.copy()
    sel.loc[sel["set"] == "primary", "weight"] = "1.0"
    mc.atomic_csv(sel, out / mc.SELECTION_NAME)
    assert mc.main(["check"]) == 1
    swap = good.copy()
    a = swap.index[swap["set"] == "primary"][0]
    b = swap.index[(swap["set"] == "frame") & (swap["cell"] == swap.at[a, "cell"])][0]
    swap.loc[[a, b], "set"] = ["frame", "primary"]
    swap.loc[[a, b], "weight"] = ["", swap.at[a, "weight"]]
    mc.atomic_csv(swap, out / mc.SELECTION_NAME)
    capsys.readouterr()
    assert mc.main(["check"]) == 1
    assert "fresh draw" in capsys.readouterr().out


def test_select_output_has_no_refs(
    world: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    assert mc.main(["select"]) == 0
    assert mc.main(["check"]) == 0
    text = capsys.readouterr()
    for ref in world["refs"]:  # type: ignore[attr-defined]
        assert ref not in text.out + text.err


def test_paths_must_stay_inside_their_roots(world: dict[str, object], tmp_path: Path) -> None:
    assert mc.main(["select", "--out-dir", str(tmp_path / "elsewhere")]) == 2
    assert mc.main(["export", "--docs-dir", str(tmp_path / "elsewhere")]) == 2
