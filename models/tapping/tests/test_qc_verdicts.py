"""Tests for models/tapping/src/qc_verdicts.py (task FT-22 helper).

Synthetic data only: made-up refs and page files under a temporary ``data/interim`` and a
temporary docs folder (``INTERIM_ROOT`` and ``DOCS_ROOT`` are patched). No conftest.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import qc_verdicts as qv  # noqa: E402

N = 234
POISON_ID = "CONTROL01_DCHA"


def refs_of(n: int = N) -> list[str]:
    return [f"{i * 7919 % 0xFFFFFF:08x}" for i in range(n)]  # hash-like, unsorted order


def write_pages(qc_dir: Path, refs: list[str]) -> None:
    """Page files like qc_sheet writes them: 24 refs per page, hash-sorted, 6 columns per row."""
    pool = sorted(refs)
    pages = max(1, -(-len(pool) // 24))
    qc_dir.mkdir(parents=True, exist_ok=True)
    for k in range(1, pages + 1):
        chunk = pool[(k - 1) * 24 : k * 24]
        rows = [
            {
                "sheet": f"sheet_page{k:02d}of{pages:02d}",
                "panel": i + 1,
                "row": i // 6 + 1,
                "col": i % 6 + 1,
                "ref": r,
                "status": "ok",
                "range": 1.0,
                "floor": 0.1,
                "max": 1.5,
                "hints": "",
                "seed": "",
                "page": k,
            }
            for i, r in enumerate(chunk)
        ]
        pd.DataFrame(rows).to_csv(qc_dir / f"sheet_page{k:02d}of{pages:02d}.csv", index=False)


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    interim = tmp_path / "interim"
    docs = tmp_path / "docs"
    docs.mkdir()
    monkeypatch.setattr(qv, "INTERIM_ROOT", interim)
    monkeypatch.setattr(qv, "DOCS_ROOT", docs)
    qc_dir = interim / "qc"
    write_pages(qc_dir, refs_of())
    # a sample sheet next to the pages must be ignored
    pd.DataFrame({"sheet": ["sheet_seed021"], "panel": [1], "ref": [refs_of()[0]]}).to_csv(
        qc_dir / "sheet_seed021.csv", index=False
    )
    return {"qc": qc_dir, "docs": docs, "verdicts": qc_dir / qv.VERDICTS_NAME}


def load(world: dict[str, Path]) -> pd.DataFrame:
    return pd.read_csv(world["verdicts"], dtype=str, keep_default_na=False)


def save(world: dict[str, Path], df: pd.DataFrame) -> None:
    df.to_csv(world["verdicts"], index=False)


def fill_all(world: dict[str, Path]) -> pd.DataFrame:
    """A valid, complete file: mostly ok, a few ok_flag and rejects (set by the test, not the tool)."""
    assert qv.main(["init"]) == 0
    df = load(world)
    df["verdict"] = "ok"
    df.loc[3, ["verdict", "note"]] = ["ok_flag", "slow start"]
    df.loc[10, ["verdict", "note"]] = ["ok_flag", "pause at the start"]
    df.loc[5, ["verdict", "reason", "note"]] = ["reject", "tracking_failure", "hand lost"]
    df.loc[100, ["verdict", "reason"]] = ["reject", "tiny_signal"]
    df.loc[200, ["verdict", "reason", "note"]] = ["reject", "other", "unclear"]
    save(world, df)
    return df


# --------------------------------------------------------------------------- init


def test_init_creates_one_row_per_ref_with_empty_verdicts(world: dict[str, Path]) -> None:
    assert qv.main(["init"]) == 0
    df = load(world)
    assert list(df.columns) == list(qv.COLUMNS) and len(df) == N and df["ref"].is_unique
    assert set(df["ref"]) == set(refs_of())  # the sample sheet adds nothing
    assert (df[["verdict", "reason", "note"]] == "").all().all()
    assert df["page"].astype(int).max() == 10 and df.groupby("page").size().tolist().count(24) == 9
    assert df.iloc[0]["page"] == "1" and df.iloc[0]["panel"] == "1"
    assert df["ref"].tolist() == sorted(df["ref"])  # pages are hash-sorted, so is the file


def test_init_refuses_to_overwrite_without_force(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert qv.main(["init"]) == 0
    df = load(world)
    df.loc[0, "verdict"] = "ok"
    save(world, df)
    before = world["verdicts"].read_bytes()
    assert qv.main(["init"]) == 2
    assert "already exists" in capsys.readouterr().err
    assert world["verdicts"].read_bytes() == before  # untouched


def test_force_keeps_filled_verdicts_by_ref(world: dict[str, Path]) -> None:
    filled = fill_all(world)
    kept = filled.set_index("ref")
    # the pages change: one ref disappears, one new ref appears, and refs move to other pages
    refs = refs_of()
    removed, added = refs[0], "ffffff01"
    write_pages(world["qc"], [r for r in refs if r != removed] + [added])
    assert qv.main(["init", "--force"]) == 0
    df = load(world).set_index("ref")
    assert removed not in df.index and df.loc[added, ["verdict", "reason", "note"]].eq("").all()
    common = [r for r in kept.index if r in df.index]
    for column in ("verdict", "reason", "note"):
        assert (df.loc[common, column] == kept.loc[common, column]).all()
    assert (df["verdict"] != "").sum() == (kept.drop(index=removed)["verdict"] != "").sum()
    new_pages = qv.read_pages(world["qc"]).set_index("ref")
    assert (df.loc[common, "page"] == new_pages.loc[common, "page"].astype(str)).all()  # moved


def test_force_without_an_existing_file_just_creates_it(world: dict[str, Path]) -> None:
    assert qv.main(["init", "--force"]) == 0 and len(load(world)) == N


def test_init_needs_complete_unique_pages(world: dict[str, Path]) -> None:
    (world["qc"] / "sheet_page03of10.csv").unlink()
    assert qv.main(["init"]) == 2  # page 3 is missing
    assert not world["verdicts"].exists()
    write_pages(world["qc"], refs_of())
    dup = pd.read_csv(world["qc"] / "sheet_page02of10.csv", dtype=str)
    dup.loc[0, "ref"] = pd.read_csv(world["qc"] / "sheet_page01of10.csv", dtype=str).loc[0, "ref"]
    dup.to_csv(world["qc"] / "sheet_page02of10.csv", index=False)
    assert qv.main(["init"]) == 2  # a ref on two pages
    for f in world["qc"].glob("sheet_page*.csv"):
        f.unlink()
    assert qv.main(["init"]) == 2  # no pages at all


def test_qc_dir_outside_interim_is_refused(world: dict[str, Path], tmp_path: Path) -> None:
    assert qv.main(["init", "--qc-dir", str(tmp_path / "elsewhere")]) == 2


# --------------------------------------------------------------------------- check


def test_check_passes_on_a_valid_complete_file_and_prints_counts(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    fill_all(world)
    capsys.readouterr()
    assert qv.main(["check"]) == 0
    out = capsys.readouterr().out
    assert "ok 229 | ok_flag 2 | reject 3 | empty 0" in out and "check passed" in out


def test_check_lists_the_empty_refs_by_page_and_exits_1(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    df = fill_all(world)
    df.loc[0, "verdict"] = ""  # page 1
    df.loc[60, "verdict"] = ""  # page 3
    save(world, df)
    capsys.readouterr()
    assert qv.main(["check"]) == 1
    out = capsys.readouterr().out
    assert "empty 2" in out and "page 01: 1 still empty" in out and "page 03: 1 still empty" in out
    assert df.loc[0, "ref"] in out and df.loc[60, "ref"] in out
    assert "incomplete: 2 refs have no verdict yet" in out


def test_the_empty_refs_are_listed_in_full_not_truncated(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert qv.main(["init"]) == 0
    capsys.readouterr()
    assert qv.main(["check"]) == 1
    out = capsys.readouterr().out
    assert "more)" not in out
    assert all(ref in out for ref in load(world)["ref"])  # all 234 refs appear


def test_a_fresh_init_is_incomplete(world: dict[str, Path]) -> None:
    assert qv.main(["init"]) == 0
    assert qv.main(["check"]) == 1  # 234 empty


def run_check(capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    capsys.readouterr()
    code = qv.main(["check"])
    return code, capsys.readouterr().out


def test_check_catches_duplicate_refs(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    df = fill_all(world)
    df.loc[1, "ref"] = df.loc[0, "ref"]  # one ref twice, another one missing
    save(world, df)
    code, out = run_check(capsys)
    assert code == 1 and "duplicate refs" in out and "missing from verdicts.csv" in out


def test_check_catches_missing_and_unknown_refs(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    df = fill_all(world)
    gone = df.loc[7, "ref"]
    df = df.drop(index=7)
    df.loc[len(df)] = ["abcdef12", "1", "1", "ok", "", ""]
    save(world, df)
    code, out = run_check(capsys)
    assert code == 1 and gone in out and "abcdef12" in out and "not on any page" in out


def test_check_catches_a_ref_that_is_not_a_hash(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    df = fill_all(world)
    df.loc[2, "ref"] = POISON_ID
    save(world, df)
    code, out = run_check(capsys)
    assert code == 1 and "not an 8-character hash" in out and POISON_ID not in out


@pytest.mark.parametrize("bad", ["OK", "maybe", "rejected", "ok "])
def test_check_catches_a_bad_verdict(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str], bad: str
) -> None:
    df = fill_all(world)
    df.loc[4, "verdict"] = bad
    save(world, df)
    code, out = run_check(capsys)
    if bad == "ok ":  # surrounding spaces are stripped, so this one is simply "ok"
        assert code == 0
    else:
        assert code == 1 and "verdict is not one of" in out and df.loc[4, "ref"] in out


def test_check_catches_reject_without_or_with_a_bad_reason(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    df = fill_all(world)
    df.loc[5, "reason"] = ""
    save(world, df)
    code, out = run_check(capsys)
    assert code == 1 and "reject without a reason" in out and df.loc[5, "ref"] in out
    df.loc[5, "reason"] = "looks_bad"
    save(world, df)
    code, out = run_check(capsys)
    assert code == 1 and "reject with a reason not in" in out


def test_check_catches_ok_flag_without_a_note(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    df = fill_all(world)
    df.loc[3, "note"] = "   "
    save(world, df)
    code, out = run_check(capsys)
    assert code == 1 and "ok_flag needs a note" in out and df.loc[3, "ref"] in out


def test_check_catches_a_reason_on_a_non_reject_row(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    df = fill_all(world)
    df.loc[0, "reason"] = "tiny_signal"
    save(world, df)
    code, out = run_check(capsys)
    assert code == 1 and "reasons belong to reject" in out


@pytest.mark.parametrize("name", ["group", "UPDRS_score", "Control", "clip_id", "participant_ref"])
def test_check_catches_label_like_columns(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str], name: str
) -> None:
    df = fill_all(world)
    df[name] = "x"
    save(world, df)
    code, out = run_check(capsys)
    assert code == 1 and "label-like columns are not allowed" in out and name in out


def test_check_catches_other_extra_or_missing_columns(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    df = fill_all(world)
    df["comment"] = "x"
    save(world, df)
    code, out = run_check(capsys)
    assert code == 1 and "extra columns: comment" in out
    save(world, df.drop(columns=["comment", "note"]))
    code, out = run_check(capsys)
    assert code == 1 and "missing columns: note" in out


def test_check_catches_a_raw_clip_id_in_a_note_without_printing_the_note(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    df = fill_all(world)
    df.loc[3, "note"] = f"same as {POISON_ID}"
    save(world, df)
    code, out = run_check(capsys)
    assert code == 1 and "raw clip ID" in out and df.loc[3, "ref"] in out
    assert POISON_ID not in out and "same as" not in out  # notes are never echoed


def test_check_catches_page_or_panel_edited_away_from_the_page_files(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    df = fill_all(world)
    df.loc[9, "panel"] = "99"
    save(world, df)
    code, out = run_check(capsys)
    assert code == 1 and "page or panel differs" in out and df.loc[9, "ref"] in out


def test_check_needs_the_file(world: dict[str, Path]) -> None:
    assert qv.main(["check"]) == 2


# --------------------------------------------------------------------------- freeze


def test_freeze_splits_sorts_and_writes_hashes_only(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    df = fill_all(world).set_index("ref")
    capsys.readouterr()
    assert qv.main(["freeze"]) == 0
    assert "retained 231 of 234, flagged 2, rejected 3" in capsys.readouterr().out
    rej = pd.read_csv(world["docs"] / qv.REJECTIONS_NAME, dtype=str, keep_default_na=False)
    ret = pd.read_csv(world["docs"] / qv.RETAINED_NAME, dtype=str, keep_default_na=False)
    assert list(rej.columns) == ["ref", "reason", "note"]
    assert list(ret.columns) == ["ref", "verdict", "note"]
    assert rej["ref"].tolist() == sorted(rej["ref"]) and ret["ref"].tolist() == sorted(ret["ref"])
    assert set(rej["ref"]) == set(df.index[df["verdict"] == "reject"]) and len(rej) == 3
    assert set(ret["ref"]) == set(df.index[df["verdict"] != "reject"]) and len(ret) == 231
    assert set(rej["ref"]).isdisjoint(ret["ref"]) and len(set(rej["ref"]) | set(ret["ref"])) == N
    assert set(ret["verdict"]) == {"ok", "ok_flag"} and (ret["verdict"] == "ok_flag").sum() == 2
    assert set(rej["reason"]) == {"tracking_failure", "tiny_signal", "other"}
    text = (world["docs"] / qv.REJECTIONS_NAME).read_text() + (
        world["docs"] / qv.RETAINED_NAME
    ).read_text()
    for word in ("clip_id", "group", "updrs", "control", "participant"):
        assert word not in text.lower()
    assert POISON_ID not in text
    assert all(len(r) == 8 for r in rej["ref"]) and all(len(r) == 8 for r in ret["ref"])
    assert sorted(p.name for p in world["docs"].iterdir()) == [qv.REJECTIONS_NAME, qv.RETAINED_NAME]


def test_freeze_sorts_by_ref_even_if_the_verdicts_file_was_reordered(
    world: dict[str, Path],
) -> None:
    df = fill_all(world)
    save(world, df.iloc[::-1].reset_index(drop=True))  # as if re-sorted in a spreadsheet
    assert qv.main(["check"]) == 0  # row order does not matter to the check
    assert qv.main(["freeze"]) == 0
    for name in (qv.REJECTIONS_NAME, qv.RETAINED_NAME):
        refs = pd.read_csv(world["docs"] / name, dtype=str)["ref"].tolist()
        assert refs == sorted(refs) and len(refs) > 1


def test_freeze_leaves_verdicts_untouched_and_is_repeatable(world: dict[str, Path]) -> None:
    fill_all(world)
    before = world["verdicts"].read_bytes()
    assert qv.main(["freeze"]) == 0
    first = (world["docs"] / qv.RETAINED_NAME).read_bytes()
    assert world["verdicts"].read_bytes() == before
    assert qv.main(["freeze"]) == 0 and (world["docs"] / qv.RETAINED_NAME).read_bytes() == first
    assert not list(world["docs"].glob(f".*{qv.TMP_MARKER}*"))


def test_freeze_refuses_when_the_check_fails_and_writes_nothing(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    df = fill_all(world)
    df.loc[5, "reason"] = ""
    save(world, df)
    capsys.readouterr()
    assert qv.main(["freeze"]) == 1
    assert "freeze refused" in capsys.readouterr().err
    assert list(world["docs"].iterdir()) == []
    df.loc[5, "reason"] = "tracking_failure"  # valid again, but one verdict is left empty
    df.loc[0, "verdict"] = ""  # incomplete
    save(world, df)
    assert qv.main(["freeze"]) == 1 and list(world["docs"].iterdir()) == []


def test_freeze_with_no_rejections_writes_a_header_only_file(world: dict[str, Path]) -> None:
    assert qv.main(["init"]) == 0
    df = load(world)
    df["verdict"] = "ok"
    save(world, df)
    assert qv.main(["freeze"]) == 0
    rej = pd.read_csv(world["docs"] / qv.REJECTIONS_NAME, dtype=str)
    assert len(rej) == 0 and list(rej.columns) == ["ref", "reason", "note"]


def test_freeze_folders_are_confined(world: dict[str, Path], tmp_path: Path) -> None:
    fill_all(world)
    assert qv.main(["freeze", "--docs-dir", str(tmp_path / "elsewhere")]) == 2
    assert qv.main(["freeze", "--qc-dir", str(tmp_path / "elsewhere")]) == 2
    assert not (tmp_path / "elsewhere").exists()
    inside = world["docs"] / "sub"
    assert (
        qv.main(["freeze", "--docs-dir", str(inside)]) == 0
        and (inside / qv.RETAINED_NAME).is_file()
    )


def test_the_tool_never_fills_or_changes_a_verdict(world: dict[str, Path]) -> None:
    df = fill_all(world)
    for command in (["check"], ["freeze"], ["init", "--force"]):
        qv.main(command)
        assert load(world)[["ref", "verdict", "reason", "note"]].equals(
            df[["ref", "verdict", "reason", "note"]]
        )


# --------------------------------------------------------------------------- hardening


def flaky_write_temp(monkeypatch: pytest.MonkeyPatch, fail_on: int) -> list[int]:
    """Make the ``fail_on``-th temporary write fail (after really writing the earlier ones)."""
    calls: list[int] = []
    real = qv._write_temp

    def flaky(df: pd.DataFrame, final: Path) -> Path:
        calls.append(1)
        if len(calls) == fail_on:
            raise OSError(28, "No space left on device", str(final))
        return real(df, final)

    monkeypatch.setattr(qv, "_write_temp", flaky)
    return calls


def test_freeze_is_all_or_nothing_on_an_empty_docs_dir(
    world: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fill_all(world)
    calls = flaky_write_temp(monkeypatch, fail_on=2)
    capsys.readouterr()
    assert qv.main(["freeze"]) == 2
    assert len(calls) == 2  # the first temp file really was written, then the second failed
    assert list(world["docs"].iterdir()) == []  # no frozen file and no temp file is left
    assert (
        "could not read/write qc_retained.csv: No space left on device" in capsys.readouterr().err
    )


def test_freeze_failure_leaves_an_older_consistent_pair_untouched(
    world: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    df = fill_all(world)
    assert qv.main(["freeze"]) == 0
    old = {n: (world["docs"] / n).read_bytes() for n in (qv.REJECTIONS_NAME, qv.RETAINED_NAME)}
    df.loc[0, ["verdict", "reason"]] = ["reject", "no_closure"]  # the new freeze would differ
    df.loc[1, "verdict"] = "ok_flag"
    df.loc[1, "note"] = "changed"
    save(world, df)
    real_writer = qv._write_temp
    flaky_write_temp(monkeypatch, fail_on=2)
    assert qv.main(["freeze"]) == 2
    new = {n: (world["docs"] / n).read_bytes() for n in (qv.REJECTIONS_NAME, qv.RETAINED_NAME)}
    assert new == old  # both old files byte-identical
    assert sorted(p.name for p in world["docs"].iterdir()) == [qv.REJECTIONS_NAME, qv.RETAINED_NAME]
    monkeypatch.setattr(qv, "_write_temp", real_writer)  # only the writer; the folders stay patched
    assert qv.main(["freeze"]) == 0  # and it works again once the failure is gone
    assert (world["docs"] / qv.REJECTIONS_NAME).read_bytes() != old[qv.REJECTIONS_NAME]


def test_first_temp_write_failure_also_leaves_nothing(
    world: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    fill_all(world)
    flaky_write_temp(monkeypatch, fail_on=1)
    assert qv.main(["freeze"]) == 2 and list(world["docs"].iterdir()) == []


@pytest.mark.skipif(
    not hasattr(__import__("os"), "geteuid") or __import__("os").geteuid() == 0,
    reason="needs a non-root user",
)
def test_unwritable_and_unreadable_files_exit_2_without_paths(
    world: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import os

    fill_all(world)
    capsys.readouterr()
    world["docs"].chmod(0o500)  # read-only docs folder
    try:
        assert qv.main(["freeze"]) == 2
    finally:
        world["docs"].chmod(0o700)
    err = capsys.readouterr().err
    assert "error: could not read/write" in err and "Permission denied" in err
    assert str(tmp_path) not in err and "Traceback" not in err
    world["verdicts"].chmod(0)  # unreadable verdicts file
    try:
        assert qv.main(["check"]) == 2
        err = capsys.readouterr().err
        assert "cannot read verdicts.csv: Permission denied" in err and str(tmp_path) not in err
    finally:
        world["verdicts"].chmod(0o600)
    page = world["qc"] / "sheet_page02of10.csv"
    page.chmod(0)
    try:
        assert qv.main(["check"]) == 2
        err = capsys.readouterr().err
        assert "cannot read sheet_page02of10.csv" in err and str(tmp_path) not in err
    finally:
        page.chmod(0o600)
    assert os.access(world["verdicts"], os.R_OK)


def test_a_write_error_in_init_exits_2_with_the_file_name_only(
    world: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(df: pd.DataFrame, final: Path) -> Path:
        raise PermissionError(13, "Permission denied", str(final))

    monkeypatch.setattr(qv, "_write_temp", broken)
    assert qv.main(["init"]) == 2
    err = capsys.readouterr().err
    assert err.strip() == "error: could not read/write verdicts.csv: Permission denied"
    assert not world["verdicts"].exists()


@pytest.mark.parametrize(
    "placeholder", ["nan", "NaN", "NULL", "None", "N/A", "#N/A", " n/a ", "none"]
)
def test_placeholders_count_as_empty_in_every_column(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str], placeholder: str
) -> None:
    df = fill_all(world)
    ok_row = df.index[df["verdict"] == "ok"][0]
    df.loc[ok_row, ["reason", "note"]] = placeholder  # no "reasons belong to reject" error
    save(world, df)
    code, out = run_check(capsys)
    assert code == 0, out
    df.loc[ok_row, "verdict"] = placeholder  # an empty verdict spelled as a placeholder
    save(world, df)
    code, out = run_check(capsys)
    assert code == 1 and "empty 1" in out and "incomplete: 1 refs have no verdict" in out
    assert "verdict is not one of" not in out


def test_an_ok_flag_whose_note_is_a_placeholder_needs_a_note(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    df = fill_all(world)
    df.loc[3, "note"] = "nan"
    save(world, df)
    code, out = run_check(capsys)
    assert code == 1 and "ok_flag needs a note" in out and df.loc[3, "ref"] in out


def test_a_placeholder_note_is_frozen_as_empty_not_as_text(world: dict[str, Path]) -> None:
    df = fill_all(world)
    rejected = df.index[df["verdict"] == "reject"]
    df.loc[rejected, "note"] = ["nan", "NULL", "N/A"]
    save(world, df)
    assert qv.main(["freeze"]) == 0
    rej = pd.read_csv(world["docs"] / qv.REJECTIONS_NAME, dtype=str, keep_default_na=False)
    assert (rej["note"] == "").all() and len(rej) == 3
    text = (world["docs"] / qv.REJECTIONS_NAME).read_text() + (
        world["docs"] / qv.RETAINED_NAME
    ).read_text()
    assert "nan" not in text.lower().replace("scan", "") and "null" not in text.lower()


def test_force_merge_does_not_carry_placeholders_over_as_text(world: dict[str, Path]) -> None:
    df = fill_all(world)
    df.loc[0, "verdict"] = "NaN"
    df.loc[1, "note"] = "N/A"
    save(world, df)
    assert qv.main(["init", "--force"]) == 0
    after = load(world)
    assert after.loc[0, "verdict"] == "" and after.loc[1, "note"] == ""


BLOCKED_NOTES = [
    "group: control", "control group", "Patient", "patients", "Parkinson's", "parkinsons",
    "parkinson", "UPDRS 3", "updrs", "healthy", "PD", "pd", "hoehn stage 2", "levodopa", "CONTROL",
    "Controls", "ID_1004", "id 1004", "ID-1004", "ID1004", "CONTROL01_DCHA", "see id_7",
]  # fmt: skip
ALLOWED_NOTES = [
    "gap_hidden_taps", "fast_tapping_blur", "spiky", "motor_control_unclear", "controlled", "idle",
    "pdf export", "valid 1 tap", "hidden hand", "slow start", "group_a", "pd5", "tracking_failure",
]  # fmt: skip


@pytest.mark.parametrize("note", BLOCKED_NOTES)
def test_label_like_words_and_raw_ids_are_blocked_in_notes(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str], note: str
) -> None:
    df = fill_all(world)
    df.loc[3, "note"] = note
    save(world, df)
    code, out = run_check(capsys)
    assert code == 1, note
    assert "label-like word" in out or "raw clip ID" in out
    assert df.loc[3, "ref"] in out
    assert note not in out and note.lower() not in out.lower().replace("label-like word", "")


@pytest.mark.parametrize("note", ALLOWED_NOTES)
def test_ordinary_notes_still_pass(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str], note: str
) -> None:
    df = fill_all(world)
    df.loc[3, "note"] = note
    save(world, df)
    code, out = run_check(capsys)
    assert code == 0, (note, out)


def test_the_allowed_reasons_pass_the_word_guard(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    df = fill_all(world)
    rejected = df.index[df["verdict"] == "reject"]
    for reason in qv.REASONS:
        df.loc[rejected[0], "reason"] = reason
        save(world, df)
        assert run_check(capsys)[0] == 0, reason
    assert not any(qv.WORD_RE.search(r) or qv.RAW_ID_RE.search(r) for r in qv.REASONS)


def test_the_word_guard_applies_to_reasons_too_and_never_names_the_word(
    world: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    df = fill_all(world)
    df.loc[5, "reason"] = "patient"
    save(world, df)
    code, out = run_check(capsys)
    assert code == 1 and "label-like word" in out and "patient" not in out.lower()


def test_the_word_boundary_rule() -> None:
    blocked = ["control", "a control.", "(group)", "x, PD, y", "parkinson's", "group:control"]
    free = ["motor_control_unclear", "controller", "pdf", "group1", "_pd_", "pd2", "hand_group"]
    assert all(qv.WORD_RE.search(t) for t in blocked)
    assert not any(qv.WORD_RE.search(t) for t in free)
    assert qv.RAW_ID_RE.search("ID_1004") and qv.RAW_ID_RE.search("id 1004")
    assert not qv.RAW_ID_RE.search("valid 1") and not qv.RAW_ID_RE.search("idle 5")
