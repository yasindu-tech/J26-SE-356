"""Record and freeze the manual visual QC verdicts of every clip (task FT-22 helper).

The person looking at the sheets (``qc_sheet.py``) makes every judgement. This tool only keeps
the book: it never invents, fills in or changes a verdict.

Three commands (from the repo root):

    python models/tapping/src/qc_verdicts.py init [--force]
    python models/tapping/src/qc_verdicts.py check
    python models/tapping/src/qc_verdicts.py freeze

``init`` reads ``data/interim/qc/sheet_page*.csv`` (all pages must be present) and writes
``data/interim/qc/verdicts.csv`` with the columns ``ref, page, panel, verdict, reason, note``;
the last three are empty for you to fill in. It refuses to overwrite an existing file (exit 2)
unless ``--force``; ``--force`` re-reads the pages and KEEPS every verdict, reason and note
already filled in for refs that are still on the pages (merge by ref); refs no longer on the
pages are dropped and new ones are added empty.

``check`` verifies ``verdicts.csv`` against the page files:

- exactly the refs of the pages, each once, and no other refs; ``page`` and ``panel`` agree
  with the page files;
- exactly the six columns above (a label-like column such as group, updrs or control is named
  in the error, any other extra column is an error too);
- ``verdict`` is one of ``ok``, ``ok_flag``, ``reject``;
- ``reject`` needs a ``reason`` from ``tracking_failure``, ``tiny_signal``, ``no_closure``,
  ``other``; ``ok_flag`` needs a non-empty ``note``; ``ok`` and ``ok_flag`` rows carry no
  reason (a reason belongs to a rejection);
- a note or reason must not contain a raw clip ID (``CONTROL01_DCHA``, ``ID1004``, and the
  spellings ``ID_1004`` and ``ID 1004``), nor a label-like WHOLE word: group, control, patient,
  parkinson, parkinsons, updrs, healthy, PD, hoehn, levodopa (and the plurals groups, controls,
  patients), in any case. Boundary rule: a word is a maximal run of letters, digits and
  underscores, so ``control`` is blocked in "control group" but NOT in ``motor_control_unclear``
  or ``controlled``; punctuation and an apostrophe end a word. The error lists refs only, never
  the word or the note;
- every row has a verdict (an empty verdict means the audit is incomplete).

When ``verdicts.csv`` is read, the placeholders ``nan``, ``NaN``, ``NULL``, ``None``, ``N/A`` and
``#N/A`` (any case, after stripping spaces) count as EMPTY in every column, so a spreadsheet or
pandas round trip can neither make an ``ok_flag`` note look filled nor put the text ``nan`` into
a frozen note.

It prints the counts per verdict and the refs that are still empty, by page. Exit 1 if anything
is wrong or incomplete. It never prints the text of a note.

``freeze`` runs ``check`` and, only if it passes, writes (all-or-nothing: both files go to
temporary names first and are renamed into place only after both were written; a failure leaves
the old pair untouched and removes the temporary files; sorted by ref)
``models/tapping/docs/qc_rejections.csv`` (``ref, reason, note``; reject rows) and
``qc_retained.csv`` (``ref, verdict, note``; ok and ok_flag rows). Hashes only: ``verdicts.csv``
itself stays under the gitignored ``data/``. It prints one line, for example
``retained 228 of 234, flagged 3, rejected 6``.

Nothing here looks at UPDRS or at the CONTROL/ID group; only hashed refs appear in any print or
file. Exit codes: 0 done, 1 check failed or incomplete, 2 an input is missing or malformed, a
folder is outside its allowed root, ``init`` would overwrite, or a file cannot be read or written
(``error: could not read/write <file name>: <reason>``, no path, no traceback).
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from signal_processing import is_inside

REPO_ROOT = Path(__file__).resolve().parents[3]
INTERIM_ROOT = REPO_ROOT / "data" / "interim"
DOCS_ROOT = REPO_ROOT / "models" / "tapping" / "docs"

VERDICTS_NAME = "verdicts.csv"
REJECTIONS_NAME = "qc_rejections.csv"
RETAINED_NAME = "qc_retained.csv"
TMP_MARKER = ".qcv-tmp-"

COLUMNS = ("ref", "page", "panel", "verdict", "reason", "note")
VERDICTS = ("ok", "ok_flag", "reject")
REASONS = ("tracking_failure", "tiny_signal", "no_closure", "other")
LABEL_WORDS = ("group", "updrs", "control", "hand", "participant", "clip_id", "diagnos", "label")
# Placeholders that spreadsheets and pandas write for an empty cell; read as EMPTY (any column).
PLACEHOLDERS = frozenset({"nan", "null", "none", "n/a", "#n/a"})
# Words that must not appear in a note or reason (hashes-only rule). Whole words, any case.
FORBIDDEN_WORDS = (
    "group", "groups", "control", "controls", "patient", "patients", "parkinson", "parkinsons",
    "updrs", "healthy", "pd", "hoehn", "levodopa",
)  # fmt: skip
# Boundary rule: a "word" is a maximal run of letters, digits and underscores, so ``control`` is
# blocked in "control group" or "group: control" but NOT inside another identifier such as
# ``motor_control_unclear`` or ``controlled``. An apostrophe or punctuation ends a word.
WORD_RE = re.compile(r"(?i)(?<![a-z0-9_])(" + "|".join(FORBIDDEN_WORDS) + r")(?![a-z0-9_])")
HASH_RE = re.compile(r"[0-9a-f]{8}")
PAGE_FILE_RE = re.compile(r"sheet_page(\d+)of(\d+)\.csv")
# Raw clip IDs: CONTROL01_DCHA, ID1004, and the spaced or underscored spellings ID_1004, ID 1004.
RAW_ID_RE = re.compile(r"(?i)(?<![a-z])(control|id)[\s_-]?\d+(_(dcha|izda))?")
LIST_LIMIT = 12


class InputError(ValueError):
    """An input is missing or malformed (exit code 2)."""


@dataclass
class Report:
    """The outcome of ``check``."""

    errors: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    empty_by_page: dict[int, list[str]] = field(default_factory=dict)
    total: int = 0
    table: pd.DataFrame | None = None

    @property
    def n_empty(self) -> int:
        return sum(len(v) for v in self.empty_by_page.values())

    @property
    def ok(self) -> bool:
        return not self.errors and self.n_empty == 0


# --------------------------------------------------------------------------- files


def _write_temp(df: pd.DataFrame, final: Path) -> Path:
    """Write ``df`` to a temporary file next to ``final`` and return its path."""
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f".{final.name}{TMP_MARKER}{os.getpid()}")
    try:
        df.to_csv(tmp, index=False)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    return tmp


def _atomic_csv(df: pd.DataFrame, final: Path) -> None:
    tmp = _write_temp(df, final)
    try:
        os.replace(tmp, final)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def _atomic_pair(items: Sequence[tuple[pd.DataFrame, Path]]) -> None:
    """Write several files all-or-nothing: every temp file first, the renames only after.

    If any temp write fails, the temp files are removed and the existing files are untouched.
    """
    temps: list[tuple[Path, Path]] = []
    try:
        for df, final in items:
            temps.append((_write_temp(df, final), final))
    except BaseException:
        for tmp, _ in temps:
            with contextlib.suppress(OSError):
                tmp.unlink()
        raise
    for tmp, final in temps:  # only renames are left: they do not fail in normal operation
        os.replace(tmp, final)


def _reason(err: Exception) -> str:
    """A short reason without any path: the OS message, or the first line of the error."""
    if isinstance(err, OSError):
        return err.strerror or type(err).__name__
    return (str(err).splitlines() or [type(err).__name__])[0][:80]


def read_pages(qc_dir: Path) -> pd.DataFrame:
    """``ref, page, panel`` of every panel on every page, sorted by page and panel.

    All pages ``1..N`` named by the files must be present, and a ref must appear once.
    """
    files = sorted(Path(qc_dir).glob("sheet_page*.csv"))
    if not files:
        raise InputError(f"no sheet_page*.csv files in {qc_dir}")
    pages: dict[int, Path] = {}
    totals: set[int] = set()
    for path in files:
        m = PAGE_FILE_RE.fullmatch(path.name)
        if m is None:
            raise InputError(f"unexpected page file name: {path.name}")
        pages[int(m.group(1))] = path
        totals.add(int(m.group(2)))
    if len(totals) != 1 or sorted(pages) != list(range(1, next(iter(totals)) + 1)):
        raise InputError(f"the page files are incomplete: have pages {sorted(pages)} of {totals}")
    frames = []
    for number, path in sorted(pages.items()):
        try:
            df = pd.read_csv(path, usecols=["panel", "ref"], dtype=str, keep_default_na=False)
        except (ValueError, OSError) as err:
            raise InputError(f"cannot read {path.name}: {_reason(err)}") from err
        df["page"] = number
        frames.append(df)
    table = pd.concat(frames, ignore_index=True)
    bad = table[~table["ref"].map(lambda r: bool(HASH_RE.fullmatch(r)))]
    if len(bad):
        raise InputError(f"{len(bad)} page rows have a ref that is not an 8-character hash")
    if table["ref"].duplicated().any():
        raise InputError("a ref appears on more than one page row")
    table["panel"] = table["panel"].astype(int)
    table = table.sort_values(["page", "panel"], kind="stable").reset_index(drop=True)
    return table[["ref", "page", "panel"]]


def read_verdicts(path: Path) -> pd.DataFrame:
    """The verdicts file as text: cells stripped, placeholders such as ``nan`` read as empty."""
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (ValueError, OSError) as err:
        raise InputError(f"cannot read {path.name}: {_reason(err)}") from err
    df = df.apply(lambda col: col.str.strip())
    return df.apply(lambda col: col.where(~col.str.lower().isin(PLACEHOLDERS), ""))


def _listed(items: Sequence[str]) -> str:
    shown = ", ".join(items[:LIST_LIMIT])
    return shown + (f" (+{len(items) - LIST_LIMIT} more)" if len(items) > LIST_LIMIT else "")


# --------------------------------------------------------------------------- init


def init_verdicts(qc_dir: Path, force: bool) -> tuple[int, str]:
    """Write ``verdicts.csv``; returns ``(exit code, message)``."""
    out = Path(qc_dir) / VERDICTS_NAME
    pages = read_pages(qc_dir)
    if out.exists() and not force:
        return 2, f"{VERDICTS_NAME} already exists; use --force to merge (filled verdicts are kept)"
    table = pages.copy()
    for column in ("verdict", "reason", "note"):
        table[column] = ""
    message = f"wrote {VERDICTS_NAME}: {len(table)} refs on {pages['page'].nunique()} pages"
    if out.exists():
        old = read_verdicts(out)
        if "ref" in old.columns and old["ref"].is_unique:
            kept = old.set_index("ref")
            present = [c for c in ("verdict", "reason", "note") if c in kept.columns]
            in_both = table["ref"].isin(kept.index)
            for column in present:
                table.loc[in_both, column] = table.loc[in_both, "ref"].map(kept[column]).to_numpy()
            dropped = len(set(kept.index) - set(table["ref"]))
            filled = int((table["verdict"] != "").sum())
            added = int((~in_both).sum())
            message += f"; kept {filled} filled verdicts, dropped {dropped} refs, added {added} new"
        else:
            return 2, f"the existing {VERDICTS_NAME} has no usable unique ref column; not merged"
    _atomic_csv(table[list(COLUMNS)], out)
    return 0, message


# --------------------------------------------------------------------------- check


def check_verdicts(qc_dir: Path) -> Report:
    """Validate ``verdicts.csv`` against the page files."""
    pages = read_pages(qc_dir)
    path = Path(qc_dir) / VERDICTS_NAME
    if not path.is_file():
        raise InputError(f"{VERDICTS_NAME} not found; run init first")
    df = read_verdicts(path)
    report = Report(total=len(pages))
    extra = [c for c in df.columns if c not in COLUMNS]
    missing_cols = [c for c in COLUMNS if c not in df.columns]
    labels = [c for c in extra if any(w in c.lower() for w in LABEL_WORDS)]
    if labels:
        report.errors.append(f"label-like columns are not allowed: {', '.join(labels)}")
    if [c for c in extra if c not in labels]:
        report.errors.append(f"extra columns: {', '.join(c for c in extra if c not in labels)}")
    if missing_cols:
        report.errors.append(f"missing columns: {', '.join(missing_cols)}")
        return report
    refs = df["ref"].tolist()
    dups = sorted({r for r in refs if refs.count(r) > 1})
    if dups:
        report.errors.append(f"duplicate refs: {_listed(dups)}")
    non_hash = [r for r in refs if not HASH_RE.fullmatch(r)]
    if non_hash:
        report.errors.append(f"{len(non_hash)} rows have a ref that is not an 8-character hash")
    expected = set(pages["ref"])
    absent = sorted(expected - set(refs))
    if absent:
        report.errors.append(f"refs missing from {VERDICTS_NAME}: {_listed(absent)}")
    unknown = sorted(r for r in set(refs) - expected if HASH_RE.fullmatch(r))
    if unknown:
        report.errors.append(f"refs not on any page: {_listed(unknown)}")
    place = pages.set_index("ref")
    moved = [
        r
        for r, p, n in zip(df["ref"], df["page"], df["panel"], strict=True)
        if r in place.index and (p != str(place.at[r, "page"]) or n != str(place.at[r, "panel"]))
    ]
    if moved:
        report.errors.append(f"page or panel differs from the page files: {_listed(sorted(moved))}")
    bad_verdict, no_reason, bad_reason, no_note, stray_reason, raw_id, label_word = (
        [] for _ in range(7)
    )
    counts = {v: 0 for v in VERDICTS}
    counts["empty"] = 0
    page_of = dict(zip(pages["ref"], pages["page"], strict=True))
    for ref, verdict, reason, note in zip(
        df["ref"], df["verdict"], df["reason"], df["note"], strict=True
    ):
        if RAW_ID_RE.search(note) or RAW_ID_RE.search(reason):
            raw_id.append(ref)
        if WORD_RE.search(note) or WORD_RE.search(reason):
            label_word.append(ref)
        if verdict == "":
            counts["empty"] += 1
            if ref in page_of:
                report.empty_by_page.setdefault(int(page_of[ref]), []).append(ref)
            continue
        if verdict not in VERDICTS:
            bad_verdict.append(ref)
            continue
        counts[verdict] += 1
        if verdict == "reject":
            if reason == "":
                no_reason.append(ref)
            elif reason not in REASONS:
                bad_reason.append(ref)
        else:
            if reason != "":
                stray_reason.append(ref)
            if verdict == "ok_flag" and note == "":
                no_note.append(ref)
    for items, text in (
        (bad_verdict, f"verdict is not one of {', '.join(VERDICTS)}"),
        (no_reason, "reject without a reason"),
        (bad_reason, f"reject with a reason not in {', '.join(REASONS)}"),
        (no_note, "ok_flag needs a note"),
        (stray_reason, "ok or ok_flag row with a reason (reasons belong to reject)"),
        (raw_id, "a note or reason contains a raw clip ID"),
        (label_word, "a note or reason contains a label-like word"),
    ):
        if items:
            report.errors.append(f"{text}: {_listed(sorted(items))}")
    report.counts = counts
    for refs_on_page in report.empty_by_page.values():
        refs_on_page.sort()
    report.table = df
    return report


def print_report(report: Report) -> None:
    c = report.counts
    print(
        f"{report.total} refs on the pages | "
        + " | ".join(f"{k} {c.get(k, 0)}" for k in (*VERDICTS, "empty"))
    )
    for page in sorted(report.empty_by_page):
        refs = report.empty_by_page[page]
        print(f"  page {page:02d}: {len(refs)} still empty: {', '.join(refs)}")
    for error in report.errors:
        print(f"ERROR: {error}")
    if report.ok:
        print("check passed: every ref has a valid verdict")
    elif not report.errors:
        print(f"incomplete: {report.n_empty} refs have no verdict yet")


# --------------------------------------------------------------------------- freeze


def freeze_verdicts(qc_dir: Path, docs_dir: Path) -> tuple[int, str]:
    """Write the two frozen lists if ``check`` passes; returns ``(exit code, summary)``."""
    report = check_verdicts(qc_dir)
    if not report.ok or report.table is None:
        print_report(report)
        return 1, "freeze refused: the check did not pass; nothing was written"
    df = report.table.sort_values("ref", kind="stable")
    rejected = df[df["verdict"] == "reject"]
    retained = df[df["verdict"] != "reject"]
    _atomic_pair(
        [
            (
                rejected[["ref", "reason", "note"]].reset_index(drop=True),
                Path(docs_dir) / REJECTIONS_NAME,
            ),
            (
                retained[["ref", "verdict", "note"]].reset_index(drop=True),
                Path(docs_dir) / RETAINED_NAME,
            ),
        ]
    )
    flagged = int((retained["verdict"] == "ok_flag").sum())
    return 0, f"retained {len(retained)} of {len(df)}, flagged {flagged}, rejected {len(rejected)}"


# --------------------------------------------------------------------------- command line


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="create verdicts.csv from the page files")
    init.add_argument("--force", action="store_true", help="merge into an existing file")
    sub.add_parser("check", help="validate verdicts.csv")
    sub.add_parser("freeze", help="check, then write the rejection and retained lists")
    for p in sub.choices.values():
        p.add_argument("--qc-dir", type=Path, default=None)
    sub.choices["freeze"].add_argument("--docs-dir", type=Path, default=None)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    qc_dir = args.qc_dir or INTERIM_ROOT / "qc"
    if not is_inside(qc_dir, INTERIM_ROOT):
        print(f"error: --qc-dir must be inside {INTERIM_ROOT}", file=sys.stderr)
        return 2
    try:
        if args.command == "init":
            code, message = init_verdicts(qc_dir, args.force)
            print(message, file=sys.stderr if code else sys.stdout)
            return code
        if args.command == "check":
            report = check_verdicts(qc_dir)
            print_report(report)
            return 0 if report.ok else 1
        docs_dir = args.docs_dir or DOCS_ROOT
        if not is_inside(docs_dir, DOCS_ROOT):
            print(f"error: --docs-dir must be inside {DOCS_ROOT}", file=sys.stderr)
            return 2
        code, message = freeze_verdicts(qc_dir, docs_dir)
        print(message, file=sys.stderr if code else sys.stdout)
        return code
    except InputError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except OSError as err:
        name = Path(err.filename).name if isinstance(err.filename, str) else "a file"
        print(f"error: could not read/write {name}: {_reason(err)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
