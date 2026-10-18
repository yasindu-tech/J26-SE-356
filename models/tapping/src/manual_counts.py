"""Choose the FT-24 clips and keep the books of the blind manual tap counts (task FT-24).

    python models/tapping/src/manual_counts.py select [--force | --replace]
    python models/tapping/src/manual_counts.py check
    python models/tapping/src/manual_counts.py export

COUNTING RULE. One tap = one opening and closing of the thumb-index pair, marked at the frame of
maximum aperture. Two maxima are two taps only if the fingers visibly start closing between them
(a clear reversal); a pause or hold while open is one tap. A tap the tracker missed still counts
if the video shows it. An opening in progress at the first or last frame whose maximum lies
outside the clip is not a tap: add it to n_edge_partials. Tremor, whole-hand shifts and video
flicker are never taps. Confidence 1 when blur, occlusion or a close double-bump call affects
more than one tap; 3 when every tap is clear.

PRE-SPECIFIED ANALYSIS (FT-25). Primary: on the 40 primary clips (rater R1), the MAE of detector
minus manual count weighted by the design weights, and the unweighted ICC(2,1). Reported
separately: per cell (cells are named "detector-rate tertile x QC flag"), R1 vs R2 on the 12
second-rater clips, and the hard set (16 seen clips, R2 only). Practice clips are never analysed.
Manual tap times are pts seconds from the video's first frame; match taps to the detector on the
video frame number (tap_frames, 0-based video frames), because a trimmed signal starts later than
the video. FT-25 compares counts only inside the detector's signal window (first to last frame of
data/interim/smoothed/<ref>.parquet); taps marked on frames outside it (trailing undetected frames
trimmed by FT-19) are reported separately and are not part of MAE/ICC.

SELECTION (pre-registered). Frame = clips in qc_retained.csv minus the 19 seen clips (practice 3,
hard 16). Cell = detector-rate tertile (cuts over all retained clips, tap_rate_hz, <= cut) x QC
flag. One random.Random(24) for all cells in CELL_ORDER: sorted refs, shuffled, taken in order,
skipping a participant already chosen. Weight = frame clips in the cell / sampled clips. Second
rater: random.Random(25), per tertile 3 unflagged + 1 flagged of the primary clips. The only
reroll: ``select --replace`` replaces a primary clip listed in unreadable.csv (written by the
counting tool) by the next unused clip of its cell in the shuffled order, one per participant;
the new row records replaced_for and takes over the second-rater flag. ``select --force`` draws
again and breaks the pre-registration.

Labels are never read: participants come from the video file names (clip_id.py), are hashed in
memory and are never printed or stored. Hashed refs only. Exit codes: 0 ok, 1 check or export
refused, 2 missing or malformed input or a refused overwrite.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import random
import sys
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from clip_id import UnparseableClipId, parse_clip_id
from hubu_fis import HubuFisError, list_videos, pseudonym
from qc_verdicts import HASH_RE, PLACEHOLDERS, RAW_ID_RE, WORD_RE
from signal_processing import is_inside

REPO_ROOT = Path(__file__).resolve().parents[3]
INTERIM_ROOT = REPO_ROOT / "data" / "interim"
DOCS_ROOT = REPO_ROOT / "models" / "tapping" / "docs"
VIDEO_ROOT = REPO_ROOT / "data" / "raw" / "HUBU-FIS" / "videos"

SEED = 24
SECOND_SEED = 25
PRACTICE = ("001ee42d", "6b06ed76", "9c76e15b")
HARD = (
    "01fe52bf", "2406c7b8", "35c23a42", "3be7d4d8", "478db593", "4884a0c8", "4ad85f58",
    "6612f371", "6b83804b", "78bd0691", "7dcf933c", "b2d95c76", "bf081476", "d8050a3c",
    "d8de6459", "de9d2b20",
)  # fmt: skip
TERTILES = ("slow", "mid", "fast")
FLAGS = ("unflagged", "flagged")
CELL_ORDER = tuple(f"{t}/{f}" for t in TERTILES for f in FLAGS)
ALLOCATION = dict(zip(CELL_ORDER, (9, 4, 9, 4, 9, 5), strict=True))
SECOND_PER_TERTILE = {"unflagged": 3, "flagged": 1}
RATERS = ("R1", "R2")
# (rater, list) -> file stem. R1 counts the primary clips and the practice clips, R2 the
# second-rater subset and the hard set.
LISTS = {
    ("R1", "main"): "R1", ("R1", "practice"): "R1_practice",
    ("R2", "main"): "R2", ("R2", "hard"): "R2_hard",
}  # fmt: skip

SELECTION_NAME = "selection.csv"
UNREADABLE_NAME = "unreadable.csv"
EXPORT_NAME = "manual_counts.csv"
TMP_MARKER = ".mc-tmp-"
SELECTION_COLUMNS = (
    "ref", "set", "second_rater", "cell", "cell_rank", "weight", "replaced_for", "seed",
)  # fmt: skip
SETS = ("primary", "frame", "replaced", "practice", "hard")
RATER_COLUMNS = (
    "rater", "order", "ref", "n_taps", "tap_frames", "tap_times_s", "n_edge_partials",
    "confidence", "note", "started_utc", "finished_utc", "tool_version",
)  # fmt: skip
EXPORT_COLUMNS = (
    "ref", "rater", "set", "n_taps", "tap_frames", "tap_times_s", "n_edge_partials", "confidence",
    "note", "cell", "weight", "window_first", "window_last",
)  # fmt: skip
UNREADABLE_COLUMNS = ("ref", "rater", "list", "utc")
LIST_LIMIT = 12


class InputError(ValueError):
    """An input is missing or malformed (exit code 2)."""


# --------------------------------------------------------------------------- small helpers


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _reason(err: Exception) -> str:
    if isinstance(err, OSError):
        return err.strerror or type(err).__name__
    return (str(err).splitlines() or [type(err).__name__])[0][:80]


def _listed(items: Sequence[str]) -> str:
    shown = ", ".join(items[:LIST_LIMIT])
    return shown + (f" (+{len(items) - LIST_LIMIT} more)" if len(items) > LIST_LIMIT else "")


def read_text(path: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """A CSV as stripped text; spreadsheet placeholders such as ``nan`` read as empty."""
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, usecols=columns)
    except (ValueError, OSError) as err:
        raise InputError(f"cannot read {Path(path).name}: {_reason(err)}") from err
    df = df.apply(lambda col: col.str.strip())
    return df.apply(lambda col: col.where(~col.str.lower().isin(PLACEHOLDERS), ""))


def atomic_csv(df: pd.DataFrame, final: Path) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f".{final.name}{TMP_MARKER}{os.getpid()}")
    try:
        df.to_csv(tmp, index=False, lineterminator="\n")
        os.replace(tmp, final)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def note_problem(note: str) -> str | None:
    """Why a note may not be kept (the reason never repeats the note), or None."""
    if RAW_ID_RE.search(note):
        return "it contains a raw clip ID"
    if WORD_RE.search(note):
        return "it contains a label-like word"
    return None


# --------------------------------------------------------------------------- inputs


def participant_keys(video_dir: Path) -> dict[str, tuple[str, str]]:
    """ref -> (hashed participant, hand), from the video file names. Never printed or stored."""
    try:
        videos, _ = list_videos(Path(video_dir))
    except (HubuFisError, OSError) as err:
        raise InputError(f"cannot list the videos: {_reason(err)}") from err
    keys: dict[str, tuple[str, str]] = {}
    for stem in videos:
        try:
            key = parse_clip_id(stem)
        except UnparseableClipId as err:
            raise InputError(f"video {pseudonym(stem)}: the file name does not parse") from err
        keys[pseudonym(stem)] = (pseudonym(key.participant), key.hand)
    return keys


def load_population(docs_dir: Path, summary_path: Path) -> pd.DataFrame:
    """The retained clips with ``tap_rate_hz, qc_status, fps, low_range, n_one_sided``."""
    retained = read_text(Path(docs_dir) / "qc_retained.csv")
    if not {"ref", "verdict"} <= set(retained.columns):
        raise InputError("qc_retained.csv lacks the columns ref, verdict")
    summary = read_text(summary_path)
    needed = ["ref", "tap_rate_hz", "qc_status", "fps", "low_range", "n_one_sided"]
    if not set(needed) <= set(summary.columns):
        raise InputError(f"{Path(summary_path).name} lacks one of: {', '.join(needed)}")
    if set(summary["ref"]) != set(retained["ref"]) or not summary["ref"].is_unique:
        raise InputError("taps_summary.csv does not hold exactly the retained clips; rerun taps.py")
    df = retained[["ref", "verdict"]].merge(summary[needed], on="ref", validate="one_to_one")
    expected = df["verdict"].map({"ok": "retained", "ok_flag": "retained_flag"})
    if (expected != df["qc_status"]).any():
        raise InputError("qc_status in taps_summary.csv disagrees with qc_retained.csv")
    try:
        df["tap_rate_hz"] = df["tap_rate_hz"].astype(float)
    except ValueError as err:
        raise InputError("tap_rate_hz is not numeric") from err
    if not np.isfinite(df["tap_rate_hz"]).all():
        raise InputError("tap_rate_hz has missing values")
    return df.sort_values("ref", kind="stable").reset_index(drop=True)


def tertile_cuts(rates: Iterable[float]) -> tuple[float, float]:
    lo, hi = np.quantile(np.asarray(list(rates), dtype=float), [1 / 3, 2 / 3])
    return float(lo), float(hi)


def cells(population: pd.DataFrame) -> pd.Series:
    """ref -> cell, e.g. ``mid/flagged``. Tertile cuts over every row (all retained clips)."""
    lo, hi = tertile_cuts(population["tap_rate_hz"])
    rate = population["tap_rate_hz"].to_numpy(dtype=float)
    tertile = np.where(rate <= lo, "slow", np.where(rate <= hi, "mid", "fast"))
    flag = np.where(population["qc_status"] == "retained_flag", "flagged", "unflagged")
    return pd.Series(
        [f"{t}/{f}" for t, f in zip(tertile, flag, strict=True)], index=population["ref"]
    )


# --------------------------------------------------------------------------- selection


def build_selection(
    population: pd.DataFrame,
    participants: dict[str, tuple[str, str]],
    seed: int = SEED,
    second_seed: int = SECOND_SEED,
) -> pd.DataFrame:
    """The pre-registered draw (see the module docstring). Pure: same inputs, same table."""
    cell_of = cells(population)
    seen = set(PRACTICE) | set(HARD)
    missing = sorted(seen - set(cell_of.index))
    if missing:
        raise InputError(f"seen clips not in the retained list: {_listed(missing)}")
    unknown = sorted(set(cell_of.index) - set(participants))
    if unknown:
        raise InputError(f"retained clips without a video file: {_listed(unknown)}")
    frame = cell_of.drop(sorted(seen))
    rng = random.Random(seed)
    taken: set[str] = set()
    rows: list[dict[str, object]] = []
    for cell in CELL_ORDER:
        pool = sorted(frame.index[frame == cell])
        rng.shuffle(pool)
        picked = 0
        for rank, ref in enumerate(pool):
            chosen = picked < ALLOCATION[cell] and participants[ref][0] not in taken
            if chosen:
                taken.add(participants[ref][0])
                picked += 1
            rows.append({"ref": ref, "set": "primary" if chosen else "frame", "cell": cell})
            rows[-1]["cell_rank"] = rank
            rows[-1]["weight"] = len(pool) / ALLOCATION[cell] if chosen else ""
        if picked < ALLOCATION[cell]:
            raise InputError(f"cell {cell}: only {picked} of {ALLOCATION[cell]} clips available")
    for name, refs in (("practice", PRACTICE), ("hard", HARD)):
        rows += [
            {"ref": r, "set": name, "cell": cell_of[r], "cell_rank": "", "weight": ""} for r in refs
        ]
    sel = pd.DataFrame(rows)
    sel["second_rater"] = 0
    sel["replaced_for"] = ""
    sel["seed"] = seed
    rng2 = random.Random(second_seed)
    for tertile in TERTILES:
        for flag in FLAGS:
            mask = (sel["set"] == "primary") & (sel["cell"] == f"{tertile}/{flag}")
            pool = sorted(sel.loc[mask, "ref"])
            rng2.shuffle(pool)
            sel.loc[sel["ref"].isin(pool[: SECOND_PER_TERTILE[flag]]), "second_rater"] = 1
    return sel[list(SELECTION_COLUMNS)]


def read_selection(out_dir: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    path = Path(out_dir) / SELECTION_NAME
    if not path.is_file():
        raise InputError(f"{SELECTION_NAME} not found; run manual_counts.py select first")
    return read_text(path, columns)


def replace_unreadable(
    sel: pd.DataFrame, bad_refs: Iterable[str], participants: dict[str, tuple[str, str]]
) -> tuple[pd.DataFrame, list[str]]:
    """Apply the replacement rule to each unreadable primary clip; returns (table, messages)."""
    sel = sel.copy()
    messages: list[str] = []
    for ref in sorted(set(bad_refs)):
        where = sel.index[sel["ref"] == ref]
        if len(where) != 1:
            messages.append(f"{ref}: not in the selection, ignored")
            continue
        old = where[0]
        kind = sel.at[old, "set"]
        if kind == "replaced":
            continue  # done on an earlier run
        if kind != "primary":
            messages.append(f"{ref}: a {kind} clip, not replaced (only primary clips are)")
            continue
        primary = sel["set"] == "primary"
        used = {participants[r][0] for r in sel.loc[primary, "ref"] if r != ref}
        pool = sel[(sel["set"] == "frame") & (sel["cell"] == sel.at[old, "cell"])]
        pool = pool.sort_values("cell_rank", key=lambda s: s.astype(int), kind="stable")
        new = next((r for r in pool["ref"] if participants[r][0] not in used), None)
        if new is None:
            raise InputError(f"{ref}: no unused clip left in its cell to replace it")
        idx = sel.index[sel["ref"] == new][0]
        sel.loc[idx, ["set", "weight", "second_rater", "replaced_for"]] = [
            "primary", sel.at[old, "weight"], sel.at[old, "second_rater"], ref
        ]  # fmt: skip
        sel.loc[old, ["set", "weight"]] = ["replaced", ""]  # second_rater kept: fixes R2 order
        messages.append(f"{ref}: replaced by {new}")
    return sel, messages


def list_refs(sel: pd.DataFrame, rater: str, kind: str) -> list[str]:
    """The clips one rater counts in one list, in that rater's fixed random order.

    The original clips (replaced ones included) are shuffled by ``random.Random(f"{SEED}:{rater}")``
    over their sorted refs and the replaced ones dropped; replacement clips follow at the end in
    the order of the selection file. A replacement therefore never reorders a list in progress.
    """
    if (rater, kind) not in LISTS:
        raise InputError(f"{rater} does not count the {kind} list")
    if kind in ("practice", "hard"):
        base = sorted(sel.loc[sel["set"] == kind, "ref"])
        random.Random(f"{SEED}:{rater}").shuffle(base)
        return base
    second = sel["second_rater"].astype(str) == "1"
    original = sel["set"].isin(["primary", "replaced"]) & (sel["replaced_for"] == "")
    added = (sel["set"] == "primary") & (sel["replaced_for"] != "")
    if rater == "R2":
        original, added = original & second, added & second
    base = sorted(sel.loc[original, "ref"])
    random.Random(f"{SEED}:{rater}").shuffle(base)
    gone = set(sel.loc[sel["set"] == "replaced", "ref"])
    return [r for r in base if r not in gone] + sel.loc[added, "ref"].tolist()


def validate_selection(
    sel: pd.DataFrame, participants: dict[str, tuple[str, str]], fresh: pd.DataFrame | None
) -> list[str]:
    """Structural rules of the selection, and agreement with a fresh draw if given."""
    errors: list[str] = []
    if list(sel.columns) != list(SELECTION_COLUMNS):
        return [f"{SELECTION_NAME} must have the columns {', '.join(SELECTION_COLUMNS)}"]
    if not sel["ref"].is_unique or not sel["ref"].map(lambda r: bool(HASH_RE.fullmatch(r))).all():
        errors.append("refs must be unique 8-character hashes")
    if (sel["seed"] != str(SEED)).any():
        errors.append(f"seed must be {SEED} in every row")
    if not sel["set"].isin(SETS).all():
        errors.append(f"set must be one of {', '.join(SETS)}")
    for name, refs in (("practice", PRACTICE), ("hard", HARD)):
        if set(sel.loc[sel["set"] == name, "ref"]) != set(refs):
            errors.append(f"the {name} set differs from the fixed list")
    primary = sel[sel["set"] == "primary"]
    in_frame = sel[sel["set"].isin(["primary", "frame", "replaced"])]
    for cell in CELL_ORDER:
        n = int((primary["cell"] == cell).sum())
        if n != ALLOCATION[cell]:
            errors.append(f"cell {cell}: {n} primary clips, expected {ALLOCATION[cell]}")
        expected = int((in_frame["cell"] == cell).sum()) / ALLOCATION[cell]
        w = pd.to_numeric(primary.loc[primary["cell"] == cell, "weight"], errors="coerce")
        if not np.allclose(w.to_numpy(dtype=float), expected):
            errors.append(f"cell {cell}: weight is not frame clips / sampled clips")
    total = pd.to_numeric(primary["weight"], errors="coerce").sum()
    if not np.isclose(total, len(in_frame)):
        errors.append(f"weights sum to {total:.3f}, not to the frame size {len(in_frame)}")
    people = [participants.get(r, ("?", ""))[0] for r in primary["ref"]]
    if "?" in people or len(set(people)) != len(people):
        errors.append("the primary clips must come from different, known participants")
    second = primary[primary["second_rater"] == "1"]
    if (sel.loc[~sel["set"].isin(["primary", "replaced"]), "second_rater"] != "0").any():
        errors.append("only primary (or replaced) clips can be second-rater clips")
    for tertile in TERTILES:
        for flag, n_expected in SECOND_PER_TERTILE.items():
            n = int((second["cell"] == f"{tertile}/{flag}").sum())
            if n != n_expected:
                errors.append(f"second rater: {n} {tertile}/{flag} clips, expected {n_expected}")
    replaced = set(sel.loc[sel["set"] == "replaced", "ref"])
    targets = sel.loc[sel["replaced_for"] != "", "replaced_for"].tolist()
    if sorted(targets) != sorted(replaced) or len(set(targets)) != len(targets):
        errors.append("every replaced clip needs exactly one replacement and vice versa")
    if fresh is not None:
        original = set(primary.loc[primary["replaced_for"] == "", "ref"]) | replaced
        same_cells = (
            fresh.set_index("ref")["cell"].to_dict() == sel.set_index("ref")["cell"].to_dict()
        )
        if not same_cells or original != set(fresh.loc[fresh["set"] == "primary", "ref"]):
            errors.append("the selection does not match a fresh draw from the current inputs")
    return errors


def selection_summary(
    sel: pd.DataFrame, population: pd.DataFrame, participants: dict[str, tuple[str, str]]
) -> pd.DataFrame:
    """One row per cell plus totals, for the primary clips: counts only, never refs."""
    pop = population.set_index("ref")
    in_frame = sel[sel["set"].isin(["primary", "frame", "replaced"])]
    rows = []
    for cell in (*CELL_ORDER, "all"):
        part = sel[(sel["set"] == "primary") & ((sel["cell"] == cell) | (cell == "all"))]
        refs = part["ref"].tolist()
        frame_n = len(in_frame) if cell == "all" else int((in_frame["cell"] == cell).sum())
        weight = pd.to_numeric(part["weight"], errors="coerce")
        rows.append(
            {
                "cell": cell,
                "N_frame": frame_n,
                "n_sample": len(refs),
                "weight": round(float(weight.iloc[0]), 3) if cell != "all" else "",
                "participants": len({participants[r][0] for r in refs}),
                "fps60": int((pop.loc[refs, "fps"].astype(float) > 45).sum()),
                "low_range": int((pop.loc[refs, "low_range"] == "True").sum()),
                "one_sided_clips": int((pop.loc[refs, "n_one_sided"].astype(int) > 0).sum()),
                "left": sum(participants[r][1] == "left" for r in refs),
                "right": sum(participants[r][1] == "right" for r in refs),
                "second_rater": int((part["second_rater"].astype(str) == "1").sum()),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- rater files


def rater_path(out_dir: Path, rater: str, kind: str) -> Path:
    if (rater, kind) not in LISTS:
        raise InputError(f"{rater} does not count the {kind} list")
    return Path(out_dir) / f"{LISTS[(rater, kind)]}.csv"


def read_rows(path: Path) -> pd.DataFrame:
    if not Path(path).is_file():
        return pd.DataFrame(columns=list(RATER_COLUMNS))
    return read_text(path)


def append_row(path: Path, row: dict[str, object]) -> None:
    """Add one finished clip to a rater file (rewritten atomically)."""
    df = read_rows(path)
    new = pd.DataFrame([{c: row[c] for c in RATER_COLUMNS}]).astype(str)
    atomic_csv(pd.concat([df, new], ignore_index=True)[list(RATER_COLUMNS)], Path(path))


def read_unreadable(out_dir: Path) -> set[str]:
    path = Path(out_dir) / UNREADABLE_NAME
    return set(read_text(path)["ref"]) if path.is_file() else set()


def add_unreadable(out_dir: Path, ref: str, rater: str, kind: str) -> None:
    path = Path(out_dir) / UNREADABLE_NAME
    df = read_text(path) if path.is_file() else pd.DataFrame(columns=list(UNREADABLE_COLUMNS))
    if ref not in set(df["ref"]):
        new = pd.DataFrame([{"ref": ref, "rater": rater, "list": kind, "utc": utc_now()}])
        atomic_csv(pd.concat([df, new], ignore_index=True), path)


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(";")] if text else []


def _floats(text: str) -> list[float]:
    return [float(x) for x in text.split(";")] if text else []


def check_rows(df: pd.DataFrame, rater: str, expected: Sequence[str], name: str) -> list[str]:
    """Rules for one rater file; errors name refs only, never a note."""
    if list(df.columns) != list(RATER_COLUMNS):
        return [f"{name}: the columns must be {', '.join(RATER_COLUMNS)}"]
    errors: list[str] = []
    if (df["rater"] != rater).any():
        errors.append(f"{name}: rater must be {rater} in every row")
    dups = sorted(df.loc[df["ref"].duplicated(), "ref"])
    if dups:
        errors.append(f"{name}: duplicate refs: {_listed(dups)}")
    stray = sorted(set(df["ref"]) - set(expected))
    if stray:
        errors.append(f"{name}: refs that are not in this list: {_listed(stray)}")
    bad: dict[str, list[str]] = {}
    for row in df.itertuples(index=False):
        ref = str(row.ref)
        try:
            frames, times = _ints(str(row.tap_frames)), _floats(str(row.tap_times_s))
            n, edges = int(str(row.n_taps)), int(str(row.n_edge_partials))
        except ValueError:
            bad.setdefault("unreadable numbers", []).append(ref)
            continue
        if n != len(frames) or n != len(times) or edges < 0:
            bad.setdefault("n_taps does not match the tap lists", []).append(ref)
        if any(b <= a for a, b in zip(frames, frames[1:], strict=False)) or any(
            b < a for a, b in zip(times, times[1:], strict=False)
        ):
            bad.setdefault("tap frames or times are not increasing", []).append(ref)
        if str(row.confidence) not in ("1", "2", "3"):
            bad.setdefault("confidence must be 1, 2 or 3", []).append(ref)
        if note_problem(str(row.note)):
            bad.setdefault("a note contains a raw clip ID or a label-like word", []).append(ref)
    errors += [f"{name}: {text}: {_listed(sorted(refs))}" for text, refs in bad.items()]
    return errors


# --------------------------------------------------------------------------- commands


def _paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    out_dir = Path(args.out_dir or INTERIM_ROOT / "manual_counts")
    docs_dir = Path(args.docs_dir or DOCS_ROOT)
    summary = Path(args.summary or INTERIM_ROOT / "taps" / "taps_summary.csv")
    video_dir = Path(args.video_dir or VIDEO_ROOT)
    return out_dir, docs_dir, summary, video_dir


def cmd_select(args: argparse.Namespace) -> int:
    out_dir, docs_dir, summary_path, video_dir = _paths(args)
    path = out_dir / SELECTION_NAME
    population = load_population(docs_dir, summary_path)
    participants = participant_keys(video_dir)
    if args.replace:
        sel, messages = replace_unreadable(
            read_selection(out_dir), read_unreadable(out_dir), participants
        )
        atomic_csv(sel, path)
        print("\n".join(messages) or "nothing to replace")
        return 0
    if path.exists() and not args.force:
        print(
            f"error: {SELECTION_NAME} already exists and is pre-registered; --force draws again "
            "and breaks the pre-registration",
            file=sys.stderr,
        )
        return 2
    sel = build_selection(population, participants)
    errors = validate_selection(sel.astype(str), participants, None)
    if errors:
        print("\n".join(f"ERROR: {e}" for e in errors), file=sys.stderr)
        return 1
    atomic_csv(sel, path)
    print(selection_summary(sel.astype(str), population, participants).to_string(index=False))
    print(f"wrote {SELECTION_NAME}: {int((sel['set'] == 'primary').sum())} primary clips")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    out_dir, docs_dir, summary_path, video_dir = _paths(args)
    sel = read_selection(out_dir)
    participants = participant_keys(video_dir)
    fresh = build_selection(load_population(docs_dir, summary_path), participants).astype(str)
    errors = validate_selection(sel, participants, fresh)
    progress = []
    for (rater, kind), stem in LISTS.items():
        expected = list_refs(sel, rater, kind)
        rows = read_rows(rater_path(out_dir, rater, kind))
        errors += check_rows(rows, rater, expected, f"{stem}.csv")
        progress.append(f"{stem}: {int(rows['ref'].isin(expected).sum())} of {len(expected)}")
    print(
        f"selection: {int((sel['set'] == 'primary').sum())} primary "
        f"({int((sel['second_rater'] == '1').sum())} second rater), "
        f"{int((sel['set'] == 'hard').sum())} hard, {int((sel['set'] == 'practice').sum())} "
        f"practice, {int((sel['set'] == 'replaced').sum())} replaced"
    )
    print("saved: " + " | ".join(progress) + f" | unreadable: {len(read_unreadable(out_dir))}")
    for error in errors:
        print(f"ERROR: {error}")
    print("check passed" if not errors else "check failed")
    return 1 if errors else 0


def window_bounds(
    smoothed_dir: Path, refs: Iterable[str]
) -> tuple[dict[str, tuple[int, int]], list[str]]:
    """First and last video frame (inclusive) of each clip's smoothed signal, and the refs
    whose smoothed file is missing or unusable."""
    bounds: dict[str, tuple[int, int]] = {}
    missing: list[str] = []
    for ref in sorted(set(refs)):
        try:
            frames = pd.read_parquet(Path(smoothed_dir) / f"{ref}.parquet", columns=["frame"])
        except (OSError, ValueError, KeyError, pa.ArrowException):
            missing.append(ref)
            continue
        if frames.empty:
            missing.append(ref)
            continue
        bounds[ref] = (int(frames["frame"].iloc[0]), int(frames["frame"].iloc[-1]))
    return bounds, missing


def build_export(out_dir: Path, smoothed_dir: Path) -> tuple[pd.DataFrame | None, list[str]]:
    sel = read_selection(out_dir)
    info = sel.set_index("ref")
    parts, problems = [], []
    for rater, kind, label, required in (
        ("R1", "main", "primary", True),
        ("R2", "main", "second", True),
        ("R2", "hard", "hard", False),
    ):
        expected = list_refs(sel, rater, kind)
        rows = read_rows(rater_path(out_dir, rater, kind))
        problems += check_rows(rows, rater, expected, f"{LISTS[(rater, kind)]}.csv")
        missing = sorted(set(expected) - set(rows["ref"]))
        if required and missing:
            problems.append(f"{label}: {len(missing)} of {len(expected)} clips have no {rater} row")
        rows = rows[rows["ref"].isin(expected)].copy()
        rows["set"] = label
        rows["cell"] = rows["ref"].map(info["cell"])
        rows["weight"] = rows["ref"].map(info["weight"]) if label == "primary" else ""
        parts.append(rows)
    if problems:
        return None, problems
    table = pd.concat(parts, ignore_index=True)
    bounds, missing = window_bounds(smoothed_dir, table["ref"])
    if missing:
        return None, [f"window: {len(missing)} clips have no smoothed file: {_listed(missing)}"]
    table["window_first"] = table["ref"].map(lambda r: str(bounds[r][0]))
    table["window_last"] = table["ref"].map(lambda r: str(bounds[r][1]))
    table = table[list(EXPORT_COLUMNS)]
    return table.sort_values(["set", "rater", "ref"], kind="stable").reset_index(drop=True), []


def cmd_export(args: argparse.Namespace) -> int:
    out_dir, docs_dir, _, _ = _paths(args)
    table, problems = build_export(out_dir, Path(args.smoothed_dir or INTERIM_ROOT / "smoothed"))
    if table is None:
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr)
        print("export refused: nothing was written", file=sys.stderr)
        return 1
    atomic_csv(table, docs_dir / EXPORT_NAME)
    counts = table["set"].value_counts()
    print(f"wrote {EXPORT_NAME}: " + ", ".join(f"{k} {counts.get(k, 0)}" for k in counts.index))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select", help="draw the pre-registered selection")
    mode = select.add_mutually_exclusive_group()
    mode.add_argument("--force", action="store_true", help="draw again (breaks pre-registration)")
    mode.add_argument("--replace", action="store_true", help="replace unreadable primary clips")
    sub.add_parser("check", help="validate the selection and the rater files")
    sub.add_parser("export", help="write models/tapping/docs/manual_counts.csv")
    for p in sub.choices.values():
        for name in ("--out-dir", "--docs-dir", "--summary", "--video-dir", "--smoothed-dir"):
            p.add_argument(name, type=Path, default=None)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir, docs_dir, summary, _ = _paths(args)
    smoothed = Path(args.smoothed_dir or INTERIM_ROOT / "smoothed")
    if not all(is_inside(p, INTERIM_ROOT) for p in (out_dir, summary, smoothed)):
        print(
            "error: --out-dir, --summary and --smoothed-dir must be inside data/interim",
            file=sys.stderr,
        )
        return 2
    if not is_inside(docs_dir, DOCS_ROOT):
        print("error: --docs-dir must be inside models/tapping/docs", file=sys.stderr)
        return 2
    try:
        command = {"select": cmd_select, "check": cmd_check, "export": cmd_export}
        return command[args.command](args)
    except InputError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except OSError as err:
        name = Path(err.filename).name if isinstance(err.filename, str) else "a file"
        print(f"error: could not read/write {name}: {_reason(err)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
