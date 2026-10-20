"""FT-26 pre-registration: the tuning protocol and the fresh confirmation sample (Part 1 only).

    python models/tapping/src/tune_detector.py select [--force | --replace]
    python models/tapping/src/tune_detector.py check

Holds the frozen protocol constants, the gate, the objective and the acceptance rule, and the
seeded draw of the fresh confirmation clips. It does not tune anything and reads no counts.
The frame is every retained clip that is not in the FT-24 selection (primary, hard, practice) and
whose participant is in none of those sets. Cells use the FROZEN detector rate (FT-24 tertile
cuts) and the QC flag. One clip per participant. Weight = frame clips in the cell / sampled.
The manual_counts selection.csv is read (never written) to confirm that no fresh participant
is in a primary, replaced, hard or practice row. Labels are never read; participants come from video file names, hashed in memory, never printed.
Exit codes: 0 ok, 1 check failed, 2 missing or malformed input, refused overwrite or bad path.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import math
import random
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import manual_counts as mc
import numpy as np
import pandas as pd
from qc_verdicts import HASH_RE
from signal_processing import is_inside

REPO_ROOT = mc.REPO_ROOT
INTERIM_ROOT = mc.INTERIM_ROOT
DOCS_ROOT = mc.DOCS_ROOT
VIDEO_ROOT = mc.VIDEO_ROOT

# ----------------------------------------------------------------------- protocol (frozen)
MAE_TARGET = 1.0  # FT-25 target: weighted MAE <= 1 tap per clip
GATE_STATUS = "not_met"  # FT-25 decision status that can start tuning
GATE_MAE_MAX = 1.3  # a grid step moves counts <= 0.32 taps/clip, so a larger gap is not closable
GATE_TUNE, GATE_CLOSE_FROZEN, GATE_CLOSE_LIMITATIONS = "tune", "close_frozen", "close_limitations"

PROMINENCE_GRID = (0.15, 0.175, 0.20, 0.225, 0.25)  # x range, the only prominence values tried
SPACING_GRID = (0.10, 0.15, 0.20)  # seconds, the only spacing values tried
FROZEN = (0.20, 0.15)  # (prominence_frac, min_spacing_s) of FT-23, always evaluated, wins ties
CANDIDATES = tuple(itertools.product(PROMINENCE_GRID, SPACING_GRID))  # prominence outer
TIE_TOL = 1e-12  # objectives closer than this are a tie

BIG_ERROR = 3  # |d| above this is penalised once more
BIG_ERROR_WEIGHT = 0.5  # objective = mean|d| + 0.5 * mean(max(|d| - 3, 0))
MIN_GAIN = 0.25  # objective gain over the frozen parameters needed to switch
BOOT_LOWER_PERCENTILE = 5.0  # lower end of the 90% participant-level paired bootstrap gain
MIN_STABILITY = 0.80  # share of leave-one-participant-out folds where the candidate is argmin
TUNING_SETS = ("hard", "practice")  # the 16 hard + 3 practice clips; primary is never tuned on
CONFOUND_CHECK = "compare R1 and R2 counts of the 12 shared clips with the size of the gain"

# ----------------------------------------------------------------------- fresh sample (frozen)
FRESH_SEED = 26
ALLOCATION = dict(zip(mc.CELL_ORDER, (4, 2, 5, 1, 5, 3), strict=True))  # 20 clips
FRESH_CELL_ORDER = tuple(c for c in mc.CELL_ORDER if c.endswith("/flagged")) + tuple(
    c for c in mc.CELL_ORDER if c.endswith("/unflagged")
)  # scarcest cells first, so a participant is not used up before a rare cell is drawn
EXPECTED_FRAME_CLIPS = 133  # retained clips of participants outside primary, hard and practice
EXPECTED_FRAME_PARTICIPANTS = 67
EXPECTED_TUNING_PRIMARY_OVERLAP = 4  # participants in both the tuning set and the primary 40
POPULATION = "clips from the 67 unseen participants"

SELECTION_NAME = "fresh_selection.csv"
SELECTION_COLUMNS = ("ref", "set", "cell", "cell_rank", "weight", "replaced_for", "seed", "order")
SETS = ("fresh", "pool", "replaced")
NOTE_SKIPPED = "note: manual_counts selection.csv not found, cross-check skipped"


# ----------------------------------------------------------------------- gate and rule


def decide_gate(status: str, mae: float) -> str:
    """What FT-26 does after FT-25: tune, close with frozen parameters, or close with limits."""
    if not math.isfinite(mae):
        raise ValueError("the primary weighted MAE must be a finite number")
    if (status == GATE_STATUS) != (mae > MAE_TARGET):
        raise ValueError("the decision status and the weighted MAE disagree")
    if mae <= MAE_TARGET:
        return GATE_CLOSE_FROZEN
    return GATE_TUNE if mae <= GATE_MAE_MAX else GATE_CLOSE_LIMITATIONS


def objective(d: Sequence[float] | np.ndarray) -> float:
    """Unweighted mean|d| + 0.5 * mean(max(|d| - 3, 0)) over the tuning clips."""
    a = np.abs(np.asarray(d, dtype=float))
    if a.size == 0:
        raise ValueError("the objective needs at least one clip")
    return float(a.mean() + BIG_ERROR_WEIGHT * np.maximum(a - BIG_ERROR, 0.0).mean())


def pick_candidate(objectives: Sequence[float]) -> int:
    """Index of the best candidate; the frozen parameters win ties, then grid order."""
    if len(objectives) != len(CANDIDATES):
        raise ValueError(f"expected {len(CANDIDATES)} objectives")
    frozen = CANDIDATES.index(FROZEN)
    best = min(objectives)
    if objectives[frozen] <= best + TIE_TOL:
        return frozen
    return next(i for i, v in enumerate(objectives) if v <= best + TIE_TOL)


def passes_rule(gain: float, boot_lower: float, stability: float) -> bool:
    """All three conditions to replace the frozen parameters."""
    return gain >= MIN_GAIN and boot_lower > 0 and stability >= MIN_STABILITY


def tuning_facts(sel: pd.DataFrame, participants: dict[str, tuple[str, str]]) -> dict[str, int]:
    """Counts for the report: the tuning set and its overlap with the primary participants."""
    tuning = sel.loc[sel["set"].isin(TUNING_SETS), "ref"]
    primary = sel.loc[sel["set"] == "primary", "ref"]
    t_people = {participants[r][0] for r in tuning}
    p_people = {participants[r][0] for r in primary}
    return {
        "tuning_clips": len(tuning),
        "tuning_participants": len(t_people),
        "tuning_primary_overlap_participants": len(t_people & p_people),
    }


# ----------------------------------------------------------------------- the fresh draw


def seen_refs(population: pd.DataFrame, participants: dict[str, tuple[str, str]]) -> set[str]:
    """Clips of the FT-24 selection that are primary, hard or practice (not the unused frame)."""
    sel = mc.build_selection(population, participants)
    return set(sel.loc[sel["set"] != "frame", "ref"])


def build_frame(population: pd.DataFrame, participants: dict[str, tuple[str, str]]) -> pd.Series:
    """ref -> cell for the retained clips whose participant is in no seen clip."""
    cell_of = mc.cells(population)
    seen = seen_refs(population, participants)
    seen_people = {participants[r][0] for r in seen}
    keep = [r for r in cell_of.index if r not in seen and participants[r][0] not in seen_people]
    return cell_of.loc[sorted(keep)]


def check_frame(frame: pd.Series, participants: dict[str, tuple[str, str]]) -> None:
    people = {participants[r][0] for r in frame.index}
    if len(frame) != EXPECTED_FRAME_CLIPS or len(people) != EXPECTED_FRAME_PARTICIPANTS:
        raise mc.InputError(
            f"the fresh frame has {len(frame)} clips and {len(people)} participants, expected "
            f"{EXPECTED_FRAME_CLIPS} and {EXPECTED_FRAME_PARTICIPANTS}"
        )


def build_fresh(
    population: pd.DataFrame,
    participants: dict[str, tuple[str, str]],
    seed: int = FRESH_SEED,
) -> pd.DataFrame:
    """The pre-registered draw. Pure: same inputs and seed give the same table."""
    frame = build_frame(population, participants)
    check_frame(frame, participants)
    rng = random.Random(seed)
    taken: set[str] = set()
    rows: list[dict[str, object]] = []
    for cell in FRESH_CELL_ORDER:
        pool = sorted(frame.index[frame == cell])
        rng.shuffle(pool)
        picked = 0
        for rank, ref in enumerate(pool):
            chosen = picked < ALLOCATION[cell] and participants[ref][0] not in taken
            if chosen:
                taken.add(participants[ref][0])
                picked += 1
            rows.append(
                {
                    "ref": ref,
                    "set": "fresh" if chosen else "pool",
                    "cell": cell,
                    "cell_rank": rank,
                    "weight": len(pool) / ALLOCATION[cell] if chosen else "",
                }
            )
        if picked < ALLOCATION[cell]:
            raise mc.InputError(f"cell {cell}: only {picked} of {ALLOCATION[cell]} clips available")
    fresh = sorted(r["ref"] for r in rows if r["set"] == "fresh")  # type: ignore[type-var]
    random.Random(f"{seed}:order").shuffle(fresh)
    number = {ref: n for n, ref in enumerate(fresh, start=1)}
    for row in rows:
        row.update(replaced_for="", seed=seed, order=number.get(str(row["ref"]), ""))
    sel = pd.DataFrame(rows)
    return sel[list(SELECTION_COLUMNS)]


def replace_unreadable(
    sel: pd.DataFrame, bad_refs: Iterable[str], participants: dict[str, tuple[str, str]]
) -> tuple[pd.DataFrame, list[str]]:
    """Replace an unreadable fresh clip by the next unused pool clip of its cell, in the shuffled
    order, from a participant not already in the sample. The replacement counts last."""
    sel = sel.copy()
    messages: list[str] = []
    for ref in sorted(set(bad_refs)):
        where = sel.index[sel["ref"] == ref]
        if len(where) != 1:
            messages.append(f"{ref}: not in the fresh selection, ignored")
            continue
        old = where[0]
        kind = sel.at[old, "set"]
        if kind == "replaced":
            continue  # done on an earlier run
        if kind != "fresh":
            messages.append(f"{ref}: a {kind} clip, not replaced (only fresh clips are)")
            continue
        used = {participants[r][0] for r in sel.loc[sel["set"] == "fresh", "ref"] if r != ref}
        pool = sel[(sel["set"] == "pool") & (sel["cell"] == sel.at[old, "cell"])]
        pool = pool.sort_values("cell_rank", key=lambda s: s.astype(int), kind="stable")
        new = next((r for r in pool["ref"] if participants[r][0] not in used), None)
        if new is None:
            raise mc.InputError(f"{ref}: no unused clip left in its cell to replace it")
        orders = pd.to_numeric(sel["order"], errors="coerce")
        nxt = int(orders.max()) + 1
        idx = sel.index[sel["ref"] == new][0]
        sel.loc[idx, ["set", "weight", "replaced_for", "order"]] = [
            "fresh", sel.at[old, "weight"], ref, str(nxt)
        ]  # fmt: skip
        sel.loc[old, ["set", "weight"]] = ["replaced", ""]
        messages.append(f"{ref}: replaced by {new}")
    return sel, messages


def validate_fresh(
    sel: pd.DataFrame,
    participants: dict[str, tuple[str, str]],
    frame: pd.Series | None,
    draw: pd.DataFrame | None,
) -> list[str]:
    """Structural rules of the file, and agreement with the current frame and a fresh draw."""
    if list(sel.columns) != list(SELECTION_COLUMNS):
        return [f"{SELECTION_NAME} must have the columns {', '.join(SELECTION_COLUMNS)}"]
    errors: list[str] = []
    if not sel["ref"].is_unique or not sel["ref"].map(lambda r: bool(HASH_RE.fullmatch(r))).all():
        errors.append("refs must be unique 8-character hashes")
    if (sel["seed"] != str(FRESH_SEED)).any():
        errors.append(f"seed must be {FRESH_SEED} in every row")
    if not sel["set"].isin(SETS).all():
        errors.append(f"set must be one of {', '.join(SETS)}")
    if not sel["cell"].isin(mc.CELL_ORDER).all():
        errors.append("cell must be one of the six cells")
    fresh = sel[sel["set"] == "fresh"]
    inside = sel[sel["set"].isin(SETS)]
    for cell, n_expected in ALLOCATION.items():
        n = int((fresh["cell"] == cell).sum())
        if n != n_expected:
            errors.append(f"cell {cell}: {n} fresh clips, expected {n_expected}")
        frame_n = int((inside["cell"] == cell).sum())
        w = pd.to_numeric(fresh.loc[fresh["cell"] == cell, "weight"], errors="coerce")
        if not np.allclose(w.to_numpy(dtype=float), frame_n / n_expected):
            errors.append(f"cell {cell}: weight is not frame clips / sampled clips")
    total = pd.to_numeric(fresh["weight"], errors="coerce").sum()
    if not np.isclose(total, len(inside)):
        errors.append(f"weights sum to {total:.3f}, not to the frame size {len(inside)}")
    people = [participants.get(r, ("?", ""))[0] for r in fresh["ref"]]
    if "?" in people or len(set(people)) != len(people):
        errors.append("the fresh clips must come from different, known participants")
    counted = sel[sel["set"].isin(["fresh", "replaced"])]
    first = pd.to_numeric(counted.loc[counted["replaced_for"] == "", "order"], errors="coerce")
    later = pd.to_numeric(counted.loc[counted["replaced_for"] != "", "order"], errors="coerce")
    n_first = len(first)
    if sorted(first.tolist()) != list(range(1, n_first + 1)) or not (
        later.notna().all() and later.is_unique and (later > n_first).all()
    ):
        errors.append("order must number the original clips 1..n, replacements after them")
    replaced = set(sel.loc[sel["set"] == "replaced", "ref"])
    targets = sel.loc[sel["replaced_for"] != "", "replaced_for"].tolist()
    if sorted(targets) != sorted(replaced) or len(set(targets)) != len(targets):
        errors.append("every replaced clip needs exactly one replacement and vice versa")
    if frame is not None and (set(inside["ref"]) != set(frame.index)):
        errors.append("the selection does not cover the current fresh frame")
    if frame is not None and (inside.set_index("ref")["cell"].to_dict() != frame.to_dict()):
        errors.append("the cells differ from the current frozen detector rates")
    if draw is not None:
        original = set(fresh.loc[fresh["replaced_for"] == "", "ref"]) | replaced
        if original != set(draw.loc[draw["set"] == "fresh", "ref"]):
            errors.append("the selection does not match a fresh draw from the current inputs")
    return errors


def selection_summary(
    sel: pd.DataFrame, population: pd.DataFrame, participants: dict[str, tuple[str, str]]
) -> pd.DataFrame:
    """One row per cell plus totals for the fresh clips: counts only, never refs."""
    pop = population.set_index("ref")
    inside = sel[sel["set"].isin(SETS)]
    rows = []
    for cell in (*mc.CELL_ORDER, "all"):
        part = sel[(sel["set"] == "fresh") & ((sel["cell"] == cell) | (cell == "all"))]
        refs = part["ref"].tolist()
        frame_n = len(inside) if cell == "all" else int((inside["cell"] == cell).sum())
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
                "left": sum(participants[r][1] == "left" for r in refs),
                "right": sum(participants[r][1] == "right" for r in refs),
            }
        )
    return pd.DataFrame(rows)


def cross_check(
    manual_dir: Path, fresh_refs: Iterable[str], participants: dict[str, tuple[str, str]]
) -> tuple[str, list[str]]:
    """Read-only check against the FT-24 selection: no fresh participant may belong to any row
    whose set is not plain "frame" (primary, replaced, hard, practice, replacements). Returns an
    info line and the errors (counts only)."""
    if not (manual_dir / mc.SELECTION_NAME).is_file():
        return NOTE_SKIPPED, []
    ft24 = mc.read_selection(manual_dir, ["ref", "set"])
    unknown = sorted(set(ft24["ref"]) - set(participants))
    if unknown:
        raise mc.InputError(f"{len(unknown)} clips of the manual_counts selection have no video")
    fresh_people = {participants[r][0] for r in fresh_refs}
    used = ft24[ft24["set"] != "frame"]
    hits = used[used["ref"].map(lambda r: participants[r][0] in fresh_people)]
    shared = {participants[r][0] for r in hits["ref"]}
    if not shared:
        return (
            f"cross-check with manual_counts selection: {len(fresh_people)} participants checked, 0 shared",
            [],
        )
    by_set = ", ".join(f"{k} {v}" for k, v in hits["set"].value_counts().sort_index().items())
    return "", [
        f"cross-check failed: {len(shared)} of {len(fresh_people)} fresh participants also "
        f"belong to the manual_counts selection (rows by set: {by_set})"
    ]


def file_hash8(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:8]


# ----------------------------------------------------------------------- command line


def _paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    out_dir = Path(args.out_dir or INTERIM_ROOT / "tuning")
    docs_dir = Path(args.docs_dir or DOCS_ROOT)
    summary = Path(args.summary or INTERIM_ROOT / "taps" / "taps_summary.csv")
    video_dir = Path(args.video_dir or VIDEO_ROOT)
    return out_dir, docs_dir, summary, video_dir


def _manual_dir(args: argparse.Namespace) -> Path:
    return Path(args.manual_dir or INTERIM_ROOT / "manual_counts")


def _load(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, tuple[str, str]], Path]:
    out_dir, docs_dir, summary, video_dir = _paths(args)
    return mc.load_population(docs_dir, summary), mc.participant_keys(video_dir), out_dir


def _print_facts(population: pd.DataFrame, participants: dict[str, tuple[str, str]]) -> None:
    facts = tuning_facts(mc.build_selection(population, participants), participants)
    print(", ".join(f"{k} {v}" for k, v in facts.items()))
    if facts["tuning_primary_overlap_participants"] != EXPECTED_TUNING_PRIMARY_OVERLAP:
        print(f"note: the expected overlap is {EXPECTED_TUNING_PRIMARY_OVERLAP} participants")


def cmd_select(args: argparse.Namespace) -> int:
    population, participants, out_dir = _load(args)
    path = out_dir / SELECTION_NAME
    if args.replace:
        if not path.is_file():
            raise mc.InputError(f"{SELECTION_NAME} not found; run tune_detector.py select first")
        sel, messages = replace_unreadable(
            mc.read_text(path), mc.read_unreadable(out_dir), participants
        )
        info, errors = cross_check(
            _manual_dir(args), sel.loc[sel["set"] == "fresh", "ref"], participants
        )
        if errors:
            print("\n".join(f"ERROR: {e}" for e in errors), file=sys.stderr)
            return 2
        mc.atomic_csv(sel, path)
        print("\n".join(messages) or "nothing to replace")
        print(info)
        print(f"{SELECTION_NAME} sha256 {file_hash8(path)}")
        return 0
    if path.exists() and not args.force:
        print(
            f"error: {SELECTION_NAME} already exists and is pre-registered; --force draws again "
            "and breaks the pre-registration",
            file=sys.stderr,
        )
        return 2
    sel = build_fresh(population, participants)
    frame = build_frame(population, participants)
    errors = validate_fresh(sel.astype(str), participants, frame, None)
    if errors:
        print("\n".join(f"ERROR: {e}" for e in errors), file=sys.stderr)
        return 1
    info, errors = cross_check(
        _manual_dir(args), sel.loc[sel["set"] == "fresh", "ref"], participants
    )
    if errors:  # nothing is written
        print("\n".join(f"ERROR: {e}" for e in errors), file=sys.stderr)
        return 2
    mc.atomic_csv(sel, path)
    print(f"population: {POPULATION}")
    print(selection_summary(sel.astype(str), population, participants).to_string(index=False))
    _print_facts(population, participants)
    print(info)
    print(f"wrote {SELECTION_NAME}: {int((sel['set'] == 'fresh').sum())} fresh clips")
    print(f"{SELECTION_NAME} sha256 {file_hash8(path)}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    population, participants, out_dir = _load(args)
    path = out_dir / SELECTION_NAME
    if not path.is_file():
        raise mc.InputError(f"{SELECTION_NAME} not found; run tune_detector.py select first")
    sel = mc.read_text(path)
    frame = build_frame(population, participants)
    check_frame(frame, participants)
    draw = build_fresh(population, participants).astype(str)
    errors = validate_fresh(sel, participants, frame, draw)
    info, cross_errors = cross_check(
        _manual_dir(args), sel.loc[sel["set"] == "fresh", "ref"], participants
    )
    errors += cross_errors
    counts = sel["set"].value_counts()
    print(
        f"frame: {len(frame)} clips, {len({participants[r][0] for r in frame.index})} participants"
    )
    print("selection: " + ", ".join(f"{k} {int(counts.get(k, 0))}" for k in SETS))
    _print_facts(population, participants)
    print(info)
    print(f"{SELECTION_NAME} sha256 {file_hash8(path)}")
    for error in errors:
        print(f"ERROR: {error}")
    print("check passed" if not errors else "check failed")
    return 1 if errors else 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select", help="draw the pre-registered fresh sample")
    mode = select.add_mutually_exclusive_group()
    mode.add_argument("--force", action="store_true", help="draw again (breaks pre-registration)")
    mode.add_argument("--replace", action="store_true", help="replace unreadable fresh clips")
    sub.add_parser("check", help="validate the fresh selection against a fresh draw")
    for p in sub.choices.values():
        for name in ("--out-dir", "--docs-dir", "--summary", "--video-dir", "--manual-dir"):
            p.add_argument(name, type=Path, default=None)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir, docs_dir, summary, _ = _paths(args)
    if not all(is_inside(p, INTERIM_ROOT) for p in (out_dir, summary, _manual_dir(args))):
        print(
            "error: --out-dir, --summary and --manual-dir must be inside data/interim",
            file=sys.stderr,
        )
        return 2
    if not is_inside(docs_dir, DOCS_ROOT):
        print("error: --docs-dir must be inside models/tapping/docs", file=sys.stderr)
        return 2
    try:
        return {"select": cmd_select, "check": cmd_check}[args.command](args)
    except mc.InputError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except OSError as err:
        name = Path(err.filename).name if isinstance(err.filename, str) else "a file"
        print(f"error: could not read/write {name}: {mc._reason(err)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
