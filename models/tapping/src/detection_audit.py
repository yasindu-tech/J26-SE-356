"""Detection log and retention AUDIT for the tracked HUBU-FIS clips (task FT-17).

Reads ``tracking_manifest.csv`` (written by ``tracking.py``, FT-15/FT-16) and the label file, and
writes:

- ``data/interim/detection_log.csv``: one row per clip, hashed reference only (no raw ID, no
  UPDRS): detection rate, frames, longest missing run, gap count, flags, resolution,
  orientation, whether the file is stored rotated, fps, group, hand. Gitignored.
- ``models/tapping/docs/detection_audit.json``: aggregates only (counts and rates, no per-clip
  rows, no IDs), in the style of ``hubu_fis_counts.json``.
- a printed summary.

The audit answers one question: if clips below a detection-rate threshold were dropped, who
would be lost? For each threshold (default 0.80, 0.90, 0.95; a clip is retained when its rate is
AT OR ABOVE the threshold, the same side as the ``low_detection`` flag) it counts clips retained
and dropped overall and by group, resolution, orientation, stored rotation and UPDRS level,
which flag or condition accompanies each drop, and how many participants (key = group plus
number) lose one or both hands. Counts are always shown with percentages; a level with fewer
than 5 clips is marked ``small_cell``, and a dropped count of 1 to 4 is marked ``few_dropped``
(a percentage built on so few clips says almost nothing).

What this is NOT:

- It does not choose the retention threshold. The rule is still to be decided with the
  supervisor, and the output says so.
- It does not change any preprocessing. Group and UPDRS are read only to stratify this audit
  (to look for differential missingness); they are never an input to a decision here.
- A clip with no detection rate (a failed clip) is never counted as 0: it is reported
  separately as ``no_rate`` and is not retained.

Exit codes: 0 ok, 1 the manifest and the label file do not match clip for clip, or an input is
unreadable or breaks a rule, 2 an input file is missing or an output is outside its allowed
folder.

Usage (from the repo root):
    python models/tapping/src/detection_audit.py
    python models/tapping/src/detection_audit.py --thresholds 0.85,0.9
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from hubu_fis import HubuFisError, load_labels, pseudonym

REPO_ROOT = Path(__file__).resolve().parents[3]
INTERIM_ROOT = REPO_ROOT / "data" / "interim"
DEFAULT_MANIFEST = INTERIM_ROOT / "tracking_manifest.csv"
DEFAULT_LABELS = REPO_ROOT / "data" / "raw" / "HUBU-FIS" / "fis_diagnostic.csv"
DEFAULT_LOG = INTERIM_ROOT / "detection_log.csv"
DEFAULT_JSON = REPO_ROOT / "models" / "tapping" / "docs" / "detection_audit.json"

DEFAULT_THRESHOLDS = (0.80, 0.90, 0.95)
MIN_CELL = 5  # a level with fewer clips than this is marked small_cell
FINE_GRID = (0.99, 0.98, 0.97, 0.96, 0.95, 0.90, 0.80)
STRATA = ("group", "resolution", "orientation", "stored_rotated", "updrs")
RAW_ID = re.compile(r"(CONTROL|ID)\d")  # a raw clip or participant ID, e.g. CONTROL9001
NOT_DECIDED = (
    "This is an audit. The retention threshold is NOT chosen here and no preprocessing "
    "depends on it; the rule is still to be decided with the supervisor."
)

LOG_COLUMNS = (
    "ref",
    "participant_ref",
    "group",
    "hand",
    "resolution",
    "orientation",
    "stored_rotated",
    "fps",
    "frames_decoded",
    "frames_detected",
    "detection_rate",
    "longest_missing_run",
    "n_gaps",
    "flags",
)
MANIFEST_NEEDED = (
    "clip_id",
    "ref",
    "participant_ref",
    "group",
    "hand",
    "resolution",
    "orientation",
    "orientation_meta",
    "fps",
    "frames_decoded",
    "frames_detected",
    "detection_rate",
    "longest_missing_run",
    "n_gaps",
    "flags",
    "status",
)


class AuditError(ValueError):
    """The inputs break a rule (command line exit code 1)."""


# --------------------------------------------------------------------------- building the log


def build_log(manifest: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """The audit frame: manifest columns plus ``updrs`` (labels), one row per clip.

    The manifest and the label file must hold exactly the same clips; a difference in either
    direction is an error (listed by hash), never a silent drop. ``updrs`` is used only to
    stratify the audit and is not part of the detection log file.
    """
    missing = [c for c in MANIFEST_NEEDED if c not in manifest.columns]
    if missing:
        raise AuditError(f"the manifest lacks columns {missing[:4]}")
    if manifest["clip_id"].duplicated().any():
        raise AuditError("the manifest has a clip more than once")
    only_manifest = sorted(set(manifest["clip_id"]) - set(labels["ID"]))
    only_labels = sorted(set(labels["ID"]) - set(manifest["clip_id"]))
    if only_manifest or only_labels:
        shown = [pseudonym(c) for c in only_manifest + only_labels][:10]
        raise AuditError(
            f"manifest and label file differ: {len(only_manifest)} clips only in the manifest, "
            f"{len(only_labels)} only in the labels (hashes: {', '.join(shown)})"
        )
    merged = manifest.merge(labels[["ID", "UPDRS"]], left_on="clip_id", right_on="ID", how="left")
    merged = merged.rename(columns={"UPDRS": "updrs"}).drop(columns="ID")
    merged["stored_rotated"] = merged["orientation_meta"].fillna(0).astype(int) != 0
    merged["flags"] = merged["flags"].fillna("").astype(str)
    return merged.sort_values("ref").reset_index(drop=True)


def detection_log(frame: pd.DataFrame) -> pd.DataFrame:
    """The per-clip detection log: hashed reference only, no raw ID and no UPDRS."""
    return frame[list(LOG_COLUMNS)].reset_index(drop=True)


# --------------------------------------------------------------------------- the audit


def retained_mask(rate: pd.Series, threshold: float) -> pd.Series:
    """True where ``rate >= threshold``. NaN is not retained (and is never treated as 0)."""
    return (rate >= threshold).fillna(False).astype(bool)


def _cell(n: int, kept: int) -> dict[str, Any]:
    dropped = n - kept
    return {
        "clips": n,
        "retained": kept,
        "dropped": dropped,
        "retained_pct": round(100.0 * kept / n, 1) if n else None,
        "small_cell": n < MIN_CELL,
        "few_dropped": 0 < dropped < MIN_CELL,
    }


def stratified(frame: pd.DataFrame, keep: pd.Series, column: str) -> dict[str, dict[str, Any]]:
    """Retained and dropped counts for every level of ``column`` (levels as strings)."""
    out: dict[str, dict[str, Any]] = {}
    for level, part in frame.groupby(column, dropna=False, sort=True):
        out[str(level)] = _cell(len(part), int(keep[part.index].sum()))
    return out


def drop_causes(frame: pd.DataFrame, keep: pd.Series) -> dict[str, Any]:
    """Which flag or condition accompanies each dropped clip (a clip can have several)."""
    dropped = frame[~keep]
    causes: Counter[str] = Counter()
    for _, row in dropped.iterrows():
        if pd.isna(row["detection_rate"]):
            causes["no_rate_failed_clip"] += 1
            continue
        names = [f for f in str(row["flags"]).split(";") if f]
        causes.update(names or ["no_flag_rate_only"])
    return {
        "dropped_clips": len(dropped),
        "flag_counts": dict(sorted(causes.items())),
        "longest_missing_run_of_dropped": sorted(
            int(v) for v in dropped["longest_missing_run"].dropna()
        ),
    }


def participant_view(frame: pd.DataFrame, keep: pd.Series) -> dict[str, Any]:
    """How many participants lose no hand, one hand or all hands (key = participant hash)."""
    work = frame.assign(_keep=keep)
    per = work.groupby("participant_ref").agg(
        group=("group", "first"), hands=("hand", "nunique"), kept=("_keep", "sum")
    )
    per["lost"] = per["hands"] - per["kept"]

    def summarise(part: pd.DataFrame) -> dict[str, int]:
        two = part[part["hands"] == 2]
        one = part[part["hands"] == 1]
        return {
            "participants": len(part),
            "with_two_hands": len(two),
            "two_hands_lose_none": int((two["lost"] == 0).sum()),
            "two_hands_lose_one": int((two["lost"] == 1).sum()),
            "two_hands_lose_both": int((two["lost"] == 2).sum()),
            "with_one_hand": len(one),
            "one_hand_kept": int((one["lost"] == 0).sum()),
            "one_hand_lost": int((one["lost"] == 1).sum()),
        }

    out = {"all": summarise(per)}
    for group, part in per.groupby("group", sort=True):
        out[str(group)] = summarise(part)
    return out


def rate_distribution(rate: pd.Series) -> dict[str, Any]:
    finite = rate.dropna()
    return {
        "clips": len(rate),
        "no_rate": int(rate.isna().sum()),
        "min": round(float(finite.min()), 4) if len(finite) else None,
        "median": round(float(finite.median()), 4) if len(finite) else None,
        "clips_below": {f"{t:g}": int((finite < t).sum()) for t in FINE_GRID},
        "clips_at_1": int((finite == 1.0).sum()),
    }


def audit(frame: pd.DataFrame, thresholds: Sequence[float] = DEFAULT_THRESHOLDS) -> dict[str, Any]:
    """The whole audit as a JSON-ready dict of aggregates (no per-clip rows)."""
    for t in thresholds:
        if not (np.isfinite(t) and 0 < t <= 1):
            raise AuditError(f"a threshold must be in (0, 1], got {t!r}")
    per_threshold: dict[str, Any] = {}
    for t in thresholds:
        keep = retained_mask(frame["detection_rate"], t)
        entry: dict[str, Any] = {
            "overall": _cell(len(frame), int(keep.sum())),
            "by": {col: stratified(frame, keep, col) for col in STRATA},
            "drop_causes": drop_causes(frame, keep),
            "participants": participant_view(frame, keep),
        }
        per_threshold[f"{t:g}"] = entry
    return {
        "note": NOT_DECIDED,
        "retained_means": "detection_rate >= threshold",
        "min_cell": MIN_CELL,
        "clips": len(frame),
        "rate_distribution": rate_distribution(frame["detection_rate"]),
        "by_threshold": per_threshold,
    }


def check_no_raw_ids(text: str) -> None:
    """Refuse to write aggregate output that contains something shaped like a raw ID."""
    found = RAW_ID.search(text)
    if found:
        raise AuditError("the aggregate output contains a raw-ID-like string; not written")


# --------------------------------------------------------------------------- printing


def format_summary(result: dict[str, Any], lowest: pd.DataFrame) -> str:
    lines = [result["note"], ""]
    dist = result["rate_distribution"]
    lines.append(
        f"{result['clips']} clips; detection rate min {dist['min']}, median {dist['median']}; "
        f"{dist['clips_at_1']} clips at 1.0; no rate: {dist['no_rate']}"
    )
    lines.append("clips below: " + ", ".join(f"{k}: {v}" for k, v in dist["clips_below"].items()))
    for t, entry in result["by_threshold"].items():
        o = entry["overall"]
        lines += ["", f"== threshold {t} (retained = rate >= {t})"]
        lines.append(
            f"overall: {o['retained']}/{o['clips']} retained ({o['retained_pct']}%), "
            f"{o['dropped']} dropped"
        )
        for col, levels in entry["by"].items():
            cells = []
            for level, c in levels.items():
                mark = "*" if c["small_cell"] else ""
                cells.append(f"{level}{mark}: {c['retained']}/{c['clips']} ({c['retained_pct']}%)")
            lines.append(f"  by {col}: " + " | ".join(cells))
        causes = entry["drop_causes"]
        lines.append(
            f"  drop causes (flags on the {causes['dropped_clips']} dropped): "
            f"{causes['flag_counts'] or 'none'}; longest missing run of dropped: "
            f"{causes['longest_missing_run_of_dropped']}"
        )
        for who, p in entry["participants"].items():
            lines.append(
                f"  participants {who}: {p['participants']} "
                f"(two hands: {p['with_two_hands']} -> lose none {p['two_hands_lose_none']}, "
                f"one {p['two_hands_lose_one']}, both {p['two_hands_lose_both']}; "
                f"one hand: {p['with_one_hand']} -> lost {p['one_hand_lost']})"
            )
    lines += [
        "",
        f"* = small cell (fewer than {result['min_cell']} clips). A dropped count of 1 to "
        f"{result['min_cell'] - 1} is marked few_dropped in the JSON: percentages built on it "
        "say almost nothing.",
        "",
        "lowest detection rates (hashes only):",
        lowest.to_string(index=False),
    ]
    return "\n".join(lines)


def lowest_clips(frame: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    cols = ["ref", "group", "resolution", "detection_rate", "longest_missing_run", "flags"]
    return frame.sort_values(["detection_rate", "ref"]).head(n)[cols].round(4)


# --------------------------------------------------------------------------- command line


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def parse_thresholds(text: str) -> tuple[float, ...]:
    try:
        return tuple(sorted({float(v) for v in text.split(",") if v.strip()}))
    except ValueError as err:
        raise AuditError(f"cannot read thresholds {text!r}") from err


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG, help="per-clip log (inside interim)")
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON, help="aggregate-only JSON")
    ap.add_argument("--interim-root", type=Path, default=INTERIM_ROOT)
    ap.add_argument("--thresholds", default=",".join(f"{t:g}" for t in DEFAULT_THRESHOLDS))
    args = ap.parse_args(argv)

    for path in (args.manifest, args.labels):
        if not path.is_file():
            print(f"error: input not found: {path.name}", file=sys.stderr)
            return 2
    if not _inside(args.log, args.interim_root):
        print(f"error: the per-clip log must be inside {args.interim_root}", file=sys.stderr)
        return 2
    try:
        thresholds = parse_thresholds(args.thresholds)
        manifest = pd.read_csv(args.manifest)
        frame = build_log(manifest, load_labels(args.labels))
        result = audit(frame, thresholds)
        text = json.dumps(result, indent=2, sort_keys=True) + "\n"
        check_no_raw_ids(text)
    except (
        AuditError,
        HubuFisError,
        OSError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as err:
        print(f"FAILED: {err}", file=sys.stderr)
        return 1

    args.log.parent.mkdir(parents=True, exist_ok=True)
    detection_log(frame).to_csv(args.log, index=False)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(text)
    print(format_summary(result, lowest_clips(frame)))
    print(f"\nwrote {args.log.name} (per clip, hashes only) and {args.json.name} (aggregates)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
