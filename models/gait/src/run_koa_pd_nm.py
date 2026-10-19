"""GAIT-32: run every KOA-PD-NM video through ``predict_video`` and save one row per video.

    python models/gait/src/run_koa_pd_nm.py [--limit 20] [--dry-run]

Writes ``artifacts/koa_pd_nm/results.csv`` (one row per clip) and ``records.jsonl`` (the full
records). Rows hold group, severity folder, a hashed person key and the scores, never a file
name. A rerun skips clips that already have a row, so a stopped run carries on. GAIT-33 reads
the CSV. Exit code 0 = finished, 2 = the model bundle or pose model cannot be used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carepd_format import sample_keys, short_id
from check_mediapipe_mapping import DEFAULT_DATA
from explain_shap import DEFAULT_BUNDLE
from feature_contract import FeatureContractError
from gait_extract import FEATURE_NAMES
from model_bundle import BundleError
from predict_video import GaitPredictor
from video_to_joints import DEFAULT_MODEL, MAX_SECONDS

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "artifacts" / "koa_pd_nm"
GROUPS = ("NM", "PD", "KOA")
KNOWN_TWO_PEOPLE = {"001_PD_01_SV", "001_PD_02_SV"}  # a helper walks with the patient
CHECKS = ("frame_rate", "leg_visibility", "filled_share", "facing", "strides")
COLUMNS = [
    "clip",
    "group",
    "level",
    "person",
    "trial",
    "known_two_people",
    "status",
    "reasons",
    "warnings",
    "walk_seconds",
    *CHECKS,
    "predicted_class",
    "p0",
    "p1",
    "p2",
    "expected_severity",
    *FEATURE_NAMES,
    "model_version",
    "run_seconds",
]


@dataclass(frozen=True)
class Clip:
    path: Path
    clip: str  # short hash of the path inside the dataset: the key for resuming
    group: str  # NM, PD or KOA
    level: str  # PD: ML, MD, SV; KOA: EL, MD, SV; NM: empty
    person: str  # hashed group, severity folder and ID: one person may have several clips
    trial: str
    known_two_people: bool


def person_key(group: str, level: str, ident: str) -> str:
    """IDs restart in each severity folder (PD_ML/001 and PD_SV/001 are two people)."""
    return "p-" + hashlib.sha1(f"{group}_{level}_{ident}".encode()).hexdigest()[:8]


def parse(path: Path, data_dir: Path) -> Clip | None:
    """``001_PD_01_SV.MOV`` → PD, SV, person of PD 001, trial 01. None for other names."""
    parts = path.stem.split("_")
    if len(parts) not in (3, 4) or parts[1] not in GROUPS:
        return None
    ident, group, trial = parts[0], parts[1], parts[2]
    level = parts[3] if len(parts) == 4 else ""
    rel = path.relative_to(data_dir).as_posix()
    return Clip(
        path, short_id(rel), group, level, person_key(group, level, ident), trial,
        path.stem in KNOWN_TWO_PEOPLE,
    )  # fmt: skip


def find_clips(data_dir: Path) -> tuple[list[Clip], int]:
    """Every clip under ``data_dir``, sorted, and how many video files had another name."""
    files = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() == ".mov")
    clips = [c for c in (parse(p, data_dir) for p in files) if c is not None]
    return clips, len(files) - len(clips)


def spread(clips: list[Clip], limit: int) -> list[Clip]:
    """About ``limit`` clips, shared equally between the groups (fewer if a group is small)."""
    keep: set[str] = set()
    for k, group in enumerate(GROUPS):
        mine = [c.clip for c in clips if c.group == group]
        share = limit // len(GROUPS) + (1 if k < limit % len(GROUPS) else 0)
        keep |= set(sample_keys(mine, share))
    return [c for c in clips if c.clip in keep]


def done_clips(csv_path: Path) -> set[str]:
    if not csv_path.is_file():
        return set()
    with csv_path.open(newline="") as f:  # a row cut short by a stopped run is not done
        return {row["clip"] for row in csv.DictReader(f) if row.get("run_seconds")}


def row_of(clip: Clip, record: dict[str, Any] | None, error: str, seconds: float) -> dict:
    row: dict[str, Any] = {
        "clip": clip.clip,
        "group": clip.group,
        "level": clip.level,
        "person": clip.person,
        "trial": clip.trial,
        "known_two_people": int(clip.known_two_people),
        "run_seconds": round(seconds, 1),
    }
    if record is None:
        return row | {"status": "failed", "reasons": error}
    row |= {
        "status": "scored" if record["available"] else "not_scored",
        "reasons": " | ".join(record["reasons"]),
        "warnings": " | ".join(record["warnings"]),
        "model_version": record["model_version"],
    }
    if record.get("walk"):
        row["walk_seconds"] = record["walk"]["seconds"]
    for name, check in (record.get("gate") or {}).get("checks", {}).items():
        if name in CHECKS:
            row[name] = check["value"]
    if record["available"]:
        p = record["probabilities"]
        row |= {"predicted_class": record["predicted_class"], "p0": p[0], "p1": p[1], "p2": p[2]}
        row["expected_severity"] = record["expected_severity"]
        row |= record["feature_vector"]
    return row


def run(
    predictor: Any, clips: list[Clip], out_dir: Path, done: set[str]
) -> dict[str, dict[str, int]]:
    """Predict every clip not in ``done``, appending a row and a record per clip."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, jsonl = out_dir / "results.csv", out_dir / "records.jsonl"
    todo = [c for c in clips if c.clip not in done]
    new_file = not csv_path.is_file()
    counts: dict[str, dict[str, int]] = {}
    start = time.perf_counter()
    with csv_path.open("a", newline="") as f, jsonl.open("a") as js:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, restval="")
        if new_file:
            writer.writeheader()
        for i, clip in enumerate(todo, start=1):
            t0 = time.perf_counter()
            record, error = None, ""
            try:
                record = predictor.predict(clip.path)
            except Exception as err:  # noqa: BLE001 - one bad clip must not stop the run
                error = f"{type(err).__name__}: {err}"[:300]
            row = row_of(clip, record, error, time.perf_counter() - t0)
            writer.writerow(row)
            f.flush()
            if record is not None:
                js.write(json.dumps({"clip": clip.clip, **record}) + "\n")
                js.flush()
            tally = counts.setdefault(clip.group, {"scored": 0, "not_scored": 0, "failed": 0})
            tally[row["status"]] += 1
            spent = time.perf_counter() - start
            left = spent / i * (len(todo) - i)
            print(
                f"[{i}/{len(todo)}] {clip.clip} {clip.group}{'-' + clip.level if clip.level else ''}"
                f": {row['status']} in {row['run_seconds']:.0f} s; about {left / 60:.0f} min left",
                flush=True,
            )
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KOA-PD-NM through predict_video (GAIT-32)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    parser.add_argument("--limit", type=int, help="only about N clips, shared between the groups")
    parser.add_argument("--dry-run", action="store_true", help="list what would run, then stop")
    args = parser.parse_args(argv)

    clips, odd = find_clips(args.data_dir)
    if args.limit:
        clips = spread(clips, args.limit)
    done = done_clips(args.out_dir / "results.csv")
    todo = [c for c in clips if c.clip not in done]
    for group in GROUPS:
        mine = [c for c in clips if c.group == group]
        people = len({c.person for c in mine})
        print(f"{group:<4}{len(mine):4d} clips, {people:3d} people")
    print(
        f"{len(clips)} clips, {len(clips) - len(todo)} already done, {len(todo)} to run"
        + (f"; {odd} video file(s) with another name skipped" if odd else "")
    )
    if args.dry_run or not todo:
        return 0
    try:
        predictor = GaitPredictor(args.bundle, args.model, args.max_seconds)
    except (BundleError, FileNotFoundError, FeatureContractError) as err:
        print(f"Cannot start: {err}")
        return 2
    counts = run(predictor, clips, args.out_dir, done)
    for group, tally in counts.items():
        print(f"{group:<4}" + ", ".join(f"{k} {v}" for k, v in tally.items()))
    print(f"Wrote {args.out_dir / 'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
