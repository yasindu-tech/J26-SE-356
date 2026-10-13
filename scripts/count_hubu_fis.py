#!/usr/bin/env python3
"""Cohort audit for the HUBU-FIS finger-tapping dataset (task FT-06).

Every HUBU-FIS count quoted in a slide, report, data card or README must be
reproducible by this script (CLAUDE.md section 3.6). It prints and writes:

- file counts, and a strict 1:1 check between the label CSV and the videos;
- participants per group, and who has both hands;
- the UPDRS 3.4 rating per clip, per hand, per group and per participant;
- left-right asymmetry in the ratings;
- both candidate targets (CONTROL vs ID, and UPDRS >= 1 vs 0) with class ratios;
- with ``--probe-video``: fps, resolution, orientation, container and length per
  clip, crossed with group, to find device settings that only one group has.

The UPDRS column is a clinician's rating of the tapping task, not a
confirmation of disease: some control clips are rated 1 or more and some
patient clips are rated 0. Neither column is a model input (section 3.1).

Rules it follows:

- Any broken integrity rule stops the run with exit code 1 and writes nothing.
- A run without ``--probe-video`` refuses (exit 1, nothing written) to overwrite
  a counts JSON that has a ``video_probe`` section, so the confound tables
  cannot be dropped by accident. ``--allow-drop-probe`` overrides this on purpose.
  The rules are in models/tapping/src/hubu_fis.py.
- Participant and clip IDs are never printed. Anything flagged is shown as a
  short hash (``hubu_fis.pseudonym``). HUBU-FIS IDs are already pseudonyms and
  the ID space is tiny, so the hash only follows the repo convention; it does
  not anonymise anything.
- Output is deterministic: sorted keys, no timestamps, the CSV's sha256 for
  provenance.

Orientation: phones store some clips rotated and record the rotation as
metadata. The probe decodes the first frame twice with OpenCV, once with
auto-rotation off (stored size) and once on (displayed size, the frames a
landmark model actually receives), and checks the two agree with the rotation
metadata. The confound tables use the displayed size.

Outputs:
    models/tapping/docs/hubu_fis_counts.json   aggregates only, no IDs (committed)
    models/tapping/docs/hubu_fis_counts.md     the same, as text for the data card
    data/interim/hubu_fis_participants.csv     one row per participant (IDs; gitignored)
    data/interim/hubu_fis_clip_qc.csv          --probe-video only, one row per clip (IDs)

Exit codes: 0 ok, 1 an integrity rule failed, a clip could not be read, or the
run would drop an existing probe section (or the existing JSON is unreadable),
2 an input file or folder is missing.

Usage (from the repo root):
    python scripts/count_hubu_fis.py
    python scripts/count_hubu_fis.py --probe-video        # also open every clip (slower)
    python scripts/count_hubu_fis.py --csv path/to/fis_diagnostic.csv --video-dir path/to/videos
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "models" / "tapping" / "src"))

from hubu_fis import (  # noqa: E402
    GROUPS,
    UPDRS_VALUES,
    HubuFisError,
    check_match,
    list_videos,
    load_labels,
    participant_table,
    pseudonym,
)

RAW = REPO_ROOT / "data" / "raw" / "HUBU-FIS"
DEFAULT_CSV = RAW / "fis_diagnostic.csv"
DEFAULT_VIDEOS = RAW / "videos"
DEFAULT_OUT = REPO_ROOT / "models" / "tapping" / "docs"
DEFAULT_TABLES = REPO_ROOT / "data" / "interim"

COUNTS_STEM = "hubu_fis_counts"
PARTICIPANTS_CSV = "hubu_fis_participants.csv"
CLIP_QC_CSV = "hubu_fis_clip_qc.csv"

# Properties crossed with group in the confound check, as (key, column) pairs.
CONFOUND_PROPERTIES = (
    ("resolution", "resolution_display"),
    ("orientation", "orientation_display"),
    ("fps", "fps_rounded"),
    ("container", "container"),
)


# --------------------------------------------------------------------------- helpers


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts().sort_index().items()}


def _crosstab(rows: pd.Series, cols: pd.Series) -> dict[str, dict[str, int]]:
    tab = pd.crosstab(rows, cols)
    return {str(r): {str(c): int(tab.loc[r, c]) for c in tab.columns} for r in tab.index}


def _plain(counts: dict[str, int]) -> str:
    return ", ".join(f"{k} {v}" for k, v in counts.items())


def _ratio(counts: dict[str, int]) -> float | None:
    """Majority / minority class size, or None if a class is empty."""
    values = list(counts.values())
    if len(values) < 2 or min(values) == 0:
        return None
    return round(max(values) / min(values), 2)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- audit


def audit(
    labels: pd.DataFrame,
    participants: pd.DataFrame,
    videos: dict[str, Path],
    ignored: list[Path],
    csv_path: Path,
) -> dict[str, Any]:
    """Every count the tapping module quotes. Aggregates only: no IDs in the result."""
    both = participants[participants["has_left"] & participants["has_right"]]
    one_hand = participants.drop(index=both.index)

    asymmetry: dict[str, dict[str, Any]] = {}
    for group in GROUPS:
        diffs = both.loc[both["group"] == group, "abs_diff_rl"].astype(int)
        asymmetry[group] = {
            "participants_with_both_hands": int(len(diffs)),
            "asymmetric": int((diffs >= 1).sum()),
            "share_asymmetric": round(float((diffs >= 1).mean()), 3) if len(diffs) else None,
            "mean_abs_diff": round(float(diffs.mean()), 2) if len(diffs) else None,
            "abs_diff_counts": _counts(diffs),
        }

    group_clip = _counts(labels["group"])
    group_participant = _counts(participants["group"])
    impaired = (labels["UPDRS"] >= 1).map({False: "0", True: ">=1"})
    impaired_p = (participants["updrs_max"] >= 1).map({False: "0", True: ">=1"})

    return {
        "source": {"csv_name": csv_path.name, "csv_sha256": _sha256(csv_path)},
        "files": {
            "video_files": len(videos),
            "video_files_by_suffix": _counts(pd.Series([p.suffix for p in videos.values()])),
            "ignored_non_video_files": len(ignored),
            "labelled_clips": int(len(labels)),
            "labels_and_videos_match_1_to_1": True,
        },
        "participants": {
            "total": int(len(participants)),
            "by_group": group_participant,
            "with_both_hands": int(len(both)),
            "with_both_hands_by_group": _counts(both["group"]),
            "with_one_hand": int(len(one_hand)),
            "one_hand_only": {
                "left": int((one_hand["has_left"]).sum()),
                "right": int((one_hand["has_right"]).sum()),
            },
        },
        "clips": {
            "by_group": group_clip,
            "by_updrs": _counts(labels["UPDRS"]),
            "updrs_never_used": [v for v in UPDRS_VALUES if v not in set(labels["UPDRS"])],
            "by_hand": _counts(labels["hand"]),
            "group_x_updrs": _crosstab(labels["group"], labels["UPDRS"]),
            "group_x_hand": _crosstab(labels["group"], labels["hand"]),
            "control_clips_rated_ge1": int(
                ((labels["group"] == "CONTROL") & (labels["UPDRS"] >= 1)).sum()
            ),
            "patient_clips_rated_0": int(
                ((labels["group"] == "ID") & (labels["UPDRS"] == 0)).sum()
            ),
        },
        "participant_level_updrs": {
            "group_x_max_updrs": _crosstab(participants["group"], participants["updrs_max"]),
        },
        "asymmetry": asymmetry,
        "targets": {
            "group_control_vs_id": {
                "clips": group_clip,
                "clip_ratio": _ratio(group_clip),
                "participants": group_participant,
                "participant_ratio": _ratio(group_participant),
            },
            "updrs_ge1_vs_0": {
                "clips": _counts(impaired),
                "clip_ratio": _ratio(_counts(impaired)),
                "participants_max_hand": _counts(impaired_p),
                "participant_ratio_max_hand": _ratio(_counts(impaired_p)),
            },
        },
    }


# --------------------------------------------------------------------------- video probe


def _orientation(width: int, height: int) -> str:
    return "portrait" if height > width else "landscape" if width > height else "square"


def _first_frame_size(path: Path, auto_rotate: bool) -> tuple[int, int] | None:
    """(width, height) of the first decoded frame, or None if nothing decodes."""
    import cv2
    from hand_landmarks import open_video  # mutes OpenCV's stderr, which prints the file path

    capture = open_video(str(path))
    try:
        if not capture.isOpened():
            return None
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1 if auto_rotate else 0)
        ok, frame = capture.read()
        return (int(frame.shape[1]), int(frame.shape[0])) if ok else None
    finally:
        capture.release()


def probe_clip(path: Path) -> dict[str, Any]:
    """Read fps, frame count, rotation and the stored and displayed frame size of one clip.

    Raises HubuFisError (message without the clip ID) if the clip cannot be read. Clips are
    opened with ``hand_landmarks.open_video``: on a failed open OpenCV itself prints the full
    file path, which is the raw clip ID, so its stderr is muted around the call.
    """
    import cv2
    from hand_landmarks import open_video

    capture = open_video(str(path))
    try:
        if not capture.isOpened():
            raise HubuFisError("cannot be opened")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        prop_size = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        rotation = int(capture.get(cv2.CAP_PROP_ORIENTATION_META)) % 360
    finally:
        capture.release()

    stored = _first_frame_size(path, auto_rotate=False)
    displayed = _first_frame_size(path, auto_rotate=True)
    if fps <= 0 or frames <= 0 or stored is None or displayed is None:
        raise HubuFisError("no fps, no frames, or no decodable first frame")

    expected = (stored[1], stored[0]) if rotation in (90, 270) else stored
    return {
        "suffix": path.suffix,
        "container": path.suffix.lower().lstrip("."),
        "fps": round(fps, 2),
        "fps_rounded": int(round(fps)),
        "frames": frames,
        "seconds": round(frames / fps, 2),
        "rotation_meta": rotation,
        "prop_width": prop_size[0],
        "prop_height": prop_size[1],
        "stored_width": stored[0],
        "stored_height": stored[1],
        "display_width": displayed[0],
        "display_height": displayed[1],
        "resolution_stored": f"{stored[0]}x{stored[1]}",
        "resolution_display": f"{displayed[0]}x{displayed[1]}",
        "orientation_stored": _orientation(*stored),
        "orientation_display": _orientation(*displayed),
        "rotation_consistent": displayed == expected,
    }


def probe_videos(videos: dict[str, Path]) -> pd.DataFrame:
    """Probe every clip, sorted by clip ID. Raises listing every unreadable clip (hashed)."""
    rows, failed = [], []
    for clip in sorted(videos):
        try:
            rows.append({"ID": clip, **probe_clip(videos[clip])})
        except HubuFisError:
            failed.append(pseudonym(clip))
    if failed:
        raise HubuFisError(f"{len(failed)} clips could not be read, hashed clip IDs: {failed}")
    return pd.DataFrame(rows)


def confounds(qc: pd.DataFrame, labels: pd.DataFrame) -> dict[str, Any]:
    """Cross every recording property with group and list group-exclusive values."""
    import cv2

    merged = qc.merge(labels[["ID", "group"]], on="ID", how="inner", validate="one_to_one")
    by_group: dict[str, dict[str, dict[str, int]]] = {}
    exclusive: list[dict[str, Any]] = []
    for key, column in CONFOUND_PROPERTIES:
        tab = _crosstab(merged[column], merged["group"])
        by_group[key] = tab
        for value, cells in tab.items():
            present = [g for g in GROUPS if cells.get(g, 0) > 0]
            if len(present) == 1:
                exclusive.append(
                    {
                        "property": key,
                        "value": value,
                        "group": present[0],
                        "clips": cells[present[0]],
                    }
                )

    exclusive_values = {(e["property"], e["value"]) for e in exclusive}
    flagged = merged[
        [
            any((key, str(row[column])) in exclusive_values for key, column in CONFOUND_PROPERTIES)
            for _, row in merged.iterrows()
        ]
    ]
    seconds = merged.groupby("group")["seconds"].describe()[["min", "50%", "max"]]
    return {
        "tool": f"opencv {cv2.__version__}",
        "size_used_in_tables": "displayed (first frame decoded with auto-rotation on)",
        "by_group": by_group,
        "fps_exact_by_group": _crosstab(merged["fps"], merged["group"]),
        "seconds_by_group": {
            g: {k: round(float(v), 2) for k, v in seconds.loc[g].items()} for g in seconds.index
        },
        "group_exclusive_values": exclusive,
        "clips_with_any_group_exclusive_value": _counts(flagged["group"]),
        "orientation_check": {
            "clips_with_rotation_meta": int((merged["rotation_meta"] != 0).sum()),
            "rotation_meta_values": _counts(merged["rotation_meta"]),
            "stored_differs_from_displayed": int(
                (merged["resolution_stored"] != merged["resolution_display"]).sum()
            ),
            "displayed_inconsistent_with_rotation_meta": int(
                (~merged["rotation_consistent"]).sum()
            ),
            "prop_size_equals_stored": int(
                (
                    (merged["prop_width"] == merged["stored_width"])
                    & (merged["prop_height"] == merged["stored_height"])
                ).sum()
            ),
            "prop_size_equals_displayed": int(
                (
                    (merged["prop_width"] == merged["display_width"])
                    & (merged["prop_height"] == merged["display_height"])
                ).sum()
            ),
            "orientation_stored_by_group": _crosstab(merged["orientation_stored"], merged["group"]),
            "resolution_stored_by_group": _crosstab(merged["resolution_stored"], merged["group"]),
            "by_container": {
                c: {
                    "clips": int(len(sub)),
                    "with_rotation_meta": int((sub["rotation_meta"] != 0).sum()),
                    "stored_differs_from_displayed": int(
                        (sub["resolution_stored"] != sub["resolution_display"]).sum()
                    ),
                }
                for c, sub in merged.groupby("container")
            },
        },
    }


# --------------------------------------------------------------------------- report


def _md_table(header: list[str], rows: list[list[object]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


def _group_table(first: str, tab: dict[str, dict[str, int]]) -> list[str]:
    rows: list[list[object]] = [[v, *(cells.get(g, 0) for g in GROUPS)] for v, cells in tab.items()]
    return _md_table([first, *GROUPS], rows)


def report(result: dict[str, Any]) -> str:
    """Markdown summary for the data card. Contains no raw IDs."""
    f, p, c, t = result["files"], result["participants"], result["clips"], result["targets"]
    out = [
        "# HUBU-FIS cohort counts",
        "",
        "Generated by `python scripts/count_hubu_fis.py` (task FT-06). Do not edit by hand.",
        f"Label CSV: `{result['source']['csv_name']}`, sha256 `{result['source']['csv_sha256']}`.",
        "",
        "Groups: CONTROL = healthy control, ID = patient. UPDRS = the clinician's MDS-UPDRS 3.4",
        "finger-tapping rating of that hand: a label, never a model input, and not a",
        "confirmation of disease.",
        "",
        "## Files",
        "",
        f"- Video files: {f['video_files']} "
        f"({', '.join(f'{n} {s}' for s, n in f['video_files_by_suffix'].items())})",
        f"- Labelled clips: {f['labelled_clips']}; labels and videos match 1:1",
        f"- Non-video files ignored: {f['ignored_non_video_files']}",
        "",
        "## Participants",
        "",
        f"- Total: {p['total']} ({', '.join(f'{g} {n}' for g, n in p['by_group'].items())})",
        f"- With both hands: {p['with_both_hands']} "
        f"({', '.join(f'{g} {n}' for g, n in p['with_both_hands_by_group'].items())})",
        f"- With one hand: {p['with_one_hand']} (left only {p['one_hand_only']['left']}, "
        f"right only {p['one_hand_only']['right']})",
        "",
        "## UPDRS rating per clip",
        "",
        *_md_table(
            ["UPDRS", "clips", "share"],
            [[r, n, f"{n / f['labelled_clips']:.1%}"] for r, n in c["by_updrs"].items()],
        ),
        "",
        f"Ratings never used: {c['updrs_never_used'] or 'none'}. "
        f"Clips by hand: {', '.join(f'{h} {n}' for h, n in c['by_hand'].items())}.",
        "",
        "### Group x UPDRS (clips)",
        "",
        *_md_table(
            ["group", *(str(v) for v in UPDRS_VALUES)],
            [
                [g, *(c["group_x_updrs"].get(g, {}).get(str(v), 0) for v in UPDRS_VALUES)]
                for g in GROUPS
            ],
        ),
        "",
        f"Control clips rated 1 or more: {c['control_clips_rated_ge1']}. "
        f"Patient clips rated 0: {c['patient_clips_rated_0']}.",
        "",
        "### Group x highest UPDRS over a participant's hands (participants)",
        "",
        *_md_table(
            ["group", *(str(v) for v in UPDRS_VALUES)],
            [
                [
                    g,
                    *(
                        result["participant_level_updrs"]["group_x_max_updrs"]
                        .get(g, {})
                        .get(str(v), 0)
                        for v in UPDRS_VALUES
                    ),
                ]
                for g in GROUPS
            ],
        ),
        "",
        "## Left-right asymmetry in the rating (participants with both hands)",
        "",
        *_md_table(
            [
                "group",
                "participants",
                "asymmetric (R and L differ by >= 1)",
                "share",
                "mean abs(R - L)",
            ],
            [
                [
                    g,
                    a["participants_with_both_hands"],
                    a["asymmetric"],
                    "n/a" if a["share_asymmetric"] is None else f"{a['share_asymmetric']:.1%}",
                    a["mean_abs_diff"],
                ]
                for g, a in result["asymmetry"].items()
            ],
        ),
        "",
        "## Candidate targets",
        "",
        *_md_table(
            ["target", "unit", "counts", "majority : minority"],
            [
                [
                    "CONTROL vs ID",
                    "clip",
                    _plain(t["group_control_vs_id"]["clips"]),
                    f"{t['group_control_vs_id']['clip_ratio']:.2f}:1",
                ],
                [
                    "CONTROL vs ID",
                    "participant",
                    _plain(t["group_control_vs_id"]["participants"]),
                    f"{t['group_control_vs_id']['participant_ratio']:.2f}:1",
                ],
                [
                    "UPDRS >= 1 vs 0",
                    "clip",
                    _plain(t["updrs_ge1_vs_0"]["clips"]),
                    f"{t['updrs_ge1_vs_0']['clip_ratio']:.2f}:1",
                ],
                [
                    "UPDRS >= 1 vs 0",
                    "participant (higher hand)",
                    _plain(t["updrs_ge1_vs_0"]["participants_max_hand"]),
                    f"{t['updrs_ge1_vs_0']['participant_ratio_max_hand']:.2f}:1",
                ],
            ],
        ),
        "",
        "Both targets are imbalanced, so raw accuracy is not a headline metric: always",
        "predicting the majority class already scores its share. Report balanced accuracy",
        "and AUROC with participant-level bootstrap 95% CIs, plus MCC for comparison with",
        "the reference pipeline. Folds must be grouped by participant so a person's two",
        "hands never sit on both sides of a split. The primary target is not chosen here.",
        "",
    ]
    out += _probe_report(result.get("video_probe"))
    return "\n".join(out)


def _probe_report(probe: dict[str, Any] | None) -> list[str]:
    if probe is None:
        return ["## Recording properties", "", "Not probed (run with `--probe-video`).", ""]
    oc = probe["orientation_check"]
    out = [
        "## Recording properties by group (confound check)",
        "",
        f"Read with {probe['tool']}. Sizes are {probe['size_used_in_tables']}.",
        "",
    ]
    for key, _ in CONFOUND_PROPERTIES:
        out += [f"### {key}", "", *_group_table(key, probe["by_group"][key]), ""]
    out += [
        "### Length (seconds)",
        "",
        *_md_table(
            ["group", "min", "median", "max"],
            [[g, s["min"], s["50%"], s["max"]] for g, s in probe["seconds_by_group"].items()],
        ),
        "",
        "### Orientation cross-check",
        "",
        f"- Clips with rotation metadata: {oc['clips_with_rotation_meta']} "
        f"(values: {oc['rotation_meta_values']})",
        f"- Stored size differs from displayed size: {oc['stored_differs_from_displayed']}",
        f"- Displayed size inconsistent with rotation metadata: "
        f"{oc['displayed_inconsistent_with_rotation_meta']}",
        f"- OpenCV width/height property equals stored size: {oc['prop_size_equals_stored']}, "
        f"equals displayed size: {oc['prop_size_equals_displayed']}",
        f"- By container: {oc['by_container']}",
        "",
        "### Values found in only one group",
        "",
    ]
    exclusive = probe["group_exclusive_values"]
    if exclusive:
        out += _md_table(
            ["property", "value", "only in", "clips"],
            [[e["property"], e["value"], e["group"], e["clips"]] for e in exclusive],
        )
        out += [
            "",
            "Clips carrying at least one group-exclusive value: "
            + ", ".join(
                f"{g} {n}" for g, n in probe["clips_with_any_group_exclusive_value"].items()
            )
            + ".",
            "",
            "Why it matters: a value seen in only one group lets a model separate the groups",
            "by camera or device instead of by tapping. Recommended handling (not done here):",
            "log these properties per clip in QC, re-run every headline result without the",
            "flagged clips as a sensitivity analysis, and use resolution-independent features",
            "(landmark distances normalised by hand size) rather than raw pixel values.",
        ]
    else:
        out.append("None.")
    out.append("")
    return out


# --------------------------------------------------------------------------- main


def check_probe_kept(out_dir: Path, probing: bool, allow_drop: bool) -> None:
    """Refuse to replace a counts JSON that has a probe section with one that has none.

    Skipped for a probe run, with ``--allow-drop-probe``, or when there is no
    existing file. An existing file that cannot be read as a JSON object is an
    error too: it is never silently overwritten.
    """
    existing = out_dir / f"{COUNTS_STEM}.json"
    if probing or allow_drop or not existing.exists():
        return
    override = "Re-run with --probe-video, or pass --allow-drop-probe to overwrite it on purpose."
    try:
        previous = json.loads(existing.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:
        raise HubuFisError(
            f"cannot read the existing {existing} ({type(err).__name__}), so it was not "
            f"checked for a video_probe section and is left untouched. {override}"
        ) from err
    if not isinstance(previous, dict):
        raise HubuFisError(
            f"the existing {existing} is not a JSON object; left untouched. {override}"
        )
    if "video_probe" in previous:
        raise HubuFisError(
            f"the existing {existing} has a video_probe section (the confound tables) and this "
            f"run has no --probe-video, so writing would drop it. Nothing was written. {override}"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="label CSV (ID, UPDRS)")
    ap.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEOS, help="folder of clips")
    ap.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="where the aggregate JSON and .md go"
    )
    ap.add_argument(
        "--tables-dir",
        type=Path,
        default=DEFAULT_TABLES,
        help="where the ID-bearing tables go (keep this gitignored)",
    )
    ap.add_argument(
        "--probe-video", action="store_true", help="open every clip to read its properties (slow)"
    )
    ap.add_argument(
        "--allow-drop-probe",
        action="store_true",
        help="without --probe-video, overwrite an existing counts JSON that has a probe section",
    )
    args = ap.parse_args(argv)

    missing = [p for p in (args.csv, args.video_dir) if not p.exists()]
    if missing:
        for path in missing:
            print(f"error: {path} not found.", file=sys.stderr)
        print(
            "HUBU-FIS lives in data/raw/HUBU-FIS/ (fis_diagnostic.csv and videos/). It is never "
            "committed; see models/tapping/README.md.",
            file=sys.stderr,
        )
        return 2

    try:
        check_probe_kept(args.out, args.probe_video, args.allow_drop_probe)
        labels = load_labels(args.csv)
        videos, ignored = list_videos(args.video_dir)
        check_match(labels, videos)
        participants = participant_table(labels)
        qc = probe_videos(videos) if args.probe_video else None
    except HubuFisError as err:
        print(f"FAILED: {err}", file=sys.stderr)
        return 1

    result = audit(labels, participants, videos, ignored, args.csv)
    if qc is not None:
        result["video_probe"] = confounds(qc, labels)
    text = report(result)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{COUNTS_STEM}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (args.out / f"{COUNTS_STEM}.md").write_text(text)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    participants.to_csv(args.tables_dir / PARTICIPANTS_CSV)
    if qc is not None:
        qc.to_csv(args.tables_dir / CLIP_QC_CSV, index=False)

    print(text)
    one_hand = participants[~(participants["has_left"] & participants["has_right"])]
    hashed = ", ".join(sorted(pseudonym(p) for p in one_hand.index)) or "none"
    print(f"One-hand participants (hashed, terminal only, not written to files): {hashed}")
    print(f"Wrote {args.out / COUNTS_STEM}.json and .md; ID tables in {args.tables_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
