"""Triage of flagged and low-quality clips: a visual review set and label-free checks (FT-18).

A high detection rate does not prove the landmarks are right: fingertips sit short of the nail,
the handedness label flickers, and ``num_hands=1`` could lock onto another hand. This tool picks
a small set of clips to LOOK at, draws them, and computes a few automatic checks that need no
labels. It decides nothing: ``visual_verdict`` in the triage table is left empty for a person.

The review set (selection uses ONLY these manifest columns: ``clip_id``, ``ref``, ``flags``,
``detection_rate``, ``longest_missing_run``, ``resolution``; never UPDRS, never group):

- (a) every clip with a review flag (any flag not starting with ``info_``);
- (b) among clips without a review flag, the N lowest detection rates and the N longest
  missing runs (default 5 each; a clip in both lists appears once, with both reasons);
- (c) control clips: no flag at all and detection rate exactly 1.0, drawn with a fixed random
  seed, one per resolution present first (where possible), the rest from what remains.

Outputs, all under ``data/interim/triage/`` (gitignored), named by the clip's 8-character hash
only (``hubu_fis.pseudonym``); raw clip IDs never appear in a file name, title, print or error:

- ``review_set.csv``: hashed ref and the reason(s) each clip is in the set;
- ``<ref>_sheet.png``: a contact sheet, 12 evenly spaced frames plus frames around the longest
  missing run (and around the largest wrist jump), landmarks drawn, cropped to the hand's reach;
- ``<ref>_overlay.gif``: an overlay GIF around the longest gap, for the flagged clips and the
  lowest or longest-gap clips only (not for control clips);
- ``triage_table.csv``: one row per reviewed clip with the automatic checks and an EMPTY
  ``visual_verdict`` column for you to fill in.

Automatic checks (``automatic_checks``; none uses a label, NaN is never zero):

- ``wrist_jump_frames``: pairs of consecutive detected frames where the wrist moves more than
  0.5 hand sizes (median wrist-to-knuckle-9 length); a hand switch or tracking jump shows here.
  ``wrist_jump_after_gap`` counts the same across a gap (last detected frame before it to the
  first after), where real movement can also be large, so read it with the sheet;
- ``edge_frames_frac``: share of detected frames with any landmark within 2% of the frame edge;
- ``distance_outlier_frac``: share of frames whose normalised thumb-index distance (FT-13) lies
  beyond three interquartile ranges outside the quartiles (the signal is bimodal, so this only
  catches gross outliers);
- ``handedness_changes``: label changes between consecutive detected frames (informational:
  the label flickers on correctly tracked hands, and the hand comes from the clip name).

Alternative confidence (``--alt-conf 0.3``): re-runs the ``low_detection`` clip(s) with
``min_hand_detection_confidence`` and ``min_hand_presence_confidence`` lowered, through
``tracking.py`` into a SEPARATE folder (``data/interim/landmarks_alt_conf03`` by default; the
main folder is never written), and reports the detection rate before and after, which frames
were newly detected, whether the frames both runs detect agree, and a sheet of the new frames.

Exit codes: 0 ok, 1 an input or a rule failed, 2 an input is missing or an output is outside
``data/interim``.

Usage (from the repo root):
    python models/tapping/src/triage.py                       # sets, sheets, GIFs, table
    python models/tapping/src/triage.py --no-gifs
    python models/tapping/src/triage.py --alt-conf 0.3
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import hand_landmarks as hl
import hand_scale as hs
import numpy as np
import overlay_gif as og
import pandas as pd
import pyarrow as pa
import tracking
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[3]
INTERIM_ROOT = REPO_ROOT / "data" / "interim"
DEFAULT_MANIFEST = INTERIM_ROOT / "tracking_manifest.csv"
DEFAULT_LANDMARKS = INTERIM_ROOT / "landmarks"
DEFAULT_OUT = INTERIM_ROOT / "triage"
DEFAULT_ALT_OUT = INTERIM_ROOT / "landmarks_alt_conf03"
SELECTION_COLUMNS = (
    "clip_id",
    "ref",
    "flags",
    "detection_rate",
    "longest_missing_run",
    "resolution",
)

SEED = 18
N_LOWEST = 5
N_LONGEST = 5
N_CONTROL = 8
JUMP_HAND_SIZES = 0.5
EDGE_MARGIN = 0.02
OUTLIER_IQR = 3.0
N_EVEN_FRAMES = 12
TILE = 240
SHEET_COLUMNS = 4
GIF_WIDTH = 360
GIF_SECONDS = 4.0
GIF_FPS = 15.0

TABLE_COLUMNS = (
    "ref",
    "reasons",
    "resolution",
    "detection_rate",
    "longest_missing_run",
    "flags",
    "wrist_jump_frames",
    "wrist_jump_after_gap",
    "edge_frames_frac",
    "distance_outlier_frac",
    "handedness_changes",
    "visual_verdict",
)


class TriageError(ValueError):
    """An input or a rule failed (command line exit code 1)."""


# --------------------------------------------------------------------------- the review set


def has_review_flag(flags: object) -> bool:
    """True if any flag in the ``;``-joined string is not informational (``info_...``)."""
    if not isinstance(flags, str):
        return False
    return any(f and not f.startswith("info_") for f in flags.split(";"))


def has_any_flag(flags: object) -> bool:
    return isinstance(flags, str) and any(flags.split(";"))


def select_review_set(
    manifest: pd.DataFrame,
    *,
    seed: int = SEED,
    n_lowest: int = N_LOWEST,
    n_longest: int = N_LONGEST,
    n_control: int = N_CONTROL,
) -> pd.DataFrame:
    """The review set: ``clip_id``, ``ref``, ``reasons`` (list of text), one row per clip.

    Reads only ``SELECTION_COLUMNS``; any other column of ``manifest`` (group, UPDRS, ...) is
    not looked at, so it cannot influence the choice.
    """
    missing = [c for c in SELECTION_COLUMNS if c not in manifest.columns]
    if missing:
        raise TriageError(f"the manifest lacks columns {missing}")
    m = manifest[list(SELECTION_COLUMNS)].copy()
    m["_review"] = m["flags"].map(has_review_flag)
    m["_any"] = m["flags"].map(has_any_flag)

    reasons: dict[str, list[str]] = {}

    def add(ref: str, why: str) -> None:
        reasons.setdefault(ref, []).append(why)

    for _, row in m[m["_review"]].sort_values("ref").iterrows():
        for flag in str(row["flags"]).split(";"):
            if flag and not flag.startswith("info_"):
                add(row["ref"], f"flag:{flag}")

    quiet = m[~m["_review"] & m["detection_rate"].notna()]
    for _, row in quiet.sort_values(["detection_rate", "ref"]).head(n_lowest).iterrows():
        add(row["ref"], "lowest_detection_unflagged")
    longest = quiet.sort_values(["longest_missing_run", "ref"], ascending=[False, True])
    for _, row in longest.head(n_longest).iterrows():
        add(row["ref"], "longest_gap_unflagged")

    pool = m[~m["_any"] & (m["detection_rate"] == 1.0) & ~m["ref"].isin(reasons)]
    for ref in control_sample(pool, seed, n_control):
        add(ref, "control_unflagged_rate_1")

    chosen = m[m["ref"].isin(reasons)].sort_values("ref")
    return pd.DataFrame(
        {
            "clip_id": chosen["clip_id"].to_numpy(),
            "ref": chosen["ref"].to_numpy(),
            "reasons": [reasons[r] for r in chosen["ref"]],
        }
    ).reset_index(drop=True)


def control_sample(pool: pd.DataFrame, seed: int, n: int) -> list[str]:
    """``n`` refs from ``pool``: one per resolution first (where possible), then the rest.

    Reproducible for a given seed: everything is sorted before the seeded shuffle.
    """
    rng = random.Random(seed)
    by_res: dict[str, list[str]] = {}
    for res, part in pool.groupby("resolution", sort=True):
        by_res[str(res)] = sorted(part["ref"])
    picked: list[str] = []
    resolutions = sorted(by_res)
    rng.shuffle(resolutions)
    for res in resolutions:
        if len(picked) >= n:
            break
        options = by_res[res]
        choice = options[rng.randrange(len(options))]
        picked.append(choice)
    rest = sorted(set(pool["ref"]) - set(picked))
    rng.shuffle(rest)
    picked += rest[: max(n - len(picked), 0)]
    return sorted(picked)


# --------------------------------------------------------------------------- automatic checks


def _wrist_xy(df: pd.DataFrame) -> np.ndarray:
    xy = np.column_stack(
        [
            df[hl.landmark_column("x_px", hs.WRIST)].to_numpy(float),
            df[hl.landmark_column("y_px", hs.WRIST)].to_numpy(float),
        ]
    )
    xy[~df["detected"].to_numpy(bool)] = np.nan
    return xy


def wrist_jumps(df: pd.DataFrame, hand_sizes: float = JUMP_HAND_SIZES) -> dict[str, Any]:
    """Wrist jumps larger than ``hand_sizes`` hand sizes, between frames and across gaps.

    Only pairs where both frames have a hand are compared: a missing frame is NaN and is never
    treated as position 0. The hand size is the clip's median wrist-to-knuckle-9 length.
    """
    lengths = hs.scale_length(df, "w9")
    lengths = lengths[np.isfinite(lengths)]
    size = float(np.median(lengths)) if lengths.size else float("nan")
    xy = _wrist_xy(df)
    ok = np.isfinite(xy).all(axis=1)
    out: dict[str, Any] = {
        "hand_size_px": size,
        "wrist_jump_frames": 0,
        "wrist_jump_after_gap": 0,
        "jump_frame": None,
    }
    if not np.isfinite(size) or size <= 0 or ok.sum() < 2:
        out["wrist_jump_frames"] = out["wrist_jump_after_gap"] = float("nan")
        return out
    idx = np.flatnonzero(ok)
    steps = np.hypot(*(xy[idx[1:]] - xy[idx[:-1]]).T) / size
    adjacent = (idx[1:] - idx[:-1]) == 1
    big = steps > hand_sizes
    out["wrist_jump_frames"] = int((big & adjacent).sum())
    out["wrist_jump_after_gap"] = int((big & ~adjacent).sum())
    if big.any():
        out["jump_frame"] = int(idx[1:][np.argmax(np.where(big, steps, -np.inf))])
    return out


def edge_fraction(df: pd.DataFrame, margin: float = EDGE_MARGIN) -> float:
    """Share of detected frames with any landmark within ``margin`` of the frame edge."""
    det = df["detected"].to_numpy(bool)
    if not det.any():
        return float("nan")
    x = df[[hl.landmark_column("x", i) for i in range(hl.N_LANDMARKS)]].to_numpy(float)[det]
    y = df[[hl.landmark_column("y", i) for i in range(hl.N_LANDMARKS)]].to_numpy(float)[det]
    near = (x < margin) | (x > 1 - margin) | (y < margin) | (y > 1 - margin)
    return float(near.any(axis=1).mean())


def distance_outlier_fraction(df: pd.DataFrame, k: float = OUTLIER_IQR) -> float:
    """Share of frames with a normalised thumb-index distance beyond ``k`` IQRs of the quartiles."""
    values = hs.normalised_distance(df).values
    finite = values[np.isfinite(values)]
    if finite.size < 4:
        return float("nan")
    q1, q3 = np.percentile(finite, [25, 75])
    spread = q3 - q1
    outside = (finite < q1 - k * spread) | (finite > q3 + k * spread)
    return float(outside.mean())


def handedness_changes(df: pd.DataFrame) -> int:
    """Label changes between consecutive detected frames (informational only)."""
    labels = df.loc[df["detected"].to_numpy(bool), "handedness_label"].tolist()
    return sum(a != b for a, b in zip(labels, labels[1:], strict=False))


def automatic_checks(df: pd.DataFrame) -> dict[str, Any]:
    jumps = wrist_jumps(df)
    return {
        "wrist_jump_frames": jumps["wrist_jump_frames"],
        "wrist_jump_after_gap": jumps["wrist_jump_after_gap"],
        "edge_frames_frac": edge_fraction(df),
        "distance_outlier_frac": distance_outlier_fraction(df),
        "handedness_changes": handedness_changes(df),
        "_jump_frame": jumps["jump_frame"],
    }


# --------------------------------------------------------------------------- sheets


def longest_gap(detected: np.ndarray) -> tuple[int, int] | None:
    """(first, last) frame index of the longest run without a hand, or None if there is none."""
    best: tuple[int, int] | None = None
    start: int | None = None
    d = np.asarray(detected, dtype=bool)
    for i, value in enumerate(np.append(d, True)):
        if not value and start is None:
            start = i
        elif value and start is not None:
            if best is None or (i - 1 - start) > (best[1] - best[0]):
                best = (start, i - 1)
            start = None
    return best


def sheet_frames(
    detected: np.ndarray, jump_frame: int | None = None, n_even: int = N_EVEN_FRAMES
) -> list[tuple[int, str]]:
    """Frame indices for a contact sheet: evenly spaced, around the longest gap, around a jump."""
    n = len(detected)
    if n == 0:
        return []
    tags: dict[int, str] = {}
    for i in np.linspace(0, n - 1, num=min(n_even, n), dtype=int):
        tags.setdefault(int(i), "")
    gap = longest_gap(detected)
    if gap is not None:
        first, last = gap
        for index, tag in (
            (first - 1, "before gap"),
            (first + (last - first) // 2, "gap"),
            (last, "gap end"),
            (last + 1, "after gap"),
        ):
            if 0 <= index < n:
                tags[index] = tag
    if jump_frame is not None:
        tags[jump_frame] = "jump"
        if jump_frame > 0:
            tags.setdefault(jump_frame - 1, "before jump")
    return sorted(tags.items())


def draw_tile(
    frame_bgr: np.ndarray, xy: np.ndarray, box: tuple[int, int, int], tag: str, index: int
) -> Image.Image:
    """One tile: the cropped frame with the skeleton, a red border when there is no hand."""
    import cv2

    x0, y0, side = box
    rgb = cv2.cvtColor(frame_bgr[y0 : y0 + side, x0 : x0 + side], cv2.COLOR_BGR2RGB)
    tile = Image.fromarray(rgb).resize((TILE, TILE), Image.Resampling.LANCZOS)
    og.draw_skeleton(tile, og.to_panel(xy, box, TILE))
    draw = ImageDraw.Draw(tile)
    if not np.isfinite(xy).any():
        draw.rectangle([0, 0, TILE - 1, TILE - 1], outline=(255, 0, 0), width=4)
    font = ImageFont.load_default(size=13)
    label = f"f{index} {tag}".strip() + ("" if np.isfinite(xy).any() else " (no hand)")
    draw.rectangle([0, 0, TILE, 17], fill=(0, 0, 0))
    draw.text((3, 2), label, fill=(255, 255, 255), font=font)
    return tile


def make_sheet(
    df: pd.DataFrame,
    meta: dict[str, Any],
    cap: Any,
    ref: str,
    jump_frame: int | None,
    chosen: list[tuple[int, str]] | None = None,
) -> Image.Image:
    """The contact sheet of one clip (landmarks drawn, cropped to the hand's whole reach)."""
    if chosen is None:
        chosen = sheet_frames(df["detected"].to_numpy(bool), jump_frame)
    size = (int(meta["width"]), int(meta["height"]))
    pts = df["pts_ms"].to_numpy(float)
    frames = og.read_frames(
        cap, [i for i, _ in chosen], {i: float(pts[i]) for i, _ in chosen}, size, ref
    )
    points = np.vstack([og.landmark_xy(df, i) for i in range(len(df))])
    hand = float(np.nanmedian(hs.scale_length(df, "w9")))
    pad = og.CROP_PAD_HAND_SIZES * (hand if np.isfinite(hand) else 0.0)
    box = og.crop_square(points, size[0], size[1], pad)
    tiles = [draw_tile(frames[i], og.landmark_xy(df, i), box, tag, i) for i, tag in chosen]
    rows = math.ceil(len(tiles) / SHEET_COLUMNS)
    head = 24
    sheet = Image.new("RGB", (SHEET_COLUMNS * TILE, head + rows * TILE), (30, 30, 30))
    ImageDraw.Draw(sheet).text(
        (4, 4),
        f"clip {ref}: {len(df)} frames",
        fill=(255, 255, 255),
        font=ImageFont.load_default(size=14),
    )
    for k, tile in enumerate(tiles):
        sheet.paste(tile, ((k % SHEET_COLUMNS) * TILE, head + (k // SHEET_COLUMNS) * TILE))
    return sheet


def gif_window(df: pd.DataFrame) -> float:
    """Start (seconds of pts_ms) of a GIF window centred on the longest gap, else at 3 s."""
    pts = df["pts_ms"].to_numpy(float)
    gap = longest_gap(df["detected"].to_numpy(bool))
    last = float(np.nanmax(pts)) / 1000.0
    centre = float(pts[(gap[0] + gap[1]) // 2]) / 1000.0 if gap else 5.0
    return float(np.clip(centre - GIF_SECONDS / 2, 0.0, max(last - GIF_SECONDS, 0.0)))


# --------------------------------------------------------------------------- the run


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def review_table(
    selected: pd.DataFrame, manifest: pd.DataFrame, checks: dict[str, dict[str, Any]]
) -> pd.DataFrame:
    """One row per reviewed clip: reasons, manifest numbers, automatic checks, empty verdict."""
    base = manifest.set_index("ref")
    rows = []
    for _, item in selected.iterrows():
        ref = item["ref"]
        m = base.loc[ref]
        c = checks[ref]
        rows.append(
            {
                "ref": ref,
                "reasons": ";".join(item["reasons"]),
                "resolution": m["resolution"],
                "detection_rate": round(float(m["detection_rate"]), 4),
                "longest_missing_run": int(m["longest_missing_run"]),
                "flags": "" if pd.isna(m["flags"]) else m["flags"],
                "wrist_jump_frames": c["wrist_jump_frames"],
                "wrist_jump_after_gap": c["wrist_jump_after_gap"],
                "edge_frames_frac": round(float(c["edge_frames_frac"]), 4),
                "distance_outlier_frac": round(float(c["distance_outlier_frac"]), 4),
                "handedness_changes": c["handedness_changes"],
                "visual_verdict": "",
            }
        )
    return pd.DataFrame(rows, columns=list(TABLE_COLUMNS))


def wants_gif(reasons: Sequence[str]) -> bool:
    """GIFs only for flagged and lowest or longest-gap clips, never for control clips."""
    return any(not r.startswith("control_") for r in reasons)


def compare_runs(main_df: pd.DataFrame, alt_df: pd.DataFrame) -> dict[str, Any]:
    """Detection before and after an alternative run of the same clip, and agreement."""
    a, b = main_df["detected"].to_numpy(bool), alt_df["detected"].to_numpy(bool)
    if len(main_df) != len(alt_df):
        raise TriageError("the two runs have different frame counts")
    both = a & b
    px = [hl.landmark_column(p, i) for p in ("x_px", "y_px") for i in range(hl.N_LANDMARKS)]
    diff = (
        float(
            np.nanmax(
                np.abs(main_df.loc[both, px].to_numpy(float) - alt_df.loc[both, px].to_numpy(float))
            )
        )
        if both.any()
        else float("nan")
    )
    return {
        "frames": len(a),
        "detected_main": int(a.sum()),
        "detected_alt": int(b.sum()),
        "rate_main": float(a.mean()),
        "rate_alt": float(b.mean()),
        "newly_detected": [int(i) for i in np.flatnonzero(b & ~a)],
        "lost": [int(i) for i in np.flatnonzero(a & ~b)],
        "max_px_difference_on_shared_frames": diff,
        "longest_gap_main": longest_gap(a),
        "longest_gap_alt": longest_gap(b),
    }


def alt_confidence_run(
    refs: Sequence[str],
    manifest: pd.DataFrame,
    args: argparse.Namespace,
) -> int:
    """Re-run ``refs`` with lowered confidence into a separate folder and compare (read-only
    for the main landmarks folder; the refusal to mix parameter sets is shown with --dry-run)."""
    conf = str(args.alt_conf)
    base = [
        "--video-dir", str(args.video_dir),
        "--min-hand-detection-confidence", conf,
        "--min-hand-presence-confidence", conf,
        "--interim-root", str(args.interim_root),
        "--workers", "1",
    ]  # fmt: skip
    ids = manifest.set_index("ref")["clip_id"]
    clip_list = ",".join(refs)
    print(f"\n== alternative confidence {conf}: {len(refs)} clip(s) -> {args.alt_out_dir.name}/")
    code = tracking.main([*base, "--clips", clip_list, "--out-dir", str(args.alt_out_dir)])
    if code != 0:
        print(f"FAILED: the alternative run exited with {code}", file=sys.stderr)
        return 1
    refused = tracking.main(
        [*base, "--clips", clip_list, "--out-dir", str(args.landmarks_dir), "--dry-run"]
    )
    print(f"mixing check on the main folder (dry run, no --force): exit {refused} (2 = refused)")
    for ref in refs:
        main_df, meta = hl.read_landmarks(args.landmarks_dir / f"{ids[ref]}.parquet")
        alt_df, _ = hl.read_landmarks(args.alt_out_dir / f"{ids[ref]}.parquet")
        result = compare_runs(main_df, alt_df)
        print(f"clip {ref}: {result}")
        new = result["newly_detected"]
        if new and not args.no_sheets:
            chosen = [(i, "new") for i in new[:: max(len(new) // 8, 1)][:8]]
            video = og.find_video(ref, args.video_dir)
            if video is not None:
                cap = og.open_video(str(video))
                try:
                    sheet = make_sheet(alt_df, meta, cap, ref, None, chosen)
                finally:
                    cap.release()
                sheet.save(args.out_dir / f"{ref}_alt_conf_new_frames.png")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--landmarks-dir", type=Path, default=DEFAULT_LANDMARKS)
    ap.add_argument("--video-dir", type=Path, default=og.DEFAULT_VIDEO_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--interim-root", type=Path, default=INTERIM_ROOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-gifs", action="store_true")
    ap.add_argument("--no-sheets", action="store_true")
    ap.add_argument(
        "--alt-conf", type=float, default=None, help="e.g. 0.3: re-run low_detection clips"
    )
    ap.add_argument("--alt-out-dir", type=Path, default=DEFAULT_ALT_OUT)
    args = ap.parse_args(argv)

    if not args.manifest.is_file() or not args.landmarks_dir.is_dir():
        print("error: the manifest or the landmarks folder is missing", file=sys.stderr)
        return 2
    outputs = [args.out_dir, *([args.alt_out_dir] if args.alt_conf is not None else [])]
    if any(not _inside(p, args.interim_root) for p in outputs):
        print(f"error: outputs must be inside {args.interim_root}", file=sys.stderr)
        return 2
    if args.alt_conf is not None and args.alt_out_dir.resolve() == args.landmarks_dir.resolve():
        print("error: --alt-out-dir must not be the main landmarks folder", file=sys.stderr)
        return 2

    try:
        manifest = pd.read_csv(args.manifest)
        selected = select_review_set(manifest, seed=args.seed)
    except (TriageError, OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as err:
        print(f"FAILED: {err}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"ref": selected["ref"], "reasons": [";".join(r) for r in selected["reasons"]]}
    ).to_csv(args.out_dir / "review_set.csv", index=False)
    print(f"review set: {len(selected)} clips")
    for _, item in selected.iterrows():
        print(f"  {item['ref']}  {';'.join(item['reasons'])}")

    checks: dict[str, dict[str, Any]] = {}
    for _, item in selected.iterrows():
        ref, clip = item["ref"], item["clip_id"]
        parquet = args.landmarks_dir / f"{clip}.parquet"
        try:
            df, meta = hl.read_landmarks(parquet)
            checks[ref] = automatic_checks(df)
            if args.no_sheets and args.no_gifs:
                continue
            video = og.find_video(ref, args.video_dir)
            if video is None:
                raise TriageError("video file not found")
            if not args.no_sheets:
                cap = og.open_video(str(video))
                try:
                    sheet = make_sheet(df, meta, cap, ref, checks[ref]["_jump_frame"])
                finally:
                    cap.release()
                sheet.save(args.out_dir / f"{ref}_sheet.png")
            if not args.no_gifs and wants_gif(item["reasons"]):
                cap = og.open_video(str(video))
                try:
                    frames, delays, _ = og.build_overlay(
                        df, meta, cap, ref, start_s=gif_window(df), duration_s=GIF_SECONDS,
                        width=GIF_WIDTH, fps=GIF_FPS,
                    )  # fmt: skip
                finally:
                    cap.release()
                og.save_gif(frames, delays, args.out_dir / f"{ref}_overlay.gif")
        except (
            TriageError,
            og.OverlayError,
            hl.HandLandmarksError,
            pa.ArrowException,
            OSError,
        ) as err:
            print(f"FAILED: clip {ref}: {err}", file=sys.stderr)
            return 1

    if args.alt_conf is not None:
        low = [
            r
            for r, f in zip(manifest["ref"], manifest["flags"], strict=True)
            if isinstance(f, str) and "low_detection" in f.split(";")
        ]
        if low and alt_confidence_run(low, manifest, args) != 0:
            return 1

    table = review_table(selected, manifest, checks)
    table.to_csv(args.out_dir / "triage_table.csv", index=False)
    print("\n" + table.drop(columns=["visual_verdict"]).to_string(index=False))
    print(f"\nwrote review_set.csv, triage_table.csv and sheets/GIFs to {args.out_dir.name}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
