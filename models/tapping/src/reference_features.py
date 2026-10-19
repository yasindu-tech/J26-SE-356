"""Reference features for baseline B2 (task FT-40): the 27 "classical" features of the HUBU-FIS
reference pipeline, recomputed label-blind from our own landmark files.

    python models/tapping/src/reference_features.py [--landmarks data/interim/landmarks]

Definitions follow github.com/arelraptor/UBU-PD-FT-Assessment (Apache-2.0; lib/video_processing.py
and lib/extract_features.py, main branch read unpinned). The code here is a reimplementation, not a
copy; the reference was read, never executed. Its TsFresh and FI+TsFresh sets are NOT used: both
select features with the UPDRS labels on the whole dataset, and tsfresh is not installed.

Per clip, as in the reference: frames without a detected hand are dropped (not interpolated); the
amplitude is |thumb tip - index tip| / |wrist - index tip| from the normalised x, y, z of landmarks
4, 8 and 0, smoothed by a centred 3-sample mean; velocity and acceleration are first differences
over 1/fps with NaN set to 0; fps is the container's nominal frame rate. Our landmarks come from the
VIDEO-mode tracker, so these are the reference definitions on our detections, not its detections.
No label, group or rating is read anywhere. Writes reference_features.csv, reference_roles.csv and
reference_meta.json under data/processed/reference. Exit codes: 0 ok, 1 inputs disagree, 2 missing
input or a path outside its folder.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv_splits as cvs
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from cv_splits import GuardFailure, InputProblem
from hubu_fis import pseudonym
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew

TOOL_VERSION = "reference-features-1"
REPO_ROOT = Path(__file__).resolve().parents[3]
INTERIM_ROOT = REPO_ROOT / "data" / "interim"
REFERENCE_ROOT = REPO_ROOT / "data" / "processed" / "reference"
LANDMARKS_DEFAULT = INTERIM_ROOT / "landmarks"
METADATA_KEY = b"tapping.hand_landmarks"
SOURCE = {
    "repository": "github.com/arelraptor/UBU-PD-FT-Assessment",
    "licence": "Apache-2.0",
    "revision": "unpinned: main branch files read once, not executed",
    "files": "lib/video_processing.py, lib/extract_features.py",
    "used": "classical features only (27)",
    "not_used": "tsfresh and fi_tsfresh: feature selection fitted on the diagnosis labels, tsfresh not installed",
}
REFERENCE_REJECTION_SHARE = (
    0.90  # the reference rejects a video at or below this share of detected frames
)
LANDMARK_IDS = {"wrist": 0, "thumb": 4, "index": 8}
LANDMARK_COLUMNS = ["detected"] + [f"{a}{k:02d}" for k in LANDMARK_IDS.values() for a in "xyz"]

AGGREGATES = (
    ("smoothed_amplitude", ("mean", "std", "max")),
    ("velocity", ("mean", "max", "std")),
    ("acceleration", ("mean", "max", "std")),
    ("distance_ang", ("mean", "std")),
    ("frequency", ("mean", "std", "count")),
)
REFERENCE_COLUMNS = [
    *[f"ref_{series}_{stat}" for series, stats in AGGREGATES for stat in stats],
    "ref_interval_std", "ref_interval_cv", "ref_amplitude_slope", "ref_frequency_drop",
    "ref_acceleration_increase", "ref_skew_amplitude", "ref_kurtosis_amplitude",
    "ref_skew_velocity", "ref_kurtosis_velocity", "ref_fft_dominant_freq_amplitude",
    "ref_fft_power_amplitude", "ref_fft_dominant_freq_velocity", "ref_fft_power_velocity",
]  # fmt: skip
FEATURE_NOTES = {
    "mean": "mean over the kept frames", "std": "sample std (ddof=1) over the kept frames",
    "max": "maximum over the kept frames",
    "count": "number of non-NaN values (peaks after the first)",
}  # fmt: skip
OUT_FEATURES, OUT_ROLES, OUT_META = (
    "reference_features.csv",
    "reference_roles.csv",
    "reference_meta.json",
)
PINNED_ROLES_SHA8 = "3373dae7"  # the registered feature list; a test fails if it ever changes
PINNED_FEATURES_SHA8 = "17fb7bbf"


# --------------------------------------------------------------------------- the reference series


def angle_degrees(vector_1: np.ndarray, vector_2: np.ndarray) -> np.ndarray:
    """Angle between two vectors per row, as the reference's angle_of_vectors (no clipping)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        u1 = vector_1 / np.linalg.norm(vector_1, axis=1, keepdims=True)
        u2 = vector_2 / np.linalg.norm(vector_2, axis=1, keepdims=True)
        return np.degrees(np.arccos((u1 * u2).sum(axis=1)))


def reference_time_series(landmarks: pd.DataFrame, fps: float) -> pd.DataFrame:
    """DISTANCE_ANG, SMOOTHED_AMPLITUDE, VELOCITY, ACCELERATION and FREQUENCY of one clip."""
    kept = landmarks[landmarks["detected"].astype(bool)]
    if len(kept) < 3:
        raise GuardFailure("fewer than 3 frames with a detected hand")
    point = {
        name: kept[[f"{a}{k:02d}" for a in "xyz"]].to_numpy(dtype=float)
        for name, k in LANDMARK_IDS.items()
    }
    wrist, thumb, index = point["wrist"], point["thumb"], point["index"]
    angle = angle_degrees(wrist - index, wrist - thumb)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalised = np.linalg.norm(thumb - index, axis=1) / np.linalg.norm(wrist - index, axis=1)
    smoothed = pd.Series(normalised).rolling(window=3, center=True).mean().bfill().ffill()
    delta_t = 1 / fps
    velocity = (smoothed.diff() / delta_t).fillna(0)
    acceleration = (velocity.diff() / delta_t).fillna(0)
    peaks, _ = find_peaks(smoothed, distance=fps // 3)
    frequency = pd.Series(np.nan, index=smoothed.index)
    if len(peaks) >= 2:
        frequency.iloc[peaks[1:]] = 1 / np.diff(peaks * (1 / fps))
    return pd.DataFrame({
        "DISTANCE_ANG": angle / 90, "SMOOTHED_AMPLITUDE": smoothed, "VELOCITY": velocity,
        "ACCELERATION": acceleration, "FREQUENCY": frequency,
    })  # fmt: skip


def fft_features(signal_values: np.ndarray) -> tuple[float, float]:
    """(dominant frequency in cycles per sample, power) over the first n//2 bins, DC included."""
    n = len(signal_values)
    magnitude = np.abs(np.fft.fft(signal_values))[: n // 2]
    freqs = np.fft.fftfreq(n)[: n // 2]
    return float(freqs[int(np.argmax(magnitude))]), float(np.sum(magnitude**2))


def half_difference(series: pd.Series) -> float:
    """Mean of the first n//2 values minus mean of the last n//2 (the reference's sign)."""
    half = len(series) // 2
    return float(series.head(half).mean() - series.tail(half).mean())


def reference_features(ts: pd.DataFrame, fps: float) -> dict[str, float]:
    """The 27 classical features of one clip, in REFERENCE_COLUMNS order."""
    out: dict[str, float] = {}
    for series, stats in AGGREGATES:
        values = ts[series.upper()]
        for stat in stats:
            out[f"ref_{series}_{stat}"] = float(getattr(values, stat)())
    amplitude = ts["SMOOTHED_AMPLITUDE"]
    peaks, _ = find_peaks(amplitude, distance=fps // 3)
    periods = np.diff(peaks * (1 / fps))
    if len(periods) >= 2:
        std = float(np.std(periods))
        mean = float(np.mean(periods))
        out["ref_interval_std"], out["ref_interval_cv"] = (
            std,
            (std / mean if mean != 0 else math.nan),
        )
    else:
        out["ref_interval_std"] = out["ref_interval_cv"] = math.nan
    out["ref_amplitude_slope"] = float(np.polyfit(np.arange(len(amplitude)), amplitude, 1)[0])
    out["ref_frequency_drop"] = half_difference(ts["FREQUENCY"])
    out["ref_acceleration_increase"] = half_difference(ts["ACCELERATION"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # constant signals: scipy returns NaN with a warning
        out["ref_skew_amplitude"] = float(skew(amplitude))
        out["ref_kurtosis_amplitude"] = float(kurtosis(amplitude))
        out["ref_skew_velocity"] = float(skew(ts["VELOCITY"]))
        out["ref_kurtosis_velocity"] = float(kurtosis(ts["VELOCITY"]))
    out["ref_fft_dominant_freq_amplitude"], out["ref_fft_power_amplitude"] = fft_features(
        amplitude.to_numpy()
    )
    out["ref_fft_dominant_freq_velocity"], out["ref_fft_power_velocity"] = fft_features(
        ts["VELOCITY"].to_numpy()
    )
    return {c: out[c] for c in REFERENCE_COLUMNS}


# --------------------------------------------------------------------------- roles and table


def roles_table() -> pd.DataFrame:
    notes = {}
    for series, stats in AGGREGATES:
        for stat in stats:
            notes[f"ref_{series}_{stat}"] = f"{FEATURE_NOTES[stat]} of {series.upper()}"
    notes.update({
        "ref_interval_std": "population std of peak-to-peak intervals (needs 2 intervals)",
        "ref_interval_cv": "interval std over interval mean (needs 2 intervals)",
        "ref_amplitude_slope": "np.polyfit slope of SMOOTHED_AMPLITUDE against the kept-frame index",
        "ref_frequency_drop": "mean FREQUENCY first half minus second half (the reference's sign)",
        "ref_acceleration_increase": "mean ACCELERATION first half minus second half (the reference's sign)",
        "ref_skew_amplitude": "scipy skew of SMOOTHED_AMPLITUDE", "ref_kurtosis_amplitude": "scipy Fisher kurtosis of SMOOTHED_AMPLITUDE",
        "ref_skew_velocity": "scipy skew of VELOCITY", "ref_kurtosis_velocity": "scipy Fisher kurtosis of VELOCITY",
        "ref_fft_dominant_freq_amplitude": "frequency (cycles per sample) of the largest of the first n//2 FFT bins of SMOOTHED_AMPLITUDE",
        "ref_fft_power_amplitude": "sum of squared magnitudes of the first n//2 FFT bins of SMOOTHED_AMPLITUDE",
        "ref_fft_dominant_freq_velocity": "as above, for VELOCITY", "ref_fft_power_velocity": "as above, for VELOCITY",
    })  # fmt: skip
    rows = [
        ("ref", "id", "hashed clip id"),
        ("participant", "group_key", "hashed participant; folds are grouped on it"),
    ]
    rows += [(c, "reference_feature", notes[c]) for c in REFERENCE_COLUMNS]
    return pd.DataFrame(rows, columns=["column", "role", "note"])


def roles_sha8() -> str:
    return cvs.roles_sha8_from_file(roles_table())


def landmark_files(directory: Path) -> dict[str, Path]:
    """Hashed ref -> parquet path. Names are raw clip IDs, so nothing here is ever printed."""
    found: dict[str, Path] = {}
    for name in sorted(glob.glob(str(directory / "*.parquet"))):
        ref = pseudonym(Path(name).stem)
        if ref in found:
            raise GuardFailure("two landmark files map to the same hashed ref")
        found[ref] = Path(name)
    return found


def clip_inputs(path: Path) -> tuple[pd.DataFrame, float]:
    landmarks = pq.read_table(path, columns=LANDMARK_COLUMNS).to_pandas()
    meta = json.loads((pq.read_schema(path).metadata or {}).get(METADATA_KEY, b"{}"))
    fps = float(meta.get("fps", math.nan))
    if not math.isfinite(fps) or fps < 3:
        raise GuardFailure("a landmark file has no usable nominal fps")
    return landmarks, fps


def build_table(features_path: Path, landmarks_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not features_path.is_file():
        raise InputProblem("missing input: features file")
    if not landmarks_dir.is_dir():
        raise InputProblem("missing input: landmarks folder")
    clips = pd.read_csv(
        features_path, dtype=str, usecols=["ref", "participant"], keep_default_na=False
    )
    if clips["ref"].duplicated().any() or (clips["participant"] == "").any():
        raise GuardFailure("features: ref must be unique and every clip needs a participant")
    clips = clips.sort_values("ref").reset_index(drop=True)
    files = landmark_files(landmarks_dir)
    missing = sum(ref not in files for ref in clips["ref"])
    if missing:
        raise GuardFailure(f"{missing} retained clip(s) have no landmark file")
    rows, shares = [], []
    for ref in clips["ref"]:
        landmarks, fps = clip_inputs(files[ref])
        shares.append(float(landmarks["detected"].astype(bool).mean()))
        rows.append(reference_features(reference_time_series(landmarks, fps), fps))
    table = pd.concat([clips, pd.DataFrame(rows, columns=REFERENCE_COLUMNS)], axis=1)
    nan_counts = {c: int(table[c].isna().sum()) for c in REFERENCE_COLUMNS}
    stats = {
        "n_clips": len(table), "n_participants": int(table["participant"].nunique()),
        "n_landmark_files_not_used": len(files) - len(clips),
        "n_clips_at_or_below_reference_detection_rule": int(sum(s <= REFERENCE_REJECTION_SHARE for s in shares)),
        "nan_counts": nan_counts, "n_features_with_nan": sum(v > 0 for v in nan_counts.values()),
    }  # fmt: skip
    return table, stats


def _atomic(final: Path, data: bytes) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f".{final.name}.tmp{os.getpid()}")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, final)
    finally:
        if tmp.exists():
            tmp.unlink()


def run(features_path: Path, landmarks_dir: Path, out_dir: Path) -> dict[str, Any]:
    table, stats = build_table(features_path, landmarks_dir)
    roles = roles_table()
    features_bytes = table.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode()
    meta = {
        "tool": TOOL_VERSION, "source": SOURCE, "reference_roles_sha8": cvs.roles_sha8_from_file(roles),
        "columns": REFERENCE_COLUMNS, "features_sha8": cvs.sha8_of(json.dumps(REFERENCE_COLUMNS)),
        "table_sha8": cvs.sha8_of(features_bytes), "inputs_features_sha8": cvs.sha8_of(features_path.read_bytes()),
        "rejection_rule_share": REFERENCE_REJECTION_SHARE, "labels_used": [], "group_exported": False,
        **stats,
    }  # fmt: skip
    _atomic(out_dir / OUT_FEATURES, features_bytes)
    _atomic(out_dir / OUT_ROLES, roles.to_csv(index=False, lineterminator="\n").encode())
    _atomic(out_dir / OUT_META, (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode())
    return meta


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--features", type=Path, default=None)
    ap.add_argument("--landmarks", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    features = Path(args.features or cvs.PROCESSED_ROOT / "features.csv")
    landmarks = Path(args.landmarks or LANDMARKS_DEFAULT)
    out_dir = Path(args.out_dir or REFERENCE_ROOT)
    for name, path, root in (
        ("--out-dir", out_dir, REFERENCE_ROOT),
        ("--landmarks", landmarks, INTERIM_ROOT),
    ):
        if not cvs.is_inside(path, root):
            print(f"error: {name} must be inside {root.parent.name}/{root.name}", file=sys.stderr)
            return 2
    try:
        meta = run(features, landmarks, out_dir)
    except InputProblem as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except GuardFailure as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    print(
        f"wrote {OUT_FEATURES}: {meta['n_clips']} clips, {len(REFERENCE_COLUMNS)} features, "
        f"{meta['n_features_with_nan']} with NaN; reference_roles_sha8 {meta['reference_roles_sha8']}; "
        f"features_sha8 {meta['features_sha8']}; clips at or below the reference's detection rule "
        f"{meta['n_clips_at_or_below_reference_detection_rule']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
