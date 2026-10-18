"""GAIT-20 addendum: how much does the severity model lose when the test joints are noisy?

Run from the repository root after ``train_xgboost.py``::

    python models/gait/src/noise_check.py

Each test walk's joints are jittered by multiples (0.5, 1, 2, 4) of the white-noise level measured
on the MediaPipe traces of ``check_mediapipe_mapping.py``, the features are re-extracted with the
real ``extract_features``, and the 25 cross-validation models (fitted on clean walks) are scored
on them. Scores are taken on the walks that still pass the cycle check at every level. Only the
size of the jitter comes from video; its timing, missing frames and pose errors are not modelled.

Writes CSVs and ``results.json`` to ``artifacts/models/noise/`` (git-ignored).
Exit code 0 = ok, 1 = level 0x differs from the feature table or a shuffled-label control failed,
2 = an input is missing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import evaluate_cohorts as ec
import numpy as np
import pandas as pd
import train_xgboost as tx
import xgboost
from carepd_format import load_config
from carepd_loader import DEFAULT_CONFIG, DEFAULT_DATA, Walk, load_all
from gait_extract import FEATURE_NAMES, extract_features
from make_folds import fingerprint, fold_problems
from mediapipe_mapping import N_JOINTS, map_landmarks

TRACE_DIR = tx.GAIT_DIR / "artifacts" / "mediapipe"
DEFAULT_OUT = tx.GAIT_DIR / "artifacts" / "models" / "noise"
MULTIPLES = (0.0, 0.5, 1.0, 2.0, 4.0)
MODELS = ("xgboost", "logistic", tx.CONTROL)
REFERENCE_LEVEL = 0.0
FEATURE_TOLERANCE = 1e-6  # the feature table has 6 decimals
MIN_RUN = 20  # a joint needs this many usable fourth differences in a trace
FOURTH_DIFFERENCE_VARIANCE = 70.0  # 1 + 16 + 36 + 16 + 1: the variance of d4 for unit white noise
MAD_TO_SD = 1.4826


def level_name(multiple: float) -> str:
    return f"{multiple:g}x"


def model_key(model: str, multiple: float) -> str:
    return f"{model}@{level_name(multiple)}"


# --------------------------------------------------------------------------- the noise
def white_noise_sd(x: np.ndarray) -> float:
    """White-noise standard deviation of one coordinate over time (NaN where unusable).

    Fourth differences of white noise have variance 70 sigma^2, while a smooth walking signal
    (a few hertz at 25-50 fps) almost cancels; the median absolute deviation ignores the outliers
    that pose estimators produce. Only runs of five finite frames are used.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("expected one coordinate over time")
    d4 = x[4:] - 4 * x[3:-1] + 6 * x[2:-2] - 4 * x[1:-3] + x[:-4]
    d4 = d4[np.isfinite(d4)]
    if len(d4) < MIN_RUN:
        return float("nan")
    centred = np.abs(d4 - np.median(d4))
    return float(MAD_TO_SD * np.median(centred) / np.sqrt(FOURTH_DIFFERENCE_VARIANCE))


def trace_jitter(world: np.ndarray) -> np.ndarray:
    """``(17,)`` jitter (metres) of the h36m joints mapped from one MediaPipe world trace."""
    joints = map_landmarks(world)
    out = np.full(N_JOINTS, np.nan)
    for j in range(N_JOINTS):
        sds = [white_noise_sd(joints[:, j, axis]) for axis in range(3)]
        if not np.all(np.isnan(sds)):
            out[j] = np.nanmedian(sds)
    return out


def jitter_profile(worlds: Sequence[np.ndarray]) -> np.ndarray:
    """``(17,)`` median over traces of each joint's jitter, in metres (1x of the check)."""
    if not worlds:
        raise ValueError("no MediaPipe traces to measure")
    per_trace = np.vstack([trace_jitter(w) for w in worlds])
    if np.isnan(per_trace).all(axis=0).any():
        raise ValueError("a joint has no usable frames in any trace")
    return np.asarray(np.nanmedian(per_trace, axis=0))


def read_worlds(trace_dir: Path) -> list[np.ndarray]:
    files = sorted(trace_dir.glob("trace_*.npz"))
    worlds = []
    for f in files:
        with np.load(f) as z:
            worlds.append(np.asarray(z["world"], dtype=float))
    return worlds


def jitter_walk(joints: np.ndarray, sigma: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """``joints`` plus white Gaussian noise, ``sigma[j]`` metres on every axis of joint j."""
    joints = np.asarray(joints, dtype=float)
    if joints.ndim != 3 or joints.shape[1:] != (N_JOINTS, 3):
        raise ValueError(f"expected (frames, {N_JOINTS}, 3) joints, got {joints.shape}")
    if sigma.shape != (N_JOINTS,) or (sigma < 0).any():
        raise ValueError("sigma must hold one non-negative value per joint")
    return joints + rng.normal(size=joints.shape) * sigma[None, :, None]


# --------------------------------------------------------------------------- the features
@dataclass
class NoisyFeatures:
    """The test inputs at one noise level, row for row with the training table."""

    multiple: float
    table: pd.DataFrame  # table with the noisy inputs; rejected rows keep their clean inputs
    passed: np.ndarray  # (rows,) bool: the noisy walk still gave the eight features
    reasons: Counter[str]
    max_gap: float  # largest |noisy - table| over the rows that passed


def noisy_features(
    walks_by_tag: dict[str, Walk],
    table: pd.DataFrame,
    profile: np.ndarray,
    multiple: float,
    level_index: int,
    noise_seed: int,
) -> NoisyFeatures:
    """Re-extract the eight inputs of every table walk from its jittered joints."""
    missing = [t for t in table["walk"] if t not in walks_by_tag]
    if missing:
        raise ValueError(f"{len(missing)} table walks were not loaded from the data")
    sigma = profile * multiple
    values = table[list(FEATURE_NAMES)].to_numpy(dtype=float).copy()
    passed = np.zeros(len(table), dtype=bool)
    reasons: Counter[str] = Counter()
    gap = 0.0
    for i, tag in enumerate(table["walk"]):
        walk = walks_by_tag[tag]
        joints = walk.joints
        if multiple > 0:
            rng = np.random.default_rng([noise_seed, level_index, zlib.crc32(tag.encode())])
            joints = jitter_walk(joints, sigma, rng)
        r = extract_features(joints, float(walk.fps))
        if not r.ok:
            reasons[r.reason] += 1
            continue
        new = np.array([r.features[n] for n in FEATURE_NAMES])
        gap = max(gap, float(np.abs(new - values[i]).max()))
        values[i] = new
        passed[i] = True
    noisy = table.copy()
    noisy[list(FEATURE_NAMES)] = values
    return NoisyFeatures(multiple, noisy, passed, reasons, gap)


# --------------------------------------------------------------------------- the scores
def rejection_table(table: pd.DataFrame, levels: Sequence[NoisyFeatures]) -> pd.DataFrame:
    rows = []
    for lv in levels:
        lost = table.loc[~lv.passed]
        for (cohort, severity), part in lost.groupby(["cohort", "severity"]):
            rows.append(
                {
                    "level": level_name(lv.multiple),
                    "cohort": cohort,
                    "severity": int(severity),  # type: ignore[call-overload]
                    "rejected": len(part),
                }
            )
    return pd.DataFrame(rows, columns=["level", "cohort", "severity", "rejected"])


def feature_shift(
    table: pd.DataFrame, levels: Sequence[NoisyFeatures], common: np.ndarray
) -> pd.DataFrame:
    """Median |noisy - clean| of every input per level, in standard deviations of the input.

    Taken over the common set; the 0x level is all zeros (it is the check that re-extraction
    reproduces the table).
    """
    clean = table[list(FEATURE_NAMES)].to_numpy(dtype=float)[common]
    spread = table[list(FEATURE_NAMES)].to_numpy(dtype=float).std(axis=0, ddof=1)
    rows: list[dict[str, Any]] = []
    for lv in levels:
        noisy = lv.table[list(FEATURE_NAMES)].to_numpy(dtype=float)[common]
        shift = np.median(np.abs(noisy - clean), axis=0) / spread
        rows.extend(
            {"level": level_name(lv.multiple), "feature": f, "median_shift_sd": float(v)}
            for f, v in zip(FEATURE_NAMES, shift, strict=True)
        )
    return pd.DataFrame(rows, columns=["level", "feature", "median_shift_sd"])


def noise_analysis(
    table: pd.DataFrame,
    assignments: pd.DataFrame,
    levels: Sequence[NoisyFeatures],
    n_boot: int,
    params: dict[str, Any] | None = None,
    models: Sequence[str] = MODELS,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Scores per level on the common set, and paired differences from the 0x level.

    Returns ``(scores, differences, common_mask)``.
    """
    if not levels or levels[0].multiple != REFERENCE_LEVEL:
        raise ValueError("the first level must be the 0x reference")
    common = np.logical_and.reduce([lv.passed for lv in levels])
    if common.sum() == 0:
        raise ValueError("no walk passes at every noise level")
    y = table["severity"].to_numpy(dtype=int)[common]
    subjects = table["subject_key"].to_numpy()[common]
    probas: dict[str, list[np.ndarray]] = {}
    for lv in levels:
        cv = tx.cross_validate(table, assignments, models, params=params, test_table=lv.table)
        for m in models:
            probas[model_key(m, lv.multiple)] = [p[common] for p in cv.proba[m]]
    scores, draws = ec.score_models(y, subjects, probas, n_boot)
    pairs = [
        (model_key(m, lv.multiple), model_key(m, REFERENCE_LEVEL))
        for m in models
        for lv in levels[1:]
    ]
    diffs = ec.difference_table(scores, draws, pairs)
    keys = scores["model"].str.split("@", expand=True)
    scores = scores.assign(model=keys[0], level=keys[1])
    scores = scores[["level", *[c for c in scores.columns if c != "level"]]]
    return scores, diffs, common


def control_problems_by_level(scores: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for _, r in scores[scores["model"] == tx.CONTROL].iterrows():
        out += ec.control_problems(f"noise {r['level']}", r["bal_acc"], r["auc"])
    return out


# --------------------------------------------------------------------------- the report
def report(
    profile: np.ndarray,
    joint_names: Sequence[str],
    levels: Sequence[NoisyFeatures],
    n_walks: int,
    n_common: int,
    scores: pd.DataFrame,
    diffs: pd.DataFrame,
    shift: pd.DataFrame,
) -> None:
    print("Jitter measured on MediaPipe output (1x), millimetres, per joint:")
    print(
        "  " + ", ".join(f"{n} {s * 1000:.1f}" for n, s in zip(joint_names, profile, strict=True))
    )
    print(
        "\nHow many walks still pass the cycle check, and the walks the scores are taken on "
        f"({n_common:,} of {n_walks:,} pass at every level):"
    )
    for lv in levels:
        lost = int((~lv.passed).sum())
        top = ", ".join(f"{k} ({v})" for k, v in lv.reasons.most_common(2)) or "-"
        print(f"  {level_name(lv.multiple):>5}: {lost:>4} rejected ({lost / n_walks:.1%})  {top}")
    print("\nMedian change of each input, in standard deviations of that input across walks:")
    print(f"{'':<26}" + "".join(f"{level_name(lv.multiple):>8}" for lv in levels[1:]))
    for f in FEATURE_NAMES:
        vals = shift[shift["feature"] == f].set_index("level")["median_shift_sd"]
        print(f"{f:<26}" + "".join(f"{vals[level_name(lv.multiple)]:>8.2f}" for lv in levels[1:]))
    by_pair = diffs.set_index(["model", "versus", "metric"])
    for model in MODELS:
        if model == tx.CONTROL:
            continue
        print(f"\n{model}: scores on the common set; change from 0x with its 95% interval")
        print(f"{'':>6}{'balanced accuracy':>22}{'change':>26}{'AUC':>22}{'macro-F1':>22}")
        for lv in levels:
            row = scores[(scores["model"] == model) & (scores["level"] == level_name(lv.multiple))]
            r = row.iloc[0]
            change = ""
            if lv.multiple != REFERENCE_LEVEL:
                d = by_pair.loc[
                    (model_key(model, lv.multiple), model_key(model, REFERENCE_LEVEL), "bal_acc")
                ]
                change = ec._delta(float(d["diff"]), float(d["lo"]), float(d["hi"]))
            print(
                f"{level_name(lv.multiple):>6}{ec._ci(r, 'bal_acc'):>22}{change:>26}"
                f"{ec._ci(r, 'auc'):>22}{ec._ci(r, 'macro_f1'):>22}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Joint-noise check of the severity model (GAIT-20)"
    )
    parser.add_argument("--table", type=Path, default=tx.DEFAULT_TABLE)
    parser.add_argument("--folds", type=Path, default=tx.DEFAULT_FOLDS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap", type=int, default=tx.N_BOOT, help="bootstrap draws")
    parser.add_argument("--noise-seed", type=int, default=0, help="seed of the noise draw")
    args = parser.parse_args(argv)

    for path, hint in (
        (args.table, "build_feature_table.py"),
        (args.folds, "make_folds.py"),
        (args.config, "the repository checkout"),
    ):
        if not path.is_file():
            print(f"MISSING: {path}\nRun {hint} first.")
            return 2
    if not args.data_dir.is_dir() or not any(args.trace_dir.glob("trace_*.npz")):
        print(
            f"MISSING: CARE-PD data in {args.data_dir} or MediaPipe traces in {args.trace_dir}\n"
            "Run check_mediapipe_mapping.py for the traces."
        )
        return 2
    started = time.time()
    table = tx.read_training_table(args.table)
    assignments = tx.read_assignments(args.folds)
    problems = fold_problems(table, assignments)
    if problems:
        for p in problems:
            print(f"  -> {p}")
        print("The folds are not valid for this table. Rebuild them with make_folds.py.")
        return 1
    cfg = load_config(args.config)
    walks, _ = load_all(args.data_dir, cfg)
    by_tag = {w.tag: w for w in walks}
    profile = jitter_profile(read_worlds(args.trace_dir))
    print(
        f"{len(table)} walks, {table['subject_key'].nunique()} subjects; folds "
        f"{fingerprint(assignments)}; table {tx.table_fingerprint(args.table)}; "
        f"xgboost {xgboost.__version__}; noise seed {args.noise_seed}.\n"
    )

    levels = []
    for i, multiple in enumerate(MULTIPLES):
        levels.append(noisy_features(by_tag, table, profile, multiple, i, args.noise_seed))
        print(f"  features at {level_name(multiple)} done ({time.time() - started:.0f} s)")
    failed: list[str] = []
    zero = levels[0]
    if (~zero.passed).any() or zero.max_gap > FEATURE_TOLERANCE:
        failed.append(
            f"at 0x the re-extracted features differ from the table (largest gap {zero.max_gap:.1e}, "
            f"{int((~zero.passed).sum())} walks not extracted): rebuild the table with "
            "build_feature_table.py"
        )
        for p in failed:
            print(f"  -> {p}")
        return 1
    print()
    scores, diffs, common = noise_analysis(table, assignments, levels, args.bootstrap)
    joint_names = cfg.skeleton.joint_order
    shift = feature_shift(table, levels, common)
    report(profile, joint_names, levels, len(table), int(common.sum()), scores, diffs, shift)
    failed = control_problems_by_level(scores)
    print(
        "\nShuffled-label controls at every noise level: "
        + ("FAILED" if failed else "all at chance, as they should be.")
    )
    for p in failed:
        print(f"  -> {p}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.out_dir / "noise_scores.csv", index=False, float_format="%.6f")
    diffs.to_csv(args.out_dir / "noise_differences.csv", index=False, float_format="%.6f")
    rejection_table(table, levels).to_csv(args.out_dir / "noise_rejections.csv", index=False)
    shift.to_csv(args.out_dir / "noise_feature_shift.csv", index=False, float_format="%.6f")
    pd.DataFrame({"joint": joint_names, "sigma_mm": profile * 1000}).to_csv(
        args.out_dir / "noise_profile.csv", index=False, float_format="%.4f"
    )
    config = {
        "multiples": list(MULTIPLES),
        "noise_seed": args.noise_seed,
        "bootstrap_draws": args.bootstrap,
        "xgboost_params": tx.XGB_PARAMS,
        "xgboost_version": xgboost.__version__,
        "folds_fingerprint": fingerprint(assignments),
        "feature_table": tx.table_fingerprint(args.table),
        "walks": len(table),
        "walks_in_common_set": int(common.sum()),
        "rejected_per_level": {level_name(lv.multiple): int((~lv.passed).sum()) for lv in levels},
        "control_problems": failed,
    }
    (args.out_dir / "results.json").write_text(json.dumps(config, indent=2))
    print(f"\nWrote {args.out_dir} in {time.time() - started:.0f} s.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
