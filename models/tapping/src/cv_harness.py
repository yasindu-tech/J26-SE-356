"""Nested-CV harness (task FT-32, part 2): models, inner tuning, pooled metrics, permutation control.

    python models/tapping/src/cv_harness.py run --model lgbm --feature-set A [--show-metrics]
    python models/tapping/src/cv_harness.py permute --model lgbm --feature-set A --draws 20

Splits come only from <splits-dir>/<target>/splits.csv (cv_splits.py); inner folds from
cv_splits.inner_folds. Everything below is pre-registered and fixed as constants: the grids
(simplest first, ties go to the first), inner scoring (pooled inner out-of-fold AUROC), the 0.5
threshold, class_weight balanced, no resampling, no early stopping, no imputation. The rating is
read only as the target. Default prints are counts and hashes; metrics print only with
--show-metrics. Exit codes: 0 ok, 1 guard failure or the permutation control failed, 2 missing
input or a path outside data/processed/cv.
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv_splits as cvs
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from cv_splits import (
    N_OUTER,
    SET_A_PRIMARY,
    SET_B_ADDED,
    GuardFailure,
    InputProblem,
    LeakageError,
    assert_disjoint,
    canonical,
    inner_folds,
    sha8_of,
    target_labels,
    validate_inputs,
)
from lightgbm import LGBMClassifier
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TOOL_VERSION = "cv-harness-0.1.0"
THRESHOLD = 0.5  # fixed, never tuned
LGBM_SEED = 20260
TARGET_STATUS = "pending_default_ge2"
PERMUTATION_MAX_MEAN_AUROC = 0.55
PPV_PREVALENCES = (0.01, 0.02, 0.05)
MODELS = ("b0", "b1", "lgbm")
FEATURE_SETS: dict[str, list[str]] = {
    "A": list(SET_A_PRIMARY),
    "B": list(SET_A_PRIMARY) + list(SET_B_ADDED),
}
LGBM_FIXED: dict[str, Any] = {
    "learning_rate": 0.05,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "class_weight": "balanced",
    "deterministic": True,
    "force_row_wise": True,
    "n_jobs": 1,
    "verbose": -1,
}
B1_FIXED: dict[str, Any] = {
    "class_weight": "balanced",
    "solver": "lbfgs",
    "max_iter": 2000,
}  # penalty is l2, the default (the explicit argument is deprecated in scikit-learn 1.8)


def real_grids() -> dict[str, list[dict[str, Any]]]:
    """The pre-registered grids, simplest first."""
    return {
        "b0": [{}],
        "b1": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
        "lgbm": [
            {"num_leaves": nl, "n_estimators": ne, "min_child_samples": mc}
            for nl, ne, mc in itertools.product((3, 5, 7), (100, 200, 300), (20, 15, 10))
        ],
    }


GRIDS = real_grids()  # tests may replace entries for speed; the real grids are pinned by a test
RUN_COLUMNS = [
    "model", "feature_set", "target", "threshold", "repeat", "seed", "auroc", "auroc_fold_mean",
    "auroc_fold_sd", "balanced_accuracy", "sensitivity", "specificity", "mcc", "n_pred_pos",
    "ppv_at_1pct", "ppv_at_2pct", "ppv_at_5pct", "n_clips", "n_pos", "n_neg", "n_features",
    "features_sha8", "splits_sha8", "inner_sha8", "grid_sha8",
]  # fmt: skip
SHOWN_METRICS = ["auroc", "balanced_accuracy", "sensitivity", "specificity", "mcc", "n_pred_pos"]


# --------------------------------------------------------------------------- models


class PrevalenceClassifier:
    """b0: a constant probability equal to the prevalence of the training labels."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> PrevalenceClassifier:
        self.prevalence_ = float(np.mean(y))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = np.full(len(X), self.prevalence_)
        return np.column_stack([1.0 - p, p])


def make_estimator(model: str, params: dict[str, Any], seed: int = LGBM_SEED) -> Any:
    if model == "b0":
        return PrevalenceClassifier()
    if model == "b1":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=params["C"], **B1_FIXED)),
            ]
        )
    if model == "lgbm":
        return LGBMClassifier(**LGBM_FIXED, random_state=seed, **params)
    raise GuardFailure(f"unknown model {model!r}")


def grid_for(model: str) -> list[dict[str, Any]]:
    return [dict(p) for p in GRIDS[model]]


def grid_sha8(model: str, grid: list[dict[str, Any]]) -> str:
    return sha8_of(json.dumps({"model": model, "grid": grid}, sort_keys=True))


def design_matrix(frame: pd.DataFrame, feature_set: str) -> tuple[np.ndarray, list[str]]:
    """Model input: the registry columns of the set as float64; NaN stays NaN."""
    if feature_set not in FEATURE_SETS:
        raise GuardFailure(f"unknown feature set {feature_set!r}")
    cols = list(FEATURE_SETS[feature_set])
    validate_inputs(frame, cols)
    return frame[cols].astype("float64").to_numpy(), cols


# --------------------------------------------------------------------------- metrics


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUROC; tied scores count 0.5; NaN when a class is missing."""
    y = np.asarray(y)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ppv(sensitivity: float, specificity: float, prevalence: float) -> float:
    denominator = sensitivity * prevalence + (1 - specificity) * (1 - prevalence)
    if np.isnan(denominator) or denominator == 0:
        return float("nan")
    return float(sensitivity * prevalence / denominator)


def threshold_metrics(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = proba >= THRESHOLD
    pos = np.asarray(y) == 1
    tp = int((pred & pos).sum())
    fn = int((~pred & pos).sum())
    tn = int((~pred & ~pos).sum())
    fp = int((pred & ~pos).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    denominator = float(np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denominator if denominator > 0 else 0.0
    out = {
        "balanced_accuracy": (sens + spec) / 2,
        "sensitivity": sens,
        "specificity": spec,
        "mcc": float(mcc),
        "n_pred_pos": float(tp + fp),
    }
    for p in PPV_PREVALENCES:
        out[f"ppv_at_{round(p * 100)}pct"] = ppv(sens, spec, p)
    return out


def repeat_metrics(y: np.ndarray, proba: np.ndarray, member: np.ndarray) -> dict[str, float]:
    """Metrics of one repeat from its pooled out-of-fold probabilities."""
    folds = np.array([auroc(y[member == f], proba[member == f]) for f in range(N_OUTER)])
    bad = np.isnan(folds).any()
    return {
        "auroc": auroc(y, proba),
        "auroc_fold_mean": float("nan") if bad else float(folds.mean()),
        "auroc_fold_sd": float("nan") if bad else float(folds.std(ddof=1)),
        **threshold_metrics(y, proba),
        "n_clips": float(len(y)),
        "n_pos": float((y == 1).sum()),
        "n_neg": float((y == 0).sum()),
    }


# --------------------------------------------------------------------------- fitting


def tune(
    model: str,
    grid: list[dict[str, Any]],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train: np.ndarray,
    repeat: int,
    outer_fold: int,
) -> tuple[dict[str, Any], float]:
    """Choose a configuration on the outer training part only: pooled inner out-of-fold AUROC."""
    if model == "b0":
        return dict(grid[0]), float("nan")
    folds = inner_folds(train, y, groups, repeat, outer_fold)
    allowed = set(train.tolist())
    for fit, val in folds:
        assert_disjoint(groups[fit], groups[val], f"inner repeat {repeat} outer {outer_fold}")
        if not (set(fit.tolist()) | set(val.tolist())) <= allowed:
            raise LeakageError(f"inner repeat {repeat} outer {outer_fold}: rows outside the part")
    val_all = np.concatenate([val for _, val in folds])
    best_params: dict[str, Any] | None = None
    best = float("-inf")
    for params in grid:
        oof = np.full(len(y), np.nan)
        for fit, val in folds:
            est = make_estimator(model, params)
            est.fit(X[fit], y[fit])
            oof[val] = est.predict_proba(X[val])[:, 1]
        score = auroc(y[val_all], oof[val_all])
        if not np.isnan(score) and score > best:  # strict: ties keep the simplest-first winner
            best, best_params = score, dict(params)
    if best_params is None:
        raise GuardFailure(f"inner repeat {repeat} outer {outer_fold}: no configuration scored")
    return best_params, best


def fit_final(
    model: str, params: dict[str, Any], X: np.ndarray, y: np.ndarray, train: np.ndarray
) -> Any:
    est = make_estimator(model, params)
    est.fit(X[train], y[train])
    return est


def fit_outer_fold(
    model: str,
    grid: list[dict[str, Any]],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train: np.ndarray,
    repeat: int,
    outer_fold: int,
) -> tuple[dict[str, Any], float, Any]:
    """Tune inside the training part, then refit the winner on all of it."""
    params, inner_auc = tune(model, grid, X, y, groups, train, repeat, outer_fold)
    return params, inner_auc, fit_final(model, params, X, y, train)


def refit_outer(
    frame: pd.DataFrame,
    model: str,
    params: dict[str, Any],
    train_positions: np.ndarray,
    feature_set: str = "A",
    target: str = cvs.DEFAULT_TARGET,
    test_positions: np.ndarray | None = None,
) -> Any:
    """Refit one outer fold from its stored best params (for FT-42); positions are ref-sorted.

    Pass the fold's test positions to have the participant-overlap assertion run here too."""
    frame = canonical(frame)
    if test_positions is not None:
        groups = frame["participant"].to_numpy()
        train_ids = groups[np.asarray(train_positions)]
        assert_disjoint(train_ids, groups[np.asarray(test_positions)], "refit_outer")
    X, _ = design_matrix(frame, feature_set)
    y = target_labels(frame, target)
    return fit_final(model, params, X, y, np.asarray(train_positions))


def membership(groups: np.ndarray, splits: pd.DataFrame, repeat: int) -> np.ndarray:
    part = splits[splits["repeat"] == repeat]
    fold_of = dict(zip(part["participant"], part["outer_fold"], strict=True))
    return np.array([fold_of[g] for g in groups])


def run_repeat(
    model: str,
    grid: list[dict[str, Any]],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    member: np.ndarray,
    repeat: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Out-of-fold probabilities of one repeat (each clip predicted exactly once)."""
    proba = np.full(len(y), np.nan)
    count = np.zeros(len(y), dtype=int)
    best_rows: list[dict[str, Any]] = []
    for fold in range(N_OUTER):
        train = np.flatnonzero(member != fold)
        test = np.flatnonzero(member == fold)
        assert_disjoint(groups[train], groups[test], f"outer repeat {repeat} fold {fold}")
        params, inner_auc, est = fit_outer_fold(model, grid, X, y, groups, train, repeat, fold)
        proba[test] = est.predict_proba(X[test])[:, 1]
        count[test] += 1
        best_rows.append(
            {
                "repeat": repeat,
                "outer_fold": fold,
                "params": json.dumps(params, sort_keys=True),
                "inner_auroc": inner_auc,
            }
        )
    if not (count == 1).all() or not np.isfinite(proba).all():
        raise GuardFailure(f"repeat {repeat}: a clip was not predicted exactly once")
    if ((proba < 0) | (proba > 1)).any():
        raise GuardFailure(f"repeat {repeat}: a probability is outside [0, 1]")
    return proba, best_rows


# --------------------------------------------------------------------------- permutation


def permute_labels(frame: pd.DataFrame, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle whole participants' label vectors among participants with the same clip count.

    A vector is ordered by (hand, ref), so hand-specific structure moves with the participant; a
    participant alone in its size class stays where it is."""
    if "hand" not in frame.columns:
        raise GuardFailure("the permutation control needs the hand column to order the clips")
    groups = frame["participant"].to_numpy()
    hands = frame["hand"].to_numpy()
    refs = frame["ref"].to_numpy()
    order: dict[str, list[int]] = {}
    for i in sorted(range(len(frame)), key=lambda k: (hands[k], refs[k])):
        order.setdefault(groups[i], []).append(i)
    by_size: dict[int, list[str]] = {}
    for person in sorted(order):
        by_size.setdefault(len(order[person]), []).append(person)
    out = y.copy()
    for size in sorted(by_size):
        people = by_size[size]
        if len(people) == 1:
            continue
        shuffled = rng.permutation(len(people))
        for i, person in enumerate(people):
            out[order[person]] = y[order[people[shuffled[i]]]]
    return out


def null_auroc(
    model: str,
    grid: list[dict[str, Any]],
    X: np.ndarray,
    y_perm: np.ndarray,
    groups: np.ndarray,
    splits: pd.DataFrame,
    repeats: int,
) -> float:
    values = []
    for repeat in range(repeats):
        proba, _ = run_repeat(
            model, grid, X, y_perm, groups, membership(groups, splits, repeat), repeat
        )
        values.append(auroc(y_perm, proba))
    return float(np.mean(values))


# --------------------------------------------------------------------------- inputs and outputs


def prepare(args: argparse.Namespace) -> SimpleNamespace:
    model, feature_set = args.model, args.feature_set
    if model == "b1" and feature_set != "A":
        raise GuardFailure("b1 is pre-registered on Set A")
    if model == "b0":
        feature_set = "A"  # the prevalence classifier ignores features
    features = Path(args.features or cvs.PROCESSED_ROOT / "features.csv")
    roles = Path(args.roles or cvs.PROCESSED_ROOT / "feature_roles.csv")
    meta = Path(args.meta or cvs.PROCESSED_ROOT / "features_meta.json")
    frame, _ = cvs.load_inputs(features, roles, meta)
    frame = canonical(frame)
    X, cols = design_matrix(frame, feature_set)
    if model == "b1" and np.isnan(X).any():
        raise GuardFailure("b1 does not impute: the Set A matrix must have no NaN")
    target_dir = Path(args.splits_dir or cvs.CV_ROOT) / args.target
    splits_path = target_dir / cvs.OUT_SPLITS
    splits = cvs.load_splits(splits_path, frame)
    splits_meta = json.loads((target_dir / cvs.OUT_META).read_text())
    if splits_meta.get("target") != args.target:
        raise GuardFailure("the splits file was made for another target")
    if splits_meta.get("seed") != args.seed:
        raise GuardFailure("--seed differs from the seed of the splits file")
    if not 1 <= args.repeats <= int(splits_meta["repeats"]):
        raise GuardFailure(f"--repeats must be 1..{splits_meta['repeats']} (repeats in the splits)")
    used = splits[splits["repeat"] < args.repeats].reset_index(drop=True)
    return SimpleNamespace(
        model=model,
        feature_set=feature_set,
        frame=frame,
        X=X,
        cols=cols,
        y=target_labels(frame, args.target),
        groups=frame["participant"].to_numpy(),
        refs=frame["ref"].to_numpy(),
        splits=used,
        splits_meta=splits_meta,
        inputs={
            **cvs.input_hashes(features, roles, meta),
            "splits": cvs.sha8_of(splits_path.read_bytes()),
        },
        roles_sha8=json.loads(meta.read_text())["roles_sha8"],
        grid=grid_for(model),
    )


def _csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode()


def _json(data: dict[str, Any]) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True) + "\n").encode()


def write_all(out_dir: Path, files: dict[str, bytes], force: bool) -> None:
    for name, data in files.items():
        path = out_dir / name
        if path.exists() and path.read_bytes() != data and not force:
            raise GuardFailure(f"refusing to overwrite {name} with different content; use --force")
    for name, data in files.items():
        cvs._atomic(out_dir / name, data)


def read_oof(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"ref": str}, float_precision="round_trip")


def same_folds(meta_a: dict[str, Any], meta_b: dict[str, Any]) -> bool:
    return bool(
        meta_a["splits_sha8"] == meta_b["splits_sha8"]
        and meta_a["inner_sha8"] == meta_b["inner_sha8"]
    )


def run_models(args: argparse.Namespace, out_root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    p = prepare(args)
    inner = cvs.inner_table(p.frame, p.splits, args.target)
    splits_h, inner_h = cvs.splits_sha8(p.splits), cvs.inner_sha8(inner)
    feat_h, grid_h = sha8_of(json.dumps(p.cols)), grid_sha8(p.model, p.grid)
    oof_parts, best_rows, run_rows = [], [], []
    for repeat in range(args.repeats):
        member = membership(p.groups, p.splits, repeat)
        proba, rows = run_repeat(p.model, p.grid, p.X, p.y, p.groups, member, repeat)
        best_rows += rows
        oof_parts.append(
            pd.DataFrame(
                {"ref": p.refs, "repeat": repeat, "outer_fold": member, "y": p.y, "proba": proba}
            )
        )
        run_rows.append(
            {
                "model": p.model, "feature_set": p.feature_set, "target": args.target,
                "threshold": THRESHOLD, "repeat": repeat, "seed": args.seed,
                **repeat_metrics(p.y, proba, member),
                "n_features": len(p.cols), "features_sha8": feat_h, "splits_sha8": splits_h,
                "inner_sha8": inner_h, "grid_sha8": grid_h,
            }
        )  # fmt: skip
    tag = f"{p.model}_{p.feature_set}"
    meta = {
        "tool": TOOL_VERSION,
        "model": p.model,
        "feature_set": p.feature_set,
        "target": args.target,
        "target_status": TARGET_STATUS,
        "seed": args.seed,
        "repeats": args.repeats,
        "columns": p.cols,
        "features_sha8": feat_h,
        "inputs_sha8": p.inputs,
        "roles_sha8": p.roles_sha8,
        "splits_sha8": splits_h,
        "splits_file_rows_sha8": p.splits_meta["splits_sha8"],
        "inner_sha8": inner_h,
        "grid": p.grid,
        "grid_sha8": grid_h,
        "constants": {
            "threshold": THRESHOLD,
            "lgbm_seed": LGBM_SEED,
            "n_outer": N_OUTER,
            "n_inner": cvs.N_INNER,
            "lgbm_fixed": LGBM_FIXED,
            "b1_fixed": {**B1_FIXED, "penalty": "l2 (default)"},
            "inner_scoring": "pooled inner out-of-fold AUROC",
            "tie_break": "first configuration in simplest-first order",
        },
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "rows": len(p.frame),
        "participants": int(pd.Series(p.groups).nunique()),
    }
    files = {
        f"oof_{tag}.csv": _csv(pd.concat(oof_parts, ignore_index=True)),
        f"best_params_{tag}.csv": _csv(pd.DataFrame(best_rows)),
        f"runs_{tag}.csv": _csv(pd.DataFrame(run_rows, columns=RUN_COLUMNS)),
        f"meta_{tag}.json": _json(meta),
    }
    return files, {"meta": meta, "runs": pd.DataFrame(run_rows), "tag": tag}


def run_command(args: argparse.Namespace) -> int:
    started = time.time()
    out_dir = Path(args.out_dir or cvs.CV_ROOT)
    if not cvs.is_inside(out_dir, cvs.CV_ROOT):
        print("error: --out-dir must be inside data/processed/cv", file=sys.stderr)
        return 2
    files, info = run_models(args, out_dir)
    write_all(out_dir / args.target, files, args.force)
    meta, runs = info["meta"], info["runs"]
    print(
        f"wrote {len(files)} files for {info['tag']} ({args.target}): runs rows {len(runs)}, "
        f"oof rows {meta['rows'] * args.repeats}, splits_sha8 {meta['splits_sha8']}, "
        f"inner_sha8 {meta['inner_sha8']}, grid_sha8 {meta['grid_sha8']}, "
        f"features_sha8 {meta['features_sha8']}, {time.time() - started:.0f} s"
    )
    if args.show_metrics:
        for name in SHOWN_METRICS:
            col = runs[name].astype(float)
            print(
                f"{name}: mean {col.mean():.4f} sd {col.std(ddof=1) if len(col) > 1 else float('nan'):.4f}"
            )
    return 0


def permute_command(args: argparse.Namespace) -> int:
    started = time.time()
    out_dir = Path(args.out_dir or cvs.CV_ROOT)
    if not cvs.is_inside(out_dir, cvs.CV_ROOT):
        print("error: --out-dir must be inside data/processed/cv", file=sys.stderr)
        return 2
    if args.draws < 1:
        raise GuardFailure("--draws must be at least 1")
    p = prepare(args)
    values = []
    for draw in range(args.draws):
        y_perm = permute_labels(p.frame, p.y, np.random.default_rng(args.seed + draw))
        values.append(null_auroc(p.model, p.grid, p.X, y_perm, p.groups, p.splits, args.repeats))
    arr = np.array(values)
    mean = float(arr.mean())
    decision = "fail" if mean > PERMUTATION_MAX_MEAN_AUROC else "pass"
    summary = {
        "model": p.model,
        "feature_set": p.feature_set,
        "target": args.target,
        "seed": args.seed,
        "repeats": args.repeats,
        "draws": args.draws,
        "mean": mean,
        "sd": float(arr.std(ddof=1)) if len(arr) > 1 else None,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "max_allowed_mean": PERMUTATION_MAX_MEAN_AUROC,
        "decision": decision,
        "splits_sha8": cvs.splits_sha8(p.splits),
        "grid_sha8": grid_sha8(p.model, p.grid),
    }
    tag = f"{p.model}_{p.feature_set}"
    table = pd.DataFrame({"draw": range(args.draws), "null_auroc": arr})
    write_all(
        out_dir / args.target,
        {f"permutation_{tag}.csv": _csv(table), f"permutation_{tag}.json": _json(summary)},
        args.force,
    )
    sd = summary["sd"]
    print(
        f"permutation control {tag} ({args.target}): null AUROC over {args.draws} draws "
        f"mean {mean:.4f} sd {'n/a' if sd is None else f'{sd:.4f}'} min {arr.min():.4f} "
        f"max {arr.max():.4f}; allowed mean <= {PERMUTATION_MAX_MEAN_AUROC}; decision {decision}; "
        f"{time.time() - started:.0f} s"
    )
    if decision == "fail":
        print("error: permutation control failed: the null AUROC is above chance, something leaks",
              file=sys.stderr)  # fmt: skip
        return 1
    return 0


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    for name in ("--features", "--roles", "--meta", "--splits-dir", "--out-dir"):
        common.add_argument(name, type=Path, default=None)
    common.add_argument("--target", choices=sorted(cvs.TARGETS), default=cvs.DEFAULT_TARGET)
    common.add_argument("--model", choices=MODELS, required=True)
    common.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="A")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--force", action="store_true")
    run = sub.add_parser("run", parents=[common])
    run.add_argument("--repeats", type=int, default=5)
    run.add_argument("--show-metrics", action="store_true")
    perm = sub.add_parser("permute", parents=[common])
    perm.add_argument("--repeats", type=int, default=1)
    perm.add_argument("--draws", type=int, default=20)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_command(args) if args.command == "run" else permute_command(args)
    except InputProblem as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except (GuardFailure, LeakageError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
