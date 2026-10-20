"""Baseline B2 (task FT-40): the reference classical features scored under our nested CV.

    python models/tapping/src/reference_baseline.py run     --model lgbm --target ge2
    python models/tapping/src/reference_baseline.py permute --model lgbm --target ge2
    python models/tapping/src/reference_baseline.py report

Fixed before any result: the 27 registered columns of reference_features.py (roles pinned by sha8),
the same 233 clips, the stored splits (splits and inner hashes must equal the official runs), the
existing grids (grid_sha8 must equal the official ones), threshold 0.5, 5 repeats x 5 outer folds.
Models: b1 (skipped if any feature is NaN: nothing is imputed, no clip is dropped) and lgbm. Calls the
existing harness functions without changing them; the leakage assertion also runs on a refit of every
fold. Writes NEW files only: oof_/runs_/meta_/best_params_/permutation_<model>_R.* and
reports_ci/reference_table.csv (CIs from bootstrap_ci's shared draws; Set A vs R is descriptive).
Exit codes: 0 ok, 1 guard failure or the permutation control failed, 2 missing input or bad path.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import baseline_report as br
import bootstrap_ci as bc
import cv_harness as ch
import cv_splits as cvs
import lightgbm
import numpy as np
import pandas as pd
import reference_features as rf
import sklearn
from cv_splits import GuardFailure, InputProblem, LeakageError

FEATURE_SET = "R"
MODELS = ("b1", "lgbm")
N_REPEATS = bc.EXPECTED_REPEATS
OFFICIAL_SET = "A"
SHOWN_METRICS = bc.METRICS[:5]  # auroc, balanced_accuracy, sensitivity, specificity, mcc
OUT_TABLE = "reference_table.csv"
TABLE_COLUMNS = [
    "target", "section", "subject", "metric", "estimate", "ci_low", "ci_high", "n_draws_used",
    "n_draws_finite", "n_excluded", "n_boot", "seed", "draws_sha8", "splits_sha8", "inner_sha8",
    "roles_sha8", "target_status", "features_sha8", "reference_roles_sha8",
]  # fmt: skip
NOTE = (
    "Set A vs R differences describe these {n} participants and are descriptive only; "
    "5 repeats are not independent evidence; no p-value is used as a decision rule."
)


# --------------------------------------------------------------------------- inputs


def load_reference(frame: pd.DataFrame, reference_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """The registered reference matrix in the order of the canonical frame; refuses any deviation."""
    files = {name: reference_dir / name for name in (rf.OUT_FEATURES, rf.OUT_ROLES, rf.OUT_META)}
    for name, path in files.items():
        if not path.is_file():
            raise InputProblem(f"missing input: {name}")
    table = pd.read_csv(
        files[rf.OUT_FEATURES], dtype={"ref": str, "participant": str}, float_precision="round_trip"
    )
    roles = pd.read_csv(files[rf.OUT_ROLES], dtype=str, keep_default_na=False)
    meta = json.loads(files[rf.OUT_META].read_text())
    if (
        cvs.roles_sha8_from_file(roles) != rf.PINNED_ROLES_SHA8
        or meta.get("reference_roles_sha8") != rf.PINNED_ROLES_SHA8
    ):
        raise GuardFailure("the reference roles do not match the registered roles_sha8")
    if list(table.columns) != ["ref", "participant", *rf.REFERENCE_COLUMNS]:
        raise GuardFailure(
            "the reference table is not exactly ref, participant and the 27 registered features"
        )
    if (
        meta.get("columns") != rf.REFERENCE_COLUMNS
        or meta.get("features_sha8") != rf.PINNED_FEATURES_SHA8
    ):
        raise GuardFailure("the reference meta file disagrees with the registered feature list")
    if table["ref"].duplicated().any() or set(table["ref"]) != set(frame["ref"]):
        raise GuardFailure("the reference table does not cover exactly the clips of features.csv")
    aligned = table.set_index("ref").loc[frame["ref"]]
    if (aligned["participant"].to_numpy() != frame["participant"].to_numpy()).any():
        raise GuardFailure("the reference table disagrees with features.csv on a participant")
    X = aligned[rf.REFERENCE_COLUMNS].to_numpy(dtype=float)
    if np.isinf(X).any():
        raise GuardFailure("the reference table holds an infinite value")
    return X, {
        "nan_counts": {c: int(aligned[c].isna().sum()) for c in rf.REFERENCE_COLUMNS},
        "meta": meta,
    }


def official_meta(cv_root: Path, target: str, model: str) -> dict[str, Any]:
    path = cv_root / target / f"meta_{model}_{OFFICIAL_SET}.json"
    if not path.is_file():
        raise InputProblem(f"missing input: official meta file for {target} {model}")
    return dict(json.loads(path.read_text()))


def prepare(args: argparse.Namespace) -> SimpleNamespace:
    cv_root = Path(args.cv_root or cvs.CV_ROOT)
    features = Path(args.features or cvs.PROCESSED_ROOT / "features.csv")
    roles = Path(args.roles or cvs.PROCESSED_ROOT / "feature_roles.csv")
    meta = Path(args.meta or cvs.PROCESSED_ROOT / "features_meta.json")
    reference_dir = Path(args.reference_dir or rf.REFERENCE_ROOT)
    frame, _ = cvs.load_inputs(features, roles, meta)
    frame = cvs.canonical(frame)
    X, info = load_reference(frame, reference_dir)
    target_dir = cv_root / args.target
    splits = cvs.load_splits(target_dir / cvs.OUT_SPLITS, frame)
    splits_meta = json.loads((target_dir / cvs.OUT_META).read_text())
    if (
        splits_meta.get("target") != args.target
        or splits_meta.get("seed") != args.seed
        or int(splits_meta["repeats"]) != N_REPEATS
    ):
        raise GuardFailure(
            "the stored splits are not the official ones (target, seed 0, 5 repeats)"
        )
    official = official_meta(cv_root, args.target, "lgbm")
    inner = cvs.inner_table(frame, splits, args.target)
    hashes = {"splits_sha8": cvs.splits_sha8(splits), "inner_sha8": cvs.inner_sha8(inner)}
    for key, value in hashes.items():
        if official.get(key) != value:
            raise GuardFailure(
                f"{key} differs from the official runs: the folds are not the stored ones"
            )
    grid = ch.grid_for(args.model)
    grid_hash = ch.grid_sha8(args.model, grid)
    if official_meta(cv_root, args.target, args.model).get("grid_sha8") != grid_hash:
        raise GuardFailure(
            f"grid_sha8 of {args.model} differs from the official runs: the grid was changed"
        )
    return SimpleNamespace(
        cv_root=cv_root, frame=frame, X=X, nan_counts=info["nan_counts"], reference_meta=info["meta"],
        y=cvs.target_labels(frame, args.target), groups=frame["participant"].to_numpy(),
        refs=frame["ref"].to_numpy(), splits=splits, official=official, grid=grid, grid_sha8=grid_hash,
        inputs={"features": cvs.sha8_of(features.read_bytes()), "reference_features": cvs.sha8_of((reference_dir / rf.OUT_FEATURES).read_bytes()), "splits": cvs.sha8_of((target_dir / cvs.OUT_SPLITS).read_bytes())},
        **hashes,
    )  # fmt: skip


def nan_total(ctx: SimpleNamespace) -> int:
    return int(sum(ctx.nan_counts.values()))


# --------------------------------------------------------------------------- fitting


def refit_fold(
    model: str,
    params: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
) -> np.ndarray:
    """Refit one outer fold from its stored best params, with the overlap assertion (cf. refit_outer)."""
    cvs.assert_disjoint(groups[train], groups[test], "refit_reference")
    return np.asarray(ch.fit_final(model, params, X, y, train).predict_proba(X[test])[:, 1])


def run_files(model: str, target: str, seed: int, ctx: SimpleNamespace) -> dict[str, bytes]:
    oof_parts, best_rows, run_rows = [], [], []
    columns_hash = cvs.sha8_of(json.dumps(rf.REFERENCE_COLUMNS))
    for repeat in range(N_REPEATS):
        member = ch.membership(ctx.groups, ctx.splits, repeat)
        proba, rows = ch.run_repeat(model, ctx.grid, ctx.X, ctx.y, ctx.groups, member, repeat)
        for fold, row in enumerate(rows):  # the same assertion, on a refit of every fold
            train, test = np.flatnonzero(member != fold), np.flatnonzero(member == fold)
            again = refit_fold(
                model, json.loads(row["params"]), ctx.X, ctx.y, ctx.groups, train, test
            )
            if not np.array_equal(again, proba[test]):
                raise GuardFailure(
                    f"repeat {repeat} fold {fold}: a refit does not reproduce the out-of-fold predictions"
                )
        best_rows += rows
        oof_parts.append(
            pd.DataFrame(
                {
                    "ref": ctx.refs,
                    "repeat": repeat,
                    "outer_fold": member,
                    "y": ctx.y,
                    "proba": proba,
                }
            )
        )
        run_rows.append({
            "model": model, "feature_set": FEATURE_SET, "target": target, "threshold": ch.THRESHOLD,
            "repeat": repeat, "seed": seed, **ch.repeat_metrics(ctx.y, proba, member),
            "n_features": len(rf.REFERENCE_COLUMNS), "features_sha8": columns_hash,
            "splits_sha8": ctx.splits_sha8, "inner_sha8": ctx.inner_sha8, "grid_sha8": ctx.grid_sha8,
        })  # fmt: skip
    official = ctx.official
    meta = {
        "tool": "reference-baseline-1", "model": model, "feature_set": FEATURE_SET, "target": target,
        "target_status": official["target_status"], "seed": seed, "repeats": N_REPEATS,
        "columns": rf.REFERENCE_COLUMNS, "features_sha8": columns_hash,
        "reference_roles_sha8": rf.PINNED_ROLES_SHA8, "nan_counts": ctx.nan_counts,
        "reference_source": ctx.reference_meta["source"], "inputs_sha8": ctx.inputs,
        "roles_sha8": official["roles_sha8"], "splits_sha8": ctx.splits_sha8, "inner_sha8": ctx.inner_sha8,
        "grid": ctx.grid, "grid_sha8": ctx.grid_sha8, "constants": official["constants"],
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                     "lightgbm": lightgbm.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "rows": len(ctx.frame), "participants": int(pd.Series(ctx.groups).nunique()),
    }  # fmt: skip
    tag = f"{model}_{FEATURE_SET}"
    return {
        f"oof_{tag}.csv": ch._csv(pd.concat(oof_parts, ignore_index=True)),
        f"best_params_{tag}.csv": ch._csv(pd.DataFrame(best_rows)),
        f"runs_{tag}.csv": ch._csv(pd.DataFrame(run_rows, columns=ch.RUN_COLUMNS)),
        f"meta_{tag}.json": ch._json(meta),
    }


def _guard_root(path: Path, name: str) -> int | None:
    if not cvs.is_inside(path, cvs.CV_ROOT):
        print(f"error: {name} must be inside data/processed/cv", file=sys.stderr)
        return 2
    return None


def run_command(args: argparse.Namespace) -> int:
    started = time.time()
    cv_root = Path(args.cv_root or cvs.CV_ROOT)
    if (code := _guard_root(cv_root, "--cv-root")) is not None:
        return code
    ctx = prepare(args)
    if args.model == "b1" and nan_total(ctx):
        bad = {k: v for k, v in ctx.nan_counts.items() if v}
        print(
            f"skipped b1 on R: {len(bad)} feature(s) have NaN {bad}; nothing is imputed and no clip is dropped"
        )
        return 0
    files = run_files(args.model, args.target, args.seed, ctx)
    ch.write_all(cv_root / args.target, files, args.force)
    print(
        f"wrote {len(files)} files for {args.model}_{FEATURE_SET} ({args.target}): runs rows {N_REPEATS}, "
        f"oof rows {len(ctx.frame) * N_REPEATS}, splits_sha8 {ctx.splits_sha8}, inner_sha8 {ctx.inner_sha8}, "
        f"grid_sha8 {ctx.grid_sha8}, features_sha8 {cvs.sha8_of(json.dumps(rf.REFERENCE_COLUMNS))}, "
        f"nan features {sum(v > 0 for v in ctx.nan_counts.values())}, {time.time() - started:.0f} s"
    )
    return 0


def permute_command(args: argparse.Namespace) -> int:
    started = time.time()
    cv_root = Path(args.cv_root or cvs.CV_ROOT)
    if (code := _guard_root(cv_root, "--cv-root")) is not None:
        return code
    if args.draws < 1:
        raise GuardFailure("--draws must be at least 1")
    ctx = prepare(args)
    if args.model == "b1" and nan_total(ctx):
        print("skipped the b1 permutation control on R: NaN features")
        return 0
    first = ctx.splits[ctx.splits["repeat"] < 1].reset_index(
        drop=True
    )  # repeat 0, as the official control
    values = [
        ch.null_auroc(args.model, ctx.grid, ctx.X, ch.permute_labels(ctx.frame, ctx.y, np.random.default_rng(args.seed + d)), ctx.groups, first, 1)
        for d in range(args.draws)
    ]  # fmt: skip
    arr = np.array(values)
    mean = float(arr.mean())
    decision = "fail" if mean > ch.PERMUTATION_MAX_MEAN_AUROC else "pass"
    summary = {
        "model": args.model, "feature_set": FEATURE_SET, "target": args.target, "seed": args.seed,
        "repeats": 1, "draws": args.draws, "mean": mean, "sd": float(arr.std(ddof=1)) if len(arr) > 1 else None,
        "min": float(arr.min()), "max": float(arr.max()), "max_allowed_mean": ch.PERMUTATION_MAX_MEAN_AUROC,
        "decision": decision, "splits_sha8": cvs.splits_sha8(first), "grid_sha8": ctx.grid_sha8,
    }  # fmt: skip
    tag = f"{args.model}_{FEATURE_SET}"
    table = pd.DataFrame({"draw": range(args.draws), "null_auroc": arr})
    ch.write_all(
        cv_root / args.target,
        {f"permutation_{tag}.csv": ch._csv(table), f"permutation_{tag}.json": ch._json(summary)},
        args.force,
    )
    sd = summary["sd"]
    print(
        f"permutation control {tag} ({args.target}): null AUROC over {args.draws} draws mean {mean:.4f} "
        f"sd {'n/a' if sd is None else f'{sd:.4f}'} min {arr.min():.4f} max {arr.max():.4f}; "
        f"allowed mean <= {ch.PERMUTATION_MAX_MEAN_AUROC}; decision {decision}; {time.time() - started:.0f} s"
    )
    if decision == "fail":
        print(
            "error: permutation control failed: the null AUROC is above chance, something leaks",
            file=sys.stderr,
        )
        return 1
    return 0


# --------------------------------------------------------------------------- the report


def report_rows(
    cv_root: Path,
    target: str,
    ref_table: pd.DataFrame,
    counts: np.ndarray,
    order: list[str],
    reference_sha8: str,
) -> list[dict[str, Any]]:
    folder = cv_root / target
    r_models = [m for m in MODELS if (folder / f"runs_{m}_{FEATURE_SET}.csv").is_file()]
    if "lgbm" not in r_models:
        raise InputProblem(f"missing input: runs file for {target} lgbm {FEATURE_SET}")
    keys = [("lgbm", OFFICIAL_SET), *[(m, FEATURE_SET) for m in r_models]]
    models = {k: bc.load_model(cv_root, target, k[0], k[1], ref_table["ref"]) for k in keys}
    shared = bc.check_agreement(target, models)
    for m in (
        r_models
    ):  # the saved runs must come from the current reference table and the official grid
        run_meta = models[(m, FEATURE_SET)]["meta"]
        if run_meta.get("inputs_sha8", {}).get("reference_features") != reference_sha8:
            raise GuardFailure(
                f"{target}: {m} {FEATURE_SET} was not produced from the current reference table"
            )
        if run_meta.get("grid_sha8") != official_meta(cv_root, target, m).get("grid_sha8"):
            raise GuardFailure(
                f"{target}: grid_sha8 of {m} {FEATURE_SET} differs from the official runs"
            )
    for m in r_models:  # every R model needs a passing permutation control
        br.load_permutation(folder, f"{m}_{FEATURE_SET}", f"{target} {m} {FEATURE_SET}")
    y = models[keys[0]]["y"]
    index = {p: i for i, p in enumerate(order)}
    weights = counts[:, np.array([index[p] for p in ref_table["participant"]])].astype(float)
    valid = (weights @ (y == 1) > 0) & (weights @ (y == 0) > 0)
    n_excluded = int((~valid).sum())
    if n_excluded > bc.MAX_EXCLUDED_SHARE * len(counts):
        raise GuardFailure(
            f"{target}: {n_excluded} of {len(counts)} draws lack a class in a repeat, more than {bc.MAX_EXCLUDED_SHARE:.0%}"
        )
    weights = weights[valid]
    ones = np.ones((1, len(y)))
    common = {
        "target": target, "n_draws_used": int(valid.sum()), "n_excluded": n_excluded, "n_boot": len(counts),
        "seed": bc.SEED, "draws_sha8": bc.draws_sha8(counts), "splits_sha8": shared["splits_sha8"],
        "inner_sha8": shared["inner_sha8"], "roles_sha8": shared["roles_sha8"],
        "target_status": shared["target_status"], "reference_roles_sha8": rf.PINNED_ROLES_SHA8,
    }  # fmt: skip
    rows: list[dict[str, Any]] = []
    for m in r_models:
        entry = models[(m, FEATURE_SET)]
        point = bc.statistic(entry["y"], entry["probas"], ones)
        boot = bc.statistic(entry["y"], entry["probas"], weights)
        for metric in SHOWN_METRICS:
            reported = entry["runs"][metric].to_numpy(dtype=float).mean()
            estimate = float(point[metric][0])
            if not (np.isnan(estimate) and np.isnan(reported)) and abs(estimate - reported) > 1e-9:
                raise GuardFailure(
                    f"{target}: {m} {FEATURE_SET}: oof and runs disagree on {metric}"
                )
            low, high, n_finite = bc.interval(boot[metric])
            rows.append({**common, "section": "performance", "subject": f"{m} {FEATURE_SET}", "metric": metric,
                         "estimate": estimate, "ci_low": low, "ci_high": high, "n_draws_finite": n_finite,
                         "features_sha8": entry["meta"]["features_sha8"]})  # fmt: skip
    auc = {
        k: np.array([bc.weighted_auroc(y, p, weights) for p in e["probas"]])
        for k, e in models.items()
    }
    auc_point = {
        k: np.array([bc.weighted_auroc(y, p, ones)[0] for p in e["probas"]])
        for k, e in models.items()
    }
    pairs = [(("lgbm", OFFICIAL_SET), ("lgbm", FEATURE_SET))]
    if "b1" in r_models:
        pairs.append((("lgbm", FEATURE_SET), ("b1", FEATURE_SET)))
    for minuend, subtrahend in pairs:
        draws = (auc[minuend] - auc[subtrahend]).mean(axis=0)
        low, high, n_finite = bc.interval(draws)
        rows.append({**common, "section": "difference", "subject": f"{minuend[0]} {minuend[1]} minus {subtrahend[0]} {subtrahend[1]}",
                     "metric": "auroc", "estimate": float((auc_point[minuend] - auc_point[subtrahend]).mean()),
                     "ci_low": low, "ci_high": high, "n_draws_finite": n_finite,
                     "features_sha8": f"{models[minuend]['meta']['features_sha8']}|{models[subtrahend]['meta']['features_sha8']}"})  # fmt: skip
    return rows


def build_report(
    cv_root: Path,
    features_path: Path,
    targets: Sequence[str],
    reference_dir: Path | None = None,
) -> tuple[pd.DataFrame, int]:
    reference_path = Path(reference_dir or rf.REFERENCE_ROOT) / rf.OUT_FEATURES
    if not reference_path.is_file():
        raise InputProblem(f"missing input: {rf.OUT_FEATURES}")
    reference_sha8 = cvs.sha8_of(reference_path.read_bytes())
    ref_table = bc.read_participants(features_path)
    order = bc.sorted_participants(ref_table["participant"].tolist())
    counts = bc.bootstrap_draws(order, bc.N_DRAWS, bc.SEED)
    rows: list[dict[str, Any]] = []
    for target in sorted(set(targets)):
        rows += report_rows(cv_root, target, ref_table, counts, order, reference_sha8)
    table = pd.DataFrame(rows, columns=TABLE_COLUMNS)
    subject_rank = {"b1 R": 0, "lgbm R": 1, "lgbm A minus lgbm R": 0, "lgbm R minus b1 R": 1}
    table["_s"] = [0 if s == "performance" else 1 for s in table["section"]]
    table["_r"] = [subject_rank[s] for s in table["subject"]]
    table["_m"] = [SHOWN_METRICS.index(m) for m in table["metric"]]
    table = (
        table.sort_values(["target", "_s", "_r", "_m"], kind="stable")
        .drop(columns=["_s", "_r", "_m"])
        .reset_index(drop=True)
    )
    return table, len(order)


def render_report(table: pd.DataFrame, n_participants: int) -> str:
    lines = [
        f"Reference baseline B2: classical reference features (R), participant-bootstrap 95% CIs ({bc.N_DRAWS} draws, seed {bc.SEED})"
    ]
    for target, part in table.groupby("target", sort=True):
        r = part.iloc[0]
        lines.append(f"{target}: splits_sha8 {r['splits_sha8']}, inner_sha8 {r['inner_sha8']}, roles_sha8 {r['roles_sha8']}, target_status {r['target_status']}, draws_sha8 {r['draws_sha8']}, reference_roles_sha8 {r['reference_roles_sha8']}; draws used {int(r['n_draws_used'])}, excluded {int(r['n_excluded'])}")  # fmt: skip
        perf = part[part["section"] == "performance"]
        header = ["model", *SHOWN_METRICS]
        grid = []
        for subject, rows in perf.groupby("subject", sort=False):
            cells = [
                f"{bc.num(x.estimate)} [{bc.num(x.ci_low)}, {bc.num(x.ci_high)}]"
                for x in rows.itertuples()
            ]
            grid.append([str(subject), *cells])
        widths = [max(len(header[c]), *(len(row[c]) for row in grid)) for c in range(len(header))]
        lines.append("  ".join(h.ljust(w) for h, w in zip(header, widths, strict=True)))
        lines += ["  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) for row in grid]
        for x in part[part["section"] == "difference"].itertuples():
            lines.append(
                f"AUROC difference {x.subject}: {bc.num(x.estimate)} [{bc.num(x.ci_low)}, {bc.num(x.ci_high)}] (descriptive)"
            )
    lines.append(NOTE.format(n=n_participants))
    return "\n".join(lines)


def report_command(args: argparse.Namespace) -> int:
    cv_root = Path(args.cv_root or cvs.CV_ROOT)
    out_dir = Path(args.out_dir or cv_root / "reports_ci")
    features = Path(args.features or cvs.PROCESSED_ROOT / "features.csv")
    for name, path in (("--cv-root", cv_root), ("--out-dir", out_dir)):
        if (code := _guard_root(path, name)) is not None:
            return code
    table, n_participants = build_report(cv_root, features, args.targets, args.reference_dir)
    data = bc.table_bytes(table)
    text = render_report(table, n_participants)
    bc._atomic(out_dir / OUT_TABLE, data)
    print(text)
    print(f"wrote {OUT_TABLE}: {len(table)} rows")
    return 0


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    for name in ("--features", "--roles", "--meta", "--cv-root", "--reference-dir"):
        common.add_argument(name, type=Path, default=None)
    common.add_argument("--target", choices=sorted(cvs.TARGETS), default=cvs.DEFAULT_TARGET)
    common.add_argument("--model", choices=MODELS, required=True)
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--force", action="store_true")
    run = sub.add_parser("run", parents=[common])
    run.set_defaults(func=run_command)
    perm = sub.add_parser("permute", parents=[common])
    perm.add_argument("--draws", type=int, default=20)
    perm.set_defaults(func=permute_command)
    rep = sub.add_parser("report")
    for name in ("--features", "--cv-root", "--out-dir", "--reference-dir"):
        rep.add_argument(name, type=Path, default=None)
    rep.add_argument("--targets", nargs="+", choices=sorted(cvs.TARGETS), default=["ge2", "ge1"])
    rep.set_defaults(func=report_command)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except InputProblem as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except (GuardFailure, LeakageError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
