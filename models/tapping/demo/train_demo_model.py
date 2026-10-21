"""Offline demo models (tasks FT-47, FT-49), run once, not imported by the app. Rule fixed in
advance: params = the mode of the official outer-fold best params per target (tie -> earliest
grid entry); LightGBM Set A fitted on all clips. Also saves the 25 official outer-fold refits
(each must reproduce the official OOF) for the score band. Not used for any reported metric.
Run: python models/tapping/demo/train_demo_model.py   (writes data/processed/demo/)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv_harness as ch  # noqa: E402
import cv_splits as cvs  # noqa: E402
import lightgbm  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import resources as res  # noqa: E402
import treeshap_heldout as tsh  # noqa: E402
from cv_splits import GuardFailure, InputProblem, LeakageError  # noqa: E402

TARGETS = ("ge2", "ge1")
RULE = (
    "mode of the 25 official outer-fold best params; ties go to the earliest configuration "
    "in the lgbm grid order (simplest first); fitted on all clips, Set A"
)


def choose_params(
    best: dict[tuple[int, int], dict[str, Any]], grid: list[dict[str, Any]]
) -> tuple[int, list[int]]:
    """Index of the most frequent grid entry (earliest on a tie) and the count per entry."""
    seen = Counter(grid.index(p) for p in best.values())
    counts = [seen.get(i, 0) for i in range(len(grid))]
    top = max(counts)
    return counts.index(top), counts  # list.index = the first (simplest) of the tied entries


def fit_all(frame: pd.DataFrame, target: str, params: dict[str, Any]) -> Any:
    x, _ = ch.design_matrix(frame, res.FEATURE_SET)
    return ch.make_estimator("lgbm", params).fit(x, cvs.target_labels(frame, target))


def model_files(
    est: Any, target: str, idx: int, counts: list[int], grid: list[dict[str, Any]],
    inputs_sha8: dict[str, str], n_clips: int,
) -> dict[str, bytes]:  # fmt: skip
    cols = list(ch.FEATURE_SETS[res.FEATURE_SET])
    model = est.booster_.model_to_string().encode()
    meta = {
        "tool": res.TOOL, "target": target, "feature_set": res.FEATURE_SET, "model": "lgbm",
        "columns": cols, "params": grid[idx], "grid_index": idx, "selection_rule": RULE,
        "config_counts": [{"grid_index": i, "params": grid[i], "n_folds": n}
                          for i, n in enumerate(counts) if n],  # fmt: skip
        "n_best_params": sum(counts), "n_clips": n_clips,
        "features_sha8": cvs.sha8_of(json.dumps(cols)),
        "grid_sha8": ch.grid_sha8("lgbm", ch.grid_for("lgbm")),
        "inputs_sha8": inputs_sha8, "lightgbm": lightgbm.__version__, "numpy": np.__version__,
        "model_sha8": cvs.sha8_of(model), "note": res.NOTE,
    }  # fmt: skip
    return {
        res.MODEL_FILE.format(target=target): model,
        res.META_FILE.format(target=target): (
            json.dumps(meta, indent=2, sort_keys=True) + "\n"
        ).encode(),
    }


def run_target(
    cv_root: Path, out_dir: Path, frame: pd.DataFrame, splits: pd.DataFrame,
    hashes: dict[str, str], target: str,
) -> dict[str, Any]:  # fmt: skip
    folder = cv_root / target
    meta_path, best_path = (
        folder / f"meta_lgbm_{res.FEATURE_SET}.json",
        folder / (f"best_params_lgbm_{res.FEATURE_SET}.csv"),
    )
    for path in (meta_path, best_path):
        if not path.is_file():
            raise InputProblem(f"missing input: {path.name} for {target}")
    official = json.loads(meta_path.read_text())
    tsh.check_official(official, frame, splits, target, res.FEATURE_SET, hashes)
    grid = ch.grid_for("lgbm")
    best = tsh.read_best_params(best_path, grid)
    idx, counts = choose_params(best, grid)
    est = fit_all(cvs.canonical(frame), target, grid[idx])  # fixed row order
    inputs = {**hashes, "best_params": cvs.sha8_of(best_path.read_bytes())}
    files = model_files(est, target, idx, counts, grid, inputs, len(frame))
    for name, data in files.items():
        cvs._atomic(out_dir / name, data)
    return {"target": target, "params": grid[idx], "counts": {i: n for i, n in enumerate(counts) if n},
            "sha8": {n: cvs.sha8_of(d) for n, d in files.items()}}  # fmt: skip


def fold_refits(
    frame: pd.DataFrame, splits: pd.DataFrame, oof: pd.DataFrame,
    best: dict[tuple[int, int], dict[str, Any]], target: str,
) -> list[dict[str, Any]]:  # fmt: skip
    """The 25 outer-fold refits as treeshap_heldout makes them; each must reproduce the OOF."""
    frame = cvs.canonical(frame)
    x, _ = ch.design_matrix(frame, res.FEATURE_SET)
    groups, refs = frame["participant"].to_numpy(), frame["ref"].to_numpy()
    official = {r: g.set_index("ref") for r, g in oof.groupby("repeat")}
    out: list[dict[str, Any]] = []
    for repeat in range(tsh.REPEATS):
        member = ch.membership(groups, splits, repeat)
        saved = official[repeat].reindex(refs)
        if saved["proba"].isna().any() or (saved["outer_fold"].to_numpy() != member).any():
            raise GuardFailure(f"repeat {repeat}: the saved OOF does not match the stored folds")
        for fold in range(cvs.N_OUTER):
            train, test = np.flatnonzero(member != fold), np.flatnonzero(member == fold)
            est = ch.refit_outer(
                frame, "lgbm", best[(repeat, fold)], train, res.FEATURE_SET, target,
                test_positions=test,
            )  # fmt: skip
            proba = est.predict_proba(x[test])[:, 1]
            diff = float(np.max(np.abs(proba - saved["proba"].to_numpy()[test])))
            if not diff <= tsh.OOF_TOL:
                raise GuardFailure(
                    f"repeat {repeat} fold {fold}: the refit does not reproduce the official "
                    f"OOF (max diff {diff:.3g})"
                )
            out.append({"repeat": repeat, "fold": fold, "est": est, "diff": diff,
                        "params": best[(repeat, fold)], "n_test": len(test)})  # fmt: skip
    return out


def fold_files(
    refits: list[dict[str, Any]], target: str, inputs_sha8: dict[str, str]
) -> dict[str, bytes]:
    cols = list(ch.FEATURE_SETS[res.FEATURE_SET])
    files = {
        res.FOLD_FILE.format(repeat=r["repeat"], fold=r["fold"]): r["est"]
        .booster_.model_to_string()
        .encode()
        for r in refits
    }
    meta = {
        "tool": res.TOOL, "target": target, "feature_set": res.FEATURE_SET, "model": "lgbm",
        "columns": cols, "n_models": len(refits), "oof_tolerance": tsh.OOF_TOL,
        "oof_max_diff": max(r["diff"] for r in refits),
        "folds": [{"repeat": r["repeat"], "fold": r["fold"], "params": r["params"],
                   "n_test_rows": r["n_test"], "oof_max_diff": r["diff"]} for r in refits],
        "files_sha8": {n: cvs.sha8_of(d) for n, d in files.items()},
        "features_sha8": cvs.sha8_of(json.dumps(cols)),
        "grid_sha8": ch.grid_sha8("lgbm", ch.grid_for("lgbm")),
        "inputs_sha8": inputs_sha8, "lightgbm": lightgbm.__version__, "numpy": np.__version__,
        "note": res.NOTE,
    }  # fmt: skip
    files[res.FOLD_META] = (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode()
    return files


def run_folds(
    cv_root: Path, out_dir: Path, frame: pd.DataFrame, splits: pd.DataFrame,
    hashes: dict[str, str], target: str,
) -> dict[str, Any]:  # fmt: skip
    folder = cv_root / target
    best_path, oof_path = folder / "best_params_lgbm_A.csv", folder / "oof_lgbm_A.csv"
    for path in (best_path, oof_path):
        if not path.is_file():
            raise InputProblem(f"missing input: {path.name} for {target}")
    best = tsh.read_best_params(best_path, ch.grid_for("lgbm"))
    refits = fold_refits(frame, splits, ch.read_oof(oof_path), best, target)
    inputs = {**hashes, "best_params": cvs.sha8_of(best_path.read_bytes()),
              "oof": cvs.sha8_of(oof_path.read_bytes())}  # fmt: skip
    files = fold_files(refits, target, inputs)
    for name, data in files.items():
        cvs._atomic(out_dir / "folds" / target / name, data)
    return {"target": target, "n_models": len(refits),
            "oof_max_diff": max(r["diff"] for r in refits),
            "meta_sha8": cvs.sha8_of(files[res.FOLD_META])}  # fmt: skip


def render(done: Sequence[dict[str, Any]]) -> str:
    lines = [res.NOTE]
    for d in done:
        lines.append(f"{d['target']}: params {json.dumps(d['params'], sort_keys=True)}")
        lines.append(f"  folds per grid index: {d['counts']}")
        lines.append("  sha8: " + ", ".join(f"{n}={s}" for n, s in d["sha8"].items()))
        f = d["folds"]
        lines.append(
            f"  fold models: {f['n_models']}, max OOF diff {f['oof_max_diff']:.3g}, "
            f"folds_meta sha8 {f['meta_sha8']}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=None, help="data/processed (default)")
    args = ap.parse_args(argv)
    root = Path(args.root or cvs.PROCESSED_ROOT)
    try:
        paths = (root / "features.csv", root / "feature_roles.csv", root / "features_meta.json")
        frame, _ = cvs.load_inputs(*paths)  # as treeshap_heldout: guards see this frame
        hashes = cvs.input_hashes(*paths)
        done = []
        for target in TARGETS:
            splits_path = root / "cv" / target / cvs.OUT_SPLITS
            splits = cvs.load_splits(splits_path, cvs.canonical(frame))
            h = {**hashes, "splits": cvs.sha8_of(splits_path.read_bytes())}
            out = res.demo_dir(root)
            entry = run_target(root / "cv", out, frame, splits, h, target)
            entry["folds"] = run_folds(root / "cv", out, frame, splits, h, target)
            done.append(entry)
    except InputProblem as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except (GuardFailure, LeakageError) as err:
        print(f"refused: {err}", file=sys.stderr)
        return 1
    print(render(done))
    return 0


if __name__ == "__main__":
    sys.exit(main())
