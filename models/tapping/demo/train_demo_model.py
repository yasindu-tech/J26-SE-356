"""Offline demo model (task FT-47), run once, not imported by the app. Rule fixed in advance:
params = the mode of the official outer-fold best params per target (tie -> earliest grid
entry); LightGBM Set A fitted on all clips. Demo model; not used for any reported metric.
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


def render(done: Sequence[dict[str, Any]]) -> str:
    lines = [res.NOTE]
    for d in done:
        lines.append(f"{d['target']}: params {json.dumps(d['params'], sort_keys=True)}")
        lines.append(f"  folds per grid index: {d['counts']}")
        lines.append("  sha8: " + ", ".join(f"{n}={s}" for n, s in d["sha8"].items()))
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
            done.append(run_target(root / "cv", res.demo_dir(root), frame, splits, h, target))
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
