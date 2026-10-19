"""TreeSHAP on held-out folds (task FT-42). SHAP shows which features contributed to the model's
score, not cause. Refits each official outer fold of lgbm (stored best params, same seed) with
cv_harness.refit_outer, explains the TEST rows only with shap.TreeExplainer (tree_path_dependent,
raw margin), and checks additivity and reproduction of the official OOF probabilities.
Writes data/processed/xai/<target>/<set>/. Exit codes: 0 ok, 1 guard failure or OOF not
reproduced, 2 missing input or a path outside data/processed/xai.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv_harness as ch
import cv_splits as cvs
import lightgbm
import numpy as np
import pandas as pd
import shap
from cv_splits import GuardFailure, InputProblem, LeakageError

TOOL = "treeshap-heldout-1"
CONFIGS = (("ge2", "A"), ("ge1", "A"), ("ge2", "B"))  # pre-registered; model is lgbm only
MODEL = "lgbm"
EXPLAINER = {"feature_perturbation": "tree_path_dependent", "model_output": "raw"}
ADDITIVITY_TOL = 1e-6
OOF_TOL = 1e-9
REPEATS = 5
TOP_N = 5
SHAP_FILE, FOLD_FILE, GLOBAL_FILE, META_FILE = (
    "shap_values.csv", "shap_importance_by_fold.csv", "shap_importance_global.csv", "shap_meta.json"
)  # fmt: skip


# --------------------------------------------------------------------------- guards


def check_official(
    meta: dict[str, Any], frame: pd.DataFrame, splits: pd.DataFrame, target: str, fs: str,
    hashes: dict[str, str],
) -> None:  # fmt: skip
    """The saved official run must be the one these inputs, grid and folds would reproduce."""
    where = f"{target} {MODEL} {fs}"
    cols = ch.FEATURE_SETS[fs]
    expected = {
        "model": MODEL, "feature_set": fs, "target": target, "repeats": REPEATS,
        "columns": cols, "features_sha8": cvs.sha8_of(json.dumps(cols)),
        "grid_sha8": ch.grid_sha8(MODEL, ch.grid_for(MODEL)),
        "splits_sha8": cvs.splits_sha8(splits),
        "inner_sha8": cvs.inner_sha8(cvs.inner_table(frame, splits, target)),
    }  # fmt: skip
    for key, value in expected.items():
        if meta.get(key) != value:
            raise GuardFailure(f"{where}: {key} differs from the official run")
    if meta.get("constants", {}).get("lgbm_seed") != ch.LGBM_SEED:
        raise GuardFailure(f"{where}: lgbm_seed differs from the official run")
    for key, value in hashes.items():
        if meta.get("inputs_sha8", {}).get(key) != value:
            raise GuardFailure(f"{where}: input {key} differs from the official run")
    versions = meta.get("versions", {})
    if versions.get("lightgbm") != lightgbm.__version__ or versions.get("numpy") != np.__version__:
        raise GuardFailure(f"{where}: lightgbm or numpy version differs from the official run")


def read_best_params(
    path: Path, grid: list[dict[str, Any]]
) -> dict[tuple[int, int], dict[str, Any]]:
    table = pd.read_csv(path)
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for row in table.itertuples(index=False):
        key = (int(row.repeat), int(row.outer_fold))
        params = json.loads(row.params)
        if key in out or params not in grid:
            raise GuardFailure("best_params: a duplicated fold or a configuration outside the grid")
        out[key] = params
    expected = {(r, f) for r in range(REPEATS) for f in range(cvs.N_OUTER)}
    if set(out) != expected:
        raise GuardFailure("best_params: expected one row per repeat and outer fold")
    return out


# --------------------------------------------------------------------------- explaining


def explain_values(est: Any, x_test: np.ndarray) -> tuple[float, np.ndarray]:
    explainer = shap.TreeExplainer(est, **EXPLAINER)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="LightGBM binary classifier with TreeExplainer")
        values = explainer.shap_values(x_test, check_additivity=False)
    if not isinstance(values, np.ndarray) or values.shape != x_test.shape:
        raise GuardFailure("TreeExplainer returned an unexpected shape for a binary classifier")
    base = np.asarray(explainer.expected_value, dtype=float).ravel()
    if base.size != 1:
        raise GuardFailure("TreeExplainer returned more than one base value")
    return float(base[0]), values


def explain_config(
    frame: pd.DataFrame, splits: pd.DataFrame, oof: pd.DataFrame,
    best: dict[tuple[int, int], dict[str, Any]], target: str, fs: str,
) -> dict[str, Any]:  # fmt: skip
    frame = cvs.canonical(frame)
    X, cols = ch.design_matrix(frame, fs)
    groups, refs = frame["participant"].to_numpy(), frame["ref"].to_numpy()
    official = {r: g.set_index("ref") for r, g in oof.groupby("repeat")}
    parts: list[pd.DataFrame] = []
    max_diff = max_add = 0.0
    for repeat in range(REPEATS):
        member = ch.membership(groups, splits, repeat)
        saved = official[repeat].reindex(refs)
        if saved["proba"].isna().any() or (saved["outer_fold"].to_numpy() != member).any():
            raise GuardFailure(f"repeat {repeat}: the saved OOF does not match the stored folds")
        seen = np.zeros(len(frame), dtype=int)
        for fold in range(cvs.N_OUTER):
            train, test = np.flatnonzero(member != fold), np.flatnonzero(member == fold)
            if set(train.tolist()) & set(test.tolist()):
                raise LeakageError(f"repeat {repeat} fold {fold}: train and test rows overlap")
            cvs.assert_disjoint(
                groups[train], groups[test], f"treeshap repeat {repeat} fold {fold}"
            )
            est = ch.refit_outer(
                frame, MODEL, best[(repeat, fold)], train, fs, target, test_positions=test
            )
            x_test = X[test]
            base, values = explain_values(est, x_test)
            raw = np.asarray(est.predict(x_test, raw_score=True), dtype=float)
            add_err = float(np.max(np.abs(base + values.sum(axis=1) - raw)))
            if not add_err <= ADDITIVITY_TOL:
                raise GuardFailure(f"repeat {repeat} fold {fold}: additivity error {add_err:.3g}")
            proba = est.predict_proba(x_test)[:, 1]
            max_diff = max(max_diff, float(np.max(np.abs(proba - saved["proba"].to_numpy()[test]))))
            max_add = max(max_add, add_err)
            seen[test] += 1
            part = pd.DataFrame(
                {"repeat": repeat, "outer_fold": fold, "clip_ref": refs[test],
                 "participant_ref": groups[test], "base_value": base, "raw_margin": raw,
                 "proba": proba, "_pos": test}
            )  # fmt: skip
            for j, c in enumerate(cols):
                part[f"shap_{c}"] = values[:, j]
            for j, c in enumerate(cols):
                part[f"value_{c}"] = x_test[:, j]  # NaN stays NaN
            parts.append(part)
        if not (seen == 1).all():
            raise GuardFailure(f"repeat {repeat}: a clip was not explained exactly once")
    table = pd.concat(parts, ignore_index=True).sort_values(["repeat", "_pos"], kind="stable")
    table = table.drop(columns="_pos").reset_index(drop=True)
    return {"table": table, "cols": cols, "max_diff": max_diff, "max_add": max_add}


def importance_tables(
    table: pd.DataFrame, cols: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (repeat, fold), g in table.groupby(["repeat", "outer_fold"], sort=True):
        for c in cols:
            rows.append({"repeat": repeat, "outer_fold": fold, "feature": c,
                         "mean_abs_shap": float(g[f"shap_{c}"].abs().mean()), "n_test_rows": len(g)})  # fmt: skip
    by_fold = pd.DataFrame(rows)
    per_repeat = {c: [float(g[f"shap_{c}"].abs().mean()) for _, g in table.groupby("repeat")]
                  for c in cols}  # fmt: skip
    glob = pd.DataFrame({"feature": list(cols),
                         "mean_abs_shap_mean_over_repeats": [float(np.mean(per_repeat[c])) for c in cols]})  # fmt: skip
    glob = glob.sort_values(["mean_abs_shap_mean_over_repeats", "feature"], ascending=[False, True],
                            kind="stable").reset_index(drop=True)  # fmt: skip
    glob["rank"] = np.arange(1, len(glob) + 1)
    return by_fold, glob


# --------------------------------------------------------------------------- run


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.17g", na_rep="").encode()


def xai_root() -> Path:
    return cvs.PROCESSED_ROOT / "xai"


def run_config(
    cv_root: Path, out_root: Path, frame: pd.DataFrame, input_hashes: dict[str, str],
    target: str, fs: str,
) -> dict[str, Any]:  # fmt: skip
    folder = cv_root / target
    paths = {
        "meta": folder / f"meta_{MODEL}_{fs}.json",
        "oof": folder / f"oof_{MODEL}_{fs}.csv",
        "best": folder / f"best_params_{MODEL}_{fs}.csv",
        "splits": folder / cvs.OUT_SPLITS,
    }
    for name, path in paths.items():
        if not path.is_file():
            raise InputProblem(f"missing input: {name} file for {target} {MODEL} {fs}")
    meta = json.loads(paths["meta"].read_text())
    splits = cvs.load_splits(paths["splits"], cvs.canonical(frame))
    hashes = {**input_hashes, "splits": cvs.sha8_of(paths["splits"].read_bytes())}
    check_official(meta, frame, splits, target, fs, hashes)
    best = read_best_params(paths["best"], ch.grid_for(MODEL))
    oof = ch.read_oof(paths["oof"])
    result = explain_config(frame, splits, oof, best, target, fs)
    table, cols = result["table"], result["cols"]
    by_fold, glob = importance_tables(table, cols)
    files = {SHAP_FILE: csv_bytes(table), FOLD_FILE: csv_bytes(by_fold),
             GLOBAL_FILE: csv_bytes(glob)}  # fmt: skip
    out_meta = {
        "tool": TOOL, "target": target, "feature_set": fs, "model": MODEL,
        "target_status": meta.get("target_status"), "columns": cols,
        "features_sha8": meta["features_sha8"], "grid_sha8": meta["grid_sha8"],
        "splits_sha8": meta["splits_sha8"], "inner_sha8": meta["inner_sha8"],
        "inputs_sha8": {**hashes, "oof": cvs.sha8_of(paths["oof"].read_bytes()),
                        "best_params": cvs.sha8_of(paths["best"].read_bytes())},  # fmt: skip
        "explainer": {"class": "shap.TreeExplainer", **EXPLAINER},
        "tolerances": {"additivity": ADDITIVITY_TOL, "oof_reproduction": OOF_TOL},
        "max_abs_diff_vs_official_oof": result["max_diff"],
        "oof_reproduced": result["max_diff"] <= OOF_TOL,
        "max_additivity_error": result["max_add"],
        "row_counts": {SHAP_FILE: len(table), FOLD_FILE: len(by_fold), GLOBAL_FILE: len(glob)},
        "versions": {"shap": shap.__version__, "lightgbm": lightgbm.__version__,
                     "numpy": np.__version__},  # fmt: skip
    }  # fmt: skip
    files[META_FILE] = (json.dumps(out_meta, indent=2, sort_keys=True) + "\n").encode()
    out_dir = out_root / target / fs
    for name, data in files.items():
        cvs._atomic(out_dir / name, data)
    return {"target": target, "fs": fs, "glob": glob, "meta": out_meta,
            "sha8": {n: cvs.sha8_of(d) for n, d in files.items()}}  # fmt: skip


def render(done: Sequence[dict[str, Any]]) -> str:
    lines = []
    for d in done:
        m = d["meta"]
        lines.append(f"== {d['target']} lgbm {d['fs']}  features {m['features_sha8']}  grid "
                     f"{m['grid_sha8']}  splits {m['splits_sha8']}")  # fmt: skip
        lines.append(f"  rows {m['row_counts'][SHAP_FILE]}  by-fold rows {m['row_counts'][FOLD_FILE]}"
                     f"  max additivity error {m['max_additivity_error']:.3g}"
                     f"  max diff vs official OOF {m['max_abs_diff_vs_official_oof']:.3g}")  # fmt: skip
        lines.append("  top features by global mean |SHAP| (contribution to the score, not cause)")
        for r in d["glob"].head(TOP_N).itertuples(index=False):
            lines.append(f"    {r.rank}. {r.feature:<22}{r.mean_abs_shap_mean_over_repeats:.4f}")
        lines.append("  sha8 " + "  ".join(f"{k.split('.')[0]} {v}" for k, v in d["sha8"].items()))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for name in ("cv-root", "features", "roles", "meta", "out-root"):
        ap.add_argument(f"--{name}", type=Path, default=None)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cv_root = Path(args.cv_root or cvs.CV_ROOT)
    out_root = Path(args.out_root or xai_root())
    features = Path(args.features or cvs.PROCESSED_ROOT / "features.csv")
    roles = Path(args.roles or cvs.PROCESSED_ROOT / "feature_roles.csv")
    meta = Path(args.meta or cvs.PROCESSED_ROOT / "features_meta.json")
    if not cvs.is_inside(out_root, xai_root()):
        print("error: --out-root must be inside data/processed/xai", file=sys.stderr)
        return 2
    try:
        frame, _ = cvs.load_inputs(features, roles, meta)
        input_hashes = cvs.input_hashes(features, roles, meta)
        done = [run_config(cv_root, out_root, frame, input_hashes, t, fs) for t, fs in CONFIGS]
    except InputProblem as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except (GuardFailure, LeakageError) as err:
        print(f"refused: {err}", file=sys.stderr)
        return 1
    print(render(done))
    bad = [f"{d['target']} {d['fs']}" for d in done if not d["meta"]["oof_reproduced"]]
    if bad:
        print(f"FAILED: the refit does not reproduce the official OOF within {OOF_TOL:g} "
              f"for {', '.join(bad)}", file=sys.stderr)  # fmt: skip
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
