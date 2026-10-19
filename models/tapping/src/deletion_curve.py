"""Deletion curve for the SHAP explanations (task FT-44). A faithfulness check of the explanation,
not evidence of cause. Refits each official outer fold of lgbm, deletes features from the held-out
rows (value -> training-fold median) in SHAP order, reverse order and 100 random orders, and
compares the pooled AUROC curves. Labels only enter the AUROC. Figure + FigureCanvasAgg only.
Exit codes: 0 ok, 1 guard failure or OOF not reproduced, 2 missing input or a path outside xai.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv_harness as ch
import cv_splits as cvs
import lightgbm
import matplotlib
import numpy as np
import pandas as pd
import shap_figures as sf
import treeshap_heldout as tsh
from cv_splits import GuardFailure, InputProblem, LeakageError
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

TOOL = "deletion-curve-1"
CONFIGS = (("ge2", "A"), ("ge1", "A"), ("ge2", "B"))  # pre-registered; model is lgbm only
N_RANDOM = 100
RANDOM_SEED = 2026
ALPHA = 0.05
OOF_TOL = 1e-9
K0_TOL = 1e-9
HALF_TOL = 0.05
FOOTER = (
    "SHAP order = per-clip |SHAP| descending; deleted value = training-fold median. "
    "Descriptive; not a cause."
)
CURVES_FILE, AREAS_FILE, SUMMARY_FILE, FIG_FILE, META_FILE = (
    "deletion_curves.csv", "deletion_random_areas.csv", "deletion_summary.csv",
    "deletion_curve.png", "deletion_meta.json",
)  # fmt: skip
CURVE_COLUMNS = [
    "order",
    "repeat",
    "k",
    "auroc",
    "mean_abs_margin_change",
    "auroc_p05",
    "auroc_p95",
]
SUMMARY_COLUMNS = [
    "area_shap", "area_reverse", "area_random_mean", "area_random_p05", "area_random_p95",
    "p_value", "falls_faster", "n_random",
]  # fmt: skip


# --------------------------------------------------------------------------- pure pieces


def shap_order(abs_shap: np.ndarray, names: Sequence[str]) -> np.ndarray:
    """Per row the feature indices by |SHAP| descending; ties broken by feature name."""
    by_name = np.argsort(np.asarray(names), kind="stable")
    sorted_cols = np.argsort(-np.asarray(abs_shap)[:, by_name], axis=1, kind="stable")
    return by_name[sorted_cols]


def random_orders(
    config_index: int, draw: int, repeats: int, n_clips: int, n_features: int
) -> np.ndarray:
    """One uniform random permutation per (repeat, clip), fixed by the seed, config and draw."""
    rng = np.random.default_rng([RANDOM_SEED, config_index, draw])
    return np.argsort(rng.random((repeats, n_clips, n_features)), axis=2)


def train_medians(x_train: np.ndarray) -> np.ndarray:
    """The deleted value of every feature: its median over the outer TRAINING rows."""
    med = np.nanmedian(x_train, axis=0)
    if np.isnan(med).any():
        raise GuardFailure("a feature has no value in the training rows: no median to delete to")
    return np.asarray(med, dtype=float)


def delete_masks(order: np.ndarray, n_features: int) -> np.ndarray:
    """masks[..., j, f] is True when feature f is among the first k=j deleted: (..., F+1, F)."""
    position = np.argsort(order, axis=-1)  # rank of every feature in the deletion order
    k = np.arange(n_features + 1)
    return position[..., None, :] < k[:, None]


def masked(x_test: np.ndarray, medians: np.ndarray, order: np.ndarray) -> np.ndarray:
    """(F+1, rows, F) copies of the test rows with the first k features replaced (k = 0..F)."""
    masks = delete_masks(order, x_test.shape[1]).transpose(1, 0, 2)  # (F+1, rows, F)
    return np.where(masks, medians[None, None, :], x_test[None, :, :])


def curve_area(y: np.ndarray) -> float:
    """Trapezoid area over x = k/F in [0, 1] for a curve sampled at k = 0..F."""
    y = np.asarray(y, dtype=float)
    return float(np.sum((y[:-1] + y[1:]) / 2.0) / (len(y) - 1))


def empirical_p(shap_area: float, random_areas: np.ndarray) -> float:
    """One-sided: (1 + #{random area <= shap area}) / (N + 1)."""
    random_areas = np.asarray(random_areas, dtype=float)
    return float((1 + int((random_areas <= shap_area).sum())) / (len(random_areas) + 1))


# --------------------------------------------------------------------------- the curves


def shap_abs_by_repeat(values: pd.DataFrame, frame: pd.DataFrame, cols: Sequence[str],
                       splits: pd.DataFrame) -> np.ndarray:  # fmt: skip
    """|SHAP| as (repeats, clips, features), clips in the ref-sorted frame order."""
    refs, groups = frame["ref"].to_numpy(), frame["participant"].to_numpy()
    out = np.empty((tsh.REPEATS, len(refs), len(cols)))
    for repeat in range(tsh.REPEATS):
        sub = values[values["repeat"] == repeat].set_index("clip_ref").reindex(refs)
        if sub[[f"shap_{c}" for c in cols]].isna().any().any():
            raise GuardFailure(f"repeat {repeat}: SHAP rows are missing for some clips")
        if (sub["outer_fold"].to_numpy() != ch.membership(groups, splits, repeat)).any():
            raise GuardFailure(f"repeat {repeat}: the SHAP rows come from other folds")
        out[repeat] = np.abs(sub[[f"shap_{c}" for c in cols]].to_numpy(dtype=float))
    return out


def compute_curves(
    frame: pd.DataFrame, splits: pd.DataFrame, oof: pd.DataFrame,
    best: dict[tuple[int, int], dict[str, Any]], abs_shap: np.ndarray, target: str, fs: str,
    config_index: int, runs_auroc: Sequence[float], n_random: int = N_RANDOM,
) -> dict[str, Any]:  # fmt: skip
    frame = cvs.canonical(frame)
    X, cols = ch.design_matrix(frame, fs)
    y = cvs.target_labels(frame, target)  # labels: only for the AUROC below
    groups, refs = frame["participant"].to_numpy(), frame["ref"].to_numpy()
    n, n_f = X.shape
    official = {r: g.set_index("ref") for r, g in oof.groupby("repeat")}
    shap_perm = np.stack([shap_order(abs_shap[r], cols) for r in range(tsh.REPEATS)])
    rand_perm = np.stack(
        [random_orders(config_index, d, tsh.REPEATS, n, n_f) for d in range(n_random)]
    )
    proba = {"shap": np.full((tsh.REPEATS, n_f + 1, n), np.nan),
             "reverse": np.full((tsh.REPEATS, n_f + 1, n), np.nan),
             "random": np.full((n_random, tsh.REPEATS, n_f + 1, n), np.nan)}  # fmt: skip
    margin = {k: np.full_like(v, np.nan) for k, v in proba.items()}
    max_oof = 0.0
    for repeat in range(tsh.REPEATS):
        member = ch.membership(groups, splits, repeat)
        saved = official[repeat].reindex(refs)
        if saved["proba"].isna().any() or (saved["outer_fold"].to_numpy() != member).any():
            raise GuardFailure(f"repeat {repeat}: the saved OOF does not match the stored folds")
        seen = np.zeros(n, dtype=int)
        for fold in range(cvs.N_OUTER):
            train, test = np.flatnonzero(member != fold), np.flatnonzero(member == fold)
            if set(train.tolist()) & set(test.tolist()):
                raise LeakageError(f"repeat {repeat} fold {fold}: train and test rows overlap")
            cvs.assert_disjoint(
                groups[train], groups[test], f"deletion repeat {repeat} fold {fold}"
            )
            est = ch.refit_outer(
                frame, tsh.MODEL, best[(repeat, fold)], train, fs, target, test_positions=test
            )
            medians = train_medians(X[train])
            x_test = X[test]
            max_oof = max(max_oof, float(np.max(np.abs(
                est.predict_proba(x_test)[:, 1] - saved["proba"].to_numpy()[test]))))  # fmt: skip
            orders = {"shap": shap_perm[repeat][test][None], "reverse": shap_perm[repeat][test][:, ::-1][None],
                      "random": rand_perm[:, repeat][:, test]}  # fmt: skip
            for key, order in orders.items():
                batch = np.stack([masked(x_test, medians, o) for o in order])  # (D, F+1, rows, F)
                flat = batch.reshape(-1, n_f)
                p = est.predict_proba(flat)[:, 1].reshape(batch.shape[:3])
                m = np.asarray(est.predict(flat, raw_score=True), dtype=float).reshape(
                    batch.shape[:3]
                )
                if key == "random":
                    proba[key][:, repeat][:, :, test], margin[key][:, repeat][:, :, test] = p, m
                else:
                    proba[key][repeat][:, test], margin[key][repeat][:, test] = p[0], m[0]
            seen[test] += 1
        if not (seen == 1).all():
            raise GuardFailure(f"repeat {repeat}: a clip was not evaluated exactly once")
    for key in proba:
        if np.isnan(proba[key]).any() or np.isnan(margin[key]).any():
            raise GuardFailure(f"{key}: a (repeat, k, clip) cell was never evaluated")
    auc = {key: _auc(y, p) for key, p in proba.items()}
    change = {key: np.abs(m - m[..., :1, :]).mean(axis=-1) for key, m in margin.items()}
    return {"auc": auc, "change": change, "cols": cols, "max_oof": max_oof, "n_features": n_f,
            "runs_auroc": list(runs_auroc), "n_random": n_random}  # fmt: skip


def _auc(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Pooled AUROC (the harness' own) for every leading index of p[..., clips]."""
    flat = p.reshape(-1, p.shape[-1])
    return np.array([ch.auroc(y, row) for row in flat]).reshape(p.shape[:-1])


def check_k0(auc: dict[str, np.ndarray], runs_auroc: Sequence[float]) -> float:
    """k = 0 is the unmasked model: every order must equal the official pooled AUROC."""
    worst = 0.0
    for a in auc.values():
        k0 = a[..., 0]  # (repeats,) or (draws, repeats)
        diff = float(np.max(np.abs(k0 - np.asarray(runs_auroc, dtype=float))))
        worst = max(worst, diff)
    if worst > K0_TOL:
        raise GuardFailure(f"the k=0 AUROC differs from the official runs file by {worst:.3g}")
    return worst


def summarise(res: dict[str, Any]) -> dict[str, Any]:
    auc, f = res["auc"], res["n_features"]
    mean_shap, mean_rev = auc["shap"].mean(axis=0), auc["reverse"].mean(axis=0)
    random_curves = auc["random"].mean(axis=1)  # (draws, F+1): mean over repeats
    areas = np.array([curve_area(c) for c in random_curves])
    area_shap = curve_area(mean_shap)
    p = empirical_p(area_shap, areas)
    summary = {
        "area_shap": area_shap, "area_reverse": curve_area(mean_rev),
        "area_random_mean": float(areas.mean()),
        "area_random_p05": float(np.percentile(areas, 5)),
        "area_random_p95": float(np.percentile(areas, 95)),
        "p_value": p, "falls_faster": bool(p <= ALPHA), "n_random": len(areas),
    }  # fmt: skip
    curves = {"shap": mean_shap, "reverse": mean_rev, "random": random_curves.mean(axis=0),
              "band_low": np.percentile(random_curves, 5, axis=0),
              "band_high": np.percentile(random_curves, 95, axis=0)}  # fmt: skip
    return {"summary": summary, "areas": areas, "curves": curves, "k": np.arange(f + 1)}


def curves_table(res: dict[str, Any], sm: dict[str, Any]) -> pd.DataFrame:
    auc, change = res["auc"], res["change"]
    k, rows = sm["k"], []
    random_by_repeat = auc["random"].mean(axis=0)  # (repeats, F+1)
    random_change = change["random"].mean(axis=0)
    parts = {"shap": (auc["shap"], change["shap"]), "reverse": (auc["reverse"], change["reverse"]),
             "random_mean": (random_by_repeat, random_change)}  # fmt: skip
    for name, (a, c) in parts.items():
        for r in range(tsh.REPEATS):
            rows += [{"order": name, "repeat": r, "k": int(j), "auroc": a[r, j],
                      "mean_abs_margin_change": c[r, j]} for j in k]  # fmt: skip
        mean_c = c.mean(axis=0)
        for j in k:
            row = {"order": name, "repeat": "mean", "k": int(j), "auroc": a.mean(axis=0)[j],
                   "mean_abs_margin_change": mean_c[j]}  # fmt: skip
            if name == "random_mean":
                row.update(
                    auroc_p05=sm["curves"]["band_low"][j], auroc_p95=sm["curves"]["band_high"][j]
                )
            rows.append(row)
    return pd.DataFrame(rows, columns=CURVE_COLUMNS)


# --------------------------------------------------------------------------- figure


def fig_curve(target: str, fs: str, sm: dict[str, Any]) -> Figure:
    fig = Figure(figsize=sf.FIGSIZE, dpi=sf.DPI, layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    k, c, s = sm["k"], sm["curves"], sm["summary"]
    ax.fill_between(k, c["band_low"], c["band_high"], color="#bbbbbb", alpha=0.6, linewidth=0,
                    label="random order, 5-95% band")  # fmt: skip
    ax.plot(k, c["random"], color=sf.TOTAL, linewidth=2, label="random order, mean")
    ax.plot(k, c["reverse"], color=sf.DOWN, linewidth=2.4, marker="o", label="reverse SHAP order")
    ax.plot(k, c["shap"], color=sf.UP, linewidth=2.4, marker="o", label="SHAP order")
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.set_xticks(k)
    ax.set_xlim(0, k[-1])
    ax.tick_params(labelsize=sf.FONT_SMALL)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("pooled out-of-fold AUROC (mean of 5 repeats)", fontsize=sf.FONT_SMALL)
    verdict = "yes" if s["falls_faster"] else "no"
    ax.set_xlabel(
        "number of features deleted\n"
        f"area: SHAP {s['area_shap']:.3f}, random mean {s['area_random_mean']:.3f} "
        f"(5-95%: {s['area_random_p05']:.3f}-{s['area_random_p95']:.3f}); one-sided p = "
        f"{s['p_value']:.3f}; SHAP order below the random band: {verdict}",
        fontsize=sf.FONT_SMALL,
    )
    top = max(float(np.max(c["shap"])), float(np.max(c["reverse"])), float(np.max(c["band_high"])))
    ax.set_ylim(0.36, top + 0.02)  # room below the chance line for the legend
    ax.text(
        0.1,
        0.505,
        "chance (0.5)",
        ha="left",
        va="bottom",
        fontsize=sf.FONT_SMALL,
        color="#555555",
    )
    ax.legend(loc="lower left", fontsize=sf.FONT_SMALL, frameon=False)
    fig.suptitle(
        f"{sf.title_of(target, fs)}: deleting features in SHAP order vs random order", fontsize=14
    )
    fig.supxlabel(FOOTER, fontsize=sf.FONT_SMALL, color="#333333")
    return fig


# --------------------------------------------------------------------------- run


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.17g", na_rep="").encode()


def run_config(
    cv_root: Path, xai: Path, frame: pd.DataFrame, input_hashes: dict[str, str], config_index: int,
    target: str, fs: str, n_random: int = N_RANDOM,
) -> dict[str, Any]:  # fmt: skip
    folder = cv_root / target
    paths = {"meta": folder / f"meta_{tsh.MODEL}_{fs}.json", "oof": folder / f"oof_{tsh.MODEL}_{fs}.csv",
             "best": folder / f"best_params_{tsh.MODEL}_{fs}.csv", "splits": folder / cvs.OUT_SPLITS,
             "runs": folder / f"runs_{tsh.MODEL}_{fs}.csv"}  # fmt: skip
    for name, path in paths.items():
        if not path.is_file():
            raise InputProblem(f"missing input: {name} file for {target} {tsh.MODEL} {fs}")
    official = json.loads(paths["meta"].read_text())
    splits = cvs.load_splits(paths["splits"], cvs.canonical(frame))
    hashes = {**input_hashes, "splits": cvs.sha8_of(paths["splits"].read_bytes())}
    tsh.check_official(official, frame, splits, target, fs, hashes)
    cols = list(ch.FEATURE_SETS[fs])
    inputs = sf.read_inputs(xai / target / fs, cols)
    sf.check_meta(inputs["meta"], official, target, fs)
    sf.check_tables(inputs, cols, f"{target} {fs}")
    runs = pd.read_csv(paths["runs"]).sort_values("repeat")
    if list(runs["repeat"]) != list(range(tsh.REPEATS)):
        raise GuardFailure(f"{target} {fs}: the runs file does not hold repeats 0-4 once each")
    canon = cvs.canonical(frame)
    abs_shap = shap_abs_by_repeat(inputs["values"], canon, cols, splits)
    res = compute_curves(
        frame, splits, ch.read_oof(paths["oof"]), tsh.read_best_params(paths["best"], ch.grid_for(tsh.MODEL)),
        abs_shap, target, fs, config_index, runs["auroc"].to_numpy(dtype=float), n_random,
    )  # fmt: skip
    if res["max_oof"] > OOF_TOL:
        raise GuardFailure(
            f"{target} {fs}: the refit does not reproduce the official OOF ({res['max_oof']:.3g})"
        )
    k0 = check_k0(res["auc"], res["runs_auroc"])
    sm = summarise(res)
    table = curves_table(res, sm)
    summary = pd.DataFrame([sm["summary"]], columns=SUMMARY_COLUMNS)
    areas = pd.DataFrame({"draw": np.arange(len(sm["areas"])), "area": sm["areas"]})
    fig = fig_curve(target, fs, sm)
    out_dir = xai / "deletion" / target / fs
    sf._atomic(
        out_dir / FIG_FILE, lambda tmp: fig.savefig(tmp, format="png", metadata={"Software": None})
    )
    files = {
        CURVES_FILE: csv_bytes(table),
        AREAS_FILE: csv_bytes(areas),
        SUMMARY_FILE: csv_bytes(summary),
    }
    for name, data in files.items():
        cvs._atomic(out_dir / name, data)
    sha8 = {n: cvs.sha8_of(d) for n, d in files.items()}
    sha8[FIG_FILE] = cvs.sha8_of((out_dir / FIG_FILE).read_bytes())
    full = {key: float(res["auc"][key][..., -1].mean()) for key in res["auc"]}
    meta = {
        "tool": TOOL, "target": target, "feature_set": fs, "model": tsh.MODEL, "columns": cols,
        "constants": {"n_random": n_random, "random_seed": RANDOM_SEED, "config_index": config_index,
                      "delete_value": "median of the outer training rows (nanmedian)", "alpha": ALPHA,
                      "oof_tolerance": OOF_TOL, "k0_tolerance": K0_TOL, "dpi": sf.DPI,
                      "figsize": list(sf.FIGSIZE)},  # fmt: skip
        "inputs_sha8": {**hashes, "oof": cvs.sha8_of(paths["oof"].read_bytes()),
                        "best_params": cvs.sha8_of(paths["best"].read_bytes()),
                        "runs": cvs.sha8_of(paths["runs"].read_bytes()),
                        **{f"xai_{n}": cvs.sha8_of((xai / target / fs / n).read_bytes()) for n in sf.SHAP_INPUTS}},
        "guards": {"max_abs_diff_vs_official_oof": res["max_oof"], "k0_max_diff_vs_runs": k0,
                   "k_full_auroc_mean": full,
                   "k_full_within_0p05_of_0p5": {key: abs(v - 0.5) <= HALF_TOL for key, v in full.items()}},
        "features_sha8": official["features_sha8"], "grid_sha8": official["grid_sha8"],
        "splits_sha8": official["splits_sha8"], "files_sha8": sha8,
        "versions": {"lightgbm": lightgbm.__version__, "matplotlib": matplotlib.__version__,
                     "numpy": np.__version__},  # fmt: skip
    }  # fmt: skip
    cvs._atomic(out_dir / META_FILE, (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode())
    sha8[META_FILE] = cvs.sha8_of((out_dir / META_FILE).read_bytes())
    return {"target": target, "fs": fs, "summary": sm["summary"], "curves": sm["curves"],
            "meta": meta, "sha8": sha8, "k": sm["k"], "n_features": res["n_features"]}  # fmt: skip


def render(done: Sequence[dict[str, Any]]) -> str:
    lines = []
    for d in done:
        s, g, c = d["summary"], d["meta"]["guards"], d["curves"]
        lines.append(
            f"== {d['target']} Set {d['fs']}: {d['n_features']} features, {s['n_random']} random draws"
        )
        lines.append(f"  area shap {s['area_shap']:.4f}  reverse {s['area_reverse']:.4f}  random mean "
                     f"{s['area_random_mean']:.4f} (5-95%: {s['area_random_p05']:.4f}-{s['area_random_p95']:.4f})")  # fmt: skip
        lines.append(f"  p {s['p_value']:.4f}  falls_faster {str(s['falls_faster']).lower()}")
        lines.append(f"  AUROC k=0 {c['shap'][0]:.4f}  k=1 shap {c['shap'][1]:.4f} reverse {c['reverse'][1]:.4f} "
                     f"random {c['random'][1]:.4f}  k=F shap {c['shap'][-1]:.4f} random {c['random'][-1]:.4f}")  # fmt: skip
        lines.append(f"  guards: k0 max diff {g['k0_max_diff_vs_runs']:.3g}  OOF max diff "
                     f"{g['max_abs_diff_vs_official_oof']:.3g}")  # fmt: skip
        lines.append("  sha8 " + "  ".join(f"{k.split('.')[0]} {v}" for k, v in d["sha8"].items()))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for name in ("cv-root", "features", "roles", "meta", "xai-root"):
        ap.add_argument(f"--{name}", type=Path, default=None)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cv_root = Path(args.cv_root or cvs.CV_ROOT)
    xai = Path(args.xai_root or tsh.xai_root())
    features = Path(args.features or cvs.PROCESSED_ROOT / "features.csv")
    roles = Path(args.roles or cvs.PROCESSED_ROOT / "feature_roles.csv")
    meta = Path(args.meta or cvs.PROCESSED_ROOT / "features_meta.json")
    if not cvs.is_inside(xai, tsh.xai_root()):
        print("error: --xai-root must be inside data/processed/xai", file=sys.stderr)
        return 2
    try:
        frame, _ = cvs.load_inputs(features, roles, meta)
        hashes = cvs.input_hashes(features, roles, meta)
        done = [
            run_config(cv_root, xai, frame, hashes, i, t, fs) for i, (t, fs) in enumerate(CONFIGS)
        ]
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
