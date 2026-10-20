"""Cross-fold stability of the SHAP explanation (task FT-45). Stability of the explanation across
refits, not evidence of cause. Reads only data/processed/xai/<target>/<set>/ (no refit, no labels):
Spearman between the 25 per-fold importance vectors (target mean rho >= 0.70), Kendall's W, top-1
and top-2 agreement and a per-clip check. Figure + FigureCanvasAgg only. Exit codes: 0 ok,
1 guard failure, 2 missing input or a path outside the xai folder.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv_splits as cvs
import matplotlib
import numpy as np
import pandas as pd
import scipy
import shap_figures as sf
import treeshap_heldout as tsh
from cv_splits import GuardFailure, InputProblem
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.stats import rankdata, spearmanr

TOOL = "explanation-stability-1"
CONFIGS = (("ge2", "A"), ("ge1", "A"), ("ge2", "B"))  # pre-registered; model is lgbm only
TARGET = 0.70
N_FOLDS = tsh.REPEATS * cvs.N_OUTER
CMAP = "viridis_r"  # rank 1 (largest mean |SHAP|) is the bright end
FOOTER = (
    "Rank of mean |SHAP| per held-out fold. Stability of the explanation, not of the disease "
    "signal; folds share most training data."
)
PAIRS_FILE, CLIPS_FILE, SUMMARY_FILE, FIG_FILE, META_FILE = (
    "stability_pairs.csv", "stability_clips.csv", "stability_summary.csv",
    "stability_ranks.png", "stability_meta.json",
)  # fmt: skip
SUMMARY_COLUMNS = [
    "mean_rho", "median_rho", "min_rho", "p05_rho", "share_ge_target", "mean_rho_within_repeat",
    "mean_rho_between_repeat", "kendall_w", "top1_agreement", "top2_agreement", "clip_rho_median",
    "clip_rho_p25", "clip_rho_p75", "n_clips_excluded", "target", "meets_target", "n_features",
    "n_folds",
]  # fmt: skip


# --------------------------------------------------------------------------- statistics


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho with average ranks for ties; NaN (never 0 or 1) if a vector is constant."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(spearmanr(a, b).statistic)


def kendall_w(vectors: np.ndarray) -> float:
    """Kendall's W of m raters (rows) over n items (columns), with the tie correction."""
    v = np.asarray(vectors, dtype=float)
    m, n = v.shape
    ranks = np.array([rankdata(row) for row in v])
    totals = ranks.sum(axis=0)
    s = float(((totals - totals.mean()) ** 2).sum())
    ties = 0.0
    for row in v:
        _, counts = np.unique(row, return_counts=True)
        ties += float((counts**3 - counts).sum())
    denominator = m**2 * (n**3 - n) - m * ties
    if denominator <= 0:
        raise GuardFailure("Kendall's W is undefined: every rater ties all items")
    return 12.0 * s / denominator


def meets_target(mean_rho: float) -> bool:
    return bool(mean_rho >= TARGET)


def fold_ids(by_fold: pd.DataFrame) -> list[tuple[int, int]]:
    return sorted(
        {(int(r), int(f)) for r, f in zip(by_fold["repeat"], by_fold["outer_fold"], strict=True)}
    )


def fold_matrix(
    by_fold: pd.DataFrame, order: Sequence[str]
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """(25, F): mean |SHAP| per (repeat, fold) row and feature, features in global-rank order."""
    ids = fold_ids(by_fold)
    if len(ids) != N_FOLDS or len(by_fold) != N_FOLDS * len(order):
        raise GuardFailure(f"expected {N_FOLDS} folds x {len(order)} features in the per-fold file")
    if by_fold.duplicated(["repeat", "outer_fold", "feature"]).any():
        raise GuardFailure("the per-fold file has a duplicated (repeat, fold, feature) row")
    table = by_fold.pivot(index=["repeat", "outer_fold"], columns="feature", values="mean_abs_shap")
    table = table.loc[ids, list(order)]
    if table.isna().any().any():
        raise GuardFailure("the per-fold file has a missing value")
    return ids, table.to_numpy(dtype=float)


def rank_text(rank: float) -> str:
    """A rank for a cell: 3, or 12\u00bd for a tie of two (a full "12.5" is wider than a cell)."""
    if abs(rank - round(rank)) < 1e-9:
        return str(int(round(rank)))
    if abs(2 * rank - round(2 * rank)) < 1e-9:
        return f"{int(rank)}\u00bd"
    return f"{rank:.1f}"


def fold_name(rf: tuple[int, int]) -> str:
    return f"r{rf[0]}f{rf[1]}"


def pair_table(ids: Sequence[tuple[int, int]], matrix: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, j in itertools.combinations(range(len(ids)), 2):
        rho = spearman(matrix[i], matrix[j])
        if np.isnan(rho):
            raise GuardFailure(f"a per-fold importance vector is constant ({fold_name(ids[i])})")
        rows.append({"fold_a": fold_name(ids[i]), "fold_b": fold_name(ids[j]),
                     "same_repeat": bool(ids[i][0] == ids[j][0]), "rho": rho})  # fmt: skip
    return pd.DataFrame(rows)


def fold_ranks(matrix: np.ndarray) -> np.ndarray:
    """Rank of each feature within each fold: 1 = largest mean |SHAP|, ties averaged."""
    return np.array([rankdata(-row) for row in matrix])


def top_agreement(
    matrix: np.ndarray, names: Sequence[str], glob: pd.DataFrame
) -> tuple[float, float]:
    """Share of folds whose rank-1 feature, and whose top-2 set, equal the global ones."""
    ranked = glob.sort_values(["rank", "feature"], kind="stable")["feature"].tolist()
    top1, top2 = ranked[0], set(ranked[:2])
    hit1 = hit2 = 0
    for row in matrix:
        order = sorted(range(len(names)), key=lambda i: (-row[i], names[i]))
        hit1 += names[order[0]] == top1
        hit2 += {names[order[0]], names[order[1]]} == top2
    return hit1 / len(matrix), hit2 / len(matrix)


def clip_table(values: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Per clip: mean Spearman rho over the repeat pairs of its signed SHAP vectors."""
    shap_cols = [f"shap_{c}" for c in cols]
    rows = []
    for ref, g in values.groupby("clip_ref", sort=True):
        g = g.sort_values("repeat")
        if list(g["repeat"]) != list(range(tsh.REPEATS)):
            raise GuardFailure("a clip is not present exactly once in every repeat")
        vectors = g[shap_cols].to_numpy(dtype=float)
        rhos = [
            spearman(vectors[i], vectors[j])
            for i, j in itertools.combinations(range(len(vectors)), 2)
        ]
        defined = [r for r in rhos if not np.isnan(r)]
        undefined_vector = any(np.ptp(v) == 0 for v in vectors)  # a constant vector: rho undefined
        rows.append({"clip_ref": ref, "mean_rho_over_repeats": (float(np.mean(defined)) if defined and not undefined_vector else np.nan),
                     "n_pairs_defined": len(defined)})  # fmt: skip
    return pd.DataFrame(rows)


def summarise(
    pairs: pd.DataFrame,
    matrix: np.ndarray,
    names: Sequence[str],
    glob: pd.DataFrame,
    clips: pd.DataFrame,
) -> dict[str, Any]:
    rho = pairs["rho"].to_numpy()
    within = pairs["same_repeat"].to_numpy(dtype=bool)
    top1, top2 = top_agreement(matrix, names, glob)
    clip_rho = clips["mean_rho_over_repeats"].dropna().to_numpy()
    mean_rho = float(rho.mean())
    return {
        "mean_rho": mean_rho, "median_rho": float(np.median(rho)), "min_rho": float(rho.min()),
        "p05_rho": float(np.percentile(rho, 5)), "share_ge_target": float((rho >= TARGET).mean()),
        "mean_rho_within_repeat": float(rho[within].mean()),
        "mean_rho_between_repeat": float(rho[~within].mean()),
        "kendall_w": kendall_w(matrix), "top1_agreement": top1, "top2_agreement": top2,
        "clip_rho_median": float(np.median(clip_rho)) if len(clip_rho) else float("nan"),
        "clip_rho_p25": float(np.percentile(clip_rho, 25)) if len(clip_rho) else float("nan"),
        "clip_rho_p75": float(np.percentile(clip_rho, 75)) if len(clip_rho) else float("nan"),
        "n_clips_excluded": int(clips["mean_rho_over_repeats"].isna().sum()),
        "target": TARGET, "meets_target": meets_target(mean_rho), "n_features": len(names),
        "n_folds": len(matrix),
    }  # fmt: skip


# --------------------------------------------------------------------------- figure


def fig_ranks(target: str, fs: str, ids: Sequence[tuple[int, int]], order: Sequence[str],
              matrix: np.ndarray, summary: dict[str, Any]) -> Figure:  # fmt: skip
    fig = Figure(figsize=sf.FIGSIZE, dpi=sf.DPI, layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ranks = fold_ranks(matrix).T  # rows = features in global-rank order, columns = the folds
    n_f, n_c = ranks.shape
    ax.imshow(ranks, cmap=CMAP, vmin=1, vmax=n_f, aspect="auto", interpolation="nearest")
    cmap = matplotlib.colormaps[CMAP]
    for i in range(n_f):
        for j in range(n_c):
            shade = cmap((ranks[i, j] - 1) / max(n_f - 1, 1))
            luminance = 0.299 * shade[0] + 0.587 * shade[1] + 0.114 * shade[2]
            ax.text(j, i, rank_text(float(ranks[i, j])), ha="center", va="center", fontsize=sf.FONT_SMALL,
                    color="black" if luminance > 0.5 else "white")  # fmt: skip
    for r in range(1, tsh.REPEATS):
        ax.axvline(r * cvs.N_OUTER - 0.5, color="white", linewidth=3)
    ax.set_xticks(range(n_c), [str(f) for _, f in ids], fontsize=sf.FONT_SMALL)
    ax.set_yticks(range(n_f), list(order), fontsize=sf.FONT_SMALL)
    ax.set_xlabel(
        "outer fold within each repeat (white lines separate the 5 repeats); "
        "cell = rank, 1 = largest mean |SHAP|",
        fontsize=sf.FONT_SMALL,
    )
    top = ax.secondary_xaxis("top")
    top.set_xticks([r * cvs.N_OUTER + (cvs.N_OUTER - 1) / 2 for r in range(tsh.REPEATS)],
                   [f"repeat {r}" for r in range(tsh.REPEATS)], fontsize=sf.FONT_SMALL)  # fmt: skip
    top.tick_params(length=0)
    verdict = "meets" if summary["meets_target"] else "does not meet"
    fig.suptitle(f"{sf.title_of(target, fs)}: feature rank by outer fold", fontsize=14)
    ax.set_title(
        f"mean Spearman rho {summary['mean_rho']:.2f} over {n_c * (n_c - 1) // 2} fold pairs "
        f"({verdict} the {TARGET:.2f} target); Kendall W {summary['kendall_w']:.2f}",
        fontsize=sf.FONT_SMALL, pad=36,
    )  # fmt: skip
    fig.supxlabel(FOOTER, fontsize=sf.FONT_SMALL, color="#333333")
    return fig


# --------------------------------------------------------------------------- run


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.17g", na_rep="").encode()


def run_config(cv_root: Path, xai: Path, target: str, fs: str) -> dict[str, Any]:
    cols = list(tsh.ch.FEATURE_SETS[fs])
    folder = xai / target / fs
    inputs = sf.read_inputs(folder, cols)
    official_path = cv_root / target / f"meta_{tsh.MODEL}_{fs}.json"
    if not official_path.is_file():
        raise InputProblem(f"missing input: official meta for {target} {fs}")
    official = json.loads(official_path.read_text())
    sf.check_meta(inputs["meta"], official, target, fs)
    sf.check_tables(inputs, cols, f"{target} {fs}")
    order = sf.feature_order(inputs["glob"])
    ids, matrix = fold_matrix(inputs["by_fold"], order)
    pairs = pair_table(ids, matrix)
    clips = clip_table(inputs["values"], cols)
    summary = summarise(pairs, matrix, order, inputs["glob"], clips)
    fig = fig_ranks(target, fs, ids, order, matrix, summary)
    out_dir = xai / "stability" / target / fs
    sf._atomic(
        out_dir / FIG_FILE, lambda tmp: fig.savefig(tmp, format="png", metadata={"Software": None})
    )
    files = {PAIRS_FILE: csv_bytes(pairs), CLIPS_FILE: csv_bytes(clips),
             SUMMARY_FILE: csv_bytes(pd.DataFrame([summary], columns=SUMMARY_COLUMNS))}  # fmt: skip
    for name, data in files.items():
        cvs._atomic(out_dir / name, data)
    sha8 = {n: cvs.sha8_of(d) for n, d in files.items()}
    sha8[FIG_FILE] = cvs.sha8_of((out_dir / FIG_FILE).read_bytes())
    meta = {
        "tool": TOOL, "target": target, "feature_set": fs, "model": tsh.MODEL, "columns": cols,
        "constants": {"target_mean_rho": TARGET, "n_folds": N_FOLDS, "n_pairs": len(pairs),
                      "fold_vector": "mean_abs_shap per feature for one (repeat, outer fold)",
                      "primary": "mean Spearman rho over all fold pairs (average ranks)",
                      "dpi": sf.DPI, "figsize": list(sf.FIGSIZE)},  # fmt: skip
        "inputs_sha8": {n: cvs.sha8_of((folder / n).read_bytes()) for n in sf.SHAP_INPUTS},
        "features_sha8": official["features_sha8"], "grid_sha8": official["grid_sha8"],
        "splits_sha8": official["splits_sha8"], "files_sha8": sha8,
        "versions": {"matplotlib": matplotlib.__version__, "numpy": np.__version__,
                     "scipy": scipy.__version__},  # fmt: skip
    }  # fmt: skip
    cvs._atomic(out_dir / META_FILE, (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode())
    sha8[META_FILE] = cvs.sha8_of((out_dir / META_FILE).read_bytes())
    return {
        "target": target,
        "fs": fs,
        "summary": summary,
        "sha8": sha8,
        "n_pairs": len(pairs),
        "n_clips": len(clips),
    }


def render(done: Sequence[dict[str, Any]]) -> str:
    lines = []
    for d in done:
        s = d["summary"]
        lines.append(f"== {d['target']} Set {d['fs']}: {s['n_folds']} folds, {s['n_features']} features, "
                     f"{d['n_pairs']} fold pairs, {d['n_clips']} clips")  # fmt: skip
        lines.append(f"  mean rho {s['mean_rho']:.4f} (target {s['target']:.2f}, meets {str(s['meets_target']).lower()})  "
                     f"median {s['median_rho']:.4f}  min {s['min_rho']:.4f}  p05 {s['p05_rho']:.4f}  "
                     f"share>=target {s['share_ge_target']:.3f}")  # fmt: skip
        lines.append(f"  within-repeat {s['mean_rho_within_repeat']:.4f}  between-repeat {s['mean_rho_between_repeat']:.4f}  "
                     f"Kendall W {s['kendall_w']:.4f}  top1 {s['top1_agreement']:.2f}  top2 {s['top2_agreement']:.2f}")  # fmt: skip
        lines.append(f"  per-clip rho median {s['clip_rho_median']:.4f} (p25 {s['clip_rho_p25']:.4f}, "
                     f"p75 {s['clip_rho_p75']:.4f})  clips excluded {s['n_clips_excluded']}")  # fmt: skip
        lines.append("  sha8 " + "  ".join(f"{k.split('.')[0]} {v}" for k, v in d["sha8"].items()))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for name in ("cv-root", "xai-root"):
        ap.add_argument(f"--{name}", type=Path, default=None)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cv_root = Path(args.cv_root or cvs.CV_ROOT)
    xai = Path(args.xai_root or tsh.xai_root())
    if not cvs.is_inside(xai, tsh.xai_root()):
        print("error: --xai-root must be inside data/processed/xai", file=sys.stderr)
        return 2
    try:
        done = [run_config(cv_root, xai, t, fs) for t, fs in CONFIGS]
    except InputProblem as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except GuardFailure as err:
        print(f"refused: {err}", file=sys.stderr)
        return 1
    print(render(done))
    return 0


if __name__ == "__main__":
    sys.exit(main())
