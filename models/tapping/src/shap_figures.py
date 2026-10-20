"""SHAP figures (task FT-43): global importance, summary dots and three illustrative clip waterfalls
per registered config, from data/processed/xai (treeshap_heldout.py). SHAP is the contribution to
the model's score, not a cause. Clips are chosen by model score only. Figure + FigureCanvasAgg
only; PNG bytes are deterministic. Exit codes: 0 ok, 1 guard failure, 2 missing input or a
path outside the xai folder.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv_splits as cvs
import matplotlib
import numpy as np
import pandas as pd
import treeshap_heldout as tsh
from cv_splits import GuardFailure, InputProblem
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.text import Text
from scipy.stats import rankdata

TOOL = "shap-figures-1"
JITTER_SEED = 2026
JITTER_HALF_WIDTH = 0.3
DPI = 200
FIGSIZE = (11.5, 6.4)
CMAP = "viridis"  # sequential and colour-blind safe
NAN_GREY = (0.6, 0.6, 0.6, 1.0)
KINDS = ("highest", "lowest", "nearest_half")
KIND_TEXT = {
    "highest": "highest model score",
    "lowest": "lowest model score",
    "nearest_half": "score nearest 0.5",
}
FOOTER = (
    "SHAP = contribution to the model's score, not a cause. "
    "Illustrative clips are chosen by model score, not representative."
)
BAR = "#3b5b92"
UP, DOWN, TOTAL = "#c4622d", "#3b6ea5", "#555555"
FONT, FONT_SMALL = 12, 11
FIG_GLOBAL, FIG_DOTS = "global_importance.png", "summary_dots.png"
META_FILE = "figures_meta.json"
SHAP_INPUTS = (tsh.SHAP_FILE, tsh.FOLD_FILE, tsh.GLOBAL_FILE, tsh.META_FILE)
FIXED_COLUMNS = ["repeat", "outer_fold", "clip_ref", "base_value", "raw_margin", "proba"]


# --------------------------------------------------------------------------- data preparation


def title_of(target: str, fs: str) -> str:
    return f"{target}, Set {fs}, LightGBM"


def read_inputs(folder: Path, cols: Sequence[str]) -> dict[str, Any]:
    for name in SHAP_INPUTS:
        if not (folder / name).is_file():
            raise InputProblem(f"missing input: {name}")
    wanted = FIXED_COLUMNS + [f"shap_{c}" for c in cols] + [f"value_{c}" for c in cols]
    values = pd.read_csv(folder / tsh.SHAP_FILE, usecols=wanted, dtype={"clip_ref": str})
    # only the named columns are read: anything else in the file never reaches the figures
    return {
        "values": values,
        "by_fold": pd.read_csv(folder / tsh.FOLD_FILE),
        "glob": pd.read_csv(folder / tsh.GLOBAL_FILE),
        "meta": json.loads((folder / tsh.META_FILE).read_text()),
    }


def check_meta(meta: dict[str, Any], official: dict[str, Any], target: str, fs: str) -> None:
    where = f"{target} {fs}"
    if meta.get("tool") != tsh.TOOL or (meta.get("target"), meta.get("feature_set")) != (
        target,
        fs,
    ):
        raise GuardFailure(f"{where}: shap_meta is not the expected treeshap_heldout output")
    if meta.get("oof_reproduced") is not True:
        raise GuardFailure(f"{where}: the SHAP refit did not reproduce the official OOF")
    for key in ("features_sha8", "grid_sha8", "splits_sha8"):
        if meta.get(key) != official.get(key):
            raise GuardFailure(f"{where}: {key} differs from the official run")


def check_tables(inputs: dict[str, Any], cols: Sequence[str], where: str) -> None:
    counts = inputs["meta"].get("row_counts", {})
    sizes = {tsh.SHAP_FILE: inputs["values"], tsh.FOLD_FILE: inputs["by_fold"],
             tsh.GLOBAL_FILE: inputs["glob"]}  # fmt: skip
    for name, frame in sizes.items():
        if counts.get(name) != len(frame):
            raise GuardFailure(f"{where}: {name} has a different row count than its meta")
    if sorted(inputs["glob"]["feature"]) != sorted(cols) or inputs["meta"].get("columns") != list(
        cols
    ):
        raise GuardFailure(f"{where}: features differ from the registered set")


def feature_order(glob: pd.DataFrame) -> list[str]:
    return list(glob.sort_values(["rank", "feature"], kind="stable")["feature"])


def per_clip_means(values: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """One row per clip: mean over the repeats of SHAP, base, margin and proba; feature values."""
    if (values.groupby("clip_ref")["repeat"].nunique() != tsh.REPEATS).any():
        raise GuardFailure("a clip is not present in every repeat")
    if values.duplicated(["repeat", "clip_ref"]).any():
        raise GuardFailure("a clip appears twice in a repeat")
    groups = values.groupby("clip_ref", sort=True)
    means = groups[["base_value", "raw_margin", "proba", *[f"shap_{c}" for c in cols]]].mean()
    value_cols = [f"value_{c}" for c in cols]
    first = groups[value_cols].first()  # skips NaN, so check consistency separately
    spread = groups[value_cols].nunique(dropna=True)
    nan_share = groups[value_cols].apply(lambda g: g.isna().mean())
    if (spread > 1).any().any() or ((nan_share > 0) & (nan_share < 1)).any().any():
        raise GuardFailure("a feature value differs between the repeats of one clip")
    first = first.where(nan_share == 0)  # all-NaN stays NaN
    return pd.concat([means, first], axis=1).reset_index()


def whiskers(by_fold: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Per repeat: mean of the fold values weighted by n_test_rows; then min and max over repeats."""
    rows = []
    for c in cols:
        sub = by_fold[by_fold["feature"] == c]
        per_repeat = [
            float(np.average(g["mean_abs_shap"], weights=g["n_test_rows"]))
            for _, g in sub.groupby("repeat", sort=True)
        ]
        if len(per_repeat) != tsh.REPEATS:
            raise GuardFailure(f"{c}: expected {tsh.REPEATS} repeats in the per-fold importance")
        rows.append({"feature": c, "low": min(per_repeat), "high": max(per_repeat),
                     "mean": float(np.mean(per_repeat))})  # fmt: skip
    return pd.DataFrame(rows)


def check_whiskers(glob: pd.DataFrame, wh: pd.DataFrame) -> None:
    merged = glob.merge(wh, on="feature")
    inside = (merged["mean_abs_shap_mean_over_repeats"] >= merged["low"] - 1e-9) & (
        merged["mean_abs_shap_mean_over_repeats"] <= merged["high"] + 1e-9
    )
    close = (merged["mean_abs_shap_mean_over_repeats"] - merged["mean"]).abs() < 1e-9
    if len(merged) != len(glob) or not inside.all() or not close.all():
        raise GuardFailure("global importance disagrees with the per-fold importance")


def percentile_ranks(v: np.ndarray) -> np.ndarray:
    """Percentile rank in [0, 1] among the non-missing values (ties averaged); NaN stays NaN."""
    v = np.asarray(v, dtype=float)
    out = np.full(len(v), np.nan)
    ok = ~np.isnan(v)
    n = int(ok.sum())
    if n == 1:
        out[ok] = 0.5
    elif n > 1:
        out[ok] = (rankdata(v[ok], method="average") - 1.0) / (n - 1.0)
    return out


def dot_colours(ranks: np.ndarray) -> np.ndarray:
    rgba = np.empty((len(ranks), 4))
    nan = np.isnan(ranks)
    rgba[~nan] = colormaps[CMAP](ranks[~nan])
    rgba[nan] = NAN_GREY  # missing is grey, never a rank of 0 and never dropped
    return rgba


def dot_layout(clips: pd.DataFrame, order: Sequence[str]) -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(JITTER_SEED)
    out = {}
    for row, c in enumerate(order):
        jitter = rng.uniform(-JITTER_HALF_WIDTH, JITTER_HALF_WIDTH, len(clips))
        ranks = percentile_ranks(clips[f"value_{c}"].to_numpy())
        out[c] = {"x": clips[f"shap_{c}"].to_numpy(), "y": row + jitter, "rank": ranks,
                  "rgba": dot_colours(ranks)}  # fmt: skip
    return out


def select_clips(clips: pd.DataFrame) -> dict[str, str]:
    """By mean model score only; ties go to the smallest clip_ref."""
    frame = clips[["clip_ref", "proba"]].assign(half=(clips["proba"] - 0.5).abs())
    pick = {
        "highest": frame.sort_values(["proba", "clip_ref"], ascending=[False, True]),
        "lowest": frame.sort_values(["proba", "clip_ref"], ascending=[True, True]),
        "nearest_half": frame.sort_values(["half", "clip_ref"], ascending=[True, True]),
    }
    return {kind: str(pick[kind]["clip_ref"].iloc[0]) for kind in KINDS}


def waterfall(clips: pd.DataFrame, ref: str, cols: Sequence[str]) -> dict[str, Any]:
    row = clips[clips["clip_ref"] == ref].iloc[0]
    bars = [(c, float(row[f"shap_{c}"]), float(row[f"value_{c}"])) for c in cols]
    bars.sort(key=lambda b: (-abs(b[1]), b[0]))
    base, final = float(row["base_value"]), float(row["raw_margin"])
    if abs(base + sum(b[1] for b in bars) - final) > 1e-9:
        raise GuardFailure("waterfall: base value plus contributions differs from the final margin")
    return {"ref": ref, "base": base, "bars": bars, "final": final, "proba": float(row["proba"])}


# --------------------------------------------------------------------------- drawing


def new_figure() -> tuple[Figure, Any]:
    fig = Figure(figsize=FIGSIZE, dpi=DPI, layout="constrained")
    FigureCanvasAgg(fig)
    return fig, fig.add_subplot(111)


def finish(fig: Figure, title: str) -> Figure:
    fig.suptitle(title, fontsize=14)
    fig.supxlabel(FOOTER, fontsize=FONT_SMALL, color="#333333")
    return fig


def style(ax: Any) -> None:
    ax.tick_params(labelsize=FONT_SMALL)
    ax.spines[["top", "right"]].set_visible(False)


def fig_global(glob: pd.DataFrame, wh: pd.DataFrame, order: Sequence[str], title: str) -> Figure:
    fig, ax = new_figure()
    value = glob.set_index("feature")["mean_abs_shap_mean_over_repeats"].loc[list(order)]
    w = wh.set_index("feature").loc[list(order)]
    y = np.arange(len(order))
    ax.barh(y, value.to_numpy(), color=BAR, height=0.62)
    ax.errorbar(value.to_numpy(), y, xerr=[(value - w["low"]).to_numpy(), (w["high"] - value).to_numpy()],
                fmt="none", ecolor="#222222", elinewidth=1.0, capsize=3)  # fmt: skip
    top = float(w["high"].max())
    for yi, v, hi in zip(y, value.to_numpy(), w["high"].to_numpy(), strict=True):
        ax.text(hi + 0.012 * top, yi, f"{v:.3f}", va="center", fontsize=FONT_SMALL)
    ax.set_xlim(0, top * 1.14)
    ax.set_yticks(y, list(order), fontsize=FONT_SMALL)
    ax.invert_yaxis()
    ax.set_xlabel("mean |SHAP| (log-odds); bar = mean over 5 repeats, whisker = range over "
                  "5 repeats, not a confidence interval", fontsize=FONT_SMALL)  # fmt: skip
    style(ax)
    return finish(fig, f"{title}: global importance")


def fig_dots(dots: dict[str, dict[str, np.ndarray]], order: Sequence[str], title: str) -> Figure:
    fig, ax = new_figure()
    any_nan = False
    for c in order:
        d = dots[c]
        any_nan |= bool(np.isnan(d["rank"]).any())
        ax.scatter(d["x"], d["y"], c=d["rgba"], s=13, linewidths=0, zorder=3)
    ax.axvline(0, color="#777777", linewidth=0.8, zorder=1)
    ax.set_yticks(range(len(order)), list(order), fontsize=FONT_SMALL)
    ax.set_ylim(len(order) - 0.4, -0.6)
    ax.set_xlabel("SHAP value (log-odds); positive = pushes toward the higher-severity class "
                  "of this target", fontsize=FONT_SMALL)  # fmt: skip
    style(ax)
    bar = fig.colorbar(ScalarMappable(Normalize(0, 1), cmap=CMAP), ax=ax, pad=0.015, fraction=0.03)
    bar.set_ticks([0, 1], labels=["low", "high"], fontsize=FONT_SMALL)
    bar.set_label("feature value (low to high)", fontsize=FONT_SMALL)
    if any_nan:
        ax.scatter([], [], c=[NAN_GREY], s=30, label="grey = value missing")
        ax.legend(loc="lower right", fontsize=FONT_SMALL, frameon=False)
    return finish(fig, f"{title}: one dot per clip (mean over 5 repeats)")


def delta_text(delta: float) -> str:
    """A signed contribution to two decimals; a value that rounds to zero carries no sign."""
    text = f"{delta:+.2f}"
    return "0.00" if text in ("+0.00", "-0.00") else text


def plain_text(x: float) -> str:
    text = f"{x:.2f}"
    return "0.00" if text == "-0.00" else text


def proba_text(p: float) -> str:
    """Two decimals, except at the extremes where 0.00 or 1.00 would hide how extreme it is."""
    if p < 0.005:
        return "<0.01"
    return ">0.99" if p > 0.995 else f"{p:.2f}"


def fig_waterfall(wf: dict[str, Any], kind: str, title: str) -> Figure:
    fig, ax = new_figure()
    labels = ["base value"]
    starts, ends, colours = [0.0], [wf["base"]], [TOTAL]
    running = wf["base"]
    for name, shap_value, feature_value in wf["bars"]:
        shown = "missing" if np.isnan(feature_value) else f"{feature_value:.3g}"
        labels.append(f"{name} = {shown}")
        starts.append(running)
        running += shap_value
        ends.append(running)
        colours.append(UP if shap_value >= 0 else DOWN)
    labels.append("final margin")
    starts.append(0.0)
    ends.append(wf["final"])
    colours.append(TOTAL)
    y = np.arange(len(labels))
    left = np.minimum(starts, ends)
    width = np.abs(np.array(ends) - np.array(starts))
    ax.barh(y, width, left=left, color=colours, height=0.62)
    span = float(max(ends + starts) - min(ends + starts + [0.0])) or 1.0
    for i in range(len(labels)):
        delta = ends[i] - starts[i]
        text = delta_text(delta) if 0 < i < len(labels) - 1 else plain_text(ends[i])
        ax.text(max(starts[i], ends[i]) + 0.012 * span, i, text, va="center", fontsize=FONT_SMALL,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0}, zorder=4)  # fmt: skip
    ax.set_yticks(y, labels, fontsize=FONT_SMALL)
    ax.invert_yaxis()
    ax.axvline(0, color="#777777", linewidth=0.8)
    lo, hi = min(ends + starts + [0.0]), max(ends + starts + [0.0])
    ax.set_xlim(lo - 0.05 * span, hi + 0.12 * span)
    ax.set_xlabel("log-odds of the higher-severity class of this target (mean over 5 repeats)",
                  fontsize=FONT_SMALL)  # fmt: skip
    ax.set_title(f"mean predicted probability {proba_text(wf['proba'])} (mean of 5 repeats)",
                 fontsize=FONT_SMALL, loc="right")  # fmt: skip
    style(ax)
    return finish(fig, f"{title}: clip {wf['ref']} ({KIND_TEXT[kind]})")


def figure_texts(fig: Figure) -> list[str]:
    return [t.get_text() for t in fig.findobj(Text) if t.get_text()]


# --------------------------------------------------------------------------- run


def save_png(fig: Figure, path: Path) -> None:
    _atomic(path, lambda tmp: fig.savefig(tmp, format="png", metadata={"Software": None}))


def _atomic(final: Path, write: Any) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f".{final.name}.tmp{os.getpid()}")
    try:
        write(tmp)
        os.replace(tmp, final)
    finally:
        if tmp.exists():
            tmp.unlink()


def sha8_file(path: Path) -> str:
    return cvs.sha8_of(path.read_bytes())


def build_config(xai: Path, cv_root: Path, target: str, fs: str) -> dict[str, Any]:
    cols = list(tsh.ch.FEATURE_SETS[fs])
    folder = xai / target / fs
    inputs = read_inputs(folder, cols)
    official_path = cv_root / target / f"meta_{tsh.MODEL}_{fs}.json"
    if not official_path.is_file():
        raise InputProblem(f"missing input: official meta for {target} {fs}")
    check_meta(inputs["meta"], json.loads(official_path.read_text()), target, fs)
    check_tables(inputs, cols, f"{target} {fs}")
    order = feature_order(inputs["glob"])
    wh = whiskers(inputs["by_fold"], order)
    check_whiskers(inputs["glob"], wh)
    clips = per_clip_means(inputs["values"], cols)
    chosen = select_clips(clips)
    title = title_of(target, fs)
    figures: dict[str, Figure] = {
        FIG_GLOBAL: fig_global(inputs["glob"], wh, order, title),
        FIG_DOTS: fig_dots(dot_layout(clips, order), order, title),
    }
    for kind in KINDS:
        figures[f"clip_{kind}_{chosen[kind]}.png"] = fig_waterfall(
            waterfall(clips, chosen[kind], cols), kind, title
        )
    return {"figures": figures, "chosen": chosen, "folder": folder, "title": title}


def write_config(xai: Path, cv_root: Path, target: str, fs: str) -> dict[str, Any]:
    built = build_config(xai, cv_root, target, fs)
    out_dir = xai / "figures" / target / fs
    sha8: dict[str, str] = {}
    for name, fig in built["figures"].items():
        save_png(fig, out_dir / name)
        sha8[name] = sha8_file(out_dir / name)
    meta = {
        "tool": TOOL, "target": target, "feature_set": fs, "title": built["title"],
        "inputs_sha8": {n: sha8_file(built["folder"] / n) for n in SHAP_INPUTS},
        "figures": list(built["figures"]), "figures_sha8": sha8, "chosen_clips": built["chosen"],
        "constants": {"jitter_seed": JITTER_SEED, "dpi": DPI, "figsize": list(FIGSIZE),
                      "cmap": CMAP, "kinds": list(KINDS)},  # fmt: skip
        "matplotlib": matplotlib.__version__,
    }  # fmt: skip
    data = (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode()
    _atomic(out_dir / META_FILE, lambda tmp: tmp.write_bytes(data))
    sha8[META_FILE] = cvs.sha8_of(data)
    return {"target": target, "fs": fs, "sha8": sha8, "chosen": built["chosen"]}


def render(done: Sequence[dict[str, Any]]) -> str:
    lines = []
    for d in done:
        lines.append(f"== {d['target']} Set {d['fs']}: {len(d['sha8'])} files")
        lines += [f"  {name}  sha8 {h}" for name, h in d["sha8"].items()]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cv-root", type=Path, default=None)
    ap.add_argument("--xai-root", type=Path, default=None)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cv_root = Path(args.cv_root or cvs.CV_ROOT)
    xai = Path(args.xai_root or tsh.xai_root())
    if not cvs.is_inside(xai, tsh.xai_root()):
        print("error: --xai-root must be inside data/processed/xai", file=sys.stderr)
        return 2
    try:
        done = [write_config(xai, cv_root, t, fs) for t, fs in tsh.CONFIGS]
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
