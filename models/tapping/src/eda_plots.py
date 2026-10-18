"""Exploratory plots of the 12 Set B features against the rating (task FT-31).

    python models/tapping/src/eda_plots.py [--features data/processed/features.csv] [--out-dir ...]

Descriptive only: these figures change no role, feature choice or threshold. The rating comes from
features.csv alone (the label file is never opened); scale, limits, jitter and order are fixed here
from the feature column, never from the rating. Spearman rho and proportions get participant-cluster
bootstrap CIs, no p-values. Exit codes: 0 ok, 1 inputs inconsistent, 2 missing input or bad path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from signal_processing import file_sha256, is_inside

TOOL_VERSION = "eda-plots-0.1.0"
REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
HASH_LEN = 8
DEFAULT_SEED, DEFAULT_BOOT = 2026, 2000
RATINGS = (0, 1, 2, 3)
DPI = 100
MARGIN = 0.05  # limits: pooled feature min/max plus 5 % (multiplicative on a log axis)
UNSTABLE_SHARE = 0.05  # more undefined bootstrap resamples than this: CI marked unstable
FEW_EVENTS = 5  # fewer events than this at a rating level: proportion CI marked few events
JITTER = 0.18  # half width of the horizontal jitter
BAND = 0.001  # iti_slope: |value| below this is shaded
BLUE, ORANGE, GREY, VERMILLION = "#0072B2", "#E69F00", "#999999", "#D55E00"  # Okabe-Ito
SUBTITLE = "exploratory; 12 features, no multiplicity correction"

SET_B = (
    "amp_slope",
    "open_vel_slope",
    "close_vel_slope",
    "amp_late_early",
    "amp_min_initial",
    "decrement_present",
    "decrement_onset_frac",
    "iti_slope",
    "n_hesitations",
    "n_halts",
    "longest_pause_ratio",
    "longest_pause_s",
)  # fmt: skip  (the column order of features.csv)
BLOCK_FIRST, BLOCK_LAST = "n_valid_taps", "nan_reasons_b"  # Set B columns of features.csv
KIND = {
    "decrement_present": "binary", "decrement_onset_frac": "conditional",
    "n_hesitations": "count", "n_halts": "count",
}  # fmt: skip
SCALE = {"longest_pause_ratio": "log", "longest_pause_s": "log"}  # all others linear
YLABEL = {
    "amp_slope": "amplitude slope (fraction of clip mean per 10 s)",
    "open_vel_slope": "opening speed slope (fraction of mean per 10 s)",
    "close_vel_slope": "closing speed slope (fraction of mean per 10 s)",
    "amp_late_early": "last third / first third amplitude (ratio)",
    "amp_min_initial": "lowest level / initial level (ratio)",
    "decrement_present": "share of clips with a sustained fall",
    "decrement_onset_frac": "onset position (fraction of the taps); among clips with a decrement",
    "n_hesitations": "hesitations per clip (count)",
    "n_halts": "halts per clip (count)",
    "longest_pause_ratio": "longest interval / median interval (ratio)",
    "longest_pause_s": "longest interval (s)",
    "iti_slope": "interval slope (fraction of mean per 10 s)",
}
OUT_META, OUT_SUMMARY_PNG, OUT_SUMMARY_CSV = "eda_meta.json", "summary_rho.png", "eda_summary.csv"


class Disagreement(ValueError):
    """The inputs are inconsistent (exit code 1)."""


class InputProblem(ValueError):
    """An input is missing or a path is outside its folder (exit code 2)."""


# --------------------------------------------------------------------------- statistics


def average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks, ties get the mean of their ranks."""
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2.0)[inverse]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho on the pairs where x is finite; NaN if fewer than 3 pairs or no variance."""
    keep = np.isfinite(x)
    if keep.sum() < 3:
        return float("nan")
    rx, ry = average_ranks(x[keep]), average_ranks(y[keep])
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def cluster_draws(
    participants: np.ndarray, boot: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Row indices of ``boot`` resamples: participants drawn with replacement, ALL their clips."""
    _, inverse = np.unique(participants, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(inverse.max() + 1)]
    k = len(members)
    return [np.concatenate([members[g] for g in rng.integers(0, k, k)]) for _ in range(boot)]


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


def rho_ci(x: np.ndarray, y: np.ndarray, draws: list[np.ndarray]) -> dict[str, Any]:
    """rho, its cluster-bootstrap percentile CI and the share of undefined resamples."""
    boots = np.array([spearman(x[d], y[d]) for d in draws])
    lo, hi = percentile_ci(boots)
    share = float(np.isnan(boots).mean())
    return {
        "rho": spearman(x, y), "ci_low": lo, "ci_high": hi, "undefined_share": share,
        "unstable": share > UNSTABLE_SHARE,
    }  # fmt: skip


def proportion_by_rating(
    ind: np.ndarray, rating: np.ndarray, draws: list[np.ndarray] | None
) -> list[dict[str, Any]]:
    """Share of 1s among the non-NaN values at each rating, with a cluster-bootstrap CI."""
    out = []
    boots = None
    if draws is not None:
        boots = np.full((len(draws), len(RATINGS)), np.nan)
        for b, d in enumerate(draws):
            i, r = ind[d], rating[d]
            for j, level in enumerate(RATINGS):
                v = i[(r == level) & np.isfinite(i)]
                if v.size:
                    boots[b, j] = v.mean()
    for j, level in enumerate(RATINGS):
        v = ind[(rating == level) & np.isfinite(ind)]
        events, n = int((v == 1).sum()), int(v.size)
        row: dict[str, Any] = {
            "n": n, "events": events, "prop": events / n if n else float("nan"),
            "low": float("nan"), "high": float("nan"), "flag": "",
        }  # fmt: skip
        if boots is not None:
            row["low"], row["high"] = percentile_ci(boots[:, j])
            row["flag"] = "few events" if events < FEW_EVENTS else ""
        out.append(row)
    return out


def rating_stats(values: np.ndarray, rating: np.ndarray) -> list[dict[str, Any]]:
    """Per rating: n, n_nan, median and quartiles (linear percentile) of the non-NaN values."""
    out = []
    for level in RATINGS:
        v = values[rating == level]
        ok = v[np.isfinite(v)]
        q1, med, q3 = np.percentile(ok, [25, 50, 75]) if ok.size else (np.nan,) * 3
        out.append({"n": int(ok.size), "n_nan": int(v.size - ok.size), "median": float(med),
                    "q1": float(q1), "q3": float(q3)})  # fmt: skip
    return out


# --------------------------------------------------------------------------- fixed choices


def choose_scale(feature: str, values: np.ndarray) -> tuple[str, str]:
    """(scale, note): the pre-registered scale, linear if a log axis is impossible."""
    want = SCALE.get(feature, "linear")
    ok = values[np.isfinite(values)]
    if want == "log" and ok.size and not (ok > 0).all():
        return "linear", "log refused: non-positive values"
    return want, ""


def axis_limits(values: np.ndarray, scale: str) -> tuple[float, float]:
    """Pooled min/max of the feature (never of a rating subset) plus the fixed margin."""
    ok = values[np.isfinite(values)]
    if ok.size == 0:
        return 0.0, 1.0
    lo, hi = float(ok.min()), float(ok.max())
    if hi == lo:
        return lo - 1.0, hi + 1.0
    if scale == "log":
        f = (hi / lo) ** MARGIN
        return lo / f, hi * f
    pad = MARGIN * (hi - lo)
    return lo - pad, hi + pad


def jitter_for(feature: str, seed: int, n: int) -> np.ndarray:
    """Horizontal jitter, seeded from sha256 of the feature name and the seed (never hash())."""
    digest = hashlib.sha256(f"{feature}:{seed}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big")).uniform(-JITTER, JITTER, n)


# --------------------------------------------------------------------------- data


class Data:
    """The columns the plots need, parsed from the text of features.csv."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.participant = frame["participant"].to_numpy(dtype=str)
        self.hand = frame["hand"].to_numpy(dtype=str)
        self.rating = np.array([int(v) for v in frame["rating"]], dtype=float)
        self.values = {f: parse_column(frame[f], f) for f in SET_B}
        self.n = len(frame)


def parse_column(text: pd.Series, name: str) -> np.ndarray:
    if name == "decrement_present":
        table = {"True": 1.0, "False": 0.0, "": float("nan")}
        bad = sorted(set(text) - set(table))
        if bad:
            raise Disagreement("decrement_present holds a value other than True, False or empty")
        return np.array([table[v] for v in text])
    try:
        return np.array([float(v) if v != "" else float("nan") for v in text])
    except ValueError as err:
        raise Disagreement(f"{name} holds a value that is not a number") from err


def validate(frame: pd.DataFrame) -> list[str]:
    """The Set B features in features.csv column order; raises Disagreement if inconsistent."""
    cols = list(frame.columns)
    if BLOCK_FIRST not in cols or BLOCK_LAST not in cols:
        raise Disagreement("the Set B block of features.csv was not found")
    block = cols[cols.index(BLOCK_FIRST) : cols.index(BLOCK_LAST) + 1]
    found = [c for c in block if not c.endswith("_sens") and c not in (BLOCK_FIRST, BLOCK_LAST)]
    missing, extra = sorted(set(SET_B) - set(found)), sorted(set(found) - set(SET_B))
    if missing or extra:
        raise Disagreement(f"Set B columns differ: {len(missing)} missing, {len(extra)} extra")
    if found != list(SET_B):
        raise Disagreement("Set B columns are not in the registry order")
    for c in ("participant", "hand", "rating"):
        if c not in cols:
            raise Disagreement(f"column {c} is missing")
    bad = [v for v in frame["rating"] if v not in ("0", "1", "2", "3")]
    if bad:
        raise Disagreement(f"{len(bad)} ratings are blank or not an integer 0-3")
    if not set(frame["hand"]) <= {"left", "right"}:
        raise Disagreement("hand must be left or right")
    present = parse_column(frame["decrement_present"], "decrement_present")
    onset = parse_column(frame["decrement_onset_frac"], "decrement_onset_frac")
    if not np.array_equal(np.isfinite(onset), present == 1.0):
        raise Disagreement("decrement_onset_frac and decrement_present are inconsistent")
    return found


# --------------------------------------------------------------------------- figures


def _strip(
    ax: Axes, d: Data, feature: str, values: np.ndarray, seed: int, note: str
) -> dict[str, Any]:
    """Jittered strip with a median line and an IQR bar per rating; returns the limits used."""
    scale, scale_note = choose_scale(feature, values)
    jit = jitter_for(feature, seed, d.n)
    stats = rating_stats(values, d.rating)
    for j, level in enumerate(RATINGS):
        pick = (d.rating == level) & np.isfinite(values)
        for hand, marker in (("right", "o"), ("left", "^")):
            m = pick & (d.hand == hand)
            if m.any():
                ax.scatter(level + jit[m], values[m], s=14, marker=marker, color=BLUE, alpha=0.7,
                           linewidths=0, gid="strip")  # fmt: skip
        s = stats[j]
        if s["n"]:
            ax.vlines(level, s["q1"], s["q3"], color=ORANGE, linewidth=6, alpha=0.8, zorder=1)
            ax.hlines(s["median"], level - 0.3, level + 0.3, color="black", linewidth=2, zorder=3)
        ax.text(level, 0.01, f"NaN {s['n_nan']}", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=7, color=GREY)  # fmt: skip
    ax.set_xticks(list(RATINGS), [f"{lv}\n(n={stats[j]['n']})" for j, lv in enumerate(RATINGS)])
    ax.set_xlim(-0.6, 3.6)
    ax.set_xlabel("rating (higher = more severe)")
    ax.set_yscale(scale)
    if scale == "log":  # plain numbers (0.5, 1, 2, 5) instead of powers of ten
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(*axis_limits(values, scale))
    ax.set_ylabel(YLABEL[feature], fontsize=8)
    ax.grid(alpha=0.25)
    return {"scale": scale, "note": scale_note or note}


def _proportion(ax: Axes, rows: list[dict[str, Any]], feature: str, with_ci: bool) -> None:
    x = np.array(RATINGS, dtype=float)
    prop = np.array([r["prop"] for r in rows])
    ax.scatter(x, prop, s=36, color=VERMILLION, zorder=3, gid="proportion")
    if with_ci:
        low, high = np.array([r["low"] for r in rows]), np.array([r["high"] for r in rows])
        ok = np.isfinite(low) & np.isfinite(high)
        ax.vlines(x[ok], low[ok], high[ok], color=VERMILLION, linewidth=2)
    for level, r in zip(RATINGS, rows, strict=True):
        ax.text(level, 1.02, f"{r['events']}/{r['n']}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8)  # fmt: skip
    ax.set_xticks(
        list(RATINGS), [f"{lv}\n(n={r['n']})" for lv, r in zip(RATINGS, rows, strict=True)]
    )
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(0, 1)
    ax.set_xlabel("rating (higher = more severe)")
    ax.set_ylabel(
        "share non-zero" if feature != "decrement_present" else YLABEL[feature], fontsize=8
    )
    ax.grid(alpha=0.25)


def feature_figure(
    feature: str, d: Data, roles: dict[str, tuple[str, str]], res: dict[str, Any], seed: int
) -> tuple[Figure, dict[str, Any]]:
    """One figure for one feature, and what was fixed for it (scale and notes)."""
    kind = KIND.get(feature, "continuous")
    wide = kind in ("count",)
    fig = Figure(figsize=(10.5 if wide else 6.4, 4.9), dpi=DPI, layout="constrained")
    FigureCanvasAgg(fig)
    values = d.values[feature]
    role, note = roles[feature]
    info: dict[str, Any] = {"scale": "linear", "note": ""}
    if kind == "binary":
        ax = fig.subplots()
        _proportion(ax, res["props"], feature, True)
    elif kind == "count":
        a1, a2 = fig.subplots(1, 2)
        info = _strip(a1, d, feature, values, seed, "")
        _proportion(a2, res["props"], feature, feature != "n_halts")
        a2.set_title(
            "counts only" if feature == "n_halts" else "non-zero share, 95% CI", fontsize=8, pad=16
        )
    else:
        ax = fig.subplots()
        info = _strip(ax, d, feature, values, seed, "")
        if kind == "conditional":
            ax.set_title("among clips with a decrement; NaN = no decrement", fontsize=8)
        if feature == "iti_slope":
            ax.axhspan(-BAND, BAND, color=GREY, alpha=0.35, zorder=0)
            ax.text(0.99, 0.97, "exploratory; split-half reliability 0.43", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8)  # fmt: skip
    rho = res["rho"]
    line = (
        "counts only, no rho"
        if feature == "n_halts"
        else (
            f"rho {rho['rho']:.2f} [{rho['ci_low']:.2f}, {rho['ci_high']:.2f}] "
            f"(exploratory, participant-cluster bootstrap{', unstable' if rho['unstable'] else ''})"
        )
    )
    fig.suptitle(feature, fontsize=11)
    fig.supxlabel(f"{role}: {note}  |  {line}", fontsize=7)
    return fig, info


def summary_figure(results: dict[str, dict[str, Any]], roles: dict[str, tuple[str, str]]) -> Figure:
    fig = Figure(figsize=(8.5, 5.6), dpi=DPI, layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    labels = []
    for pos, f in enumerate(SET_B):
        r = results[f]["rho"]
        tag = roles[f][0].removeprefix("feature_")
        if f == "n_halts":
            labels.append(f"{f} ({tag}; counts only)")
            continue
        extra = ", conditional" if f == "decrement_onset_frac" else ""
        extra += ", unstable" if r["unstable"] else ""
        labels.append(f"{f} ({tag}{extra})")
        ax.hlines(pos, r["ci_low"], r["ci_high"], color=BLUE, linewidth=2)
        ax.scatter([r["rho"]], [pos], color=BLUE, s=30, zorder=3)
    ax.axvline(0, color=GREY, linewidth=1)
    ax.set_yticks(range(len(SET_B)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman rho with the rating (95% participant-cluster bootstrap CI)")
    ax.set_title(SUBTITLE, fontsize=9)
    ax.grid(alpha=0.25, axis="x")
    return fig


# --------------------------------------------------------------------------- run


def _atomic(final: Path, write: Any) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f".{final.name}.tmp{os.getpid()}")
    try:
        write(tmp)
        os.replace(tmp, final)
    finally:
        if tmp.exists():
            tmp.unlink()


def save_png(fig: Figure, path: Path) -> None:
    _atomic(path, lambda tmp: fig.savefig(tmp, format="png", metadata={"Software": None}))


def roles_sha8(roles: pd.DataFrame) -> str:
    table = {r.column: [r.role, r.note] for r in roles.itertuples()}
    text = json.dumps(table, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:HASH_LEN]


def read_inputs(features: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    paths = (
        features,
        features.with_name("feature_roles.csv"),
        features.with_name("features_meta.json"),
    )
    for p in paths:
        if not p.is_file():
            raise InputProblem(f"input not found: {p.name}")
    frame = pd.read_csv(paths[0], dtype=str, keep_default_na=False)
    roles = pd.read_csv(paths[1], dtype=str, keep_default_na=False)
    meta = json.loads(paths[2].read_text(encoding="utf-8"))
    if roles_sha8(roles) != meta.get("roles_sha8"):
        raise Disagreement("the roles file does not match roles_sha8 in features_meta.json")
    if list(roles["column"]) != list(frame.columns):
        raise Disagreement("feature_roles.csv and features.csv list different columns")
    return frame, roles, meta


def run(features: Path, out_dir: Path, seed: int, boot: int) -> dict[str, Any]:
    frame, roles_df, _ = read_inputs(features)
    validate(frame)
    roles = {r.column: (r.role, r.note) for r in roles_df.itertuples()}
    d = Data(frame)
    draws = cluster_draws(d.participant, boot, np.random.default_rng(seed))
    results: dict[str, dict[str, Any]] = {}
    for f in SET_B:
        x = d.values[f]
        kind = KIND.get(f, "continuous")
        res: dict[str, Any] = {"rho": {"rho": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                                       "undefined_share": np.nan, "unstable": False}}  # fmt: skip
        if f != "n_halts":
            res["rho"] = rho_ci(x, d.rating, draws)
        if kind in ("binary", "count"):
            ind = x if kind == "binary" else np.where(np.isfinite(x), (x > 0).astype(float), np.nan)
            res["props"] = proportion_by_rating(ind, d.rating, None if f == "n_halts" else draws)
        results[f] = res
    scales: dict[str, str] = {}
    rows = []
    for f in SET_B:
        fig, info = feature_figure(f, d, roles, results[f], seed)
        save_png(fig, out_dir / f"{f}.png")
        scales[f] = info["scale"] + (f" ({info['note']})" if info["note"] else "")
        stats = rating_stats(d.values[f], d.rating)
        rho = results[f]["rho"]
        flags = []
        if f == "n_halts":
            flags.append("counts only")
        if f == "decrement_onset_frac":
            flags.append("conditional")
        if f == "iti_slope":
            flags.append("exploratory")
        if rho["unstable"]:
            flags.append("unstable")
        for j, level in enumerate(RATINGS):
            p = results[f].get("props", [None] * 4)[j]
            row = {"feature": f, "rating": level, **stats[j],
                   "n_true": p["events"] if p and KIND.get(f) == "binary" else np.nan,
                   "n_nonzero": p["events"] if p and KIND.get(f) == "count" else np.nan,
                   "prop": p["prop"] if p else np.nan, "prop_low": p["low"] if p else np.nan,
                   "prop_high": p["high"] if p else np.nan, "prop_flag": p["flag"] if p else "",
                   "rho": rho["rho"], "ci_low": rho["ci_low"], "ci_high": rho["ci_high"],
                   "undefined_share": rho["undefined_share"], "flags": ";".join(flags)}  # fmt: skip
            if KIND.get(f) == "binary":
                row.update(median=np.nan, q1=np.nan, q3=np.nan)
            rows.append(row)
    save_png(summary_figure(results, roles), out_dir / OUT_SUMMARY_PNG)
    table = pd.DataFrame(rows)
    _atomic(out_dir / OUT_SUMMARY_CSV, lambda p: table.to_csv(p, index=False, lineterminator="\n"))
    meta = {
        "tool": TOOL_VERSION,
        "inputs_sha8": {
            "features_csv": file_sha256(features)[:HASH_LEN],
            "feature_roles_csv": file_sha256(features.with_name("feature_roles.csv"))[:HASH_LEN],
            "features_meta_json": file_sha256(features.with_name("features_meta.json"))[:HASH_LEN],
        },
        "roles_sha8": roles_sha8(roles_df),
        "seed": seed,
        "boot": boot,
        "versions": {
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },  # fmt: skip
        "scale": scales,
        "rows": d.n,
        "statement": SUBTITLE,
    }
    _atomic(
        out_dir / OUT_META,
        lambda p: p.write_text(json.dumps(meta, sort_keys=True, indent=1) + "\n", encoding="utf-8"),
    )
    return meta


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--features", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--boot", type=int, default=DEFAULT_BOOT)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    features = Path(args.features or PROCESSED_ROOT / "features.csv")
    out_dir = Path(args.out_dir or PROCESSED_ROOT / "eda")
    if not is_inside(out_dir, PROCESSED_ROOT):
        print(f"error: --out-dir must be inside {PROCESSED_ROOT.parent.name}/{PROCESSED_ROOT.name}",
              file=sys.stderr)  # fmt: skip
        return 2
    if args.boot < 1:
        print("error: --boot must be at least 1", file=sys.stderr)
        return 2
    try:
        meta = run(features, out_dir, args.seed, args.boot)
    except Disagreement as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    except InputProblem as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    print(f"wrote {len(SET_B) + 3} files: {meta['rows']} clips, seed {args.seed}, boot {args.boot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
