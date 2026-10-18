"""Does the fast eddy profile change the DTI features, and does the model score move?

The model was trained on features made with FSL's default eddy settings ("full"). The
"fast" profile (fewer iterations) is only worth using if it gives the same numbers. This
compares the two on the SAME people, and reports how far they differ. It trains nothing.

Two kinds of input, mix freely:

    # feature tables (dti_features.py output), paired on the `subject` column
    python models/mri/src/compare_dti_eddy.py \\
        --csv data/processed/features/dti_features.csv \\
              data/processed/features/dti_features_fast.csv

    # two predict_scan.py result files for the same person (full run, then fast run)
    python models/mri/src/compare_dti_eddy.py \\
        --pair 3104 result_full.json result_fast.json

Per feature it reports the mean difference, the mean relative difference and, with at
least MIN_PEOPLE people, the intraclass correlation ICC(2,1) between the two profiles.
The limits below were set BEFORE seeing any result. They are provisional (not from a
paper): change them only by writing down why. With fewer than MIN_PEOPLE people the
verdict is always "too few people": two people can show a difference, not prove agreement.

Writes comparison.md and feature_agreement.csv to data/processed/results/eddy_comparison_*.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]

# PROVISIONAL limits, fixed before looking at any fast-vs-full result.
MIN_PEOPLE = 5  # below this an ICC means nothing
ICC_MIN = 0.90  # a feature "agrees" only if ICC(2,1) is at least this ...
REL_DIFF_MAX_PCT = 2.0  # ... and its mean relative difference is at most this many percent
SHARE_OK_MIN = 0.90  # and at least this share of features must agree for the overall verdict


class CompareError(Exception):
    """The inputs cannot be compared (not the same people, no features, bad file)."""


def icc_2_1(full: np.ndarray, fast: np.ndarray) -> float:
    """ICC(2,1): absolute agreement between two measurements of the same people.

    Penalises a constant shift between the profiles, not just a loss of correlation.
    Returns NaN when it cannot be computed (under 2 people, or no variation at all).
    """
    x = np.column_stack([np.asarray(full, float), np.asarray(fast, float)])
    n, k = x.shape
    if n < 2:
        return float("nan")
    grand = x.mean()
    rows, cols = x.mean(axis=1), x.mean(axis=0)
    msr = k * ((rows - grand) ** 2).sum() / (n - 1)
    msc = n * ((cols - grand) ** 2).sum() / (k - 1)
    sse = ((x - rows[:, None] - cols[None, :] + grand) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    if denom <= 0:
        return float("nan")
    return float((msr - mse) / denom)


def paired_features(full: pd.DataFrame, fast: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The same people and the same dti_* columns, in the same order, from both tables."""
    for name, df in (("full", full), ("fast", fast)):
        if "subject" not in df.columns:
            raise CompareError(f"the {name} table has no 'subject' column")
        if df["subject"].duplicated().any():
            raise CompareError(f"the {name} table lists the same person twice")
    people = sorted(set(full["subject"]) & set(fast["subject"]))
    if not people:
        raise CompareError("the two tables share no people: nothing to compare")
    cols = [c for c in full.columns if c.startswith("dti_") and c in fast.columns]
    if not cols:
        raise CompareError("the two tables share no dti_* feature columns")
    a = full.set_index("subject").loc[people, cols].astype(float)
    b = fast.set_index("subject").loc[people, cols].astype(float)
    return a, b


def agreement(full: pd.DataFrame, fast: pd.DataFrame) -> pd.DataFrame:
    """One row per feature: how far apart the two profiles are, over the paired people."""
    a, b = paired_features(full, fast)
    n = len(a)
    rows = []
    for c in a.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(x != 0, np.abs(y - x) / np.abs(x) * 100.0, np.nan)
        icc = icc_2_1(x, y) if n >= MIN_PEOPLE else float("nan")
        rel_mean = float(np.nanmean(rel)) if np.isfinite(rel).any() else float("nan")
        rows.append(
            {
                "feature": c,
                "n": n,
                "mean_diff": float(np.mean(y - x)),  # fast minus full: the bias
                "mean_abs_diff": float(np.mean(np.abs(y - x))),
                "mean_rel_diff_pct": rel_mean,
                "max_rel_diff_pct": float(np.nanmax(rel)) if np.isfinite(rel).any() else np.nan,
                "icc": icc,
                "agrees": bool(
                    n >= MIN_PEOPLE
                    and np.isfinite(icc)
                    and icc >= ICC_MIN
                    and np.isfinite(rel_mean)
                    and rel_mean <= REL_DIFF_MAX_PCT
                ),
            }
        )
    return pd.DataFrame(rows)


def verdict(table: pd.DataFrame) -> str:
    n = int(table["n"].iloc[0])
    if n < MIN_PEOPLE:
        return (
            f"TOO FEW PEOPLE ({n} < {MIN_PEOPLE}): the differences below are descriptive only. "
            "This cannot show that the fast profile agrees."
        )
    share = float(table["agrees"].mean())
    if share >= SHARE_OK_MIN:
        return (
            f"AGREES within the provisional limits: {share:.0%} of {len(table)} features "
            f"(ICC >= {ICC_MIN}, mean relative difference <= {REL_DIFF_MAX_PCT}%)."
        )
    return (
        f"DIFFERS: only {share:.0%} of {len(table)} features meet the limits "
        f"(need {SHARE_OK_MIN:.0%}). Do not use the fast profile with the trained model."
    )


# ----------------------------------------------------------- predict_scan result files


@dataclass
class ScanPair:
    """The full and fast predict_scan results for one person."""

    name: str
    full: dict
    fast: dict
    notes: list[str] = field(default_factory=list)


def load_result(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise CompareError(f"cannot read result file {path}: {e}") from e
    if data.get("modality") != "mri":
        raise CompareError(f"{path} is not an MRI result")
    return data


def result_features(result: dict) -> dict[str, float]:
    """The raw DTI features of one predict_scan result, by name."""
    vec = result.get("feature_vector")
    if not vec:
        raise CompareError(f"no feature vector (status: {result.get('status')}, abstained?)")
    return {
        n: float(v) for n, v in zip(vec["names"], vec["raw"], strict=True) if n.startswith("dti_")
    }


def pairs_to_tables(pairs: list[ScanPair]) -> tuple[pd.DataFrame, pd.DataFrame]:
    full, fast = [], []
    for p in pairs:
        full.append({"subject": p.name, **result_features(p.full)})
        fast.append({"subject": p.name, **result_features(p.fast)})
    return pd.DataFrame(full), pd.DataFrame(fast)


def score_rows(pairs: list[ScanPair]) -> pd.DataFrame:
    """Model score (log-odds), interval and eddy time for each person under both profiles."""
    rows = []
    for p in pairs:
        row: dict[str, object] = {"person": p.name}
        for tag, res in (("full", p.full), ("fast", p.fast)):
            s = res.get("score") or {}
            lo, hi = (s.get("interval_95") or [np.nan, np.nan])[:2]
            eddy = ((res.get("inputs") or {}).get("dti") or {}).get("eddy") or {}
            row |= {
                f"score_{tag}": s.get("value", np.nan),
                f"lo_{tag}": lo,
                f"hi_{tag}": hi,
                f"eddy_s_{tag}": eddy.get("seconds"),
                f"profile_{tag}": eddy.get("profile"),
            }
        rows.append(row)
    df = pd.DataFrame(rows)
    df["score_diff"] = df["score_fast"] - df["score_full"]
    # Does the fast score fall inside the full score's own 95% interval?
    df["fast_inside_full_interval"] = (df["score_fast"] >= df["lo_full"]) & (
        df["score_fast"] <= df["hi_full"]
    )
    return df


def check_profiles(pairs: list[ScanPair]) -> None:
    """Refuse to compare two runs that did not use the profiles they claim."""
    for p in pairs:
        for tag, res in (("full", p.full), ("fast", p.fast)):
            got = (((res.get("inputs") or {}).get("dti") or {}).get("eddy") or {}).get("profile")
            if got is not None and got != tag:
                raise CompareError(
                    f"{p.name}: the file given as '{tag}' was made with the '{got}' profile"
                )


# ----------------------------------------------------------------------- report


def timing_line(scores: pd.DataFrame) -> str:
    both = scores.dropna(subset=["eddy_s_full", "eddy_s_fast"])
    if both.empty:
        return "Timing: eddy seconds were not recorded in these files."
    full, fast = both["eddy_s_full"].astype(float), both["eddy_s_fast"].astype(float)
    return (
        f"Eddy time, {len(both)} people: full {full.mean() / 60:.1f} min, "
        f"fast {fast.mean() / 60:.1f} min, {full.sum() / fast.sum():.2f}x faster. "
        "Only a fair comparison if both ran on the same machine, one person at a time."
    )


def render(table: pd.DataFrame, scores: pd.DataFrame | None) -> str:
    n = int(table["n"].iloc[0])
    worst = table.sort_values("mean_rel_diff_pct", ascending=False).head(8)
    lines = [
        "# Fast vs full eddy: DTI feature agreement",
        "",
        f"People compared: **{n}**. Features: **{len(table)}**.",
        "",
        f"**Verdict: {verdict(table)}**",
        "",
        f"Limits (provisional, set before seeing results): ICC(2,1) >= {ICC_MIN}, mean relative "
        f"difference <= {REL_DIFF_MAX_PCT}%, at least {SHARE_OK_MIN:.0%} of features, "
        f"at least {MIN_PEOPLE} people.",
        "",
        f"Median relative difference over features: {table['mean_rel_diff_pct'].median():.2f}%. "
        f"Median ICC: {table['icc'].median():.3f}."
        if n >= MIN_PEOPLE
        else f"Median relative difference over features: "
        f"{table['mean_rel_diff_pct'].median():.2f}%.",
        "",
        "## Features that differ most",
        "",
        "| feature | mean fast-full | mean rel. diff % | ICC |",
        "|---|---|---|---|",
    ]
    for _, r in worst.iterrows():
        icc = "n/a" if pd.isna(r["icc"]) else format(r["icc"], ".3f")
        lines.append(
            f"| {r['feature']} | {r['mean_diff']:+.4g} | {r['mean_rel_diff_pct']:.2f} | {icc} |"
        )
    if scores is not None and not scores.empty:
        lines += ["", "## Model score (log-odds, uncalibrated)", ""]
        lines += ["| person | full | fast | difference | fast inside full 95% interval |"]
        lines += ["|---|---|---|---|---|"]
        for _, r in scores.iterrows():
            lines.append(
                f"| {r['person']} | {r['score_full']:.3f} | {r['score_fast']:.3f} | "
                f"{r['score_diff']:+.3f} | {'yes' if r['fast_inside_full_interval'] else 'NO'} |"
            )
        lines += ["", timing_line(scores)]
    lines += [
        "",
        "Reading this: a fast profile that is only a little quicker but moves the features is "
        "not worth it, and one that agrees on 2 people proves nothing. Quote the people count "
        "with every number from this report.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--csv", nargs=2, type=Path, metavar=("FULL", "FAST"))
    ap.add_argument(
        "--pair",
        nargs=3,
        action="append",
        default=[],
        metavar=("NAME", "FULL_JSON", "FAST_JSON"),
        help="predict_scan results for one person, full run then fast run (repeatable)",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if not args.csv and not args.pair:
        ap.error("give --csv FULL FAST and/or --pair NAME FULL_JSON FAST_JSON")

    try:
        pairs = [
            ScanPair(name, load_result(Path(f)), load_result(Path(g))) for name, f, g in args.pair
        ]
        check_profiles(pairs)
        full_parts, fast_parts = [], []
        if args.csv:
            full_parts.append(pd.read_csv(args.csv[0]))
            fast_parts.append(pd.read_csv(args.csv[1]))
        scores = None
        if pairs:
            f, g = pairs_to_tables(pairs)
            full_parts.append(f)
            fast_parts.append(g)
            scores = score_rows(pairs)
        full = pd.concat(full_parts, ignore_index=True)
        fast = pd.concat(fast_parts, ignore_index=True)
        table = agreement(full, fast)
    except (CompareError, OSError, pd.errors.ParserError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    out = args.out or REPO / "data/processed/results" / (
        "eddy_comparison_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "feature_agreement.csv", index=False)
    if scores is not None:
        scores.to_csv(out / "scores.csv", index=False)
    text = render(table, scores)
    (out / "comparison.md").write_text(text)
    print(text)
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
