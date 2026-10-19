"""GAIT-33: the real-video results, from the KOA-PD-NM run (``run_koa_pd_nm.py``).

    python models/gait/src/report_koa_pd_nm.py

Scores are averaged per person, then compared between groups (AUC with a bootstrap 95% interval
resampled by person) and against the severity folders (Spearman). Simple baselines, a
frame-rate-matched subset and the run without the helper clips show how much of the result could
come from the groups being filmed or aged differently. Indicative only: KOA-PD-NM has no UPDRS
labels. Writes ``artifacts/koa_pd_nm/report/``. Exit code 0 = ok, 2 = no results to read.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from run_koa_pd_nm import DEFAULT_OUT
from scipy.stats import spearmanr

LEVEL_RANK = {"PD": {"ML": 1, "MD": 2, "SV": 3}, "KOA": {"EL": 1, "MD": 2, "SV": 3}}
BOOTSTRAP = 2000
SEED = 0
# name, column, sign: +1 when a higher value should mean more impaired
SCORES = [
    ("model: expected severity", "expected_severity", 1),
    ("baseline: step length only", "step_length_norm", -1),
    ("baseline: cadence only", "cadence", -1),
    ("confounder: walk length only", "walk_seconds", 1),
]


def load(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, dtype={"trial": str, "level": str})
    table = table[table["run_seconds"].notna()].copy()  # rows cut short by a stopped run
    table["level"] = table["level"].fillna("")
    # group + folder + person: IDs restart in each severity folder
    table["who"] = table["group"] + "|" + table["level"] + "|" + table["person"]
    return table


def per_person(scored: pd.DataFrame) -> pd.DataFrame:
    cols = [c for _, c, _ in SCORES]
    return scored.groupby(["group", "level", "who"], as_index=False)[cols].mean()


def pair_auc(case: np.ndarray, control: np.ndarray) -> float:
    """Share of case/control pairs ranked the right way (ties count half): the AUC."""
    if len(case) == 0 or len(control) == 0:
        return float("nan")
    diff = case[:, None] - control[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def _sides(
    people: pd.DataFrame, case: str, control: str, col: str, sign: int
) -> tuple[np.ndarray, np.ndarray]:
    a = sign * people.loc[people["group"] == case, col].to_numpy(dtype=float)
    b = sign * people.loc[people["group"] == control, col].to_numpy(dtype=float)
    return a, b


def auc(people: pd.DataFrame, case: str, control: str, col: str, sign: int) -> float:
    return pair_auc(*_sides(people, case, control, col, sign))


def auc_interval(
    people: pd.DataFrame, case: str, control: str, col: str, sign: int
) -> tuple[float, float]:
    """95% interval, resampling people within each group."""
    a, b = _sides(people, case, control, col, sign)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(SEED)
    values = [pair_auc(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(BOOTSTRAP)]
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def spearman(rank: np.ndarray, score: np.ndarray) -> float:
    if len(rank) < 3 or len(np.unique(rank)) < 2 or len(np.unique(score)) < 2:
        return float("nan")
    return float(spearmanr(rank, score).statistic)


def severity_order(people: pd.DataFrame, group: str) -> dict[str, Any]:
    """Spearman between the severity folder and the person's expected severity, with a 95%
    interval resampling people."""
    x = people[people["group"] == group]
    rank = x["level"].map(LEVEL_RANK[group]).to_numpy(dtype=float)
    score = x["expected_severity"].to_numpy(dtype=float)
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(BOOTSTRAP if len(x) >= 3 else 0):
        pick = rng.integers(0, len(x), len(x))
        values.append(spearman(rank[pick], score[pick]))
    finite = [v for v in values if np.isfinite(v)]
    low, high = np.percentile(finite, [2.5, 97.5]) if finite else (np.nan, np.nan)
    return {
        "group": group,
        "spearman": round(spearman(rank, score), 3),
        "ci_low": round(float(low), 3),
        "ci_high": round(float(high), 3),
        "people": len(x),
    }


def group_table(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group, level), x in table.groupby(["group", "level"]):
        s = x[x["status"] == "scored"]
        classes = s["predicted_class"].value_counts()
        rows.append(
            {
                "group": group,
                "level": level or "-",
                "people": x["who"].nunique(),
                "clips": len(x),
                "scored": len(s),
                "not_scored": int((x["status"] != "scored").sum()),
                "expected_severity_mean": round(s["expected_severity"].mean(), 2),
                "expected_severity_median": round(s["expected_severity"].median(), 2),
                "class_0": int(classes.get(0, 0)),
                "class_1": int(classes.get(1, 0)),
                "class_2_3": int(classes.get(2, 0)),
            }
        )
    return pd.DataFrame(rows)


def comparisons(scored: pd.DataFrame) -> pd.DataFrame:
    subsets = {
        "all scored clips": scored,
        "without the known helper clips": scored[scored["known_two_people"] == 0],
        "30 fps clips only": scored[scored["frame_rate"] == 30],
        "50 fps clips only": scored[scored["frame_rate"] == 50],
    }
    rows = []
    for subset, clips in subsets.items():
        people = per_person(clips)
        for case, control in (("PD", "NM"), ("KOA", "NM"), ("PD", "KOA")):
            n_case = int((people["group"] == case).sum())
            n_control = int((people["group"] == control).sum())
            for name, col, sign in SCORES:
                if subset != "all scored clips" and col != "expected_severity":
                    continue
                low, high = auc_interval(people, case, control, col, sign)
                rows.append(
                    {
                        "subset": subset,
                        "comparison": f"{case} vs {control}",
                        "score": name,
                        "people": f"{n_case} vs {n_control}",
                        "auc": round(auc(people, case, control, col, sign), 3),
                        "ci_low": round(low, 3),
                        "ci_high": round(high, 3),
                    }
                )
    return pd.DataFrame(rows)


def not_scored(table: pd.DataFrame) -> pd.DataFrame:
    x = table[table["status"] != "scored"].copy()
    x["reason"] = x["reasons"].fillna("").str.replace(r"\d+", "N", regex=True).str.slice(0, 70)
    return x.groupby(["group", "reason"]).size().reset_index(name="clips")


def plot(scored: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    people = per_person(scored)
    order = [
        ("NM", ""),
        ("PD", "ML"),
        ("PD", "MD"),
        ("PD", "SV"),
        ("KOA", "EL"),
        ("KOA", "MD"),
        ("KOA", "SV"),
    ]
    labels, values = [], []
    for group, level in order:
        v = people[(people["group"] == group) & (people["level"] == level)]["expected_severity"]
        if len(v):
            labels.append(f"{group} {level}".strip() + f"\n(n={len(v)})")
            values.append(v.to_numpy())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(values, showfliers=False)
    for i, v in enumerate(values, start=1):
        ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, s=12, alpha=0.7)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_ylabel("expected severity per person (0-2)")
    ax.set_ylim(0, 2)
    ax.set_title("KOA-PD-NM: model score per person, by group and severity folder")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KOA-PD-NM real-video results (GAIT-33)")
    parser.add_argument("--results", type=Path, default=DEFAULT_OUT / "results.csv")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT / "report")
    args = parser.parse_args(argv)
    if not args.results.is_file():
        print(f"MISSING: {args.results}\nRun run_koa_pd_nm.py first.")
        return 2

    table = load(args.results)
    scored = table[table["status"] == "scored"]
    people = per_person(scored)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    groups, comps, missing = group_table(table), comparisons(scored), not_scored(table)
    order = pd.DataFrame([severity_order(people, g) for g in LEVEL_RANK])
    groups.to_csv(args.out_dir / "groups.csv", index=False)
    comps.to_csv(args.out_dir / "comparisons.csv", index=False)
    order.to_csv(args.out_dir / "severity_order.csv", index=False)
    missing.to_csv(args.out_dir / "not_scored.csv", index=False)
    plot(scored, args.out_dir / "severity_by_group.png")

    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print(f"{len(table)} clips, {len(scored)} scored, {table['who'].nunique()} people\n")
        print(groups.to_string(index=False), "\n")
        print(comps.to_string(index=False), "\n")
        print("Severity folder against expected severity (per person):")
        print(order.to_string(index=False), "\n")
        print(missing.to_string(index=False))
    print(f"\nWrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
