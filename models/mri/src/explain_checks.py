"""Are the MRI explanations faithful and stable? Two checks on the cross-validation people.

The app explains each score with exact TreeSHAP contributions. "Exact" only means the
numbers add up to the score; it does not show that the top-ranked measurements are what the
model actually relies on, or that another copy of the model would rank them the same way.
These two checks measure that, on people each model never saw:

1. Deletion test (faithfulness). For every person in a test fold, set their k most important
   measurements (by |SHAP| from that fold's model) to the training fold's median, and see how
   far the score moves. Do the same for 100 random sets of k measurements. A faithful
   explanation moves the score more when its top measurements are removed than random ones.
   Per person: (top-k change) - (mean random-k change), averaged over the repeats in which the
   person was tested. PASS if the bootstrap 95% CI of the mean over people is above 0.

2. Stability test. Each person is tested by 10 copies of the model (one per repeat), each
   trained on different people. For every pair of those copies: the Spearman correlation of
   their |SHAP| rankings (over the measurements at least one of the two used) and how many of
   their top-5 measurements are the same. Averaged per person, then over people with a
   bootstrap 95% CI. MEETS TARGET if the mean correlation is at least 0.70 (the same target
   the tapping component pre-registered); the top-5 overlap is compared with chance (5 / the
   number of measurements).

Both rules were fixed before the checks were run on real data. The models are refitted
exactly as the deployed bundle's fold copies (mri_bundle.fit_member, same folds and seeds).
Both describe the explanation of the model; neither is evidence of a cause in the brain.

    python models/mri/src/explain_checks.py                    # paired set, newest labels
    python models/mri/src/explain_checks.py --set t1 --labels data/interim/mri_labels.csv

Writes explain_checks.json and explain_checks.md to data/processed/results/explain_<time>/.
Only pseudonyms and summary numbers are written.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from mri_bundle import Member, fit_member
from scipy.stats import spearmanr
from train_mri import RESULTS, make_folds
from training_data import DEFAULT_LABELS, FEATURES, MIN_BATCH, ModelData, load_all

TOP_K = 5
CURVE_K = (1, 3, 5, 10)
N_RANDOM = 100
N_BOOT = 2000
STABILITY_TARGET = 0.70
SEED = 2026

# attribution(member, rows) -> |importance| per row and column, same shape as rows
Attribution = Callable[[Member, pd.DataFrame], np.ndarray]


def shap_abs(member: Member, rows: pd.DataFrame) -> np.ndarray:
    """|TreeSHAP| per person and measurement, from LightGBM's exact pred_contrib."""
    c = member.booster.predict(rows[member.columns], pred_contrib=True)
    return np.abs(np.asarray(c)[:, :-1])


def harmonised_rows(member: Member, data: ModelData, idx: np.ndarray) -> pd.DataFrame:
    """The people at ``idx`` through this member's frozen ComBat (as the app does)."""
    cov = data.covariates()[idx]
    parts = []
    for g, cols in member.kept.items():
        x = data.features.iloc[idx][cols].to_numpy(dtype=float)
        parts.append(pd.DataFrame(member.combat[g].transform(x, data.batches[g][idx], cov), columns=cols))
    return pd.concat(parts, axis=1)


def _scores(member: Member, rows: pd.DataFrame) -> np.ndarray:
    return np.asarray(member.booster.predict(rows[member.columns], raw_score=True), dtype=float)


def _boot_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_folds(data: ModelData, n_splits: int, n_repeats: int, seed: int):
    """(fold, member, test rows, training medians) for every fold, refitted like the bundle."""
    for f in make_folds(data, data.y, n_splits, n_repeats, seed):
        member = fit_member(data, f.train_idx, seed + f.repeat)
        train = harmonised_rows(member, data, f.train_idx)
        test = harmonised_rows(member, data, f.test_idx)
        yield f, member, test, train.median()


def deletion_test(
    folds, n_people: int, attribution: Attribution = shap_abs, seed: int = SEED
) -> dict:
    rng = np.random.default_rng(seed)
    per_person: dict[int, list[float]] = {}
    curve = {k: {"top": [], "random": []} for k in CURVE_K}
    for f, member, test, medians in folds:
        cols = member.columns
        base = _scores(member, test)
        imp = attribution(member, test)
        for row_i, person in enumerate(f.test_idx):
            order = np.argsort(-imp[row_i])
            row = test.iloc[[row_i]]
            for k in CURVE_K:
                kk = min(k, len(cols))
                top = row.copy()
                top.iloc[0, order[:kk]] = medians[cols].to_numpy()[order[:kk]]
                top_change = abs(_scores(member, top)[0] - base[row_i])
                rand_sets = [rng.choice(len(cols), size=kk, replace=False) for _ in range(N_RANDOM)]
                rand = pd.concat([row] * N_RANDOM, ignore_index=True)
                for j, s in enumerate(rand_sets):
                    rand.iloc[j, s] = medians[cols].to_numpy()[s]
                rand_change = float(np.mean(np.abs(_scores(member, rand) - base[row_i])))
                curve[k]["top"].append(top_change)
                curve[k]["random"].append(rand_change)
                if k == TOP_K:
                    per_person.setdefault(int(person), []).append(top_change - rand_change)
    diffs = np.array([np.mean(v) for v in per_person.values()])
    low, high = _boot_mean_ci(diffs, np.random.default_rng(seed))
    return {
        "k": TOP_K,
        "n_people": len(diffs),
        "n_random_sets": N_RANDOM,
        "mean_top_minus_random": float(diffs.mean()),
        "ci_95": [low, high],
        "share_people_top_above_random": float(np.mean(diffs > 0)),
        "passed": bool(low > 0),
        "curve": {
            str(k): {
                "mean_change_top": float(np.mean(v["top"])),
                "mean_change_random": float(np.mean(v["random"])),
            }
            for k, v in curve.items()
        },
        "rule": "PASS if the 95% CI of mean(top-k change - random-k change) over people is above 0",
    }


def stability_test(
    folds, attribution: Attribution = shap_abs, seed: int = SEED
) -> dict:
    by_person: dict[int, list[pd.Series]] = {}
    n_features = 0
    for f, member, test, _ in folds:
        imp = attribution(member, test)
        n_features = max(n_features, len(member.columns))
        for row_i, person in enumerate(f.test_idx):
            by_person.setdefault(int(person), []).append(pd.Series(imp[row_i], index=member.columns))
    rhos, overlaps = [], []
    for vectors in by_person.values():
        p_rho, p_top = [], []
        for a, b in itertools.combinations(vectors, 2):
            a, b = a.align(b, fill_value=0.0)  # a column ComBat dropped in one fold counts as 0
            used = (a > 0) | (b > 0)  # ignore measurements neither copy used (both exactly 0)
            if used.sum() >= 3:
                rho = spearmanr(a[used], b[used])[0]
                if np.isfinite(rho):
                    p_rho.append(rho)
            top_a = set(a.nlargest(TOP_K).index)
            top_b = set(b.nlargest(TOP_K).index)
            p_top.append(len(top_a & top_b) / TOP_K)
        if p_rho:
            rhos.append(np.mean(p_rho))
        if p_top:
            overlaps.append(np.mean(p_top))
    rhos_a, over_a = np.array(rhos), np.array(overlaps)
    rng = np.random.default_rng(seed)
    return {
        "n_people": len(rhos_a),
        "copies_per_person": int(np.median([len(v) for v in by_person.values()])),
        "mean_spearman": float(rhos_a.mean()),
        "spearman_ci_95": list(_boot_mean_ci(rhos_a, rng)),
        "median_spearman": float(np.median(rhos_a)),
        "share_people_at_target": float(np.mean(rhos_a >= STABILITY_TARGET)),
        "mean_top5_overlap": float(over_a.mean()),
        "top5_overlap_ci_95": list(_boot_mean_ci(over_a, rng)),
        "top5_overlap_by_chance": TOP_K / n_features if n_features else None,
        "target": STABILITY_TARGET,
        "meets_target": bool(rhos_a.mean() >= STABILITY_TARGET),
        "rule": "MEETS TARGET if the mean Spearman correlation between copies is at least 0.70",
    }


def run(data: ModelData, n_splits: int = 5, n_repeats: int = 10, seed: int = 0) -> dict:
    folds = list(run_folds(data, n_splits, n_repeats, seed))
    return {
        "set": data.name,
        "people": int(data.n),
        "folds": n_splits,
        "repeats": n_repeats,
        "deletion": deletion_test(folds, data.n),
        "stability": stability_test(folds),
    }


def markdown(r: dict) -> str:
    d, s = r["deletion"], r["stability"]
    lines = [
        f"# Explanation checks: {r['set']} set ({r['people']} people, "
        f"{r['folds']} folds x {r['repeats']} repeats)",
        "",
        "These describe the model's explanation, not a cause in the brain.",
        "",
        f"## Deletion test (faithfulness): **{'PASS' if d['passed'] else 'FAIL'}**",
        "",
        f"Removing each person's top {d['k']} measurements (set to the training median) moved the "
        f"score by {d['mean_top_minus_random']:+.3f} log-odds more than removing {d['k']} random "
        f"ones (95% CI {d['ci_95'][0]:+.3f} to {d['ci_95'][1]:+.3f}; {d['n_people']} people, "
        f"{d['n_random_sets']} random sets each). Top beat random for "
        f"{d['share_people_top_above_random']:.0%} of people.",
        "",
        "| k | mean score change, top-k | mean score change, random-k |",
        "|---|---|---|",
    ]
    for k, v in d["curve"].items():
        lines.append(f"| {k} | {v['mean_change_top']:.3f} | {v['mean_change_random']:.3f} |")
    lines += [
        "",
        f"Rule: {d['rule']}.",
        "",
        f"## Stability test: **{'MEETS TARGET' if s['meets_target'] else 'BELOW TARGET'}**",
        "",
        f"Across the {s['copies_per_person']} model copies that tested each person: mean Spearman "
        f"correlation of their rankings {s['mean_spearman']:.2f} (95% CI "
        f"{s['spearman_ci_95'][0]:.2f} to {s['spearman_ci_95'][1]:.2f}; median "
        f"{s['median_spearman']:.2f}; {s['share_people_at_target']:.0%} of people at or above "
        f"{s['target']:.2f}). Top-5 overlap {s['mean_top5_overlap']:.0%} (95% CI "
        f"{s['top5_overlap_ci_95'][0]:.0%} to {s['top5_overlap_ci_95'][1]:.0%}); by chance it "
        f"would be {s['top5_overlap_by_chance']:.0%}.",
        "",
        f"Rule: {s['rule']}.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", default="paired", choices=["paired", "t1", "dti"])
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--features", type=Path, default=FEATURES)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    warnings.filterwarnings("ignore", category=UserWarning)
    data = load_all(args.labels, args.features, MIN_BATCH)[args.set]
    started = time.time()
    result = run(data, args.folds, args.repeats)
    result["seconds"] = round(time.time() - started, 1)
    out = args.out or RESULTS / ("explain_" + time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "explain_checks.json").write_text(json.dumps(result, indent=2))
    text = markdown(result)
    (out / "explain_checks.md").write_text(text)
    print(text)
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
