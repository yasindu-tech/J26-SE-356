"""Are the voice explanations faithful and stable? (raw-audio model; exploratory)

Every uploaded voice gets SHAP reasons ("what mattered most"). This script checks
two things about them, on the honest person-level CV (each outer fold refitted
exactly as ``pipeline.nested_cv_predict`` does, so no test person is ever seen in
training). It follows the tapping module's FT-44 and FT-45 checks.

1. **Deletion curve (faithfulness).** For each held-out person, the features the
   model uses are "deleted" one at a time (value -> training-fold median) in the
   order of that person's |SHAP|, largest first. If the explanation is faithful,
   this should change the person's score faster than deleting the same features
   in a random order, and the pooled AUC should fall faster. Compared with the
   reverse order and ``N_RANDOM`` random orders; one-sided empirical p-values.
   The random orders are drawn from the features the model USES, the harder
   baseline: drawing from all 56 would mostly delete unused features and make
   SHAP look better than it is.

2. **Stability.** (a) Across the 5 folds: the Spearman correlation between the
   folds' mean |SHAP| rankings (target 0.70, as for tapping). (b) For each held-
   out person: the fold model is refitted on ``N_REFITS`` bootstrap resamples of
   its training people, and the person's top-5 reasons are compared with the fold
   model's (Jaccard overlap, top-1 agreement, sign agreement). Stable reasons mean
   the result card would show much the same "why" if the training people had
   been slightly different. Both are also measured for measure GROUPS (pitch,
   jitter, shimmer, ...), built exactly as the desktop card builds its visible
   "What mattered most": the top-5 SHAP features summed by group, top 3 groups.
   Single features appear only in the card's technical details.

For the L1 logistic model SHAP is exact and linear, so the deletion check mainly
confirms the implementation and the median baseline; it matters more for
LightGBM, which is checked too. Labels are used only to fit the models and to
compute AUC. A faithful explanation describes what the model used, never what
causes Parkinson's disease.

Outputs (gitignored, aggregates only, no ids):
    models/voice/artifacts/xai_faithfulness.json
    models/voice/artifacts/xai_deletion_curve.png

Usage (from the repo root):
    python models/voice/src/xai_faithfulness.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import audio_evidence
import audio_features
import numpy as np
import pandas as pd
import pipeline
import splitter
import voice_module
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
MODELS: tuple[pipeline.Model, ...] = ("l1_logistic", "lightgbm")
MAX_K = 10  # deletion steps; the card shows 5 reasons
N_RANDOM = 100
N_REFITS = 20
TOP_N = voice_module.TOP_K
CARD_GROUPS = 3  # the desktop card shows the top 3 groups (voicePlain.ts plainDrivers)
GROUP_OF = {name: info[0] for name, info in audio_features.FEATURE_INFO.items()}
RANDOM_SEED = 2026
ALPHA = 0.05
STABILITY_TARGET = 0.70  # mean Spearman between folds, as in models/tapping (FT-45)
EPS = 1e-12


@dataclass
class FoldModel:
    train: np.ndarray
    test: np.ndarray
    model: Pipeline
    medians: np.ndarray  # training-fold median of every feature: the "deleted" value


# --------------------------------------------------------------------------- pure pieces


def full_shap(model: Pipeline, X: np.ndarray, names: Sequence[str]) -> np.ndarray:
    """SHAP (log-odds) for every feature, rows x features; 0 for features the model drops."""
    frame = voice_module.shap_values(model, X, names).drop(columns="_baseline")
    return frame.reindex(columns=list(names), fill_value=0.0).to_numpy(float)


def used_features(model: Pipeline) -> np.ndarray:
    """Indices of the features the model actually uses (selected, and non-zero for L1)."""
    selected = model["select"].get_support(indices=True)
    estimator = model["model"]
    if hasattr(estimator, "coef_"):
        return selected[np.abs(estimator.coef_[0]) > EPS]
    return selected


def shap_order(abs_shap: np.ndarray, used: np.ndarray, names: Sequence[str]) -> np.ndarray:
    """Per row, the used features by |SHAP| descending; ties broken by feature name."""
    used = np.asarray(used)[np.argsort(np.asarray(names)[used], kind="stable")]
    by_value = np.argsort(-abs_shap[:, used], axis=1, kind="stable")
    return used[by_value]


def random_orders(n_rows: int, used: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One random permutation of the used features per row."""
    return np.array([rng.permutation(used) for _ in range(n_rows)]).reshape(n_rows, len(used))


def delete_top(X: np.ndarray, medians: np.ndarray, order: np.ndarray, k: int) -> np.ndarray:
    """Copy of X with each row's first k features in ``order`` set to the training median."""
    out = X.copy()
    if k == 0 or order.shape[1] == 0:
        return out
    cols = order[:, : min(k, order.shape[1])]
    rows = np.repeat(np.arange(len(X)), cols.shape[1])
    out[rows, cols.ravel()] = medians[cols.ravel()]
    return out


def log_odds(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def empirical_p(value: float, random_values: np.ndarray, larger_is_better: bool) -> float:
    """One-sided p: share of random orders at least as good as SHAP (with the +1 correction)."""
    as_good = random_values >= value if larger_is_better else random_values <= value
    return float((1 + np.sum(as_good)) / (1 + len(random_values)))


def top_set(shap_row: np.ndarray, n: int = TOP_N) -> set[int]:
    """The n features with the largest |SHAP| that are non-zero."""
    nonzero = np.flatnonzero(np.abs(shap_row) > EPS)
    ranked = nonzero[np.argsort(-np.abs(shap_row[nonzero]), kind="stable")]
    return set(ranked[:n].tolist())


def card_groups(shap_row: np.ndarray, names: Sequence[str], n: int = CARD_GROUPS) -> list[str]:
    """The groups the desktop card shows: top-5 features summed by group, top n by |total|."""
    totals: dict[str, float] = {}
    for i in top_set(shap_row):
        group = GROUP_OF[names[i]]
        totals[group] = totals.get(group, 0.0) + float(shap_row[i])
    return sorted(totals, key=lambda g: -abs(totals[g]))[:n]


def jaccard(a: set, b: set) -> float:
    return 1.0 if not a and not b else len(a & b) / len(a | b)


# --------------------------------------------------------------------------- the checks


def fit_folds(
    X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, model: pipeline.Model, seed: int
) -> list[FoldModel]:
    """The outer folds of the honest nested CV, each tuned on its own training people."""
    X_arr = X.to_numpy(float)
    folds = []
    for train, test in splitter.person_folds(groups, y, seed=seed):
        fitted, _ = pipeline.fit_tuned(
            X.iloc[train], y[train], groups[train], model=model, seed=seed,
            k_grid=audio_evidence.AUDIO_K_GRID,
        )  # fmt: skip
        folds.append(FoldModel(train, test, fitted, np.median(X_arr[train], axis=0)))
    return folds


def deletion_curve(
    X: np.ndarray, y: np.ndarray, folds: list[FoldModel], names: Sequence[str], seed: int
) -> dict[str, Any]:
    """Pooled AUC and mean |change in log-odds| after deleting 0..MAX_K features per person."""
    rng = np.random.default_rng([RANDOM_SEED, seed])
    ks = range(MAX_K + 1)
    proba = {name: np.zeros((MAX_K + 1, len(y))) for name in ("shap", "reverse")}
    random_proba = np.zeros((N_RANDOM, MAX_K + 1, len(y)))
    for fold in folds:
        Xt = X[fold.test]
        used = used_features(fold.model)
        order = shap_order(np.abs(full_shap(fold.model, Xt, names)), used, names)
        orders = {"shap": order, "reverse": order[:, ::-1]}
        for name, o in orders.items():
            for k in ks:
                Xk = delete_top(Xt, fold.medians, o, k)
                proba[name][k, fold.test] = fold.model.predict_proba(Xk)[:, 1]
        for r in range(N_RANDOM):
            o = random_orders(len(Xt), used, rng)
            for k in ks:
                Xk = delete_top(Xt, fold.medians, o, k)
                random_proba[r, k, fold.test] = fold.model.predict_proba(Xk)[:, 1]

    def auc_curve(p: np.ndarray) -> np.ndarray:
        return np.array([roc_auc_score(y, p[k]) for k in ks])

    def margin_curve(p: np.ndarray) -> np.ndarray:
        return np.array([np.mean(np.abs(log_odds(p[k]) - log_odds(p[0]))) for k in ks])

    auc = {name: auc_curve(p) for name, p in proba.items()}
    margin = {name: margin_curve(p) for name, p in proba.items()}
    random_auc = np.array([auc_curve(p) for p in random_proba])
    random_margin = np.array([margin_curve(p) for p in random_proba])
    # Areas over k = 1..MAX_K: a faithful order has a LOWER AUC area, a HIGHER margin area.
    area = {name: float(curve[1:].mean()) for name, curve in auc.items()}
    random_area = random_auc[:, 1:].mean(axis=1)
    margin_area = {name: float(curve[1:].mean()) for name, curve in margin.items()}
    random_margin_area = random_margin[:, 1:].mean(axis=1)
    p_auc = empirical_p(area["shap"], random_area, larger_is_better=False)
    p_margin = empirical_p(margin_area["shap"], random_margin_area, larger_is_better=True)
    return {
        "k": list(ks),
        "auc": {
            "shap": _r(auc["shap"]),
            "reverse": _r(auc["reverse"]),
            "random_mean": _r(random_auc.mean(axis=0)),
            "random_p05": _r(np.percentile(random_auc, 5, axis=0)),
            "random_p95": _r(np.percentile(random_auc, 95, axis=0)),
        },
        "mean_abs_log_odds_change": {
            "shap": _r(margin["shap"]),
            "reverse": _r(margin["reverse"]),
            "random_mean": _r(random_margin.mean(axis=0)),
        },
        "auc_area": {
            "shap": round(area["shap"], 4),
            "reverse": round(area["reverse"], 4),
            "random_mean": round(float(random_area.mean()), 4),
            "p_value": round(p_auc, 4),
            "auc_falls_faster": p_auc < ALPHA,
        },
        "margin_area": {
            "shap": round(margin_area["shap"], 4),
            "reverse": round(margin_area["reverse"], 4),
            "random_mean": round(float(random_margin_area.mean()), 4),
            "p_value": round(p_margin, 4),
            "score_changes_faster": p_margin < ALPHA,
        },
        "n_random": N_RANDOM,
        "used_features_per_fold": [int(len(used_features(f.model))) for f in folds],
    }


def fold_stability(X: np.ndarray, folds: list[FoldModel], names: Sequence[str]) -> dict[str, Any]:
    """Spearman between the folds' mean |SHAP| rankings, on features any fold uses."""
    importance = np.array(
        [np.abs(full_shap(f.model, X[f.test], names)).mean(axis=0) for f in folds]
    )
    keep = importance.max(axis=0) > EPS
    rhos = [
        float(spearmanr(importance[a, keep], importance[b, keep]).statistic)
        for a, b in itertools.combinations(range(len(folds)), 2)
    ]
    tops = [set(np.argsort(-row)[:TOP_N].tolist()) for row in importance]
    top_pairs = [jaccard(a, b) for a, b in itertools.combinations(tops, 2)]
    overall = importance.mean(axis=0)
    group_names = sorted(set(GROUP_OF.values()))
    by_group = np.array(
        [[row[[GROUP_OF[n] == g for n in names]].sum() for g in group_names] for row in importance]
    )
    group_rhos = [
        float(spearmanr(by_group[a], by_group[b]).statistic)
        for a, b in itertools.combinations(range(len(folds)), 2)
    ]
    return {
        "group_mean_spearman": round(float(np.mean(group_rhos)), 4),
        "group_meets_target": bool(np.mean(group_rhos) >= STABILITY_TARGET),
        "top3_groups_overall": [
            group_names[i] for i in np.argsort(-by_group.mean(axis=0))[:CARD_GROUPS]
        ],
        "n_features_compared": int(keep.sum()),
        "mean_spearman": round(float(np.mean(rhos)), 4),
        "min_spearman": round(float(np.min(rhos)), 4),
        "meets_target": bool(np.mean(rhos) >= STABILITY_TARGET),
        "target": STABILITY_TARGET,
        "mean_top5_jaccard_between_folds": round(float(np.mean(top_pairs)), 4),
        "top5_overall": [names[i] for i in np.argsort(-overall)[:TOP_N]],
    }


def person_stability(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: list[FoldModel],
    names: Sequence[str], seed: int,
) -> dict[str, Any]:  # fmt: skip
    """Each held-out person's top-5 reasons: fold model vs N_REFITS bootstrap refits."""
    rng = np.random.default_rng([RANDOM_SEED, seed, 1])
    overlaps, top1, signs, group_overlaps, group_top1 = [], [], [], [], []
    for fold in folds:
        Xt = X[fold.test]
        main = full_shap(fold.model, Xt, names)
        main_tops = [top_set(row) for row in main]
        main_first = [int(np.argmax(np.abs(row))) for row in main]
        main_groups = [card_groups(row, names) for row in main]
        for _ in range(N_REFITS):
            rows = fold.train[voice_module._bootstrap_rows(groups[fold.train], rng)]
            refit = clone(fold.model).fit(X[rows], y[rows])
            other = full_shap(refit, Xt, names)
            for i, row in enumerate(other):
                overlaps.append(jaccard(main_tops[i], top_set(row)))
                shown = card_groups(row, names)
                group_overlaps.append(jaccard(set(main_groups[i]), set(shown)))
                group_top1.append(bool(shown) and shown[0] == main_groups[i][0])
                top1.append(main_first[i] == int(np.argmax(np.abs(row))))
                shared = list(main_tops[i] & top_set(row))
                if shared:
                    signs.append(np.mean(np.sign(main[i, shared]) == np.sign(row[shared])))
    return {
        "n_refits_per_fold": N_REFITS,
        "mean_top5_jaccard": round(float(np.mean(overlaps)), 4),
        "top1_same_share": round(float(np.mean(top1)), 4),
        "shared_reasons_same_direction_share": round(float(np.mean(signs)), 4),
        "card_top3_groups_jaccard": round(float(np.mean(group_overlaps)), 4),
        "card_top_group_same_share": round(float(np.mean(group_top1)), 4),
    }


def _r(values: np.ndarray) -> list[float]:
    return [round(float(v), 4) for v in values]


# --------------------------------------------------------------------------- run and report


def run(table: pd.DataFrame, seed: int = pipeline.SEED) -> dict[str, Any]:
    if table.person.duplicated().any():
        raise ValueError("expected one recording per person; average per person first")
    names = list(audio_features.FEATURE_NAMES)
    X = table[names]
    y = table.label.to_numpy(int)
    groups = table.person.to_numpy()
    X_arr = X.to_numpy(float)
    result: dict[str, Any] = {
        "status": "exploratory",
        "metric_unit": "person",
        "n_people": int(len(table)),
        "deleted_value": "training-fold median",
        "random_orders_drawn_from": "features the model uses",
        "models": {},
    }
    for model in MODELS:
        print(f"  {model}: refitting the 5 outer folds ...", flush=True)
        folds = fit_folds(X, y, groups, model, seed)
        print(f"  {model}: deletion curve ({N_RANDOM} random orders) ...", flush=True)
        deletion = deletion_curve(X_arr, y, folds, names, seed)
        print(f"  {model}: stability ({N_REFITS} refits per fold) ...", flush=True)
        result["models"][model] = {
            "deletion": deletion,
            "stability_between_folds": fold_stability(X_arr, folds, names),
            "stability_per_person": person_stability(X_arr, y, groups, folds, names, seed),
        }
    return result


def plot(result: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(result["models"])
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4.2), squeeze=False)
    for ax, model in zip(axes[0], models, strict=True):
        d = result["models"][model]["deletion"]
        k, auc = d["k"], d["auc"]
        ax.fill_between(k, auc["random_p05"], auc["random_p95"], alpha=0.25, label="random 5-95%")
        ax.plot(k, auc["random_mean"], "--", label="random mean")
        ax.plot(k, auc["shap"], "o-", label="SHAP order (largest first)")
        ax.plot(k, auc["reverse"], "s:", label="reverse order")
        ax.axhline(0.5, color="grey", lw=0.8)
        ax.set_title(f"{model}: p = {d['auc_area']['p_value']:.3f}")
        ax.set_xlabel("features deleted per person (value -> training median)")
        ax.set_ylabel("pooled AUC")
        ax.legend(fontsize=8)
    fig.suptitle("Voice deletion curve. What the model used; not a cause.", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args(argv)
    table = audio_evidence.load_passed()
    print(f"XAI faithfulness on {len(table)} people ...", flush=True)
    result = run(table)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS / "xai_faithfulness.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    plot(result, ARTIFACTS / "xai_deletion_curve.png")
    for model, r in result["models"].items():
        a, m = r["deletion"]["auc_area"], r["deletion"]["margin_area"]
        s, p = r["stability_between_folds"], r["stability_per_person"]
        print(
            f"{model}: AUC area SHAP {a['shap']:.3f} vs random {a['random_mean']:.3f} "
            f"(p {a['p_value']:.3f}); score change SHAP {m['shap']:.3f} vs random "
            f"{m['random_mean']:.3f} (p {m['p_value']:.3f}); folds rho {s['mean_spearman']:.2f}; "
            f"person top-5 overlap {p['mean_top5_jaccard']:.2f}, top-1 same {p['top1_same_share']:.2f}"
        )
        print(
            f"  card groups: folds rho {s['group_mean_spearman']:.2f}; person top-3 groups "
            f"overlap {p['card_top3_groups_jaccard']:.2f}, top group same "
            f"{p['card_top_group_same_share']:.2f}; shared reasons same direction "
            f"{p['shared_reasons_same_direction_share']:.2f}"
        )
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
