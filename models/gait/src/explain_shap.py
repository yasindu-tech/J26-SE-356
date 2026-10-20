"""GAIT-23: SHAP explanations of the saved severity model.

Run from the repository root after ``train_xgboost.py``::

    python models/gait/src/explain_shap.py

Explains the model bundle with SHAP's TreeExplainer, in log-odds of the class the model predicted.
Writes a global importance chart, a top-three chart for three example walks and the tables behind
them to ``artifacts/models/shap/`` (git-ignored). ``Explainer.explain`` gives the top three inputs
of one walk, for the plain-words step and ``predict_video``.
Exit code 0 = ok, 1 = SHAP does not add up to the model's output, 2 = an input is missing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import train_xgboost as tx
from explain_words import TOP, WordedExplanation, describe
from gait_extract import FEATURE_NAMES
from make_folds import CLASS_LABELS
from model_bundle import Bundle, BundleError, load_bundle

DEFAULT_OUT = tx.GAIT_DIR / "artifacts" / "models" / "shap"
DEFAULT_BUNDLE = tx.DEFAULT_OUT / "bundle"
ADDITIVITY_TOL = 1e-4
CLASS_COLORS = ("#0072B2", "#E69F00", "#009E73")
TOWARDS, AWAY = "#0072B2", "#999999"


@dataclass(frozen=True)
class Contribution:
    feature: str
    value: float
    shap: float

    @property
    def direction(self) -> str:
        return "towards" if self.shap > 0 else "away"


@dataclass(frozen=True)
class Explanation:
    predicted_class: int
    probabilities: tuple[float, ...]
    top: tuple[Contribution, ...]


class Explainer:
    """TreeExplainer of a bundle's model."""

    def __init__(self, bundle: Bundle) -> None:
        import shap

        self.bundle = bundle
        self.tree = shap.TreeExplainer(bundle.model)

    def values(self, x: np.ndarray) -> np.ndarray:
        """SHAP values (walks, inputs, classes), in log-odds."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return tx.shap_array(self.tree.shap_values(x))

    def additivity_error(self, x: np.ndarray) -> float:
        """Largest gap between base value + SHAP and the model's raw output."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        raw = self.bundle.model.predict(x, output_margin=True)
        total = self.values(x).sum(axis=1) + np.asarray(self.tree.expected_value)
        return float(np.abs(total - raw).max())

    def explain(self, x: np.ndarray) -> Explanation:
        """The inputs that moved the predicted class's score most for one walk."""
        x = np.asarray(x, dtype=float).reshape(1, -1)
        proba = self.bundle.predict_proba(x)[0]
        predicted = int(proba.argmax())
        shap_values = self.values(x)[0, :, predicted]
        order = np.argsort(-np.abs(shap_values), kind="stable")[:TOP]
        top = tuple(
            Contribution(self.bundle.features[j], float(x[0, j]), float(shap_values[j]))
            for j in order
        )
        return Explanation(predicted, tuple(float(p) for p in proba), top)

    def in_words(self, x: np.ndarray) -> WordedExplanation:
        """``explain`` with one plain-words sentence per input (needs the bundle's reference)."""
        return describe(self.explain(x), self.bundle.reference)


# --------------------------------------------------------------------------- tables
def global_importance(values: np.ndarray) -> pd.DataFrame:
    """Mean |SHAP| per input and class, sorted by the mean over classes."""
    mean_abs = np.abs(values).mean(axis=0)
    out = pd.DataFrame(
        mean_abs, index=list(FEATURE_NAMES), columns=[CLASS_LABELS[c] for c in range(3)]
    )
    out["overall"] = out.mean(axis=1)
    return out.sort_values("overall", ascending=False).rename_axis("feature").reset_index()


def top_features(values: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Indices (walks, TOP) of the inputs with the largest |SHAP| for each walk's predicted class."""
    own = values[np.arange(len(values)), :, predicted]
    return np.argsort(-np.abs(own), axis=1, kind="stable")[:, :TOP]


def top_frequency(top: np.ndarray, predicted: np.ndarray) -> pd.DataFrame:
    """Share of walks, per predicted class, in which each input is among the top three."""
    out = {}
    for c in range(3):
        rows = top[predicted == c]
        out[CLASS_LABELS[c]] = [
            float((rows == j).any(axis=1).mean()) if len(rows) else float("nan")
            for j in range(len(FEATURE_NAMES))
        ]
    return pd.DataFrame(out, index=list(FEATURE_NAMES)).rename_axis("feature").reset_index()


def rank_agreement(saved: pd.DataFrame, reference: pd.DataFrame) -> dict[str, float]:
    """Spearman correlation of the input ranking with the out-of-fold table from training."""
    from scipy.stats import spearmanr

    saved, reference = saved.set_index("feature"), reference.set_index("feature")
    reference = reference.loc[saved.index]
    return {col: float(spearmanr(saved[col], reference[col]).statistic) for col in saved.columns}


def pick_examples(table: pd.DataFrame, oof: pd.DataFrame, saved_pred: np.ndarray) -> dict[int, str]:
    """One walk per class: right out of fold, right in the saved model, median confidence.

    The saved model has seen every walk, so being right in it proves little; the out-of-fold
    prediction (a model that never saw the subject) decides, and its confidence picks the walk.
    """
    probs = oof[oof["model"] == "xgboost"].groupby("walk")[list(tx.PROBA_COLUMNS)].mean()
    frame = table[["walk", "severity"]].assign(saved=saved_pred)
    frame = frame.join(probs, on="walk")
    frame["oof"] = frame[list(tx.PROBA_COLUMNS)].to_numpy().argmax(axis=1)
    frame["confidence"] = frame[list(tx.PROBA_COLUMNS)].max(axis=1)
    picked = {}
    for c in range(3):
        ok = frame[(frame["severity"] == c) & (frame["oof"] == c) & (frame["saved"] == c)]
        if ok.empty:
            continue
        gap = (ok["confidence"] - ok["confidence"].median()).abs()
        picked[c] = str(ok.loc[gap.sort_values(kind="stable").index[0], "walk"])
    return picked


# --------------------------------------------------------------------------- charts
def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_global(importance: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    ranked = importance.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    left = np.zeros(len(ranked))
    for c in range(3):
        width = ranked[CLASS_LABELS[c]].to_numpy()
        ax.barh(ranked["feature"], width, left=left, color=CLASS_COLORS[c], label=CLASS_LABELS[c])
        left += width
    ax.set_xlabel("mean |SHAP value| per class, stacked (log-odds)")
    ax.set_title("What the model used, over all walks", loc="left", fontsize=11)
    ax.legend(title="class explained", frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.01,
        0.01,
        "The saved model has seen every walk: this describes the model, not a result.",
        fontsize=7.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_walks(examples: list[tuple[str, int, Explanation]], path: Path) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, len(examples), figsize=(5.2 * len(examples), 3.6), squeeze=False)
    for ax, (_walk, label, exp) in zip(axes[0], examples, strict=True):
        shown = exp.top[::-1]
        names = [f"{c.feature}\n= {c.value:.3g}" for c in shown]
        ax.barh(
            names, [c.shap for c in shown], color=[TOWARDS if c.shap > 0 else AWAY for c in shown]
        )
        for i, c in enumerate(shown):
            ax.text(
                c.shap,
                i,
                f" {c.shap:+.2f} ",
                va="center",
                ha="left" if c.shap > 0 else "right",
                fontsize=8,
            )
        ax.axvline(0, color="#333333", lw=0.8)
        ax.margins(x=0.25)
        ax.set_title(
            f"{CLASS_LABELS[label]} walk, model's pick: {CLASS_LABELS[exp.predicted_class]}\n"
            f"saved-model score {exp.probabilities[exp.predicted_class]:.2f}",
            loc="left",
            fontsize=9,
        )
        ax.set_xlabel(f"SHAP, log-odds of {CLASS_LABELS[exp.predicted_class]}")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "The three inputs the model used most (blue: towards its pick, grey: away)",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.text(
        0.01,
        0.01,
        "Walks: right out of fold, median out-of-fold score, per class. The saved model has seen "
        "them; this shows what the model used, not a cause. The first input is firm; the order "
        "of the others is approximate.",
        fontsize=7.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- command line
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SHAP explanations of the saved model (GAIT-23)")
    parser.add_argument("--table", type=Path, default=tx.DEFAULT_TABLE)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--oof", type=Path, default=tx.DEFAULT_OUT / "oof_predictions.csv")
    parser.add_argument("--shap-table", type=Path, default=tx.DEFAULT_OUT / "shap_mean_abs.csv")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    for path, hint in (
        (args.table, "build_feature_table.py"),
        (args.oof, "train_xgboost.py"),
        (args.bundle / "bundle.json", "train_xgboost.py"),
    ):
        if not path.is_file():
            print(f"MISSING: {path}\nRun {hint} first.")
            return 2
    try:
        bundle = load_bundle(args.bundle)
        bundle.reference  # noqa: B018 - refuses a bundle without the plain-words reference
    except BundleError as err:
        print(f"The bundle cannot be used: {err}")
        return 2
    table = tx.read_training_table(args.table)
    if bundle.info.get("feature_table") not in (None, tx.table_fingerprint(args.table)):
        print("The bundle was trained on a different feature table. Rerun train_xgboost.py.")
        return 2

    x = table[list(FEATURE_NAMES)].to_numpy(dtype=float)
    explainer = Explainer(bundle)
    values = explainer.values(x)
    error = explainer.additivity_error(x)
    saved_pred = bundle.predict_proba(x).argmax(axis=1)
    print(f"Model {bundle.version}, {len(table)} walks. SHAP additivity error {error:.1e}.")

    importance = global_importance(values)
    frequency = top_frequency(top_features(values, saved_pred), saved_pred)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(args.out_dir / "global_importance.csv", index=False, float_format="%.6f")
    frequency.to_csv(args.out_dir / "top3_frequency.csv", index=False, float_format="%.4f")
    plot_global(importance, args.out_dir / "global_importance.png")

    print(
        f"\n{'feature':<26}"
        + "".join(f"{CLASS_LABELS[c]:>12}" for c in range(3))
        + f"{'overall':>10}"
    )
    for _, r in importance.iterrows():
        print(
            f"{r['feature']:<26}"
            + "".join(f"{r[CLASS_LABELS[c]]:>12.3f}" for c in range(3))
            + f"{r['overall']:>10.3f}"
        )
    if args.shap_table.is_file():
        agree = rank_agreement(importance, pd.read_csv(args.shap_table))
        print(
            "\nInput ranking against the out-of-fold table from training (Spearman): "
            + ", ".join(f"{k} {v:.2f}" for k, v in agree.items())
        )

    oof = pd.read_csv(args.oof, dtype={"walk": str})
    picked = pick_examples(table, oof, saved_pred)
    examples = []
    rows = []
    for label, walk in picked.items():
        i = int(table.index[table["walk"] == walk][0])
        words = explainer.in_words(x[i])
        exp = explainer.explain(x[i])
        examples.append((walk, label, exp))
        for rank, (c, e) in enumerate(zip(exp.top, words.entries, strict=True), start=1):
            rows.append(
                {
                    "label_class": CLASS_LABELS[label],
                    "walk": walk,
                    "predicted": CLASS_LABELS[exp.predicted_class],
                    "rank": rank,
                    "feature": c.feature,
                    "value": c.value,
                    "shap": c.shap,
                    "direction": c.direction,
                    "strength": e.strength,
                    "sentence": e.sentence,
                }
            )
    missing = [CLASS_LABELS[c] for c in range(3) if c not in picked]
    if missing:
        print(
            f"\nNo example walk for {', '.join(missing)}: none right both out of fold and in the saved model."
        )
    if examples:
        pd.DataFrame(rows).to_csv(
            args.out_dir / "example_walks.csv", index=False, float_format="%.6f"
        )
        plot_walks(examples, args.out_dir / "example_walks.png")
        print("\nExample walks:")
        for walk, label, _ in examples:
            print(f"  {walk} ({CLASS_LABELS[label]}):")
            for sentence in explainer.in_words(
                x[int(table.index[table["walk"] == walk][0])]
            ).sentences:
                print(f"    {sentence}")
    print(f"\nWrote {args.out_dir}.")

    if error > ADDITIVITY_TOL:
        print(f"FAILED: SHAP values plus base value differ from the model output by {error:.1e}.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
