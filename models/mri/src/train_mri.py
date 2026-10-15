"""Cross-validated training of the MRI models, with the baseline ladder and both controls.

For each training set (see training_data.py: "t1", "dti", "paired") this runs repeated,
stratified, subject-wise K-fold (default 5 x 10). Inside every fold, and only on that fold's
training people:

1. the feature contract is checked for every model's inputs;
2. ComBat is fitted per modality (T1 columns with the T1 scanner group, DTI columns with the
   DTI scanner group; age and sex protected as covariates, diagnosis never), then applied
   unchanged to the test people;
3. every rung of the baseline ladder is fitted and scores the test people:

       age_sex       logistic regression on age and sex only
       best_single   logistic regression on the one feature that separates the training
                     people best (chosen inside the fold)
       lgbm_t1       LightGBM on T1 features
       lgbm_dti      LightGBM on DTI features
       lgbm_t1dti    LightGBM on T1 and DTI features together ("paired" set only)

Every person is scored by models that never saw them. Reported per model: AUC and balanced
accuracy as mean +- SD over repeats, a bootstrap 95% CI over people (scores averaged over
repeats), sensitivity, specificity, PPV at 1/2/5% prevalence, and the paired AUC difference
against age + sex with its CI (the primary success criterion: that CI excludes zero).

The two controls run on every set:

* site-prediction gate: the scanner group must be no easier to guess from the out-of-fold
  harmonised features than with shuffled scanner labels;
* label-permutation control: the whole pipeline (folds, ComBat, model) on shuffled diagnosis
  labels must score at chance.

The report is always written, so a failure can be read. If either control fails, the report
is marked INVALID and the run exits with an error: those numbers must not be quoted.

LightGBM uses fixed, small-data settings (shallow trees, many-sample leaves). There is no
hyperparameter tuning in this version; tuning would need its own inner loop.

Progress and results also go to the monitoring dashboard (pipeline_monitor.py, page
/train): the step each set is on, how the people were split, every model's metrics and both
controls. Only counts and metrics are sent, never per-person rows.

Command line (writes data/processed/results/<time>/):

    python models/mri/src/train_mri.py                       # every set whose features exist
    python models/mri/src/train_mri.py --sets t1 --repeats 2  # quick look
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from folds import assert_subject_disjoint, subject_folds
from harmonise import (
    ComBat,
    ControlResult,
    label_permutation_control,
    out_of_fold_harmonise,
    site_predictability,
    site_prediction_gate,
)
from metrics import auc, balanced_accuracy, paired_difference_ci, summarise
from mri_contract import check_mri_features
from progress import NullRecorder, RunRecorder
from training_data import DEFAULT_LABELS, FEATURES, MIN_BATCH, SETS, ModelData, load_all

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "data" / "processed" / "results"
LGBM_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 7,
    "max_depth": 3,
    "min_child_samples": 5,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "class_weight": "balanced",
    "n_jobs": 1,
    "verbose": -1,
}
MAIN_MODEL = {"t1": "lgbm_t1", "dti": "lgbm_dti", "paired": "lgbm_t1dti"}
TRAIN_STEPS = ["cv", "metrics", "site_gate", "permutation"]


class ResultsInvalid(RuntimeError):
    """A mandatory control failed; the numbers in this run must not be quoted."""


# ------------------------------------------------------------------ models


def models_for(data: ModelData) -> dict[str, list[str] | None]:
    """Model name -> feature groups it uses (None = age and sex only)."""
    out: dict[str, list[str] | None] = {"age_sex": None, "best_single": list(data.groups)}
    for g in data.groups:
        out[f"lgbm_{g}"] = [g]
    if len(data.groups) > 1:
        out["lgbm_" + "".join(data.groups)] = list(data.groups)
    return out


def _logistic():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)
    )


def _lgbm(seed: int):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(random_state=seed, **LGBM_PARAMS)


def best_single_feature(x: pd.DataFrame, y: np.ndarray) -> str:
    """The column whose values separate the classes best in these (training) people."""
    scores = {c: abs(auc(y, x[c].to_numpy()) - 0.5) for c in x.columns}
    return max(scores, key=lambda c: (scores[c], c))


# ------------------------------------------------------------------ one fold


def harmonise_fold(
    data: ModelData, train: np.ndarray, test: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ComBat per modality, fitted on the training people only, applied to the test people."""
    cov = data.covariates()
    tr_parts, te_parts = [], []
    for g, cols in data.groups.items():
        x = data.features[cols].to_numpy(dtype=float)
        b = data.batches[g]
        combat = ComBat().fit(x[train], b[train], cov[train])
        tr_parts.append(
            pd.DataFrame(combat.transform(x[train], b[train], cov[train]), columns=cols)
        )
        te_parts.append(pd.DataFrame(combat.transform(x[test], b[test], cov[test]), columns=cols))
    return pd.concat(tr_parts, axis=1), pd.concat(te_parts, axis=1)


def fit_predict_fold(
    data: ModelData,
    train: np.ndarray,
    test: np.ndarray,
    models: dict[str, list[str] | None],
    seed: int,
) -> dict[str, np.ndarray]:
    """Test-set probabilities of every model, all learned from the training people only."""
    assert_subject_disjoint(train, test, data.subjects)
    y_tr = data.y[train]
    h_tr, h_te = harmonise_fold(data, train, test)
    out = {}
    for name, groups in models.items():
        if groups is None:
            check_mri_features(["age", "sex"], allow_demographics=True)
            cov = data.covariates()
            m = _logistic().fit(cov[train], y_tr)
            out[name] = m.predict_proba(cov[test])[:, 1]
            continue
        cols = [c for g in groups for c in data.groups[g]]
        check_mri_features(cols)
        if name == "best_single":
            col = best_single_feature(h_tr[cols], y_tr)
            m = _logistic().fit(h_tr[[col]].to_numpy(), y_tr)
            out[name] = m.predict_proba(h_te[[col]].to_numpy())[:, 1]
        else:
            m = _lgbm(seed).fit(h_tr[cols], y_tr)
            out[name] = m.predict_proba(h_te[cols])[:, 1]
    return out


def fold_groups(data: ModelData) -> np.ndarray:
    """The ComBat batches of every modality, joined into one key per person.

    Folds deal each of these groups evenly (``spread`` in folds.subject_folds), so every
    training part keeps enough people of every scanner group for ComBat, and no test person
    comes from a scanner group the training part has not seen. Stratifying on the label
    alone let a small group cluster in one test fold: a group of 3 could leave 1 person in
    training, which ComBat refuses.
    """
    parts = [np.asarray(b).astype(str) for b in data.batches.values()]
    key = parts[0]
    for p in parts[1:]:
        key = np.char.add(np.char.add(key, "|"), p)
    return key


def make_folds(data: ModelData, labels: np.ndarray, n_splits: int, n_repeats: int, seed: int):
    """Subject-wise folds with every scanner group spread evenly (see ``fold_groups``)."""
    return subject_folds(data.subjects, labels, n_splits, n_repeats, seed, spread=fold_groups(data))


def run_cv(
    data: ModelData,
    models: dict[str, list[str] | None],
    y: np.ndarray | None = None,
    n_splits: int = 5,
    n_repeats: int = 10,
    seed: int = 0,
    on_fold=None,
) -> dict[str, np.ndarray]:
    """Out-of-fold probabilities, shape (repeats, people), for every model.

    ``y`` replaces the true labels (used by the permutation control); the folds are then
    built on the replaced labels too, as a real run would. ``on_fold(done, total)`` is
    called after each fold (progress reporting only).
    """
    labels = data.y if y is None else np.asarray(y)
    run = replace(data, y=labels)
    folds = make_folds(run, labels, n_splits, n_repeats, seed)
    probs = {m: np.full((n_repeats, run.n), np.nan) for m in models}
    for i, f in enumerate(folds, 1):
        pred = fit_predict_fold(run, f.train_idx, f.test_idx, models, seed + f.repeat)
        for m, p in pred.items():
            probs[m][f.repeat, f.test_idx] = p
        if on_fold is not None:
            on_fold(i, len(folds))
    for m, p in probs.items():
        if np.isnan(p).any():
            raise RuntimeError(f"{m}: some people were never scored")
    return probs


# ------------------------------------------------------------------ summarising


@dataclass
class ModelReport:
    model: str
    auc_mean: float
    auc_sd: float
    bal_acc_mean: float
    bal_acc_sd: float
    auc_ci: tuple[float, float]
    bal_acc_ci: tuple[float, float]
    sensitivity: float
    specificity: float
    ppv: dict[str, float]
    auc_vs_age_sex: tuple[float, float, float] | None = None  # diff, low, high

    @property
    def beats_age_sex(self) -> bool | None:
        return None if self.auc_vs_age_sex is None else self.auc_vs_age_sex[1] > 0


@dataclass
class SetReport:
    name: str
    n: int
    n_pd: int
    n_features: int
    scanner_groups: dict[str, dict[str, int]]
    models: list[ModelReport] = field(default_factory=list)
    site_gate: dict[str, dict] = field(default_factory=dict)
    permutation: dict | None = None
    split: dict = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        gates_ok = all(g["passed"] for g in self.site_gate.values())
        return gates_ok and bool(self.permutation and self.permutation["passed"])


def summarise_models(y: np.ndarray, probs: dict[str, np.ndarray], n_boot: int, seed: int):
    reports = []
    base = probs["age_sex"].mean(axis=0)
    for m, p in probs.items():
        per_repeat_auc = [auc(y, r) for r in p]
        per_repeat_ba = [balanced_accuracy(y, r) for r in p]
        pooled = p.mean(axis=0)
        s = summarise(y, pooled, n_boot=n_boot, seed=seed)
        vs = None
        if m != "age_sex":
            vs = paired_difference_ci(y, pooled, base, auc, n_boot=n_boot, seed=seed)
        reports.append(
            ModelReport(
                model=m,
                auc_mean=float(np.mean(per_repeat_auc)),
                auc_sd=float(np.std(per_repeat_auc)),
                bal_acc_mean=float(np.mean(per_repeat_ba)),
                bal_acc_sd=float(np.std(per_repeat_ba)),
                auc_ci=s.auc_ci,
                bal_acc_ci=s.balanced_accuracy_ci,
                sensitivity=s.sensitivity,
                specificity=s.specificity,
                ppv={f"{k:.0%}": v for k, v in s.ppv.items()},
                auc_vs_age_sex=vs,
            )
        )
    return reports


def _control(result: ControlResult) -> dict:
    return {
        "score": result.score,
        "threshold": result.threshold,
        "passed": bool(result.passed),
        "null_mean": float(np.mean(result.null_scores)),
    }


def run_set(
    data: ModelData,
    n_splits: int = 5,
    n_repeats: int = 10,
    n_permutations: int = 20,
    permutation_repeats: int = 2,
    gate_permutations: int = 100,
    n_boot: int = 2000,
    seed: int = 0,
    recorder: RunRecorder | NullRecorder | None = None,
) -> tuple[SetReport, dict[str, np.ndarray]]:
    rec = recorder or NullRecorder()
    name = data.name
    models = models_for(data)
    with rec.step(name, "cv"):
        probs = run_cv(
            data,
            models,
            None,
            n_splits,
            n_repeats,
            seed,
            on_fold=lambda i, n: rec.note(name, "cv", f"fold {i} of {n}", every_s=2.0),
        )
    with rec.step(name, "metrics"):
        report = SetReport(
            name=name,
            n=data.n,
            n_pd=int(data.y.sum()),
            n_features=data.features.shape[1],
            scanner_groups={
                g: {str(k): int(v) for k, v in pd.Series(b).value_counts().items()}
                for g, b in data.batches.items()
            },
            models=summarise_models(data.y, probs, n_boot, seed),
        )

    # site gate on out-of-fold harmonised features (one repeat of the same kind of folds;
    # with the same seed this is exactly repeat 0 of the CV above)
    folds = make_folds(data, data.y, n_splits, 1, seed)
    report.split = split_summary(data, folds, n_splits, n_repeats)
    pairs = [(f.train_idx, f.test_idx) for f in folds]
    with rec.step(name, "site_gate"):
        for g, cols in data.groups.items():
            x = data.features[cols].to_numpy(dtype=float)
            oof, _ = out_of_fold_harmonise(x, data.batches[g], pairs, data.covariates())
            gate = site_prediction_gate(
                oof, data.batches[g], gate_permutations, seed=seed, raise_on_fail=False
            )
            report.site_gate[g] = _control(gate) | {
                "before_harmonisation": site_predictability(x, data.batches[g], seed)
            }

    # label permutation control: the whole pipeline for the main model on shuffled labels
    main = MAIN_MODEL[name]
    done = [0]

    def evaluate(y_perm: np.ndarray) -> float:
        p = run_cv(data, {main: models[main]}, y_perm, n_splits, permutation_repeats, seed)[main]
        done[0] += 1
        rec.note(name, "permutation", f"shuffle {done[0]} of {n_permutations}", every_s=2.0)
        return float(np.mean([balanced_accuracy(y_perm, r) for r in p]))

    with rec.step(name, "permutation"):
        perm = label_permutation_control(
            evaluate, data.y, data.subjects, n_permutations, seed=seed, raise_on_fail=False
        )
    report.permutation = _control(perm) | {"model": main}
    rec.report(name, dashboard_summary(report))
    return report, probs


def split_summary(data: ModelData, folds, n_splits: int, n_repeats: int) -> dict:
    """How the people were divided: sizes of the training and test parts of each fold."""
    train = [len(f.train_idx) for f in folds]
    test = [len(f.test_idx) for f in folds]
    test_pd = [int(data.y[f.test_idx].sum()) for f in folds]
    return {
        "folds": n_splits,
        "repeats": n_repeats,
        "fits_per_model": n_splits * n_repeats,
        "train_people": [min(train), max(train)],
        "test_people": [min(test), max(test)],
        "test_pd": [min(test_pd), max(test_pd)],
        "min_train_per_scanner_group": min(
            int(pd.Series(fold_groups(data)[f.train_idx]).value_counts().min()) for f in folds
        ),
    }


def dashboard_summary(report: SetReport) -> dict:
    """What the dashboard shows for one set: counts, split, metrics and controls only."""
    summary = asdict(report) | {"valid": report.valid, "main_model": MAIN_MODEL[report.name]}
    return json.loads(json.dumps(summary, default=float))


# ------------------------------------------------------------------ report


def _span(pair) -> str:
    return str(pair[0]) if pair[0] == pair[1] else f"{pair[0]}-{pair[1]}"


def _fmt_ci(ci) -> str:
    return f"{ci[0]:.2f} to {ci[1]:.2f}"


def markdown(reports: list[SetReport], settings: dict) -> str:
    lines = ["# MRI model results", ""]
    lines.append(f"Run: {settings['time']}  |  code: {settings['commit']}")
    lines.append(
        f"{settings['folds']}-fold x {settings['repeats']} repeats, subject-wise. "
        "Every score comes from models that never saw that person."
    )
    lines.append("")
    for r in reports:
        status = "VALID" if r.valid else "INVALID: a mandatory control failed, do not quote"
        lines += [f"## Set: {r.name} ({status})", ""]
        lines.append(
            f"{r.n} people ({r.n_pd} PD, {r.n - r.n_pd} control), {r.n_features} features. "
            f"Scanner groups: {r.scanner_groups}"
        )
        if r.split:
            sp = r.split
            lines.append(
                f"Split: each fold trains on {_span(sp['train_people'])} people and tests on "
                f"{_span(sp['test_people'])} ({_span(sp['test_pd'])} PD)."
            )
        lines += [
            "",
            "| Model | AUC mean +- SD | AUC 95% CI | Bal. acc mean +- SD | Bal. acc 95% CI "
            "| Sens | Spec | PPV 1% / 2% / 5% | AUC vs age+sex (95% CI) |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for m in r.models:
            vs = "baseline"
            if m.auc_vs_age_sex is not None:
                d, lo, hi = m.auc_vs_age_sex
                vs = f"{d:+.2f} ({lo:+.2f} to {hi:+.2f})" + (" better" if m.beats_age_sex else "")
            ppv = " / ".join(f"{v:.3f}" for v in m.ppv.values())
            lines.append(
                f"| {m.model} | {m.auc_mean:.2f} +- {m.auc_sd:.2f} | {_fmt_ci(m.auc_ci)} "
                f"| {m.bal_acc_mean:.2f} +- {m.bal_acc_sd:.2f} | {_fmt_ci(m.bal_acc_ci)} "
                f"| {m.sensitivity:.2f} | {m.specificity:.2f} | {ppv} | {vs} |"
            )
        lines += ["", "Controls:", ""]
        for g, gate in r.site_gate.items():
            lines.append(
                f"- Site gate ({g}): scanner guessed with balanced accuracy "
                f"{gate['before_harmonisation']:.2f} before and {gate['score']:.2f} after "
                f"harmonisation; shuffled-label threshold {gate['threshold']:.2f} -> "
                f"{'pass' if gate['passed'] else 'FAIL'}"
            )
        if r.permutation:
            p = r.permutation
            lines.append(
                f"- Label permutation ({p['model']}): shuffled labels score {p['score']:.2f} "
                f"on average; limit {p['threshold']:.2f} -> {'pass' if p['passed'] else 'FAIL'}"
            )
        lines.append("")
    lines.append(
        "Balanced accuracy uses a 0.5 probability threshold. PPV is computed from sensitivity "
        "and specificity at screening prevalence, not at the research-set prevalence."
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    out_dir: Path,
    reports: list[SetReport],
    probs: dict[str, dict[str, np.ndarray]],
    sets: dict[str, ModelData],
    settings: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "settings": settings,
        "sets": [asdict(r) | {"valid": r.valid} for r in reports],
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, default=str))
    (out_dir / "report.md").write_text(markdown(reports, settings))
    for name, p in probs.items():
        data = sets[name]
        df = pd.DataFrame({"subject": data.subjects, "y": data.y})  # pseudonyms only
        for m, arr in p.items():
            df[m] = arr.mean(axis=0)
        df.to_csv(out_dir / f"predictions_{name}.csv", index=False)


def enforce_controls(reports: list[SetReport]) -> None:
    bad = [r.name for r in reports if not r.valid]
    if bad:
        raise ResultsInvalid(
            f"mandatory control failed for set(s) {bad}; the report is marked INVALID and "
            "its numbers must not be quoted. See the Controls section."
        )


def _commit() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO
            ).stdout.strip()
            or "unknown"
        )
    except OSError:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    # sklearn can warn about strata smaller than the number of folds; the folds here are
    # dealt by scanner group (fold_groups), so the warning is noise.
    warnings.filterwarnings("ignore", message="The least populated class", category=UserWarning)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sets", nargs="+", choices=SETS, help="default: every set with features")
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--features", type=Path, default=FEATURES)
    ap.add_argument("--out", type=Path, default=None, help="default: data/processed/results/<time>")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--permutations", type=int, default=20)
    ap.add_argument("--permutation-repeats", type=int, default=2)
    ap.add_argument("--gate-permutations", type=int, default=100)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--min-batch", type=int, default=MIN_BATCH)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--events-url", default=None, help="send progress to a remote dashboard")
    args = ap.parse_args(argv)

    sets = load_all(args.labels, args.features, args.min_batch)
    chosen = [s for s in (args.sets or SETS) if s in sets]
    if not chosen:
        print("error: no feature tables found for the requested sets", file=sys.stderr)
        return 2
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or RESULTS / stamp
    settings = {
        "time": stamp,
        "commit": _commit(),
        "folds": args.folds,
        "repeats": args.repeats,
        "permutations": args.permutations,
        "permutation_repeats": args.permutation_repeats,
        "gate_permutations": args.gate_permutations,
        "bootstrap": args.bootstrap,
        "min_batch": args.min_batch,
        "seed": args.seed,
        "lightgbm": LGBM_PARAMS,
    }
    reports, probs = [], {}
    rec = RunRecorder("train", chosen, TRAIN_STEPS, hash_ids=False, events_url=args.events_url)
    print("watch live: python models/mri/src/pipeline_monitor.py, then open /train")
    for name in chosen:
        print(f"[{name}] {sets[name].n} people: cross-validating, then the two controls ...")
        rec.subject_start(name)
        try:
            r, p = run_set(
                sets[name],
                args.folds,
                args.repeats,
                args.permutations,
                args.permutation_repeats,
                args.gate_permutations,
                args.bootstrap,
                args.seed,
                recorder=rec,
            )
        except Exception as e:
            rec.subject_end(name, "failed", type(e).__name__)
            rec.run_end()
            raise
        main_model = next(m for m in r.models if m.model == MAIN_MODEL[name])
        rec.subject_end(
            name,
            "ok" if r.valid else "failed",
            "" if r.valid else "a mandatory control failed: do not quote",
            metrics={"auc": main_model.auc_mean, "balanced_accuracy": main_model.bal_acc_mean},
        )
        reports.append(r)
        probs[name] = p
        print(f"[{name}] done ({'valid' if r.valid else 'INVALID'})")
    rec.run_end()
    write_outputs(out_dir, reports, probs, sets, settings)
    print(f"report: {out_dir / 'report.md'}")
    try:
        enforce_controls(reports)
    except ResultsInvalid as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
