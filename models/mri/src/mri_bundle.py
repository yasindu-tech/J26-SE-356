"""The frozen MRI model bundle that predict_scan.py loads to score one new person.

train_mri.py measures how well the pipeline works (cross-validated, with both controls).
This file builds what is deployed, for each training set ("paired", "t1", "dti"):

* the final pipeline fitted on every person of the set: ComBat per modality (frozen
  parameters) and one LightGBM model, with the same fixed settings as in train_mri.py;
* the same pipeline fitted on the training part of every cross-validation fold (5 x 10 =
  50 copies). A new person's score is reported with the spread of these copies' scores as a
  95% interval: how much the score depends on which people the model happened to learn from;
* the scanner map: which scanner models each ComBat batch was learned from. A new scan from
  a scanner model that is not in the map cannot be harmonised, and is not scored;
* the range of each harmonised feature in the training people, to flag a scan that falls
  outside what the model has seen;
* a model card: the people it was trained on, and the cross-validated performance with 95%
  CIs, copied from the newest VALID train_mri.py result. A set without a valid result is
  not bundled: a model is never deployed without an honest measurement of it.

Everything is plain JSON (ComBat parameters as lists, LightGBM models as text), never pickle,
so a bundle can be read and checked by eye, and loading one cannot run code.

Nothing here changes how performance is measured; the numbers on the model card are the
cross-validated ones, not scores of the final model on its own training people.

Command line (writes data/models/mri/<version>/ and data/models/mri/LATEST):

    python models/mri/src/mri_bundle.py
    python models/mri/src/mri_bundle.py --sets paired t1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from harmonise import ComBat, harmonisable_columns
from mri_contract import check_mri_features
from train_mri import LGBM_PARAMS, MAIN_MODEL, RESULTS, _commit, _lgbm, make_folds
from training_data import (
    DEFAULT_LABELS,
    FEATURES,
    MIN_BATCH,
    SAME_MODEL,
    SETS,
    ModelData,
    load_all,
)

REPO = Path(__file__).resolve().parents[3]
BUNDLES = REPO / "data" / "models" / "mri"
SCHEMA_VERSION = "mri-bundle/1"
PREFERENCE = ("paired", "t1", "dti")  # which set scores a new person, by what was uploaded
MODALITIES = {"paired": ("t1", "dti"), "t1": ("t1",), "dti": ("dti",)}


class BundleError(RuntimeError):
    """The bundle cannot be built or loaded; the message says why."""


def normalise_model(name: str) -> str:
    """Scanner model name as the batch rule sees it (training_data.SAME_MODEL)."""
    name = str(name or "").strip()
    return SAME_MODEL.get(name, name)


# ------------------------------------------------------------------ one fitted copy


@dataclass
class Member:
    """One fitted copy of the pipeline: ComBat per modality, then one LightGBM model."""

    combat: dict[str, ComBat]  # modality -> fitted ComBat
    kept: dict[str, list[str]]  # modality -> the columns ComBat could estimate
    booster: Any  # lightgbm.Booster

    @property
    def columns(self) -> list[str]:
        return [c for cols in self.kept.values() for c in cols]

    def harmonise(
        self, raw: dict[str, float], batch: dict[str, str], age: float, sex: float
    ) -> pd.DataFrame:
        """One person's raw features -> the harmonised row this model was trained on."""
        cov = np.array([[age, sex]], dtype=float)
        parts = []
        for g, cols in self.kept.items():
            x = np.array([[raw[c] for c in cols]], dtype=float)
            parts.append(pd.DataFrame(self.combat[g].transform(x, [batch[g]], cov), columns=cols))
        return pd.concat(parts, axis=1)

    def score(self, row: pd.DataFrame) -> float:
        """The model's raw score (log-odds of the PD class). Not a calibrated probability."""
        return float(self.booster.predict(row[self.columns], raw_score=True)[0])

    def contributions(self, row: pd.DataFrame) -> tuple[dict[str, float], float]:
        """Exact TreeSHAP contributions (log-odds) per feature, and the base value."""
        c = self.booster.predict(row[self.columns], pred_contrib=True)[0]
        return dict(zip(self.columns, map(float, c[:-1]), strict=True)), float(c[-1])

    def to_dict(self) -> dict:
        return {
            "combat": {g: m.to_dict() for g, m in self.combat.items()},
            "kept": self.kept,
            "model": self.booster.model_to_string(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Member:
        from lightgbm import Booster

        return cls(
            combat={g: ComBat.from_dict(v) for g, v in d["combat"].items()},
            kept={g: list(v) for g, v in d["kept"].items()},
            booster=Booster(model_str=d["model"]),
        )


def fit_member(data: ModelData, idx: np.ndarray, seed: int) -> Member:
    """Fit ComBat and LightGBM on the people at ``idx`` only (same rules as train_mri)."""
    cov = data.covariates()
    combat, kept, parts = {}, {}, []
    for g, cols in data.groups.items():
        check_mri_features(cols)
        x = data.features[cols].to_numpy(dtype=float)
        b = data.batches[g]
        keep = harmonisable_columns(x[idx], b[idx], cov[idx])
        kept[g] = [c for c, k in zip(cols, keep, strict=True) if k]
        if not kept[g]:
            raise BundleError(f"no {g} feature can be harmonised")
        combat[g] = ComBat().fit(x[idx][:, keep], b[idx], cov[idx])
        parts.append(
            pd.DataFrame(combat[g].transform(x[idx][:, keep], b[idx], cov[idx]), columns=kept[g])
        )
    h = pd.concat(parts, axis=1)
    model = _lgbm(seed).fit(h, data.y[idx])
    return Member(combat=combat, kept=kept, booster=model.booster_)


# ------------------------------------------------------------------ one set


@dataclass
class SetBundle:
    name: str
    final: Member
    members: list[Member]
    scanner_map: dict[str, dict[str, str]]  # modality -> scanner model -> ComBat batch
    feature_ranges: dict[str, list[float]]  # harmonised training range per feature
    card: dict = field(default_factory=dict)

    @property
    def modalities(self) -> tuple[str, ...]:
        return MODALITIES[self.name]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "final": self.final.to_dict(),
            "members": [m.to_dict() for m in self.members],
            "scanner_map": self.scanner_map,
            "feature_ranges": self.feature_ranges,
            "card": self.card,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SetBundle:
        return cls(
            name=d["name"],
            final=Member.from_dict(d["final"]),
            members=[Member.from_dict(m) for m in d["members"]],
            scanner_map=d["scanner_map"],
            feature_ranges=d["feature_ranges"],
            card=d["card"],
        )


def scanner_map(data: ModelData, scanners: dict[str, pd.DataFrame]) -> dict[str, dict[str, str]]:
    """Scanner model -> ComBat batch, per modality, from the people the set was built on.

    ``scanners[modality]`` has ``scanner_model`` per pseudonym (the feature table columns).
    A model whose people ended up in two batches would make the map ambiguous: refused.
    """
    out: dict[str, dict[str, str]] = {}
    for g in data.groups:
        names = scanners[g].loc[data.subjects, "scanner_model"].map(normalise_model).to_numpy()
        pairs = pd.DataFrame({"model": names, "batch": data.batches[g]}).drop_duplicates()
        if pairs["model"].duplicated().any():
            raise BundleError(f"{g}: one scanner model falls into two ComBat batches")
        out[g] = dict(zip(pairs["model"], pairs["batch"], strict=True))
    return out


def cv_result(set_name: str, results_dir: Path = RESULTS) -> dict:
    """The newest VALID cross-validated result of this set (train_mri.py results.json)."""
    for run in sorted((p for p in results_dir.glob("*/results.json")), reverse=True):
        payload = json.loads(run.read_text())
        for s in payload.get("sets", []):
            if s.get("name") == set_name and s.get("valid"):
                return {"run": run.parent.name, "settings": payload.get("settings", {}), **s}
    raise BundleError(
        f"no VALID cross-validated result for set {set_name!r} in {results_dir.name}/; "
        "run train_mri.py first (a model is not deployed without an honest measurement)"
    )


def track_record(set_name: str, run: str, results_dir: Path = RESULTS) -> dict | None:
    """How often the main model's lean was right in cross-validation, per direction.

    Read from the run's out-of-fold predictions (``predictions_<set>.csv``: one row per person,
    the probability averaged over repeats, from models that never saw that person). A person
    "leans PD" when that probability is above 0.5, the same boundary as log-odds 0 in the app.
    This is what lets a single result say how far its direction has held up in testing.
    Returns None when the run has no predictions file.
    """
    path = results_dir / run / f"predictions_{set_name}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    p, y = df[MAIN_MODEL[set_name]].to_numpy(dtype=float), df["y"].to_numpy(dtype=int)
    pd_side = p > 0.5
    return {
        "source": f"{run}, out-of-fold predictions",
        "leans_pd": {"n": int(pd_side.sum()), "correct": int(y[pd_side].sum())},
        "leans_control": {"n": int((~pd_side).sum()), "correct": int((1 - y[~pd_side]).sum())},
    }


def explanation_checks(set_name: str, results_dir: Path = RESULTS) -> dict | None:
    """Summary of the newest explain_checks.py run for this set, or None if there is none."""
    for path in sorted(results_dir.glob("explain_*/explain_checks.json"), reverse=True):
        r = json.loads(path.read_text())
        if r.get("set") == set_name:
            d, s = r["deletion"], r["stability"]
            return {
                "source": path.parent.name,
                "people": r.get("people"),
                "deletion_passed": d["passed"],
                "deletion_mean_top_minus_random": d["mean_top_minus_random"],
                "deletion_ci_95": d["ci_95"],
                "stability_mean_spearman": s["mean_spearman"],
                "stability_ci_95": s["spearman_ci_95"],
                "stability_meets_target": s["meets_target"],
                "top5_overlap": s["mean_top5_overlap"],
                "top5_overlap_by_chance": s["top5_overlap_by_chance"],
            }
    return None


def model_card(data: ModelData, cv: dict, dropped: dict[str, list[str]]) -> dict:
    main = next(m for m in cv["models"] if m["model"] == MAIN_MODEL[data.name])
    return {
        "set": data.name,
        "model": MAIN_MODEL[data.name],
        "purpose": "Screening and triage aid for prioritising specialist review. Not a diagnosis.",
        "training_people": {
            "total": int(data.n),
            "pd": int(data.y.sum()),
            "control": int(data.n - data.y.sum()),
        },
        "source": "PPMI (PP1 subset)",
        "scanner_groups": {
            g: {str(k): int(v) for k, v in pd.Series(b).value_counts().items()}
            for g, b in data.batches.items()
        },
        "features_not_used": dropped,
        "cross_validated": {
            "result_run": cv["run"],
            "folds": cv.get("split", {}).get("folds"),
            "repeats": cv.get("split", {}).get("repeats"),
            "auc_mean": main["auc_mean"],
            "auc_ci": main["auc_ci"],
            "balanced_accuracy_mean": main["bal_acc_mean"],
            "balanced_accuracy_ci": main["bal_acc_ci"],
            "sensitivity": main["sensitivity"],
            "specificity": main["specificity"],
            "auc_vs_age_sex": main["auc_vs_age_sex"],
            "controls_passed": bool(cv.get("valid")),
        },
        "track_record": track_record(data.name, cv["run"]) if "run" in cv else None,
        "explanation_checks": explanation_checks(data.name),
        "score_meaning": (
            "Raw model score in log-odds of the PD class: 0 is the boundary the model learned, "
            "above 0 leans PD, below leans control. Not a calibrated probability."
        ),
        "known_limits": [
            "Trained on few people; the cross-validated CI shows how uncertain performance is.",
            "Only scanner models seen in training can be harmonised; others are not scored.",
            "PD participants were already diagnosed; this is not evidence of earlier detection.",
            "Attributions show what moved this model's score, not a biological cause.",
        ],
    }


def build_set(
    data: ModelData,
    scanners: dict[str, pd.DataFrame],
    cv: dict,
    n_splits: int = 5,
    n_repeats: int = 10,
    seed: int = 0,
) -> SetBundle:
    everyone = np.arange(data.n)
    final = fit_member(data, everyone, seed)
    folds = make_folds(data, data.y, n_splits, n_repeats, seed)
    members = [fit_member(data, f.train_idx, seed + f.repeat) for f in folds]
    h = final.harmonise
    rows = pd.concat(
        [
            h(
                data.features.iloc[i].to_dict(),
                {g: data.batches[g][i] for g in data.groups},
                float(data.age[i]),
                float(data.sex[i]),
            )
            for i in everyone
        ],
        ignore_index=True,
    )
    ranges = {c: [float(rows[c].min()), float(rows[c].max())] for c in rows.columns}
    dropped = {g: [c for c in data.groups[g] if c not in final.kept[g]] for g in data.groups}
    return SetBundle(
        name=data.name,
        final=final,
        members=members,
        scanner_map=scanner_map(data, scanners),
        feature_ranges=ranges,
        card=model_card(data, cv, dropped),
    )


# ------------------------------------------------------------------ whole bundle


@dataclass
class Bundle:
    version: str
    sets: dict[str, SetBundle]
    created: str = ""
    commit: str = ""

    def choose(self, available: set[str]) -> SetBundle | None:
        """The first set in PREFERENCE whose inputs were all uploaded and usable.

        Both T1 and DTI -> paired; only one -> that one. If the paired model is not
        bundled, T1 alone is used before DTI alone.
        """
        for name in PREFERENCE:
            if name in self.sets and set(MODALITIES[name]) <= available:
                return self.sets[name]
        return None


def save_bundle(bundle: Bundle, root: Path = BUNDLES) -> Path:
    out = root / bundle.version
    out.mkdir(parents=True, exist_ok=True)
    body = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "version": bundle.version,
            "created": bundle.created,
            "commit": bundle.commit,
            "lightgbm": LGBM_PARAMS,
            "sets": {k: v.to_dict() for k, v in bundle.sets.items()},
        }
    )
    (out / "bundle.json").write_text(body)
    cards = {k: v.card for k, v in bundle.sets.items()}
    (out / "model_card.json").write_text(json.dumps(cards, indent=2))
    digest = hashlib.sha256(body.encode()).hexdigest()
    (out / "manifest.json").write_text(
        json.dumps({"version": bundle.version, "bundle_sha256": digest}, indent=2)
    )
    (root / "LATEST").write_text(bundle.version + "\n")
    return out


def load_bundle(path: Path | None = None, root: Path = BUNDLES) -> Bundle:
    """Load a bundle directory (default: the one named in data/models/mri/LATEST).

    The bundle file is checked against its recorded hash, so a changed or partly copied
    bundle is refused instead of silently giving different scores.
    """
    if path is None:
        latest = root / "LATEST"
        if not latest.exists():
            raise BundleError("no MRI model bundle yet; run mri_bundle.py")
        path = root / latest.read_text().strip()
    body_path, manifest_path = path / "bundle.json", path / "manifest.json"
    if not body_path.exists() or not manifest_path.exists():
        raise BundleError(f"incomplete bundle: {path.name}")
    body = body_path.read_text()
    expected = json.loads(manifest_path.read_text())["bundle_sha256"]
    if hashlib.sha256(body.encode()).hexdigest() != expected:
        raise BundleError(f"bundle {path.name} does not match its manifest hash")
    d = json.loads(body)
    if d.get("schema_version") != SCHEMA_VERSION:
        raise BundleError(f"bundle schema {d.get('schema_version')!r} is not {SCHEMA_VERSION}")
    return Bundle(
        version=d["version"],
        sets={k: SetBundle.from_dict(v) for k, v in d["sets"].items()},
        created=d.get("created", ""),
        commit=d.get("commit", ""),
    )


def main(argv: list[str] | None = None) -> int:
    import warnings

    warnings.filterwarnings("ignore", message="The least populated class", category=UserWarning)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sets", nargs="+", choices=SETS, help="default: every set with features")
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--features", type=Path, default=FEATURES)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--out", type=Path, default=BUNDLES)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    data = load_all(args.labels, args.features, MIN_BATCH)
    chosen = [s for s in (args.sets or PREFERENCE) if s in data]
    tables = {
        g: pd.read_csv(args.features / f"{g}_features.csv", dtype=str).set_index("subject")
        for g in ("t1", "dti")
        if (args.features / f"{g}_features.csv").exists()
    }
    commit = _commit()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    sets = {}
    for name in chosen:
        try:
            cv = cv_result(name, args.results)
        except BundleError as e:
            print(f"[{name}] skipped: {e}", file=sys.stderr)
            continue
        print(f"[{name}] fitting the final model and {args.folds * args.repeats} fold copies ...")
        sets[name] = build_set(data[name], tables, cv, args.folds, args.repeats, args.seed)
    if not sets:
        print("error: nothing to bundle", file=sys.stderr)
        return 2
    out = save_bundle(Bundle(f"{stamp}_{commit}", sets, stamp, commit), args.out)
    print(f"bundle: {out} (sets: {', '.join(sets)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
