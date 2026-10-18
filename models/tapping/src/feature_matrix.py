"""Export the feature matrix (task FT-30): features.csv, feature_roles.csv, features_meta.json.

    python models/tapping/src/feature_matrix.py [--features-dir ...] [--labels ...] [--out-dir ...]

Joins Set A and Set B (one row per QC-retained clip) with the rating on the exact key
ref == hubu_fis.pseudonym(ID). Feature cells are read and written as text, so they are copied
unchanged; only the two nan_reasons columns are renamed. The raw ID, the group (CONTROL/ID), the
line number and the label file's order are never stored. ROLES is frozen before any label is
joined and pinned by a test. Prints are counts only. Exit codes: 0 ok, 1 inputs disagree, a clip
has no label or the label file is invalid, 2 missing input or a path outside its folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from hubu_fis import HubuFisError, load_labels, pseudonym
from signal_processing import file_sha256, is_inside

TOOL_VERSION = "feature-matrix-0.1.0"
REPO_ROOT = Path(__file__).resolve().parents[3]
INTERIM_ROOT = REPO_ROOT / "data" / "interim"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
DOCS_ROOT = REPO_ROOT / "models" / "tapping" / "docs"
LABELS_PATH = REPO_ROOT / "data" / "raw" / "HUBU-FIS" / "fis_diagnostic.csv"
HASH_LEN = 8
OUT_FEATURES, OUT_ROLES, OUT_META = "features.csv", "feature_roles.csv", "features_meta.json"
ID_COLUMNS = ("ref", "participant", "hand", "rating")
RENAMED = {"nan_reasons": ("nan_reasons_a", "nan_reasons_b")}  # the only departure from verbatim
META_AGREE = ("tap_params_sha8", "taps_summary_sha8", "qc_list_sha8")  # A and B must match
NON_MODAL_MIN = 10  # fewer non-modal clips than this: descriptive, not a model input
FPS_HIGH = 45.0  # a clip above this frame rate counts as 60 fps

VOCABULARY = (
    "id", "group_key", "label", "covariate_candidate", "feature_primary", "feature_twin",
    "feature_secondary", "feature_exploratory", "feature_descriptive", "feature_sensitivity",
    "audit_count",
)  # fmt: skip
FEATURE_ROLES = tuple(r for r in VOCABULARY if r.startswith("feature_"))
SENS_A = (
    "tap_rate_hz", "iti_mean_s", "iti_cv", "open_vel_peak", "open_vel_peak_median",
    "close_vel_peak", "close_vel_peak_median", "open_vel_mean", "open_vel_mean_median",
    "close_vel_mean", "close_vel_mean_median", "amp_mean", "amp_max",
)  # fmt: skip
SENS_B = (
    "amp_slope", "open_vel_slope", "close_vel_slope", "amp_late_early", "amp_min_initial",
    "decrement_present", "decrement_onset_frac", "iti_slope", "n_hesitations", "n_halts",
    "longest_pause_ratio", "longest_pause_s",
)  # fmt: skip

# Frozen before any label is joined. {column: (role, note)}
ROLES: dict[str, tuple[str, str]] = {
    "ref": ("id", "hashed clip id"),
    "participant": ("group_key", "hashed participant; folds are grouped on it"),
    "rating": ("label", "clinician rating; label or stratifier only"),
    "hand": ("covariate_candidate", "left or right"),
    "tap_rate_hz": ("feature_primary", "tapping rate"),
    "iti_cv": ("feature_primary", "interval variability"),
    "open_vel_peak": ("feature_primary", "peak opening speed"),
    "close_vel_peak": ("feature_primary", "peak closing speed"),
    "amp_mean": ("feature_primary", "mean amplitude (hypokinesia)"),
    "amp_slope": ("feature_primary", "amplitude trend"),
    "open_vel_slope": ("feature_primary", "opening speed trend"),
    "close_vel_slope": ("feature_primary", "closing speed trend"),
    "amp_min_initial": ("feature_primary", "lowest level over initial level"),
    "decrement_present": ("feature_primary", "sustained fall present"),
    "decrement_onset_frac": ("feature_primary", "NaN when no decrement; imputation only in the CV"),
    "n_hesitations": ("feature_primary", "passes the 10 non-modal clips rule"),
    "longest_pause_ratio": ("feature_primary", "longest interval over the median"),
    "amp_late_early": ("feature_twin", "twin of amp_slope"),
    "iti_mean_s": ("feature_twin", "reciprocal of tap_rate_hz"),
    "amp_max": ("feature_twin", "twin of amp_mean"),
    "n_taps": ("feature_twin", "twin of tap_rate_hz; depends on clip length"),
    "n_over_duration_hz": ("feature_secondary", "rate over the whole clip"),
    "open_vel_mean": ("feature_secondary", "amplitude over rise time"),
    "close_vel_mean": ("feature_secondary", "amplitude over fall time"),
    "open_vel_peak_median": ("feature_secondary", "median variant"),
    "close_vel_peak_median": ("feature_secondary", "median variant"),
    "open_vel_mean_median": ("feature_secondary", "median variant"),
    "close_vel_mean_median": ("feature_secondary", "median variant"),
    "longest_pause_s": ("feature_secondary", "longest interval in seconds"),
    "iti_slope": ("feature_exploratory", "split-half reliability below 0.6; not primary"),
    "n_halts": ("feature_descriptive", "fewer than 10 non-modal clips; not a model input"),
    "n_valid_taps": ("audit_count", "taps used for amplitude features"),
    "n_one_sided_excluded": ("audit_count", "taps left out of amplitude features"),
    "n_iti_excluded": ("audit_count", "intervals left out"),
    "n_short_iti": ("audit_count", "intervals under 0.2 s"),
    "fps": ("audit_count", "measured frame rate"),
    "duration_s": ("audit_count", "signal duration"),
    "qc_status": ("audit_count", "retained or retained_flag"),
    "nan_reasons_a": ("audit_count", "why a Set A value is missing"),
    "nan_reasons_b": ("audit_count", "why a Set B value is missing"),
    **{f + "_sens": ("feature_sensitivity", "sensitivity variant") for f in SENS_A},
    **{f + "_sens": ("feature_sensitivity", "sensitivity variant") for f in SENS_B},
}


def columns_with_role(*roles: str) -> list[str]:
    """Column names holding any of the given roles, in registry order."""
    unknown = set(roles) - set(VOCABULARY)
    if unknown:
        raise ValueError(f"unknown roles: {sorted(unknown)}")
    return [c for c, (r, _) in ROLES.items() if r in roles]


PRIMARY_COLUMNS = columns_with_role("feature_primary")


def roles_sha8() -> str:
    text = json.dumps({c: list(v) for c, v in ROLES.items()}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:HASH_LEN]


class Disagreement(ValueError):
    """The inputs disagree, a clip has no label or the labels are invalid (exit code 1)."""


class InputProblem(ValueError):
    """An input is missing or a path is outside its folder (exit code 2)."""


# --------------------------------------------------------------------------- inputs


def sha8(path: Path) -> str:
    return file_sha256(path)[:HASH_LEN]


def read_text_table(path: Path) -> pd.DataFrame:
    """A CSV as text: every cell is kept exactly as written, an empty cell stays empty."""
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, ValueError) as err:
        raise InputProblem(f"cannot read {path.name}") from err


def read_meta(path: Path) -> dict[str, Any]:
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise InputProblem(f"cannot read {path.name}") from err
    return data


def merge_features(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Set A and Set B side by side on ref, as text; one qc_status; nan_reasons renamed."""
    for name, df in (("features_a", a), ("features_b", b)):
        if "ref" not in df.columns or not df["ref"].is_unique:
            raise Disagreement(f"{name}.csv needs a unique ref column")
    if set(a["ref"]) != set(b["ref"]) or len(a) != len(b):
        raise Disagreement("features_a.csv and features_b.csv hold different clips")
    a, b = a.set_index("ref"), b.set_index("ref")
    b = b.loc[a.index]
    if not (a["qc_status"] == b["qc_status"]).all():
        raise Disagreement("qc_status differs between features_a.csv and features_b.csv")
    a = a.rename(columns={"nan_reasons": RENAMED["nan_reasons"][0]})
    b = b.rename(columns={"nan_reasons": RENAMED["nan_reasons"][1]}).drop(columns="qc_status")
    clash = set(a.columns) & set(b.columns)
    if clash:
        raise Disagreement(f"{len(clash)} column names are in both feature files")
    return pd.concat([a, b], axis=1).reset_index()


def label_table(path: Path) -> pd.DataFrame:
    """ref, participant, hand, rating (text) from the label file; nothing else is kept."""
    if not path.is_file():
        raise InputProblem("the label file was not found")
    try:
        raw = load_labels(path)
    except HubuFisError as err:
        reason = re.sub(r"CSV lines:.*", "", str(err).splitlines()[0]).strip(" ,")
        raise Disagreement(f"the label file is invalid: {reason}") from err
    out = pd.DataFrame(
        {
            "ref": raw["ID"].map(pseudonym),
            "participant": raw["participant"].map(pseudonym),
            "hand": raw["hand"],
            "rating": raw["UPDRS"].astype(int).astype(str),
        }
    )  # ID, group and line are dropped here and never used again
    if not out["ref"].is_unique:
        raise Disagreement("two label rows share a hashed clip key")
    return out


def non_modal(values: pd.Series) -> int:
    """Clips whose (non-empty) value is not the most common one."""
    v = values[values != ""]
    return int((v != v.mode().iloc[0]).sum()) if len(v) else 0


# --------------------------------------------------------------------------- build


def build(
    a: pd.DataFrame, b: pd.DataFrame, labels: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    """The feature matrix (sorted by ref) and the join counts. Exact key, no fuzzy matching."""
    feats = merge_features(a, b)
    joined = feats[["ref"]].merge(labels, on="ref", how="left", validate="one_to_one")
    unmatched = int(joined["rating"].isna().sum())
    if unmatched:
        raise Disagreement(f"{unmatched} clips have no label")
    out = joined.merge(feats, on="ref", how="left", validate="one_to_one")
    columns = [*ID_COLUMNS, *[c for c in feats.columns if c != "ref"]]
    out = out[columns].sort_values("ref", kind="stable").reset_index(drop=True)
    counts = {
        "matched": len(out),
        "unmatched": unmatched,
        "labels_unused": len(set(labels["ref"]) - set(out["ref"])),
    }
    return out, counts


def check_roles(columns: Sequence[str]) -> None:
    missing = sorted(set(columns) - set(ROLES))
    extra = sorted(set(ROLES) - set(columns))
    if missing or extra:
        raise Disagreement(
            f"roles and columns differ: {len(missing)} without a role, {len(extra)} unused"
        )


def roles_table(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": list(columns),
            "role": [ROLES[c][0] for c in columns],
            "note": [ROLES[c][1] for c in columns],
        }
    )


def _atomic(final: Path, write: Any) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f".{final.name}.tmp{os.getpid()}")
    try:
        write(tmp)
        os.replace(tmp, final)
    finally:
        if tmp.exists():
            tmp.unlink()


def run(features_dir: Path, labels_path: Path, out_dir: Path, docs_dir: Path) -> dict[str, Any]:
    paths = {
        "features_a_csv": features_dir / "features_a.csv",
        "features_b_csv": features_dir / "features_b.csv",
        "features_a_meta": features_dir / "features_a_meta.json",
        "features_b_meta": features_dir / "features_b_meta.json",
        "qc_list": docs_dir / "qc_retained.csv",
    }
    for key, path in paths.items():
        if not path.is_file():
            raise InputProblem(f"input not found: {key}")
    ma, mb = read_meta(paths["features_a_meta"]), read_meta(paths["features_b_meta"])
    for key in META_AGREE:
        if ma.get(key) is None or ma.get(key) != mb.get(key):
            raise Disagreement(f"features_a_meta and features_b_meta disagree on {key}")
    if ma["qc_list_sha8"] != sha8(paths["qc_list"]):
        raise Disagreement("the features were made with another QC list")
    a, b = read_text_table(paths["features_a_csv"]), read_text_table(paths["features_b_csv"])
    labels = label_table(labels_path)
    matrix, join = build(a, b, labels)
    check_roles(list(matrix.columns))
    roles = roles_table(list(matrix.columns))
    descriptive = {
        c: non_modal(matrix[c]) for c in ("n_halts", "n_hesitations", "decrement_present")
    }
    fps = pd.to_numeric(matrix["fps"], errors="coerce")
    meta = {
        "tool": TOOL_VERSION,
        "inputs_sha8": {
            **{k: sha8(p) for k, p in paths.items()},
            "labels": sha8(labels_path),
        },
        "definitions": {
            "set_a_version": ma.get("definitions_version"),
            "set_b_version": mb.get("definitions_version"),
            "set_a_definitions_sha8": ma.get("definitions_sha8"),
            "set_b_definitions_sha8": mb.get("definitions_sha8"),
        },
        "tap_params_sha8": ma["tap_params_sha8"],
        "roles_sha8": roles_sha8(),
        "rows": len(matrix),
        "columns": len(matrix.columns),
        "participants": int(matrix["participant"].nunique()),
        "join": join,
        "columns_per_role": {r: int((roles["role"] == r).sum()) for r in VOCABULARY},
        "non_modal_clips": descriptive,
        "non_modal_min": NON_MODAL_MIN,
        "clips_above_45_fps": int((fps > FPS_HIGH).sum()),
        "labels_used": ["UPDRS"],
        "group_dropped": True,
    }
    _atomic(out_dir / OUT_FEATURES, lambda p: matrix.to_csv(p, index=False, lineterminator="\n"))
    _atomic(out_dir / OUT_ROLES, lambda p: roles.to_csv(p, index=False, lineterminator="\n"))
    _atomic(
        out_dir / OUT_META,
        lambda p: p.write_text(json.dumps(meta, sort_keys=True, indent=1) + "\n", encoding="utf-8"),
    )
    return meta


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for name in ("--features-dir", "--labels", "--out-dir", "--docs-dir"):
        ap.add_argument(name, type=Path, default=None)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    features_dir = Path(args.features_dir or INTERIM_ROOT / "features")
    labels = Path(args.labels or LABELS_PATH)
    out_dir = Path(args.out_dir or PROCESSED_ROOT)
    docs_dir = Path(args.docs_dir or DOCS_ROOT)
    for name, path, root in (
        ("--features-dir", features_dir, INTERIM_ROOT),
        ("--out-dir", out_dir, PROCESSED_ROOT),
        ("--docs-dir", docs_dir, DOCS_ROOT),
    ):
        if not is_inside(path, root):
            print(f"error: {name} must be inside {root.parent.name}/{root.name}", file=sys.stderr)
            return 2
    try:
        meta = run(features_dir, labels, out_dir, docs_dir)
    except Disagreement as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    except InputProblem as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    print(
        f"wrote {OUT_FEATURES}: {meta['rows']} rows, {meta['columns']} columns, "
        f"{meta['participants']} participants, matched {meta['join']['matched']}, "
        f"unmatched {meta['join']['unmatched']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
