"""Nested-CV splits and leakage guards (task FT-32, part 1; the models come in part 2).

    python models/tapping/src/cv_splits.py [--target ge2] [--repeats 5] [--seed 0] [--force]

Outer: StratifiedGroupKFold(5, shuffle, seed + repeat) on (rating >= threshold, participant).
Inner: StratifiedGroupKFold(3, shuffle, 1000 * repeat + outer_fold + 1) on the outer training part
only, so it depends on labels, groups and positions, never on the model or the feature set. The
rating is read only as the stratification target. Rows are put in ref order first, so the splits do
not depend on the row order of features.csv. Every outer test fold and inner validation fold needs
at least 5 positive and 5 negative participants. Writes <out-dir>/<target>/splits.csv and
splits_meta.json (participant level, hashed ids, no timestamps). Prints are counts only.
Exit codes: 0 ok, 1 guard, gate or input disagreement, 2 missing input or path outside the folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

TOOL_VERSION = "cv-splits-0.1.0"
REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
CV_ROOT = PROCESSED_ROOT / "cv"
HASH_LEN = 8

SET_A_PRIMARY = ["tap_rate_hz", "iti_cv", "open_vel_peak", "close_vel_peak", "amp_mean"]
SET_B_ADDED = [
    "amp_slope",
    "open_vel_slope",
    "close_vel_slope",
    "amp_min_initial",
    "decrement_present",
    "decrement_onset_frac",
    "n_hesitations",
    "longest_pause_ratio",
]
MODEL_COLUMNS = SET_A_PRIMARY + SET_B_ADDED
TARGETS = {"ge1": 1, "ge2": 2, "ge3": 3}  # fixed, never fitted
DEFAULT_TARGET = "ge2"
N_OUTER = 5
N_INNER = 3
MIN_POS_PARTICIPANTS = 5
SPLIT_COLUMNS = ["participant", "repeat", "outer_fold"]
ID_PATTERN = re.compile(r"[0-9a-f]{8}")
ID_ROLES = {"ref": "id", "participant": "group_key", "rating": "label"}
OUT_SPLITS = "splits.csv"
OUT_META = "splits_meta.json"


class GuardFailure(ValueError):
    """A guard or the validity gate failed, or the inputs disagree (exit code 1)."""


class ContractError(GuardFailure):
    """A model column is not allowed (exit code 1)."""


class GateFailure(GuardFailure):
    """A fold has too few positive or negative participants (exit code 1)."""


class LeakageError(AssertionError):
    """A participant is on both sides of a split (exit code 1)."""


class InputProblem(ValueError):
    """An input is missing or a path is outside its folder (exit code 2)."""


# --------------------------------------------------------------------------- small helpers


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha8_of(text: str | bytes) -> str:
    data = text.encode() if isinstance(text, str) else text
    return hashlib.sha256(data).hexdigest()[:HASH_LEN]


def is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def outer_seed(seed: int, repeat: int) -> int:
    return seed + repeat


def inner_seed(repeat: int, outer_fold: int) -> int:
    return 1000 * repeat + outer_fold + 1


def roles_sha8_from_file(roles: pd.DataFrame) -> str:
    """sha8 of the roles registry, computed as feature_matrix.roles_sha8 does (kept local)."""
    table = {
        c: [r, n] for c, r, n in zip(roles["column"], roles["role"], roles["note"], strict=True)
    }
    text = json.dumps(table, sort_keys=True, separators=(",", ":"))
    return sha8_of(text)


# --------------------------------------------------------------------------- inputs


def load_inputs(
    features_path: Path, roles_path: Path, meta_path: Path
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Read and verify features.csv against its roles file and meta; return frame and columns."""
    for name, path in (("features", features_path), ("roles", roles_path), ("meta", meta_path)):
        if not Path(path).is_file():
            raise InputProblem(f"missing input: {name} file")
    meta = json.loads(Path(meta_path).read_text())
    roles = pd.read_csv(roles_path, dtype=str, keep_default_na=False)
    if list(roles.columns) != ["column", "role", "note"]:
        raise GuardFailure("roles file must have the columns column, role, note")
    if roles["column"].duplicated().any():
        raise GuardFailure("roles file lists a column twice")
    if roles_sha8_from_file(roles) != meta.get("roles_sha8"):
        raise GuardFailure("roles_sha8 of the roles file differs from the one in the meta file")
    role_of = dict(zip(roles["column"], roles["role"], strict=True))
    for column, role in ID_ROLES.items():
        if role_of.get(column) != role:
            raise GuardFailure(f"roles file: {column} must have role {role}")
    primary = [c for c in roles["column"] if role_of[c] == "feature_primary"]
    if sorted(primary) != sorted(MODEL_COLUMNS):
        raise GuardFailure(
            "feature_primary in the roles file differs from the pinned model columns "
            f"(missing {sorted(set(MODEL_COLUMNS) - set(primary))}, "
            f"extra {sorted(set(primary) - set(MODEL_COLUMNS))})"
        )

    frame = pd.read_csv(features_path, dtype={"ref": str, "participant": str})
    for column in ("ref", "participant", "rating", *MODEL_COLUMNS):
        if column not in frame.columns:
            raise GuardFailure(f"features file lacks the column {column}")
    if len(frame) != meta.get("rows"):
        raise GuardFailure(f"features has {len(frame)} rows, meta says {meta.get('rows')}")
    if frame["ref"].isna().any() or frame["ref"].duplicated().any():
        raise GuardFailure("ref must be present and unique")
    if frame["participant"].isna().any() or (frame["participant"] == "").any():
        raise GuardFailure("participant must be present and non-empty")
    for column in ("ref", "participant"):
        if not frame[column].map(lambda v: bool(ID_PATTERN.fullmatch(v))).all():
            raise GuardFailure(f"{column} must be an 8-character hash everywhere")
    n_participants = meta.get("participants")
    if n_participants is not None and frame["participant"].nunique() != n_participants:
        raise GuardFailure("participant count differs from the meta file")
    rating = frame["rating"]
    if rating.dtype.kind not in "iu" or not rating.between(0, 3).all():
        raise GuardFailure("rating must be an integer 0..3 in every row")
    validate_inputs(frame, MODEL_COLUMNS)
    return frame, {"model": list(MODEL_COLUMNS), "set_a": list(SET_A_PRIMARY),
                   "set_b_added": list(SET_B_ADDED)}  # fmt: skip


def validate_inputs(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """The validator before any fit (CLAUDE.md 3.1): only feature_primary columns, once each."""
    cols = list(columns)
    if not cols:
        raise ContractError("no model columns given")
    duplicated = sorted({c for c in cols if cols.count(c) > 1})
    if duplicated:
        raise ContractError(f"duplicate model columns: {duplicated}")
    refused = [c for c in cols if c not in MODEL_COLUMNS]
    if refused:
        raise ContractError(f"not feature_primary, refused as model input: {refused}")
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ContractError(f"model columns missing from the frame: {missing}")
    for c in cols:
        if frame[c].dtype.kind not in "biuf":
            raise ContractError(f"model column {c} is not numeric or boolean")


def canonical(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows in ref order, so positions mean the same thing in every module."""
    if frame["ref"].duplicated().any():
        raise GuardFailure("ref must be unique")
    return frame.sort_values("ref", kind="stable").reset_index(drop=True)


def target_labels(frame: pd.DataFrame, target: str) -> np.ndarray:
    if target not in TARGETS:
        raise GuardFailure(f"unknown target {target!r}; choose from {sorted(TARGETS)}")
    return (frame["rating"].to_numpy() >= TARGETS[target]).astype(int)


# --------------------------------------------------------------------------- guards


def assert_disjoint(train_groups: Iterable[Any], test_groups: Iterable[Any], where: str) -> None:
    """Raise LeakageError if a group is on both sides or a side is empty (counts only)."""
    train = set(train_groups)
    test = set(test_groups)
    if not train or not test:
        raise LeakageError(f"{where}: empty side (train groups {len(train)}, test {len(test)})")
    shared = train & test
    if shared:
        raise LeakageError(
            f"{where}: {len(shared)} group(s) on both sides (train groups {len(train)}, "
            f"test groups {len(test)})"
        )


def _split(
    n_splits: int, seed: int, y: np.ndarray, groups: np.ndarray, where: str
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # the gate reports thin classes
            return list(splitter.split(np.zeros(len(y)), y, groups))
    except ValueError as err:
        raise GateFailure(
            f"{where}: cannot make {n_splits} folds from {len(set(groups))} groups "
            f"and {int(y.sum())} positive rows"
        ) from err


def inner_folds(
    train_positions: np.ndarray, y: np.ndarray, groups: np.ndarray, repeat: int, outer_fold: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Inner (fit, validation) positions inside one outer training part."""
    train_positions = np.asarray(train_positions)
    where = f"inner repeat {repeat} outer {outer_fold}"
    pairs = _split(
        N_INNER,
        inner_seed(repeat, outer_fold),
        y[train_positions],
        groups[train_positions],
        where,
    )
    out = []
    for k, (fit, val) in enumerate(pairs):
        fit_pos, val_pos = train_positions[fit], train_positions[val]
        assert_disjoint(groups[fit_pos], groups[val_pos], f"{where} fold {k}")
        out.append((fit_pos, val_pos))
    return out


def check_partition(splits: pd.DataFrame, participants: set[str], repeats: int) -> None:
    """Every participant exactly once per repeat, folds in range, repeats 0..R-1."""
    if list(splits.columns) != SPLIT_COLUMNS:
        raise GuardFailure(f"splits must have the columns {SPLIT_COLUMNS}")
    if sorted(splits["repeat"].unique()) != list(range(repeats)):
        raise GuardFailure(f"splits must hold the repeats 0..{repeats - 1}")
    if not splits["outer_fold"].between(0, N_OUTER - 1).all():
        raise GuardFailure(f"outer_fold must be in 0..{N_OUTER - 1}")
    for repeat, part in splits.groupby("repeat"):
        ids = part["participant"]
        if ids.duplicated().any():
            raise GuardFailure(
                f"repeat {repeat}: {int(ids.duplicated().sum())} participant row(s) repeated"
            )
        if set(ids) != participants:
            raise GuardFailure(
                f"repeat {repeat}: {len(set(ids) ^ participants)} participant(s) missing or extra"
            )
        if part["outer_fold"].nunique() != N_OUTER:
            raise GuardFailure(f"repeat {repeat}: not all {N_OUTER} outer folds are used")


# --------------------------------------------------------------------------- hashes


def splits_sha8(splits: pd.DataFrame) -> str:
    rows = sorted(zip(splits["participant"], splits["repeat"], splits["outer_fold"], strict=True))
    return sha8_of("".join(f"{p},{r},{f}\n" for p, r, f in rows))


def inner_sha8(inner: pd.DataFrame) -> str:
    cols = ["repeat", "outer_fold", "inner_fold", "role", "ref"]
    rows = sorted(zip(*(inner[c] for c in cols), strict=True))
    return sha8_of("".join(",".join(str(v) for v in row) + "\n" for row in rows))


# --------------------------------------------------------------------------- building


def outer_table(frame: pd.DataFrame, target: str, repeats: int, seed: int) -> pd.DataFrame:
    frame = canonical(frame)
    if repeats < 1 or seed < 0:
        raise GuardFailure("repeats must be at least 1 and seed non-negative")
    y = target_labels(frame, target)
    groups = frame["participant"].to_numpy()
    rows: list[tuple[str, int, int]] = []
    for repeat in range(repeats):
        pairs = _split(N_OUTER, outer_seed(seed, repeat), y, groups, f"outer repeat {repeat}")
        for fold, (train, test) in enumerate(pairs):
            assert_disjoint(groups[train], groups[test], f"outer repeat {repeat} fold {fold}")
            rows.extend((p, repeat, fold) for p in sorted(set(groups[test])))
    splits = pd.DataFrame(rows, columns=SPLIT_COLUMNS)
    splits = splits.sort_values(["repeat", "outer_fold", "participant"]).reset_index(drop=True)
    check_partition(splits, set(groups), repeats)
    return splits


def _outer_membership(groups: np.ndarray, splits: pd.DataFrame, repeat: int) -> np.ndarray:
    fold_of = dict(
        zip(
            splits.loc[splits["repeat"] == repeat, "participant"],
            splits.loc[splits["repeat"] == repeat, "outer_fold"],
            strict=True,
        )
    )
    return np.array([fold_of[g] for g in groups])


def inner_table(frame: pd.DataFrame, splits: pd.DataFrame, target: str) -> pd.DataFrame:
    """Every inner assignment of the run: repeat, outer_fold, inner_fold, role, position, ref."""
    frame = canonical(frame)
    y = target_labels(frame, target)
    groups = frame["participant"].to_numpy()
    refs = frame["ref"].to_numpy()
    repeats = int(splits["repeat"].max()) + 1
    check_partition(splits, set(groups), repeats)
    parts: list[pd.DataFrame] = []
    for repeat in range(repeats):
        membership = _outer_membership(groups, splits, repeat)
        for fold in range(N_OUTER):
            train = np.flatnonzero(membership != fold)
            test = np.flatnonzero(membership == fold)
            assert_disjoint(groups[train], groups[test], f"outer repeat {repeat} fold {fold}")
            for k, (fit, val) in enumerate(inner_folds(train, y, groups, repeat, fold)):
                for role, pos in (("fit", fit), ("val", val)):
                    parts.append(
                        pd.DataFrame(
                            {
                                "repeat": repeat,
                                "outer_fold": fold,
                                "inner_fold": k,
                                "role": role,
                                "position": pos,
                                "ref": refs[pos],
                            }
                        )
                    )
    return pd.concat(parts, ignore_index=True)


def _composition(y: np.ndarray, groups: np.ndarray, positions: np.ndarray) -> dict[str, int]:
    yy, gg = y[positions], groups[positions]
    return {
        "pos_clips": int(yy.sum()),
        "neg_clips": int((1 - yy).sum()),
        "pos_participants": len(set(gg[yy == 1])),
        "neg_participants": len(set(gg[yy == 0])),
    }


def _summary(rows: list[dict[str, int]]) -> dict[str, dict[str, int]]:
    return {k: {"min": min(r[k] for r in rows), "max": max(r[k] for r in rows)} for k in rows[0]}


def check_gate(
    frame: pd.DataFrame, splits: pd.DataFrame, inner: pd.DataFrame, target: str
) -> dict[str, Any]:
    """Every outer test and inner validation fold needs MIN_POS_PARTICIPANTS of each class."""
    frame = canonical(frame)
    y = target_labels(frame, target)
    groups = frame["participant"].to_numpy()
    outer_rows, inner_rows = [], []
    for repeat in sorted(splits["repeat"].unique()):
        membership = _outer_membership(groups, splits, int(repeat))
        for fold in range(N_OUTER):
            outer_rows.append(_composition(y, groups, np.flatnonzero(membership == fold)))
    val = inner[inner["role"] == "val"]
    for _, part in val.groupby(["repeat", "outer_fold", "inner_fold"]):
        inner_rows.append(_composition(y, groups, part["position"].to_numpy()))

    def thin(rows: list[dict[str, int]]) -> int:
        return sum(
            r["pos_participants"] < MIN_POS_PARTICIPANTS
            or r["neg_participants"] < MIN_POS_PARTICIPANTS
            for r in rows
        )

    result: dict[str, Any] = {
        "min_participants_per_class": MIN_POS_PARTICIPANTS,
        "outer_test": {"folds": len(outer_rows), "thin_folds": thin(outer_rows),
                       **_summary(outer_rows)},
        "inner_validation": {"folds": len(inner_rows), "thin_folds": thin(inner_rows),
                             **_summary(inner_rows)},
    }  # fmt: skip
    bad_outer, bad_inner = (
        result["outer_test"]["thin_folds"],
        result["inner_validation"]["thin_folds"],
    )
    if bad_outer or bad_inner:
        raise GateFailure(
            f"gate failed ({target}): {bad_outer} of {len(outer_rows)} outer test folds and "
            f"{bad_inner} of {len(inner_rows)} inner validation folds hold fewer than "
            f"{MIN_POS_PARTICIPANTS} positive or negative participants; fewest positive "
            f"participants outer {result['outer_test']['pos_participants']['min']}, inner "
            f"{result['inner_validation']['pos_participants']['min']}; fewest negative "
            f"participants outer {result['outer_test']['neg_participants']['min']}, inner "
            f"{result['inner_validation']['neg_participants']['min']}"
        )
    return result


class SplitPlan:
    """Outer splits, inner assignments, gate counts and the two hashes of one run."""

    def __init__(self, splits: pd.DataFrame, inner: pd.DataFrame, gate: dict[str, Any]) -> None:
        self.splits = splits
        self.inner = inner
        self.gate = gate
        self.splits_sha8 = splits_sha8(splits)
        self.inner_sha8 = inner_sha8(inner)


def build_plan(frame: pd.DataFrame, target: str, repeats: int, seed: int) -> SplitPlan:
    splits = outer_table(frame, target, repeats, seed)
    inner = inner_table(frame, splits, target)
    gate = check_gate(frame, splits, inner, target)
    return SplitPlan(splits, inner, gate)


def make_splits(frame: pd.DataFrame, target: str, repeats: int, seed: int) -> pd.DataFrame:
    """One row per participant, repeat and outer fold; the gate and all guards have run."""
    return build_plan(frame, target, repeats, seed).splits


# --------------------------------------------------------------------------- files


def _atomic(final: Path, data: bytes) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f".{final.name}.tmp{os.getpid()}")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, final)
    finally:
        if tmp.exists():
            tmp.unlink()


def splits_bytes(splits: pd.DataFrame) -> bytes:
    return splits.to_csv(index=False, lineterminator="\n").encode()


def load_splits(path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    """Read splits.csv, verify it against splits_meta.json and the frame, and return it."""
    path = Path(path)
    meta_path = path.with_name(OUT_META)
    if not path.is_file() or not meta_path.is_file():
        raise InputProblem("splits.csv or splits_meta.json is missing")
    meta = json.loads(meta_path.read_text())
    if sha8_of(path.read_bytes()) != meta.get("splits_file_sha8"):
        raise GuardFailure("splits.csv does not match splits_meta.json (file hash)")
    splits = pd.read_csv(path, dtype={"participant": str})
    if list(splits.columns) != SPLIT_COLUMNS:
        raise GuardFailure(f"splits must have the columns {SPLIT_COLUMNS}")
    if splits_sha8(splits) != meta.get("splits_sha8"):
        raise GuardFailure("splits.csv does not match splits_meta.json (row hash)")
    check_partition(splits, set(frame["participant"]), int(meta["repeats"]))
    return splits


def input_hashes(features: Path, roles: Path, meta: Path) -> dict[str, str]:
    return {
        "features": file_sha256(features)[:HASH_LEN],
        "roles": file_sha256(roles)[:HASH_LEN],
        "features_meta": file_sha256(meta)[:HASH_LEN],
    }


def write_outputs(
    out_dir: Path,
    plan: SplitPlan,
    meta: dict[str, Any],
    force: bool,
) -> None:
    splits_path = out_dir / OUT_SPLITS
    data = splits_bytes(plan.splits)
    if splits_path.exists() and splits_path.read_bytes() != data and not force:
        raise GuardFailure(
            f"refusing to overwrite an existing {OUT_SPLITS} with different content; "
            "use --force (part 2 relies on the file being stable)"
        )
    meta = {**meta, "splits_file_sha8": sha8_of(data)}
    _atomic(splits_path, data)
    _atomic(out_dir / OUT_META, (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode())


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--features", type=Path, default=None)
    ap.add_argument("--roles", type=Path, default=None)
    ap.add_argument("--meta", type=Path, default=None)
    ap.add_argument("--target", choices=sorted(TARGETS), default=DEFAULT_TARGET)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    features = Path(args.features or PROCESSED_ROOT / "features.csv")
    roles = Path(args.roles or PROCESSED_ROOT / "feature_roles.csv")
    meta_path = Path(args.meta or PROCESSED_ROOT / "features_meta.json")
    out_dir = Path(args.out_dir or CV_ROOT)
    if not is_inside(out_dir, CV_ROOT):
        print("error: --out-dir must be inside data/processed/cv", file=sys.stderr)
        return 2
    try:
        frame, _ = load_inputs(features, roles, meta_path)
        plan = build_plan(frame, args.target, args.repeats, args.seed)
        roles_meta = json.loads(meta_path.read_text())["roles_sha8"]
        meta = {
            "tool": TOOL_VERSION,
            "target": args.target,
            "threshold": TARGETS[args.target],
            "seed": args.seed,
            "repeats": args.repeats,
            "outer_seeds": [outer_seed(args.seed, r) for r in range(args.repeats)],
            "inner_seed_formula": "1000 * repeat + outer_fold + 1",
            "n_outer": N_OUTER,
            "n_inner": N_INNER,
            "rows": len(frame),
            "participants": int(frame["participant"].nunique()),
            "inputs_sha8": input_hashes(features, roles, meta_path),
            "roles_sha8": roles_meta,
            "gate": plan.gate,
            "splits_sha8": plan.splits_sha8,
            "inner_sha8": plan.inner_sha8,
            "versions": {
                "python": platform.python_version(),
                "scikit_learn": sklearn.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        }
        write_outputs(out_dir / args.target, plan, meta, args.force)
    except InputProblem as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except (GuardFailure, LeakageError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    gate = plan.gate
    print(
        f"wrote {args.target}/{OUT_SPLITS}: {meta['participants']} participants x "
        f"{args.repeats} repeats, gate ok"
    )
    for part in ("outer_test", "inner_validation"):
        g = gate[part]
        print(
            f"{part}: folds {g['folds']}, positive clips {g['pos_clips']['min']}-"
            f"{g['pos_clips']['max']}, positive participants {g['pos_participants']['min']}-"
            f"{g['pos_participants']['max']}, negative participants "
            f"{g['neg_participants']['min']}-{g['neg_participants']['max']}"
        )
    print(f"splits_sha8 {plan.splits_sha8}  inner_sha8 {plan.inner_sha8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
