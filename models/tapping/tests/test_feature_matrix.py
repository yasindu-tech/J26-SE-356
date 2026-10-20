"""Tests for models/tapping/src/feature_matrix.py (task FT-30). Synthetic fixtures only, with a
decoy label file whose raw IDs carry CONTROL/ID prefixes; the last test checks the real export's
counts and skips when the data is absent."""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import feature_matrix as fm  # noqa: E402
from hubu_fis import pseudonym  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src"
IDS = [
    "CONTROL01_DCHA", "CONTROL01_IZDA", "CONTROL02_DCHA", "CONTROL02_IZDA",
    "ID1001_DCHA", "ID1001_IZDA", "ID1002_DCHA", "ID1002_IZDA",
]  # fmt: skip
EXTRA = "ID1003_DCHA"  # labelled but not retained (like the rejected clip)
SET_A_COLUMNS = (
    "ref", "n_taps", "duration_s", "fps", "qc_status", "n_over_duration_hz", "tap_rate_hz",
    "iti_mean_s", "iti_cv", "open_vel_peak", "open_vel_peak_median", "close_vel_peak",
    "close_vel_peak_median", "open_vel_mean", "open_vel_mean_median", "close_vel_mean",
    "close_vel_mean_median", "amp_mean", "amp_max", "n_one_sided_excluded", "n_iti_excluded",
    "n_short_iti", "tap_rate_hz_sens", "iti_mean_s_sens", "iti_cv_sens", "open_vel_peak_sens",
    "open_vel_peak_median_sens", "close_vel_peak_sens", "close_vel_peak_median_sens",
    "open_vel_mean_sens", "open_vel_mean_median_sens", "close_vel_mean_sens",
    "close_vel_mean_median_sens", "amp_mean_sens", "amp_max_sens", "nan_reasons",
)  # fmt: skip
SET_B_COLUMNS = (
    "ref", "qc_status", "n_valid_taps", "amp_slope", "open_vel_slope", "close_vel_slope",
    "amp_late_early", "amp_min_initial", "decrement_present", "decrement_onset_frac", "iti_slope",
    "n_hesitations", "n_halts", "longest_pause_ratio", "longest_pause_s", "amp_slope_sens",
    "open_vel_slope_sens", "close_vel_slope_sens", "amp_late_early_sens", "amp_min_initial_sens",
    "decrement_present_sens", "decrement_onset_frac_sens", "iti_slope_sens", "n_hesitations_sens",
    "n_halts_sens", "longest_pause_ratio_sens", "longest_pause_s_sens", "nan_reasons",
)  # fmt: skip
RATING = {i: n % 4 for n, i in enumerate(IDS)}
PINNED_ROLES_SHA256 = "4f92d429da7809d9d570040e9b93e0c0b462093a8035c4f0e1d2ed4337b93ba7"
RAW_PATTERN = re.compile(r"(CONTROL|ID)\d+|DCHA|IZDA|CONTROL", re.IGNORECASE)


def sha8(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def feature_row(
    columns: tuple[str, ...], ref: str, rng: random.Random, flag: bool
) -> dict[str, str]:
    row = {}
    for c in columns:
        if c == "ref":
            row[c] = ref
        elif c == "qc_status":
            row[c] = "retained_flag" if flag else "retained"
        elif c == "nan_reasons":
            row[c] = (
                ""
                if "n_halts" not in columns
                else ("decrement_onset_frac:no_decrement" if flag else "")
            )
        elif c == "decrement_present":
            row[c] = rng.choice(["True", "False", ""])
        elif c.startswith(("n_", "decrement_present")) and not c.endswith(("_hz", "_sens")):
            row[c] = str(rng.randint(0, 40))
        elif rng.random() < 0.08:
            row[c] = ""  # an empty cell must stay empty
        else:
            row[c] = repr(rng.random() * 3 + 1e-9)  # many digits: rounding would show
    return row


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    interim, processed, docs = tmp_path / "interim", tmp_path / "processed", tmp_path / "docs"
    features = interim / "features"
    features.mkdir(parents=True)
    docs.mkdir()
    raw = tmp_path / "raw"
    raw.mkdir()
    refs = sorted(pseudonym(i) for i in IDS)
    rng = random.Random(5)
    flags = {r: rng.random() < 0.4 for r in refs}
    rows_a = [feature_row(SET_A_COLUMNS, r, rng, flags[r]) for r in refs]
    rows_b = [feature_row(SET_B_COLUMNS, r, rng, flags[r]) for r in refs]
    pd.DataFrame(rows_a, columns=list(SET_A_COLUMNS)).to_csv(
        features / "features_a.csv", index=False
    )
    pd.DataFrame(rows_b, columns=list(SET_B_COLUMNS)).to_csv(
        features / "features_b.csv", index=False
    )
    pd.DataFrame({"ref": refs, "verdict": "ok", "note": ""}).to_csv(
        docs / "qc_retained.csv", index=False
    )
    meta = {"tap_params_sha8": "3c477a5e", "taps_summary_sha8": "11111111",
            "qc_list_sha8": sha8(docs / "qc_retained.csv")}  # fmt: skip
    (features / "features_a_meta.json").write_text(
        json.dumps({**meta, "definitions_version": "features-a-1"})
    )
    (features / "features_b_meta.json").write_text(
        json.dumps({**meta, "definitions_version": "features-b-1", "definitions_sha8": "65bce75b"})
    )
    labels = raw / "fis_diagnostic.csv"
    write_labels(labels, [*IDS, EXTRA], lambda i: RATING.get(i, 2))
    for module in (fm,):
        monkeypatch.setattr(module, "INTERIM_ROOT", interim)
        monkeypatch.setattr(module, "PROCESSED_ROOT", processed)
        monkeypatch.setattr(module, "DOCS_ROOT", docs)
    return {"features": features, "processed": processed, "docs": docs, "labels": labels,
            "refs": refs, "root": tmp_path, "rows_a": rows_a, "rows_b": rows_b}  # fmt: skip


def write_labels(path: Path, ids: list[str], rating: Any) -> None:
    pd.DataFrame({"ID": ids, "UPDRS": [str(rating(i)) for i in ids]}).to_csv(path, index=False)


def run(w: dict[str, Any], *extra: str) -> int:
    return fm.main(["--features-dir", str(w["features"]), "--labels", str(w["labels"]),
                    "--out-dir", str(w["processed"]), "--docs-dir", str(w["docs"]), *extra])  # fmt: skip


def read_out(w: dict[str, Any], name: str = fm.OUT_FEATURES) -> pd.DataFrame:
    return pd.read_csv(w["processed"] / name, dtype=str, keep_default_na=False)


# ----------------------------------------------------------------------- roles registry


def test_roles_cover_the_real_column_set_exactly() -> None:
    expected = [*fm.ID_COLUMNS]
    expected += [c for c in SET_A_COLUMNS if c not in ("ref", "nan_reasons")] + ["nan_reasons_a"]
    expected += [c for c in SET_B_COLUMNS if c not in ("ref", "qc_status", "nan_reasons")]
    expected += ["nan_reasons_b"]
    assert sorted(fm.ROLES) == sorted(expected) and len(expected) == 65
    assert len(set(expected)) == 65


def test_roles_vocabulary_is_closed_and_counts_are_fixed() -> None:
    assert {r for r, _ in fm.ROLES.values()} <= set(fm.VOCABULARY)
    per_role = {r: len(fm.columns_with_role(r)) for r in fm.VOCABULARY}
    assert per_role == {
        "id": 1, "group_key": 1, "label": 1, "covariate_candidate": 1, "feature_primary": 13,
        "feature_twin": 4, "feature_secondary": 8, "feature_exploratory": 1,
        "feature_descriptive": 1, "feature_sensitivity": 25, "audit_count": 9,
    }  # fmt: skip
    assert all(c.endswith("_sens") for c in fm.columns_with_role("feature_sensitivity"))
    assert all(c.endswith("_sens") for c in fm.ROLES if fm.ROLES[c][0] == "feature_sensitivity")
    assert all(fm.ROLES[c][0] == "feature_sensitivity" for c in fm.ROLES if c.endswith("_sens"))
    with pytest.raises(ValueError):
        fm.columns_with_role("feature_primry")


def test_specific_roles_decided_before_the_join() -> None:
    r = {c: v[0] for c, v in fm.ROLES.items()}
    assert r["iti_slope"] == "feature_exploratory" and r["n_halts"] == "feature_descriptive"
    assert r["amp_late_early"] == "feature_twin" and r["amp_slope"] == "feature_primary"
    assert r["decrement_onset_frac"] == "feature_primary" and r["fps"] == "audit_count"
    assert (
        r["rating"] == "label"
        and r["participant"] == "group_key"
        and r["hand"] == "covariate_candidate"
    )
    assert (
        fm.columns_with_role("feature_primary") == fm.PRIMARY_COLUMNS
        and len(fm.PRIMARY_COLUMNS) == 13
    )


def test_identity_label_and_audit_columns_are_in_no_feature_role() -> None:
    feature = {c for c, (r, _) in fm.ROLES.items() if r.startswith("feature_")}
    for c in (*fm.ID_COLUMNS, "n_valid_taps", "n_one_sided_excluded", "n_iti_excluded", "n_short_iti",
              "fps", "duration_s", "qc_status", "nan_reasons_a", "nan_reasons_b"):  # fmt: skip
        assert c not in feature, c
    assert not any(
        fm.ROLES[c][0] in fm.FEATURE_ROLES
        for c in fm.columns_with_role("id", "label", "audit_count")
    )


def test_roles_registry_is_pinned() -> None:
    text = json.dumps(
        {c: list(v) for c, v in fm.ROLES.items()}, sort_keys=True, separators=(",", ":")
    )
    assert hashlib.sha256(text.encode()).hexdigest() == PINNED_ROLES_SHA256
    assert fm.roles_sha8() == PINNED_ROLES_SHA256[:8]


# ----------------------------------------------------------------------- the export


def test_export_columns_rows_and_exact_join(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(world) == 0
    out = read_out(world)
    assert list(out.columns[:4]) == ["ref", "participant", "hand", "rating"]
    assert out["ref"].tolist() == sorted(pseudonym(i) for i in IDS)  # sorted by ref only
    by_ref = {pseudonym(i): i for i in IDS}
    for _, row in out.iterrows():
        raw_id = by_ref[row["ref"]]
        assert row["rating"] == str(RATING[raw_id])
        assert row["hand"] == ("right" if raw_id.endswith("DCHA") else "left")
        assert row["participant"] == pseudonym(raw_id.split("_")[0])
    assert (
        "wrote features.csv: 8 rows, 65 columns, 4 participants, matched 8, unmatched 0"
        in capsys.readouterr().out
    )
    roles = read_out(world, fm.OUT_ROLES)
    assert list(roles.columns) == ["column", "role", "note"] and roles["column"].tolist() == list(
        out.columns
    )
    assert len(roles) == 65 and roles["role"].isin(fm.VOCABULARY).all()


def test_feature_cells_are_copied_byte_for_byte(world: dict[str, Any]) -> None:
    assert run(world) == 0
    out = read_out(world).set_index("ref")
    a = pd.read_csv(
        world["features"] / "features_a.csv", dtype=str, keep_default_na=False
    ).set_index("ref")
    b = pd.read_csv(
        world["features"] / "features_b.csv", dtype=str, keep_default_na=False
    ).set_index("ref")
    for col in a.columns:
        name = "nan_reasons_a" if col == "nan_reasons" else col
        assert out[name].tolist() == a.loc[out.index, col].tolist(), col
    for col in b.columns:
        if col == "qc_status":
            continue
        name = "nan_reasons_b" if col == "nan_reasons" else col
        assert out[name].tolist() == b.loc[out.index, col].tolist(), col
    assert out["qc_status"].tolist() == a.loc[out.index, "qc_status"].tolist()
    assert "nan_reasons" not in out.columns and out.columns.tolist().count("qc_status") == 1
    assert any(v == "" for v in out.to_numpy().ravel())  # empty cells stay empty, never 0
    long_float = [v for v in out["tap_rate_hz"] if len(v) > 12]
    assert long_float, "digits must survive unrounded"


def test_shuffling_the_label_file_changes_nothing(world: dict[str, Any]) -> None:
    assert run(world) == 0
    first = (world["processed"] / fm.OUT_FEATURES).read_bytes()
    labels = pd.read_csv(world["labels"], dtype=str)
    labels.sample(frac=1, random_state=3).to_csv(world["labels"], index=False)
    assert run(world) == 0
    assert (world["processed"] / fm.OUT_FEATURES).read_bytes() == first


def test_second_run_is_byte_identical(world: dict[str, Any]) -> None:
    assert run(world) == 0
    first = {
        n: (world["processed"] / n).read_bytes()
        for n in (fm.OUT_FEATURES, fm.OUT_ROLES, fm.OUT_META)
    }
    assert run(world) == 0
    assert {n: (world["processed"] / n).read_bytes() for n in first} == first


def test_no_raw_id_group_or_line_anywhere(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(world) == 0
    shown = capsys.readouterr()
    text = shown.out + shown.err
    for name in (fm.OUT_FEATURES, fm.OUT_ROLES, fm.OUT_META):
        text += (world["processed"] / name).read_text()
    assert not RAW_PATTERN.search(text)
    columns = read_out(world).columns
    assert not any(re.search(r"group|^id$|line", c, re.IGNORECASE) for c in columns)
    meta = json.loads((world["processed"] / fm.OUT_META).read_text())
    assert meta["labels_used"] == ["UPDRS"] and meta["group_dropped"] is True


def test_meta_records_inputs_counts_and_hashes(world: dict[str, Any]) -> None:
    assert run(world) == 0
    meta = json.loads((world["processed"] / fm.OUT_META).read_text())
    ins = meta["inputs_sha8"]
    assert ins["features_a_csv"] == sha8(world["features"] / "features_a.csv")
    assert ins["features_b_csv"] == sha8(world["features"] / "features_b.csv")
    assert ins["features_a_meta"] == sha8(world["features"] / "features_a_meta.json")
    assert ins["features_b_meta"] == sha8(world["features"] / "features_b_meta.json")
    assert ins["labels"] == sha8(world["labels"]) and ins["qc_list"] == sha8(
        world["docs"] / "qc_retained.csv"
    )
    assert meta["roles_sha8"] == PINNED_ROLES_SHA256[:8] and meta["tap_params_sha8"] == "3c477a5e"
    assert meta["definitions"]["set_b_definitions_sha8"] == "65bce75b"
    assert (meta["rows"], meta["columns"], meta["participants"]) == (8, 65, 4)
    assert meta["join"] == {"matched": 8, "unmatched": 0, "labels_unused": 1}
    assert (
        meta["columns_per_role"]["feature_primary"] == 13
        and meta["columns_per_role"]["audit_count"] == 9
    )
    assert set(meta["non_modal_clips"]) == {"n_halts", "n_hesitations", "decrement_present"}
    text = json.dumps(meta)
    assert not re.search(
        r"\d{4}-\d{2}-\d{2}|utc|timestamp|created", text, re.IGNORECASE
    )  # no clock


# ----------------------------------------------------------------------- refusals


def test_unmatched_clip_is_exit_1_and_never_a_rating(world: dict[str, Any]) -> None:
    labels = pd.read_csv(world["labels"], dtype=str)
    labels[labels["ID"] != IDS[3]].to_csv(world["labels"], index=False)
    assert run(world) == 1
    assert not (world["processed"] / fm.OUT_FEATURES).exists()


def test_near_miss_keys_do_not_match(world: dict[str, Any]) -> None:
    # a feature ref that differs from the label's hash only in its last two characters
    for name in ("features_a.csv", "features_b.csv"):
        df = pd.read_csv(world["features"] / name, dtype=str, keep_default_na=False)
        near = df.loc[0, "ref"][:6] + ("00" if df.loc[0, "ref"][6:] != "00" else "ff")
        df.loc[0, "ref"] = near
        df.sort_values("ref").to_csv(world["features"] / name, index=False)
    assert run(world) == 1  # no prefix or fuzzy matching: the clip has no label
    assert not (world["processed"] / fm.OUT_FEATURES).exists()


def test_duplicate_label_rows_are_an_error(world: dict[str, Any]) -> None:
    labels = pd.read_csv(world["labels"], dtype=str)
    pd.concat([labels, labels.iloc[[0]]]).to_csv(world["labels"], index=False)
    assert run(world) == 1


def test_blank_rating_is_an_error_not_zero(world: dict[str, Any]) -> None:
    labels = pd.read_csv(world["labels"], dtype=str, keep_default_na=False)
    labels.loc[0, "UPDRS"] = ""
    labels.to_csv(world["labels"], index=False)
    assert run(world) == 1


@pytest.mark.parametrize("key", ["tap_params_sha8", "taps_summary_sha8", "qc_list_sha8"])
def test_disagreeing_metas_are_exit_1(world: dict[str, Any], key: str) -> None:
    path = world["features"] / "features_b_meta.json"
    meta = json.loads(path.read_text())
    meta[key] = "deadbeef"
    path.write_text(json.dumps(meta))
    assert run(world) == 1


def test_stale_qc_list_is_exit_1(world: dict[str, Any]) -> None:
    (world["docs"] / "qc_retained.csv").write_text("ref,verdict,note\n")
    assert run(world) == 1


def test_different_ref_sets_are_exit_1(world: dict[str, Any]) -> None:
    b = pd.read_csv(world["features"] / "features_b.csv", dtype=str, keep_default_na=False)
    b.loc[0, "ref"] = pseudonym(EXTRA)
    b.to_csv(world["features"] / "features_b.csv", index=False)
    assert run(world) == 1


def test_qc_status_mismatch_is_exit_1(world: dict[str, Any]) -> None:
    b = pd.read_csv(world["features"] / "features_b.csv", dtype=str, keep_default_na=False)
    b.loc[0, "qc_status"] = "retained_flag" if b.loc[0, "qc_status"] == "retained" else "retained"
    b.to_csv(world["features"] / "features_b.csv", index=False)
    assert run(world) == 1


def test_a_new_column_without_a_role_is_exit_1(world: dict[str, Any]) -> None:
    a = pd.read_csv(world["features"] / "features_a.csv", dtype=str, keep_default_na=False)
    a["brand_new"] = "1"
    a.to_csv(world["features"] / "features_a.csv", index=False)
    assert run(world) == 1


def test_missing_inputs_are_exit_2(world: dict[str, Any]) -> None:
    world["labels"].unlink()
    assert run(world) == 2
    write_labels(world["labels"], IDS, lambda i: 1)
    (world["features"] / "features_a_meta.json").unlink()
    assert run(world) == 2


@pytest.mark.parametrize("flag", ["--out-dir", "--features-dir", "--docs-dir"])
def test_paths_outside_their_folders_are_exit_2_and_write_nothing(
    world: dict[str, Any], tmp_path: Path, flag: str
) -> None:
    elsewhere = tmp_path / "elsewhere"
    base = {"--features-dir": world["features"], "--out-dir": world["processed"],
            "--docs-dir": world["docs"]}  # fmt: skip
    base[flag] = elsewhere
    argv = ["--labels", str(world["labels"])] + [x for k, v in base.items() for x in (k, str(v))]
    assert fm.main(argv) == 2
    assert not elsewhere.exists() and not world["processed"].exists()


# ----------------------------------------------------------------------- leakage


def test_feature_modules_never_touch_labels_or_identity() -> None:
    pattern = re.compile(
        r"\b(hubu_fis|clip_id|fis_diagnostic|pseudonym|load_labels|feature_matrix)\b"
    )
    for name in ("features_a.py", "features_b.py", "tap_table.py"):
        if (SRC / name).exists():  # features_b.py is absent on a branch cut before FT-28
            assert not pattern.search((SRC / name).read_text()), name


def test_only_the_allowed_inputs_are_opened(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    decoy = world["root"] / "elsewhere"
    decoy.mkdir()
    (decoy / "other_labels.csv").write_text("ID,UPDRS\nCONTROL09_DCHA,0\n")
    allowed = [world["features"].resolve(), world["labels"].resolve(), world["processed"].resolve(),
               world["docs"].resolve()]  # fmt: skip
    seen: list[str] = []

    def guard(path: Any) -> None:
        if isinstance(path, str | Path) and str(path):
            p = Path(path).resolve()
            seen.append(p.name)
            ok = p.suffix in (".py", ".pyc", ".so") or any(
                p == a or p.is_relative_to(a) for a in allowed
            )
            if not ok:
                raise AssertionError(f"read outside the allowed inputs: {p.name}")

    def wrap(fn: Any) -> Any:
        def inner(path: Any, *a: Any, **k: Any) -> Any:
            guard(path)
            return fn(path, *a, **k)

        return inner

    for owner, name in ((builtins, "open"), (io, "open"), (pd, "read_csv"), (pd, "read_parquet"),
                        (pq, "read_table")):  # fmt: skip
        monkeypatch.setattr(owner, name, wrap(getattr(owner, name)))
    assert run(world) == 0
    assert seen and "other_labels.csv" not in seen
    with pytest.raises(AssertionError):
        pd.read_csv(decoy / "other_labels.csv")


# ----------------------------------------------------------------------- the real export


def test_real_export_counts(capsys: pytest.CaptureFixture[str]) -> None:
    features = fm.PROCESSED_ROOT / "features.csv"
    roles = fm.PROCESSED_ROOT / "feature_roles.csv"
    if not (features.is_file() and roles.is_file()):
        pytest.skip("real export not available")
    df = pd.read_csv(features, dtype=str, keep_default_na=False)
    rl = pd.read_csv(roles, dtype=str, keep_default_na=False)
    assert len(df) == 233 and (df["rating"] == "").sum() == 0
    assert df["participant"].nunique() == 117 and df["ref"].is_unique
    assert set(df.columns) == set(fm.ROLES) and rl["column"].tolist() == df.columns.tolist()
    assert df["ref"].tolist() == sorted(df["ref"])
    with capsys.disabled():
        print(f"\nreal export: {len(df)} rows, {df.shape[1]} columns, every column has a role")
