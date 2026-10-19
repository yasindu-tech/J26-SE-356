"""Tests for models/tapping/src/bootstrap_ci.py (task FT-38). Two kinds of input: a small hand-built
world with known answers (independent recomputation of the bootstrap), and a world produced by the
real cv_harness on a synthetic matrix. No real data is used."""

from __future__ import annotations

import builtins
import io
import json
import re
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bootstrap_ci as bc  # noqa: E402
import cv_harness as ch  # noqa: E402
import cv_splits as cvs  # noqa: E402
from test_cv_harness import SMALL_LGBM, build_world, harness, make_data  # noqa: E402
from test_cv_splits import hid  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src"
HASH = re.compile(r"\b[0-9a-f]{8}\b")
MODEL_KEYS = list(bc.MODELS)


# --------------------------------------------------------------------------- a hand-built world


def craft(
    tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
    kinds: dict[tuple[str, str], str] | None = None,
    n_part: int = 12,
    singleton: bool = True,
    duplicate_clips: bool = False,
    seed: int = 0,
    target: str = "ge2",
) -> dict[str, Any]:
    """Features, oof, runs and meta files for the four models. kinds maps a model to how its
    scores are made: signal, ties, const, repeat_copy, or same_as_<index of MODEL_KEYS>."""
    rng = np.random.default_rng(seed)
    cv_root = tmp / "data" / "processed" / "cv"
    monkeypatch.setattr(cvs, "CV_ROOT", cv_root)
    rows = []
    z = rng.permutation(np.arange(n_part) % 2)
    for i in range(n_part):
        for c in range(1 if (singleton and i == n_part - 1) else 2):
            rows.append(
                (
                    hid("cc", 100 * i + c),
                    hid("cp", i),
                    int(z[i])
                    if (duplicate_clips or c == 0)
                    else int(rng.random() < 0.3 + 0.4 * z[i]),
                )
            )
    frame = (
        pd.DataFrame(rows, columns=["ref", "participant", "y"])
        .sort_values("ref")
        .reset_index(drop=True)
    )
    features = tmp / "data" / "processed" / "features.csv"
    features.parent.mkdir(parents=True, exist_ok=True)
    frame[["ref", "participant"]].to_csv(features, index=False)
    y = frame["y"].to_numpy()
    base = {
        p: rng.uniform(size=5) for p in frame["participant"].unique()
    }  # participant-level noise
    kinds = kinds or {
        MODEL_KEYS[0]: "const",
        MODEL_KEYS[1]: "signal",
        MODEL_KEYS[2]: "ties",
        MODEL_KEYS[3]: "same_as_2",
    }
    probas: dict[tuple[str, str], np.ndarray] = {}
    for key in MODEL_KEYS:
        kind = kinds[key]
        if kind.startswith("same_as_"):
            probas[key] = probas[MODEL_KEYS[int(kind[-1])]]
            continue
        p = np.zeros((5, len(y)))
        for r in range(5 if kind != "repeat_copy" else 1):
            noise = rng.uniform(size=len(y))
            if duplicate_clips:  # the two clips of a participant get identical scores
                noise = np.array([base[q][r] for q in frame["participant"]])
            p[r] = {
                "const": np.full(len(y), 0.3),
                "signal": 0.25 * y + 0.75 * noise,
                "ties": np.round(0.3 * y + 0.7 * noise, 1),
                "repeat_copy": 0.3 * y + 0.7 * noise,
            }[kind]
        if kind == "repeat_copy":
            p[:] = p[0]
        probas[key] = np.clip(p, 0, 1)
    folder = cv_root / target
    folder.mkdir(parents=True)
    for (model, fs), p in probas.items():
        tag = f"{model}_{fs}"
        oof = pd.concat(
            [
                pd.DataFrame(
                    {
                        "ref": frame["ref"],
                        "repeat": r,
                        "outer_fold": rng.integers(0, 5, len(y)),
                        "y": y,
                        "proba": p[r],
                    }
                )
                for r in range(5)
            ],
            ignore_index=True,
        )
        oof.to_csv(folder / f"oof_{tag}.csv", index=False, float_format="%.17g")
        runs = []
        for r in range(5):
            m = ch.threshold_metrics(y, p[r])
            runs.append(
                {
                    "repeat": r,
                    "n_clips": len(y),
                    "n_pos": int(y.sum()),
                    "n_neg": int((1 - y).sum()),
                    "auroc": ch.auroc(y, p[r]),
                    **{k2: v for k2, v in m.items()},
                }
            )
        pd.DataFrame(runs).to_csv(folder / f"runs_{tag}.csv", index=False, float_format="%.17g")
        meta = {
            "target": target,
            "model": model,
            "feature_set": fs,
            "repeats": 5,
            "splits_sha8": "11111111",
            "inner_sha8": "22222222",
            "roles_sha8": "33333333",
            "target_status": "pending_default_ge2",
            "features_sha8": "aaaaaaaa" if fs == "A" else "bbbbbbbb",
        }
        (folder / f"meta_{tag}.json").write_text(json.dumps(meta))
    return {
        "root": cv_root,
        "features": features,
        "frame": frame,
        "y": y,
        "probas": probas,
        "tmp": tmp,
        "target": target,
    }


def reference(
    world: dict[str, Any],
    key: tuple[str, str],
    n_draws: int,
    by: str = "participant",
    rng_seed: int = 7,
) -> dict[str, tuple[float, float, float, int]]:
    """Straightforward bootstrap: build each resampled dataset explicitly and use the harness's own
    metric functions. Returns metric -> (estimate, low, high, n_used)."""
    frame, y, p = world["frame"], world["y"], world["probas"][key]
    order = bc.sorted_participants(frame["participant"].tolist())
    pidx = np.array([order.index(q) for q in frame["participant"]])
    if by == "participant":
        counts = bc.bootstrap_draws(frame["participant"].tolist(), n_draws, bc.SEED)

        def resample(d: int) -> np.ndarray:
            return np.repeat(np.arange(len(y)), counts[d][pidx])
    else:
        rng = np.random.default_rng(rng_seed)

        def resample(d: int) -> np.ndarray:
            return rng.integers(0, len(y), len(y))

    def stat(idx: np.ndarray) -> dict[str, float]:
        out: dict[str, list[float]] = {m: [] for m in bc.METRICS}
        for r in range(5):
            m = {**ch.threshold_metrics(y[idx], p[r][idx]), "auroc": ch.auroc(y[idx], p[r][idx])}
            for name in bc.METRICS:
                out[name].append(m[name])
        return {name: float(np.mean(v)) for name, v in out.items()}

    point = stat(np.arange(len(y)))
    draws = []
    for d in range(n_draws):
        idx = resample(d)
        if y[idx].sum() == 0 or y[idx].sum() == len(idx):
            continue
        draws.append(stat(idx))
    result = {}
    for name in bc.METRICS:
        vals = np.array([s[name] for s in draws])
        finite = vals[np.isfinite(vals)]
        lo, hi = np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan)
        result[name] = (point[name], float(lo), float(hi), len(draws))
    return result


@pytest.fixture
def small(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(bc, "N_DRAWS", 60)
    return craft(tmp_path, monkeypatch)


def run_ci(world: dict[str, Any], *extra: str, features: Path | None = None) -> int:
    targets = [] if "--targets" in extra else ["--targets", world["target"]]
    return bc.main(
        [
            "--cv-root",
            str(world["root"]),
            "--features",
            str(features or world["features"]),
            *targets,
            *extra,
        ]
    )


def table(world: dict[str, Any]) -> pd.DataFrame:
    text = {
        c: str for c in ("draws_sha8", "splits_sha8", "inner_sha8", "roles_sha8", "features_sha8")
    }
    path = world["root"] / "reports_ci" / "ci_table.csv"
    return pd.read_csv(path, float_precision="round_trip", dtype=text)


def row(t: pd.DataFrame, key: tuple[str, str], metric: str) -> pd.Series:
    return t[(t["model"] == key[0]) & (t["feature_set"] == key[1]) & (t["metric"] == metric)].iloc[
        0
    ]


# --------------------------------------------------------------------------- the design is pinned


def test_the_preregistered_constants_are_pinned() -> None:
    assert (bc.SEED, bc.N_DRAWS, bc.CI_PERCENTILES, bc.MAX_EXCLUDED_SHARE) == (
        2026,
        2000,
        (2.5, 97.5),
        0.01,
    )
    assert (bc.THRESHOLD, bc.PPV_PREVALENCES, bc.EXPECTED_REPEATS) == (0.5, (0.01, 0.02, 0.05), 5)
    assert bc.METRICS == (
        "auroc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "mcc",
        "ppv_at_1pct",
        "ppv_at_2pct",
        "ppv_at_5pct",
    )
    assert bc.MODELS == (("b0", "A"), ("b1", "A"), ("lgbm", "A"), ("lgbm", "B"))
    assert bc.sorted_participants(["b", "a", "b"]) == ["a", "b"]


def test_the_draws_are_deterministic_and_independent_of_the_participant_order() -> None:
    people = [hid("d", i) for i in range(20)]
    a = bc.bootstrap_draws(people, 50, 2026)
    assert a.shape == (50, 20) and (a.sum(axis=1) == 20).all() and a.min() >= 0
    assert np.array_equal(a, bc.bootstrap_draws(list(reversed(people)), 50, 2026))
    assert np.array_equal(
        a, bc.bootstrap_draws(people + people, 50, 2026)
    )  # a clip-level list gives the same draws
    assert not np.array_equal(a, bc.bootstrap_draws(people, 50, 2027))
    assert bc.draws_sha8(a) == bc.draws_sha8(a.copy()) and bc.draws_sha8(a) != bc.draws_sha8(
        bc.bootstrap_draws(people, 50, 2027)
    )
    assert bc.bootstrap_draws(["p1", "p2", "p3", "p4"], 3, 2026).tolist() == [
        [2, 0, 1, 1],
        [1, 3, 0, 0],
        [0, 1, 1, 2],
    ]  # numpy stream guard


def test_the_weighted_metrics_equal_the_harness_on_unit_and_integer_weights() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 80)
    p = np.round(rng.uniform(size=80), 1)  # many ties
    unit = bc.weighted_threshold_metrics(y, p, np.ones((1, 80)))
    expected = ch.threshold_metrics(y, p)
    for name, value in unit.items():
        assert value[0] == pytest.approx(expected[name], nan_ok=True)
    assert bc.weighted_auroc(y, p, np.ones((1, 80)))[0] == pytest.approx(ch.auroc(y, p))
    w = rng.integers(0, 4, (3, 80)).astype(float)
    for d in range(3):  # a weight is the number of copies of the clip
        idx = np.repeat(np.arange(80), w[d].astype(int))
        assert bc.weighted_auroc(y, p, w)[d] == pytest.approx(ch.auroc(y[idx], p[idx]))
        for name, value in bc.weighted_threshold_metrics(y, p, w).items():
            assert value[d] == pytest.approx(
                ch.threshold_metrics(y[idx], p[idx])[name], nan_ok=True
            )


def test_mcc_is_zero_and_ppv_is_nan_for_a_classifier_that_never_predicts_positive() -> None:
    y = np.array([1, 1, 0, 0, 0, 0])
    m = bc.weighted_threshold_metrics(y, np.full(6, 0.2), np.ones((1, 6)))
    assert (
        m["mcc"][0] == 0.0
        and np.isnan(m["ppv_at_2pct"][0])
        and m["sensitivity"][0] == 0.0
        and m["specificity"][0] == 1.0
    )


# --------------------------------------------------------------------------- the statistic


def test_the_table_matches_an_independent_bootstrap(small: dict[str, Any]) -> None:
    assert run_ci(small) == 0
    t = table(small)
    for key in (("b1", "A"), ("lgbm", "A"), ("b0", "A")):
        expected = reference(small, key, 60)
        for metric in bc.METRICS:
            r = row(t, key, metric)
            est, lo, hi, used = expected[metric]
            assert r["estimate"] == pytest.approx(est, abs=1e-9, nan_ok=True)
            assert r["ci_low"] == pytest.approx(lo, abs=1e-9, nan_ok=True) and r[
                "ci_high"
            ] == pytest.approx(hi, abs=1e-9, nan_ok=True)
            assert r["n_draws_used"] == used == 60


def test_the_point_estimate_is_the_mean_of_the_runs_over_repeats(small: dict[str, Any]) -> None:
    assert run_ci(small) == 0
    t = table(small)
    for key in MODEL_KEYS:
        runs = pd.read_csv(
            small["root"] / small["target"] / f"runs_{key[0]}_{key[1]}.csv",
            float_precision="round_trip",
        )
        for metric in bc.METRICS:
            assert row(t, key, metric)["estimate"] == pytest.approx(
                runs[metric].mean(), abs=1e-9, nan_ok=True
            )


def test_draws_are_paired_across_models_and_feature_sets(
    small: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real = bc.bootstrap_draws

    def counting(*a: Any, **k: Any) -> np.ndarray:
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(bc, "bootstrap_draws", counting)
    assert run_ci(small) == 0
    assert len(calls) == 1  # one draw set for the whole run
    t = table(small)
    for metric in bc.METRICS:  # lgbm B is a copy of lgbm A, so equal draws give equal intervals
        a, b = row(t, ("lgbm", "A"), metric), row(t, ("lgbm", "B"), metric)
        assert (a["estimate"], a["ci_low"], a["ci_high"]) == pytest.approx(
            (b["estimate"], b["ci_low"], b["ci_high"]), nan_ok=True
        )
    assert t["draws_sha8"].nunique() == 1 and t["n_draws_used"].nunique() == 1


def test_the_same_draws_are_used_for_every_repeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bc, "N_DRAWS", 80)
    kinds = {k: "repeat_copy" for k in MODEL_KEYS}
    world = craft(tmp_path, monkeypatch, kinds=kinds, n_part=14)
    assert run_ci(world) == 0
    one_repeat = reference(world, ("b1", "A"), 80)
    for metric in bc.METRICS:
        r = row(
            table(world), ("b1", "A"), metric
        )  # five identical repeats: the mean over repeats is one repeat
        assert r["ci_low"] == pytest.approx(one_repeat[metric][1], abs=1e-9, nan_ok=True)
        assert r["ci_high"] == pytest.approx(one_repeat[metric][2], abs=1e-9, nan_ok=True)


def test_resampling_participants_gives_a_wider_interval_than_resampling_clips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bc, "N_DRAWS", 500)
    world = craft(
        tmp_path,
        monkeypatch,
        n_part=40,
        singleton=False,
        duplicate_clips=True,
        kinds={k: "signal" for k in MODEL_KEYS},
    )
    assert (world["frame"].groupby("participant").size() == 2).all()
    assert run_ci(world) == 0
    r = row(table(world), ("b1", "A"), "auroc")
    participant_width = r["ci_high"] - r["ci_low"]
    clip = reference(world, ("b1", "A"), 500, by="clip")["auroc"]
    clip_width = clip[2] - clip[1]
    assert (
        participant_width > 1.1 * clip_width
    )  # two nearly identical clips per participant: clip resampling is too narrow


def test_draws_where_a_class_is_missing_are_excluded_for_every_model(
    small: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real = bc.bootstrap_draws
    people = small["frame"]["participant"].tolist()
    order = bc.sorted_participants(people)
    y_of = small["frame"].groupby("participant")["y"].max()

    def planted(participants: Any, n_draws: int, seed: int) -> np.ndarray:
        counts = real(participants, 100, seed)
        counts[7] = np.array(
            [0 if y_of[p] == 1 else 1 for p in order]
        )  # no positive clip in this draw
        return counts

    monkeypatch.setattr(bc, "bootstrap_draws", planted)
    assert run_ci(small) == 0  # 1 of 100 is not more than 1%
    t = table(small)
    assert (
        (t["n_excluded"] == 1).all()
        and (t["n_draws_used"] == 99).all()
        and (t["n_boot"] == 100).all()
    )
    assert "excluded 1" in capsys.readouterr().out
    expected = reference(small, ("b1", "A"), 100)  # the reference skips draws that lack a class
    for metric in bc.METRICS:
        r = row(t, ("b1", "A"), metric)
        assert r["ci_low"] == pytest.approx(expected[metric][1], abs=1e-9, nan_ok=True)
        assert r["ci_high"] == pytest.approx(expected[metric][2], abs=1e-9, nan_ok=True)
        assert r["n_draws_finite"] == 99 and expected[metric][3] == 99
    assert (
        t["draws_sha8"].nunique() == 1
    )  # the hash covers all 100 draws, including the excluded one


def test_more_than_one_percent_excluded_is_refused(
    small: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real = bc.bootstrap_draws
    order = bc.sorted_participants(small["frame"]["participant"].tolist())
    y_min = small["frame"].groupby("participant")["y"].min()
    y_max = small["frame"].groupby("participant")["y"].max()
    assert (y_min == 1).any() and (y_max == 0).any()  # both planted draws below are possible

    def planted(participants: Any, n_draws: int, seed: int) -> np.ndarray:
        counts = real(participants, 100, seed)
        counts[3] = np.array(
            [1 if y_max[p] == 0 else 0 for p in order]
        )  # only all-negative participants
        counts[9] = np.array(
            [1 if y_min[p] == 1 else 0 for p in order]
        )  # only all-positive participants
        return counts

    monkeypatch.setattr(bc, "bootstrap_draws", planted)
    assert run_ci(small) == 1
    captured = capsys.readouterr()
    assert (
        "2 of 100 draws lack a class" in captured.err
        and not (small["root"] / "reports_ci").exists()
    )


def test_nan_stays_nan_and_is_never_zero(small: dict[str, Any]) -> None:
    assert run_ci(small) == 0
    r = row(
        table(small), ("b0", "A"), "ppv_at_2pct"
    )  # b0 never predicts positive: PPV is undefined
    assert (
        np.isnan(r["estimate"])
        and np.isnan(r["ci_low"])
        and np.isnan(r["ci_high"])
        and r["n_draws_finite"] == 0
    )
    raw = pd.read_csv(
        small["root"] / "reports_ci" / "ci_table.csv", dtype=str, keep_default_na=False
    )
    cell = raw[(raw["model"] == "b0") & (raw["metric"] == "ppv_at_2pct")].iloc[0]
    assert cell["estimate"] == cell["ci_low"] == cell["ci_high"] == ""
    mcc = row(table(small), ("b0", "A"), "mcc")
    assert (
        mcc["estimate"] == 0.0 and mcc["ci_low"] == 0.0 and mcc["ci_high"] == 0.0
    )  # a defined zero, not a missing value


# --------------------------------------------------------------------------- the output


def test_output_shape_order_columns_and_note(
    small: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_ci(small) == 0
    t = table(small)
    assert list(t.columns) == bc.TABLE_COLUMNS and len(t) == len(MODEL_KEYS) * len(bc.METRICS)
    assert list(zip(t["model"], t["feature_set"], t["metric"], strict=True)) == [
        (m, f, k) for m, f in MODEL_KEYS for k in bc.METRICS
    ]
    assert not [c for c in t.columns if re.search(r"p_value|pval|best|rank|delta", c)]
    assert (
        (t["seed"] == 2026).all()
        and (t["splits_sha8"] == "11111111").all()
        and set(t["features_sha8"]) == {"aaaaaaaa", "bbbbbbbb"}
    )
    assert set(t["inner_sha8"]) == {"22222222"} and set(t["roles_sha8"]) == {"33333333"}
    out = capsys.readouterr().out
    n = small["frame"]["participant"].nunique()
    expected = f"CIs are over {n} participants; 5 repeats are not independent evidence; coverage is approximate with this sample size."
    assert (
        expected in out.splitlines()
        and bc.note(117)
        == "CIs are over 117 participants; 5 repeats are not independent evidence; coverage is approximate with this sample size."
    )
    assert "splits_sha8 11111111" in out and "draws_sha8" in out


def test_rerun_is_byte_identical_atomic_and_only_one_file(
    small: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert run_ci(small) == 0
    folder = small["root"] / "reports_ci"
    first = (folder / "ci_table.csv").read_bytes()
    assert [p.name for p in folder.iterdir()] == ["ci_table.csv"]
    assert run_ci(small) == 0 and (folder / "ci_table.csv").read_bytes() == first

    def broken(src: Any, dst: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(bc.os, "replace", broken)
    with pytest.raises(OSError):
        run_ci(small)
    assert (folder / "ci_table.csv").read_bytes() == first and [
        p.name for p in folder.iterdir()
    ] == ["ci_table.csv"]


def test_no_ids_anywhere_and_the_label_file_is_never_opened(
    small: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    opened: list[str] = []
    real_open = builtins.open

    def spy(file: Any, *args: Any, **kwargs: Any) -> Any:
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(io, "open", spy)
    assert run_ci(small) == 0
    ids = set(small["frame"]["ref"]) | set(small["frame"]["participant"])
    blob = capsys.readouterr().out + (small["root"] / "reports_ci" / "ci_table.csv").read_text()
    assert not [i for i in ids if i in blob]
    assert not [p for p in opened if "fis_diagnostic" in p or "splits.csv" in p]
    assert any("oof_b1_A.csv" in p for p in opened) and not re.search(
        r"20\d\d-\d\d-\d\d|T\d\d:\d\d", blob
    )


def test_source_names_no_label_file_and_imports_neither_gait_nor_the_harness() -> None:
    source = (SRC / "bootstrap_ci.py").read_text()
    assert "fis_diagnostic" not in source and "UPDRS" not in source
    imports = re.findall(r"^\s*(?:from|import)\s+([\w.]+)", source, flags=re.M)
    assert not {"cv_harness", "feature_matrix", "hubu_fis", "gait"} & {
        i.split(".")[0] for i in imports
    }


# --------------------------------------------------------------------------- refusals


def edit_csv(path: Path, change: Any) -> None:
    frame = pd.read_csv(path, dtype={"ref": str}, float_precision="round_trip")
    change(frame).to_csv(path, index=False, float_format="%.17g")


def edit_json(path: Path, **updates: Any) -> None:
    data = json.loads(path.read_text())
    data.update(updates)
    path.write_text(json.dumps(data))


def refusal_cases() -> list[Any]:
    def folder(w: dict[str, Any]) -> Path:
        return w["root"] / w["target"]

    def flip_label_everywhere(
        w: dict[str, Any],
    ) -> None:  # one model sees one clip with another label, consistently
        ref = w["frame"]["ref"].iloc[0]
        edit_csv(
            folder(w) / "oof_b1_A.csv",
            lambda f: f.assign(y=np.where(f["ref"] == ref, 1 - f["y"], f["y"])),
        )
        delta = 1 if w["y"][0] == 0 else -1
        edit_csv(
            folder(w) / "runs_b1_A.csv",
            lambda f: f.assign(n_pos=f["n_pos"] + delta, n_neg=f["n_neg"] - delta),
        )

    def flip_label_one_repeat(w: dict[str, Any]) -> None:
        ref = w["frame"]["ref"].iloc[0]
        edit_csv(
            folder(w) / "oof_b1_A.csv",
            lambda f: f.assign(
                y=np.where((f["ref"] == ref) & (f["repeat"] == 2), 1 - f["y"], f["y"])
            ),
        )

    def features_edit(change: Any) -> Any:
        return lambda w: edit_csv(w["features"], change)

    return [
        (
            "splits_sha8",
            lambda w: edit_json(folder(w) / "meta_lgbm_B.json", splits_sha8="deadbeef"),
            "differs from the other models in splits_sha8",
        ),
        (
            "inner_sha8",
            lambda w: edit_json(folder(w) / "meta_lgbm_B.json", inner_sha8="deadbeef"),
            "differs from the other models in inner_sha8",
        ),
        (
            "roles_sha8",
            lambda w: edit_json(folder(w) / "meta_b0_A.json", roles_sha8="deadbeef"),
            "differs from the other models in roles_sha8",
        ),
        (
            "target_status",
            lambda w: edit_json(folder(w) / "meta_b1_A.json", target_status="settled"),
            "differs from the other models in target_status",
        ),
        (
            "clip_missing",
            lambda w: edit_csv(
                folder(w) / "oof_b1_A.csv", lambda f: f.drop(index=f.index[(f["repeat"] == 2)][0])
            ),
            "does not predict every clip exactly once",
        ),
        (
            "clip_twice",
            lambda w: edit_csv(
                folder(w) / "oof_b1_A.csv", lambda f: pd.concat([f, f.iloc[[0]]], ignore_index=True)
            ),
            "does not predict every clip exactly once",
        ),
        (
            "unknown_clip",
            lambda w: edit_csv(
                folder(w) / "oof_b1_A.csv",
                lambda f: f.assign(ref=np.where(f.index == 0, "deadbeef", f["ref"])),
            ),
            "not in the features file",
        ),
        ("features_clip_missing", features_edit(lambda f: f.iloc[1:]), "not in the features file"),
        (
            "participant_missing",
            features_edit(
                lambda f: f.assign(participant=np.where(f.index == 0, "", f["participant"]))
            ),
            "have no participant",
        ),
        (
            "ref_duplicated_in_features",
            features_edit(lambda f: pd.concat([f, f.iloc[[0]]], ignore_index=True)),
            "ref must be present and unique",
        ),
        (
            "runs_auroc",
            lambda w: edit_csv(
                folder(w) / "runs_lgbm_A.csv", lambda f: f.assign(auroc=f["auroc"] + 0.01)
            ),
            "oof and runs disagree on auroc",
        ),
        (
            "runs_ppv",
            lambda w: edit_csv(
                folder(w) / "runs_lgbm_B.csv", lambda f: f.assign(ppv_at_5pct=0.123)
            ),
            "oof and runs disagree on ppv_at_5pct",
        ),
        (
            "runs_counts",
            lambda w: edit_csv(
                folder(w) / "runs_b0_A.csv", lambda f: f.assign(n_pos=f["n_pos"] + 1)
            ),
            "oof and runs disagree on the label counts",
        ),
        (
            "runs_rows",
            lambda w: edit_csv(
                folder(w) / "runs_b0_A.csv", lambda f: f.assign(repeat=[0, 0, 1, 2, 3])
            ),
            "exactly one row per repeat",
        ),
        ("labels_between_models", flip_label_everywhere, "different labels from the other models"),
        ("labels_between_repeats", flip_label_one_repeat, "labels differ between repeats"),
        (
            "repeats_in_oof",
            lambda w: edit_csv(folder(w) / "oof_b1_A.csv", lambda f: f[f["repeat"] != 4]),
            "does not hold repeats 0..4",
        ),
        (
            "repeats_in_meta",
            lambda w: edit_json(folder(w) / "meta_b1_A.json", repeats=4),
            "meta has 4 repeats",
        ),
        (
            "meta_model",
            lambda w: edit_json(folder(w) / "meta_b1_A.json", model="b0"),
            "another target, model or feature set",
        ),
        (
            "oof_columns",
            lambda w: edit_csv(
                folder(w) / "oof_b1_A.csv", lambda f: f.rename(columns={"proba": "p"})
            ),
            "unexpected columns",
        ),
        (
            "oof_nan",
            lambda w: edit_csv(
                folder(w) / "oof_b1_A.csv",
                lambda f: f.assign(proba=np.where(f.index == 0, np.nan, f["proba"])),
            ),
            "missing label or probability",
        ),
    ]


@pytest.mark.parametrize("name,damage,needle", refusal_cases(), ids=[c[0] for c in refusal_cases()])
def test_refusals_exit_1_with_counts_only_and_write_nothing(
    small: dict[str, Any], capsys: pytest.CaptureFixture[str], name: str, damage: Any, needle: str
) -> None:
    damage(small)
    assert run_ci(small) == 1
    captured = capsys.readouterr()
    assert needle in captured.err and captured.err.startswith("refused:")
    assert (
        not HASH.search(captured.err)
        and not captured.out
        and not (small["root"] / "reports_ci").exists()
    )


@pytest.mark.parametrize("which", ["oof_b1_A.csv", "runs_lgbm_B.csv", "meta_b0_A.json"])
def test_missing_model_files_exit_2(
    small: dict[str, Any], capsys: pytest.CaptureFixture[str], which: str
) -> None:
    (small["root"] / small["target"] / which).unlink()
    assert (
        run_ci(small) == 2
        and "missing input" in capsys.readouterr().err
        and not (small["root"] / "reports_ci").exists()
    )


def test_missing_features_file_exits_2(
    small: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    small["features"].unlink()
    assert run_ci(small) == 2 and "missing input: features file" in capsys.readouterr().err


def test_paths_outside_data_processed_cv_exit_2_and_write_nothing(
    small: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    outside = tmp_path / "elsewhere"
    assert run_ci(small, "--out-dir", str(outside)) == 2
    assert (
        "--out-dir must be inside data/processed/cv" in capsys.readouterr().err
        and not outside.exists()
    )
    shutil.copytree(small["root"], outside)
    assert bc.main(["--cv-root", str(outside), "--features", str(small["features"])]) == 2
    assert (
        "--cv-root must be inside data/processed/cv" in capsys.readouterr().err
        and not (outside / "reports_ci").exists()
    )


def test_the_parser_rejects_unknown_targets(small: dict[str, Any]) -> None:
    with pytest.raises(SystemExit) as err:
        run_ci(small, "--targets", "ge9")
    assert err.value.code == 2


# --------------------------------------------------------------------------- the real harness output


@pytest.fixture(scope="module")
def official(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    patch = pytest.MonkeyPatch()
    patch.setitem(ch.GRIDS, "lgbm", [dict(p) for p in SMALL_LGBM])
    w = build_world(
        tmp_path_factory.mktemp("official"), patch, make_data("noise", seed=4), repeats=5
    )
    proc = w["proc"]
    assert cvs.main(["--features", str(proc / "features.csv"), "--roles", str(proc / "feature_roles.csv"),
                     "--meta", str(proc / "features_meta.json"), "--target", "ge1", "--repeats", "5",
                     "--out-dir", str(w["cv"])]) == 0  # fmt: skip
    for target in ("ge2", "ge1"):
        for model, fs in bc.MODELS:
            assert (
                harness(
                    w,
                    "run",
                    "--model",
                    model,
                    "--feature-set",
                    fs,
                    "--target",
                    target,
                    "--repeats",
                    "5",
                )
                == 0
            )
    yield w
    patch.undo()


def test_real_harness_output_reads_back_and_the_estimates_equal_the_runs(
    official: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    copy = tmp_path / "data" / "processed" / "cv"
    shutil.copytree(official["cv"], copy)
    monkeypatch.setattr(cvs, "CV_ROOT", copy)
    monkeypatch.setattr(bc, "N_DRAWS", 100)
    features = official["proc"] / "features.csv"
    assert bc.main(["--cv-root", str(copy), "--features", str(features)]) == 0
    t = pd.read_csv(copy / "reports_ci" / "ci_table.csv", float_precision="round_trip")
    assert (
        len(t) == 2 * 4 * 8
        and list(t["target"].unique()) == ["ge1", "ge2"]
        and (t["n_excluded"] == 0).all()
    )
    for (target, model, fs), part in t.groupby(["target", "model", "feature_set"]):
        runs = pd.read_csv(copy / target / f"runs_{model}_{fs}.csv", float_precision="round_trip")
        for metric in bc.METRICS:
            est = part[part["metric"] == metric]["estimate"].iloc[0]
            assert est == pytest.approx(runs[metric].mean(), abs=1e-9, nan_ok=True)
            r = part[part["metric"] == metric].iloc[0]
            if np.isfinite(r["estimate"]) and r["n_draws_finite"]:
                assert r["ci_low"] <= r["ci_high"]
    ids = set(pd.read_csv(features, dtype=str, usecols=["ref", "participant"]).stack())
    blob = capsys.readouterr().out + (copy / "reports_ci" / "ci_table.csv").read_text()
    assert not [i for i in ids if i in blob]
    first = (copy / "reports_ci" / "ci_table.csv").read_bytes()
    assert bc.main(["--cv-root", str(copy), "--features", str(features)]) == 0
    assert (copy / "reports_ci" / "ci_table.csv").read_bytes() == first
