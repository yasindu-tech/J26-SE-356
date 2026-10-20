"""Tests for models/tapping/src/auroc_difference.py (task FT-39). A small hand-built world with a
planted improvement, an explicit bootstrap and an O(n^2) DeLong reference. No real data is used."""

from __future__ import annotations

import builtins
import io
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import auroc_difference as ad  # noqa: E402
import bootstrap_ci as bc  # noqa: E402
import cv_harness as ch  # noqa: E402
import cv_splits as cvs  # noqa: E402
from test_cv_splits import hid  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src"
HASH = re.compile(r"\b[0-9a-f]{8}\b")
A, B, R = ("lgbm", "A"), ("lgbm", "B"), ("b1", "A")


# --------------------------------------------------------------------------- a hand-built world


def craft(
    tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
    targets: tuple[str, ...] = ("ge2",),
    n_part: int = 20,
    mode: str = "improve",
    seed: int = 0,
) -> dict[str, Any]:
    """mode improve: lgbm B = lgbm A plus a planted gain; identical: lgbm B is a copy of lgbm A."""
    rng = np.random.default_rng(seed)
    cv_root = tmp / "data" / "processed" / "cv"
    monkeypatch.setattr(cvs, "CV_ROOT", cv_root)
    z = rng.permutation(np.arange(n_part) % 2)
    rows = []
    for i in range(n_part):
        for c in range(1 if i == n_part - 1 else 2):
            rows.append(
                (
                    hid("dc", 100 * i + c),
                    hid("dp", i),
                    int(z[i]) if c == 0 else int(rng.random() < 0.3 + 0.4 * z[i]),
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
    y, n = frame["y"].to_numpy(), len(frame)
    world: dict[str, Any] = {
        "root": cv_root,
        "features": features,
        "frame": frame,
        "y": y,
        "targets": targets,
        "probas": {},
    }
    for target in targets:
        folder = cv_root / target
        folder.mkdir(parents=True)
        pa, pb, pc = np.zeros((5, n)), np.zeros((5, n)), np.zeros((5, n))
        for r in range(5):
            ua, ub, uc = rng.uniform(size=(3, n))
            pa[r] = np.round(np.clip(0.30 * y + 0.70 * ua, 0, 1), 2)
            pb[r] = (
                pa[r]
                if mode == "identical"
                else np.round(np.clip(pa[r] + 0.12 * (y - 0.5) + 0.03 * (ub - 0.5), 0, 1), 2)
            )
            pc[r] = np.round(np.clip(0.10 * y + 0.90 * uc, 0, 1), 2)
        world["probas"][target] = {A: pa, B: pb, R: pc}
        for (model, fs), p in world["probas"][target].items():
            tag = f"{model}_{fs}"
            oof = pd.concat(
                [
                    pd.DataFrame(
                        {"ref": frame["ref"], "repeat": r, "outer_fold": 0, "y": y, "proba": p[r]}
                    )
                    for r in range(5)
                ],
                ignore_index=True,
            )
            oof.to_csv(folder / f"oof_{tag}.csv", index=False, float_format="%.17g")
            runs = pd.DataFrame(
                [
                    {
                        "repeat": r,
                        "n_clips": n,
                        "n_pos": int(y.sum()),
                        "n_neg": int((1 - y).sum()),
                        "auroc": ch.auroc(y, p[r]),
                    }
                    for r in range(5)
                ]
            )
            runs.to_csv(folder / f"runs_{tag}.csv", index=False, float_format="%.17g")
            meta = {
                "target": target,
                "model": model,
                "feature_set": fs,
                "repeats": 5,
                "splits_sha8": "11111111",
                "inner_sha8": "22222222",
                "roles_sha8": "33333333",
                "target_status": "pending_default_ge2",
                "features_sha8": "bbbbbbbb" if (model, fs) == B else "aaaaaaaa",
            }
            (folder / f"meta_{tag}.json").write_text(json.dumps(meta))
    return world


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(bc, "N_DRAWS", 60)
    return craft(tmp_path, monkeypatch)


def run_ad(w: dict[str, Any], *extra: str, features: Path | None = None) -> int:
    targets = [] if "--targets" in extra else ["--targets", *w["targets"]]
    return ad.main(
        [
            "--cv-root",
            str(w["root"]),
            "--features",
            str(features or w["features"]),
            *targets,
            *extra,
        ]
    )


def table(w: dict[str, Any]) -> pd.DataFrame:
    text = {
        c: str
        for c in (
            "draws_sha8",
            "splits_sha8",
            "inner_sha8",
            "roles_sha8",
            "features_sha8_minuend",
            "features_sha8_subtrahend",
        )
    }
    return pd.read_csv(
        w["root"] / "reports_ci" / "auroc_difference.csv", float_precision="round_trip", dtype=text
    )


def pick(t: pd.DataFrame, target: str, comparison: int, method: str) -> pd.Series:
    return t[
        (t["target"] == target)
        & (t["comparison"] == ad.comparison_name(comparison))
        & (t["method"] == method)
    ].iloc[0]


# --------------------------------------------------------------------------- independent references


def ref_draws(
    w: dict[str, Any],
    minuend: tuple[str, str],
    subtrahend: tuple[str, str],
    target: str = "ge2",
    counts: np.ndarray | None = None,
    counts_sub: np.ndarray | None = None,
) -> np.ndarray:
    """Explicit bootstrap: build every resampled dataset and average the per-repeat AUROC differences."""
    frame, y = w["frame"], w["y"]
    order = bc.sorted_participants(frame["participant"].tolist())
    pidx = np.array([order.index(q) for q in frame["participant"]])
    counts = (
        bc.bootstrap_draws(frame["participant"].tolist(), bc.N_DRAWS, bc.SEED)
        if counts is None
        else counts
    )
    counts_sub = counts if counts_sub is None else counts_sub
    out = []
    for d in range(len(counts)):
        idx, idx_sub = (
            np.repeat(np.arange(len(y)), counts[d][pidx]),
            np.repeat(np.arange(len(y)), counts_sub[d][pidx]),
        )
        if min(y[idx].sum(), (1 - y[idx]).sum(), y[idx_sub].sum(), (1 - y[idx_sub]).sum()) == 0:
            continue
        pm, ps = w["probas"][target][minuend], w["probas"][target][subtrahend]
        out.append(
            np.mean(
                [
                    ch.auroc(y[idx], pm[r][idx]) - ch.auroc(y[idx_sub], ps[r][idx_sub])
                    for r in range(5)
                ]
            )
        )
    return np.array(out)


def ref_p(d: np.ndarray) -> float:
    b = len(d)
    return min(1.0, 2 * min((1 + np.sum(d <= 0)) / (b + 1), (1 + np.sum(d >= 0)) / (b + 1)))


def ref_delong(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> dict[str, float]:
    """O(n^2) DeLong with explicit loops and explicit sample covariances."""
    pos = [i for i, v in enumerate(y) if v == 1]
    neg = [i for i, v in enumerate(y) if v == 0]
    m, n = len(pos), len(neg)

    def psi(a: float, b: float) -> float:
        return 1.0 if a > b else (0.5 if a == b else 0.0)

    v10 = [[sum(psi(p[i], p[j]) for j in neg) / n for i in pos] for p in (p1, p2)]
    v01 = [[sum(psi(p[i], p[j]) for i in pos) / m for j in neg] for p in (p1, p2)]
    auc = [sum(v) / m for v in v10]

    def cov(a: list[float], b: list[float]) -> float:
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        return sum((x - ma) * (w - mb) for x, w in zip(a, b, strict=True)) / (len(a) - 1)

    var = (cov(v10[0], v10[0]) + cov(v10[1], v10[1]) - 2 * cov(v10[0], v10[1])) / m + (
        cov(v01[0], v01[0]) + cov(v01[1], v01[1]) - 2 * cov(v01[0], v01[1])
    ) / n
    diff = auc[0] - auc[1]
    from math import erfc, sqrt

    z = diff / sqrt(var) if var > 0 else float("nan")
    return {
        "diff": diff,
        "var": var,
        "z": z,
        "p": erfc(abs(z) / sqrt(2)) if var > 0 else float("nan"),
    }


# --------------------------------------------------------------------------- design


def test_the_design_is_pinned_and_the_constants_come_from_bootstrap_ci() -> None:
    assert ad.MODELS_USED == (("b1", "A"), ("lgbm", "A"), ("lgbm", "B"))
    assert ad.COMPARISONS == ((B, A), (A, R)) and ad.METHODS == ("participant_bootstrap", "delong")
    assert ad.CONFIRMATORY == ("ge2", 0, "participant_bootstrap")
    assert (bc.SEED, bc.N_DRAWS, bc.CI_PERCENTILES) == (2026, 2000, (2.5, 97.5))
    assert (
        ad.comparison_name(0) == "lgbm B minus lgbm A"
        and ad.comparison_name(1) == "lgbm A minus b1 A"
    )
    source = (SRC / "auroc_difference.py").read_text()
    assert "bc.N_DRAWS" in source and "bc.SEED" in source  # imported, not copied
    assert not re.search(r"^(SEED|N_DRAWS|CI_PERCENTILES)\s*=", source, flags=re.M)


def test_bootstrap_rows_match_an_explicit_bootstrap(world: dict[str, Any]) -> None:
    assert run_ad(world) == 0
    t = table(world)
    for i, (minuend, subtrahend) in enumerate(ad.COMPARISONS):
        d = ref_draws(world, minuend, subtrahend)
        r = pick(t, "ge2", i, "participant_bootstrap")
        lo, hi = np.percentile(d, [2.5, 97.5])
        assert r["ci_low"] == pytest.approx(lo, abs=1e-9) and r["ci_high"] == pytest.approx(
            hi, abs=1e-9
        )
        assert r["p"] == pytest.approx(ref_p(d), abs=1e-9) and r["n_draws_used"] == len(d) == 60
        assert r["n_boot"] == bc.N_DRAWS == 60 and r["seed"] == bc.SEED


def test_the_point_estimate_equals_the_difference_of_the_runs_means(world: dict[str, Any]) -> None:
    assert run_ad(world) == 0
    t = table(world)
    runs = {
        k: pd.read_csv(
            world["root"] / "ge2" / f"runs_{k[0]}_{k[1]}.csv", float_precision="round_trip"
        )["auroc"].mean()
        for k in (A, B, R)
    }
    for i, (minuend, subtrahend) in enumerate(ad.COMPARISONS):
        expected = runs[minuend] - runs[subtrahend]
        for method in ad.METHODS:
            assert pick(t, "ge2", i, method)["estimate"] == pytest.approx(expected, abs=1e-9)


def test_a_planted_gain_is_found_with_a_positive_difference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bc, "N_DRAWS", 300)
    w = craft(tmp_path, monkeypatch, n_part=40)
    assert run_ad(w) == 0
    r = pick(table(w), "ge2", 0, "participant_bootstrap")
    assert r["estimate"] > 0 and r["ci_low"] > 0 and r["p"] < 0.05
    assert (
        pick(table(w), "ge2", 1, "participant_bootstrap")["estimate"] > 0
    )  # lgbm A is better than the weak b1 A


def test_identical_models_give_zero_and_p_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bc, "N_DRAWS", 60)
    w = craft(tmp_path, monkeypatch, mode="identical")
    assert run_ad(w) == 0
    t = table(w)
    boot, delong = pick(t, "ge2", 0, "participant_bootstrap"), pick(t, "ge2", 0, "delong")
    assert (
        boot["estimate"] == 0 and boot["ci_low"] == 0 and boot["ci_high"] == 0 and boot["p"] == 1.0
    )
    assert (
        delong["estimate"] == 0
        and delong["p"] == 1.0
        and delong["p_min"] == 1.0
        and delong["p_max"] == 1.0
    )


def test_swapping_the_order_flips_the_sign_and_the_interval(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert run_ad(world) == 0
    forward = pick(table(world), "ge2", 0, "participant_bootstrap")
    forward_delong = pick(table(world), "ge2", 0, "delong")
    monkeypatch.setattr(ad, "COMPARISONS", ((A, B), (R, A)))
    assert run_ad(world) == 0
    t = table(world)
    swapped = pick(t, "ge2", 0, "participant_bootstrap")  # the name is now "lgbm A minus lgbm B"
    assert swapped["comparison"] == "lgbm A minus lgbm B"
    assert (
        swapped["estimate"] == pytest.approx(-forward["estimate"])
        and swapped["ci_low"] == pytest.approx(-forward["ci_high"])
        and swapped["ci_high"] == pytest.approx(-forward["ci_low"])
    )
    assert swapped["p"] == pytest.approx(forward["p"])
    d = pick(t, "ge2", 0, "delong")
    assert d["estimate"] == pytest.approx(-forward_delong["estimate"]) and d["p"] == pytest.approx(
        forward_delong["p"]
    )


# --------------------------------------------------------------------------- pairing


def test_both_models_use_the_same_draws_and_pairing_narrows_the_interval(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Any, ...]] = []
    real = bc.bootstrap_draws

    def counting(*a: Any, **k: Any) -> np.ndarray:
        calls.append(a)
        return real(*a, **k)

    monkeypatch.setattr(bc, "bootstrap_draws", counting)
    assert run_ad(world) == 0
    assert len(calls) == 1 and calls[0][1:] == (60, 2026)  # one draw set, the FT-38 settings
    paired = pick(table(world), "ge2", 0, "participant_bootstrap")
    people = world["frame"]["participant"].tolist()
    other = real(people, 60, 2027)  # an independent resample for the subtrahend
    unpaired = ref_draws(world, B, A, counts=real(people, 60, 2026), counts_sub=other)
    lo, hi = np.percentile(unpaired, [2.5, 97.5])
    assert (paired["ci_high"] - paired["ci_low"]) < 0.6 * (
        hi - lo
    )  # unpaired draws would be much wider
    assert paired["draws_sha8"] == bc.draws_sha8(real(people, 60, 2026))


# --------------------------------------------------------------------------- DeLong


def test_delong_matches_the_o_n_squared_reference() -> None:
    rng = np.random.default_rng(3)
    for _ in range(4):
        y = rng.integers(0, 2, 24)
        y[:3], y[3:6] = 1, 0
        p1, p2 = (
            np.round(rng.uniform(size=24), 1),
            np.round(0.5 * rng.uniform(size=24) + 0.4 * y * rng.uniform(size=24), 1),
        )
        got, ref = ad.delong_test(y, p1, p2), ref_delong(y, p1, p2)
        for key in ("diff", "var", "z", "p"):
            assert got[key] == pytest.approx(ref[key], rel=1e-9, abs=1e-12)


def test_delong_components_are_the_structural_components() -> None:
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.4, 0.4, 0.2, 0.6])
    v10, v01 = ad.delong_components(y, p)
    assert v10.tolist() == pytest.approx(
        [1.0, (0.5 + 1 + 0) / 3]
    ) and v01.tolist() == pytest.approx([(1 + 0.5) / 2, 1.0, 0.5])
    assert v10.mean() == pytest.approx(v01.mean()) == pytest.approx(roc_auc_score(y, p))


def test_delong_invariants() -> None:
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 40)
    y[:4], y[4:8] = 1, 0
    p1, p2 = rng.uniform(size=40), np.clip(0.5 * rng.uniform(size=40) + 0.3 * y, 0, 1)
    base = ad.delong_test(y, p1, p2)
    assert base["auc_1"] == pytest.approx(roc_auc_score(y, p1)) and base["auc_2"] == pytest.approx(
        roc_auc_score(y, p2)
    )
    same = ad.delong_test(y, p1, p1)
    assert same["diff"] == 0 and same["var"] == pytest.approx(0, abs=1e-12) and same["p"] == 1.0
    swapped = ad.delong_test(y, p2, p1)
    assert (
        swapped["diff"] == pytest.approx(-base["diff"])
        and swapped["var"] == pytest.approx(base["var"])
        and swapped["p"] == pytest.approx(base["p"])
    )
    monotone = ad.delong_test(
        y, p1, np.exp(5 * p2) + 3
    )  # a rank-preserving transform changes nothing
    assert monotone["diff"] == pytest.approx(base["diff"]) and monotone["p"] == pytest.approx(
        base["p"]
    )
    assert 0 <= base["p"] <= 1 and base["z"] == pytest.approx(base["diff"] / np.sqrt(base["var"]))
    with pytest.raises(cvs.GuardFailure):
        ad.delong_test(np.array([1, 0, 0, 0]), np.arange(4.0), np.arange(4.0))


def test_the_covariance_between_the_models_matters() -> None:
    rng = np.random.default_rng(8)
    y = rng.integers(0, 2, 60)
    y[:5], y[5:10] = 1, 0
    p1 = rng.uniform(size=60) + 0.5 * y
    p2 = p1 + 0.05 * rng.uniform(size=60)  # strongly correlated with p1
    c1, c2 = ad.delong_components(y, p1), ad.delong_components(y, p2)
    m, n = int(y.sum()), int((1 - y).sum())
    independent = (np.var(c1[0], ddof=1) + np.var(c2[0], ddof=1)) / m + (
        np.var(c1[1], ddof=1) + np.var(c2[1], ddof=1)
    ) / n
    assert (
        ad.delong_test(y, p1, p2)["var"] < 0.5 * independent
    )  # dropping the covariance term would overstate the variance


def test_the_delong_row_summarises_the_five_repeats(world: dict[str, Any]) -> None:
    assert run_ad(world) == 0
    r = pick(table(world), "ge2", 0, "delong")
    per_repeat = [
        ref_delong(world["y"], world["probas"]["ge2"][B][k], world["probas"]["ge2"][A][k])
        for k in range(5)
    ]
    assert r["estimate"] == pytest.approx(np.mean([x["diff"] for x in per_repeat]), abs=1e-9)
    ps = [x["p"] for x in per_repeat]
    assert (
        r["p"] == pytest.approx(np.median(ps))
        and r["p_min"] == pytest.approx(min(ps))
        and r["p_max"] == pytest.approx(max(ps))
    )
    assert (
        np.isnan(r["ci_low"]) and np.isnan(r["ci_high"]) and np.isnan(r["n_draws_used"])
    )  # no CI is claimed for DeLong


# --------------------------------------------------------------------------- p value and exclusion


def test_a_delong_result_that_disagrees_with_the_bootstrap_point_estimate_is_refused(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real = ad.delong_test
    monkeypatch.setattr(
        ad, "delong_test", lambda y, a, b: {**real(y, a, b), "diff": real(y, a, b)["diff"] + 0.01}
    )
    assert run_ad(world) == 1
    assert (
        "DeLong and bootstrap point estimates disagree" in capsys.readouterr().err
        and not (world["root"] / "reports_ci").exists()
    )


def test_the_bootstrap_p_formula() -> None:
    assert ad.bootstrap_p(np.array([0.0] * 10)) == 1.0
    assert ad.bootstrap_p(np.arange(1.0, 100.0)) == pytest.approx(2 * (1 + 0) / 100)  # no draw <= 0
    assert ad.bootstrap_p(-np.arange(1.0, 100.0)) == pytest.approx(2 / 100)
    d = np.array([-1.0, 0.0, 1.0, 2.0])
    assert ad.bootstrap_p(d) == pytest.approx(min(1, 2 * min((1 + 2) / 5, (1 + 3) / 5)))
    assert ad.bootstrap_p(np.array([1.0, -1.0] * 50)) == 1.0  # never above 1


def test_excluded_draws_are_removed_for_both_models_and_counted(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real = bc.bootstrap_draws
    order = bc.sorted_participants(world["frame"]["participant"].tolist())
    y_max = world["frame"].groupby("participant")["y"].max()

    def planted(participants: Any, n_draws: int, seed: int) -> np.ndarray:
        counts = real(participants, 100, seed)
        counts[7] = np.array([1 if y_max[p] == 0 else 0 for p in order])  # no positive clip
        return counts

    monkeypatch.setattr(bc, "bootstrap_draws", planted)
    shapes: list[int] = []
    real_auroc = bc.weighted_auroc

    def recording(y: np.ndarray, proba: np.ndarray, weights: np.ndarray) -> np.ndarray:
        shapes.append(weights.shape[0])
        return real_auroc(y, proba, weights)

    monkeypatch.setattr(bc, "weighted_auroc", recording)
    assert run_ad(world) == 0  # 1 of 100 is not more than 1%
    assert set(shapes) == {
        99,
        1,
    }  # the excluded draw never reaches the metric (99 draws, plus the unit-weight point)
    t = table(world)
    boot = t[t["method"] == "participant_bootstrap"]
    assert (
        (boot["n_excluded"] == 1).all()
        and (boot["n_draws_used"] == 99).all()
        and "excluded 1" in capsys.readouterr().out
    )
    d = ref_draws(world, B, A, counts=planted(world["frame"]["participant"].tolist(), 100, 2026))
    r = pick(t, "ge2", 0, "participant_bootstrap")
    assert (
        len(d) == 99
        and r["ci_low"] == pytest.approx(np.percentile(d, 2.5), abs=1e-9)
        and r["p"] == pytest.approx(ref_p(d))
    )


def test_more_than_one_percent_excluded_is_refused(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real = bc.bootstrap_draws
    order = bc.sorted_participants(world["frame"]["participant"].tolist())
    y_max = world["frame"].groupby("participant")["y"].max()

    def planted(participants: Any, n_draws: int, seed: int) -> np.ndarray:
        counts = real(participants, 100, seed)
        for k in (3, 9):
            counts[k] = np.array([1 if y_max[p] == 0 else 0 for p in order])
        return counts

    monkeypatch.setattr(bc, "bootstrap_draws", planted)
    assert run_ad(world) == 1
    captured = capsys.readouterr()
    assert (
        "2 of 100 draws lack a class" in captured.err
        and not (world["root"] / "reports_ci").exists()
    )


# --------------------------------------------------------------------------- output


def test_roles_order_columns_and_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(bc, "N_DRAWS", 40)
    w = craft(tmp_path, monkeypatch, targets=("ge2", "ge1"))
    assert run_ad(w) == 0
    t = table(w)
    assert list(t.columns) == ad.TABLE_COLUMNS and len(t) == 8
    assert list(zip(t["target"], t["comparison"], t["method"], strict=True)) == [
        (tg, ad.comparison_name(i), m) for tg in ("ge1", "ge2") for i in (0, 1) for m in ad.METHODS
    ]
    assert t[t["role"] == "confirmatory"][["target", "comparison", "method"]].values.tolist() == [
        ["ge2", "lgbm B minus lgbm A", "participant_bootstrap"]
    ]
    assert not [c for c in t.columns if re.search(r"best|rank|delta", c)]
    out = capsys.readouterr().out
    n = w["frame"]["participant"].nunique()
    assert (
        ad.note(n)
        == f"Differences are over {n} participants; 5 repeats are not independent evidence; only lgbm B minus lgbm A on ge2 is confirmatory; the rest are descriptive."
    )
    assert (
        ad.note(117)
        == "Differences are over 117 participants; 5 repeats are not independent evidence; only lgbm B minus lgbm A on ge2 is confirmatory; the rest are descriptive."
    )
    assert ad.note(n) in out.splitlines()
    assert (
        "anti-conservative" in out
        and f"{n - 1} of {n} participants contribute two clips" in out
        and "no multiplicity adjustment" in out
    )
    assert (
        "approximate and descriptive" in out
        and "splits_sha8 11111111" in out
        and "draws_sha8" in out
    )


def test_rerun_is_byte_identical_atomic_and_one_file(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert run_ad(world) == 0
    folder = world["root"] / "reports_ci"
    first = (folder / "auroc_difference.csv").read_bytes()
    assert [p.name for p in folder.iterdir()] == ["auroc_difference.csv"]
    assert run_ad(world) == 0 and (folder / "auroc_difference.csv").read_bytes() == first

    def broken(src: Any, dst: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(ad.os, "replace", broken)
    with pytest.raises(OSError):
        run_ad(world)
    assert (folder / "auroc_difference.csv").read_bytes() == first and [
        p.name for p in folder.iterdir()
    ] == ["auroc_difference.csv"]


def test_no_ids_anywhere_and_no_label_file_is_opened(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    opened: list[str] = []
    real_open = builtins.open

    def spy(file: Any, *args: Any, **kwargs: Any) -> Any:
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(io, "open", spy)
    assert run_ad(world) == 0
    ids = set(world["frame"]["ref"]) | set(world["frame"]["participant"])
    blob = (
        capsys.readouterr().out
        + (world["root"] / "reports_ci" / "auroc_difference.csv").read_text()
    )
    assert not [i for i in ids if i in blob] and not [
        p for p in opened if "fis_diagnostic" in p or "splits.csv" in p
    ]
    assert not re.search(r"20\d\d-\d\d-\d\d|T\d\d:\d\d", blob)


def test_source_names_no_label_file_and_imports_neither_gait_nor_the_harness() -> None:
    source = (SRC / "auroc_difference.py").read_text()
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
    def f(w: dict[str, Any]) -> Path:
        return w["root"] / "ge2"

    def flip_label_in_one_model(w: dict[str, Any]) -> None:
        ref = w["frame"]["ref"].iloc[0]
        edit_csv(
            f(w) / "oof_lgbm_A.csv",
            lambda d: d.assign(y=np.where(d["ref"] == ref, 1 - d["y"], d["y"])),
        )
        delta = 1 if w["y"][0] == 0 else -1
        edit_csv(
            f(w) / "runs_lgbm_A.csv",
            lambda d: d.assign(n_pos=d["n_pos"] + delta, n_neg=d["n_neg"] - delta),
        )

    return [
        (
            "splits_sha8",
            lambda w: edit_json(f(w) / "meta_lgbm_B.json", splits_sha8="deadbeef"),
            "differs from the other models in splits_sha8",
        ),
        (
            "inner_sha8",
            lambda w: edit_json(f(w) / "meta_b1_A.json", inner_sha8="deadbeef"),
            "differs from the other models in inner_sha8",
        ),
        (
            "roles_sha8",
            lambda w: edit_json(f(w) / "meta_lgbm_A.json", roles_sha8="deadbeef"),
            "differs from the other models in roles_sha8",
        ),
        (
            "target_status",
            lambda w: edit_json(f(w) / "meta_lgbm_B.json", target_status="settled"),
            "differs from the other models in target_status",
        ),
        (
            "labels_between_models",
            flip_label_in_one_model,
            "different labels from the other models",
        ),
        (
            "runs_auroc_minuend",
            lambda w: edit_csv(
                f(w) / "runs_lgbm_B.csv", lambda d: d.assign(auroc=d["auroc"] + 0.01)
            ),
            "lgbm B: oof and runs disagree on auroc",
        ),
        (
            "runs_auroc_subtrahend",
            lambda w: edit_csv(
                f(w) / "runs_lgbm_A.csv", lambda d: d.assign(auroc=d["auroc"] - 0.01)
            ),
            "lgbm A: oof and runs disagree on auroc",
        ),
        (
            "runs_auroc_b1",
            lambda w: edit_csv(f(w) / "runs_b1_A.csv", lambda d: d.assign(auroc=0.5)),
            "b1 A: oof and runs disagree on auroc",
        ),
        (
            "runs_counts",
            lambda w: edit_csv(f(w) / "runs_b1_A.csv", lambda d: d.assign(n_pos=d["n_pos"] + 1)),
            "oof and runs disagree on the label counts",
        ),
        (
            "clip_missing",
            lambda w: edit_csv(
                f(w) / "oof_lgbm_B.csv", lambda d: d.drop(index=d.index[d["repeat"] == 3][0])
            ),
            "does not predict every clip exactly once",
        ),
        (
            "repeats_in_meta",
            lambda w: edit_json(f(w) / "meta_lgbm_A.json", repeats=4),
            "meta has 4 repeats",
        ),
        (
            "meta_model",
            lambda w: edit_json(f(w) / "meta_lgbm_B.json", model="b1"),
            "another target, model or feature set",
        ),
        (
            "participant_missing",
            lambda w: edit_csv(
                w["features"],
                lambda d: d.assign(participant=np.where(d.index == 0, "", d["participant"])),
            ),
            "have no participant",
        ),
    ]


@pytest.mark.parametrize("name,damage,needle", refusal_cases(), ids=[c[0] for c in refusal_cases()])
def test_refusals_exit_1_with_counts_only_and_write_nothing(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str], name: str, damage: Any, needle: str
) -> None:
    damage(world)
    assert run_ad(world) == 1
    captured = capsys.readouterr()
    assert needle in captured.err and captured.err.startswith("refused:")
    assert (
        not HASH.search(captured.err)
        and not captured.out
        and not (world["root"] / "reports_ci").exists()
    )


@pytest.mark.parametrize("which", ["oof_lgbm_B.csv", "runs_b1_A.csv", "meta_lgbm_A.json"])
def test_missing_model_files_exit_2(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str], which: str
) -> None:
    (world["root"] / "ge2" / which).unlink()
    assert (
        run_ad(world) == 2
        and "missing input" in capsys.readouterr().err
        and not (world["root"] / "reports_ci").exists()
    )


def test_missing_features_file_exits_2(
    world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    world["features"].unlink()
    assert run_ad(world) == 2 and "missing input: features file" in capsys.readouterr().err


def test_paths_outside_data_processed_cv_exit_2_and_write_nothing(
    world: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    outside = tmp_path / "elsewhere"
    assert run_ad(world, "--out-dir", str(outside)) == 2
    assert (
        "--out-dir must be inside data/processed/cv" in capsys.readouterr().err
        and not outside.exists()
    )
    shutil.copytree(world["root"], outside)
    assert (
        ad.main(
            ["--cv-root", str(outside), "--features", str(world["features"]), "--targets", "ge2"]
        )
        == 2
    )
    assert (
        "--cv-root must be inside data/processed/cv" in capsys.readouterr().err
        and not (outside / "reports_ci").exists()
    )


def test_the_parser_rejects_unknown_targets(world: dict[str, Any]) -> None:
    with pytest.raises(SystemExit) as err:
        run_ad(world, "--targets", "ge9")
    assert err.value.code == 2
