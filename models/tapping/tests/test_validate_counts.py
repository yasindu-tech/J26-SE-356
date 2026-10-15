"""Tests for models/tapping/src/validate_counts.py (task FT-25).

Synthetic data only: made-up clips (empty video files, tiny signals, detector peaks 20 frames apart)
under temporary roots (mc.INTERIM_ROOT, mc.DOCS_ROOT and mc.VIDEO_ROOT are patched). No real
count, label or clip ID is used.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hubu_fis  # noqa: E402
import manual_counts as mc  # noqa: E402
import validate_counts as vc  # noqa: E402

CELL_N = dict(zip(mc.CELL_ORDER, (62, 12, 65, 10, 52, 13), strict=True))  # arbitrary weights
ERRORS = [0] * 40  # detector - manual per primary clip, in primary order
for _i, _e in {0: 1, 2: -2, 10: -1, 20: 2, 33: 1, 39: -1}.items():
    ERRORS[_i] = _e
RAW_ID = re.compile(r"(?i)(control|id)[\s_-]?\d+|dcha|izda|updrs")
HEX8 = re.compile(r"[0-9a-f]{8}")


def stem(n: int, hand: str = "DCHA") -> str:
    return f"{'CONTROL' if n % 2 == 0 else 'ID'}{n:02d}_{hand}"


def cell_of(i: int) -> str:
    edges = np.cumsum([mc.ALLOCATION[c] for c in mc.CELL_ORDER])
    return mc.CELL_ORDER[int(np.searchsorted(edges, i, side="right"))]


class World:
    """233 retained clips (made-up rates and flags), 19 of them seen (3 practice, 16 hard from 13
    participants). The 40 primary clips are drawn per cell from the other 214; the export weights
    are the frame sizes counted here in numpy, independently of manual_counts."""

    def __init__(self, root: Path) -> None:
        self.interim, self.docs, self.videos = root / "interim", root / "docs", root / "raw" / "v"
        for d in (self.interim / "smoothed", self.interim / "taps", self.docs, self.videos):
            d.mkdir(parents=True)
        (self.videos.parent / "fis_diagnostic.csv").write_text("ID,UPDRS\nCONTROL00_DCHA,0\n")
        rng = np.random.default_rng(11)
        stems = {f"f{k}": stem(200 + k) for k in range(214)}
        stems |= {f"h{i}": stem(100 + i) for i in range(13)}
        stems |= {f"h{13 + i}": stem(100 + i, "IZDA") for i in range(3)}  # 3 participants twice
        stems |= {f"x{i}": stem(150 + i) for i in range(3)}  # practice
        for s in stems.values():
            (self.videos / f"{s}.mp4").write_bytes(b"")
        ref_of = {key: hubu_fis.pseudonym(s) for key, s in stems.items()}
        self.hard = [ref_of[f"h{i}"] for i in range(16)]
        self.practice = [ref_of[f"x{i}"] for i in range(3)]
        rate = {r: float(rng.uniform(0.4, 3.8)) for r in ref_of.values()}
        flagged = {r: bool(rng.random() < 0.22) for r in ref_of.values()}
        lo, hi = np.quantile(list(rate.values()), [1 / 3, 2 / 3])
        cell = {
            r: ("slow" if rate[r] <= lo else "mid" if rate[r] <= hi else "fast")
            + ("/flagged" if flagged[r] else "/unflagged")
            for r in rate
        }
        seen = set(self.hard) | set(self.practice)
        frame = sorted(r for r in rate if r not in seen)
        self.frame_refs = frame
        self.cell_n = {c: sum(cell[r] == c for r in frame) for c in mc.CELL_ORDER}
        assert sum(self.cell_n.values()) == 214 and min(self.cell_n.values()) > 9
        self.primary = [
            r for c in mc.CELL_ORDER for r in [x for x in frame if cell[x] == c][: mc.ALLOCATION[c]]
        ]
        self.clips: dict[str, dict[str, Any]] = {}
        for k, ref in enumerate(self.primary + self.hard):
            first, n_taps = k % 3, 6 + k % 5
            peaks = [first + 15 + 20 * j for j in range(n_taps)]
            self.clips[ref] = {
                "first": first,
                "peaks": peaks,
                "last": peaks[-1] + 18,
                "cell": cell[ref],
            }
        for i, ref in enumerate(self.primary):
            self.clips[ref] |= {"error": ERRORS[i], "index": i}
        for ref in self.hard:
            self.clips[ref] |= {"error": 1, "index": -1}
        second: list[str] = []
        for name in mc.CELL_ORDER:
            members = [r for r in self.primary if self.clips[r]["cell"] == name]
            flag = "flagged" if name.endswith("/flagged") else "unflagged"
            second += members[: mc.SECOND_PER_TERTILE[flag]]
        self.second = second
        for ref, c in self.clips.items():
            self._write_clip(ref, c)
        rows = [
            {"ref": r, "fps": "30.04", "tap_rate_hz": rate[r],
             "qc_status": "retained_flag" if flagged[r] else "retained", "low_range": "False",
             "n_one_sided": "0",
             "n_taps": len(self.clips[r]["peaks"]) if r in self.clips else 20,
             "n_open_edge_ends": self.clips[r].get("index", 0) % 2 if r in self.clips else 0}
            for r in sorted(rate)
        ]  # fmt: skip
        pd.DataFrame(rows).to_csv(self.interim / "taps" / "taps_summary.csv", index=False)
        pd.DataFrame(
            {
                "ref": sorted(rate),
                "verdict": ["ok_flag" if flagged[r] else "ok" for r in sorted(rate)],
                "note": ["x" if flagged[r] else "" for r in sorted(rate)],
            }
        ).to_csv(self.docs / "qc_retained.csv", index=False)
        pd.DataFrame({"ref": [], "reason": [], "note": []}).to_csv(
            self.docs / "qc_rejections.csv", index=False
        )
        self.write_export()

    def weight(self, cell: str) -> float:
        return self.cell_n[cell] / mc.ALLOCATION[cell]

    def _write_clip(self, ref: str, c: dict[str, Any]) -> None:
        frames = np.arange(c["first"], c["last"] + 1)
        value = 1.0 + 0.5 * np.sin(frames / 5.0)
        if c.get("index") == 3:  # a tracking gap inside the signal
            value[(frames >= c["peaks"][2] + 12) & (frames <= c["peaks"][2] + 16)] = np.nan
        pd.DataFrame({"frame": frames, "t_s": frames / 30.0, "value": value}).to_parquet(
            self.interim / "smoothed" / f"{ref}.parquet"
        )
        rel = lambda f: (f - c["first"]) / 30.0  # noqa: E731 - the taps.py clock starts at the first sample
        pd.DataFrame(
            {
                "frame": c["peaks"],
                "t_peak_s": [rel(p) for p in c["peaks"]],
                "trough_left_s": [rel(p - 10) for p in c["peaks"]],
                "trough_right_s": [rel(p + 10) for p in c["peaks"]],
                "flags": ["narrow" if j % 4 == 0 else "" for j in range(len(c["peaks"]))],
            }
        ).to_parquet(self.interim / "taps" / f"{ref}.parquet")

    def manual(self, ref: str, error: int) -> list[int]:
        peaks = self.clips[ref]["peaks"]
        frames = [p + (j % 3) - 1 for j, p in enumerate(peaks)]
        if error > 0:
            frames = frames[:-error]
        frames += [peaks[j] + 4 for j in range(max(-error, 0))]  # double bumps in the same window
        return sorted(frames)

    def export_rows(self) -> list[dict[str, Any]]:
        out = []
        plan = [("primary", "R1", r) for r in self.primary]
        plan += [("second", "R2", r) for r in self.second] + [("hard", "R2", r) for r in self.hard]
        for kind, rater, ref in plan:
            c = self.clips[ref]
            error = c["error"] if kind != "second" else 0
            frames = self.manual(ref, error) if kind != "second" else self.manual(ref, 0)[:-1]
            out.append(
                {"ref": ref, "rater": rater, "set": kind, "n_taps": len(frames),
                 "tap_frames": ";".join(map(str, frames)),
                 "tap_times_s": ";".join(f"{f / 30 + 0.5:.3f}" for f in frames),  # a different clock
                 "n_edge_partials": 1 if c["peaks"][0] % 2 else 0, "confidence": 3, "note": "",
                 "cell": c["cell"], "weight": self.weight(c["cell"]) if kind == "primary" else "",
                 "window_first": c["first"], "window_last": c["last"]}
            )  # fmt: skip
        return out

    def write_export(self, rows: list[dict[str, Any]] | None = None) -> None:
        df = pd.DataFrame(rows if rows is not None else self.export_rows()).astype(str)
        df[list(mc.EXPORT_COLUMNS)].to_csv(self.docs / mc.EXPORT_NAME, index=False)

    def read_export(self) -> pd.DataFrame:
        return pd.read_csv(self.docs / mc.EXPORT_NAME, dtype=str, keep_default_na=False)

    def write_selection(self) -> None:
        d = self.interim / "manual_counts"
        d.mkdir()
        rows = [
            {"ref": r, "set": "primary", "second_rater": "1" if r in self.second else "0",
             "cell": self.clips[r]["cell"], "cell_rank": "0",
             "weight": str(self.weight(self.clips[r]["cell"])), "replaced_for": "", "seed": "24"}
            for r in self.primary
        ]  # fmt: skip
        rows += [
            {"ref": r, "set": "hard", "second_rater": "0", "cell": self.clips[r]["cell"],
             "cell_rank": "", "weight": "", "replaced_for": "", "seed": "24"}
            for r in self.hard
        ]  # fmt: skip
        pd.DataFrame(rows)[list(mc.SELECTION_COLUMNS)].to_csv(d / mc.SELECTION_NAME, index=False)


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    w = World(tmp_path)
    monkeypatch.setattr(mc, "INTERIM_ROOT", w.interim)
    monkeypatch.setattr(mc, "DOCS_ROOT", w.docs)
    monkeypatch.setattr(mc, "VIDEO_ROOT", w.videos)
    monkeypatch.setattr(mc, "HARD", tuple(w.hard))
    monkeypatch.setattr(mc, "PRACTICE", tuple(w.practice))
    return w


def analysed(
    w: World, boot: int = 200, seed: int = 5, kind: str = "inter_rater",
    frame: dict[str, int] | None = None,
) -> vc.Analysis:  # fmt: skip
    rows = vc.parse_rows(vc.read_export(w.docs / mc.EXPORT_NAME))
    people = mc.participant_keys(w.videos)
    sizes = w.cell_n if frame is None else frame
    return vc.analyse(rows, vc.load_inputs(w.interim), people, sizes, seed, boot, kind, False)


def run(args: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    code = vc.main(args)
    out = capsys.readouterr()
    return code, out.out, out.err


# --------------------------------------------------------------------------- error metrics


def test_hand_computed_errors() -> None:
    man, det, w = (
        np.array([10, 10, 10.0]),
        np.array([11, 10, 8.0]),
        np.array([1, 2, 3.0]),
    )  # d = +1, 0, -2
    m = vc.error_metrics(man, det, w)
    assert m["mae"] == pytest.approx(1.0) and m["bias"] == pytest.approx(-1 / 3)
    assert m["mae_weighted"] == pytest.approx(7 / 6) and m["bias_weighted"] == pytest.approx(-5 / 6)
    assert m["max_abs_error"] == 2
    assert m["share_abs_le_1"] == pytest.approx(2 / 3) and m["share_abs_le_2"] == 1.0
    assert m["share_abs_le_1_weighted"] == pytest.approx(0.5)  # |d| = 1 counts: <= 1, not < 1
    assert m["relative_mae"] == pytest.approx(0.1) and m["relative_mae_weighted"] == pytest.approx(
        0.7 / 6
    )


def test_sign_convention_is_detector_minus_manual() -> None:
    assert vc.error_metrics(np.array([10.0, 20.0]), np.array([12.0, 21.0]))[
        "bias"
    ] == pytest.approx(1.5)
    assert vc.error_metrics(np.array([10.0, 20.0]), np.array([8.0, 19.0]))["bias"] == pytest.approx(
        -1.5
    )


def test_relative_mae_skips_zero_manual_counts() -> None:
    m = vc.error_metrics(np.array([0.0, 10.0]), np.array([2.0, 11.0]))
    assert m["relative_mae_n"] == 1 and m["relative_mae"] == pytest.approx(0.1)


def test_bland_altman_limits() -> None:
    man = np.array([10.0, 20, 30, 40, 50])
    det = man + np.array([1, -1, 0, 2, -2])  # d: mean 0, SD sqrt(2.5)
    ba = vc.bland_altman(man, det)
    assert ba["bias"] == pytest.approx(0) and ba["sd"] == pytest.approx(np.sqrt(2.5))
    assert ba["loa_upper"] == pytest.approx(1.96 * np.sqrt(2.5))
    assert ba["loa_upper_ci"][0] < ba["loa_upper"] < ba["loa_upper_ci"][1]
    assert ba["loa_lower"] == pytest.approx(-ba["loa_upper"])


# --------------------------------------------------------------------------- ICC


def ref_icc(x: np.ndarray, kind: int = 2) -> float:
    """ICC(2,1) or ICC(3,1) written out from the definitions, loop by loop."""
    n, k = x.shape
    g = sum(x[i, j] for i in range(n) for j in range(k)) / (n * k)
    rm = [sum(x[i, j] for j in range(k)) / k for i in range(n)]
    cm = [sum(x[i, j] for i in range(n)) / n for j in range(k)]
    msr = k * sum((m - g) ** 2 for m in rm) / (n - 1)
    msc = n * sum((m - g) ** 2 for m in cm) / (k - 1)
    sse = sum((x[i, j] - rm[i] - cm[j] + g) ** 2 for i in range(n) for j in range(k))
    mse = sse / ((n - 1) * (k - 1))
    if kind == 3:
        return (msr - mse) / (msr + (k - 1) * mse)
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def test_icc_hand_example() -> None:
    r = vc.icc2(np.array([10.0, 20, 30]), np.array([12.0, 20, 28]))
    assert r["icc"] == pytest.approx(0.98361, abs=1e-5)  # MSR 162, MSE 2, MSC 0: 160 / 162.667
    assert r["n"] == 3


def test_icc_shrout_fleiss_example() -> None:
    sf = np.array(
        [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7.0]]
    )
    assert float(vc.icc_stack(sf[None])[0]) == pytest.approx(0.2898, abs=1e-4)  # published .29


def test_icc_matches_the_loop_definition_on_random_data() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        t = rng.normal(30, 10, 25)
        man, det = t + rng.normal(0, 2, 25), t + rng.normal(1, 2, 25)
        assert vc.icc2(man, det)["icc"] == pytest.approx(ref_icc(np.c_[man, det]), abs=1e-10)


def test_icc_equals_pingouin_when_installed() -> None:
    pg = pytest.importorskip("pingouin")
    rng = np.random.default_rng(2)
    for _ in range(10):
        n = int(rng.integers(8, 40))
        t = rng.normal(30, 12, n)
        man, det = t + rng.normal(0, 3, n), t + rng.normal(2, 3, n)
        long = pd.DataFrame(
            {"s": np.repeat(np.arange(n), 2), "r": np.tile([0, 1], n), "v": np.c_[man, det].ravel()}
        )
        table = pg.intraclass_corr(long, "s", "r", "v")
        row = table.iloc[1]  # position 1 = ICC(2,1); its label is "ICC2" or "ICC(A,1)" by version
        assert row["Type"] in ("ICC2", "ICC(A,1)")
        mine = vc.icc2(man, det)
        assert mine["icc"] == pytest.approx(float(row["ICC"]), abs=5e-4)
        ci = row["CI95%"] if "CI95%" in table.columns else row["CI95"]
        assert mine["ci_f"][0] == pytest.approx(float(ci[0]), abs=6e-3)
        assert mine["ci_f"][1] == pytest.approx(float(ci[1]), abs=6e-3)


def test_icc_21_differs_from_31_with_a_constant_bias() -> None:
    t = np.linspace(10, 60, 30)
    x = np.c_[t, t + 3.0 + np.tile([-0.5, 0.5], 15)]
    assert ref_icc(x, 3) > 0.995 and vc.icc2(x[:, 0], x[:, 1])["icc"] < ref_icc(x, 3) - 0.005
    assert vc.icc2(x[:, 0], x[:, 1])["icc"] == pytest.approx(ref_icc(x, 2), abs=1e-10)


def test_icc_degenerate_inputs_give_nan_not_errors() -> None:
    same = vc.icc2(np.array([5.0, 5, 5, 5]), np.array([5.0, 5, 5, 5]))
    assert np.isnan(same["icc"]) and np.isnan(same["ci_f"][0])
    perfect = vc.icc2(np.array([5.0, 9, 12]), np.array([5.0, 9, 12]))
    assert perfect["icc"] == pytest.approx(1.0) and np.isnan(perfect["ci_f"][0])


# --------------------------------------------------------------------------- decision wording


def test_decision_established() -> None:
    d = vc.decide(0.8, [0.5, 0.95], 0.98, [0.96, 0.99])
    assert (d["target_met"], d["established"], d["not_met"]) == (True, True, False)
    assert d["statements"] == [] and d["status"] == "established" and not d["triggers_ft26"]


def test_decision_met_on_the_point_estimate_only() -> None:
    d = vc.decide(0.9, [0.6, 1.3], 0.93, [0.85, 0.97])
    assert (d["target_met"], d["established"], d["not_met"]) == (True, False, False)
    assert d["statements"] == [
        "Met on the point estimate; the 95% CI (0.60-1.30) includes values above 1.0",
        "Met on the point estimate; the 95% CI (0.85-0.97) includes values below 0.90",
    ]
    assert d["status"] == "met_on_point_estimate"


def test_decision_not_met_triggers_ft26() -> None:
    d = vc.decide(1.4, [1.1, 1.7], 0.99, [0.97, 0.995])
    assert (d["target_met"], d["established"], d["not_met"]) == (False, False, True)
    assert d["triggers_ft26"] and d["statements"] == [] and d["status"] == "not_met"


def test_decision_boundaries_are_inclusive() -> None:
    d = vc.decide(1.0, [0.8, 1.0], 0.90, [0.90, 0.95])
    assert d["target_met"] and d["established"] and not d["not_met"]
    assert vc.decide(1.01, [0.8, 1.2], 0.95, [0.9, 0.97])["not_met"]
    assert vc.decide(0.5, [0.3, 0.7], 0.85, [0.8, 0.9])["status"] == "icc_below_target"


# --------------------------------------------------------------------------- bootstrap


def test_bootstrap_is_reproducible_and_seed_dependent() -> None:
    rng = np.random.default_rng(0)
    cells = np.array([c for c in mc.CELL_ORDER for _ in range(mc.ALLOCATION[c])])
    man = rng.integers(10, 60, 40).astype(float)
    det = man + rng.integers(-2, 3, 40)
    w = np.array([CELL_N[c] / mc.ALLOCATION[c] for c in cells])
    a, b = vc.boot_primary(man, det, w, cells, 300, 7), vc.boot_primary(man, det, w, cells, 300, 7)
    assert a == b
    assert vc.boot_primary(man, det, w, cells, 300, 8) != a
    s1, s2 = vc.boot_simple(man, det, 300, 7, 2), vc.boot_simple(man, det, 300, 7, 2)
    assert s1 == s2 and vc.boot_simple(man, det, 300, 7, 3) != s1  # another stream, other draws


def test_resampling_stays_within_cells() -> None:
    cells = np.array([c for c in mc.CELL_ORDER for _ in range(mc.ALLOCATION[c])])
    idx = vc.resample_indices(cells, 500, np.random.default_rng(3))
    assert idx.shape == (500, 40)
    taken = cells[idx]
    for cell in mc.CELL_ORDER:
        members = set(np.flatnonzero(cells == cell))
        assert ((taken == cell).sum(axis=1) == len(members)).all()  # each row keeps the cell's size
        assert set(np.unique(idx[taken == cell])) <= members  # and draws only from the cell


def test_cluster_resampling_keeps_participants_whole() -> None:
    groups = ["a", "a", "b", "c", "c", "c", "d"]
    draws = vc.resample_clusters(groups, 400, np.random.default_rng(4))
    for idx in draws:
        for g in set(groups):
            members = [i for i, x in enumerate(groups) if x == g]
            counts = [int((idx == i).sum()) for i in members]
            assert len(set(counts)) == 1  # a participant's clips are drawn together
    assert any(len(idx) != len(groups) for idx in draws)  # the clip count varies with the draw


def test_primary_clips_must_be_different_known_participants(world: World) -> None:
    rows = vc.parse_rows(vc.read_export(world.docs / mc.EXPORT_NAME))
    people = mc.participant_keys(world.videos)
    inputs = vc.load_inputs(world.interim)
    twin = dict(people)
    twin[world.primary[1]] = twin[world.primary[0]]  # two primary clips of one participant
    with pytest.raises(vc.Refused, match="different, known participants"):
        vc.analyse(rows, inputs, twin, world.cell_n, 1, 20, "inter_rater", False)
    unknown = {k: v for k, v in people.items() if k != world.primary[5]}
    with pytest.raises(vc.Refused, match="different, known participants"):
        vc.analyse(rows, inputs, unknown, world.cell_n, 1, 20, "inter_rater", False)


def test_hard_set_uses_participant_resampling(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[str]] = []
    real = vc.resample_clusters

    def spy(groups: Any, boot: int, rng: Any) -> Any:
        seen.append(list(groups))
        return real(groups, boot, rng)

    monkeypatch.setattr(vc, "resample_clusters", spy)
    report = analysed(world).report
    assert len(seen) == 1 and len(seen[0]) == 16 and len(set(seen[0])) == 13
    assert report["hard_set"]["n_participants"] == 13 and report["hard_set"]["n"] == 16


# --------------------------------------------------------------------------- the whole analysis


def test_primary_numbers_match_an_independent_calculation(world: World) -> None:
    rep = analysed(world).report["primary"]
    err = np.array(ERRORS, float)
    w = np.array([world.weight(cell_of(i)) for i in range(40)])
    assert rep["metrics"]["mae_weighted"] == pytest.approx((w * np.abs(err)).sum() / w.sum())
    assert rep["metrics"]["mae"] == pytest.approx(np.abs(err).mean()) and rep["metrics"]["n"] == 40
    assert rep["metrics"]["bias"] == pytest.approx(err.mean())
    assert rep["metrics"]["bias_weighted"] == pytest.approx((w * err).sum() / w.sum())
    assert rep["metrics"]["max_abs_error"] == 2
    assert rep["metrics"]["share_abs_le_1"] == pytest.approx(np.mean(np.abs(err) <= 1))
    clips = analysed(world).clips
    prim = clips[clips["set"] == "primary"]
    man, det = prim["n_man"].to_numpy(float), prim["n_det"].to_numpy(float)
    assert (det - man == err).all()
    assert rep["icc"]["icc"] == pytest.approx(ref_icc(np.c_[man, det]), abs=1e-10)
    assert rep["n_outside"] == 0
    assert w.sum() == pytest.approx(214)


def test_weights_apply_to_mae_only(world: World) -> None:
    base = analysed(world).report["primary"]
    df = world.read_export()
    prim = df["set"] == "primary"
    df.loc[prim, "weight"] = df.loc[prim, "weight"].astype(float).map(lambda x: str(x * 2))
    df.to_csv(world.docs / mc.EXPORT_NAME, index=False)
    doubled = {c: 2 * n for c, n in world.cell_n.items()}  # the frame sizes that go with them
    scaled = analysed(world, frame=doubled).report["primary"]
    assert scaled["icc"] == base["icc"] and scaled["metrics"]["mae"] == base["metrics"]["mae"]
    assert scaled["metrics"]["mae_weighted"] == pytest.approx(base["metrics"]["mae_weighted"])
    df.loc[df["cell"] == "mid/flagged", "weight"] = "25.0"  # reweight one cell (and its frame size)
    df.to_csv(world.docs / mc.EXPORT_NAME, index=False)
    reweighted = doubled | {"mid/flagged": 25 * mc.ALLOCATION["mid/flagged"]}
    moved = analysed(world, frame=reweighted).report["primary"]
    assert moved["metrics"]["mae_weighted"] != pytest.approx(base["metrics"]["mae_weighted"])
    assert moved["icc"] == base["icc"] and moved["metrics"]["mae"] == base["metrics"]["mae"]


def test_per_cell_table(world: World) -> None:
    cells = analysed(world).report["per_cell"]
    assert [c["cell"] for c in cells] == list(mc.CELL_ORDER)
    assert [c["n"] for c in cells] == list(mc.ALLOCATION.values())
    first = cells[0]  # clips 0..8: errors +1 and -2 -> MAE 3/9
    assert first["mae"] == pytest.approx(3 / 9) and first["mae_weighted"] == pytest.approx(3 / 9)
    assert first["bias"] == pytest.approx(-1 / 9)


def test_primary_and_hard_sets_are_not_pooled(world: World) -> None:
    rep = analysed(world).report
    assert rep["primary"]["metrics"]["n"] == 40 and rep["hard_set"]["n"] == 16
    assert rep["hard_set"]["pooled_with_primary"] is False
    assert rep["hard_set"]["metrics"]["mae"] == pytest.approx(1.0)  # every hard clip has error +1
    assert rep["primary"]["metrics"]["mae"] == pytest.approx(
        np.abs(ERRORS).mean()
    )  # unaffected by it


def test_second_rater_section(world: World) -> None:
    sec = analysed(world).report["second_rater"]  # R2 = detector peaks minus the last one
    assert sec["n"] == 12 and sec["second_rater_kind"] == "inter_rater" and sec["descriptive_only"]
    clips = analysed(world).clips
    prim = clips[clips["set"] == "primary"].set_index("ref")
    second = clips[clips["set"] == "second"].set_index("ref")
    refs = sorted(second.index)
    r1, r2, det = (prim.loc[refs, "n_man"], second.loc[refs, "n_man"], second.loc[refs, "n_det"])
    assert sec["r1_vs_r2"]["mae"] == pytest.approx(np.abs(r1 - r2).mean())
    assert sec["detector_vs_r1"]["mae"] == pytest.approx(np.abs(det - r1).mean())
    assert sec["detector_vs_r2"]["mae"] == pytest.approx(np.abs(det - r2).mean())
    assert sec["detector_vs_mean_r1_r2"]["mae"] == pytest.approx(np.abs(det - (r1 + r2) / 2).mean())
    assert sec["detector_vs_r2"]["mae"] == pytest.approx(1.0)  # R2 always misses the last tap
    assert sec["detector_vs_r2"]["mae"] != pytest.approx(
        sec["detector_vs_r1"]["mae"]
    )  # raters not swapped
    assert (
        analysed(world, kind="intra_rater_recount").report["second_rater"]["second_rater_kind"]
        == "intra_rater_recount"
    )
    assert sec["bias_definition"] == "detector - manual; for r1_vs_r2: R2 - R1"
    assert sec["r1_vs_r2"]["bias"] == pytest.approx((r2 - r1).mean())
    assert "p_value" not in json.dumps(sec)  # descriptive only: no hypothesis tests


def test_missing_r2_gives_no_second_rater_section(world: World) -> None:
    df = world.read_export()
    df[df["set"] != "second"].to_csv(world.docs / mc.EXPORT_NAME, index=False)
    rep = analysed(world).report
    assert "second_rater" not in rep and rep["primary"]["metrics"]["n"] == 40
    assert "hard_set" in rep


def test_partial_r2_is_refused(world: World) -> None:
    df = world.read_export()
    df.drop(df.index[df["set"] == "second"][:3]).to_csv(world.docs / mc.EXPORT_NAME, index=False)
    with pytest.raises(vc.Refused, match="second-rater rows"):
        analysed(world)


def test_missing_hard_set_is_allowed(world: World) -> None:
    df = world.read_export()
    df[df["set"] != "hard"].to_csv(world.docs / mc.EXPORT_NAME, index=False)
    rep = analysed(world).report
    assert "hard_set" not in rep and "second_rater" in rep


def test_practice_rows_are_refused(world: World) -> None:
    df = world.read_export()
    extra = df[df["set"] == "primary"].iloc[:3].copy()
    extra["set"] = "practice"
    pd.concat([df, extra]).to_csv(world.docs / mc.EXPORT_NAME, index=False)
    with pytest.raises(vc.Refused, match="unknown set"):
        analysed(world)


def test_a_41st_primary_clip_is_refused(world: World) -> None:
    df = world.read_export()
    extra = df[df["set"] == "hard"].iloc[:1].copy()
    extra["set"], extra["rater"], extra["cell"], extra["weight"] = (
        "primary",
        "R1",
        "slow/unflagged",
        "6.9",
    )
    pd.concat([df, extra]).to_csv(world.docs / mc.EXPORT_NAME, index=False)
    with pytest.raises(vc.Refused):
        analysed(world)


# --------------------------------------------------------------------------- window rule


def count_row(world: World, ref: str, frames: list[int], n_edge: int = 0) -> vc.CountRow:
    c = world.clips[ref]
    return vc.CountRow(
        ref,
        "R1",
        "primary",
        tuple(frames),
        len(frames),
        n_edge,
        "3",
        c["cell"],
        1.0,
        c["first"],
        c["last"],
    )


def test_window_edges_are_inclusive(world: World) -> None:
    ref = next(r for r in world.primary if world.clips[r]["first"] == 1)  # the signal starts late
    c = world.clips[ref]
    first, last = c["first"], c["last"]
    inputs = vc.load_inputs(world.interim)
    clip, taps = vc.clip_result(
        count_row(world, ref, [first - 1, first, last, last + 1, last + 2]), inputs
    )
    assert clip["n_man"] == 2 and clip["n_outside"] == 3  # first and last frame counted
    assert [t["manual_frame"] for t in taps if t["kind"] != "FP"] == [first, last]
    assert vc.clip_result(count_row(world, ref, [first - 1]), inputs)[0]["n_man"] == 0
    assert vc.clip_result(count_row(world, ref, [last + 1]), inputs)[0]["n_man"] == 0
    assert vc.clip_result(count_row(world, ref, [first]), inputs)[0]["n_man"] == 1
    assert vc.clip_result(count_row(world, ref, [last]), inputs)[0]["n_man"] == 1


def test_edge_partials_never_count(world: World) -> None:
    ref = world.primary[0]
    peaks = world.clips[ref]["peaks"]
    clip, _ = vc.clip_result(count_row(world, ref, peaks, n_edge=2), vc.load_inputs(world.interim))
    assert clip["n_man"] == len(peaks) and clip["manual_edge_partials"] == 2 and clip["d"] == 0
    assert clip["detector_edge_ends"] == 0


def test_matching_uses_frames_not_the_times_in_the_export(world: World) -> None:
    ref = next(r for r in world.primary if world.clips[r]["first"] == 2)  # a late-starting signal
    df = world.read_export()
    row = df[df["ref"] == ref].iloc[0]
    assert (
        row["tap_times_s"].split(";")[0] != f"{int(row['tap_frames'].split(';')[0]) / 30:.3f}"
    )  # clocks differ
    clips = analysed(world).clips
    one = clips[(clips["ref"] == ref) & (clips["set"] == "primary")].iloc[0]
    assert one["tp"] == min(one["n_det"], one["n_man"]) and one["n_outside"] == 0


def test_window_must_agree_with_the_smoothed_signal(world: World) -> None:
    df = world.read_export()
    df.loc[0, "window_last"] = str(int(df.loc[0, "window_last"]) + 1)
    df.to_csv(world.docs / mc.EXPORT_NAME, index=False)
    with pytest.raises(vc.Refused, match="window differs"):
        analysed(world)


def test_taps_file_and_summary_must_agree(world: World) -> None:
    path = world.interim / "taps" / "taps_summary.csv"
    df = pd.read_csv(path, dtype=str)
    df.loc[0, "n_taps"] = str(int(df.loc[0, "n_taps"]) + 1)
    df.to_csv(path, index=False)
    with pytest.raises(vc.Refused, match="disagree"):
        analysed(world)


# --------------------------------------------------------------------------- tap matching

PEAKS, LEFT, RIGHT = np.array([15, 35, 55]), np.array([5, 25, 45]), np.array([25, 45, 65])
FLAGS = ["a", "b", "c"]


def match(manual: list[int], gaps: set[int] | None = None) -> list[dict[str, Any]]:
    return vc.match_taps(manual, PEAKS, LEFT, RIGHT, FLAGS, gaps or set(), 0, 70)


def kinds(rows: list[dict[str, Any]]) -> list[tuple[str, Any, Any]]:
    return [(r["kind"], r["manual_frame"], r["detector_frame"]) for r in rows]


def test_all_hits() -> None:
    rows = match([14, 36, 55])
    assert kinds(rows) == [("TP", 14, 15), ("TP", 36, 35), ("TP", 55, 55)]
    assert [r["offset_frames"] for r in rows] == [-1, 1, 0]
    assert [r["detector_flags"] for r in rows] == FLAGS


def test_a_double_bump_is_one_tp_and_one_fn() -> None:
    rows = match([15, 18, 35, 55])
    assert sorted(kinds(rows), key=str) == sorted(
        [("TP", 15, 15), ("FN", 18, 15), ("TP", 35, 35), ("TP", 55, 55)], key=str
    )
    assert sum(r["kind"] == "TP" for r in rows) == 3  # never one-to-many
    rows = match([12, 17, 35, 55])  # the nearer one wins, whatever the order
    assert [r["manual_frame"] for r in rows if r["kind"] == "TP"][0] == 17


def test_an_extra_detector_tap_is_a_false_positive() -> None:
    rows = match([15, 55])
    assert ("FP", None, 35) in kinds(rows) and sum(r["kind"] == "TP" for r in rows) == 2
    fp = next(r for r in rows if r["kind"] == "FP")
    assert (
        fp["offset_frames"] is None and fp["detector_flags"] == "b" and fp["position"] == "middle"
    )


def test_a_manual_tap_in_no_window_is_a_false_negative_and_a_nan_gap_is_noted() -> None:
    rows = match([15, 35, 55, 68, 80], gaps={68})
    fns = [r for r in rows if r["kind"] == "FN"]
    assert [(r["manual_frame"], r["detector_frame"], r["in_nan_gap"]) for r in fns] == [
        (68, None, True), (80, None, False)
    ]  # fmt: skip
    assert all(r["detector_flags"] == "" and r["offset_frames"] is None for r in fns)
    assert match([15, 35, 55], gaps={68})[0]["in_nan_gap"] is False


def test_a_shared_trough_goes_to_the_window_with_the_nearer_peak() -> None:
    left, right, names = np.array([5, 25]), np.array([25, 55]), ["x", "y"]
    earlier = vc.match_taps([25], np.array([15, 38]), left, right, names, set(), 0, 70)
    assert [(r["kind"], r["detector_frame"]) for r in earlier if r["kind"] == "TP"] == [("TP", 15)]
    later = vc.match_taps([25], np.array([15, 30]), left, right, names, set(), 0, 70)
    assert [(r["kind"], r["detector_frame"]) for r in later if r["kind"] == "TP"] == [("TP", 30)]
    assert [(r["kind"], r["detector_frame"]) for r in later if r["kind"] == "FP"] == [("FP", 15)]
    tie = vc.match_taps([25], np.array([15, 35]), left, right, names, set(), 0, 70)
    assert [r["detector_frame"] for r in tie if r["kind"] == "TP"] == [15]  # tie: earlier window


def test_matching_is_one_to_one() -> None:
    rng = np.random.default_rng(6)
    for _ in range(50):
        manual = sorted(set(int(x) for x in rng.integers(0, 70, int(rng.integers(0, 12)))))
        rows = match(manual)
        assert sorted(r["manual_frame"] for r in rows if r["manual_frame"] is not None) == manual
        tp = [r["detector_frame"] for r in rows if r["kind"] == "TP"]
        assert len(tp) == len(set(tp))  # a detector tap is hit once
        assert sum(r["kind"] in ("TP", "FP") for r in rows) == len(PEAKS)
        assert sum(r["kind"] in ("TP", "FN") for r in rows) == len(manual)


def test_positions() -> None:
    peaks = np.array([15, 35, 55, 75, 95])
    assert [vc.position_of(f, peaks, 0, 110) for f in (10, 34, 35, 60, 75, 76, 100)] == [
        "first_interval", "first_interval", "middle", "middle", "middle", "last_interval", "last_interval"
    ]  # fmt: skip
    assert vc.position_of(30, np.array([15]), 0, 100) == "first_interval"
    assert vc.position_of(80, np.array([15, 35]), 0, 100) == "last_interval"
    assert vc.position_of(5, np.array([], dtype=int), 0, 100) == "first_interval"


def test_tap_table_for_ft26(world: World) -> None:
    taps = analysed(world).taps
    assert list(taps.columns) == list(vc.TAP_COLUMNS)
    assert set(taps["kind"]) <= {"TP", "TP_offset", "FN", "FP"} and set(taps["qc_status"]) <= {
        "retained",
        "retained_flag",
    }
    assert set(taps["position"]) <= {"first_interval", "middle", "last_interval"}
    prim = analysed(world).clips
    prim = prim[prim["set"] == "primary"]
    mine = taps[taps["set"] == "primary"]
    assert (mine["kind"] == "FN").sum() == prim["fn"].sum() and (
        mine["kind"] == "FP"
    ).sum() == prim["fp"].sum()
    # +1 error clips drop a tap (FP), -2 clips add two double bumps (FN)
    assert prim["fp"].sum() == 1 + 2 + 1 and prim["fn"].sum() == 2 + 1 + 1


def test_nan_gap_frames_are_marked(world: World) -> None:
    ref = world.primary[3]  # the clip with a tracking gap
    peaks = world.clips[ref]["peaks"]
    gap = peaks[2] + 14
    inputs = vc.load_inputs(world.interim)
    frames = [p for p in peaks if p != peaks[2]] + [gap]
    _, taps = vc.clip_result(count_row(world, ref, sorted(frames)), inputs)
    nan_rows = [t for t in taps if t["in_nan_gap"]]
    assert [t["manual_frame"] for t in nan_rows] == [gap]


# --------------------------------------------------------------------------- command line


def test_report_on_the_synthetic_export_writes_the_outputs(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run(["report", "--boot", "200"], capsys)
    assert (code, err) == (0, "")
    assert "primary n=40: weighted MAE" in out and "ICC(2,1)" in out and "SYNTHETIC" not in out
    out_dir = world.interim / "validation"
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "bland_altman.png", "clip_errors.csv", "error_by_cell.png", "tap_matches.csv"
    ]  # fmt: skip
    rep = json.loads((world.docs / vc.REPORT_NAME).read_text())
    assert rep["data_kind"] == "real" and rep["parameters"]["boot"] == 200
    assert (
        rep["parameters"]["second_rater_kind"] == "inter_rater"
        and "window_rule" in rep["parameters"]
    )
    for key in ("per_cell", "primary", "second_rater", "hard_set"):
        assert key in rep
    assert (
        "ICC(2,1) within the stratified validation sample of 40 clips"
        in rep["primary"]["icc_wording"]
    )
    assert "not an estimate of the ICC in the full dataset" in rep["primary"]["icc_wording"]
    assert "weak gate" in rep["primary"]["icc_wording"]
    csv = pd.read_csv(out_dir / "clip_errors.csv", dtype=str)
    assert list(csv.columns) == list(vc.CLIP_COLUMNS) and len(csv) == 40 + 12 + 16


def test_second_rater_kind_option(world: World, capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        run(["report", "--boot", "50", "--second-rater-kind", "intra_rater_recount"], capsys)[0]
        == 0
    )
    rep = json.loads((world.docs / vc.REPORT_NAME).read_text())
    assert rep["parameters"]["second_rater_kind"] == "intra_rater_recount"
    assert rep["second_rater"]["second_rater_kind"] == "intra_rater_recount"
    with pytest.raises(SystemExit):
        vc.main(["report", "--second-rater-kind", "nonsense"])


def test_incomplete_r1_is_refused(world: World, capsys: pytest.CaptureFixture[str]) -> None:
    df = world.read_export()
    df.drop(df.index[df["set"] == "primary"][-1]).to_csv(world.docs / mc.EXPORT_NAME, index=False)
    code, out, err = run(["report", "--boot", "50"], capsys)
    assert code == 1 and "39 primary rows" in err and "no partial counts" in err and out == ""
    assert (
        not (world.docs / vc.REPORT_NAME).exists() and not (world.interim / "validation").exists()
    )


def test_missing_or_malformed_export(world: World, capsys: pytest.CaptureFixture[str]) -> None:
    (world.docs / mc.EXPORT_NAME).unlink()
    assert run(["report"], capsys)[0] == 2
    world.write_export()
    old = world.read_export().drop(columns=["window_last"])
    old.to_csv(world.docs / mc.EXPORT_NAME, index=False)
    code, _, err = run(["report"], capsys)
    assert code == 2 and "must have the columns" in err
    world.write_export()
    bad = world.read_export()
    bad.loc[0, "tap_frames"] = "x;y"
    bad.to_csv(world.docs / mc.EXPORT_NAME, index=False)
    assert run(["report"], capsys)[0] == 2


@pytest.mark.parametrize(
    "mutate",
    [
        lambda df: df.drop(index=[0, 1]),  # drops two rows
        lambda df: pd.concat([df, df.iloc[:1]]),  # a duplicated row
    ],
)
def test_inconsistent_exports_are_refused(
    world: World, capsys: pytest.CaptureFixture[str], mutate: Any
) -> None:
    mutate(world.read_export()).to_csv(world.docs / mc.EXPORT_NAME, index=False)
    assert run(["report", "--boot", "50"], capsys)[0] == 1


def test_inconsistent_weights_and_cells_are_refused(world: World) -> None:
    df = world.read_export()
    prim = df[df["set"] == "primary"].index
    df.loc[prim[0], "weight"] = "7.0"  # two weights in one cell
    df.to_csv(world.docs / mc.EXPORT_NAME, index=False)
    with pytest.raises(vc.Refused, match="one positive number"):
        analysed(world)
    df = world.read_export()
    df.loc[df.index[df["set"] == "primary"][0], "cell"] = "fast/flagged"
    df.to_csv(world.docs / mc.EXPORT_NAME, index=False)
    with pytest.raises(vc.Refused, match="cell"):
        analysed(world)


def test_path_guards(world: World, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["report", "--out-dir", str(tmp_path / "elsewhere")], capsys)[0] == 2
    assert run(["report", "--docs-dir", str(tmp_path / "elsewhere")], capsys)[0] == 2
    assert run(["report", "--boot", "0"], capsys)[0] == 2


def test_synthetic_never_writes_docs(world: World, capsys: pytest.CaptureFixture[str]) -> None:
    world.write_selection()
    (world.docs / mc.EXPORT_NAME).unlink()  # not even a real export is needed
    before = sorted((p.name, p.stat().st_mtime_ns) for p in world.docs.iterdir())
    code, out, err = run(["report", "--synthetic", "--boot", "100"], capsys)
    assert code == 0 and err == "" and out.startswith("data: SYNTHETIC")
    assert sorted((p.name, p.stat().st_mtime_ns) for p in world.docs.iterdir()) == before
    assert not (world.interim / "validation").exists()
    names = sorted(p.name for p in (world.interim / "validation_synthetic").iterdir())
    assert names and all(n.startswith("SYNTHETIC_") for n in names)
    assert "SYNTHETIC_validation_report.json" in names and "SYNTHETIC_truth.csv" in names
    rep = json.loads(
        (world.interim / "validation_synthetic" / "SYNTHETIC_validation_report.json").read_text()
    )
    assert rep["data_kind"] == "SYNTHETIC" and rep["synthetic_truth_check"] == "passed for 68 clips"
    # a --docs-dir given with --synthetic is still never written
    assert (
        run(["report", "--synthetic", "--boot", "20", "--docs-dir", str(world.docs)], capsys)[0]
        == 0
    )
    assert sorted((p.name, p.stat().st_mtime_ns) for p in world.docs.iterdir()) == before


def test_synthetic_errors_are_recovered_and_reproducible(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    world.write_selection()
    assert run(["report", "--synthetic", "--boot", "50", "--seed", "11"], capsys)[0] == 0
    out = world.interim / "validation_synthetic"
    first = (
        (out / "SYNTHETIC_truth.csv").read_text(),
        (out / "SYNTHETIC_clip_errors.csv").read_text(),
    )
    assert run(["report", "--synthetic", "--boot", "50", "--seed", "11"], capsys)[0] == 0
    assert (
        (out / "SYNTHETIC_truth.csv").read_text(),
        (out / "SYNTHETIC_clip_errors.csv").read_text(),
    ) == first
    truth = pd.read_csv(out / "SYNTHETIC_truth.csv")
    clips = pd.read_csv(out / "SYNTHETIC_clip_errors.csv")
    merged = truth.merge(clips, on=["ref", "set"])
    assert (merged["injected_error"] == merged["d"]).all() and len(merged) == 68


def test_plot_titles_say_synthetic() -> None:
    prim = pd.DataFrame(
        {"ref": ["aaaaaaaa", "bbbbbbbb"], "n_man": [10, 20], "n_det": [10, 24],
         "cell": ["slow/unflagged", "mid/flagged"]}
    )  # fmt: skip
    assert "SYNTHETIC" in vc.bland_altman_figure(prim, "SYNTHETIC").axes[0].get_title()
    assert "SYNTHETIC" in vc.error_by_cell_figure(prim, "SYNTHETIC").axes[0].get_title()
    labels = [t.get_text() for t in vc.bland_altman_figure(prim, "x").axes[0].texts]
    assert labels == ["bbbbbbbb"]  # only |d| > 2 is labelled, by its hashed ref


# --------------------------------------------------------------------------- privacy


def collect_strings(obj: Any) -> list[str]:
    if isinstance(obj, dict):
        return [str(k) for k in obj] + [s for v in obj.values() for s in collect_strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in collect_strings(v)]
    return [obj] if isinstance(obj, str) else []


def test_nothing_written_or_printed_has_ids_paths_or_labels(
    world: World,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_read_csv = pd.read_csv

    def guarded(path: object, *args: Any, **kwargs: Any) -> pd.DataFrame:
        assert "fis_diagnostic" not in str(path), "the labels file was opened"
        return real_read_csv(path, *args, **kwargs)  # type: ignore[call-overload,no-any-return]

    monkeypatch.setattr(pd, "read_csv", guarded)
    monkeypatch.setattr(
        hubu_fis, "load_labels", lambda *a, **k: (_ for _ in ()).throw(AssertionError("labels"))
    )
    world.write_selection()
    code1, out1, err1 = run(["report", "--boot", "100"], capsys)
    code2, out2, err2 = run(["report", "--synthetic", "--boot", "100"], capsys)
    assert code1 == code2 == 0
    texts = [out1, err1, out2, err2]
    files = [
        p
        for d in (world.interim / "validation", world.interim / "validation_synthetic")
        for p in d.iterdir()
    ]
    files.append(world.docs / vc.REPORT_NAME)
    for p in files:
        if p.suffix in (".csv", ".json"):
            texts.append(p.read_text())
        assert RAW_ID.search(p.name) is None
    for text in texts:
        assert RAW_ID.search(text) is None and str(tmp_path) not in text and "Users" not in text
        assert not re.search(r"(?i)control|patient|healthy|parkinson", text)
    report = json.loads((world.docs / vc.REPORT_NAME).read_text())
    strings = collect_strings(report)
    assert not [s for s in strings if HEX8.fullmatch(s)]  # no per-clip refs in the committed report
    assert not set(strings) & set(world.clips)
    src = Path(vc.__file__).read_text()
    assert "fis_diagnostic" not in src and "load_labels" not in src and "UPDRS" not in src


def test_committed_report_has_no_per_clip_rows(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(["report", "--boot", "50"], capsys)[0] == 0
    report = json.loads((world.docs / vc.REPORT_NAME).read_text())

    def has_ref_key(o: Any) -> bool:
        if isinstance(o, dict):
            return "ref" in o or any(has_ref_key(v) for v in o.values())
        return isinstance(o, list) and any(has_ref_key(v) for v in o)

    assert not has_ref_key(report)
    assert isinstance(report["per_cell"], list) and len(report["per_cell"]) == 6
    assert set(report["primary"]["tap_matching"]) == {"tp", "tp_offset", "fn", "fp"}


def test_report_json_is_stable_for_a_fixed_seed(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(["report", "--boot", "80", "--seed", "3"], capsys)[0] == 0
    a = (world.docs / vc.REPORT_NAME).read_text()
    assert run(["report", "--boot", "80", "--seed", "3"], capsys)[0] == 0
    assert (world.docs / vc.REPORT_NAME).read_text() == a
    assert run(["report", "--boot", "80", "--seed", "4"], capsys)[0] == 0
    assert (world.docs / vc.REPORT_NAME).read_text() != a
    json.loads(a)  # valid JSON with no NaN literals
    assert "NaN" not in a


# --------------------------------------------------------------------------- weights vs the frame


def test_expected_frame_matches_an_independent_count(world: World) -> None:
    got = vc.expected_frame(world.docs, world.interim / "taps" / "taps_summary.csv")
    assert got == world.cell_n and sum(got.values()) == 233 - 19 == 214
    assert list(got) == list(mc.CELL_ORDER)


def test_expected_frame_needs_the_seen_clips_in_the_retained_list(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mc, "PRACTICE", ("deadbeef", *world.practice[1:]))
    with pytest.raises(vc.Refused, match="seen clips are not in the retained list"):
        vc.expected_frame(world.docs, world.interim / "taps" / "taps_summary.csv")


def test_correct_weights_pass(world: World, capsys: pytest.CaptureFixture[str]) -> None:
    assert analysed(world).report["primary"]["metrics"]["n"] == 40
    code, _, err = run(["report", "--boot", "50"], capsys)
    assert (code, err) == (0, "")


def test_every_weight_doubled_is_refused(world: World, capsys: pytest.CaptureFixture[str]) -> None:
    df = world.read_export()
    prim = df["set"] == "primary"
    df.loc[prim, "weight"] = df.loc[prim, "weight"].astype(float).map(lambda x: str(x * 2))
    df.to_csv(world.docs / mc.EXPORT_NAME, index=False)
    code, out, err = run(["report", "--boot", "50"], capsys)
    assert code == 1 and out == "" and "the weights sum to 428, the frame size is 214" in err
    assert "cell slow/unflagged: weight x clips is " in err and "refused" in err
    assert (
        not (world.docs / vc.REPORT_NAME).exists() and not (world.interim / "validation").exists()
    )


def test_an_offsetting_weight_change_is_refused(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    df = world.read_export()
    a, b = "slow/unflagged", "mid/unflagged"
    prim = df["set"] == "primary"
    df.loc[prim & (df["cell"] == a), "weight"] = str((world.cell_n[a] + 1) / mc.ALLOCATION[a])
    df.loc[prim & (df["cell"] == b), "weight"] = str((world.cell_n[b] - 1) / mc.ALLOCATION[b])
    df.to_csv(world.docs / mc.EXPORT_NAME, index=False)
    assert sum(df.loc[prim, "weight"].astype(float)) == pytest.approx(214)  # the sum is still right
    code, out, err = run(["report", "--boot", "50"], capsys)
    assert code == 1 and out == ""
    assert (
        f"cell {a}: weight x clips is {world.cell_n[a] + 1}, the frame has {world.cell_n[a]}" in err
    )
    assert f"cell {b}:" in err and "the weights sum to" not in err
    assert not (world.docs / vc.REPORT_NAME).exists()


def test_the_frame_is_recomputed_from_the_repo_lists_not_the_export(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    gone = next(r for r in world.frame_refs if r not in world.clips)  # a frame clip, not exported
    for path in (world.docs / "qc_retained.csv", world.interim / "taps" / "taps_summary.csv"):
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        df[df["ref"] != gone].to_csv(path, index=False)
    code, _, err = run(["report", "--boot", "50"], capsys)
    assert code == 1 and "the frame size is 213" in err  # the export still says 214


# --------------------------------------------------------------------------- second matching pass

NARROW = {
    "peaks": np.array([15, 35, 55]), "left": np.array([13, 33, 53]), "right": np.array([17, 37, 57]),
}  # fmt: skip


def match2(manual: list[int], tol: int = 5, **kw: Any) -> list[dict[str, Any]]:
    n = NARROW
    return vc.match_taps(
        manual,
        n["peaks"],
        n["left"],
        n["right"],
        ["a", "b", "c"],
        kw.get("gaps", set()),
        0,
        70,
        tol,
    )


def test_tolerance_in_frames_is_0_15_s_at_the_nominal_rate() -> None:
    assert [vc.tolerance_frames(f) for f in (30.04, 29.97, 29.6051, 30.0)] == [5, 5, 5, 5]
    assert vc.tolerance_frames(59.96) == 9 and vc.tolerance_frames(25.0) == 4


def test_a_manual_tap_just_outside_its_window_becomes_tp_offset() -> None:
    rows = match2([15, 40, 55])  # 40 is 3 frames outside the window [33, 37] of the peak at 35
    off = [r for r in rows if r["kind"] == "TP_offset"]
    assert len(off) == 1 and off[0]["match_pass"] == 2
    assert (off[0]["manual_frame"], off[0]["detector_frame"], off[0]["offset_frames"]) == (
        40,
        35,
        5,
    )
    assert off[0]["detector_flags"] == "b" and off[0]["position"] == "last_interval"
    assert [r["kind"] for r in rows].count("TP") == 2 and not [
        r for r in rows if r["kind"] in ("FN", "FP")
    ]
    assert all(r["match_pass"] == 1 for r in rows if r["kind"] != "TP_offset")


def test_the_second_pass_can_be_switched_off_and_has_a_limit() -> None:
    assert sorted(r["kind"] for r in match2([15, 40, 55], tol=0)) == ["FN", "FP", "TP", "TP"]
    far = match2([15, 41, 55])  # 6 frames from the free peak: too far
    assert sorted(r["kind"] for r in far) == ["FN", "FP", "TP", "TP"]
    assert sorted(r["kind"] for r in match2([15, 41, 55], tol=6)).count("TP_offset") == 1


def test_a_double_bump_is_still_tp_plus_fn_when_the_free_peak_is_far() -> None:
    rows = vc.match_taps([15, 18, 55], PEAKS, LEFT, RIGHT, FLAGS, set(), 0, 70, 5)
    got = sorted(((r["kind"], r["manual_frame"], r["detector_frame"]) for r in rows), key=str)
    assert got == sorted(
        [("TP", 15, 15), ("FN", 18, 15), ("TP", 55, 55), ("FP", None, 35)], key=str
    )  # the free peak at 35 is 17 frames from the double bump: no TP_offset


def test_the_second_pass_is_one_to_one_and_nearest_first() -> None:
    peaks, left, right = np.array([20, 30, 70]), np.array([18, 28, 68]), np.array([22, 32, 72])
    flags = ["a", "b", "c"]
    both = vc.match_taps([24, 26, 70], peaks, left, right, flags, set(), 0, 80, 5)
    pairs = sorted(
        (r["manual_frame"], r["detector_frame"]) for r in both if r["kind"] == "TP_offset"
    )
    assert pairs == [(24, 20), (26, 30)]  # each manual tap takes its own nearest free peak
    one = vc.match_taps([17, 24, 70], peaks, left, right, flags, set(), 0, 80, 5)
    pairs = [(r["manual_frame"], r["detector_frame"]) for r in one if r["kind"] == "TP_offset"]
    assert pairs == [(17, 20)]  # distance 3 beats 4; the peak at 20 is not used twice
    assert [r["manual_frame"] for r in one if r["kind"] == "FN"] == [24]
    assert [r["detector_frame"] for r in one if r["kind"] == "FP"] == [30]
    assert sum(r["kind"] in ("TP", "TP_offset") and r["detector_frame"] == 20 for r in one) == 1


def test_no_tap_is_used_twice_in_either_pass() -> None:
    rng = np.random.default_rng(8)
    peaks, left, right = np.arange(10, 200, 12), np.arange(8, 200, 12), np.arange(12, 200, 12)
    flags = [""] * len(peaks)
    for _ in range(300):
        manual = sorted({int(x) for x in rng.integers(0, 205, int(rng.integers(0, 30)))})
        rows = vc.match_taps(manual, peaks, left, right, flags, set(), 0, 205, 5)
        used_m = [r["manual_frame"] for r in rows if r["manual_frame"] is not None]
        assert sorted(used_m) == manual  # every manual tap once
        hit = [r["detector_frame"] for r in rows if r["kind"] in ("TP", "TP_offset")]
        assert len(hit) == len(set(hit))  # a detector tap is hit once
        n_hit = len(hit)
        assert n_hit + sum(r["kind"] == "FP" for r in rows) == len(peaks)
        assert n_hit + sum(r["kind"] == "FN" for r in rows) == len(manual)
        assert {r["match_pass"] for r in rows if r["kind"] == "TP_offset"} <= {2}
        assert {r["match_pass"] for r in rows if r["kind"] != "TP_offset"} <= {1}


def add_narrow_clip(world: World, ref: str = "aaaa0001") -> None:
    frames = np.arange(0, 91)
    pd.DataFrame({"frame": frames, "t_s": frames / 30.0, "value": np.ones(len(frames))}).to_parquet(
        world.interim / "smoothed" / f"{ref}.parquet"
    )
    peaks = np.array([15, 35, 55, 75])
    pd.DataFrame(
        {"frame": peaks, "t_peak_s": peaks / 30.0, "trough_left_s": (peaks - 2) / 30.0,
         "trough_right_s": (peaks + 2) / 30.0, "flags": [""] * 4}
    ).to_parquet(world.interim / "taps" / f"{ref}.parquet")  # fmt: skip
    path = world.interim / "taps" / "taps_summary.csv"
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    row = df.iloc[0].copy()
    row["ref"], row["n_taps"], row["fps"] = ref, "4", "30.04"
    pd.concat([df, row.to_frame().T]).to_csv(path, index=False)


def test_tp_offset_is_counted_apart_in_the_clip_table(world: World) -> None:
    add_narrow_clip(world)
    row = vc.CountRow("aaaa0001", "R1", "primary", (15, 40, 55, 75), 4, 0, "3", "x", 1.0, 0, 90)
    clip, taps = vc.clip_result(row, vc.load_inputs(world.interim))
    assert (clip["tp"], clip["tp_offset"], clip["fn"], clip["fp"]) == (3, 1, 0, 0)
    assert clip["n_man"] == 4 and clip["d"] == 0  # counts, MAE and ICC are not affected by pairing
    kinds_ = {t["kind"]: t["match_pass"] for t in taps}
    assert kinds_ == {"TP": 1, "TP_offset": 2}
    assert [t["offset_frames"] for t in taps if t["kind"] == "TP_offset"] == [5]


def test_a_60_fps_clip_uses_a_wider_tolerance(world: World) -> None:
    add_narrow_clip(world, "aaaa0002")
    path = world.interim / "taps" / "taps_summary.csv"
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.loc[df["ref"] == "aaaa0002", "fps"] = "59.96"
    df.to_csv(path, index=False)
    row = vc.CountRow("aaaa0002", "R1", "primary", (15, 43, 55, 75), 4, 0, "3", "x", 1.0, 0, 90)
    clip, _ = vc.clip_result(row, vc.load_inputs(world.interim))
    assert (clip["tp_offset"], clip["fn"], clip["fp"]) == (1, 0, 0)  # 8 frames <= 9 at 60 fps
    add_narrow_clip(world, "aaaa0003")  # the same tap 8 frames away at 30 fps is too far
    row = vc.CountRow("aaaa0003", "R1", "primary", (15, 43, 55, 75), 4, 0, "3", "x", 1.0, 0, 90)
    clip, _ = vc.clip_result(row, vc.load_inputs(world.interim))
    assert (clip["tp_offset"], clip["fn"], clip["fp"]) == (0, 1, 1)


def test_tap_matching_totals_keep_tp_offset_apart(world: World) -> None:
    rep = analysed(world).report
    clips = analysed(world).clips
    prim = clips[clips["set"] == "primary"]
    tm = rep["primary"]["tap_matching"]
    assert set(tm) == {"tp", "tp_offset", "fn", "fp"} and set(
        rep["hard_set"]["tap_matching"]
    ) == set(tm)
    assert tm["tp"] + tm["tp_offset"] + tm["fn"] == prim["n_man"].sum()
    assert tm["tp"] + tm["tp_offset"] + tm["fp"] == prim["n_det"].sum()
    assert list(clips.columns) == list(vc.CLIP_COLUMNS) and "tp_offset" in clips.columns
    assert (clips["tp"] + clips["tp_offset"] + clips["fn"] == clips["n_man"]).all()
    assert (clips["tp"] + clips["tp_offset"] + clips["fp"] == clips["n_det"]).all()
    taps = analysed(world).taps
    assert list(taps.columns) == list(vc.TAP_COLUMNS) and set(taps["match_pass"]) == {1}


# --------------------------------------------------------------------------- plots


def legend_texts(ax: Any) -> list[str]:
    legend = ax.get_legend()
    assert legend is not None
    return [t.get_text() for t in legend.get_texts()]


def figure_data(n: int = 12) -> pd.DataFrame:
    cells = [c for c in mc.CELL_ORDER for _ in range(2)]
    man = np.arange(20, 20 + n)
    return pd.DataFrame(
        {"ref": [f"{i:08x}" for i in range(n)], "n_man": man, "n_det": man + np.tile([0, 1, -1], 4)[:n],
         "cell": cells[:n]}
    )  # fmt: skip


def test_bland_altman_legend_names_the_bias_and_the_limits() -> None:
    ax = vc.bland_altman_figure(figure_data(), "SYNTHETIC").axes[0]
    texts = legend_texts(ax)
    assert any(t.startswith("bias ") for t in texts)
    assert any(t.startswith("limits of agreement ") and " to " in t for t in texts)
    assert any("identical points" in t for t in texts)


def test_identical_points_are_drawn_larger_and_translucent() -> None:
    prim = pd.DataFrame(
        {"ref": ["a" * 8] * 5 + ["b" * 8], "n_man": [10] * 5 + [20], "n_det": [10] * 5 + [22],
         "cell": ["slow/unflagged"] * 6}
    )  # fmt: skip
    coll = vc.bland_altman_figure(prim, "t").axes[0].collections[0]
    sizes = sorted(float(s) for s in coll.get_sizes())  # type: ignore[attr-defined]
    assert len(sizes) == 2 and sizes[1] == 22 + 22 * 4 and sizes[0] == 22  # 5 identical points
    alpha = coll.get_alpha()
    assert alpha is not None and alpha < 1


def test_cell_plot_labels_show_n_and_mean_error() -> None:
    prim = figure_data()
    ax = vc.error_by_cell_figure(prim, "t").axes[0]
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert len(labels) == 6 and labels[0].startswith("slow/unflagged\nn=2, mean ")
    d = (prim.n_det - prim.n_man)[prim.cell == "slow/unflagged"].mean()
    assert labels[0].endswith(f"{d:+.2f}")
    assert "mean error" in legend_texts(ax)


def test_plot_text_is_only_words_numbers_and_hashed_refs() -> None:
    fig = vc.bland_altman_figure(figure_data(), "SYNTHETIC")
    words = " ".join(t.get_text() for a in fig.axes for t in [*a.texts, a.title])
    assert not RAW_ID.search(words) and "/Users" not in words
