"""Tests for the joint-noise check (GAIT-20 addendum)."""

from __future__ import annotations

import json
from pathlib import Path

import evaluate_cohorts as ec
import make_folds as mf
import noise_check as nc
import numpy as np
import pandas as pd
import pytest
import train_xgboost as tx
from carepd_loader import Walk
from gait_extract import FEATURE_NAMES, extract_features, rejected
from test_evaluate_cohorts import FAST, cohort_table
from test_mediapipe_mapping import mediapipe_world
from test_preprocess import make_walk


# --------------------------------------------------------------------------- measuring jitter
def test_white_noise_sd_recovers_the_noise_and_ignores_the_walk() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(900) / 30
    walk = 0.15 * np.sin(2 * np.pi * 1.0 * t) + 0.05 * np.sin(2 * np.pi * 2.0 * t)
    assert nc.white_noise_sd(walk) < 2e-4  # a smooth signal has no white noise
    for sigma in (0.003, 0.01, 0.02):
        got = nc.white_noise_sd(walk + rng.normal(0, sigma, t.size))
        assert got == pytest.approx(sigma, rel=0.12)


def test_white_noise_sd_ignores_outliers_and_gaps() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.01, 600)
    clean = nc.white_noise_sd(x)
    x[::50] += 1.0  # a pose estimator's occasional jump
    x[100:140] = np.nan  # a lost person
    assert nc.white_noise_sd(x) == pytest.approx(clean, rel=0.15)


def test_white_noise_sd_is_nan_when_there_is_too_little_to_measure() -> None:
    assert np.isnan(nc.white_noise_sd(np.arange(10.0)))
    assert np.isnan(nc.white_noise_sd(np.full(100, np.nan)))
    with pytest.raises(ValueError, match="one coordinate"):
        nc.white_noise_sd(np.zeros((10, 3)))


def noisy_world(sigma_ankle: float, seed: int = 0) -> np.ndarray:
    joints, _, _ = make_walk(fps=30, seconds=10)
    world = mediapipe_world(joints)
    rng = np.random.default_rng(seed)
    for landmark in (27, 28):
        world[:, landmark] += rng.normal(0, sigma_ankle, world[:, landmark].shape)
    return world


def test_trace_jitter_finds_the_noisy_joints_in_h36m_order() -> None:
    names = ["pelvis", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee",
             "left_ankle"]  # fmt: skip
    got = nc.trace_jitter(noisy_world(0.01))
    assert got.shape == (17,) and np.isfinite(got).all()
    ankles = [names.index("right_ankle"), names.index("left_ankle")]
    assert got[ankles] == pytest.approx([0.01, 0.01], rel=0.15)
    quiet = [names.index(n) for n in ("pelvis", "right_hip", "right_knee", "left_knee")]
    assert (got[quiet] < 0.002).all()


def test_jitter_profile_is_the_median_over_traces() -> None:
    worlds = [noisy_world(s, seed=i) for i, s in enumerate((0.004, 0.01, 0.03))]
    profile = nc.jitter_profile(worlds)
    assert profile[3] == pytest.approx(0.01, rel=0.15)  # right ankle: the middle trace
    with pytest.raises(ValueError, match="no MediaPipe traces"):
        nc.jitter_profile([])


def test_read_worlds_reads_the_saved_traces(tmp_path: Path) -> None:
    world = noisy_world(0.01)
    for name in ("trace_a_30s.npz", "trace_b_30s.npz"):
        np.savez(tmp_path / name, world=world, fps=30.0)
    (tmp_path / "other.npz").write_bytes(b"not a trace")
    worlds = nc.read_worlds(tmp_path)
    assert len(worlds) == 2 and worlds[0].shape == world.shape


# --------------------------------------------------------------------------- the noise
def test_jitter_walk_scales_with_sigma_per_joint_and_is_reproducible() -> None:
    joints, _, _ = make_walk(fps=30, seconds=20)
    sigma = np.zeros(17)
    sigma[3], sigma[6] = 0.01, 0.02
    a = nc.jitter_walk(joints, sigma, np.random.default_rng(5))
    b = nc.jitter_walk(joints, sigma, np.random.default_rng(5))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, nc.jitter_walk(joints, sigma, np.random.default_rng(6)))
    diff = a - joints
    assert diff[:, 3].std() == pytest.approx(0.01, rel=0.1)
    assert diff[:, 6].std() == pytest.approx(0.02, rel=0.1)
    assert np.array_equal(diff[:, [0, 1, 2, 4, 5]], np.zeros((len(joints), 5, 3)))


def test_jitter_walk_with_no_noise_returns_the_walk_and_rejects_bad_input() -> None:
    joints, _, _ = make_walk()
    assert np.array_equal(nc.jitter_walk(joints, np.zeros(17), np.random.default_rng(0)), joints)
    with pytest.raises(ValueError, match="expected"):
        nc.jitter_walk(joints[:, :10], np.zeros(17), np.random.default_rng(0))
    with pytest.raises(ValueError, match="sigma"):
        nc.jitter_walk(joints, np.full(17, -1.0), np.random.default_rng(0))
    with pytest.raises(ValueError, match="sigma"):
        nc.jitter_walk(joints, np.zeros(5), np.random.default_rng(0))


# --------------------------------------------------------------------------- the features
def walk_table(n: int = 6) -> tuple[dict[str, Walk], pd.DataFrame]:
    """Walks from make_walk with slightly different strides, and the table built from them."""
    walks, rows = {}, []
    for i in range(n):
        joints, _, _ = make_walk(fps=30, seconds=8, stride=1.0 + 0.03 * i)
        w = Walk("A" if i % 2 else "B", f"s{i}", "w1", None, 30, joints, i % 3, i % 3)
        r = extract_features(w.joints, 30.0)
        assert r.ok
        walks[w.tag] = w
        rows.append((w.tag, w.cohort, f"s{i}", i % 3, *[r.features[f] for f in FEATURE_NAMES]))
    cols = ["walk", "cohort", "subject_key", "severity", *FEATURE_NAMES]
    return walks, pd.DataFrame(rows, columns=cols)


PROFILE = np.full(17, 0.004)


def test_level_zero_reproduces_the_table_exactly() -> None:
    walks, table = walk_table()
    lv = nc.noisy_features(walks, table, PROFILE, 0.0, 0, 0)
    assert lv.passed.all() and lv.max_gap == 0.0 and not lv.reasons
    assert lv.table[list(FEATURE_NAMES)].equals(table[list(FEATURE_NAMES)])


def test_noise_changes_the_inputs_of_passing_walks_and_is_reproducible() -> None:
    walks, table = walk_table()
    a = nc.noisy_features(walks, table, PROFILE, 1.0, 2, 7)
    b = nc.noisy_features(walks, table, PROFILE, 1.0, 2, 7)
    assert a.table.equals(b.table) and np.array_equal(a.passed, b.passed)
    assert a.passed.any() and a.max_gap > 0
    other_seed = nc.noisy_features(walks, table, PROFILE, 1.0, 2, 8)
    other_level = nc.noisy_features(walks, table, PROFILE, 1.0, 3, 7)
    for other in (other_seed, other_level):
        assert not other.table[list(FEATURE_NAMES)].equals(a.table[list(FEATURE_NAMES)])
    # the meta columns never change
    assert a.table[["walk", "cohort", "subject_key", "severity"]].equals(
        table[["walk", "cohort", "subject_key", "severity"]]
    )


def test_more_noise_means_more_change() -> None:
    walks, table = walk_table()
    knee = "knee_flexion_range_deg"  # the synthetic walk has no hip motion, so compare the knee
    change = {}
    for multiple in (0.1, 4.0):
        lv = nc.noisy_features(walks, table, PROFILE, multiple, 1, 0)
        change[multiple] = float(np.abs(lv.table[knee] - table[knee]).mean())
    assert 0 < change[0.1] < change[4.0]


def test_a_walk_that_no_longer_passes_is_counted_and_keeps_its_clean_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    walks, table = walk_table()
    clean = {id(w.joints) for w in walks.values()}
    real = nc.extract_features
    calls: list[int] = []

    def fails_on_the_first_noisy_walk(joints, fps):  # type: ignore[no-untyped-def]
        if id(joints) not in clean and not calls:
            calls.append(1)
            return rejected("stub: lost the heel strikes")
        return real(joints, fps)

    monkeypatch.setattr(nc, "extract_features", fails_on_the_first_noisy_walk)
    lv = nc.noisy_features(walks, table, np.full(17, 0.002), 1.0, 1, 0)
    lost = ~lv.passed
    assert lost.any() and sum(lv.reasons.values()) == int(lost.sum())
    assert "stub" in next(iter(lv.reasons))
    kept = lv.table.loc[lost, list(FEATURE_NAMES)]
    assert kept.equals(table.loc[lost, list(FEATURE_NAMES)])
    assert nc.rejection_table(table, [lv])["rejected"].sum() == int(lost.sum())


def test_a_table_walk_that_was_not_loaded_is_an_error() -> None:
    walks, table = walk_table()
    walks.pop(table["walk"].iloc[0])
    with pytest.raises(ValueError, match="not loaded"):
        nc.noisy_features(walks, table, PROFILE, 0.0, 0, 0)


# --------------------------------------------------------------------------- cross_validate
def test_test_table_replaces_the_test_rows_and_never_the_training_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = cohort_table(n_subjects=40)
    assignments = mf.make_assignments(table, 5, (0,))
    noisy = table.copy()
    noisy[list(FEATURE_NAMES)] += 100.0
    seen: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    real = tx.LEARNERS["xgboost"]

    def spy(train, test, y, w, seed, params):  # type: ignore[no-untyped-def]
        seen.append((train, test))
        return real(train, test, y, w, seed, params)

    monkeypatch.setitem(tx.LEARNERS, "xgboost", spy)
    tx.cross_validate(table, assignments, ["xgboost"], params=FAST, test_table=noisy)
    assert len(seen) == 5
    for train, test in seen:
        assert (train[list(FEATURE_NAMES)].to_numpy() < 50).all()  # clean
        assert (test[list(FEATURE_NAMES)].to_numpy() > 50).all()  # noisy
        assert not set(train["subject_key"]) & set(test["subject_key"])


def test_test_table_equal_to_the_table_changes_nothing_and_a_wrong_one_is_refused() -> None:
    table = cohort_table(n_subjects=40)
    assignments = mf.make_assignments(table, 5, (0,))
    plain = tx.cross_validate(table, assignments, ["xgboost"], params=FAST)
    same = tx.cross_validate(table, assignments, ["xgboost"], params=FAST, test_table=table.copy())
    assert np.array_equal(plain.proba["xgboost"][0], same.proba["xgboost"][0])
    with pytest.raises(ValueError, match="same walks"):
        tx.cross_validate(table, assignments, ["xgboost"], params=FAST, test_table=table.iloc[::-1])
    shifted = table.assign(subject_key=table["subject_key"].str.replace("s", "t"))
    with pytest.raises(ValueError, match="same walks"):
        tx.cross_validate(table, assignments, ["xgboost"], params=FAST, test_table=shifted)


# --------------------------------------------------------------------------- the analysis
def fake_levels(
    table: pd.DataFrame, noise: tuple[float, ...], dropped: dict[int, np.ndarray] | None = None
) -> list[nc.NoisyFeatures]:
    """Levels made by adding Gaussian noise (sd = ``noise``) to the inputs of the test table."""
    out = []
    for i, sd in enumerate(noise):
        noisy = table.copy()
        rng = np.random.default_rng(100 + i)
        noisy[list(FEATURE_NAMES)] = table[list(FEATURE_NAMES)].to_numpy() + rng.normal(
            0, sd, (len(table), len(FEATURE_NAMES))
        )
        passed = np.ones(len(table), dtype=bool)
        if dropped and i in dropped:
            passed[dropped[i]] = False
        out.append(nc.NoisyFeatures(float(i) / 2, noisy, passed, nc.Counter(), 0.0))
    return out


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return cohort_table(n_subjects=64, signal=2.0)


@pytest.fixture(scope="module")
def assignments(table: pd.DataFrame) -> pd.DataFrame:
    return mf.make_assignments(table, 5, (0, 1))


def test_noise_lowers_the_scores_and_level_zero_equals_the_clean_run(
    table: pd.DataFrame, assignments: pd.DataFrame
) -> None:
    levels = fake_levels(table, (0.0, 1.0, 6.0))
    scores, diffs, common = nc.noise_analysis(table, assignments, levels, 120, FAST)
    assert common.all()
    clean = tx.cross_validate(table, assignments, ["xgboost"], params=FAST)
    y = table["severity"].to_numpy(dtype=int)
    expected = ec.score_models(
        y, table["subject_key"].to_numpy(), {"x": clean.proba["xgboost"]}, 120
    )
    row = scores[(scores["model"] == "xgboost") & (scores["level"] == "0x")].iloc[0]
    assert row["bal_acc"] == pytest.approx(expected[0].iloc[0]["bal_acc"], abs=1e-12)
    by_level = scores[scores["model"] == "xgboost"].set_index("level")["bal_acc"]
    assert by_level["0x"] > by_level["0.5x"] > by_level["1x"]
    d = diffs[(diffs["metric"] == "bal_acc") & (diffs["model"] == "xgboost@1x")].iloc[0]
    assert d["versus"] == "xgboost@0x" and d["diff"] < 0 and d["hi"] < 0
    assert list(scores.columns[:2]) == ["level", "model"]
    assert {"macro_f1", "macro_f1_lo", "macro_f1_hi"} <= set(scores.columns)


def test_scores_are_taken_on_the_walks_that_pass_at_every_level(
    table: pd.DataFrame, assignments: pd.DataFrame
) -> None:
    lost_early = np.arange(0, len(table), 7)  # lost at the middle level only
    lost_late = np.arange(3, len(table), 11)  # lost at the last level only
    levels = fake_levels(table, (0.0, 0.5, 1.0), dropped={1: lost_early, 2: lost_late})
    scores, _, common = nc.noise_analysis(table, assignments, levels, 60, FAST, models=("xgboost",))
    assert not common[lost_early].any() and not common[lost_late].any()
    assert int(common.sum()) == len(table) - len(np.union1d(lost_early, lost_late))
    # the 0x row is scored on the same reduced set, so it differs from the full clean score
    full = nc.noise_analysis(
        table, assignments, fake_levels(table, (0.0, 0.5)), 60, FAST, models=("xgboost",)
    )[0]
    a = scores[(scores["level"] == "0x") & (scores["model"] == "xgboost")]["bal_acc"].iloc[0]
    b = full[(full["level"] == "0x") & (full["model"] == "xgboost")]["bal_acc"].iloc[0]
    assert a != b


def test_the_reference_level_must_come_first_and_something_must_pass(
    table: pd.DataFrame, assignments: pd.DataFrame
) -> None:
    levels = fake_levels(table, (0.0, 0.5))[::-1]
    with pytest.raises(ValueError, match="0x reference"):
        nc.noise_analysis(table, assignments, levels, 20, FAST)
    everything = {1: np.arange(len(table))}
    with pytest.raises(ValueError, match="no walk passes"):
        nc.noise_analysis(
            table, assignments, fake_levels(table, (0.0, 0.5), dropped=everything), 20, FAST
        )


def test_the_shuffled_label_control_stays_at_chance_at_every_level(
    table: pd.DataFrame, assignments: pd.DataFrame
) -> None:
    levels = fake_levels(table, (0.0, 1.0))
    scores, _, _ = nc.noise_analysis(table, assignments, levels, 20, FAST)
    assert nc.control_problems_by_level(scores) == []
    cheat = scores.copy()
    cheat.loc[(cheat["model"] == tx.CONTROL) & (cheat["level"] == "0.5x"), "bal_acc"] = 0.7
    problems = nc.control_problems_by_level(cheat)
    assert problems and "noise 0.5x" in problems[0]


def test_feature_shift_is_in_standard_deviations_of_the_input_on_the_common_set(
    table: pd.DataFrame,
) -> None:
    levels = fake_levels(table, (0.0, 0.5, 2.0))
    common = np.ones(len(table), dtype=bool)
    common[::5] = False
    out = nc.feature_shift(table, levels, common)
    assert list(out.columns) == ["level", "feature", "median_shift_sd"]
    assert len(out) == 3 * len(FEATURE_NAMES)
    assert (out[out["level"] == "0x"]["median_shift_sd"] == 0).all()
    # Gaussian noise of sd s has median |noise| = 0.6745 s; the inputs have sd about 1 or more
    spread = table[list(FEATURE_NAMES)].std(ddof=1).to_numpy()
    for level, sd in (("0.5x", 0.5), ("1x", 2.0)):
        got = out[out["level"] == level]["median_shift_sd"].to_numpy()
        assert got == pytest.approx(0.6745 * sd / spread, rel=0.25)


def test_rejection_table_counts_by_level_cohort_and_class(table: pd.DataFrame) -> None:
    levels = fake_levels(table, (0.0, 0.5), dropped={1: np.arange(30)})
    out = nc.rejection_table(table, levels)
    assert set(out["level"]) == {"0.5x"} and out["rejected"].sum() == 30
    assert set(out.columns) == {"level", "cohort", "severity", "rejected"}
    assert nc.rejection_table(table, fake_levels(table, (0.0,))).empty


# --------------------------------------------------------------------------- the command line
def test_missing_inputs_exit_with_code_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = nc.main(["--table", str(tmp_path / "nope.csv"), "--out-dir", str(tmp_path)])
    assert code == 2 and "MISSING" in capsys.readouterr().out


def cli_files(tmp_path: Path, table: pd.DataFrame, assignments: pd.DataFrame) -> list[str]:
    t, f, traces = tmp_path / "table.csv", tmp_path / "folds.csv", tmp_path / "traces"
    table.to_csv(t, index=False)
    assignments.to_csv(f, index=False)
    traces.mkdir()
    np.savez(traces / "trace_a.npz", world=noisy_world(0.01), fps=30.0)
    return [
        "--table", str(t), "--folds", str(f), "--trace-dir", str(traces),
        "--data-dir", str(tmp_path), "--bootstrap", "30", "--out-dir", str(tmp_path / "out"),
    ]  # fmt: skip


def test_the_command_line_writes_every_file(
    tmp_path: Path,
    table: pd.DataFrame,
    assignments: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(nc, "load_all", lambda *a, **k: ([], None))
    monkeypatch.setattr(nc, "MODELS", ("xgboost", tx.CONTROL))
    monkeypatch.setattr(tx, "XGB_PARAMS", FAST)
    monkeypatch.setattr(tx, "CHANCE_MARGIN", 0.15)  # one shuffled model on 64 subjects is loose
    levels = fake_levels(table, (0.0, 1.0, 1.5, 2.0, 3.0))
    monkeypatch.setattr(nc, "noisy_features", lambda *a, **k: levels[a[4]])
    code = nc.main(cli_files(tmp_path, table, assignments))
    out = capsys.readouterr().out
    assert code == 0, out
    for name in ("noise_scores.csv", "noise_differences.csv", "noise_rejections.csv",
                 "noise_profile.csv", "results.json"):  # fmt: skip
        assert (tmp_path / "out" / name).is_file()
    meta = json.loads((tmp_path / "out" / "results.json").read_text())
    assert meta["multiples"] == list(nc.MULTIPLES) and meta["control_problems"] == []
    assert "Jitter measured on MediaPipe output" in out and "all at chance" in out
    assert "right_ankle" in out


def test_features_at_zero_noise_that_differ_from_the_table_stop_the_run(
    tmp_path: Path,
    table: pd.DataFrame,
    assignments: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(nc, "load_all", lambda *a, **k: ([], None))
    levels = fake_levels(table, (0.0, 1.0, 1.5, 2.0, 3.0))
    levels[0].max_gap = 1e-3
    monkeypatch.setattr(nc, "noisy_features", lambda *a, **k: levels[a[4]])
    code = nc.main(cli_files(tmp_path, table, assignments))
    assert code == 1 and "rebuild the table" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()
