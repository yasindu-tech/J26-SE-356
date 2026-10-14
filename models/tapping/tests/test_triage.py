"""Tests for models/tapping/src/triage.py (task FT-18).

Synthetic only: made-up manifests and landmark tables, clip IDs such as ``CONTROL9001_DCHA``,
a fake capture. No real clip, no conftest.py (voice and gait have one).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hand_landmarks as hl  # noqa: E402
import overlay_gif as og  # noqa: E402
import triage as tr  # noqa: E402
from hubu_fis import pseudonym  # noqa: E402
from test_hand_scale import make_table, open_close  # noqa: E402
from test_overlay_gif import FakeCapture, make_pts  # noqa: E402
from test_overlay_gif import make_table as overlay_table  # noqa: E402

RAW_PARTS = ("CONTROL9000", "CONTROL9003", "CONTROL9013", "CONTROL90")


def manifest(rows: list[dict[str, Any]]) -> pd.DataFrame:
    base = {
        "flags": np.nan,
        "detection_rate": 1.0,
        "longest_missing_run": 0,
        "resolution": "1080x1920",
        "group": "CONTROL",  # labels: present but must never be used
        "updrs": 0,
    }
    out = []
    for i, row in enumerate(rows):
        clip = row.get("clip_id", f"CONTROL90{i:02d}_DCHA")
        out.append({**base, **row, "clip_id": clip, "ref": pseudonym(clip)})
    return pd.DataFrame(out)


def big_manifest() -> pd.DataFrame:
    """40 clips: 3 review-flagged, an info-only clip, low rates, long gaps, 3 resolutions."""
    rows: list[dict[str, Any]] = []
    resolutions = ["1080x1920"] * 24 + ["1920x1080"] * 10 + ["352x640"] * 6
    for i, res in enumerate(resolutions):
        rows.append({"clip_id": f"CONTROL90{i:02d}_DCHA", "resolution": res})
    rows[0]["flags"] = "low_detection;gaps_not_fillable"
    rows[0]["detection_rate"] = 0.5
    rows[1]["flags"] = "scale_tap_coupling"
    rows[2]["flags"] = "landmarks_out_of_frame"
    rows[3]["flags"] = "info_irregular_timing"  # informational only: not a review flag
    for k, i in enumerate(range(4, 11)):  # unflagged clips with falling rates
        rows[i]["detection_rate"] = 0.99 - 0.01 * k
        rows[i]["longest_missing_run"] = 3 * k  # and growing gaps (k = 6 is the longest)
    return manifest(rows)


# --------------------------------------------------------------------------- the review set


def test_review_flag_ignores_informational_flags_and_nan() -> None:
    assert tr.has_review_flag("low_detection;info_irregular_timing")
    assert not tr.has_review_flag("info_irregular_timing")
    assert not tr.has_review_flag(np.nan) and not tr.has_review_flag("")
    assert tr.has_any_flag("info_irregular_timing") and not tr.has_any_flag(np.nan)


def test_every_review_flagged_clip_is_in_the_set_with_its_reason() -> None:
    m = big_manifest()
    sel = tr.select_review_set(m)
    reasons = dict(zip(sel["ref"], sel["reasons"], strict=True))
    for i, flag in ((0, "low_detection"), (0, "gaps_not_fillable"), (1, "scale_tap_coupling")):
        assert f"flag:{flag}" in reasons[m.loc[i, "ref"]]
    assert "flag:landmarks_out_of_frame" in reasons[m.loc[2, "ref"]]
    assert m.loc[3, "ref"] not in reasons  # info-only flag: not flagged, not lowest, not control


def test_lowest_and_longest_are_taken_among_unflagged_clips_only() -> None:
    m = big_manifest()
    sel = tr.select_review_set(m, n_lowest=3, n_longest=2, n_control=0)
    reasons = dict(zip(sel["ref"], sel["reasons"], strict=True))
    lowest = [r for r, why in reasons.items() if "lowest_detection_unflagged" in why]
    longest = [r for r, why in reasons.items() if "longest_gap_unflagged" in why]
    assert sorted(lowest) == sorted(m.loc[[8, 9, 10], "ref"])  # 0.97, 0.96, 0.95
    assert sorted(longest) == sorted(m.loc[[9, 10], "ref"])  # runs 15 and 18
    assert m.loc[0, "ref"] not in lowest  # the 0.5 clip is flagged, so it is not "unflagged"
    for r in set(lowest) & set(longest):  # a clip in both lists appears once, both reasons
        assert reasons[r].count("lowest_detection_unflagged") == 1 and len(reasons[r]) == 2
    assert len(sel) == len(set(sel["ref"]))


def test_control_clips_are_unflagged_rate_one_and_never_flagged_ones() -> None:
    m = big_manifest()
    sel = tr.select_review_set(m, n_control=8)
    controls = [r for r, why in zip(sel["ref"], sel["reasons"], strict=True)
                if why == ["control_unflagged_rate_1"]]  # fmt: skip
    assert len(controls) == 8
    chosen = m[m["ref"].isin(controls)]
    assert (chosen["detection_rate"] == 1.0).all() and chosen["flags"].isna().all()
    assert m.loc[3, "ref"] not in controls  # the info-flagged clip has "a flag at all"
    flagged = set(m.loc[m["flags"].map(tr.has_any_flag), "ref"])
    assert not flagged & set(controls)


def test_control_sampling_is_reproducible_and_covers_every_resolution() -> None:
    m = big_manifest()
    a = tr.select_review_set(m, seed=18)
    b = tr.select_review_set(m, seed=18)
    pd.testing.assert_frame_equal(a, b)
    other = tr.select_review_set(m, seed=19)
    assert a["ref"].tolist() != other["ref"].tolist()  # the seed does matter
    controls = a[a["reasons"].map(lambda w: w == ["control_unflagged_rate_1"])]
    resolutions = set(m[m["ref"].isin(controls["ref"])]["resolution"])
    assert resolutions == {"1080x1920", "1920x1080", "352x640"}  # one per resolution at least


def test_control_sample_handles_small_pools_and_few_slots() -> None:
    pool = manifest([{"resolution": "a"}, {"resolution": "b"}, {"resolution": "b"}])
    assert len(tr.control_sample(pool, 1, 10)) == 3  # asks for more than exist
    assert len(tr.control_sample(pool, 1, 1)) == 1  # fewer slots than resolutions
    assert tr.control_sample(pool.iloc[:0], 1, 5) == []


def test_selection_never_looks_at_group_or_updrs() -> None:
    m = big_manifest()
    scrambled = m.copy()
    rng = np.random.default_rng(0)
    scrambled["group"] = rng.choice(["CONTROL", "ID"], len(m))
    scrambled["updrs"] = rng.integers(0, 4, len(m))
    pd.testing.assert_frame_equal(tr.select_review_set(m), tr.select_review_set(scrambled))
    pd.testing.assert_frame_equal(
        tr.select_review_set(m), tr.select_review_set(m.drop(columns=["group", "updrs"]))
    )
    with pytest.raises(tr.TriageError, match="lacks columns"):
        tr.select_review_set(m.drop(columns=["resolution"]))
    assert not any("group" in c or "updrs" in c for c in tr.SELECTION_COLUMNS)


# --------------------------------------------------------------------------- automatic checks


def moving_wrist(distances: Any, shifts: Any) -> pd.DataFrame:
    """A table whose whole hand is shifted right by ``shifts`` pixels per frame (hand size 100)."""
    df = make_table(distances)
    for i in range(hl.N_LANDMARKS):
        col = hl.landmark_column("x_px", i)
        df[col] = df[col] + np.asarray(shifts, dtype=float)
    return df


def test_wrist_jump_threshold_is_more_than_half_a_hand_size() -> None:
    n = 6
    steps = np.cumsum([0, 40, 0, 51, 0, 50])  # steps of 40, -, 51, -, 50 px; hand size 100
    j = tr.wrist_jumps(moving_wrist(np.full(n, 30.0), steps))
    assert j["hand_size_px"] == pytest.approx(100.0)
    assert j["wrist_jump_frames"] == 1  # only the 51 px step; 40 and exactly 50 are not "more"
    assert j["jump_frame"] == 3
    assert j["wrist_jump_after_gap"] == 0


def test_a_missing_frame_is_nan_never_a_jump_to_zero() -> None:
    d = np.full(8, 30.0)
    d[3] = np.nan
    steady = tr.wrist_jumps(moving_wrist(d, np.zeros(8)))
    assert steady["wrist_jump_frames"] == 0 and steady["wrist_jump_after_gap"] == 0
    assert steady["jump_frame"] is None
    shifts = np.array([0, 0, 0, 0, 80, 80, 80, 80])  # the hand moved 0.8 hand sizes in the gap
    moved = tr.wrist_jumps(moving_wrist(d, shifts))
    assert moved["wrist_jump_frames"] == 0 and moved["wrist_jump_after_gap"] == 1
    edges = np.full(8, 30.0)
    edges[[0, 7]] = np.nan  # gaps at the ends: nothing to compare with, and nothing at (0, 0)
    assert tr.wrist_jumps(moving_wrist(edges, np.zeros(8)))["wrist_jump_frames"] == 0


def test_all_nan_or_one_frame_gives_nan_not_zero() -> None:
    none = tr.wrist_jumps(make_table(np.full(5, np.nan)))
    assert np.isnan(none["wrist_jump_frames"]) and np.isnan(none["wrist_jump_after_gap"])
    one = tr.wrist_jumps(make_table([30.0]))
    assert np.isnan(one["wrist_jump_frames"])


def with_normalised(df: pd.DataFrame, x: float, y: float = 0.5) -> pd.DataFrame:
    for i in range(hl.N_LANDMARKS):
        df.loc[df["detected"], hl.landmark_column("x", i)] = 0.5
        df.loc[df["detected"], hl.landmark_column("y", i)] = y
    df.loc[df["detected"], hl.landmark_column("x", 5)] = x
    return df


def test_edge_fraction_counts_frames_with_any_landmark_near_the_edge() -> None:
    df = with_normalised(make_table([30.0, 30.0, np.nan, 30.0]), 0.5)
    assert tr.edge_fraction(df) == 0.0
    df.loc[0, hl.landmark_column("x", 5)] = 0.019  # inside the 2% margin
    df.loc[1, hl.landmark_column("y", 3)] = 0.981
    assert tr.edge_fraction(df) == pytest.approx(2 / 3)  # the NaN frame is not a denominator
    df.loc[0, hl.landmark_column("x", 5)] = 0.02  # exactly at the margin: not "within"
    assert tr.edge_fraction(df) == pytest.approx(1 / 3)
    assert np.isnan(tr.edge_fraction(make_table(np.full(3, np.nan))))


def test_distance_outlier_fraction_finds_gross_outliers_only() -> None:
    d = open_close(300)
    assert tr.distance_outlier_fraction(make_table(d)) == 0.0
    spiky = d.copy()
    spiky[[10, 11]] = 5000.0  # absurd aperture on two frames
    assert tr.distance_outlier_fraction(make_table(spiky)) == pytest.approx(2 / 300)
    nan = d.copy()
    nan[:296] = np.nan
    assert np.isnan(tr.distance_outlier_fraction(make_table(nan)))  # too few frames


def test_handedness_changes_count_only_between_detected_frames() -> None:
    df = make_table(np.full(6, 30.0))
    df["handedness_label"] = ["Left", "Right", "Right", "Left", "Left", "Right"]
    assert tr.handedness_changes(df) == 3
    df.loc[1, "detected"] = False  # the label of a frame without a hand is ignored
    assert tr.handedness_changes(df) == 3  # Left, Right, Left, Left, Right
    assert tr.automatic_checks(df)["handedness_changes"] == 3


def test_longest_gap_and_sheet_frames() -> None:
    d = np.array([1, 1, 0, 0, 1, 0, 0, 0, 1, 1], bool)
    assert tr.longest_gap(d) == (5, 7) and tr.longest_gap(np.ones(4, bool)) is None
    assert tr.longest_gap(np.array([0, 0, 1], bool)) == (0, 1)
    assert tr.longest_gap(np.array([1, 0, 0], bool)) == (1, 2)  # a gap at the very end
    frames = dict(tr.sheet_frames(np.ones(100, bool), None))
    assert len(frames) == 12 and 0 in frames and 99 in frames
    around = dict(tr.sheet_frames(d, None, n_even=4))
    assert around[4] == "before gap" and around[6] == "gap" and around[7] == "gap end"
    assert around[8] == "after gap"
    jump = dict(tr.sheet_frames(np.ones(50, bool), 20))
    assert jump[20] == "jump" and jump[19] == "before jump"
    assert tr.sheet_frames(np.array([], bool)) == []


def test_gif_window_is_centred_on_the_gap_and_stays_inside_the_clip() -> None:
    det = np.ones(600, bool)
    det[300:320] = False
    df = make_table(np.full(600, 30.0))
    df["detected"] = det
    start = tr.gif_window(df)
    assert 300 / 30 - tr.GIF_SECONDS / 2 - 0.1 < start < 310 / 30 - tr.GIF_SECONDS / 2 + 0.1
    df["detected"] = True
    assert tr.gif_window(df) == pytest.approx(5.0 - tr.GIF_SECONDS / 2)
    short = make_table(np.full(30, 30.0))
    assert tr.gif_window(short) == 0.0


def test_gifs_only_for_non_control_clips() -> None:
    assert tr.wants_gif(["flag:low_detection"]) and tr.wants_gif(["longest_gap_unflagged"])
    assert not tr.wants_gif(["control_unflagged_rate_1"])


def test_compare_runs_reports_new_lost_and_agreement() -> None:
    a = make_table(np.full(10, 30.0))
    b = make_table(np.full(10, 30.0))
    a.loc[[2, 3], "detected"] = False
    b.loc[[3, 4], "detected"] = False
    r = tr.compare_runs(a, b)
    assert r["newly_detected"] == [2] and r["lost"] == [4]
    assert r["detected_main"] == 8 and r["detected_alt"] == 8
    assert r["max_px_difference_on_shared_frames"] == pytest.approx(0.0)
    assert r["longest_gap_main"] == (2, 3) and r["longest_gap_alt"] == (3, 4)
    b["x_px08"] = b["x_px08"] + 25.0
    assert tr.compare_runs(a, b)["max_px_difference_on_shared_frames"] == pytest.approx(25.0)
    with pytest.raises(tr.TriageError):
        tr.compare_runs(a, make_table(np.full(9, 30.0)))


# --------------------------------------------------------------------------- end to end


def write_world(tmp_path: Path) -> dict[str, Path]:
    """Fourteen made-up clips: 1 flagged (with a gap), 5 low-rate/long-gap, 8 controls."""
    interim = tmp_path / "interim"
    landmarks = interim / "landmarks"
    names = [f"CONTROL90{i:02d}_DCHA" for i in range(14)]
    rows: list[dict[str, Any]] = []
    for i, name in enumerate(names):
        df = overlay_table(60, no_hand=range(20, 26) if i == 0 else ())
        df["handedness_label"] = "Right"
        hl.write_landmarks(df, landmarks / f"{name}.parquet")
        row: dict[str, Any] = {"clip_id": name}
        if i == 0:
            row.update(flags="low_detection", detection_rate=0.9, longest_missing_run=6)
        elif i <= 5:
            row.update(detection_rate=0.99 - 0.001 * i, longest_missing_run=i)
        rows.append(row)
    man = interim / "tracking_manifest.csv"
    manifest(rows).to_csv(man, index=False)
    return {"landmarks": landmarks, "manifest": man, "interim": interim,
            "out": interim / "triage", "flagged": Path(names[0])}  # fmt: skip


def run(world: dict[str, Path], *extra: str, monkeypatch: pytest.MonkeyPatch) -> int:
    pts = make_pts(60)
    monkeypatch.setattr(og, "find_video", lambda ref, folder: Path("fake.mp4"))
    monkeypatch.setattr(og, "open_video", lambda path: FakeCapture(pts))
    return tr.main(
        [
            "--manifest", str(world["manifest"]),
            "--landmarks-dir", str(world["landmarks"]),
            "--out-dir", str(world["out"]),
            "--interim-root", str(world["interim"]),
            *extra,
        ]
    )  # fmt: skip


def test_the_run_writes_sheets_gifs_and_tables_with_hashes_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    world = write_world(tmp_path)
    assert run(world, monkeypatch=monkeypatch) == 0
    captured = capsys.readouterr()
    out = world["out"]
    names = sorted(p.name for p in out.iterdir())
    flagged = pseudonym("CONTROL9000_DCHA")
    lowest = [pseudonym(f"CONTROL90{i:02d}_DCHA") for i in range(1, 6)]
    controls = [pseudonym(f"CONTROL90{i:02d}_DCHA") for i in range(6, 14)]
    assert all(f"{r}_overlay.gif" in names for r in lowest)  # lowest / longest-gap: GIF
    assert f"{flagged}_sheet.png" in names and f"{flagged}_overlay.gif" in names  # flagged: GIF
    for ref in controls:
        assert f"{ref}_sheet.png" in names and f"{ref}_overlay.gif" not in names  # no GIF
    assert {"review_set.csv", "triage_table.csv"} <= set(names)
    assert (out / f"{flagged}_sheet.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    everything = captured.out + captured.err + " ".join(names)
    everything += "".join((out / n).read_text() for n in ("review_set.csv", "triage_table.csv"))
    assert not any(raw in everything for raw in RAW_PARTS)
    table = pd.read_csv(out / "triage_table.csv", keep_default_na=False)
    assert list(table.columns) == list(tr.TABLE_COLUMNS)
    assert (table["visual_verdict"] == "").all()  # left empty for a person
    assert "group" not in table.columns and "updrs" not in table.columns
    row = table.set_index("ref").loc[flagged]
    assert row["longest_missing_run"] == 6 and "flag:low_detection" in row["reasons"]
    assert row["handedness_changes"] == 0


def test_exit_codes_and_output_guard(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    world = write_world(tmp_path)
    assert run({**world, "out": tmp_path / "elsewhere"}, monkeypatch=monkeypatch) == 2
    assert not (tmp_path / "elsewhere").exists()
    assert run(world, "--alt-conf", "0.3", "--alt-out-dir", str(world["landmarks"]),
               monkeypatch=monkeypatch) == 2  # fmt: skip
    assert "must not be the main landmarks folder" in capsys.readouterr().err
    world["manifest"].write_text("")
    assert run(world, monkeypatch=monkeypatch) == 1
    assert capsys.readouterr().err.startswith("FAILED:")
    world["manifest"].unlink()
    assert run(world, monkeypatch=monkeypatch) == 2
    assert not world["out"].exists()


def test_a_clip_that_cannot_be_read_fails_by_hash_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    world = write_world(tmp_path)
    (world["landmarks"] / "CONTROL9002_DCHA.parquet").write_bytes(b"not a parquet file")
    assert run(world, monkeypatch=monkeypatch) == 1
    err = capsys.readouterr().err
    assert err.startswith("FAILED: clip ") and pseudonym("CONTROL9002_DCHA") in err
    assert not any(raw in err for raw in RAW_PARTS)
