"""Tests for preprocessing and heel strikes (GAIT-12). Synthetic walks with known landings."""

from __future__ import annotations

from pathlib import Path

import check_heel_strikes as chs
import numpy as np
import pytest
from carepd_format import L_ANKLE, PELVIS, R_ANKLE, load_config
from carepd_loader import Walk
from preprocess import (
    centre_on_pelvis,
    detect_foot,
    detect_heel_strikes,
    preprocess,
    smooth,
)

GAIT_DIR = Path(__file__).resolve().parents[1]
REAL_DATA = GAIT_DIR.parents[1] / "data" / "raw" / "CARE-PD"
STRIDE_S = 1.1
SWING_S = 0.4
AMP_M = 0.12


def swing_height(t: np.ndarray, first_swing: float, stride: float = STRIDE_S, amp: float = AMP_M):
    """Ankle height above the floor: a smooth bump during each swing, 0 in stance."""
    h = np.zeros_like(t)
    landings = []
    start = first_swing
    while start < t[-1]:
        u = (t - start) / SWING_S
        inside = (u >= 0) & (u <= 1)
        h[inside] = amp * np.sin(np.pi * u[inside]) ** 2
        landings.append(start + SWING_S)
        start += stride
    return h, np.array([x for x in landings if x <= t[-1]])


def make_walk(
    fps: float = 30,
    seconds: float = 6.0,
    stride: float = STRIDE_S,
    amp: float = AMP_M,
    bob: float = 0.02,
):
    """A walk of 17 joints: pelvis moving forward and bobbing, ankles swinging in turn."""
    t = np.arange(int(round(seconds * fps))) / fps
    joints = np.zeros((t.size, 17, 3))
    pelvis_y = 0.95 + bob * np.cos(4 * np.pi * t / stride)  # highest around mid-stance
    joints[:, :, 1] = pelvis_y[:, None]
    joints[:, :, 2] = 1.1 * t[:, None]  # forward along +z
    joints[:, PELVIS, 1] = pelvis_y
    left_h, left_land = swing_height(t, 0.3, stride, amp)
    right_h, right_land = swing_height(t, 0.3 + stride / 2, stride, amp)
    joints[:, L_ANKLE, 1] = left_h
    joints[:, R_ANKLE, 1] = right_h
    joints[:, L_ANKLE, 0], joints[:, R_ANKLE, 0] = 0.1, -0.1
    return joints, left_land, right_land


def rel_left_ankle(joints: np.ndarray, fps: float) -> np.ndarray:
    return preprocess(joints, fps)[:, L_ANKLE, 1]


# --------------------------------------------------------------------------- preprocessing
def test_centring_puts_the_pelvis_at_the_origin() -> None:
    joints, _, _ = make_walk()
    c = centre_on_pelvis(joints)
    assert np.allclose(c[:, PELVIS], 0)
    np.testing.assert_allclose(c[:, L_ANKLE, 1], joints[:, L_ANKLE, 1] - joints[:, PELVIS, 1])


def test_bad_walks_are_rejected() -> None:
    joints, _, _ = make_walk()
    with pytest.raises(ValueError, match="17, 3"):
        centre_on_pelvis(joints[:, :10])
    bad = joints.copy()
    bad[5, 3, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        preprocess(bad, 30)
    with pytest.raises(ValueError, match="too short"):
        smooth(joints[:10], 30)
    with pytest.raises(ValueError, match="cutoff"):
        smooth(joints, 10)  # 6 Hz is not below the 5 Hz Nyquist limit


def test_smoothing_does_not_shift_timing_and_removes_jitter() -> None:
    fps = 30
    t = np.arange(180) / fps
    bump = np.exp(-(((t - 3.0) / 0.15) ** 2))
    jitter = 0.05 * np.sin(2 * np.pi * 13 * t)  # 13 Hz, above the 6 Hz cutoff
    walk = np.zeros((t.size, 17, 3))
    walk[:, 3, 1] = bump + jitter
    out = smooth(walk, fps)[:, 3, 1]
    assert abs(int(np.argmax(out)) - int(np.argmax(bump))) <= 1  # zero-phase: no delay
    inner = slice(10, -10)  # filters ring a little at the very ends of a signal
    assert np.abs(out - bump)[inner].max() < 0.02


# --------------------------------------------------------------------------- heel strikes
@pytest.mark.parametrize("fps", [25, 30])
def test_strikes_match_known_landings_at_25_and_30_fps(fps: int) -> None:
    joints, left_land, _ = make_walk(fps=fps)
    found = detect_foot(rel_left_ankle(joints, fps), fps).strikes / fps
    inside = left_land[(left_land > 0.5) & (left_land < joints.shape[0] / fps - 0.1)]
    assert len(found) == len(inside)
    assert np.abs(found - inside).max() <= 2 / fps  # within 2 frames


def test_same_walk_in_seconds_gives_the_same_times_at_either_fps() -> None:
    times = []
    for fps in (25, 30):
        joints, _, _ = make_walk(fps=fps)
        times.append(detect_foot(rel_left_ankle(joints, fps), fps).strikes / fps)
    assert len(times[0]) == len(times[1])
    assert np.abs(times[0] - times[1]).max() <= 2 / 25


def test_plain_minimum_is_later_than_the_landing() -> None:
    joints, _, _ = make_walk()
    x = rel_left_ankle(joints, 30)
    onset = detect_foot(x, 30, "onset").strikes
    minimum = detect_foot(x, 30, "minimum").strikes
    n = min(len(onset), len(minimum))
    assert n >= 3 and np.all(minimum[-n:] - onset[-n:] >= 3)  # the minimum is mid-stance


def test_small_noise_adds_no_strikes() -> None:
    joints, _, _ = make_walk()
    clean = detect_foot(rel_left_ankle(joints, 30), 30).strikes
    noisy = joints + np.random.default_rng(0).normal(0, 0.001, joints.shape)
    assert len(detect_foot(rel_left_ankle(noisy, 30), 30).strikes) == len(clean)


def test_standing_still_has_no_strikes() -> None:
    joints, _, _ = make_walk(amp=0.0, bob=0.0)  # standing: no swing and no pelvis bob
    assert detect_foot(rel_left_ankle(joints, 30), 30).strikes.size == 0


def test_last_landing_is_kept_when_the_walk_ends_in_stance() -> None:
    fps = 30
    t = np.arange(150) / fps
    h, landings = swing_height(t, 0.3)
    found = detect_foot(h, fps).strikes / fps
    assert landings[-1] < t[-1] - 0.2  # the walk ends in stance
    assert abs(found[-1] - landings[-1]) <= 2 / fps


def test_walk_that_ends_mid_descent_has_not_landed() -> None:
    fps = 30
    t = np.arange(150) / fps
    h, landings = swing_height(t, 0.3)
    cut = int((landings[-1] - 0.1) * fps)  # stop 0.1 s before the last landing
    found = detect_foot(h[:cut], fps).strikes / fps
    assert found.size == len(landings) - 1 and found[-1] < landings[-1] - 0.5


def test_slow_drift_at_the_start_is_not_a_step() -> None:
    fps = 30
    t = np.arange(200) / fps
    h, _ = swing_height(t, 1.5)
    h = h + np.where(t < 1.2, 0.02 * (1 - t / 1.2), 0.0)  # starts 2 cm high and drifts down
    found = detect_foot(h, fps).strikes / fps
    assert found.min() > 1.5  # nothing before the first real swing


def test_two_dips_in_one_stance_count_once() -> None:
    fps = 30
    t = np.arange(200) / fps
    h, landings = swing_height(t, 0.3, stride=1.6)
    stance_wobble = 0.006 * np.sin(2 * np.pi * t / 0.6)  # a small rise within each stance
    found = detect_foot(h + np.where(h == 0, stance_wobble, 0), fps).strikes
    assert len(found) == len([x for x in landings if x < t[-1] - 0.05])
    assert np.all(np.diff(found) > 0.5 * fps)


def test_both_feet_and_stride_times() -> None:
    joints, left_land, right_land = make_walk(seconds=8.0)
    hs = detect_heel_strikes(joints, 30)
    assert hs.left.strikes.size >= 5 and hs.right.strikes.size >= 5
    np.testing.assert_allclose(np.median(hs.all_stride_times()), STRIDE_S, atol=1 / 30)
    assert hs.method == "onset"


# --------------------------------------------------------------------------- command line
def synthetic_walks(stride: float = STRIDE_S, bob: float = 0.02) -> list[Walk]:
    walks = []
    for i, (cohort, fps) in enumerate((("3DGait", 30), ("PD-GaM", 25))):
        joints, _, _ = make_walk(fps=fps, seconds=8.0, stride=stride, bob=bob)
        walks.append(Walk(cohort, f"secretsubj{i}", "w1", None, fps, joints, score=1, severity=1))
    return walks


def test_cli_summary_and_plot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(chs, "load_all", lambda *a, **k: (synthetic_walks(), []))
    rc = chs.main(["--out-dir", str(tmp_path)])
    text = capsys.readouterr().out
    assert rc == 0
    assert "3DGait" in text and "PD-GaM" in text and "1.10" in text
    assert "secretsubj" not in text  # walks are shown by hash only
    assert (tmp_path / "heel_strike_overlay.png").stat().st_size > 1000


def test_cli_flags_implausible_stride_times(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # No pelvis bob here: this test is about the exit code, not the detector. (A 4 cm bob in a
    # 2.6 s stride, which real walkers do not have, would add dips within the long stance.)
    walks = synthetic_walks(stride=2.6, bob=0.0)
    monkeypatch.setattr(chs, "load_all", lambda *a, **k: (walks, []))
    assert chs.main(["--no-plot"]) == 1
    assert "outside" in capsys.readouterr().out


def test_coordinate_method_needs_a_straight_walk_of_some_length() -> None:
    joints, _, _ = make_walk(seconds=8.0)
    straight = Walk("3DGait", "s", "w", None, 30, joints, score=0, severity=0)
    assert chs.coordinate_heel_strikes(straight) is not None
    turning = joints.copy()
    t = np.arange(joints.shape[0]) / 30
    turning[:, :, 0] = np.cos(t)[:, None]  # pelvis goes round in a circle
    turning[:, :, 2] = np.sin(t)[:, None]
    assert chs.coordinate_heel_strikes(Walk("3DGait", "s", "w", None, 30, turning, 0, 0)) is None
    short = joints.copy()
    short[:, :, 2] = 0.0  # walking on the spot
    assert chs.coordinate_heel_strikes(Walk("3DGait", "s", "w", None, 30, short, 0, 0)) is None


def test_cli_missing_files_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert chs.main(["--data-dir", str(tmp_path), "--no-plot"]) == 2
    assert "MISSING" in capsys.readouterr().out


# --------------------------------------------------------------------------- real data
@pytest.mark.skipif(not REAL_DATA.is_dir(), reason="CARE-PD data not downloaded")
def test_real_stride_times_are_plausible_and_agree_with_the_coordinate_method() -> None:
    from carepd_loader import load_all

    cfg = load_config(GAIT_DIR / "configs" / "carepd_format.toml")
    walks, _ = load_all(REAL_DATA, cfg)
    for cohort in ("3DGait", "T-SDU-PD", "BMCLab", "PD-GaM"):
        s = chs.summarise(cohort, [w for w in walks if w.cohort == cohort])
        assert 0.8 <= s.stride_s[1] <= 2.0
        assert s.stride_s_coordinate is not None
        assert abs(s.stride_s[1] - s.stride_s_coordinate) <= 0.05
        assert s.offset_ms is not None and abs(s.offset_ms[1]) <= 150  # within ~4 frames
