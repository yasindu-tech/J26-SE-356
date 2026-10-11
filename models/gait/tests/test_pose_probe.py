"""Tests for the MediaPipe clip check (GAIT-09, Check C). Synthetic data only: no patient video."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import check_mediapipe_clips as cli
import numpy as np
import pytest
from pose_probe import (
    CYCLES_PASS,
    DETECTION_PASS,
    L_ANKLE_IDX,
    L_HIP_IDX,
    LEG_IDX,
    N_LANDMARKS,
    R_ANKLE_IDX,
    R_HIP_IDX,
    PoseTrace,
    VideoInfo,
    count_cycles,
    evaluate,
    fixed_window,
    landmarks_from_result,
    motion_windows,
    plot_clip,
    probe_video,
    run_pose,
)

GAIT_DIR = Path(__file__).resolve().parents[1]
MODEL = GAIT_DIR / "artifacts" / "pose_landmarker_heavy.task"
FPS = 30.0


def sine(seconds: float, hz: float = 1.0, amp: float = 0.1, fps: float = FPS, phase: float = 0.0):
    t = np.arange(int(seconds * fps)) / fps
    return amp * np.sin(2 * np.pi * hz * t + phase)


def make_trace(
    left: np.ndarray, right: np.ndarray, detected: np.ndarray | None = None, fps: float = FPS
) -> PoseTrace:
    """A trace whose ankle heights (relative to the hips) are exactly ``left`` / ``right``."""
    n = len(left)
    det = np.ones(n, dtype=bool) if detected is None else detected
    vis = np.full((n, N_LANDMARKS), 0.9)
    image = np.full((n, N_LANDMARKS, 2), 0.5)
    world = np.zeros((n, N_LANDMARKS, 3))
    world[:, L_HIP_IDX, 1] = 0.0
    world[:, R_HIP_IDX, 1] = 0.0
    world[:, L_ANKLE_IDX, 1] = -left  # MediaPipe y points down
    world[:, R_ANKLE_IDX, 1] = -right
    image[:, L_ANKLE_IDX, 1] = 0.8 - 0.02 * left / 0.1
    image[:, R_ANKLE_IDX, 1] = 0.8 - 0.02 * right / 0.1
    vis[~det] = np.nan
    image[~det] = np.nan
    world[~det] = np.nan
    return PoseTrace(
        fps=fps, width=1920, height=1080, detected=det, visibility=vis, image=image, world=world
    )


# --------------------------------------------------------------------------- cycle counting
def test_regular_steps_are_counted() -> None:
    res = count_cycles(sine(6.0), FPS, abs_floor=0.02)
    assert res.cycles in (4, 5)  # six 1-second steps give 5 intervals, minus edge effects
    assert res.interval_cv is not None and res.interval_cv < 0.15


def test_flat_noisy_trace_has_no_cycles() -> None:
    rng = np.random.default_rng(0)
    res = count_cycles(rng.normal(0, 0.002, 180), FPS, abs_floor=0.02)
    assert res.cycles == 0


def test_tiny_movement_below_the_floor_is_not_a_step() -> None:
    assert count_cycles(sine(6.0, amp=0.004), FPS, abs_floor=0.02).cycles == 0


def test_all_missing_does_not_crash() -> None:
    res = count_cycles(np.full(120, np.nan), FPS, abs_floor=0.02)
    assert (res.n_minima, res.cycles, res.interval_cv) == (0, 0, None)


def test_short_gap_is_bridged_but_long_gap_splits() -> None:
    full = count_cycles(sine(8.0), FPS, abs_floor=0.02).cycles
    short = sine(8.0)
    short[100:106] = np.nan  # 0.2 s
    assert count_cycles(short, FPS, abs_floor=0.02).cycles == full
    long = sine(8.0)
    long[100:160] = np.nan  # 2 s
    split = count_cycles(long, FPS, abs_floor=0.02).cycles
    assert 2 <= split < full


# --------------------------------------------------------------------------- verdict
def test_good_clip_passes() -> None:
    m = evaluate(make_trace(sine(6.0), sine(6.0, phase=np.pi)))
    assert m.passed and m.reasons == ()
    assert m.detection_rate == 1.0 and m.visible_legs == 1.0
    assert m.gait_cycles >= CYCLES_PASS


def test_one_hidden_foot_still_passes_on_the_other() -> None:
    left = sine(6.0)
    right = np.full_like(left, np.nan)  # far foot never visible in a side view
    assert evaluate(make_trace(left, right)).passed


def test_low_detection_fails_with_a_reason() -> None:
    n = 180
    det = np.arange(n) % 2 == 0  # found in half the frames
    m = evaluate(make_trace(sine(6.0), sine(6.0), det))
    assert not m.passed
    assert any("of frames" in r for r in m.reasons)
    assert m.detection_rate < DETECTION_PASS


def test_no_steps_fails_even_with_full_detection() -> None:
    m = evaluate(make_trace(np.zeros(180), np.zeros(180)))
    assert not m.passed and any("gait cycles" in r for r in m.reasons)


def test_nothing_detected_fails_without_crashing() -> None:
    det = np.zeros(90, dtype=bool)
    m = evaluate(make_trace(sine(3.0), sine(3.0), det))
    assert not m.passed and m.visible_all == 0.0 and len(m.reasons) == 3
    assert m.clean_s == 0.0


def test_late_entry_fails_whole_clip_but_passes_once_in_view() -> None:
    n = 240  # 8 s; the walker is only found from 2 s onwards
    det = np.arange(n) >= 60
    m = evaluate(make_trace(sine(8.0), sine(8.0, phase=np.pi), det))
    assert m.detection_rate == 0.75 and not m.passed
    assert m.first_found_s == pytest.approx(2.0, abs=0.1)
    assert m.in_view_rate == 1.0 and m.longest_gap_s == 0.0
    assert m.passed_in_view


def test_gap_inside_the_walk_lowers_in_view_rate() -> None:
    det = np.ones(240, dtype=bool)
    det[100:160] = False  # 2 s lost mid-walk
    m = evaluate(make_trace(sine(8.0), sine(8.0, phase=np.pi), det))
    assert m.in_view_rate == pytest.approx(180 / 240)
    assert m.longest_gap_s == pytest.approx(2.0, abs=0.1)
    assert not m.passed_in_view


def test_nothing_found_has_no_first_time() -> None:
    m = evaluate(make_trace(sine(3.0), sine(3.0), np.zeros(90, dtype=bool)))
    assert m.first_found_s is None and m.in_view_rate == 0.0 and not m.passed_in_view


def junk_then_walk(good_s: float, junk_s: float = 4.0, good: bool = True) -> PoseTrace:
    """Junk (legs barely visible, big erratic ankle motion), then ``good_s`` of steady walking."""
    rng = np.random.default_rng(3)
    n_junk = int(junk_s * FPS)
    junk = rng.normal(0, 0.15, n_junk)
    walk = sine(good_s) if good else rng.normal(0, 0.003, int(good_s * FPS))
    left = np.concatenate([junk, walk])
    right = np.concatenate([rng.normal(0, 0.15, n_junk), walk])
    trace = make_trace(left, right)
    trace.visibility[:n_junk][:, list(LEG_IDX)] = 0.2  # "found", but the legs are not seen
    return trace


def test_junk_segment_does_not_produce_cycles() -> None:
    m = evaluate(junk_then_walk(8.0, good=False))  # flat after the junk
    assert m.raw_world_cycles >= CYCLES_PASS  # without the filter the junk looks like walking
    assert m.gait_cycles < CYCLES_PASS and not m.passed
    assert any("gait cycles" in r for r in m.reasons)


def test_cycles_after_junk_are_counted_only_on_clean_frames() -> None:
    m = evaluate(junk_then_walk(8.0))
    assert m.passed and m.clean_s == pytest.approx(8.0, abs=0.2)
    assert m.raw_world_cycles > m.gait_cycles
    first_clean = int(4.0 * FPS)
    assert all(i >= first_clean for i in m.world_left.minima)


def test_too_little_clean_tracking_fails_with_a_reason() -> None:
    m = evaluate(junk_then_walk(2.0))
    assert not m.passed and m.clean_s < 3.0
    assert any("clean tracking" in r for r in m.reasons)
    assert not m.passed_in_view


# --------------------------------------------------------------------------- landmarks
def fake_result(n: int = N_LANDMARKS, x: float = 0.5, y: float = 0.5) -> Any:
    img = [SimpleNamespace(x=x, y=y, visibility=0.9) for _ in range(n)]
    world = [SimpleNamespace(x=0.0, y=0.1, z=0.0) for _ in range(n)]
    return SimpleNamespace(pose_landmarks=[img], pose_world_landmarks=[world])


def test_landmarks_from_result() -> None:
    found = landmarks_from_result(fake_result())
    assert found is not None
    vis, xy, world = found
    assert vis.shape == (33,) and xy.shape == (33, 2) and world.shape == (33, 3)
    empty = SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[])
    assert landmarks_from_result(empty) is None
    assert landmarks_from_result(fake_result(n=20)) is None


class FakeLandmarker:
    def __init__(self, found: list[bool], x: float = 0.0, y: float = 1.0) -> None:
        self.found, self.x, self.y = found, x, y
        self.stamps: list[int] = []
        self.shapes: list[tuple[int, ...]] = []

    def detect_for_video(self, image: np.ndarray, ts: int) -> Any:
        i = len(self.stamps)
        self.stamps.append(ts)
        self.shapes.append(image.shape)
        if self.found[i]:
            return fake_result(x=self.x, y=self.y)
        return SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[])


def frames(n: int, h: int = 100, w: int = 200):
    for _ in range(n):
        yield np.zeros((h, w, 3), dtype=np.uint8)


def test_run_pose_maps_crop_back_to_the_full_frame() -> None:
    info = VideoInfo(fps=30, width=200, height=100, n_frames=3)
    lm = FakeLandmarker([True, False, True])
    win = [(50, 20, 150, 80)] * 3
    trace = run_pose(frames(3), info, lm, win, make_image=lambda a: a)
    assert trace.detected.tolist() == [True, False, True]
    assert lm.shapes[0] == (60, 100, 3)  # the crop, not the full frame
    # landmark (0.0, 1.0) in the crop is pixel (50, 80) of the full frame
    assert trace.image[0, 0].tolist() == pytest.approx([0.25, 0.8])
    assert np.isnan(trace.image[1]).all() and np.isnan(trace.world[1]).all()


def test_timestamps_strictly_increase_even_at_absurd_frame_rates() -> None:
    info = VideoInfo(fps=100000, width=200, height=100, n_frames=5)
    lm = FakeLandmarker([False] * 5)
    run_pose(frames(5), info, lm, make_image=lambda a: a)
    assert all(b > a for a, b in zip(lm.stamps, lm.stamps[1:], strict=False))


def test_zero_fps_falls_back_instead_of_dividing_by_zero() -> None:
    info = VideoInfo(fps=0, width=200, height=100, n_frames=2)
    trace = run_pose(frames(2), info, FakeLandmarker([False, False]), make_image=lambda a: a)
    assert trace.fps == 30.0


def test_frame_count_larger_than_the_window_list_reuses_the_last_window() -> None:
    info = VideoInfo(fps=30, width=200, height=100, n_frames=2)
    lm = FakeLandmarker([False] * 4)
    run_pose(frames(4), info, lm, [(0, 0, 100, 50)] * 2, make_image=lambda a: a)
    assert lm.shapes[-1] == (50, 100, 3)


def test_empty_video_raises() -> None:
    info = VideoInfo(fps=30, width=200, height=100, n_frames=0)
    with pytest.raises(ValueError, match="no frames"):
        run_pose(frames(0), info, FakeLandmarker([]), make_image=lambda a: a)


# --------------------------------------------------------------------------- crop windows
def test_fixed_window_in_pixels() -> None:
    assert fixed_window((0.25, 0.1, 0.75, 1.0), 1920, 1080) == (480, 108, 1440, 1080)


@pytest.mark.parametrize(
    "frac",
    [(0.5, 0.0, 0.4, 1.0), (0.0, 0.0, 1.2, 1.0), (-0.1, 0.0, 1.0, 1.0), (0.0, 0.0, 0.01, 1.0)],
)
def test_bad_crops_are_rejected(frac: tuple[float, float, float, float]) -> None:
    with pytest.raises(ValueError):
        fixed_window(frac, 1920, 1080)


def walker_video(
    n: int = 60, h: int = 90, w: int = 160
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Grey background with a small bright 'person' crossing it; also returns its centres."""
    gray = np.full((n, h, w), 100, dtype=np.uint8)
    centres: list[tuple[float, float]] = []
    for i in range(n):
        x0 = 10 + int(130 * i / (n - 1))
        gray[i, 40:70, x0 : x0 + 10] = 220
        centres.append((x0 + 5, 55))
    return gray, centres


def test_follow_crop_tracks_the_walker() -> None:
    gray, centres = walker_video()
    wins = motion_windows(gray, (1920, 1080))
    assert len(wins) == len(gray)
    for (x0, y0, x1, y1), (cx, cy) in list(zip(wins, centres, strict=True))[5:]:
        assert 0 <= x0 < x1 <= 1920 and 0 <= y0 < y1 <= 1080
        assert x0 <= cx * 12 <= x1 and y0 <= cy * 12 <= y1  # the walker is inside the window
        assert (y1 - y0) < 1080  # and the window is smaller than the frame
        assert (x1 - x0) / (y1 - y0) == pytest.approx(0.75, abs=0.02)


def test_follow_crop_with_no_motion_returns_the_whole_frame() -> None:
    still = np.full((20, 90, 160), 100, dtype=np.uint8)
    assert set(motion_windows(still, (1920, 1080))) == {(0, 0, 1920, 1080)}


# --------------------------------------------------------------------------- plot
def test_plot_is_written(tmp_path: Path) -> None:
    trace = make_trace(sine(5.0), sine(5.0, phase=1.0))
    out = tmp_path / "sub" / "clip.png"
    plot_clip(trace, evaluate(trace), "healthy (w-test)", out)
    assert out.is_file() and out.stat().st_size > 1000


# --------------------------------------------------------------------------- command line
def touch(path: Path, size: int = 10) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def patch_probe(monkeypatch: pytest.MonkeyPatch, trace: PoseTrace) -> None:
    info = VideoInfo(fps=trace.fps, width=1920, height=1080, n_frames=trace.n_frames)
    monkeypatch.setattr(cli, "probe_video", lambda *a, **k: (info, trace))


def test_missing_model_or_clip_exits_with_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clip = touch(tmp_path / "a.MOV")
    rc = cli.main(["--model", str(tmp_path / "none.task"), "--clip", f"healthy={clip}"])
    assert rc == 2 and "MISSING" in capsys.readouterr().out
    model = touch(tmp_path / "m.task")
    assert cli.main(["--model", str(model), "--clip", f"healthy={tmp_path / 'nope.MOV'}"]) == 2


def test_zero_byte_clip_counts_as_missing(tmp_path: Path) -> None:
    model = touch(tmp_path / "m.task")
    empty = touch(tmp_path / "empty.MOV", size=0)
    assert cli.main(["--model", str(model), "--clip", f"healthy={empty}"]) == 2


def test_bad_arguments_are_rejected() -> None:
    with pytest.raises(SystemExit):
        cli.main(["--clip", "no-equals-sign"])
    with pytest.raises(SystemExit):
        cli.main(["--crop", "1,2,3"])
    with pytest.raises(SystemExit):
        cli.main(["--follow", "--crop", "0,0,1,1"])


def test_passing_run_prints_table_without_file_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    model = touch(tmp_path / "m.task")
    clip = touch(tmp_path / "007_PD_01_ML_secret.MOV")
    patch_probe(monkeypatch, make_trace(sine(6.0), sine(6.0, phase=np.pi)))
    out_dir = tmp_path / "out"
    rc = cli.main(["--model", str(model), "--clip", f"mild={clip}", "--out-dir", str(out_dir)])
    text = capsys.readouterr().out
    assert rc == 0 and "plan's mark on the whole clip: 1/1 pass" in text
    assert "counted from when the person is found: 1/1 pass" in text
    assert "mild" in text and "w-" in text
    assert "secret" not in text and "007_PD" not in text  # file names never printed
    results = (out_dir / "check_c_results_full.json").read_text()
    assert "secret" not in results and json.loads(results)[0]["passed"] is True
    assert (out_dir / "check_c_mild_full.png").is_file()


def test_failing_run_exits_1_and_reports_zero_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    model = touch(tmp_path / "m.task")
    clip = touch(tmp_path / "a.MOV")
    patch_probe(monkeypatch, make_trace(np.zeros(120), np.zeros(120), np.zeros(120, dtype=bool)))
    rc = cli.main(
        ["--model", str(model), "--clip", f"severe={clip}", "--out-dir", str(tmp_path / "o")]
    )
    text = capsys.readouterr().out
    assert rc == 1 and "plan's mark on the whole clip: 0/1 pass" in text
    assert "counted from when the person is found: 0/1 pass" in text


# --------------------------------------------------------------------------- real MediaPipe
@pytest.mark.skipif(not MODEL.is_file(), reason="pose model not downloaded")
def test_real_model_runs_on_a_video_with_no_person(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("mediapipe")
    path = tmp_path / "noise.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 240))
    rng = np.random.default_rng(1)
    for _ in range(25):
        writer.write(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
    writer.release()
    info, trace = probe_video(path, MODEL)
    assert (info.width, info.height) == (320, 240) and trace.n_frames >= 20
    m = evaluate(trace)
    assert m.detection_rate == 0.0 and not m.passed  # nobody to find in noise
    _, followed = probe_video(path, MODEL, follow=True)  # the follow-crop path runs end to end
    assert followed.n_frames == trace.n_frames
