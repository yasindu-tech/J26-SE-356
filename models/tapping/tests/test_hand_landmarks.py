"""Tests for models/tapping/src/hand_landmarks.py (task FT-11).

Synthetic data only: small made-up frames, a fake capture and a fake landmarker passed in
through the factories, so the logic is tested without MediaPipe and without any real clip.
Clip names such as ``CONTROL9001_DCHA`` are made up; the numbers are not real participants.
Two tests at the end use the real MediaPipe model and skip when it is not downloaded.

No conftest.py on purpose (voice and gait each have one); ``src`` goes on ``sys.path`` here.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hand_landmarks as hl  # noqa: E402
from hubu_fis import pseudonym  # noqa: E402

N = hl.N_LANDMARKS
STEM = "CONTROL9001_DCHA"
MODEL_FILE = hl.DEFAULT_MODEL


# --------------------------------------------------------------------------- fakes


def bgr_frame(height: int = 48, width: int = 32, bgr: tuple[int, int, int] = (10, 20, 30)) -> Any:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


class FakeCapture:
    """The few cv2.VideoCapture methods the tool uses."""

    def __init__(
        self,
        frames: list[Any],
        *,
        fps: float = 30.0,
        pts: list[float] | None = None,
        container_count: int | None = None,
        auto_rotation_works: bool = True,
        reported_size: tuple[int, int] | None = None,
        opened: bool = True,
    ) -> None:
        self.frames = frames
        self.pts = pts
        self.opened = opened
        self.auto_rotation_works = auto_rotation_works
        h, w = (frames[0].shape[:2]) if frames else (0, 0)
        self.props: dict[int, float] = {
            cv2.CAP_PROP_FPS: fps,
            cv2.CAP_PROP_FRAME_COUNT: len(frames) if container_count is None else container_count,
            cv2.CAP_PROP_FRAME_WIDTH: float(reported_size[0] if reported_size else w),
            cv2.CAP_PROP_FRAME_HEIGHT: float(reported_size[1] if reported_size else h),
            cv2.CAP_PROP_ORIENTATION_AUTO: 0.0,
            cv2.CAP_PROP_ORIENTATION_META: 0.0,
        }
        self.set_calls: list[tuple[int, float]] = []
        self.position = 0
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def set(self, prop: int, value: float) -> bool:
        self.set_calls.append((prop, value))
        if prop == cv2.CAP_PROP_ORIENTATION_AUTO and self.auto_rotation_works:
            self.props[prop] = float(value)
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_POS_MSEC:
            if self.pts is None:
                return (self.position - 1) * 1000.0 / self.props[cv2.CAP_PROP_FPS]
            return self.pts[self.position - 1]
        return self.props.get(prop, 0.0)

    def read(self) -> tuple[bool, Any]:
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame.copy()

    def release(self) -> None:
        self.released = True


def hand_result(
    base_x: float = 0.5, base_y: float = 0.25, label: str = "Left", score: float = 0.9
) -> Any:
    """A MediaPipe-shaped result for one hand; landmark i sits at x = base_x + i / 1000."""
    pts = [SimpleNamespace(x=base_x + i / 1000, y=base_y + i / 2000, z=i / 100) for i in range(N)]
    world = [SimpleNamespace(x=i / 100, y=-i / 100, z=i / 50) for i in range(N)]
    return SimpleNamespace(
        hand_landmarks=[pts],
        hand_world_landmarks=[world],
        handedness=[[SimpleNamespace(category_name=label, score=score)]],
    )


def no_hand_result() -> Any:
    return SimpleNamespace(hand_landmarks=[], hand_world_landmarks=[], handedness=[])


class FakeLandmarker:
    def __init__(self, results: list[Any]) -> None:
        self.results = results
        self.calls: list[tuple[Any, int]] = []
        self.closed = False
        self.options: dict[str, Any] = {}

    def detect_for_video(self, rgb: Any, timestamp_ms: int) -> Any:
        self.calls.append((rgb.copy(), timestamp_ms))
        return self.results[(len(self.calls) - 1) % len(self.results)]

    def close(self) -> None:
        self.closed = True


def run(
    frames: list[Any],
    results: list[Any],
    tmp_path: Path,
    *,
    stem: str = STEM,
    capture: FakeCapture | None = None,
    **options: Any,
) -> tuple[pd.DataFrame, FakeCapture, FakeLandmarker]:
    capture = capture or FakeCapture(frames)
    landmarker = FakeLandmarker(results)

    def make_landmarker(model_path: Path, **opts: Any) -> FakeLandmarker:
        landmarker.options = {"model_path": model_path, **opts}
        return landmarker

    df = hl.track_clip(
        tmp_path / f"{stem}.mp4",
        tmp_path / "model.task",
        landmarker_factory=make_landmarker,
        capture_factory=lambda path: capture,
        **options,
    )
    return df, capture, landmarker


# --------------------------------------------------------------------------- timestamps


@pytest.mark.parametrize("fps", [30.0, 29.97, 60.0, 30.04, 59.96, 25.0, 1000.0])
def test_timestamps_are_strictly_increasing_integers(fps: float) -> None:
    last, stamps = -1, []
    for i in range(6000):
        last = hl.next_timestamp_ms(i, fps, last)
        stamps.append(last)
    assert stamps[0] == 0
    assert all(isinstance(t, int) for t in stamps)
    assert all(b > a for a, b in zip(stamps, stamps[1:], strict=False))


def test_timestamps_follow_the_clip_clock() -> None:
    last, stamps = -1, []
    for i in range(5):
        last = hl.next_timestamp_ms(i, 30.0, last)
        stamps.append(last)
    assert stamps == [0, 33, 67, 100, 133]


def test_timestamp_guard_never_repeats_or_goes_back() -> None:
    """If rounding would repeat or step back, the value is raised to previous + 1."""
    assert hl.next_timestamp_ms(5, 30.0, last_ms=500) == 501
    assert hl.next_timestamp_ms(5, 30.0, last_ms=167) == 168  # round(166.7) = 167 is not above 167
    assert hl.next_timestamp_ms(5, 30.0, last_ms=100) == 167  # normal case is untouched


@pytest.mark.parametrize("fps", [0.0, -30.0, float("nan"), float("inf"), 1000.5, 5000.0])
def test_invalid_fps_is_rejected(fps: float) -> None:
    with pytest.raises(hl.HandLandmarksError, match="frame rate"):
        hl.check_fps(fps)


def test_track_clip_rejects_a_zero_fps_clip(tmp_path: Path) -> None:
    capture = FakeCapture([bgr_frame()], fps=0.0)
    with pytest.raises(hl.HandLandmarksError, match="frame rate"):
        run([bgr_frame()], [no_hand_result()], tmp_path, capture=capture)


def test_timestamps_reach_the_landmarker_and_the_table(tmp_path: Path) -> None:
    frames = [bgr_frame() for _ in range(6)]
    df, _, landmarker = run(
        frames, [no_hand_result()], tmp_path, capture=FakeCapture(frames, fps=29.97)
    )
    sent = [ts for _, ts in landmarker.calls]
    assert sent == df["timestamp_ms"].tolist()
    assert all(b > a for a, b in zip(sent, sent[1:], strict=False))
    assert df["timestamp_ms"].dtype == np.int64


# --------------------------------------------------------------------------- pts_ms


def test_pts_ms_is_the_container_time_and_differs_from_the_index_clock(tmp_path: Path) -> None:
    frames = [bgr_frame() for _ in range(5)]
    pts = [0.0, 33.3, 91.4, 124.6, 158.0]  # an irregular clip: one long gap
    df, _, landmarker = run(
        frames, [no_hand_result()], tmp_path, capture=FakeCapture(frames, pts=pts)
    )
    assert df["pts_ms"].tolist() == pts
    assert df["timestamp_ms"].tolist() == [0, 33, 67, 100, 133]
    assert [ts for _, ts in landmarker.calls] == df[
        "timestamp_ms"
    ].tolist()  # MediaPipe not given pts
    assert not df["pts_ms"].equals(df["timestamp_ms"].astype(float))


@pytest.mark.parametrize("bad", [float("nan"), -1.0, float("inf")])
def test_missing_container_time_is_nan_not_zero(tmp_path: Path, bad: float) -> None:
    frames = [bgr_frame() for _ in range(3)]
    df, _, _ = run(
        frames, [no_hand_result()], tmp_path, capture=FakeCapture(frames, pts=[0.0, bad, 66.0])
    )
    assert df["pts_ms"].isna().tolist() == [False, True, False]


# --------------------------------------------------------------------------- landmarks


def test_a_frame_with_no_hand_is_nan_never_zero(tmp_path: Path) -> None:
    frames = [bgr_frame() for _ in range(4)]
    results = [hand_result(), no_hand_result(), no_hand_result(), hand_result()]
    df, _, _ = run(frames, results, tmp_path)
    assert df["detected"].tolist() == [True, False, False, True]
    missing = df.loc[~df["detected"], list(hl.LANDMARK_COLUMNS)]
    assert missing.isna().all().all()
    assert not (missing == 0).any().any()
    assert df.loc[~df["detected"], "handedness_label"].isna().all()
    assert df.loc[~df["detected"], "handedness_score"].isna().all()
    assert df.loc[df["detected"], list(hl.LANDMARK_COLUMNS)].notna().all().all()


def test_there_are_exactly_21_landmarks_with_all_coordinates(tmp_path: Path) -> None:
    df, _, _ = run([bgr_frame()], [hand_result()], tmp_path)
    assert list(df.columns) == list(hl.COLUMNS)
    for prefix in ("x", "y", "z", "x_px", "y_px", "wx", "wy", "wz"):
        names = [c for c in df.columns if c.startswith(prefix) and c[len(prefix) :].isdigit()]
        assert names == [f"{prefix}{i:02d}" for i in range(21)]
    assert len(df.columns) == 6 + 8 * 21
    summary = hl.summarize(df, df.attrs["hand_landmarks"])
    assert (
        summary["landmarks_per_detected_frame_min"],
        summary["landmarks_per_detected_frame_max"],
    ) == (21, 21)


def test_landmark_index_maps_to_the_right_column(tmp_path: Path) -> None:
    df, _, _ = run([bgr_frame()], [hand_result(base_x=0.2)], tmp_path)
    row = df.iloc[0]
    assert row["x00"] == pytest.approx(0.2)
    assert row["x20"] == pytest.approx(0.22)
    assert row["z04"] == pytest.approx(0.04)
    assert row["wx05"] == pytest.approx(0.05) and row["wy05"] == pytest.approx(-0.05)
    assert row["wz10"] == pytest.approx(0.2)


def test_wrong_landmark_count_fails(tmp_path: Path) -> None:
    bad = hand_result()
    bad.hand_landmarks[0] = bad.hand_landmarks[0][:20]
    with pytest.raises(hl.HandLandmarksError, match="expected 21 landmarks, got 20"):
        run([bgr_frame()], [bad], tmp_path)


def test_handedness_is_stored_as_reported(tmp_path: Path) -> None:
    frames = [bgr_frame() for _ in range(3)]
    df, _, _ = run(
        frames,
        [
            hand_result(label="Left", score=0.8),
            hand_result(label="Right", score=0.6),
            no_hand_result(),
        ],
        tmp_path,
    )
    assert df["handedness_label"].tolist()[:2] == ["Left", "Right"]
    assert df["handedness_score"].tolist()[:2] == [0.8, 0.6]


# --------------------------------------------------------------------------- pixels and rotation


def test_pixels_use_the_displayed_size_from_the_first_frame(tmp_path: Path) -> None:
    """A portrait 352x640 frame: x_px = x * 352 and y_px = y * 640, not the other way round."""
    frame = bgr_frame(height=640, width=352)
    df, _, _ = run([frame], [hand_result(base_x=0.5, base_y=0.25)], tmp_path)
    assert (
        df.attrs["hand_landmarks"]["width"] == 352 and df.attrs["hand_landmarks"]["height"] == 640
    )
    assert df["x_px00"].iloc[0] == pytest.approx(0.5 * 352)
    assert df["y_px00"].iloc[0] == pytest.approx(0.25 * 640)
    assert df["x_px20"].iloc[0] == pytest.approx((0.5 + 0.02) * 352)
    assert df["y_px20"].iloc[0] == pytest.approx((0.25 + 0.01) * 640)


def test_pixels_are_nan_when_nothing_is_detected(tmp_path: Path) -> None:
    df, _, _ = run([bgr_frame()], [no_hand_result()], tmp_path)
    assert df[[f"x_px{i:02d}" for i in range(21)]].isna().all().all()


def test_auto_rotation_is_switched_on_and_read_back(tmp_path: Path) -> None:
    df, capture, _ = run([bgr_frame()], [no_hand_result()], tmp_path)
    assert (cv2.CAP_PROP_ORIENTATION_AUTO, 1) in capture.set_calls
    assert df.attrs["hand_landmarks"]["orientation_auto"] == 1
    assert capture.released


def test_auto_rotation_that_cannot_be_enabled_fails(tmp_path: Path) -> None:
    capture = FakeCapture([bgr_frame()], auto_rotation_works=False)
    with pytest.raises(hl.HandLandmarksError, match="auto-rotation"):
        run([bgr_frame()], [no_hand_result()], tmp_path, capture=capture)
    assert capture.released


def test_frame_size_that_disagrees_with_opencv_fails(tmp_path: Path) -> None:
    frame = bgr_frame(height=48, width=32)
    capture = FakeCapture([frame], reported_size=(48, 32))  # width and height swapped
    with pytest.raises(hl.HandLandmarksError, match="rotation may not have been applied"):
        run([frame], [no_hand_result()], tmp_path, capture=capture)


def test_frame_size_change_inside_a_clip_fails(tmp_path: Path) -> None:
    frames = [bgr_frame(48, 32), bgr_frame(32, 48)]
    with pytest.raises(hl.HandLandmarksError, match="frame size changed"):
        run(frames, [no_hand_result()], tmp_path)


def test_frames_are_given_to_the_landmarker_as_rgb(tmp_path: Path) -> None:
    _, _, landmarker = run([bgr_frame(bgr=(10, 20, 30))], [no_hand_result()], tmp_path)
    rgb = landmarker.calls[0][0]
    assert tuple(rgb[0, 0]) == (30, 20, 10)


# --------------------------------------------------------------------------- parameters, counts, metadata


def test_detector_parameters_are_passed_through_and_recorded(tmp_path: Path) -> None:
    df, _, landmarker = run(
        [bgr_frame()],
        [no_hand_result()],
        tmp_path,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.2,
        min_tracking_confidence=0.1,
    )
    assert landmarker.options["num_hands"] == 2
    assert landmarker.options["min_hand_detection_confidence"] == 0.3
    detector = df.attrs["hand_landmarks"]["detector"]
    assert detector == {
        "running_mode": "VIDEO",
        "num_hands": 2,
        "min_hand_detection_confidence": 0.3,
        "min_hand_presence_confidence": 0.2,
        "min_tracking_confidence": 0.1,
    }
    assert landmarker.closed


def test_defaults_are_one_hand_and_half_confidence(tmp_path: Path) -> None:
    _, _, landmarker = run([bgr_frame()], [no_hand_result()], tmp_path)
    assert landmarker.options["num_hands"] == 1
    assert {landmarker.options[k] for k in landmarker.options if k.startswith("min_")} == {0.5}


def test_container_frame_count_mismatch_is_recorded_not_hidden(tmp_path: Path) -> None:
    frames = [bgr_frame() for _ in range(4)]
    df, _, _ = run(
        frames, [no_hand_result()], tmp_path, capture=FakeCapture(frames, container_count=6)
    )
    meta = df.attrs["hand_landmarks"]
    assert (meta["frames_decoded"], meta["frames_container"]) == (4, 6)
    assert len(df) == 4


def test_a_clip_with_no_decodable_frames_fails(tmp_path: Path) -> None:
    with pytest.raises(hl.HandLandmarksError, match="no frames could be decoded"):
        run([], [no_hand_result()], tmp_path, capture=FakeCapture([]))


def test_unopenable_capture_fails_with_the_hash_not_the_name(tmp_path: Path) -> None:
    capture = FakeCapture([bgr_frame()], opened=False)
    with pytest.raises(hl.HandLandmarksError) as err:
        run([bgr_frame()], [no_hand_result()], tmp_path, capture=capture)
    assert pseudonym(STEM) in str(err.value) and "CONTROL9001" not in str(err.value)
    assert capture.released


# --------------------------------------------------------------------------- parquet


def test_parquet_round_trip_with_metadata_and_geometry_reader(tmp_path: Path) -> None:
    model = tmp_path / "model.task"
    model.write_bytes(b"not a real model")
    frames = [bgr_frame(height=64, width=36) for _ in range(5)]
    results = [hand_result(), no_hand_result(), hand_result(), hand_result(), no_hand_result()]
    df, _, _ = run(frames, results, tmp_path, capture=FakeCapture(frames, fps=29.97))
    path = hl.output_path(tmp_path / f"{STEM}.mp4", tmp_path / "out")
    hl.write_landmarks(df, path)

    back, meta = hl.read_landmarks(path)
    # A missing label is None in memory and a null (NaN in pandas 3) after reading: same meaning.
    pd.testing.assert_frame_equal(
        back.drop(columns="handedness_label"),
        df.drop(columns="handedness_label"),
        check_dtype=False,
    )
    assert back["handedness_label"].isna().tolist() == df["handedness_label"].isna().tolist()
    assert back["handedness_label"].dropna().tolist() == df["handedness_label"].dropna().tolist()
    assert back.loc[~back["detected"], list(hl.LANDMARK_COLUMNS)].isna().all().all()
    geometry = hl.read_clip_geometry(path)
    assert geometry == hl.ClipGeometry(fps=29.97, width=36, height=64)
    assert meta["model_sha256"] is not None and meta["model_matches_expected"] is False
    assert meta["detector"]["num_hands"] == 1 and meta["n_landmarks"] == 21
    assert meta["opencv"] == cv2.__version__ and "mediapipe" in meta
    for forbidden in ("fps", "width", "height"):
        assert forbidden not in back.columns  # stored once in the metadata, not per row


def test_reading_a_file_without_our_metadata_fails(tmp_path: Path) -> None:
    other = tmp_path / "other.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(other)
    with pytest.raises(hl.HandLandmarksError, match="no hand_landmarks metadata"):
        hl.read_clip_geometry(other)


def test_output_file_is_named_by_the_clip_id_in_the_out_dir(tmp_path: Path) -> None:
    assert (
        hl.output_path(Path("somewhere") / f"{STEM}.mp4", tmp_path) == tmp_path / f"{STEM}.parquet"
    )


# --------------------------------------------------------------------------- summary


def test_summary_numbers(tmp_path: Path) -> None:
    frames = [bgr_frame() for _ in range(6)]
    results = [
        hand_result(label="Left"),
        hand_result(label="Left"),
        no_hand_result(),
        hand_result(label="Right"),
        hand_result(base_x=1.2, label="Right"),  # x of the last landmarks falls outside [0, 1]
        no_hand_result(),
    ]
    df, _, _ = run(frames, results, tmp_path)
    s = hl.summarize(df, df.attrs["hand_landmarks"], expected="right")
    assert (s["frames"], s["detected_frames"]) == (6, 4)
    assert s["detection_rate"] == pytest.approx(4 / 6)
    assert s["timestamp_ms_strictly_increasing"] and s["pts_ms_strictly_increasing"]
    assert s["handedness_counts"] == {"Left": 2, "Right": 2}
    assert s["handedness_switches"] == 1
    assert s["frames_label_equals_expected"] == 2 and s["expected_label"] == "Right"
    assert s["landmark_values_outside_0_1"] == N  # all 21 x values of the one shifted frame
    assert s["frames_with_any_value_outside_0_1"] == 1
    assert (s["undetected_rows_with_zero_values"], s["undetected_rows_not_nan"]) == (0, 0)


def test_summary_reports_pts_drift(tmp_path: Path) -> None:
    frames = [bgr_frame() for _ in range(3)]
    df, _, _ = run(
        frames, [no_hand_result()], tmp_path, capture=FakeCapture(frames, pts=[0.0, 33.3, 90.0])
    )
    s = hl.summarize(df, df.attrs["hand_landmarks"])
    assert s["max_abs_pts_minus_index_over_fps_ms"] == pytest.approx(90.0 - 2000 / 30, abs=0.01)
    assert math.isfinite(s["max_abs_pts_minus_index_over_fps_ms"])


def test_expected_hand_comes_from_the_clip_name() -> None:
    assert hl.expected_hand(f"{STEM}.mp4") == "right"
    assert hl.expected_hand("ID9101_IZDA.MOV") == "left"
    assert hl.expected_hand("something_else.mp4") is None


# --------------------------------------------------------------------------- command line


def make_clip(tmp_path: Path, stem: str = STEM) -> Path:
    clip = tmp_path / f"{stem}.mp4"
    clip.write_bytes(b"synthetic placeholder, never decoded")
    return clip


def cli(tmp_path: Path, *extra: str, **factories: Any) -> int:
    model = tmp_path / "model.task"
    model.write_bytes(b"x")
    return hl.main(
        [
            "--video",
            str(make_clip(tmp_path)),
            "--model",
            str(model),
            "--out-dir",
            str(tmp_path / "out"),
            *extra,
        ],
        **factories,
    )


def fake_factories(frames: list[Any], results: list[Any]) -> dict[str, Any]:
    return {
        "capture_factory": lambda path: FakeCapture(frames),
        "landmarker_factory": lambda model_path, **opts: FakeLandmarker(results),
    }


def test_cli_success_writes_a_file_and_prints_no_raw_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    frames = [bgr_frame() for _ in range(4)]
    code = cli(tmp_path, **fake_factories(frames, [hand_result(), no_hand_result()]))
    captured = capsys.readouterr()
    assert code == 0
    assert (tmp_path / "out" / f"{STEM}.parquet").is_file()
    for text in (captured.out, captured.err):
        assert "CONTROL9001" not in text and STEM not in text and f"{STEM}.parquet" not in text
    assert pseudonym(STEM) in captured.out
    assert "detection: 2/4 frames" in captured.out
    assert "must be 0" in captured.out


def test_cli_missing_video_exits_2_with_the_hash_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = hl.main(["--video", str(tmp_path / f"{STEM}.mp4"), "--out-dir", str(tmp_path)])
    err = capsys.readouterr().err
    assert code == 2
    assert pseudonym(STEM) in err and "CONTROL9001" not in err


def test_cli_missing_model_exits_2_and_points_to_the_fetch_script(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clip = make_clip(tmp_path)
    code = hl.main(
        ["--video", str(clip), "--model", str(tmp_path / "absent.task"), "--out-dir", str(tmp_path)]
    )
    err = capsys.readouterr().err
    assert code == 2
    assert "README.md" in err and "not found" in err
    assert not (tmp_path / f"{STEM}.parquet").exists()


def test_track_clip_missing_model_raises_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(hl.InputMissingError, match="README.md"):
        hl.track_clip(
            make_clip(tmp_path),
            tmp_path / "absent.task",
            capture_factory=lambda p: FakeCapture([bgr_frame()]),
        )


def test_cli_unreadable_video_exits_1_and_opencv_does_not_leak_the_path(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """A garbage file: OpenCV itself prints the full path on a failed open unless muted."""
    clip = tmp_path / f"{STEM}.mp4"
    clip.write_bytes(np.random.default_rng(0).bytes(4096))
    model = tmp_path / "model.task"
    model.write_bytes(b"x")
    code = hl.main(
        ["--video", str(clip), "--model", str(model), "--out-dir", str(tmp_path / "out")]
    )
    captured = capfd.readouterr()
    assert code == 1
    assert "FAILED" in captured.err and pseudonym(STEM) in captured.err
    assert "CONTROL9001" not in captured.out + captured.err
    assert not (tmp_path / "out").exists()


def test_cli_passes_detector_arguments_through(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def landmarker_factory(model_path: Path, **opts: Any) -> FakeLandmarker:
        seen.update(opts)
        return FakeLandmarker([no_hand_result()])

    code = cli(
        tmp_path,
        "--num-hands",
        "2",
        "--min-hand-detection-confidence",
        "0.25",
        "--min-hand-presence-confidence",
        "0.35",
        "--min-tracking-confidence",
        "0.45",
        capture_factory=lambda path: FakeCapture([bgr_frame()]),
        landmarker_factory=landmarker_factory,
    )
    assert code == 0
    assert seen == {
        "num_hands": 2,
        "min_hand_detection_confidence": 0.25,
        "min_hand_presence_confidence": 0.35,
        "min_tracking_confidence": 0.45,
    }


def test_stderr_is_restored_after_the_open_call(
    capfd: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    clip = tmp_path / "x.mp4"
    clip.write_bytes(b"junk")
    hl.open_video(str(clip)).release()
    print("after", file=sys.stderr)
    assert "after" in capfd.readouterr().err


# --------------------------------------------------------------------------- real model (skipped when absent)

needs_model = pytest.mark.skipif(
    not MODEL_FILE.is_file(),
    reason="models/tapping/artifacts/hand_landmarker.task is missing; see models/tapping/README.md",
)


def write_blank_video(
    path: Path, n: int = 6, size: tuple[int, int] = (160, 120), fps: float = 30.0
) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, size)
    assert writer.isOpened()
    for _ in range(n):
        writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    writer.release()


@needs_model
def test_real_model_runs_in_video_mode_and_finds_no_hand_in_a_blank_clip(tmp_path: Path) -> None:
    pytest.importorskip("mediapipe")
    clip = tmp_path / f"{STEM}.mp4"
    write_blank_video(clip)
    df = hl.track_clip(clip, MODEL_FILE)
    assert len(df) == 6 and list(df.columns) == list(hl.COLUMNS)
    assert not df["detected"].any()
    assert df[list(hl.LANDMARK_COLUMNS)].isna().all().all()
    meta = df.attrs["hand_landmarks"]
    assert meta["model_matches_expected"] is True
    assert (meta["width"], meta["height"]) == (160, 120)
    assert df["timestamp_ms"].is_monotonic_increasing and df["timestamp_ms"].is_unique


@needs_model
def test_real_model_through_the_command_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("mediapipe")
    clip = tmp_path / f"{STEM}.mp4"
    write_blank_video(clip)
    code = hl.main(["--video", str(clip), "--out-dir", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert code == 0 and pseudonym(STEM) in out and STEM not in out
    assert hl.read_clip_geometry(tmp_path / "out" / f"{STEM}.parquet").width == 160
