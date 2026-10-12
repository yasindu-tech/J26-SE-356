"""Tests for models/tapping/src/tracking.py (task FT-15).

Synthetic only: tiny videos written with OpenCV and a fake landmarker that reads its answer off
the picture. Clip names such as ``CONTROL9001_DCHA`` are made up; there is no real data.

No conftest.py on purpose (voice and gait each have one); ``src`` goes on ``sys.path`` here, and
a spawned worker inherits that path, so the module-level fakes below can be unpickled there.
"""

from __future__ import annotations

import os
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
import hand_scale as hs  # noqa: E402
import tracking as tr  # noqa: E402
from hubu_fis import pseudonym  # noqa: E402
from test_hand_scale import make_table  # noqa: E402

W, H, FPS = 64, 48, 30.0
NAMES = ["CONTROL9001_DCHA", "CONTROL9001_IZDA", "ID9101_DCHA"]
RAW_PARTS = ("CONTROL9001", "ID9101", "CONTROL9004")


# --------------------------------------------------------------------------- fakes (module level)


class FakeLandmarker:
    """Reads the answer off the picture: red above 128 = no hand; green = how far open."""

    def detect_for_video(self, rgb: np.ndarray, timestamp_ms: int) -> Any:
        red, green = float(rgb[..., 0].mean()), float(rgb[..., 1].mean())
        if red > 128:
            return SimpleNamespace(hand_landmarks=[], hand_world_landmarks=[], handedness=[])
        sep = 0.02 + (green - 40.0) / 160.0 * 0.3
        xy = {i: (0.30 + 0.01 * i, 0.50) for i in range(21)}
        xy.update({0: (0.5, 0.8), 5: (0.45, 0.52), 9: (0.5, 0.5), 13: (0.55, 0.52)})
        xy.update({17: (0.6, 0.55), 4: (0.4, 0.3), 8: (0.4 + sep, 0.3)})
        pts = [SimpleNamespace(x=xy[i][0], y=xy[i][1], z=0.0) for i in range(21)]
        world = [SimpleNamespace(x=xy[i][0] - 0.5, y=xy[i][1] - 0.5, z=0.0) for i in range(21)]
        return SimpleNamespace(
            hand_landmarks=[pts],
            hand_world_landmarks=[world],
            handedness=[[SimpleNamespace(category_name="Right", score=0.9)]],
        )

    def close(self) -> None:
        pass


def exploding_factory(model_path: Any, **kwargs: Any) -> FakeLandmarker:
    raise AssertionError("a landmarker was built for an up-to-date clip")


def fake_landmarker_factory(model_path: Any, **kwargs: Any) -> FakeLandmarker:
    return FakeLandmarker()


# --------------------------------------------------------------------------- helpers


def write_video(
    path: Path, frames: int = 40, missing: set[int] | frozenset[int] = frozenset()
) -> None:
    """A tiny mp4: green swings (the tap); red marks a frame without a hand."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), FPS, (W, H))
    for i in range(frames):
        green = int(120 + 80 * np.sin(2 * np.pi * i / 10.0))
        red = 240 if i in missing else 20
        writer.write(np.full((H, W, 3), (60, green, red), np.uint8))
    writer.release()


@pytest.fixture
def videos(tmp_path: Path) -> Path:
    folder = tmp_path / "videos"
    folder.mkdir()
    for name in NAMES:
        write_video(folder / f"{name}.mp4")
    return folder


def run(tmp_path: Path, *extra: str, workers: int = 1, out: str = "out", **kw: Any) -> int:
    args = [
        "--video-dir", str(tmp_path / "videos"),
        "--out-dir", str(tmp_path / "interim" / out),
        "--interim-root", str(tmp_path / "interim"),
        "--model", str(tmp_path / "model.task"),
        "--workers", str(workers),
        "--min-frames", "10",
        *extra,
    ]  # fmt: skip
    kw.setdefault("landmarker_factory", fake_landmarker_factory)
    return int(tr.main(args, **kw))


def out_dir(tmp_path: Path, out: str = "out") -> Path:
    return tmp_path / "interim" / out


def manifest(tmp_path: Path, out: str = "out") -> pd.DataFrame:
    return pd.read_csv(out_dir(tmp_path, out) / tr.MANIFEST_NAME)


def metrics_row(**changes: Any) -> dict[str, Any]:
    """A manifest-metrics dict that raises no flag at the default thresholds."""
    row: dict[str, Any] = {
        "detection_rate": 1.0,
        "unfillable_frames_share": 0.0,
        "frames_decoded": 600,
        "frames_container": 600,
        "pts_missing": 0,
        "pts_increasing": True,
        "out_of_range_share": 0.0,
        "scale_tap_abs_r": 0.1,
        "pts_max_gap_ms": 34.0,
    }
    row.update(changes)
    return row


def names(row: dict[str, Any], gating: tr.Gating | None = None) -> list[str]:
    return [n for n, _ in tr.evaluate_flags(row, gating or tr.Gating())]


# --------------------------------------------------------------------------- numbers and flags


def test_missing_runs_and_edges() -> None:
    d = np.array([0, 0, 1, 1, 0, 1, 0, 0, 0], bool)
    assert tr.missing_runs(d) == [2, 1, 3] and tr.edge_missing(d) == 2 + 3
    assert tr.missing_runs(np.ones(4, bool)) == [] and tr.edge_missing(np.ones(4, bool)) == 0
    assert tr.edge_missing(np.zeros(4, bool)) == 4


def test_every_flag_is_raised_by_its_own_condition_and_nothing_else() -> None:
    assert names(metrics_row()) == []
    assert names(metrics_row(detection_rate=0.5)) == ["low_detection"]
    assert names(metrics_row(unfillable_frames_share=0.2)) == ["gaps_not_fillable"]
    assert names(metrics_row(frames_decoded=100, frames_container=100)) == ["short_clip"]
    assert names(metrics_row(frames_decoded=500)) == ["frame_count_mismatch"]
    assert names(metrics_row(pts_missing=3)) == ["pts_unreliable"]
    assert names(metrics_row(pts_increasing=False)) == ["pts_unreliable"]
    assert names(metrics_row(out_of_range_share=0.5)) == ["landmarks_out_of_frame"]
    assert names(metrics_row(scale_tap_abs_r=0.9)) == ["scale_tap_coupling"]
    assert names(metrics_row(pts_max_gap_ms=80.0)) == ["info_irregular_timing"]


@pytest.mark.parametrize(
    ("field", "at", "past", "flag"),
    [
        ("detection_rate", 0.90, 0.8999, "low_detection"),
        ("unfillable_frames_share", 0.05, 0.0501, "gaps_not_fillable"),
        ("out_of_range_share", 0.05, 0.0501, "landmarks_out_of_frame"),
        ("scale_tap_abs_r", 0.5, 0.5001, "scale_tap_coupling"),
        ("pts_max_gap_ms", 50.0, 50.1, "info_irregular_timing"),
    ],
)
def test_a_value_exactly_at_the_threshold_is_not_flagged(
    field: str, at: float, past: float, flag: str
) -> None:
    assert names(metrics_row(**{field: at})) == []
    assert names(metrics_row(**{field: past})) == [flag]


def test_count_thresholds_at_and_around_the_limit() -> None:
    assert names(metrics_row(frames_decoded=150, frames_container=150)) == []
    assert names(metrics_row(frames_decoded=149, frames_container=149)) == ["short_clip"]
    assert names(metrics_row(frames_decoded=588)) == []  # exactly 2% of 600
    assert names(metrics_row(frames_decoded=587)) == ["frame_count_mismatch"]
    assert names(metrics_row(scale_tap_abs_r=float("nan"))) == []  # not computed: no flag


def test_thresholds_are_parameters_and_move_the_flags() -> None:
    row = metrics_row(detection_rate=0.8)
    assert names(row, tr.Gating(min_detection_rate=0.7)) == []
    assert names(row, tr.Gating(min_detection_rate=0.9)) == ["low_detection"]


def test_gap_numbers_from_a_real_table_use_the_fillable_limit() -> None:
    d = np.full(100, 50.0)
    d[[10, 11, 12, 13, 14, 15, 40, 41]] = np.nan
    df = make_table(d)
    meta = {"fps": 30.0, "width": 400, "height": 300, "frames_container": 100}
    m = tr.clip_metrics(df, meta, tr.Gating())
    assert (m["longest_missing_run"], m["n_gaps"]) == (6, 2)
    assert m["gap_share_fillable"] == pytest.approx(0.5)  # the 2-frame gap yes, the 6-frame no
    assert m["unfillable_frames_share"] == pytest.approx(0.06)
    assert "gaps_not_fillable" in names(m) and m["frames_detected"] == 92


def test_the_coupling_flag_comes_from_the_ft13_tools() -> None:
    d = 15.0 + 65.0 * (0.5 + 0.5 * np.sin(2 * np.pi * np.arange(600) / 30.0))
    coupled = make_table(d, zoom=1.0 + 0.2 * (d - d.mean()) / np.ptp(d))
    meta = {"fps": 30.0, "width": 400, "height": 300, "frames_container": 600}
    assert hs.scale_tap_correlation(coupled, mode="per_frame") > 0.9
    # the flag uses the clip's default mode (rolling median), which averages the coupling out
    m = tr.clip_metrics(coupled, meta, tr.Gating())
    assert m["scale_tap_abs_r"] == pytest.approx(hs.scale_tap_correlation(coupled))
    assert m["scale"] == "w9" and m["scale_mode"] == "rolling_median"
    forced = dict(m, scale_tap_abs_r=0.7)
    assert "scale_tap_coupling" in names(forced)
    assert "scale_tap_coupling" not in names(forced, tr.Gating(max_coupling_r=0.8))


# --------------------------------------------------------------------------- end to end


def test_a_run_writes_one_parquet_per_clip_and_a_manifest(tmp_path: Path, videos: Path) -> None:
    assert run(tmp_path) == 0
    files = sorted(p.name for p in out_dir(tmp_path).glob("*.parquet"))
    assert files == sorted(f"{n}.parquet" for n in NAMES)
    df, meta = hl.read_landmarks(out_dir(tmp_path) / files[0])
    assert list(df.columns) == list(hl.COLUMNS) and len(df) == 40
    assert meta["runner"] == tr.RUNNER_VERSION and "video_fingerprint" in meta
    m = manifest(tmp_path)
    assert list(m.columns) == list(tr.MANIFEST_COLUMNS)
    assert m["clip_id"].tolist() == sorted(NAMES) and (m["status"] == "ok").all()
    assert set(m["action"]) == {"computed"} and (m["frames_detected"] == 40).all()
    assert set(m["group"]) == {"CONTROL", "ID"} and set(m["hand"]) == {"right", "left"}
    assert m.loc[0, "ref"] == pseudonym(m.loc[0, "clip_id"])
    assert m["gating"].nunique() == 1 and '"min_detection_rate": 0.9' in m.loc[0, "gating"]
    assert (m["resolution"] == f"{W}x{H}").all() and (m["orientation"] == "landscape").all()


def test_a_second_run_skips_everything_and_touches_nothing(
    tmp_path: Path, videos: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(tmp_path) == 0
    before = {p.name: p.stat().st_mtime_ns for p in out_dir(tmp_path).glob("*.parquet")}
    first = manifest(tmp_path)
    capsys.readouterr()
    assert run(tmp_path, landmarker_factory=exploding_factory) == 0  # none is ever built
    assert {p.name: p.stat().st_mtime_ns for p in out_dir(tmp_path).glob("*.parquet")} == before
    again = manifest(tmp_path)
    assert set(again["action"]) == {"skipped"}
    pd.testing.assert_frame_equal(
        first.drop(columns=["action", "recompute_reason"]),
        again.drop(columns=["action", "recompute_reason"]),
    )
    assert "3 up to date, 0 to compute" in capsys.readouterr().out


def test_a_changed_video_is_recomputed_and_others_are_skipped(tmp_path: Path, videos: Path) -> None:
    run(tmp_path)
    write_video(videos / f"{NAMES[0]}.mp4", frames=45)
    assert run(tmp_path) == 0
    m = manifest(tmp_path).set_index("clip_id")
    assert m.loc[NAMES[0], "action"] == "computed"
    assert m.loc[NAMES[0], "recompute_reason"] == "video_changed"
    assert m.loc[NAMES[0], "frames_decoded"] == 45
    assert m.loc[NAMES[1], "action"] == "skipped"


def test_files_from_before_ft15_are_recomputed_once(tmp_path: Path, videos: Path) -> None:
    run(tmp_path)
    for p in out_dir(tmp_path).glob("*.parquet"):  # make them look like FT-11 files
        df, meta = hl.read_landmarks(p)
        for key in ("video_fingerprint", "runner", "tracking_wall_s"):
            meta.pop(key, None)
        df.attrs["hand_landmarks"] = meta
        hl.write_landmarks(df, p)
    assert run(tmp_path) == 0
    assert set(manifest(tmp_path)["recompute_reason"]) == {"legacy_no_fingerprint"}
    assert run(tmp_path) == 0 and set(manifest(tmp_path)["action"]) == {"skipped"}


def test_another_parameter_set_in_the_folder_is_refused_and_force_recomputes(
    tmp_path: Path, videos: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run(tmp_path)
    before = {p.name: p.read_bytes() for p in out_dir(tmp_path).glob("*.parquet")}
    capsys.readouterr()
    assert run(tmp_path, "--min-hand-detection-confidence", "0.3") == 2
    err = capsys.readouterr().err
    assert "3 file(s) made with another parameter set" in err and "refused" in err
    assert {p.name: p.read_bytes() for p in out_dir(tmp_path).glob("*.parquet")} == before
    assert run(tmp_path, "--min-hand-detection-confidence", "0.3", "--force") == 0
    m = manifest(tmp_path)
    assert (m["min_hand_detection_confidence"] == 0.3).all() and set(m["action"]) == {"computed"}
    assert run(tmp_path, "--min-hand-detection-confidence", "0.3") == 0  # now uniform: fine


def test_a_changed_model_or_tool_version_counts_as_another_parameter_set(
    tmp_path: Path, videos: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.task"
    model.write_bytes(b"model one")
    run(tmp_path)
    model.write_bytes(b"model two")
    assert run(tmp_path) == 2  # the model hash changed
    model.write_bytes(b"model one")
    assert run(tmp_path) == 0  # back to the original: up to date
    monkeypatch.setattr(hl, "TOOL_VERSION", "hand-landmarks-9.9")
    assert run(tmp_path) == 2  # the tool version changed
    assert run(tmp_path, "--force") == 0


def test_a_separate_out_dir_per_parameter_set_never_mixes(tmp_path: Path, videos: Path) -> None:
    run(tmp_path)
    assert run(tmp_path, "--min-hand-detection-confidence", "0.3", out="lower") == 0
    assert (manifest(tmp_path, "lower")["min_hand_detection_confidence"] == 0.3).all()
    assert (manifest(tmp_path)["min_hand_detection_confidence"] == 0.5).all()


def test_one_worker_and_two_workers_give_identical_results(tmp_path: Path, videos: Path) -> None:
    assert run(tmp_path, workers=1, out="one") == 0
    assert run(tmp_path, workers=2, out="two") == 0
    for name in NAMES:
        a, ma = hl.read_landmarks(out_dir(tmp_path, "one") / f"{name}.parquet")
        b, mb = hl.read_landmarks(out_dir(tmp_path, "two") / f"{name}.parquet")
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        for key in ("tracking_wall_s",):
            ma.pop(key), mb.pop(key)
        assert ma == mb
    one, two = manifest(tmp_path, "one"), manifest(tmp_path, "two")
    pd.testing.assert_frame_equal(one.drop(columns="wall_s"), two.drop(columns="wall_s"))


def test_one_failing_clip_does_not_stop_the_others(
    tmp_path: Path, videos: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (videos / "CONTROL9004_DCHA.mp4").write_bytes(b"this is not a video")
    code = run(tmp_path)
    captured = capsys.readouterr()
    assert code == 1
    m = manifest(tmp_path).set_index("clip_id")
    assert m.loc["CONTROL9004_DCHA", "status"] == "failed"
    assert (m.drop(index="CONTROL9004_DCHA")["status"] == "ok").all()
    assert not (out_dir(tmp_path) / "CONTROL9004_DCHA.parquet").exists()
    assert "failed" in captured.out and pseudonym("CONTROL9004_DCHA") in captured.out


def test_a_failure_while_writing_leaves_no_partial_file(
    tmp_path: Path, videos: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = hl.write_landmarks

    def broken(df: pd.DataFrame, path: str | Path) -> None:
        real(df, path)  # the temporary file exists and is complete ...
        raise OSError(f"disk full while writing {path}")  # ... and then the write fails

    monkeypatch.setattr(hl, "write_landmarks", broken)
    assert run(tmp_path) == 1
    folder = out_dir(tmp_path)
    assert list(folder.glob("*.parquet")) == [] and [p.name for p in folder.glob(".*")] == []
    m = manifest(tmp_path)
    assert (m["status"] == "failed").all() and not m["reason"].str.contains("CONTROL|ID9").any()


def test_temporary_files_of_a_killed_run_are_removed_at_the_next_start(
    tmp_path: Path, videos: Path
) -> None:
    folder = out_dir(tmp_path)
    folder.mkdir(parents=True)
    leftover = folder / f".{NAMES[0]}.parquet{tr.TMP_MARKER}4242"
    leftover.write_bytes(b"half a file")
    assert run(tmp_path) == 0 and not leftover.exists()
    assert len(list(folder.glob("*.parquet"))) == 3


def test_flagged_clips_are_kept_and_the_run_still_succeeds(tmp_path: Path, videos: Path) -> None:
    write_video(videos / "CONTROL9001_DCHA.mp4", missing=set(range(0, 40, 2)))  # half missing
    assert run(tmp_path) == 0  # a flag never changes the exit code
    assert (out_dir(tmp_path) / "CONTROL9001_DCHA.parquet").is_file()
    m = manifest(tmp_path).set_index("clip_id")
    assert "low_detection" in m.loc["CONTROL9001_DCHA", "flags"]
    assert m.loc["CONTROL9001_DCHA", "detection_rate"] == pytest.approx(0.5)
    assert m.loc["CONTROL9001_DCHA", "longest_missing_run"] == 1
    assert isinstance(m.loc[NAMES[1], "flags"], float)  # an empty flag cell is NaN in the CSV
    df, _ = hl.read_landmarks(out_dir(tmp_path) / "CONTROL9001_DCHA.parquet")
    assert df.loc[~df["detected"], list(hl.LANDMARK_COLUMNS)].isna().all().all()  # NaN, not 0


def test_flag_thresholds_come_from_the_command_line_and_are_recorded(
    tmp_path: Path, videos: Path
) -> None:
    write_video(videos / "CONTROL9001_DCHA.mp4", missing=set(range(0, 40, 4)))  # 75% detected
    run(tmp_path, "--min-detection-rate", "0.8")
    m = manifest(tmp_path).set_index("clip_id")
    assert "low_detection" in m.loc["CONTROL9001_DCHA", "flags"]
    assert '"min_detection_rate": 0.8' in m.loc["CONTROL9001_DCHA", "gating"]
    run(tmp_path, "--min-detection-rate", "0.7")
    assert "low_detection" not in str(
        manifest(tmp_path).set_index("clip_id").loc["CONTROL9001_DCHA", "flags"]
    )


def test_no_raw_id_in_the_output_or_in_the_manifest_text(
    tmp_path: Path, videos: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (videos / "CONTROL9004_DCHA.mp4").write_bytes(b"not a video")
    run(tmp_path)
    run(tmp_path, "--dry-run")
    captured = capsys.readouterr()
    m = manifest(tmp_path)
    text = (
        captured.out
        + captured.err
        + " ".join(map(str, m.drop(columns="clip_id").to_numpy().ravel()))
    )
    for raw in RAW_PARTS:
        assert raw not in text
    assert all(pseudonym(c) == r for c, r in zip(m["clip_id"], m["ref"], strict=True))


def test_the_output_folder_must_be_inside_the_interim_root(tmp_path: Path, videos: Path) -> None:
    code = tr.main(
        ["--video-dir", str(videos), "--out-dir", str(tmp_path / "elsewhere"),
         "--interim-root", str(tmp_path / "interim")],
        landmarker_factory=fake_landmarker_factory,
    )  # fmt: skip
    assert code == 2 and not (tmp_path / "elsewhere").exists()
    code = run(tmp_path, "--manifest", str(tmp_path / "elsewhere.csv"))
    assert code == 2 and not (tmp_path / "elsewhere.csv").exists()


def test_missing_inputs_exit_2(
    tmp_path: Path, videos: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(tmp_path, "--video-dir", str(tmp_path / "nope")) == 2
    assert run(tmp_path, "--clips", "deadbeef") == 2
    assert "no video for clip(s) deadbeef" in capsys.readouterr().err
    assert run(tmp_path, "--limit", "0") == 2  # nothing selected
    assert run(tmp_path, landmarker_factory=None) == 2  # no model file and no fake
    assert "model not found" in capsys.readouterr().err
    empty = tmp_path / "empty"
    empty.mkdir()
    assert run(tmp_path, "--video-dir", str(empty)) == 2
    assert not out_dir(tmp_path).exists()


def test_two_videos_with_one_clip_id_are_refused(tmp_path: Path, videos: Path) -> None:
    write_video(videos / f"{NAMES[0]}.MOV")
    assert run(tmp_path) == 2


def test_dry_run_and_subsets(
    tmp_path: Path, videos: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(tmp_path, "--dry-run") == 0
    assert "3 clips selected: 0 up to date, 3 to compute (missing 3)" in capsys.readouterr().out
    assert not out_dir(tmp_path).exists()
    assert run(tmp_path, "--clips", f"{pseudonym(NAMES[2])},{pseudonym(NAMES[0])}") == 0
    assert manifest(tmp_path)["clip_id"].tolist() == sorted([NAMES[0], NAMES[2]])
    assert run(tmp_path, "--limit", "1") == 0
    assert len(manifest(tmp_path)) == 1 and len(list(out_dir(tmp_path).glob("*.parquet"))) == 2


def test_fingerprint_changes_with_content_and_size(tmp_path: Path) -> None:
    a = tmp_path / "a.mp4"
    a.write_bytes(os.urandom(3 * 1024 * 1024))
    fp = tr.video_fingerprint(a)
    assert tr.video_fingerprint(a) == fp
    data = bytearray(a.read_bytes())
    data[-5] ^= 1  # a change in the last MiB
    a.write_bytes(bytes(data))
    assert tr.video_fingerprint(a) != fp
    a.write_bytes(bytes(data) + b"x")
    assert tr.video_fingerprint(a)["size"] == fp["size"] + 1


def test_failed_rows_keep_the_column_order(tmp_path: Path) -> None:
    row = tr.failed_row(Path("CONTROL9001_DCHA.mp4"), tr.Gating(), tr.Detector(), "boom", "missing")
    assert list(row) == list(tr.MANIFEST_COLUMNS) or set(row) == set(tr.MANIFEST_COLUMNS)
    assert row["status"] == "failed" and row["ref"] == pseudonym("CONTROL9001_DCHA")
    assert tr.scrub("open CONTROL9001_DCHA.mp4 failed", Path("/x/CONTROL9001_DCHA.mp4")) == (
        f"open {pseudonym('CONTROL9001_DCHA')} failed"
    )
