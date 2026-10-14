"""Tests for models/tapping/src/overlay_gif.py (task FT-14).

Synthetic data only: made-up landmark tables, a fake capture, no real clip. Clip names such as
``CONTROL9001_DCHA`` are made up; the numbers are not real participants.

No conftest.py on purpose (voice and gait each have one); ``src`` goes on ``sys.path`` here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pytest
from PIL import GifImagePlugin, Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hand_landmarks as hl  # noqa: E402
import overlay_gif as og  # noqa: E402
from hubu_fis import pseudonym  # noqa: E402

STEM = "CONTROL9001_DCHA"
REF = pseudonym(STEM)
VIDEO_W, VIDEO_H = 400, 300
FPS = 30.0


def make_pts(n: int, gaps: dict[int, float] | None = None) -> np.ndarray:
    """Container times at 30 fps; ``gaps`` {frame: extra ms} makes the timing irregular."""
    pts = np.arange(n) * 1000.0 / FPS
    for frame, extra in (gaps or {}).items():
        pts[frame:] += extra
    return pts


def make_table(n: int = 60, *, pts: np.ndarray | None = None, no_hand: Any = ()) -> pd.DataFrame:
    """All 21 landmarks at known pixel positions; landmark 8 swings so the distance changes.

    Landmark i is at (100 + 10 i, 200 + 5 i) before the swing; wrist (0) to knuckle (9) is
    about 100.6 px.
    """
    df = pd.DataFrame(np.nan, index=range(n), columns=list(hl.COLUMNS))
    df["frame"] = np.arange(n)
    pts = make_pts(n) if pts is None else pts
    df["pts_ms"] = pts
    df["timestamp_ms"] = np.round(np.arange(n) * 1000 / FPS).astype(np.int64)
    detected = np.ones(n, bool)
    detected[list(no_hand)] = False
    df["detected"] = detected
    swing = 25.0 * np.sin(np.arange(n) / 4.0)
    for i in range(hl.N_LANDMARKS):
        x = np.full(n, 100.0 + 10 * i) + (swing if i == 8 else 0.0)
        y = np.full(n, 200.0 + 5 * i)
        df.loc[detected, landmark_x(i)] = x[detected]
        df.loc[detected, landmark_y(i)] = y[detected]
    df.attrs["hand_landmarks"] = {"fps": FPS, "width": VIDEO_W, "height": VIDEO_H}
    return df


def landmark_x(i: int) -> str:
    return hl.landmark_column("x_px", i)


def landmark_y(i: int) -> str:
    return hl.landmark_column("y_px", i)


class FakeCapture:
    """Just enough of cv2.VideoCapture: frame ``i`` is uniformly ``10 + i`` grey."""

    def __init__(
        self,
        pts: np.ndarray,
        *,
        shift: int = 0,
        rotation_ok: bool = True,
        size: tuple[int, int] = (VIDEO_W, VIDEO_H),
    ) -> None:
        self.pts = np.asarray(pts, dtype=float)
        self.shift = shift  # a decoder that reports the wrong frame time
        self.rotation_ok = rotation_ok
        self.size = size
        self.index = -1
        self.orientation = 0.0
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV's name
        return True

    def set(self, prop: int, value: float) -> bool:
        if prop == cv2.CAP_PROP_ORIENTATION_AUTO and self.rotation_ok:
            self.orientation = float(value)
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_ORIENTATION_AUTO:
            return self.orientation
        if prop == cv2.CAP_PROP_POS_MSEC:
            return float(self.pts[min(max(self.index + self.shift, 0), len(self.pts) - 1)])
        return 0.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.index += 1
        if self.index >= len(self.pts):
            return False, None
        frame = np.full((self.size[1], self.size[0], 3), 10 + self.index, np.uint8)
        return True, frame

    def release(self) -> None:
        self.released = True


def run_overlay(df: pd.DataFrame, cap: FakeCapture, **kwargs: Any) -> Any:
    args: dict[str, Any] = {"start_s": 0.0, "duration_s": 2.0, "width": 160, "fps": 30.0}
    args.update(kwargs)
    return og.build_overlay(df, df.attrs["hand_landmarks"], cap, REF, **args)


# --------------------------------------------------------------------------- window and timing


def test_window_is_selected_by_pts_not_by_frame_index() -> None:
    pts = make_pts(60, gaps={30: 500.0})  # a half-second hole after frame 29
    assert og.select_window(pts, 0.0, 1.0).tolist() == list(range(30))
    assert og.select_window(pts, 1.0, 1.0).tolist() == list(range(30, 45))  # 1.5 s to 2 s
    assert og.select_window(np.array([0.0, np.nan, 40.0]), 0.0, 1.0).tolist() == [0, 2]


def test_stride_follows_the_clip_rate() -> None:
    assert og.stride_for(30.0, 15.0) == 2
    assert og.stride_for(30.0, 30.0) == 1
    assert og.stride_for(25.0, 60.0) == 1  # never above the clip's own rate
    assert og.stride_for(60.0, 15.0) == 4


def test_regular_30_fps_gives_whole_10_ms_delays_that_add_up() -> None:
    pts = make_pts(31)
    delays = og.gif_delays_ms(pts)
    assert all(d % 10 == 0 and d >= og.MIN_DELAY_MS for d in delays)
    assert set(delays) <= {30, 40}
    span = pts[-1] - pts[0]
    assert abs(sum(delays[:-1]) - span) <= 5.0  # the running total stays within 5 ms


def test_irregular_gaps_are_kept_in_the_delays() -> None:
    pts = np.array([0.0, 33.3, 100.0, 140.0, 300.0, 333.0])
    delays = og.gif_delays_ms(pts)
    assert delays[:5] == [30, 70, 40, 160, 30]
    assert abs(sum(delays[:-1]) - 333.0) <= 5.0


def test_no_delay_is_below_the_viewer_minimum_and_the_total_is_kept() -> None:
    pts = np.array([0.0, 8.0, 16.0, 24.0, 100.0])
    delays = og.gif_delays_ms(pts)
    assert min(delays) >= og.MIN_DELAY_MS and all(d % 10 == 0 for d in delays)
    assert og.gif_delays_ms([5.0]) == [og.MIN_DELAY_MS] and og.gif_delays_ms([]) == []


def test_bad_frame_times_are_rejected() -> None:
    for bad in ([0.0, 0.0, 40.0], [0.0, np.nan, 40.0], [10.0, 5.0]):
        with pytest.raises(og.OverlayError, match="frame times"):
            og.gif_delays_ms(bad)


# --------------------------------------------------------------------------- geometry


def test_crop_is_a_square_around_the_hand_and_stays_inside_the_frame() -> None:
    pts = np.array([[100.0, 100.0], [200.0, 150.0], [np.nan, np.nan]])
    assert og.crop_square(pts, 400, 300, pad=10.0) == (90, 65, 120)  # 100 wide + 2 x 10 pad
    x0, y0, side = og.crop_square(np.array([[5.0, 5.0], [60.0, 40.0]]), 400, 300, pad=50.0)
    assert (x0, y0) == (0, 0) and side == 155  # shifted back into the frame
    assert og.crop_square(np.array([[0.0, 0.0], [390.0, 290.0]]), 400, 300, 20.0)[2] == 300
    assert og.crop_square(np.full((3, 2), np.nan), 400, 300, 5.0) == (50, 0, 300)


def test_points_map_into_the_panel() -> None:
    xy = np.array([[150.0, 90.0], [np.nan, 10.0]])
    out = og.to_panel(xy, (100, 50, 200), 400)
    assert out[0].tolist() == [100.0, 80.0] and np.isnan(out[1, 0])


# --------------------------------------------------------------------------- drawing


def pixel(img: Image.Image, x: float, y: float) -> tuple[int, ...]:
    return tuple(img.getpixel((round(x), round(y))))  # type: ignore[arg-type]


def test_skeleton_is_drawn_at_the_given_pixel_positions() -> None:
    xy = np.full((21, 2), np.nan)
    xy[0], xy[1], xy[4], xy[8] = (20.0, 100.0), (60.0, 100.0), (40.0, 20.0), (80.0, 20.0)
    img = Image.new("RGB", (120, 120), (0, 0, 0))
    og.draw_skeleton(img, xy)
    assert pixel(img, 20, 100) == og.JOINT and pixel(img, 60, 100) == og.JOINT
    assert pixel(img, 40, 100) == og.BONE  # on the bone 0-1, halfway
    assert pixel(img, 40, 20) == og.THUMB_COLOUR and pixel(img, 80, 20) == og.INDEX_COLOUR
    assert pixel(img, 60, 20) == og.LINK  # the line between the two tips
    assert pixel(img, 110, 110) == (0, 0, 0) and pixel(img, 0, 0) == (0, 0, 0)


def test_a_frame_without_a_hand_draws_nothing() -> None:
    img = Image.new("RGB", (80, 80), (7, 8, 9))
    before = img.tobytes()
    og.draw_skeleton(img, np.full((21, 2), np.nan))
    assert img.tobytes() == before  # no skeleton at (0, 0), no leftovers


def test_a_missing_landmark_skips_only_its_own_bones() -> None:
    xy = np.full((21, 2), np.nan)
    xy[0], xy[1], xy[2] = (10.0, 50.0), (40.0, 50.0), (70.0, 50.0)
    xy[1] = np.nan
    img = Image.new("RGB", (100, 100), (0, 0, 0))
    og.draw_skeleton(img, xy)
    assert pixel(img, 25, 50) == (0, 0, 0) and pixel(img, 55, 50) == (0, 0, 0)  # no bone 0-1, 1-2
    assert pixel(img, 10, 50) == og.JOINT and pixel(img, 70, 50) == og.JOINT


def test_trace_cursor_and_gaps() -> None:
    values = np.array([0.1, 0.5, np.nan, np.nan, 0.9, 0.2])
    img = og.draw_trace((200, 80), values, cursor=4, ymax=1.0)
    cursor_x = 8 + (200 - 16) * 4 / 5
    assert pixel(img, cursor_x, 40) == og.TEXT  # the cursor line
    col = {tuple(img.getpixel((x, y))) for x in range(80, 120) for y in range(30, 70)}  # type: ignore[arg-type]
    assert og.TRACE_LIVE not in col and og.TRACE_DIM not in col  # the line breaks over the NaN gap
    assert og.trace_ymax(np.array([np.nan])) == 1.1 and og.trace_ymax(np.array([2.0])) == 2.2


def test_blur_leaves_the_hand_and_softens_the_rest() -> None:
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (160, 160, 3), dtype=np.uint8))
    xy = np.full((21, 2), np.nan)
    xy[0], xy[1], xy[2], xy[3] = (60.0, 60.0), (100.0, 60.0), (100.0, 100.0), (60.0, 100.0)
    out = og.blur_outside(img, xy)
    assert np.abs(np.subtract(pixel(out, 80, 80), pixel(img, 80, 80))).max() <= 4
    far = np.asarray(out.crop((0, 0, 20, 20)), float)
    assert far.std() < np.asarray(img.crop((0, 0, 20, 20)), float).std() / 2
    assert og.blur_outside(img, np.full((21, 2), np.nan)).size == img.size


# --------------------------------------------------------------------------- building the frames


def test_each_kept_frame_comes_from_the_right_source_frame() -> None:
    df = make_table(60)
    frames, delays, _ = run_overlay(df, FakeCapture(df["pts_ms"].to_numpy()), fps=15.0)
    assert len(frames) == len(delays) == 30  # every second frame of 60
    for k, frame in enumerate(frames):
        grey = frame.getpixel((2, 2))
        assert grey[0] == 10 + 2 * k  # frame 2k, not 2k +- 1


def test_frame_selection_and_delays_follow_pts_on_an_irregular_clip() -> None:
    pts = make_pts(60, gaps={20: 400.0})  # a 400 ms hole
    df = make_table(60, pts=pts)
    frames, delays, _ = run_overlay(df, FakeCapture(pts), start_s=0.0, duration_s=3.0)
    assert len(frames) == 60
    assert max(delays) >= 430 and 400 <= delays[19] <= 450  # the hole is in the delays
    assert abs(sum(delays[:-1]) - (pts[-1] - pts[0])) <= 5.0
    assert delays[19] > 5 * delays[5]  # not uniform 1/fps


def test_a_decoder_that_is_one_frame_off_is_caught() -> None:
    df = make_table(60)
    for shift in (1, -2):
        cap = FakeCapture(df["pts_ms"].to_numpy(), shift=shift)
        with pytest.raises(og.OverlayError, match="does not match the landmarks"):
            run_overlay(df, cap)


def test_auto_rotation_must_read_back_and_the_size_must_match() -> None:
    df = make_table(60)
    pts = df["pts_ms"].to_numpy()
    with pytest.raises(og.OverlayError, match="auto-rotation"):
        run_overlay(df, FakeCapture(pts, rotation_ok=False))
    with pytest.raises(og.OverlayError, match="is not 400x300"):
        run_overlay(df, FakeCapture(pts, size=(300, 400)))  # a rotated frame


def test_frames_without_a_hand_show_the_picture_and_no_skeleton() -> None:
    df = make_table(30, no_hand=range(10, 20))
    frames, _, _ = run_overlay(df, FakeCapture(df["pts_ms"].to_numpy()), duration_s=1.0)
    colours = {og.BONE, og.THUMB_COLOUR, og.INDEX_COLOUR, og.JOINT, og.LINK}
    for k in (5, 12, 25):
        panel = frames[k].crop((0, 0, 160, 160))
        found = {c for _, c in panel.getcolors(maxcolors=100000)} & colours  # type: ignore[union-attr]
        assert bool(found) == (k not in range(10, 20)), k
    assert frames[12].getpixel((2, 2))[0] == 22  # the picture is still that frame's


def test_the_poster_is_the_most_open_frame() -> None:
    df = make_table(60)
    frames, _, poster = run_overlay(df, FakeCapture(df["pts_ms"].to_numpy()), duration_s=2.0)
    from hand_scale import normalised_distance

    values = normalised_distance(df).values
    assert poster.tobytes() == frames[int(np.nanargmax(values))].tobytes()


def test_the_drawn_tips_sit_on_the_parquet_coordinates() -> None:
    df = make_table(60)
    frames, _, _ = run_overlay(df, FakeCapture(df["pts_ms"].to_numpy()), duration_s=2.0, width=320)
    points = np.vstack([og.landmark_xy(df, i) for i in range(60)])
    box = og.crop_square(points, VIDEO_W, VIDEO_H, og.CROP_PAD_HAND_SIZES * 100.6)
    for k in (0, 17, 41):
        arr = np.asarray(frames[k].crop((0, 0, 320, 320)))
        for tip, colour in ((8, og.INDEX_COLOUR), (4, og.THUMB_COLOUR)):
            ys, xs = np.nonzero((arr == np.array(colour)).all(axis=2))
            expected = og.to_panel(og.landmark_xy(df, k)[tip], box, 320)
            assert np.hypot(xs.mean() - expected[0], ys.mean() - expected[1]) < 1.5


# --------------------------------------------------------------------------- limits and the file


def test_limits_are_enforced(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    df = make_table(90)
    pts = df["pts_ms"].to_numpy()
    for kwargs, match in (
        ({"width": 100}, "width"),
        ({"width": 700}, "width"),
        ({"duration_s": 7.0}, "duration"),
        ({"duration_s": 0.0}, "duration"),
        ({"fps": 0.0}, "fps"),
        ({"start_s": 50.0}, "fewer than 2"),
    ):
        with pytest.raises(og.OverlayError, match=match):
            run_overlay(df, FakeCapture(pts), **kwargs)
    monkeypatch.setattr(og, "MAX_FRAMES", 10)
    with pytest.raises(og.OverlayError, match="frames is above"):
        run_overlay(df, FakeCapture(pts), duration_s=2.0)
    monkeypatch.setattr(og, "MAX_FRAMES", 150)
    frames, delays, _ = run_overlay(df, FakeCapture(pts), duration_s=1.0)
    monkeypatch.setattr(og, "MAX_BYTES", 500)
    path = tmp_path / "x.gif"
    with pytest.raises(og.OverlayError, match="limit"):
        og.save_gif(frames, delays, path)
    assert not path.exists()  # an oversize file is not left behind


def test_the_gif_reloads_with_the_same_frames_delays_and_loop(tmp_path: Path) -> None:
    pts = make_pts(60, gaps={10: 300.0})
    df = make_table(60, pts=pts)
    frames, delays, _ = run_overlay(df, FakeCapture(pts), duration_s=3.0)
    path = tmp_path / "out.gif"
    size = og.save_gif(frames, delays, path)
    assert size == path.stat().st_size <= og.MAX_BYTES
    with Image.open(path) as gif:
        assert isinstance(gif, GifImagePlugin.GifImageFile)
        assert gif.n_frames == len(frames) and gif.info["loop"] == 0
        got = []
        for i in range(gif.n_frames):
            gif.seek(i)
            got.append(int(gif.info["duration"]))
    assert got == delays and all(d % 10 == 0 for d in got)


# --------------------------------------------------------------------------- privacy and command line


def write_clip(tmp_path: Path, stem: str = STEM) -> Path:
    df = make_table(60)
    path = tmp_path / "landmarks" / f"{stem}.parquet"
    hl.write_landmarks(df, path)
    return path


def run_main(
    tmp_path: Path,
    *extra: str,
    cap: FakeCapture | None = None,
    stem: str = STEM,
    fake_video: bool = True,
) -> int:
    """Run the command line; ``fake_video=False`` uses the real video lookup (no video on disk)."""
    factory = (lambda _path: cap or FakeCapture(make_pts(60))) if fake_video else None
    return int(
        og.main(
            [
                "--clip",
                pseudonym(stem),
                "--landmarks-dir",
                str(tmp_path / "landmarks"),
                "--video-dir",
                str(tmp_path / "videos"),
                "--out-dir",
                str(tmp_path / "interim" / "gifs"),
                "--interim-root",
                str(tmp_path / "interim"),
                "--width",
                "160",
                "--duration-s",
                "1.0",
                *extra,
            ],
            capture_factory=factory,
        )
    )


def test_cli_writes_gif_and_poster_named_by_hash_with_no_raw_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    write_clip(tmp_path)
    texts: list[str] = []
    original = ImageDraw.ImageDraw.text

    def spy(self: Any, xy: Any, text: Any, *args: Any, **kwargs: Any) -> Any:
        texts.append(str(text))
        return original(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", spy)
    assert run_main(tmp_path) == 0
    captured = capsys.readouterr()
    out = tmp_path / "interim" / "gifs"
    assert sorted(p.name for p in out.iterdir()) == [
        f"{REF}_overlay.gif",
        f"{REF}_overlay_poster.png",
    ]
    everything = captured.out + captured.err + " ".join(p.name for p in out.iterdir())
    assert "CONTROL" not in everything and "9001" not in everything
    assert texts and not any("CONTROL" in t or "9001" in t or REF in t for t in texts)
    assert not any(w in " ".join(texts).lower() for w in ("updrs", "diagnos", "patient", "group"))


def test_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_clip(tmp_path)
    assert run_main(tmp_path, stem="CONTROL9002_IZDA") == 2  # no landmarks for this hash
    assert run_main(tmp_path, fake_video=False) == 2  # no video on disk
    assert "video file not found" in capsys.readouterr().err
    assert run_main(tmp_path, cap=FakeCapture(make_pts(60), rotation_ok=False)) == 1
    err = capsys.readouterr().err
    assert "FAILED" in err and "CONTROL" not in err and REF in err
    capsys.readouterr()
    assert run_main(tmp_path, "--width", "9000") == 1
    assert "width" in capsys.readouterr().err
    assert not (tmp_path / "interim" / "gifs").exists()


def test_the_output_folder_must_stay_inside_the_interim_root(tmp_path: Path) -> None:
    write_clip(tmp_path)
    code = og.main(
        [
            "--clip",
            REF,
            "--landmarks-dir",
            str(tmp_path / "landmarks"),
            "--out-dir",
            str(tmp_path / "elsewhere"),
            "--interim-root",
            str(tmp_path / "interim"),
        ],
        capture_factory=lambda _p: FakeCapture(make_pts(60)),
    )
    assert code == 1 and not (tmp_path / "elsewhere").exists()


def test_a_corrupt_landmarks_file_exits_1_without_the_raw_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "landmarks" / "CONTROL9003_DCHA.parquet"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"not a parquet file")
    assert run_main(tmp_path, stem="CONTROL9003_DCHA") == 1
    err = capsys.readouterr().err
    assert "CONTROL9003" not in err and pseudonym("CONTROL9003_DCHA") in err
