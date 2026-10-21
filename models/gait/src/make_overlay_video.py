"""Draw the pose skeleton and the heel strikes on a walking video (for slides and demos).

    python models/gait/src/make_overlay_video.py VIDEO [VIDEO ...] [--out FILE | --out-dir DIR]
        [--trace TRACE.npz]

Uses the same pose, walk finding and heel-strike steps as ``predict_video``, so what is drawn is
what the model used. ``--trace`` (one video) takes the pose trace ``predict_video --trace`` saved,
so the pose model is not run twice. The scored stretch gets a green border; the rest is greyed
out. A video with no walk in it gets no overlay. Output names use the short hash of the input name
unless ``--out`` is given. Exit code 0 = all done.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from carepd_format import short_id
from cycles import CycleCheck, segment_cycles
from gait_extract import MAX_FPS, MIN_FPS
from mediapipe_mapping import walks_from_trace
from pose_probe import PoseTrace, VideoInfo, probe_video, read_info, read_rgb_frames
from pose_trace_io import load_trace
from preprocess import detect_heel_strikes
from video_to_joints import DEFAULT_MODEL, MAX_SECONDS

LEFT_HEEL, RIGHT_HEEL = 29, 30
MIN_VISIBILITY = 0.5
RING_SECONDS = 0.35  # how long a heel-strike ring stays on screen
MAX_SIDE = 1280  # longer videos are scaled down so the file stays small and quick to write
HEAD = 0
# (landmark a, landmark b, side): shoulders and hips are the torso; the rest follow the body.
BONES = (
    (11, 12, "mid"), (23, 24, "mid"), (11, 23, "left"), (12, 24, "right"),
    (11, 13, "left"), (13, 15, "left"), (12, 14, "right"), (14, 16, "right"),
    (23, 25, "left"), (25, 27, "left"), (27, 29, "left"), (29, 31, "left"), (27, 31, "left"),
    (24, 26, "right"), (26, 28, "right"), (28, 30, "right"), (30, 32, "right"), (28, 32, "right"),
)  # fmt: skip
# BGR colours: the left side orange, the right side blue, the torso white.
COLOURS = {"left": (40, 150, 255), "right": (230, 140, 40), "mid": (240, 240, 240)}
GREEN, GREY, AMBER = (90, 180, 60), (150, 150, 150), (40, 170, 240)


@dataclass
class Plan:
    """What to draw: where the scored stretch is and which frames show a counted heel strike."""

    state: str  # "scored", "few_strides", "no_walk" or "frame_rate"
    start: int = 0  # first frame of the stretch the model read
    stop: int = 0  # one past its last frame
    strides: int = 0
    strikes: dict[int, list[str]] = field(default_factory=dict)  # clip frame -> feet


def counted_strikes(check: CycleCheck, offset: int) -> dict[int, list[str]]:
    """Frames of the heel strikes that open or close a stride the model counted."""
    found: dict[int, set[str]] = {}
    for cycle in check.cycles:
        if cycle.plausible:
            for frame in (cycle.start, cycle.end):
                found.setdefault(frame + offset, set()).add(cycle.foot)
    return {frame: sorted(feet) for frame, feet in sorted(found.items())}


def build_plan(trace: PoseTrace) -> Plan:
    """The longest usable walk of the trace and its counted heel strikes."""
    walks = walks_from_trace(trace)
    if not walks:
        return Plan("no_walk")
    walk = max(walks, key=lambda w: w.joints.shape[0])
    plan = Plan("scored", walk.start, walk.stop)
    if not MIN_FPS <= walk.fps <= MAX_FPS:
        plan.state = "frame_rate"
        return plan
    try:
        check = segment_cycles(detect_heel_strikes(walk.joints, walk.fps))
    except ValueError:  # too few frames to find any step
        plan.state = "few_strides"
        return plan
    plan.strides = check.strides()
    plan.strikes = counted_strikes(check, walk.start)
    if not (check.passed and check.strides("left") > 0 and check.strides("right") > 0):
        plan.state = "few_strides"
    return plan


def caption(plan: Plan, fps: float, frame: int) -> tuple[str, tuple[int, int, int]]:
    """The banner text and border colour for one frame."""
    inside = plan.start <= frame < plan.stop
    span = f"{plan.start / fps:.1f} s to {plan.stop / fps:.1f} s"
    if plan.state == "no_walk":
        return "NOT SCORED: no usable walk found", GREY
    if plan.state == "frame_rate":
        return "NOT SCORED: frame rate not supported", GREY
    if not inside:
        return "not used: the model read only the walk", GREY
    if plan.state == "few_strides":
        return f"NOT SCORED: too few strides ({plan.strides}) in {span}", AMBER
    return f"SCORED WALK {span} | {plan.strides} strides counted", GREEN


def ring_radius(age: int, fps: float, size: int) -> int | None:
    """Radius of a heel-strike ring ``age`` frames after the strike, or None once it has faded."""
    if age < 0 or age > RING_SECONDS * fps:
        return None
    return int(size * (0.012 + 0.03 * age / (RING_SECONDS * fps)))


def draw_frame(
    bgr: np.ndarray, image: np.ndarray, visibility: np.ndarray, plan: Plan, frame: int, fps: float
) -> np.ndarray:
    """One frame with the skeleton, the heel-strike rings and the banner drawn on a copy."""
    import cv2

    out = bgr.copy()
    h, w = out.shape[:2]
    scale = max(h, w) / 1000
    inside = plan.start <= frame < plan.stop and plan.state != "no_walk"
    text, border = caption(plan, fps, frame)

    def point(i: int) -> tuple[int, int] | None:
        if np.isnan(image[i]).any() or not visibility[i] >= MIN_VISIBILITY:
            return None
        return int(image[i, 0] * w), int(image[i, 1] * h)

    if not inside:
        out = cv2.addWeighted(out, 0.45, np.full_like(out, 128), 0.55, 0)
    line = max(2, int(3 * scale))
    for a, b, side in BONES:
        pa, pb = point(a), point(b)
        if pa and pb:
            cv2.line(out, pa, pb, COLOURS[side], line, cv2.LINE_AA)
    for i in (*range(11, 17), *range(23, 33), HEAD):
        p = point(i)
        if p:
            cv2.circle(out, p, max(3, int(4.5 * scale)), (255, 255, 255), -1, cv2.LINE_AA)
    if inside:
        for strike, feet in plan.strikes.items():
            radius = ring_radius(frame - strike, fps, max(h, w))
            if radius is None:
                continue
            for foot in feet:
                p = point(LEFT_HEEL if foot == "left" else RIGHT_HEEL)
                if p:
                    cv2.circle(out, p, radius, COLOURS[foot], max(2, int(3 * scale)), cv2.LINE_AA)
                    cv2.circle(out, p, max(3, int(5 * scale)), (0, 0, 255), -1, cv2.LINE_AA)
    thick = max(4, int(8 * scale))
    bar = int(46 * scale)  # the banner is at the top: a video player's controls cover the bottom
    cv2.rectangle(out, (0, 0), (w, bar), (30, 30, 30), -1)
    margin = int(26 * scale)
    if inside and plan.strikes:  # a dark panel keeps the text readable on a bright background
        panel, right = int(110 * scale), int(min(w, 640 * scale))
    else:
        panel, right = int(66 * scale), int(min(w, 220 * scale))
    out[bar : bar + panel, :right] = (out[bar : bar + panel, :right] * 0.4).astype(np.uint8)

    def label(msg: str, y: int, size: float) -> None:  # white with a dark outline
        for colour, extra in (((0, 0, 0), 4), ((255, 255, 255), 0)):
            cv2.putText(out, msg, (margin, y), cv2.FONT_HERSHEY_SIMPLEX, size * scale, colour,
                        max(1, int(2 * scale)) + int(extra * scale), cv2.LINE_AA)  # fmt: skip

    label(text, bar - int(14 * scale), 0.9)
    label(f"{frame / fps:.1f} s", bar + int(52 * scale), 1.0)
    if inside and plan.strikes:
        done = sum(1 for s in plan.strikes if s <= frame)
        label(f"heel strikes counted: {done} of {len(plan.strikes)}  (orange left, blue right)",
              bar + int(92 * scale), 0.7)  # fmt: skip
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), border, thick)  # last, so nothing dims it
    return out


class Sink:
    """Writes frames as H.264 through ffmpeg when it is installed, else as plain mp4v."""

    def __init__(self, path: Path, fps: float, width: int, height: int) -> None:
        self.path, self.proc, self.writer = path, None, None
        path.parent.mkdir(parents=True, exist_ok=True)
        exe = shutil.which("ffmpeg")
        if exe:
            self.proc = subprocess.Popen(
                [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23", "-movflags", "+faststart", str(path)],
                stdin=subprocess.PIPE,
            )  # fmt: skip
        else:
            import cv2

            for code in ("avc1", "mp4v"):  # H.264 where this OpenCV can write it, else plain MPEG-4
                self.writer = cv2.VideoWriter(
                    str(path), cv2.VideoWriter.fourcc(*code), fps, (width, height)
                )
                if self.writer.isOpened():
                    break

    def write(self, frame: np.ndarray) -> None:
        if self.proc and self.proc.stdin:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        elif self.writer:
            self.writer.write(frame)

    def close(self) -> None:
        if self.proc and self.proc.stdin:
            self.proc.stdin.close()
            self.proc.wait()
        elif self.writer:
            self.writer.release()


def fit(bgr: np.ndarray, limit: int = MAX_SIDE) -> np.ndarray:
    """The frame, scaled down (never up) so its longer side is at most ``limit`` pixels."""
    import cv2

    h, w = bgr.shape[:2]
    if max(h, w) <= limit:
        return bgr
    scale = limit / max(h, w)
    size = (max(2, round(w * scale) // 2 * 2), max(2, round(h * scale) // 2 * 2))  # even, for H.264
    return cv2.resize(bgr, size, interpolation=cv2.INTER_LINEAR)


def frames_bgr(path: Path) -> Iterator[np.ndarray]:
    """The video's frames as BGR, read the same way the pose step reads them."""
    for rgb in read_rgb_frames(path):
        yield np.ascontiguousarray(rgb[:, :, ::-1])  # a flipped view makes OpenCV slow


def make_overlay(
    video: Path, out: Path, model: Path = DEFAULT_MODEL, trace_file: Path | None = None
) -> Plan:
    """Write the overlay video for the first ``MAX_SECONDS`` of ``video`` and return its plan.

    The pose trace comes from ``trace_file`` when given, else the pose model is run. Nothing is
    written when the video holds no walk.
    """
    info: VideoInfo = read_info(video)
    trace = (
        load_trace(trace_file)
        if trace_file
        else probe_video(video, model, max_seconds=MAX_SECONDS)[1]
    )
    plan = build_plan(trace)
    if plan.state == "no_walk":
        return plan
    sink: Sink | None = None
    try:
        for i, full in enumerate(frames_bgr(video)):
            if i >= trace.n_frames:
                break
            bgr = fit(full)
            if sink is None:
                height, width = bgr.shape[:2]
                sink = Sink(out, trace.fps if trace.fps > 0 else info.fps, width, height)
            sink.write(draw_frame(bgr, trace.image[i], trace.visibility[i], plan, i, trace.fps))
    finally:
        if sink:
            sink.close()
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Skeleton and heel-strike overlay video")
    parser.add_argument("videos", type=Path, nargs="+")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, help="output file (one video only)")
    parser.add_argument("--trace", type=Path, help="pose trace saved by predict_video (one video)")
    parser.add_argument("--out-dir", type=Path, help="folder for the outputs (default: artifacts)")
    args = parser.parse_args(argv)
    if args.out and len(args.videos) != 1:
        parser.error("--out takes one video; use --out-dir for several")
    if args.trace and len(args.videos) != 1:
        parser.error("--trace takes one video")
    folder = args.out_dir or Path(__file__).resolve().parents[1] / "artifacts" / "overlay"
    status = 0
    for video in args.videos:
        tag = short_id(video.name)  # never the file name
        out = args.out or folder / f"overlay_{tag}.mp4"
        try:
            plan = make_overlay(video, out, args.model, args.trace)
        except (ValueError, OSError) as err:
            print(f"{tag}: {err}")
            status = 2
            continue
        if plan.state == "no_walk":
            print(f"{tag}: no walk found, no overlay written")
            continue
        print(
            f"{tag}: {plan.state}, {plan.strides} strides, {len(plan.strikes)} heel strikes -> {out}"
        )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
