"""Blind manual tap counter (task FT-24): an OpenCV window that plays one clip at a time.

    python models/tapping/src/manual_count_tool.py --selftest
    python models/tapping/src/manual_count_tool.py --rater R1 [--practice] [--resume]
    python models/tapping/src/manual_count_tool.py --rater R2 [--hard] [--resume]

Count by the rule in manual_counts.py. The window (title "FT-24 counter") shows only the clip
number (k of N), the time, your own marks, edge partials, confidence and note state, the speed and
the keys. The tool never shows or prints a ref, file name, cell, rate, QC status or detector
count, and reads none of them: selection.csv is read for ref and list membership only.

Keys: space play/pause | , . one frame back/forward | [ ] slower/faster (0.125x to 1x, start
0.5x) | m mark a tap at this frame and pause (a held key marks once) | u undo the last mark or edge | e add an edge partial | 1 2 3
confidence | n type a note (Enter keeps it, Esc cancels) | s save and next clip | q quit (the
clip in progress is kept for --resume).

Each clip is decoded ONCE front to back into an in-memory JPEG cache (upright, long side 960 px,
quality 85) with its pts times from that same read; seeking is never used (it returned wrong
frames on the 60 fps clips). Playback follows the wall clock, not waitKey delays. Every change is
written at once to a journal, so a crash loses nothing. A clip that cannot be decoded is listed
in unreadable.csv for ``manual_counts.py select --replace``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import manual_counts as mc
import numpy as np
from hubu_fis import HubuFisError, list_videos, pseudonym
from scipy.signal import find_peaks
from signal_processing import is_inside

TOOL_VERSION = "ft24-counter-0.1.0"
WINDOW_TITLE = "FT-24 counter"
SPEEDS = (0.125, 0.25, 0.5, 1.0)
START_SPEED = 2  # index into SPEEDS: 0.5x
LONG_SIDE = 960
JPEG_QUALITY = 85
CANVAS_MIN_WIDTH = 720
LINE_HEIGHT = 22
HELP = (
    "space play/pause   , . step   [ ] speed   m mark (pauses)   u undo   e edge",
    "1 2 3 confidence   n note   s save + next   q quit (resume later)",
)
TOOL_COLUMNS = ("ref", "set", "second_rater", "replaced_for")  # all the tool reads
QUIT_KEY = ord("q")


class DecodeError(RuntimeError):
    """A video cannot be decoded into a usable clip."""


class SessionError(ValueError):
    """The session cannot start (exit code 2)."""


# --------------------------------------------------------------------------- frame cache


@dataclass
class Clip:
    """Upright JPEG frames in decode order and their pts times (s from the first frame)."""

    frames: list[bytes]
    times: np.ndarray

    def __len__(self) -> int:
        return len(self.frames)

    def image(self, index: int) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(self.frames[index], np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise DecodeError("a cached frame could not be decoded")
        return image


def encode(frame: np.ndarray) -> bytes:
    h, w = frame.shape[:2]
    scale = LONG_SIDE / max(h, w)
    if scale < 1:
        frame = cv2.resize(
            frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
        )
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise DecodeError("a frame could not be encoded")
    return bytes(buf.tobytes())


def make_clip(frames: list[bytes], pts_ms: Sequence[float]) -> Clip:
    if len(frames) < 2:
        raise DecodeError("fewer than 2 frames")
    pts = np.round(np.asarray(pts_ms, dtype=float), 3)
    if not np.isfinite(pts).all() or not (np.diff(pts) > 0).all():
        raise DecodeError("the frame times are not strictly increasing")
    return Clip(frames, (pts - pts[0]) / 1000.0)


def decode_clip(path: Path) -> Clip:
    """Read the whole video once, in order; pts from the same read as each frame."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise DecodeError("the video cannot be opened")
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        frames: list[bytes] = []
        pts: list[float] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            pts.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)))
            frames.append(encode(frame))
    finally:
        cap.release()
    return make_clip(frames, pts)


# --------------------------------------------------------------------------- player and marks


@dataclass
class Player:
    """Clock-driven playback over a clip's pts times."""

    times: np.ndarray
    index: int = 0
    playing: bool = False
    speed_i: int = START_SPEED
    _wall0: float = 0.0
    _media0: float = 0.0

    @property
    def speed(self) -> float:
        return SPEEDS[self.speed_i]

    def play(self, now: float) -> None:
        if self.index >= len(self.times) - 1:
            self.index = 0
        self.playing, self._wall0, self._media0 = True, now, float(self.times[self.index])

    def toggle(self, now: float) -> None:
        if self.playing:
            self.playing = False
        else:
            self.play(now)

    def step(self, delta: int) -> None:
        self.playing = False
        self.index = int(np.clip(self.index + delta, 0, len(self.times) - 1))

    def change_speed(self, delta: int, now: float) -> None:
        self.speed_i = int(np.clip(self.speed_i + delta, 0, len(SPEEDS) - 1))
        if self.playing:
            self.play(now)

    def tick(self, now: float) -> int:
        if self.playing:
            target = self._media0 + (now - self._wall0) * self.speed
            i = int(np.searchsorted(self.times, target, side="right")) - 1
            if i >= len(self.times) - 1:
                i, self.playing = len(self.times) - 1, False
            self.index = max(i, 0)
        return self.index


@dataclass
class Count:
    """The rater's input for one clip; ``actions`` is the undo stack (["m", frame] or ["e"])."""

    actions: list[list[Any]] = field(default_factory=list)
    confidence: int | None = None
    note: str = ""
    started_utc: str = ""

    def marks(self) -> list[int]:
        return sorted({int(a[1]) for a in self.actions if a[0] == "m"})

    def edges(self) -> int:
        return sum(a[0] == "e" for a in self.actions)

    def mark(self, frame: int) -> bool:
        if frame in self.marks():
            return False
        self.actions.append(["m", int(frame)])
        return True

    def edge(self) -> None:
        self.actions.append(["e"])

    def undo(self) -> bool:
        return bool(self.actions) and self.actions.pop() is not None

    def to_json(self) -> dict[str, Any]:
        return {"actions": self.actions, "confidence": self.confidence, "note": self.note}

    @classmethod
    def from_json(cls, data: dict[str, Any], started_utc: str) -> Count:
        return cls(list(data["actions"]), data["confidence"], str(data["note"]), started_utc)


class Journal:
    """The clip in progress, rewritten after every change; ``path=None`` keeps it in memory."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.memory: dict[str, Any] | None = None

    def save(self, data: dict[str, Any]) -> None:
        if self.path is None:
            self.memory = data
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, self.path)

    def load(self) -> dict[str, Any] | None:
        if self.path is None:
            return self.memory
        if not self.path.is_file():
            return None
        data: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        return data

    def clear(self) -> None:
        self.memory = None
        if self.path is not None:
            with contextlib.suppress(FileNotFoundError):
                self.path.unlink()


# --------------------------------------------------------------------------- display


class UI(Protocol):
    def show(self, image: np.ndarray) -> None: ...
    def key(self, wait_ms: int) -> int: ...
    def close(self) -> None: ...


class CvUI:
    """The real OpenCV window."""

    title = WINDOW_TITLE

    def __init__(self) -> None:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)

    def show(self, image: np.ndarray) -> None:
        cv2.imshow(WINDOW_TITLE, image)

    def key(self, wait_ms: int) -> int:
        k = cv2.waitKey(wait_ms)
        if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
            return QUIT_KEY  # the window was closed: keep the progress and stop
        return k & 0xFF if k >= 0 else -1

    def close(self) -> None:
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def overlay_lines(
    k: int, n: int, player: Player, count: Count, typing: str | None, message: str
) -> list[str]:
    """Everything the window writes. Nothing here may name a clip, cell or detector result."""
    i = player.index
    marks = count.marks()
    here = "   [this frame is marked]" if i in marks else ""
    lines = [
        f"clip {k} of {n}   {player.times[i]:6.2f} s of {player.times[-1]:.2f} s   "
        f"frame {i} (0-{len(player.times) - 1})   speed {player.speed:g}x   "
        + ("PLAY" if player.playing else "PAUSED"),
        f"taps marked {len(marks)}{here}   edge partials {count.edges()}   "
        f"confidence {count.confidence or '-'}   note {'yes' if count.note else 'no'}",
    ]
    if typing is not None:
        lines.append(f"note: {typing}_   (Enter keeps, Esc cancels)")
    if message:
        lines.append(message)
    return lines + list(HELP)


def compose(image: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    h, w = image.shape[:2]
    width = max(w, CANVAS_MIN_WIDTH)
    canvas = np.zeros((h + LINE_HEIGHT * len(lines) + 12, width, 3), np.uint8)
    canvas[:h, (width - w) // 2 : (width - w) // 2 + w] = image
    for j, text in enumerate(lines):
        y = h + 18 + LINE_HEIGHT * j
        cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1)
    return canvas


# --------------------------------------------------------------------------- one clip


@dataclass
class Keys:
    """Key handling without a window: ``handle`` applies one key and returns "save", "quit" or None."""

    player: Player
    count: Count
    on_change: Callable[[], None]
    typing: str | None = None
    message: str = ""

    def handle(self, key: int, now: float, shown: int) -> str | None:
        """``shown`` is the frame index on screen when the key arrived; a mark uses it."""
        count, player = self.count, self.player
        self.message = ""
        if self.typing is not None:
            self._type(key)
            return None
        ch = chr(key) if 0 <= key < 256 else ""
        if ch == " ":
            player.toggle(now)
        elif ch in (",", "."):
            player.step(-1 if ch == "," else 1)
        elif ch in ("[", "]"):
            player.change_speed(-1 if ch == "[" else 1, now)
        elif ch == "m":
            # Pause on the marked frame: a held key then repeats on the same frame and is ignored.
            player.playing, player.index = False, shown
            if count.mark(shown):
                self.on_change()
            else:
                self.message = "this frame is already marked"
        elif ch == "u":
            if count.undo():
                self.on_change()
            else:
                self.message = "nothing to undo"
        elif ch == "e":
            count.edge()
            self.on_change()
        elif ch in ("1", "2", "3"):
            count.confidence = int(ch)
            self.on_change()
        elif ch == "n":
            self.typing = count.note
        elif ch == "s":
            if count.confidence is None:
                self.message = "set the confidence (1, 2 or 3) before saving"
            else:
                return "save"
        elif ch == "q":
            self.on_change()
            return "quit"
        return None

    def _type(self, key: int) -> None:
        typing = self.typing or ""
        if key in (10, 13):
            problem = mc.note_problem(typing)
            if problem:
                self.message = f"note not kept: {problem}"
            else:
                self.count.note, self.typing = typing.strip(), None
                self.on_change()
        elif key == 27:
            self.typing = None
        elif key in (8, 127):
            self.typing = typing[:-1]
        elif 32 <= key < 127:
            self.typing = typing + chr(key)


def count_clip(
    ui: UI,
    clip: Clip,
    count: Count,
    k: int,
    n: int,
    on_change: Callable[[], None],
    clock: Callable[[], float],
) -> str:
    """Play and count one clip until the rater saves ("save") or quits ("quit")."""
    keys = Keys(Player(clip.times), count, on_change)
    while True:
        now = clock()
        i = keys.player.tick(now)
        lines = overlay_lines(k, n, keys.player, count, keys.typing, keys.message)
        ui.show(compose(clip.image(i), lines))
        key = ui.key(10)
        if key < 0:
            continue
        done = keys.handle(key, now, i)
        if done:
            return done


def result_row(rater: str, k: int, ref: str, clip: Clip, count: Count) -> dict[str, object]:
    marks = count.marks()
    return {
        "rater": rater,
        "order": k,
        "ref": ref,
        "n_taps": len(marks),
        "tap_frames": ";".join(str(f) for f in marks),
        "tap_times_s": ";".join(f"{clip.times[f]:.3f}" for f in marks),
        "n_edge_partials": count.edges(),
        "confidence": count.confidence,
        "note": count.note,
        "started_utc": count.started_utc,
        "finished_utc": mc.utc_now(),
        "tool_version": TOOL_VERSION,
    }


# --------------------------------------------------------------------------- session


def video_loader(video_dir: Path) -> Callable[[str], Clip]:
    try:
        videos, _ = list_videos(Path(video_dir))
    except (HubuFisError, OSError) as err:
        raise SessionError("cannot list the video folder") from err
    if not videos:
        raise SessionError("the video folder has no videos")
    by_ref = {pseudonym(stem): path for stem, path in videos.items()}

    def load(ref: str) -> Clip:
        if ref not in by_ref:
            raise DecodeError("no video file")
        return decode_clip(by_ref[ref])

    return load


def run_session(
    rater: str,
    kind: str,
    resume: bool,
    *,
    out_dir: Path,
    ui: UI,
    load: Callable[[str], Clip],
    clock: Callable[[], float] = time.monotonic,
) -> int:
    sel = mc.read_selection(out_dir, TOOL_COLUMNS)
    refs = mc.list_refs(sel, rater, kind)
    path = mc.rater_path(out_dir, rater, kind)
    journal = Journal(path.with_name(f".{path.stem}.journal.json"))
    if (path.exists() or journal.load() is not None) and not resume:
        raise SessionError(f"a {rater} {kind} session already exists; add --resume to continue it")
    done = set(mc.read_rows(path)["ref"])
    n = len(refs)
    for k, ref in enumerate(refs, start=1):
        if ref in done or ref in mc.read_unreadable(out_dir):
            continue
        try:
            clip = load(ref)
        except DecodeError:
            mc.add_unreadable(out_dir, ref, rater, kind)
            print(
                f"clip {k} of {n} could not be decoded: noted for manual_counts.py select --replace"
            )
            continue
        saved = journal.load()
        if saved is not None and saved.get("ref") == ref:
            count = Count.from_json(saved["count"], saved["started_utc"])
        else:
            count = Count(started_utc=mc.utc_now())

        def keep(ref: str = ref, count: Count = count) -> None:
            journal.save({"ref": ref, "started_utc": count.started_utc, "count": count.to_json()})

        keep()
        print(f"clip {k} of {n}: counting")
        if count_clip(ui, clip, count, k, n, keep, clock) == "quit":
            print(f"stopped at clip {k} of {n}; your marks are kept: run again with --resume")
            return 0
        mc.append_row(path, result_row(rater, k, ref, clip, count))
        journal.clear()
        print(f"clip {k} of {n} saved")
    left = [r for r in refs if r not in set(mc.read_rows(path)["ref"])]
    print(f"list finished: {n - len(left)} of {n} clips saved, {len(left)} not counted")
    return 0


# --------------------------------------------------------------------------- self-test

# Keyframes (seconds, aperture) of the synthetic clip, eased with a cosine between keyframes:
# three plain taps, a double bump with a clear reversal (two taps), a hold-open tap (one), one
# more tap, then an opening still rising at the last frame (an edge partial, not a tap).
SELFTEST_KEYS = (
    (0.0, 0.08), (0.4, 0.08), (0.7, 1.0), (1.0, 0.08), (1.3, 1.0), (1.6, 0.08), (1.9, 1.0),
    (2.2, 0.08), (2.5, 1.0), (2.75, 0.55), (3.0, 0.95), (3.3, 0.08), (3.6, 1.0), (4.4, 1.0),
    (4.7, 0.08), (5.0, 1.0), (5.3, 0.08), (5.6, 0.08), (6.3, 1.0),
)  # fmt: skip
SELFTEST_FPS = 30.0
SELFTEST_SECONDS = 6.0
SELFTEST_TAPS = 7
REPEAT_FRAMES = 3  # marks closer than this look like key auto-repeat
SELFTEST_EDGES = 1
SELFTEST_PEAK_TIMES = (0.7, 1.3, 1.9, 2.5, 3.0, 4.0, 5.0)  # the hold peak is its middle


def selftest_aperture() -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(round(SELFTEST_SECONDS * SELFTEST_FPS) + 1) / SELFTEST_FPS
    kt, ka = np.array(SELFTEST_KEYS).T
    j = np.clip(np.searchsorted(kt, t, side="right") - 1, 0, len(kt) - 2)
    u = (t - kt[j]) / (kt[j + 1] - kt[j])
    return t, ka[j] + (ka[j + 1] - ka[j]) * (1 - np.cos(np.pi * u)) / 2


def selftest_peaks(aperture: np.ndarray) -> np.ndarray:
    peaks, _ = find_peaks(aperture, prominence=0.05)
    return peaks


def make_selftest_clip() -> Clip:
    """Two dots (thumb and index tip) on a dark frame, in memory only."""
    t, aperture = selftest_aperture()
    frames = []
    for a in aperture:
        img = np.full((960, 540, 3), 40, np.uint8)
        palm, thumb = (390, 700), (250, 640)
        index = (250, int(600 - 380 * a))
        cv2.line(img, palm, thumb, (150, 180, 220), 22)
        cv2.line(img, palm, index, (150, 180, 220), 22)
        cv2.circle(img, thumb, 16, (60, 200, 255), -1)
        cv2.circle(img, index, 16, (60, 200, 255), -1)
        frames.append(encode(img))
    return make_clip(frames, (t * 1000).tolist())


def selftest_diagnosis(marks: Sequence[int], expected: Sequence[int]) -> list[str]:
    """Per mark: the nearest expected peak frame and the offset; marks closer than 3 frames."""
    lines = []
    for m in marks:
        near = min(expected, key=lambda e: (abs(m - e), e))
        lines.append(f"  mark {m}: nearest expected {near}, offset {m - near:+d}")
    close = [(a, b) for a, b in zip(marks, marks[1:], strict=False) if b - a < REPEAT_FRAMES]
    if close:
        lines.append(f"  possible key repeat: marks closer than {REPEAT_FRAMES} frames: {close}")
    return lines


def selftest(ui: UI, clock: Callable[[], float]) -> int:
    clip = make_selftest_clip()
    count = Count(started_utc=mc.utc_now())
    count_clip(ui, clip, count, 1, 1, lambda: None, clock)
    marks = count.marks()
    expected = [int(round(s * SELFTEST_FPS)) for s in SELFTEST_PEAK_TIMES]
    print(
        f"selftest: expected {SELFTEST_TAPS} taps and {SELFTEST_EDGES} edge partial; you counted "
        f"{len(marks)} taps and {count.edges()} edge partials"
    )
    print(f"  expected peak frames: {expected}")
    print(f"  your marked frames:   {marks}")
    ok = len(marks) == SELFTEST_TAPS and count.edges() == SELFTEST_EDGES
    if not ok:
        for line in selftest_diagnosis(marks, expected):
            print(line)
    print("selftest passed" if ok else "selftest: counts differ, read the counting rule again")
    return 0 if ok else 1


# --------------------------------------------------------------------------- command line


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--selftest", action="store_true", help="count a synthetic clip")
    ap.add_argument("--rater", choices=mc.RATERS)
    which = ap.add_mutually_exclusive_group()
    which.add_argument("--practice", action="store_true", help="R1 practice clips")
    which.add_argument("--hard", action="store_true", help="R2 hard set")
    ap.add_argument("--resume", action="store_true", help="continue a started list")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--video-dir", type=Path, default=None)
    return ap


def main(
    argv: Sequence[str] | None = None,
    ui_factory: Callable[[], UI] = CvUI,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    args = build_parser().parse_args(argv)
    if not args.selftest and args.rater is None:
        print("error: give --rater R1|R2 or --selftest", file=sys.stderr)
        return 2
    kind = "practice" if args.practice else "hard" if args.hard else "main"
    out_dir = Path(args.out_dir or mc.INTERIM_ROOT / "manual_counts")
    if not is_inside(out_dir, mc.INTERIM_ROOT):
        print("error: --out-dir must be inside data/interim", file=sys.stderr)
        return 2
    ui: UI | None = None
    try:
        if args.selftest:
            ui = ui_factory()
            return selftest(ui, clock)
        if (args.rater, kind) not in mc.LISTS:
            print(f"error: {args.rater} does not count the {kind} list", file=sys.stderr)
            return 2
        load = video_loader(Path(args.video_dir or mc.VIDEO_ROOT))
        ui = ui_factory()
        return run_session(
            args.rater, kind, args.resume, out_dir=out_dir, ui=ui, load=load, clock=clock
        )
    except (SessionError, mc.InputError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("stopped; your marks are kept: run again with --resume", file=sys.stderr)
        return 1
    except Exception as err:  # noqa: BLE001 - the message may name a file: show the type only
        print(
            f"error: {type(err).__name__} (details hidden: they may name a file)", file=sys.stderr
        )
        return 2
    finally:
        if ui is not None:
            ui.close()


if __name__ == "__main__":
    sys.exit(main())
