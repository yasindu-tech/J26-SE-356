"""GAIT-26: decide whether a video's walk is good enough to score, and say why not if it is not.

Takes the result of ``video_to_joints`` (or the ``VideoError`` it raised) and returns a
``GateResult``: passed or not, plain reasons, warnings, and each check's value beside its limit.
A rejected video gets no score. Limits are judgement, set outside the range of the eight KOA-PD-NM
clips tried (see the constants); ``people_in_view`` is where a person counter plugs in later.

    python models/gait/src/quality_gate.py VIDEO [--min-strides 3] [--people N]

Exit code 0 = passed, 1 = rejected.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from carepd_format import short_id
from cycles import CAREPD_MIN_STRIDES, segment_cycles
from mediapipe_mapping import MIN_FACING_AGREEMENT
from preprocess import detect_heel_strikes
from video_to_joints import DEFAULT_MODEL, MAX_SECONDS, VideoError, VideoWalk, video_to_joints

MIN_LEG_VISIBILITY = 0.80  # the good clips had 0.92 or more
MAX_FILLED_SHARE = 0.20  # the good clips had 0.08 or less
MIN_FPS = 15.0  # no data behind this one: step timing needs enough frames per second
ONE_PERSON_ASSUMED = "One person in view is assumed: the video was not checked for other people."


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: tuple[str, ...] = ()  # why it was rejected, in plain words
    warnings: tuple[str, ...] = ()  # shown with the score; they do not stop it
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)  # name -> value, limit, ok

    def as_record(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "checks": {k: dict(v) for k, v in self.checks.items()},
        }


def reject(error: VideoError) -> GateResult:
    """A video that could not even give a walk, as a rejected result with the reason."""
    return GateResult(False, reasons=(str(error),))


def check_walk(
    walk: VideoWalk,
    *,
    min_strides: int = CAREPD_MIN_STRIDES,
    people_in_view: int | None = None,
) -> GateResult:
    """Judge one walk. ``people_in_view`` is a count from a person detector, if one was run."""
    if people_in_view is not None and people_in_view < 0:
        raise ValueError("people_in_view cannot be negative")
    strides: int | None = None  # below MIN_FPS the step filter cannot run: the rate is the reason
    if walk.fps >= MIN_FPS:
        try:
            strides = segment_cycles(
                detect_heel_strikes(walk.joints, walk.fps), min_strides
            ).strides()
        except ValueError:  # too few frames to filter: no steps can be found
            strides = 0
    filled = walk.filled_frames / walk.joints.shape[0]
    rules = [  # name, value, limit, ok, message when it fails
        (
            "frame_rate",
            walk.fps,
            MIN_FPS,
            walk.fps >= MIN_FPS,
            f"The video has {walk.fps:.0f} frames per second; at least {MIN_FPS:.0f} are needed "
            "to time the steps.",
        ),
        (
            "leg_visibility",
            walk.leg_visibility,
            MIN_LEG_VISIBILITY,
            walk.leg_visibility >= MIN_LEG_VISIBILITY,
            f"The legs were clearly visible in only {walk.leg_visibility:.0%} of the walk "
            f"(at least {MIN_LEG_VISIBILITY:.0%} is needed).",
        ),
        (
            "filled_share",
            filled,
            MAX_FILLED_SHARE,
            filled <= MAX_FILLED_SHARE,
            f"{filled:.0%} of the walk had to be filled in where the person was lost from view "
            f"(at most {MAX_FILLED_SHARE:.0%} is allowed).",
        ),
        (
            "facing",
            walk.facing_agreement,
            MIN_FACING_AGREEMENT,
            walk.facing_agreement >= MIN_FACING_AGREEMENT,
            "The direction the person faces could not be worked out reliably (toes ahead of "
            f"heels in {walk.facing_agreement:.0%} of frames; at least "
            f"{MIN_FACING_AGREEMENT:.0%} is needed).",
        ),
    ]
    if strides is not None:
        rules.append(
            (
                "strides",
                strides,
                min_strides,
                strides >= min_strides,
                f"The walk has {strides} real strides on both feet; at least {min_strides} are needed.",
            )
        )
    warnings: list[str] = []
    if people_in_view is None:
        warnings.append(ONE_PERSON_ASSUMED)
    else:
        rules.append(
            (
                "people",
                people_in_view,
                1,
                people_in_view <= 1,
                f"{people_in_view} people are in view; the video must show one person walking.",
            )
        )
    if walk.truncated:
        warnings.append(
            f"Only the first {walk.analysed_seconds:.0f} s of the {walk.video_seconds:.0f} s "
            "video were read."
        )
    if walk.stretches > 1:
        warnings.append(
            f"{walk.stretches - 1} other stretch(es) of usable video ({walk.unused_seconds:.1f} s) "
            "were not used: only the longest walk is scored."
        )
    return GateResult(
        passed=all(ok for _, _, _, ok, _ in rules),
        reasons=tuple(msg for _, _, _, ok, msg in rules if not ok),
        warnings=tuple(warnings),
        checks={
            name: {"value": float(value), "limit": float(limit), "ok": bool(ok)}
            for name, value, limit, ok, _ in rules
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quality gate for one video (GAIT-26)")
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    parser.add_argument("--min-strides", type=int, default=CAREPD_MIN_STRIDES)
    parser.add_argument("--people", type=int, help="people counted in the video, if known")
    args = parser.parse_args(argv)

    try:
        walk = video_to_joints(args.video, args.model, max_seconds=args.max_seconds)
        result = check_walk(walk, min_strides=args.min_strides, people_in_view=args.people)
    except VideoError as err:
        result = reject(err)
    print(f"{short_id(args.video.name)}: {'PASSED' if result.passed else 'REJECTED'}")
    for name, c in result.checks.items():
        print(
            f"  {name:<15}{c['value']:>8.2f}  limit {c['limit']:.2f}  {'ok' if c['ok'] else 'FAIL'}"
        )
    for line in result.reasons:
        print(f"  reason: {line}")
    for line in result.warnings:
        print(f"  warning: {line}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
