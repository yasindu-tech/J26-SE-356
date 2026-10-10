"""GAIT-12: run preprocessing and heel-strike detection on every CARE-PD walk.

Run from the repo root (venv active):

    python models/gait/src/check_heel_strikes.py

Prints per cohort: strikes per foot, stride time, how many walks have fewer than 3 strides, and
how closely the heel strikes agree with an independent coordinate-based method (heel strike =
ankle furthest in front of the pelvis, Zeni et al. 2008), computed on walks that travel at
least 0.5 m roughly straight (the comparison fails on walks that turn). Saves an overlay plot (one walk per cohort) to
``models/gait/artifacts/heel_strikes/heel_strike_overlay.png`` (git-ignored).

Exit code 0 = every cohort's median stride time is plausible (0.8 to 2.0 s), 1 = it is not,
2 = a file is missing. Walks are shown by a short hash only, never by subject.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from carepd_format import L_ANKLE, PELVIS, R_ANKLE, FormatConfig, load_config
from carepd_loader import DEFAULT_CONFIG, DEFAULT_DATA, Walk, load_all, pick_walk_for_plot
from preprocess import HeelStrikes, ankle_heights, centre_on_pelvis, detect_heel_strikes, preprocess
from scipy.signal import find_peaks

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "artifacts" / "heel_strikes"
PLAUSIBLE_STRIDE_S = (0.8, 2.0)
MIN_STRIDES = 3  # GAIT-13 will flag walks below this; here it is only counted
MIN_TRAVEL_M = 0.5  # the coordinate-based comparison needs a clear walking direction
MIN_STRAIGHTNESS = 0.8  # and a roughly straight walk: net distance / path length of the pelvis
MATCH_WINDOW_S = 0.4


@dataclass(frozen=True)
class CohortSummary:
    name: str
    n_walks: int
    strikes_per_foot: float  # median over feet
    stride_s: tuple[float, float, float]  # 25th, 50th, 75th percentile
    walks_few_strides: int  # walks where neither foot has MIN_STRIDES strides
    offset_ms: tuple[float, float, float] | None  # vs coordinate method: 25th, 50th, 75th
    stride_s_coordinate: float | None  # median stride time from the coordinate method


def coordinate_heel_strikes(walk: Walk) -> tuple[np.ndarray, np.ndarray] | None:
    """Heel strikes as the ankle's furthest point in front of the pelvis, per foot.

    Needs one walking direction, taken from the pelvis path; returns None if the walk travels
    less than ``MIN_TRAVEL_M`` or turns (net distance below ``MIN_STRAIGHTNESS`` of the path
    length). Used only to check ``detect_heel_strikes``.
    """
    pelvis = walk.joints[:, PELVIS][:, [0, 2]]
    travel = pelvis[-1] - pelvis[0]
    dist = float(np.linalg.norm(travel))
    path = float(np.linalg.norm(np.diff(pelvis, axis=0), axis=1).sum())
    if dist < MIN_TRAVEL_M or dist < MIN_STRAIGHTNESS * path:
        return None  # too short, or the walk turns: one fixed direction would be wrong
    forward = travel / dist
    prepared = preprocess(walk.joints, walk.fps)
    out = []
    for idx in (L_ANKLE, R_ANKLE):
        ahead = prepared[:, idx, 0] * forward[0] + prepared[:, idx, 2] * forward[1]
        peaks, _ = find_peaks(ahead, distance=max(int(0.5 * walk.fps), 1), prominence=0.1)
        out.append(peaks)
    return out[0], out[1]


def _offsets_ms(found: np.ndarray, reference: np.ndarray, fps: float) -> list[float]:
    if found.size == 0 or reference.size == 0:
        return []
    out = []
    for f in found:
        j = int(np.argmin(np.abs(reference - f)))
        if abs(int(reference[j]) - int(f)) <= MATCH_WINDOW_S * fps:
            out.append((int(f) - int(reference[j])) / fps * 1000)
    return out


def summarise(name: str, walks: list[Walk]) -> CohortSummary:
    per_foot: list[int] = []
    strides: list[float] = []
    coord_strides: list[float] = []
    offsets: list[float] = []
    few = 0
    for w in walks:
        hs = detect_heel_strikes(w.joints, w.fps)
        feet = (hs.left, hs.right)
        per_foot += [f.strikes.size for f in feet]
        strides += list(hs.all_stride_times())
        if max(f.strikes.size - 1 for f in feet) < MIN_STRIDES:
            few += 1
        ref = coordinate_heel_strikes(w)
        if ref is not None:
            for foot, r in zip(feet, ref, strict=True):
                offsets += _offsets_ms(foot.strikes, r, w.fps)
                coord_strides += list(np.diff(r) / w.fps)
    q = np.percentile(strides, [25, 50, 75]) if strides else np.full(3, np.nan)
    o = np.percentile(offsets, [25, 50, 75]) if offsets else None
    return CohortSummary(
        name=name,
        n_walks=len(walks),
        strikes_per_foot=float(np.median(per_foot)) if per_foot else 0.0,
        stride_s=(float(q[0]), float(q[1]), float(q[2])),
        walks_few_strides=few,
        offset_ms=None if o is None else (float(o[0]), float(o[1]), float(o[2])),
        stride_s_coordinate=float(np.median(coord_strides)) if coord_strides else None,
    )


def plot_overlay(examples: list[Walk], out_png: Path) -> None:
    """One row per walk: ankle height vs pelvis, raw and smoothed, with heel strikes marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(examples), 1, figsize=(11, 2.6 * len(examples)), squeeze=False)
    for ax, w in zip(axes[:, 0], examples, strict=True):
        t = np.arange(w.n_frames) / w.fps
        raw = centre_on_pelvis(w.joints)
        left, right = ankle_heights(preprocess(w.joints, w.fps))
        hs: HeelStrikes = detect_heel_strikes(w.joints, w.fps)
        for y_raw, y, foot, colour, label in (
            (raw[:, L_ANKLE, 1], left, hs.left, "tab:blue", "left"),
            (raw[:, R_ANKLE, 1], right, hs.right, "tab:red", "right"),
        ):
            ax.plot(t, y_raw, color=colour, lw=0.8, alpha=0.3)
            ax.plot(t, y, color=colour, lw=1.4, label=f"{label} ankle (smoothed)")
            ax.plot(
                t[foot.strikes],
                y[foot.strikes],
                "v",
                color=colour,
                ms=8,
                label=f"{label} heel strike",
            )
            ax.plot(t[foot.minima], y[foot.minima], "x", color=colour, ms=5, alpha=0.6)
        ax.set_title(
            f"{w.cohort} walk {w.tag}: {w.fps} fps, {w.n_frames} frames, UPDRS gait {w.score}",
            fontsize=10,
        )
        ax.set_ylabel("ankle height\nvs pelvis (m)")
    axes[0, 0].legend(loc="upper right", fontsize=7, ncol=2)
    axes[-1, 0].set_xlabel(
        "time (s)    (faint = before smoothing; triangle = heel strike; x = lowest point of the dip)"
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Heel strikes on CARE-PD (GAIT-12)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    cfg: FormatConfig = load_config(args.config)
    try:
        walks, _ = load_all(args.data_dir, cfg)
    except FileNotFoundError as err:
        print(f"MISSING: {err}")
        return 2

    by_cohort: dict[str, list[Walk]] = {}
    for w in walks:
        by_cohort.setdefault(w.cohort, []).append(w)

    print("Heel strikes on pelvis-centred, smoothed CARE-PD walks (start of each ankle dip)")
    print(
        f"{'cohort':<10}{'walks':>7}{'strikes/foot':>14}{'stride time (s)':>20}"
        f"{'< 3 strides':>13}{'vs coordinate method (ms)':>28}"
    )
    status = 0
    for name, ws in by_cohort.items():
        s = summarise(name, ws)
        q1, med, q3 = s.stride_s
        off = (
            "-"
            if s.offset_ms is None
            else f"{s.offset_ms[1]:+.0f} ({s.offset_ms[0]:+.0f} to {s.offset_ms[2]:+.0f})"
        )
        print(
            f"{name:<10}{s.n_walks:>7}{s.strikes_per_foot:>14.1f}"
            f"{f'{med:.2f} ({q1:.2f}-{q3:.2f})':>20}{s.walks_few_strides:>13}{off:>28}"
        )
        if not PLAUSIBLE_STRIDE_S[0] <= med <= PLAUSIBLE_STRIDE_S[1]:
            print(f"  -> {name}: median stride time {med:.2f} s is outside {PLAUSIBLE_STRIDE_S} s")
            status = 1
    print(
        "\nstride time: median (25th-75th percentile) over both feet. Coordinate method: heel strike"
        "\n= ankle furthest in front of the pelvis; offsets are median (25th to 75th), negative ="
        f"\nour strike comes earlier. '< 3 strides': walks with fewer than {MIN_STRIDES} strides on"
        "\nboth feet (GAIT-13 decides what to do with them)."
    )

    if not args.no_plot:
        examples = [pick_walk_for_plot(ws) for ws in by_cohort.values()]
        out = args.out_dir / "heel_strike_overlay.png"
        plot_overlay(examples, out)
        print(f"\nSaved overlay plot: {out}")
    return status


if __name__ == "__main__":
    sys.exit(main())
