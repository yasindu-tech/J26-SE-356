"""Per-tap amplitude curve for the demo (task FT-50), no streamlit import. The taps and the
fitted line come from the research code: tap table (tap_table), valid taps and the Theil-Sen fit
of features_b (the same routine that gives amp_slope). Texts are fixed templates, no free text.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import features_b as fb  # noqa: E402
import numpy as np  # noqa: E402
import tap_table  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

MIN_TAPS = fb.MIN_TAPS  # the official minimum of valid taps for a slope
# Display convention for the wording only (2% of the mean per 10 s); not a clinical cut-off.
FLAT_BAND = 0.02
POINT_COLOUR, LINK_COLOUR, LINE_COLOUR = "#1b4f9c", "#9db4d6", "#c4622d"
NOTE = (
    "The score above uses Set A, which has no amplitude-trend feature. "
    "This curve explains the signal; it did not change the score."
)
REASONS = {
    "few": f"Fewer than {MIN_TAPS} taps with a valid amplitude, so no line is fitted.",
    "mean": "The mean amplitude is not positive, so no line is fitted.",
    "missing": "The per-tap curve is not available for this clip.",
}


@dataclass(eq=False)
class TapCurve:
    times: np.ndarray  # s, tap peak times of the plotted taps (plus the display offset)
    amps: np.ndarray  # amplitude in hand sizes
    slope_per10s_frac: float  # Theil-Sen slope as a fraction of the mean amplitude per 10 s
    line_x: np.ndarray
    line_y: np.ndarray
    n_taps: int  # taps plotted
    n_found: int  # valid taps given
    n_nan: int  # taps left out for a missing amplitude
    usable: bool
    reason: str
    n_edge: int = 0  # taps left out by the research code as one-sided (amplitude from one side)


def tap_curve(
    tap_times: Any, tap_amps: Any, time_offset_s: float = 0.0, n_all: int | None = None
) -> TapCurve:
    """Points and the fitted line; the slope is the official amp_slope routine on the same arrays.

    A tap without an amplitude is left out (never read as 0). The line is drawn only when the
    official minimum number of taps is present. ``n_all`` = all taps of the clip, if the given
    ones are only its valid taps (the difference is reported as left out)."""
    t = np.asarray(tap_times, dtype=float)
    a = np.asarray(tap_amps, dtype=float)
    keep = np.isfinite(t) & np.isfinite(a)
    t, a = t[keep], a[keep]
    given = len(keep)
    base = {"times": t + time_offset_s, "amps": a, "n_taps": len(t),
            "n_found": given if n_all is None else n_all, "n_nan": int((~keep).sum()),
            "n_edge": 0 if n_all is None else n_all - given}  # fmt: skip
    empty = np.empty(0)
    slope, why = fb.slope_per_10s(t, a)
    if why is not None:
        key = "mean" if why.startswith("mean") else "few"
        return TapCurve(**base, slope_per10s_frac=float("nan"), line_x=empty, line_y=empty,
                        usable=False, reason=REASONS[key])  # fmt: skip
    fit_slope, intercept = fb.theil_sen_line(t, a)
    x = np.array([t[0], t[-1]])
    return TapCurve(**base, slope_per10s_frac=float(slope), line_x=x + time_offset_s,
                    line_y=intercept + fit_slope * x, usable=True, reason="")  # fmt: skip


def curve_for_table(table: Any, time_offset_s: float = 0.0) -> TapCurve:
    """The curve of an official tap table (the valid taps, as Set B uses them)."""
    taps = fb.valid_taps(table)
    return tap_curve(taps["t_peak_s"], taps["amplitude"], time_offset_s, n_all=len(table))


def curve_for_upload(result: Any) -> TapCurve:
    table = tap_table.build_tap_table(result.taps_table)
    return curve_for_table(table, result.time_offset_s)


def slope_sentence(c: TapCurve) -> str:
    if not c.usable:
        return c.reason
    s = c.slope_per10s_frac
    word = "falls" if s < -FLAT_BAND else "rises" if s > FLAT_BAND else "is flat"
    return (
        f"Fitted line (Theil-Sen): amplitude changes by {abs(s) * 100:.1f}% of its mean "
        f"per 10 s ({word})."
    )


def count_sentence(c: TapCurve) -> str:
    text = f"Taps shown: {c.n_taps} of {c.n_found}."
    if c.n_nan:
        text += f" {c.n_nan} without a valid amplitude left out."
    if c.n_edge:
        noun = "tap" if c.n_edge == 1 else "taps"
        text += f" {c.n_edge} {noun} at the clip edges left out (one-sided amplitude)."
    return text


def curve_figure(c: TapCurve, ref: str, compact: bool = False) -> Figure:
    """The amplitude figure; ``compact`` is the page-size variant with larger relative fonts."""
    big, mid, title = (9, 8.5, 10) if compact else (12, 11, 14)
    fig = Figure(figsize=(5.2, 2.3) if compact else (11.5, 4.4), dpi=110, layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.plot(c.times, c.amps, color=LINK_COLOUR, lw=0.8, zorder=1)
    ax.plot(c.times, c.amps, "o", color=POINT_COLOUR, ms=4 if compact else 6, ls="none",
            label="tap amplitude", zorder=2)  # fmt: skip
    if c.usable:
        ax.plot(c.line_x, c.line_y, color=LINE_COLOUR, lw=2.2, label="fitted line (Theil-Sen)",
                zorder=3)  # fmt: skip
    ax.set_xlabel("time (s)", fontsize=big)
    ax.set_ylabel("tap amplitude (hand sizes)", fontsize=big)
    ax.tick_params(labelsize=mid)
    ax.spines[["top", "right"]].set_visible(False)
    where: Literal["outside lower center", "outside upper right"] = (
        "outside lower center" if compact else "outside upper right"
    )  # never over the title
    fig.legend(fontsize=mid, loc=where, frameon=False, ncols=2)  # off the data
    fig.suptitle(f"Per-tap amplitude (clip {ref})", fontsize=title)
    return fig
