"""One-page PDF summary for the demo (task FT-51), no streamlit import, no new dependency:
a matplotlib Figure of A4 size saved as PDF into memory. Texts are fixed templates (texts.py,
tap_curve.py, scoring.py); figures are the PNGs already shown on screen. Nothing is written to
disk, and the metadata is fixed (no author, no date, no path) so equal inputs give equal bytes.
"""

from __future__ import annotations

import functools
import hashlib
import io
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lightgbm  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402
import numpy as np  # noqa: E402
import shap  # noqa: E402
import tap_curve as tc  # noqa: E402
import texts as tx  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.backends.backend_pdf import FigureCanvasPdf  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402

TOOL = "demo-summary-1"
TITLE = "Finger-tapping research prototype - one-page summary"
METADATA: dict[str, Any] = {
    "Title": TITLE, "Creator": "Finger-tapping research prototype (demo)",
    "Producer": "matplotlib pdf backend", "CreationDate": None, "ModDate": None,
}  # fmt: skip
NA = "not available"
W, H = 595.0, 842.0  # A4 in points
MARGIN, BODY, HEAD, SMALL, LEAD = 36.0, 9.5, 11.5, 8.0, 1.28
TEXT_W = W - 2 * MARGIN
MIN_SCALE = 0.55  # the figures may shrink to this share of their nominal size to fit the page
GAP = 7.0
FPS_HIGH = 45.0
LOW_TRACKED = 0.9  # the research gating flags fewer than this share of frames with a hand
INK, MUTED = "#1a1a1a", "#444444"


@dataclass
class SummaryData:
    ref: str  # hashed clip ref
    mode: str  # "upload" or "example"
    target: str
    date_text: str
    recording: dict[str, Any] = field(default_factory=dict)  # duration_s, fps, tracked_share, ...
    qc_flags: list[tuple[str, str]] | None = None  # None: not computed for this clip
    notes: list[str] = field(default_factory=list)  # data-quality notes beyond the flags
    score: float | None = None
    band_text: str | None = None
    threshold_text: str | None = None
    auroc_text: str | None = None
    source_text: str | None = None
    sentences: list[str] = field(default_factory=list)
    shap_png: bytes | None = None
    signal_png: bytes | None = None
    curve_png: bytes | None = None
    curve_count: str | None = None
    curve_slope: str | None = None
    model_sha8: str | None = None
    input_sha8: list[str] = field(default_factory=list)

    def sha8(self) -> str:
        h = hashlib.sha256()
        for key, value in sorted(vars(self).items()):
            h.update(key.encode() + repr(value).encode())
        return h.hexdigest()[:8]


def quality_notes(
    *, mode: str, target: str, fps: float | None, tracked_share: float | None,
    qc_flags: list[tuple[str, str]] | None, outside: list[str], taps_note: str | None,
    curve_reason: str | None,
) -> list[str]:  # fmt: skip
    """The data-quality notes of the PDF, each a fixed template (or the flag's plain text)."""
    notes: list[str] = []
    if qc_flags is None:
        notes.append("Automatic checks are not available for example clips.")
    elif not qc_flags:
        notes.append("Automatic checks: no flags.")
    else:
        notes += [f"Automatic check {name}: {why}." for name, why in qc_flags]
    if fps is not None and np.isfinite(fps) and fps > FPS_HIGH:
        notes.append(tx.FPS_NOTE.format(fps=fps))
    if tracked_share is not None and np.isfinite(tracked_share) and tracked_share < LOW_TRACKED:
        notes.append("The hand was found in fewer than 90% of frames.")
    if taps_note:
        notes.append(taps_note)
    if outside:
        verb = "is" if len(outside) == 1 else "are"
        notes.append(tx.RANGE_NOTE.format(names=", ".join(outside) + f" {verb}"))
    notes.append(
        "Scored with the demo model (trained on all research clips)."
        if mode == "upload" else tx.HELD_OUT
    )  # fmt: skip
    if target == "ge1":
        notes.append(tx.SECONDARY_NOTE)
    if curve_reason:
        notes.append(curve_reason)
    return notes


def _number(value: Any, fmt: str) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return NA
    return fmt.format(v) if np.isfinite(v) else NA


def recording_lines(rec: dict[str, Any]) -> list[str]:
    """The recording summary: every missing or NaN number reads 'not available'."""
    share = rec.get("tracked_share")
    shown = rec.get("n_shown")
    found = rec.get("n_found")
    return [
        f"Duration: {_number(rec.get('duration_s'), '{:.1f} s')}   Frame rate: "
        f"{_number(rec.get('fps'), '{:.0f} fps')}   Hand found in: "
        f"{_number(None if share is None else float(share) * 100, '{:.0f}% of frames')}",
        f"Taps found: {_number(found, '{:.0f}')}   Taps shown in the amplitude curve: "
        f"{_number(shown, '{:.0f}')}",
    ]


# --------------------------------------------------------------------------- layout


@functools.lru_cache(maxsize=16)
def _decode(png: bytes) -> Any:
    return mpimg.imread(io.BytesIO(png), format="png")


_MEASURE = FigureCanvasAgg(Figure(dpi=72)).get_renderer()  # text widths in points


@functools.lru_cache(maxsize=8192)
def _width(s: str, size: float, weight: str) -> float:
    prop = FontProperties(size=size, weight=weight)
    return float(_MEASURE.get_text_width_height_descent(s, prop, False)[0])


@functools.lru_cache(maxsize=4096)
def _wrap(s: str, size: float, width: float, weight: str) -> tuple[str, ...]:
    lines: list[str] = []
    cur = ""
    for word in s.split():
        trial = f"{cur} {word}".strip()
        if cur and _width(trial, size, weight) > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return tuple(lines + ([cur] if cur else []))


class Page:
    """A4 page in points: texts and images placed from the top with a running cursor. A dry page
    only measures (the cursor moves, nothing is drawn): used to find the layout that fits."""

    def __init__(self, dry: bool = False) -> None:
        self.dry = dry
        self.fig = Figure(figsize=(W / 72, H / 72), dpi=72)
        FigureCanvasPdf(self.fig)
        self.y = MARGIN
        self.texts: list[dict[str, Any]] = []

    def width(self, s: str, size: float, weight: str = "normal") -> float:
        return _width(s, size, weight)

    def wrap(self, s: str, size: float, width: float, weight: str = "normal") -> list[str]:
        return list(_wrap(s, size, width, weight))

    def text(self, s: str, size: float = BODY, *, weight: str = "normal", x: float = MARGIN,
             color: str = INK, width: float = TEXT_W, kind: str = "body") -> None:  # fmt: skip
        for line in self.wrap(s, size, width - (x - MARGIN), weight):
            if not self.dry:
                self.fig.text(x / W, 1 - self.y / H, line, fontsize=size, fontweight=weight,
                              color=color, va="top", ha="left")  # fmt: skip
            self.texts.append({"text": line, "size": size, "kind": kind, "y": self.y, "x": x})
            self.y += size * LEAD

    def space(self, pts: float) -> None:
        self.y += pts

    def heading(self, s: str) -> None:
        self.space(GAP)
        self.text(s, HEAD, weight="bold", kind="heading")
        self.space(2)

    def image(self, png: bytes, x: float, width: float) -> float:
        img = _decode(png)
        height = width * img.shape[0] / img.shape[1]
        if self.dry:
            return float(height)
        ax = self.fig.add_axes((x / W, 1 - (self.y + height) / H, width / W, height / H))
        ax.imshow(img, interpolation="antialiased", aspect="auto")
        ax.axis("off")
        return height

    def lines_height(self, s: str, size: float, width: float = TEXT_W, x: float = MARGIN) -> float:
        return len(self.wrap(s, size, width - (x - MARGIN))) * size * LEAD


NOMINAL = {"shap": 440.0, "small": 262.0}  # nominal image widths in points


MORE = "Further notes are not shown on this page."


def build_page(d: SummaryData) -> Page:
    """Place every section on one page: the figures shrink (down to MIN_SCALE) first, then the
    notes are cut (with a fixed sentence saying so) until the page is not exceeded."""
    notes = list(d.notes)
    plans = [(notes, float(s)) for s in np.arange(1.0, MIN_SCALE - 1e-9, -0.05)]
    plans += [(notes[:keep] + [MORE], MIN_SCALE) for keep in range(len(notes) - 1, -1, -1)]
    for shown, scale in plans:
        if _place(d, shown, scale, dry=True).y <= H - MARGIN:
            return _place(d, shown, scale)
    raise ValueError("the summary does not fit on one page")


def _place(d: SummaryData, notes: list[str], scale: float, dry: bool = False) -> Page:
    p = Page(dry)
    p.text(TITLE, 15, weight="bold", kind="title")
    p.space(3)
    p.text(tx.DISCLAIMER, BODY, kind="disclaimer")
    p.text(f"Clip {d.ref}   Date {d.date_text}   Target {d.target}", BODY, color=MUTED)

    p.heading("Recording")
    for line in recording_lines(d.recording):
        p.text(line)

    p.heading("Data-quality notes")
    for note in notes or ["No notes."]:
        p.text(f"- {note}", x=MARGIN + 6)

    p.heading("Score")
    if d.score is None:
        p.text(f"Model score: {NA}", 16, weight="bold", kind="score")
    else:
        p.text(f"Model score {d.score:.2f}", 20, weight="bold", kind="score")
    for score_line in (d.band_text, d.threshold_text, d.auroc_text, d.source_text):
        if score_line:
            p.text(score_line)

    p.heading("Why this score (SHAP)")
    if d.shap_png is not None:
        width = NOMINAL["shap"] * scale
        h = p.image(d.shap_png, (W - width) / 2, width)
        p.y += h + 3
    else:
        p.text(f"Contribution figure: {NA}")
    for s in d.sentences:
        p.text(f"- {s}", x=MARGIN + 6)
    p.text(tx.SHAP_CAPTION, BODY, color=MUTED, kind="caption")

    p.heading("Signal and per-tap amplitude")
    width = NOMINAL["small"] * scale
    height = 0.0
    for png, x, missing in (
        (d.signal_png, MARGIN, f"Signal figure: {NA} for this clip."),
        (d.curve_png, W - MARGIN - width, f"Per-tap amplitude figure: {NA} for this clip."),
    ):
        if png is not None:
            height = max(height, p.image(png, x, width))
        else:
            lines = p.wrap(missing, BODY, width)
            for k, line in enumerate(lines):
                if not p.dry:
                    p.fig.text(x / W, 1 - (p.y + k * BODY * LEAD) / H, line, fontsize=BODY,
                               color=MUTED, va="top", ha="left")  # fmt: skip
                p.texts.append({"text": line, "size": BODY, "kind": "body", "y": p.y, "x": x})
            height = max(height, len(lines) * BODY * LEAD)
    p.y += height + 3
    for curve_line in (d.curve_count, d.curve_slope):
        if curve_line:
            p.text(curve_line)
    p.text(tc.NOTE, BODY, color=MUTED, kind="caption")

    p.space(GAP)
    foot = [
        f"Model: {d.model_sha8 or NA} (demo model, not used for metrics).   "
        f"Inputs: {', '.join(d.input_sha8) if d.input_sha8 else NA}",
        f"Tools: {TOOL}; numpy {np.__version__}, lightgbm {lightgbm.__version__}, "
        f"shap {shap.__version__}, matplotlib {matplotlib.__version__}",
    ]
    for foot_line in foot:
        p.text(foot_line, SMALL, color=MUTED, kind="footer")
    return p


def build_pdf(d: SummaryData) -> bytes:
    """The one-page PDF as bytes (in memory only)."""
    page = build_page(d)
    buf = io.BytesIO()
    page.fig.savefig(buf, format="pdf", metadata=dict(METADATA))
    return buf.getvalue()
