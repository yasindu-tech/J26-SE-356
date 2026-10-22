import type { TappingOutcome, TappingRunStatus, TappingScored } from "./tappingTypes";

// What the finger-tapping screens show, as pure functions (no React): fixed texts, formatting,
// the geometry of the three charts. The result's own sentences come from the Python program
// and are shown as they are. The score is a research model output; it is never called a
// probability, never a priority, and the 5-95% band is a spread of models, not an error bar.

/** The fixed sentences. */
export const SCORE_LINE =
  "Research model score on a 0 to 1 scale. Not a calibrated probability. Does not diagnose.";
export const CONTRIBUTION_LINE =
  "Contributions explain the model's score, not the cause of any condition.";
export const NOT_COMBINED =
  "This is a screening aid for prioritising specialist review. It gives no referral priority: the reading comes from one part of the platform, and the step that combines parts is not built yet.";
export const CHIP_AVAILABLE = "Available";
export const CHIP_CHOSEN = "Video chosen";
export const VIDEO_STAYS_HERE =
  "The video is sent to the analysis service and deleted there as soon as the analysis ends. Only the result is kept.";

export const TAPPING_MAX_MB = 200;
export const TAPPING_EXTENSIONS = [".mp4", ".mov", ".m4v"] as const;

export function extensionOf(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot < 0 ? "" : name.slice(dot).toLowerCase();
}

/** What is wrong with the chosen file, or null. The text never repeats the file name. */
export function videoProblem(name: string, bytes: number): { title: string; text: string } | null {
  if (!TAPPING_EXTENSIONS.some((e) => e === extensionOf(name))) {
    return { title: "This file type cannot be read", text: "Choose a video: MP4, MOV or M4V." };
  }
  if (bytes < 1) return { title: "This file is empty", text: "Choose another video." };
  if (bytes > TAPPING_MAX_MB * 1024 * 1024) {
    return {
      title: "This video is too large",
      text: `The limit is ${TAPPING_MAX_MB} MB. Film again at the normal video setting, or trim the clip to the tapping.`,
    };
  }
  return null;
}

/** "12.3 MB · 18 s · clip f0e37b01": no file name, only the hashed ref. */
export function videoSummary(bytes: number, seconds: number | null, ref: string | null): string {
  const parts = [`${(bytes / (1024 * 1024)).toFixed(1)} MB`];
  if (seconds !== null && Number.isFinite(seconds)) parts.push(`${Math.round(seconds)} s`);
  if (ref !== null) parts.push(`clip ${ref}`);
  return parts.join(" · ");
}

// ---------------------------------------------------------------------------- outcomes

const NOT_SCORED = "The video could not be scored.";

/** How an analysis ended: a scored result, or a plain message. A stored "failed" result is the
 * program's own reason (for example "Hand not found in enough frames"). */
export function outcomeOf(run: TappingRunStatus): TappingOutcome {
  const { result } = run;
  if (run.state === "scored" && result !== null && result.status === "scored") {
    return { kind: "scored", result };
  }
  if (result !== null && result.status === "failed") {
    return { kind: "failed", message: result.reason, ref: result.clip_ref };
  }
  return { kind: "failed", message: run.error ?? NOT_SCORED, ref: null };
}

/** The word in the parts strip: scored, failed or still running. */
export function stripState(outcome: TappingOutcome | null): "running" | "scored" | "failed" {
  if (outcome === null) return "running";
  return outcome.kind === "scored" ? "scored" : "failed";
}

// ---------------------------------------------------------------------------- numbers

const MISSING = "not available";

export function fixed(value: number | null, digits: number, unit = ""): string {
  return value === null || !Number.isFinite(value) ? MISSING : `${value.toFixed(digits)}${unit}`;
}

export function scoreHeadline(r: TappingScored): string {
  return `Model score ${r.score.value.toFixed(2)}`;
}

/** The 5-95% band as a range of models. It is not a confidence interval and is never called one. */
export function bandText(r: TappingScored): string {
  return `Range ${r.score.band_low.toFixed(2)} to ${r.score.band_high.toFixed(2)} (${r.score.band_basis}).`;
}

export function thresholdText(r: TappingScored): string {
  const side = r.score.above_threshold ? "At or above" : "Below";
  return `${side} the fixed study threshold of ${r.score.threshold}. The threshold was set for the study; it is not a clinical cut-off.`;
}

/** The cross-validation line, or null when the study figures are missing. */
export function crossValidationText(r: TappingScored): string | null {
  const { cv_auroc: est, cv_ci_low: lo, cv_ci_high: hi } = r.score;
  if (est === null || lo === null || hi === null) return null;
  const which = r.target === "ge2" ? "the primary" : "the secondary";
  return `In cross-validation on the research clips this model reached AUROC ${est.toFixed(2)} (95% interval ${lo.toFixed(2)} to ${hi.toFixed(2)}) for ${which} study target.`;
}

export interface ScaleGeometry {
  valuePercent: number;
  lowPercent: number;
  highPercent: number;
  thresholdPercent: number;
}

const clampPercent = (v: number): number => Math.max(0, Math.min(100, v * 100));

export function scaleGeometry(r: TappingScored): ScaleGeometry {
  return {
    valuePercent: clampPercent(r.score.value),
    lowPercent: clampPercent(r.score.band_low),
    highPercent: clampPercent(r.score.band_high),
    thresholdPercent: clampPercent(r.score.threshold),
  };
}

export function scaleDescription(r: TappingScored): string {
  return `${scoreHeadline(r)} on a scale from 0 to 1. ${bandText(r)} ${thresholdText(r)}`;
}

export function recordingRows(r: TappingScored): { label: string; value: string }[] {
  const rec = r.recording;
  return [
    { label: "Clip", value: r.clip_ref },
    { label: "Length", value: fixed(rec.duration_s, 1, " s") },
    { label: "Frame rate", value: fixed(rec.fps, 0, " fps") },
    { label: "Hand found in", value: fixed(rec.hand_found_pct, 0, "% of frames") },
    { label: "Taps found", value: String(rec.n_taps) },
  ];
}

export function auditLine(r: TappingScored): string {
  return `clip ${r.clip_ref} · demo model ${r.model.demo_model_sha8} · fold models ${r.model.folds_sha8}`;
}

export function previewAlt(t: number | null): string {
  return `Video frame${t === null ? "" : ` at ${t.toFixed(1)} s`} with the thumb and index fingertips marked`;
}

// ---------------------------------------------------------------------------- SHAP bars

export type Direction = "up" | "down" | "none";

export interface ShapBar {
  key: string;
  label: string;
  valueText: string;
  direction: Direction;
  /** Left edge and width on a 0 to 100 axis whose middle (50) is no effect. */
  x: number;
  width: number;
  sentence: string;
}

export function directionOf(shap: number | null): Direction {
  if (shap === null || !Number.isFinite(shap) || shap === 0) return "none";
  return shap > 0 ? "up" : "down";
}

export function directionWord(d: Direction): string {
  return d === "up" ? "Pushed the score up" : d === "down" ? "Pushed the score down" : "No effect";
}

export function shapBars(r: TappingScored): ShapBar[] {
  const largest = Math.max(0, ...r.contributions.map((c) => Math.abs(c.shap ?? 0)));
  return r.contributions.map((c) => {
    const direction = directionOf(c.shap);
    const half = largest === 0 || c.shap === null ? 0 : (Math.abs(c.shap) / largest) * 50;
    return {
      key: c.feature_label,
      label: c.feature_label,
      valueText: c.value_text,
      direction,
      x: direction === "down" ? 50 - half : 50,
      width: direction === "none" ? 0 : half,
      sentence: c.sentence,
    };
  });
}

export function shapDescription(bars: ShapBar[]): string {
  return `How much each measurement moved the score. ${bars
    .map((b) => `${b.label}: ${directionWord(b.direction).toLowerCase()}`)
    .join("; ")}.`;
}

// ---------------------------------------------------------------------------- line charts

export const CHART = { width: 600, height: 230, left: 52, right: 14, top: 14, bottom: 40 } as const;

export interface Tick {
  at: number;
  label: string;
}

export interface LineChart {
  /** SVG path data; a gap in the data starts a new sub-path. */
  path: string;
  /** Marker centres in chart units. */
  markers: { x: number; y: number }[];
  /** The straight fitted line, when there is one. */
  line: { x1: number; y1: number; x2: number; y2: number } | null;
  xTicks: Tick[];
  yTicks: Tick[];
  description: string;
}

function finitePairs(xs: (number | null)[], ys: (number | null)[]): ([number, number] | null)[] {
  return xs.map((x, i) => {
    const y = ys[i] ?? null;
    return x !== null && y !== null && Number.isFinite(x) && Number.isFinite(y) ? [x, y] : null;
  });
}

function span(values: number[], pad: number): [number, number] {
  if (values.length === 0) return [0, 1];
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const gap = hi - lo === 0 ? 1 : (hi - lo) * pad;
  return [lo - gap, hi + gap];
}

function ticks(lo: number, hi: number, count: number, digits: number): number[] {
  return Array.from({ length: count }, (_, i) => Number((lo + ((hi - lo) * i) / (count - 1)).toFixed(digits)));
}

function frame(xs: number[], ys: number[]): {
  sx: (v: number) => number;
  sy: (v: number) => number;
  x: [number, number];
  y: [number, number];
} {
  const x = span(xs, 0);
  const y = span(ys, 0.08);
  const w = CHART.width - CHART.left - CHART.right;
  const h = CHART.height - CHART.top - CHART.bottom;
  return {
    x,
    y,
    sx: (v) => CHART.left + ((v - x[0]) / (x[1] - x[0] || 1)) * w,
    sy: (v) => CHART.top + h - ((v - y[0]) / (y[1] - y[0] || 1)) * h,
  };
}

const r1 = (v: number): number => Math.round(v * 10) / 10;

function axisTicks(f: ReturnType<typeof frame>): { xTicks: Tick[]; yTicks: Tick[] } {
  return {
    xTicks: ticks(f.x[0], f.x[1], 5, 1).map((v) => ({ at: r1(f.sx(v)), label: v.toFixed(0) })),
    yTicks: ticks(f.y[0], f.y[1], 4, 2).map((v) => ({ at: r1(f.sy(v)), label: v.toFixed(2) })),
  };
}

/** The smoothed thumb-index distance with a marker at each tap peak. */
export function signalChart(r: TappingScored): LineChart | null {
  const pairs = finitePairs(r.signal.t, r.signal.smoothed);
  const peaks = finitePairs(r.signal.peak_t, r.signal.peak_v);
  const points = pairs.filter((p): p is [number, number] => p !== null);
  if (points.length < 2) return null;
  const marks = peaks.filter((p): p is [number, number] => p !== null);
  const f = frame([...points.map((p) => p[0]), ...marks.map((p) => p[0])], [...points.map((p) => p[1]), ...marks.map((p) => p[1])]);
  let path = "";
  let open = false;
  for (const p of pairs) {
    if (p === null) {
      open = false;
      continue;
    }
    path += `${open ? "L" : "M"}${r1(f.sx(p[0]))} ${r1(f.sy(p[1]))}`;
    open = true;
  }
  return {
    path,
    markers: marks.map((p) => ({ x: r1(f.sx(p[0])), y: r1(f.sy(p[1])) })),
    line: null,
    ...axisTicks(f),
    description: `Thumb-index distance over time, in hand sizes, with ${marks.length} tap peaks marked over ${fixed(r.recording.duration_s, 1, " s")}.`,
  };
}

/** The amplitude of each tap, with the fitted line only when the program made one. */
export function curveChart(r: TappingScored): LineChart | null {
  const pairs = finitePairs(r.curve.times, r.curve.amps);
  const points = pairs.filter((p): p is [number, number] => p !== null);
  if (points.length < 1) return null;
  const lines = r.curve.usable ? finitePairs(r.curve.line_x, r.curve.line_y).filter((p): p is [number, number] => p !== null) : [];
  const f = frame([...points.map((p) => p[0]), ...lines.map((p) => p[0])], [...points.map((p) => p[1]), ...lines.map((p) => p[1])]);
  const first = lines[0];
  const last = lines[lines.length - 1];
  return {
    path: "",
    markers: points.map((p) => ({ x: r1(f.sx(p[0])), y: r1(f.sy(p[1])) })),
    line:
      first && last && lines.length >= 2
        ? { x1: r1(f.sx(first[0])), y1: r1(f.sy(first[1])), x2: r1(f.sx(last[0])), y2: r1(f.sy(last[1])) }
        : null,
    ...axisTicks(f),
    description: `Amplitude of each tap over time, in hand sizes, ${points.length} taps. ${r.curve.slope_sentence}`,
  };
}
