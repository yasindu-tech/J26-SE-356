import type { GaitCheckName, GaitResult } from "./gaitTypes";

// The plain-language reading of a gait result, for a general clinician. The model grades
// how a person walks (no, mild, or moderate-to-severe gait impairment); it does not say
// whether they have Parkinson's disease. No percentage and no High/Moderate/Low priority:
// the scores are not calibrated and the fusion step that sets a priority is not built.
// A lower reading is never presented as a clearance.

const CLASS_WORDS = [
  "no gait impairment",
  "mild gait impairment",
  "moderate-to-severe gait impairment",
] as const;

export const CLASS_SHORT = ["No impairment", "Mild", "Moderate to severe"] as const;
const CLASS_RATING = ["UPDRS gait rating 0", "UPDRS gait rating 1", "UPDRS gait rating 2 to 3"] as const;

/** When the top two scores are closer than this, the model does not place the walk clearly. */
export const MIN_MARGIN = 0.15;

/** Below this many seconds of walking the reading rests on few steps. */
export const SHORT_WALK_SECONDS = 5;

export function isScored(r: GaitResult): boolean {
  return r.available && r.predicted_class !== null && r.probabilities !== null;
}

export interface GaitReading {
  /** "clear" when the model puts one level ahead; "none" when the scores are too close. */
  kind: "clear" | "none";
  headline: string;
  detail: string;
}

function topTwo(probabilities: readonly number[]): [number, number] {
  const sorted = [...probabilities].sort((a, b) => b - a);
  return [sorted[0] ?? 0, sorted[1] ?? 0];
}

export function plainGaitReading(r: GaitResult): GaitReading {
  const cls = r.predicted_class ?? 0;
  const [first, second] = topTwo(r.probabilities ?? []);
  if (first - second < MIN_MARGIN) {
    return {
      kind: "none",
      headline: "No clear pattern",
      detail:
        "The model’s scores for the three levels are close together, so it cannot place this walk clearly. It does not point towards a gait impairment, and it does not point away from one.",
    };
  }
  const words = CLASS_WORDS[cls] ?? CLASS_WORDS[0];
  const closing =
    cls === 0
      ? "This does not rule anything out; a lower reading is never a clearance."
      : "Weigh it alongside the rest of your assessment. It is not a diagnosis.";
  return {
    kind: "clear",
    headline: `Walk pattern closest to: ${words}`,
    detail: `The model places this walk closest to the ${words} group. ${closing}`,
  };
}

function seconds(n: number): string {
  return n < 10 ? n.toFixed(1) : String(Math.round(n));
}

/** How much of the video was scored. */
export function walkLine(r: GaitResult): string {
  return `Scored ${seconds(r.walk.seconds)} s of walking, from ${r.walk.start_s.toFixed(1)} s to ${r.walk.end_s.toFixed(1)} s of the video.`;
}

/** How far to lean on the model, from its own test results (models/gait/README.md). */
export const RELIABILITY_LINE =
  "On test walks recorded with motion capture, the model put people in the right level about half the time, against about a third by guessing. It has not been tested on phone video. Treat the reading as weak evidence.";

export interface ClassBar {
  key: string;
  label: string;
  /** The UPDRS gait rating this level stands for, shown under the label. */
  rating: string;
  /** Bar length as a share of the full bar, 1 to 100. Drawn to scale, never printed. */
  widthPercent: number;
  isPick: boolean;
}

/** The three model scores as bars to scale. A bar's length is not printed as a number. */
export function classBars(r: GaitResult): ClassBar[] {
  const probs = r.probabilities ?? [];
  return CLASS_SHORT.map((label, i) => ({
    key: `class-${i}`,
    label,
    rating: CLASS_RATING[i] ?? "",
    widthPercent: Math.min(100, Math.max(1, Math.round((probs[i] ?? 0) * 100))),
    isPick: i === r.predicted_class,
  }));
}

export interface DriverRow {
  key: string;
  text: string;
  /** "Strongly towards", "Moderately away", and so on, relative to the level the model picked. */
  tag: string;
  towards: boolean;
}

/** The inputs that moved the score most, as the model's own plain-words sentences. */
export function driverRows(r: GaitResult, count = 3): DriverRow[] {
  return r.top_features.slice(0, count).map((f) => ({
    key: f.name,
    text: f.sentence,
    tag: `${f.strength.charAt(0).toUpperCase()}${f.strength.slice(1)} ${f.direction}`,
    towards: f.direction === "towards",
  }));
}

const INPUT_SHORT: Record<string, string> = {
  stride_time_mean: "Stride time",
  cadence: "Walking pace",
  stride_time_cv: "Stride-time variability",
  step_length_norm: "Step length",
  step_length_asymmetry: "Step length asymmetry",
  knee_flexion_range_deg: "Knee bend range",
  hip_flexion_range_deg: "Hip swing range",
  arm_swing_range_norm: "Arm swing range",
};

export interface ShapBar {
  key: string;
  label: string;
  /** Bar length as a share of the longest bar, 3 to 100. Drawn to scale, never printed. */
  widthPercent: number;
  /** Towards the model's highest-scoring level (drawn right, solid) or away (left, outlined). */
  towards: boolean;
  tag: string;
}

/** The inputs that moved the score most, as a diverging bar chart. Lengths are to scale. */
export function shapBars(r: GaitResult, count = 3): ShapBar[] {
  const features = r.top_features.slice(0, count);
  const longest = Math.max(...features.map((f) => Math.abs(f.shap)), 0);
  return features.map((f) => ({
    key: f.name,
    label: INPUT_SHORT[f.name] ?? f.name,
    widthPercent:
      longest === 0 ? 3 : Math.min(100, Math.max(3, Math.round((Math.abs(f.shap) / longest) * 100))),
    towards: f.direction === "towards",
    tag: `${f.strength.charAt(0).toUpperCase()}${f.strength.slice(1)} ${f.direction}`,
  }));
}

export interface CheckRow {
  key: GaitCheckName;
  label: string;
  value: string;
  need: string;
  ok: boolean;
}

const pct = (x: number): string => `${Math.round(x * 100)}%`;

/** The video checks, in a fixed order, each with what was found and what is needed. */
export function checkRows(r: GaitResult): CheckRow[] {
  const c = r.gate.checks;
  return [
    {
      key: "frame_rate",
      label: "Frame rate",
      value: `${Math.round(c.frame_rate.value)} frames per second`,
      need: "needs 20 to 60",
      ok: c.frame_rate.ok,
    },
    {
      key: "leg_visibility",
      label: "Legs visible",
      value: `${pct(c.leg_visibility.value)} of frames`,
      need: `needs at least ${pct(c.leg_visibility.limit)}`,
      ok: c.leg_visibility.ok,
    },
    {
      key: "filled_share",
      label: "Frames filled in",
      value: pct(c.filled_share.value),
      need: `needs at most ${pct(c.filled_share.limit)}`,
      ok: c.filled_share.ok,
    },
    {
      key: "facing",
      label: "Toes ahead of heels",
      value: `${pct(c.facing.value)} of frames`,
      need: `needs at least ${pct(c.facing.limit)}`,
      ok: c.facing.ok,
    },
    {
      key: "strides",
      label: "Strides found",
      value: `${Math.round(c.strides.value)} in total, on both feet`,
      need: `needs at least ${Math.round(c.strides.limit)}`,
      ok: c.strides.ok,
    },
  ];
}

const ONE_PERSON_WARNINGS = ["You confirmed one person", "One person in view is assumed"];

/** Warnings worth their own notice. The one-person warnings are covered under Keep in mind. */
export function visibleWarnings(r: GaitResult): string[] {
  return r.warnings.filter((w) => !ONE_PERSON_WARNINGS.some((p) => w.startsWith(p)));
}

/** What to keep in mind for this result. */
export function keepInMind(r: GaitResult): string[] {
  const items = [
    "The model grades how a person walks. It does not say whether they have Parkinson’s disease, and other causes of a slow or uneven walk, such as joint pain, can look similar.",
    "It learned from motion-capture recordings. Joint positions from a phone video are less precise, and the model has not been tested as a Parkinson’s check on video.",
    r.one_person_confirmed
      ? "You confirmed one person is walking. The video itself is not checked for other people, so if someone else was in view, do not rely on this reading."
      : "One person walking is assumed. The video itself is not checked for other people.",
  ];
  if (!isScored(r)) return items;
  items.push(
    "The bars are the model’s scores, not chances: they have not been calibrated, so a long bar does not mean a high probability.",
  );
  return items;
}

export interface Notice {
  title: string;
  text: string;
}

/** Things worth knowing about this video: the pipeline's warnings, and a short walk. */
export function notices(r: GaitResult): Notice[] {
  const items: Notice[] = visibleWarnings(r).map((text) => ({ title: "Check the video", text }));
  if (isScored(r) && r.walk.seconds < SHORT_WALK_SECONDS) {
    items.push({
      title: "Short walk",
      text: `Only ${seconds(r.walk.seconds)} s of walking was scored, so the reading rests on few steps.`,
    });
  }
  return items;
}

/** One line for traceability: which model, which video, who confirmed one person. */
export function auditLine(r: GaitResult): string {
  const who = r.one_person_confirmed ? "one person confirmed by you" : "one person assumed";
  return `Model ${r.model_version} \u00b7 video ${r.video} \u00b7 ${who}`;
}

export interface TimelineBar {
  leftPercent: number;
  widthPercent: number;
  /** "0 s" and the video length, for the two ends. */
  endLabel: string;
  text: string;
}

/** Which part of the video was scored, as a bar. Null when the video length is not known. */
export function timelineBar(r: GaitResult, videoSeconds: number | null): TimelineBar | null {
  if (videoSeconds === null || videoSeconds <= 0 || r.walk.seconds <= 0) return null;
  const total = Math.max(videoSeconds, r.walk.end_s);
  const clamp = (n: number): number => Math.min(100, Math.max(0, n));
  const left = clamp((r.walk.start_s / total) * 100);
  const width = Math.max(2, clamp((r.walk.seconds / total) * 100));
  const short = r.walk.seconds < SHORT_WALK_SECONDS ? " That is short." : "";
  return {
    leftPercent: Math.round(left * 10) / 10,
    widthPercent: Math.round(width * 10) / 10,
    endLabel: `${Math.round(total)} s`,
    text: `Scored ${seconds(r.walk.seconds)} s of walking, from ${r.walk.start_s.toFixed(1)} s to ${r.walk.end_s.toFixed(1)} s.${short}`,
  };
}

// ----------------------------------------------------------------- the picker

export const GAIT_VIDEO_EXTENSIONS = [".mp4", ".mov", ".m4v", ".avi"] as const;
/** The backend's limit (services/backend settings: gait max upload). */
export const GAIT_MAX_UPLOAD_MB = 500;
/** The model reads at most this many seconds of a video. */
export const GAIT_READ_SECONDS = 30;

/** What is wrong with the chosen file, or null when it can be sent. */
export function videoProblem(name: string, bytes: number): { title: string; text: string } | null {
  const lower = name.toLowerCase();
  if (!GAIT_VIDEO_EXTENSIONS.some((ext) => lower.endsWith(ext))) {
    return { title: "This file type cannot be read", text: "Choose a video: MP4, MOV, M4V or AVI." };
  }
  if (bytes > GAIT_MAX_UPLOAD_MB * 1024 * 1024) {
    const size = bytes >= 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(1)} GB` : `${Math.round(bytes / 1024 ** 2)} MB`;
    return {
      title: "This video is too large",
      text: `${name} is ${size}. The limit is ${GAIT_MAX_UPLOAD_MB} MB. Film again at the normal video setting, or trim the clip to the walk.`,
    };
  }
  return null;
}

/** "86.4 MB · 21 s · MP4" for the chosen file. */
export function videoSummary(name: string, bytes: number, seconds_: number | null): string {
  const ext = name.includes(".") ? (name.split(".").pop() ?? "").toUpperCase() : "";
  const size = `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return [size, seconds_ === null ? null : `${Math.round(seconds_)} s`, ext || null].filter(Boolean).join(" \u00b7 ");
}

/** The word for a part's state in the parts strip. */
export function gaitStateText(state: "running" | "scored" | "abstained" | "failed" | "queued"): string {
  switch (state) {
    case "scored":
      return "Scored";
    case "abstained":
      return "Not scored";
    case "failed":
      return "Failed";
    default:
      return "Processing";
  }
}
