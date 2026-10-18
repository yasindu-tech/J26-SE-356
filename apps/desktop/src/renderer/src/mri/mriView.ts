import {
  MRI_STEPS,
  type AnalysisStatus,
  type EddyProfile,
  type FinalStatus,
  type MriAnalysisStatus,
  type MriDriver,
  type MriResult,
  type MriScore,
} from "./mriTypes";

// Pure helpers for the MRI screen, kept out of the components so they can be tested.

export const STEP_LABELS: Record<string, string> = {
  convert: "Convert DICOM",
  t1: "T1 pipeline",
  dti: "DTI pipeline",
  features: "Measure features",
  score: "Harmonise and score",
  explain: "Explain",
};

export function isFinal(status: AnalysisStatus): status is FinalStatus {
  return status === "scored" || status === "abstained" || status === "failed";
}

export type StepState = "done" | "current" | "waiting" | "stopped";

/** How one step is drawn in the progress list. */
export function stepState(
  step: string,
  status: AnalysisStatus,
  done: readonly string[],
  current: string | null,
): StepState {
  if (done.includes(step)) return "done";
  if (step === current && !isFinal(status)) return "current";
  if (isFinal(status)) return "stopped";
  return "waiting";
}

export function progressFraction(done: readonly string[]): number {
  return Math.min(1, done.length / MRI_STEPS.length);
}

// Tones stay out of the screening palette (attention/calm): whether a scan was
// scored says nothing about the person's priority.
export type StatusTone = "neutral" | "working" | "done" | "error";

/** Screening wording only: never "positive", "negative" or "diagnosis". */
export function statusLabel(status: AnalysisStatus): { text: string; tone: StatusTone } {
  switch (status) {
    case "queued":
      return { text: "Waiting to start", tone: "working" };
    case "running":
      return { text: "Processing", tone: "working" };
    case "scored":
      return { text: "Scored", tone: "done" };
    case "abstained":
      return { text: "Not scored", tone: "neutral" };
    case "failed":
      return { text: "Analysis failed", tone: "error" };
    default: {
      const exhaustive: never = status;
      return exhaustive;
    }
  }
}

/** Before an analysis runs (Settings, confirmation): the same warning, in the future tense. */
export const FAST_EDDY_WARNING =
  "New analyses will use the fast DTI setting. It has not been checked against the standard (full) setting, so the DTI measurements, and the score, may differ from what the full setting would give.";

export const EDDY_PROFILE_LABELS: Record<EddyProfile, string> = {
  full: "Full (standard)",
  fast: "Fast (quicker, not validated)",
};

/** Shown whenever the fast DTI setting was used, so a score is never read without it. */
export const FAST_EDDY_NOTICE =
  "DTI motion correction ran with the fast setting. It has not been checked against the " +
  "standard (full) setting, so the DTI measurements, and this score, may differ from what " +
  "the full setting would give.";

/** Which eddy setting produced a result's DTI features; null when there was no DTI. */
export function eddyProfileOf(result: Pick<MriResult, "inputs">): EddyProfile | null {
  return result.inputs["dti"]?.eddy?.profile ?? null;
}

export interface ScoreReading {
  lean: "towards PD" | "towards control" | "on the boundary";
  intervalCrossesBoundary: boolean;
  sentence: string;
}

/** Plain-language reading of the raw log-odds score and its 95% interval. */
export function readScore(score: MriScore): ScoreReading {
  const [low, high] = score.interval_95;
  const crosses = low <= 0 && high >= 0;
  const lean =
    score.value > 0 ? "towards PD" : score.value < 0 ? "towards control" : "on the boundary";
  const sentence = crosses
    ? "The 95% interval crosses 0, so the model cannot place this scan clearly on either side of its boundary."
    : low > 0
      ? "The whole 95% interval is above 0: the copies of the model agree the score is on the higher-likelihood side of its boundary. That is a signal for review, not a diagnosis."
      : "The whole 95% interval is below 0: the copies of the model agree the score is on the lower-likelihood side of its boundary. A lower score is never a clearance.";
  return { lean, intervalCrossesBoundary: crosses, sentence };
}

/** Position (0–100%) of a log-odds value on the score axis drawn in the UI. */
export const SCORE_AXIS: readonly [number, number] = [-4, 4];

export function axisPercent(value: number, axis: readonly [number, number] = SCORE_AXIS): number {
  const [min, max] = axis;
  const clamped = Math.max(min, Math.min(max, value));
  return ((clamped - min) / (max - min)) * 100;
}

export function formatSigned(value: number, digits = 2): string {
  return `${value > 0 ? "+" : value < 0 ? "−" : ""}${Math.abs(value).toFixed(digits)}`;
}

/** The feature vector as JSON, for handing to the fusion work. */
export function vectorJson(names: readonly string[], values: readonly number[]): string {
  const entries = names.map((name, i) => [name, values[i]] as const);
  return JSON.stringify(Object.fromEntries(entries), null, 2);
}

// ------------------------------------------------------------ progress screen

/** The backend sends UTC times without a zone ("2026-10-18T12:00:00"). */
export function parseBackendTime(value: string): Date {
  return new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(value) ? value : `${value}Z`);
}

/** Elapsed time as 02:14, or 1:02:14 once it passes an hour. Never negative. */
export function formatElapsed(totalSeconds: number): string {
  const s = Math.max(0, Math.floor(totalSeconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const pad = (n: number): string => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(s % 60)}` : `${pad(m)}:${pad(s % 60)}`;
}

export function formatClock(at: Date): string {
  const pad = (n: number): string => String(n).padStart(2, "0");
  return `${pad(at.getHours())}:${pad(at.getMinutes())}:${pad(at.getSeconds())}`;
}

export interface ActivityEntry {
  key: string; // what happened, so the same event is never listed twice
  at: string; // ISO time this screen saw it
  text: string;
}

/**
 * The activity list: only what this screen has actually seen the backend report (a step
 * finishing, a new progress note). It starts when the screen opens and is not a stored log.
 */
export function updateActivity(
  log: readonly ActivityEntry[],
  status: Pick<MriAnalysisStatus, "created_at" | "done" | "current_step" | "note">,
  now: Date,
): ActivityEntry[] {
  const next = [...log];
  const seen = new Set(log.map((e) => e.key));
  const add = (key: string, text: string, at: Date): void => {
    if (!seen.has(key)) {
      seen.add(key);
      next.push({ key, at: at.toISOString(), text });
    }
  };
  add("requested", "Analysis requested", parseBackendTime(status.created_at));
  for (const step of status.done) {
    add(`done:${step}`, `${STEP_LABELS[step] ?? step}: finished`, now);
  }
  if (status.current_step && status.note) {
    add(`note:${status.current_step}:${status.note}`, status.note, now);
  }
  return next;
}

// --------------------------------------------------------------------- result page

export interface ScoreBand {
  zero: number; // all values are 0–100, positions on the axis
  point: number;
  low: number;
  high: number;
  width: number;
}

/** Where the zero line, the score and its 95% interval sit on the axis. */
export function scoreBand(score: MriScore, axis: readonly [number, number] = SCORE_AXIS): ScoreBand {
  const low = axisPercent(score.interval_95[0], axis);
  const high = axisPercent(score.interval_95[1], axis);
  return {
    zero: axisPercent(0, axis),
    point: axisPercent(score.value, axis),
    low,
    high,
    width: Math.max(0, high - low),
  };
}

export interface DriverBar {
  feature: string;
  label: string;
  contribution: number;
  /** Which side of the zero line the bar is on, and how it is drawn: solid on the right
   *  (pushed the score higher), outlined on the left (pushed it lower). Direction never
   *  depends on colour alone. */
  side: "higher" | "lower";
  widthPct: number; // 0–100 of the bar lane, relative to the largest contribution shown
}

export function driverBars(drivers: readonly MriDriver[]): DriverBar[] {
  const largest = Math.max(1e-9, ...drivers.map((d) => Math.abs(d.contribution)));
  return drivers.map((d) => ({
    feature: d.feature,
    label: d.label,
    contribution: d.contribution,
    side: d.contribution >= 0 ? "higher" : "lower",
    widthPct: (Math.abs(d.contribution) / largest) * 100,
  }));
}

/** Attribution wording: what contributed to the prediction, never what caused anything. */
export function driverSentence(d: Pick<MriDriver, "label" | "contribution">): string {
  const dir = d.contribution >= 0 ? "higher" : "lower";
  return `${d.label} contributed ${formatSigned(d.contribution)} to the score (pushed it ${dir}).`;
}

export const MODEL_SET_LABELS: Record<string, string> = {
  t1: "T1 only",
  dti: "DTI only",
  t1dti: "T1 + DTI",
};

export function modelSetLabel(set: string | null): string {
  return set ? (MODEL_SET_LABELS[set] ?? set.toUpperCase()) : "no";
}

export interface MissingRow {
  key: "t1" | "dti";
  what: string;
  detail: string;
  effect: string;
  /** Label of the action offered, or null when there is nothing to do. */
  action: string | null;
}

/** Each scan part that did not contribute, what that does to the result, what to do. */
export function missingRows(result: Pick<MriResult, "availability" | "inputs" | "status" | "model_set">): MissingRow[] {
  const rows: MissingRow[] = [];
  for (const key of ["t1", "dti"] as const) {
    if (result.availability[key]) continue;
    const input = result.inputs[key];
    const found = input?.found ?? false;
    rows.push({
      key,
      what: key === "t1" ? "T1 scan" : "DTI scan",
      detail: !found
        ? "Not in the upload"
        : input?.status === "ok"
          ? "Found, but not used by the model that scored this scan"
          : "Found, but it did not pass processing or its quality checks",
      effect:
        result.status === "scored"
          ? `The score comes from the ${modelSetLabel(result.model_set)} model only. Scores from different models are not interchangeable, so compare a score only with scores from the same model.`
          : "No MRI score was produced, and nothing is counted as zero or as normal.",
      action: found ? "Check the scan and upload again" : "Upload a study that includes it",
    });
  }
  return rows;
}

export interface Caveats {
  population: string | null;
  limits: string[];
  outsideRange: { name: string; label: string }[];
}

/** The caveats block: who the model was trained on, its known limits, and anything in this
 *  scan that falls outside what it saw in training. */
export function caveats(
  result: Pick<MriResult, "model_card" | "feature_vector">,
): Caveats {
  const people = result.model_card?.training_people;
  const fv = result.feature_vector;
  return {
    population: people
      ? `Trained on ${people.total} people (${people.pd} with Parkinson's disease, ${people.control} controls). With a sample this small the model's uncertainty is large.`
      : null,
    limits: result.model_card?.known_limits ?? [],
    outsideRange: (fv?.outside_training_range ?? []).map((name) => {
      const i = fv ? fv.names.indexOf(name) : -1;
      return { name, label: fv && i >= 0 ? (fv.labels[i] ?? name) : name };
    }),
  };
}

// ------------------------------------------------------------------------- home

/** "Good morning" / "Good afternoon" / "Good evening" from the local hour. */
export function greeting(now: Date): string {
  const h = now.getHours();
  if (h < 12) return "Good morning";
  if (h < 18) return "Good afternoon";
  return "Good evening";
}

export interface HomeSummary {
  /** Analyses counted (the most recent ones the page loaded). */
  counted: number;
  inProgress: MriAnalysisStatus[];
  scored: number;
  notScored: number;
  failed: number;
}

/** Counts for the home page, from real analyses only. Queued and running are "in progress". */
export function homeSummary(items: readonly MriAnalysisStatus[]): HomeSummary {
  return {
    counted: items.length,
    inProgress: items.filter((i) => i.status === "queued" || i.status === "running"),
    scored: items.filter((i) => i.status === "scored").length,
    notScored: items.filter((i) => i.status === "abstained").length,
    failed: items.filter((i) => i.status === "failed").length,
  };
}

// -------------------------------------------------------------------- assessment

/** A pseudonymous label: letters, digits, "-" and "_", up to 32. Mirrors the backend. */
export function isValidSubjectLabel(value: string): boolean {
  return /^[A-Za-z0-9_-]{1,32}$/.test(value);
}

export type PartKey = "gait" | "tapping" | "voice" | "mri";

export interface PartInfo {
  key: PartKey;
  label: string;
  /** What the clinician provides for this part. */
  provides: string;
  /** Whether this build can upload and process it. Only MRI can today. */
  available: boolean;
}

/** The four parts of the platform, each processed and shown on its own. */
export const PARTS: readonly PartInfo[] = [
  { key: "gait", label: "Gait", provides: "A phone video of the person walking", available: false },
  { key: "tapping", label: "Finger tapping", provides: "A phone recording of finger tapping", available: false },
  { key: "voice", label: "Voice", provides: "A phone voice recording", available: false },
  { key: "mri", label: "MRI", provides: "One person's T1 and DTI study (zipped DICOM)", available: true },
];

export const PART_NOT_BUILT = "Not in this build yet";
export const NO_COMBINED_RESULT =
  "Each part is processed on its own and shown on its own. There is no combined result: the step that would combine them is not built yet.";
