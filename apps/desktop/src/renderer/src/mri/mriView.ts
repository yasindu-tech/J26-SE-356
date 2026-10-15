import { MRI_STEPS, type AnalysisStatus, type FinalStatus, type MriScore } from "./mriTypes";

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
    : `The whole 95% interval is ${low > 0 ? "above" : "below"} 0: the copies of the model agree this scan leans ${lean}.`;
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
