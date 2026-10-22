import type { MriResult, MriScore } from "./mriTypes";

// Plain wording for the MRI result page. It states what the model's output is, without
// advice: no "weigh it", no "treat with caution". The numbers behind each line are in the
// page's "More details" sections. It is a reading of one modality's raw score, not a
// priority: no High/Moderate/Low and no percentage (the score is not calibrated).

export type Lean = "none" | "higher" | "lower";

export interface PlainReading {
  lean: Lean;
  headline: string;
  detail: string;
}

/** Which side of the model's boundary (log-odds 0) the score's 95% interval sits on. */
export function plainReading(score: Pick<MriScore, "interval_95">): PlainReading {
  const [low, high] = score.interval_95;
  if (low <= 0 && high >= 0) {
    return {
      lean: "none",
      headline: "No clear lean",
      detail: "The score's 95% interval reaches both sides of the model's boundary.",
    };
  }
  if (low > 0) {
    return {
      lean: "higher",
      headline: "Leans towards a Parkinson's-like pattern",
      detail: "The whole 95% interval of the score is on the Parkinson's side of the model's boundary.",
    };
  }
  return {
    lean: "lower",
    headline: "Leans towards a control-like pattern",
    detail: "The whole 95% interval of the score is on the control side of the model's boundary.",
  };
}

export type ScanPart = "t1" | "dti";

export const PART_TITLES: Record<ScanPart, string> = {
  t1: "T1 scan",
  dti: "DTI scan",
};

/** What the quality-check picture of each part shows (rows of the picture, top to bottom). */
export const PART_CAPTIONS: Record<ScanPart, string> = {
  t1: "Top row: the T1 scan aligned to the standard brain template (side, front and top views). Bottom row: the tissue map the measurements come from.",
  dti: "Top row: FA map, aligned to the template (bright where water moves in one direction, as along white-matter tracts). Bottom row: MD map (how freely water moves).",
};

export type PartState = "used" | "not-used" | "missing";

export function partState(result: Pick<MriResult, "availability" | "inputs">, part: ScanPart): PartState {
  if (result.availability[part]) return "used";
  return result.inputs[part]?.found ? "not-used" : "missing";
}

export const PART_STATE_LABELS: Record<PartState, string> = {
  used: "Used",
  "not-used": "Not used",
  missing: "Not in the upload",
};

/** Only a data: URL of an image is shown; anything else is ignored. */
export function safeImage(src: string | null | undefined): string | null {
  return typeof src === "string" && src.startsWith("data:image/") ? src : null;
}

/** "Mean FA in the left corticospinal tract pushed the reading higher." */
export function driverText(d: { label: string; contribution: number }): string {
  return `${d.label} pushed the reading ${d.contribution >= 0 ? "higher" : "lower"}.`;
}

/** Eddy wall time in minutes, or null when it was not recorded. */
export function eddyMinutes(seconds: number | null | undefined): number | null {
  return typeof seconds === "number" && Number.isFinite(seconds) ? Math.round(seconds / 60) : null;
}
