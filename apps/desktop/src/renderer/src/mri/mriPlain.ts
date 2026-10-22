import type { MriResult, MriScore, MriTrackRecord } from "./mriTypes";

// Plain wording for the MRI result page. It states what the model's output is, without
// advice: no "weigh it", no "treat with caution". The numbers behind each line are in the
// page's "More details" sections. It is a reading of one modality's raw score, not a
// priority: no High/Moderate/Low and no percentage (the score is not calibrated). The
// strength labels describe how stable the lean is across model copies, never risk.

export type Lean = "none" | "higher" | "lower";
export type Strength = "consistent" | "moderate" | "weak";

export interface PlainReading {
  lean: Lean;
  /** How stable the lean is across the fold copies; null when there is no lean. */
  strength: Strength | null;
  headline: string;
  detail: string;
  /** "41 of 50 model copies agree", or null when the result has no copy counts. */
  agreement: string | null;
  /** "In testing, when the model leaned this way it was right for 10 of 16 people." */
  trackRecord: string | null;
}

/** Share of fold copies that must agree with the model's side for a "moderate" lean.
 *  Fixed on 22 October 2026, before any further test scans were read, and applied to
 *  every result the same way. Do not change it to make one case come out differently. */
export const MODERATE_AGREEMENT = 0.8;

export const STRENGTH_LABELS: Record<Strength, string> = {
  consistent: "Consistent",
  moderate: "Mostly consistent",
  weak: "Inconsistent",
};

export const STRENGTH_TEXT: Record<Strength, string> = {
  consistent: "The whole 95% interval is on this side: every part of the model agrees.",
  moderate: "Most model copies agree, but the 95% interval reaches the other side.",
  weak: "The model copies disagree, so this lean depends on which people the model learned from.",
};

type ScoreLike = Pick<MriScore, "value" | "interval_95"> &
  Partial<Pick<MriScore, "n_copies" | "copies_above_zero">>;

/** The reading of one result: which side the model's score is on (direction), how many
 *  fold copies agree (strength), and how often that direction was right in testing. */
export function plainReading(score: ScoreLike, track?: MriTrackRecord | null): PlainReading {
  const [low, high] = score.interval_95;
  if (score.value === 0) {
    return {
      lean: "none",
      strength: null,
      headline: "No lean",
      detail: "The score is exactly on the model's boundary.",
      agreement: null,
      trackRecord: null,
    };
  }
  const pd = score.value > 0;
  const lean: Lean = pd ? "higher" : "lower";
  const n = score.n_copies ?? null;
  const above = score.copies_above_zero ?? null;
  const agreeing = n !== null && above !== null ? (pd ? above : n - above) : null;
  const wholeInterval = pd ? low > 0 : high < 0;
  const strength: Strength = wholeInterval
    ? "consistent"
    : agreeing !== null && n !== null && n > 0 && agreeing / n >= MODERATE_AGREEMENT
      ? "moderate"
      : "weak";
  const record = track ? (pd ? track.leans_pd : track.leans_control) : null;
  return {
    lean,
    strength,
    headline: pd
      ? "Leans towards a Parkinson's-like pattern"
      : "Leans towards a control-like pattern",
    detail: STRENGTH_TEXT[strength],
    agreement:
      agreeing !== null && n !== null ? `${agreeing} of ${n} model copies agree with this side.` : null,
    trackRecord:
      record && record.n > 0
        ? `In testing, when the model leaned this way it was right for ${record.correct} of ${record.n} people.`
        : null,
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

/** The pipeline's reasons for not using this part (e.g. "DTI failed a processing step or
 *  check: ..."), from the result's validation problems. Empty when there are none. */
export function partProblems(result: Pick<MriResult, "validation">, part: ScanPart): string[] {
  const tag = part === "t1" ? /\bT1\b/ : /\bDTI\b/;
  return (result.validation?.problems ?? []).filter((p) => tag.test(p));
}

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
