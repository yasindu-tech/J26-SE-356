// The voice module's result vector (models/voice/README.md decision D5), as
// written by models/voice/src/demo_voice.py into demo_results.json.
//
// The file arrives as untrusted JSON, so it is parsed here into a typed shape
// before anything is shown. A file that would make the card say something
// unsafe (a diagnosis word, a score without its confidence interval) is
// rejected with a reason, never shown in part.

/** Screening output, never a diagnosis (CLAUDE.md section 1, decision D6). */
export type TriagePriority = "prioritised_referral" | "not_prioritised";

export interface ShapDriver {
  feature: string;
  /** Contribution to the model's log-odds for this person; > 0 pushes the score up. */
  shapLogOdds: number;
}

interface VoiceResultBase {
  /** Pseudonymised by the voice module (data.pseudonym), never a raw dataset id. */
  person: string;
  /** The dataset's own label for this hold-out person. Demo check only. */
  referenceLabel: string;
  sexFairnessNote: string;
  pipelineVersion: string;
  nTraining: number;
  note: string;
}

export interface ScoredVoiceResult extends VoiceResultBase {
  kind: "scored";
  riskScore: number;
  ciLower: number;
  ciUpper: number;
  triagePriority: TriagePriority;
  shapTop5: ShapDriver[];
}

/** Input failed the quality gate: no score, voice is treated as absent. */
export interface AbstainedVoiceResult extends VoiceResultBase {
  kind: "abstained";
  problems: string[];
}

export type VoiceResult = ScoredVoiceResult | AbstainedVoiceResult;

export type ParseOutcome = { ok: true; results: VoiceResult[] } | { ok: false; message: string };

/** The voice module's referral threshold (voice_module.TRIAGE_THRESHOLD). */
export const TRIAGE_THRESHOLD = 0.5;

const TRIAGE_VALUES: readonly TriagePriority[] = ["prioritised_referral", "not_prioritised"];

class InvalidResult extends Error {}

type Fields = Record<string, unknown>;

function isFields(value: unknown): value is Fields {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function text(fields: Fields, key: string): string {
  const value = fields[key];
  if (typeof value !== "string") throw new InvalidResult(`"${key}" must be text`);
  return value;
}

function count(fields: Fields, key: string): number {
  const value = fields[key];
  if (typeof value !== "number" || !Number.isInteger(value) || value < 0) {
    throw new InvalidResult(`"${key}" must be a whole number`);
  }
  return value;
}

function probability(fields: Fields, key: string): number {
  const value = fields[key];
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) {
    throw new InvalidResult(`"${key}" must be a number from 0 to 1`);
  }
  return value;
}

function textList(fields: Fields, key: string): string[] {
  const value = fields[key];
  if (!Array.isArray(value) || !value.every((item) => typeof item === "string")) {
    throw new InvalidResult(`"${key}" must be a list of text`);
  }
  return value;
}

function triagePriority(fields: Fields): TriagePriority {
  const value = fields["triage_priority"];
  const match = TRIAGE_VALUES.find((allowed) => allowed === value);
  if (match === undefined) {
    throw new InvalidResult(
      `"triage_priority" must be "prioritised_referral" or "not_prioritised", got ${JSON.stringify(value)}`,
    );
  }
  return match;
}

function shapDrivers(fields: Fields): ShapDriver[] {
  const value = fields["shap_top5"];
  if (!Array.isArray(value)) throw new InvalidResult(`"shap_top5" must be a list`);
  return value.map((item) => {
    if (!isFields(item)) throw new InvalidResult(`"shap_top5" entries must be objects`);
    const shapLogOdds = item["shap_log_odds"];
    if (typeof shapLogOdds !== "number" || !Number.isFinite(shapLogOdds)) {
      throw new InvalidResult(`"shap_log_odds" must be a number`);
    }
    return { feature: text(item, "feature"), shapLogOdds };
  });
}

function parseOne(value: unknown): VoiceResult {
  if (!isFields(value)) throw new InvalidResult("each result must be an object");
  if (value["modality"] !== "voice") throw new InvalidResult(`"modality" must be "voice"`);

  const base: VoiceResultBase = {
    person: text(value, "person"),
    referenceLabel: text(value, "reference_label"),
    sexFairnessNote: text(value, "sex_fairness_note"),
    pipelineVersion: text(value, "pipeline_version"),
    nTraining: count(value, "n_training"),
    note: text(value, "note"),
  };

  if (value["abstained"] === true) {
    return { kind: "abstained", ...base, problems: textList(value, "problems") };
  }
  if (value["abstained"] !== false) throw new InvalidResult(`"abstained" must be true or false`);

  const riskScore = probability(value, "risk_score");
  const ciLower = probability(value, "ci_lower");
  const ciUpper = probability(value, "ci_upper");
  if (!(ciLower <= riskScore && riskScore <= ciUpper)) {
    throw new InvalidResult("the risk score must lie inside its confidence interval");
  }
  const priority = triagePriority(value);
  const expected: TriagePriority =
    riskScore >= TRIAGE_THRESHOLD ? "prioritised_referral" : "not_prioritised";
  if (priority !== expected) {
    throw new InvalidResult(
      `"triage_priority" does not match the risk score at threshold ${TRIAGE_THRESHOLD}`,
    );
  }
  return {
    kind: "scored",
    ...base,
    riskScore,
    ciLower,
    ciUpper,
    triagePriority: priority,
    shapTop5: shapDrivers(value),
  };
}

/** Parses a voice result file: one result object, or a list of them (demo_results.json). */
export function parseVoiceResults(content: unknown): ParseOutcome {
  const items = Array.isArray(content) ? content : [content];
  if (items.length === 0) return { ok: false, message: "The file contains no results." };
  try {
    return { ok: true, results: items.map(parseOne) };
  } catch (error) {
    if (error instanceof InvalidResult) {
      return { ok: false, message: `This is not a valid voice result file: ${error.message}.` };
    }
    throw error;
  }
}

/** Display text for a triage priority. Never "positive", "negative" or "cleared". */
export function triageLabel(priority: TriagePriority): string {
  switch (priority) {
    case "prioritised_referral":
      return "Prioritised referral";
    case "not_prioritised":
      return "Not prioritised";
    default: {
      const exhaustive: never = priority;
      return exhaustive;
    }
  }
}
