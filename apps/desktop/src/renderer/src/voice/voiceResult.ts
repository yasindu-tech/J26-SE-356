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
  /** Plain-language explanations (analyze_audio.py). Absent in older result files. */
  explanations?: Explanations;
}

/** One SHAP driver, explained in plain words (models/voice/src/audio_explain.py). */
export interface ExplainedReason {
  feature: string;
  meaning: string;
  /** Measure group from the voice module, e.g. "jitter", "mfcc"; "" in older files. */
  group: string;
  sentence: string;
  effect: "raised" | "lowered";
  /** Where the value sits among healthy training voices (0-100). */
  percentileAmongHealthy: number;
  /** False when new audio measures this on a different scale from the training data. */
  comparable: boolean;
}

/** What the quality gate measured on the recording. Every field may be missing. */
export interface MeasuredQuality {
  sourceFormat?: string;
  sourceSampleRate?: number;
  durationS?: number;
  vowelS?: number;
  snrDb?: number;
  clippedFraction?: number;
}

export interface Explanations {
  summary: string;
  topReasons: ExplainedReason[];
  uncertainty: string[];
  quality: { passed: boolean; problems: string[]; measured: MeasuredQuality };
  modelOverview: { sentence: string; topGroups: { group: string; share: number }[] };
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

// Explanation text is shown to clinicians as written, so it must never use
// diagnostic wording (CLAUDE.md section 1) or claim a cause.
const UNSAFE_WORDS = /diagnos|\bpositive\b|\bnegative\b|\bcleared\b|discharged|\bcauses\b/i;

function safeText(fields: Fields, key: string): string {
  const value = text(fields, key);
  if (UNSAFE_WORDS.test(value)) {
    throw new InvalidResult(`"${key}" uses diagnostic wording`);
  }
  return value;
}

function safeTextList(fields: Fields, key: string): string[] {
  const values = textList(fields, key);
  if (values.some((value) => UNSAFE_WORDS.test(value))) {
    throw new InvalidResult(`"${key}" uses diagnostic wording`);
  }
  return values;
}

function child(fields: Fields, key: string): Fields {
  const value = fields[key];
  if (!isFields(value)) throw new InvalidResult(`"${key}" must be an object`);
  return value;
}

function optionalNumber(fields: Fields, key: string): number | undefined {
  const value = fields[key];
  if (value === undefined || value === null) return undefined;
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new InvalidResult(`"${key}" must be a number`);
  }
  return value;
}

function explainedReason(item: unknown): ExplainedReason {
  if (!isFields(item)) throw new InvalidResult(`"top_reasons" entries must be objects`);
  const effect = item["effect"];
  if (effect !== "raised" && effect !== "lowered") {
    throw new InvalidResult(`"effect" must be "raised" or "lowered"`);
  }
  const percentile = optionalNumber(item, "percentile_among_healthy");
  if (percentile === undefined || percentile < 0 || percentile > 100) {
    throw new InvalidResult(`"percentile_among_healthy" must be a number from 0 to 100`);
  }
  return {
    feature: text(item, "feature"),
    meaning: safeText(item, "meaning"),
    group: typeof item["group"] === "string" ? item["group"] : "",
    sentence: safeText(item, "sentence"),
    effect,
    percentileAmongHealthy: percentile,
    comparable: item["comparable"] !== false, // files written before this field existed
  };
}

function explanations(value: unknown): Explanations | undefined {
  if (value === undefined) return undefined;
  if (!isFields(value)) throw new InvalidResult(`"explanations" must be an object`);
  const quality = child(value, "quality");
  const measured = child(quality, "measured");
  const overview = child(value, "model_overview");
  const groups = overview["top_groups"];
  const reasons = value["top_reasons"];
  if (!Array.isArray(groups) || !Array.isArray(reasons)) {
    throw new InvalidResult(`"top_groups" and "top_reasons" must be lists`);
  }
  const format = measured["source_format"];
  return {
    summary: safeText(value, "summary"),
    topReasons: reasons.map(explainedReason),
    uncertainty: safeTextList(value, "uncertainty"),
    quality: {
      passed: quality["passed"] === true,
      problems: safeTextList(quality, "problems"),
      measured: {
        sourceFormat: typeof format === "string" ? format : undefined,
        sourceSampleRate: optionalNumber(measured, "source_sample_rate"),
        durationS: optionalNumber(measured, "duration_s"),
        vowelS: optionalNumber(measured, "vowel_s"),
        snrDb: optionalNumber(measured, "snr_db"),
        clippedFraction: optionalNumber(measured, "clipped_fraction"),
      },
    },
    modelOverview: {
      sentence: safeText(overview, "sentence"),
      topGroups: groups.map((group) => {
        if (!isFields(group)) throw new InvalidResult(`"top_groups" entries must be objects`);
        const share = optionalNumber(group, "share");
        if (share === undefined || share < 0 || share > 1) {
          throw new InvalidResult(`"share" must be a number from 0 to 1`);
        }
        return { group: text(group, "group"), share };
      }),
    },
  };
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
    explanations: explanations(value["explanations"]),
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
