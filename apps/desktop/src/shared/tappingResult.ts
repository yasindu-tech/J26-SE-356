// The result of one finger-tapping analysis ("tapping-result/1"), written by
// models/tapping/src/predict_clip.py. The main process checks it with `parseTappingResult`
// before it reaches the renderer, so everything past that point can trust the shape.
// A number that was not available arrives as null; it is never read as 0.

export const TAPPING_SCHEMA = "tapping-result/1";
export const MAX_RESULT_BYTES = 20 * 1024 * 1024;
export const MAX_POINTS = 600;
export const MAX_PREVIEWS = 12;
export const MAX_PREVIEW_CHARS = 400_000;
export const MAX_PDF_CHARS = 14 * 1024 * 1024;
export const TARGETS = ["ge2", "ge1"] as const;

export type TappingTarget = (typeof TARGETS)[number];
export type Num = number | null;

export interface TappingContribution {
  feature_label: string;
  value_text: string;
  percentile_text: string;
  shap: Num;
  sentence: string;
}

export interface TappingScored {
  schema_version: typeof TAPPING_SCHEMA;
  status: "scored";
  clip_ref: string;
  target: TappingTarget;
  recording: { duration_s: Num; fps: Num; hand_found_pct: Num; n_taps: number };
  quality: { flags: { name: string; text: string }[]; notes: string[] };
  score: {
    value: number;
    band_low: number;
    band_high: number;
    band_basis: string;
    threshold: number;
    above_threshold: boolean;
    cv_auroc: Num;
    cv_ci_low: Num;
    cv_ci_high: Num;
  };
  contributions: TappingContribution[];
  curve: {
    times: Num[];
    amps: Num[];
    line_x: Num[];
    line_y: Num[];
    slope_sentence: string;
    count_sentence: string;
    note: string;
    usable: boolean;
  };
  signal: { t: Num[]; smoothed: Num[]; peak_t: Num[]; peak_v: Num[] };
  preview_frames: { t: Num; jpeg_base64: string }[];
  model: { demo_model_sha8: string; folds_sha8: string };
  pdf_base64: string;
}

export interface TappingFailed {
  schema_version: typeof TAPPING_SCHEMA;
  status: "failed";
  reason: string;
  clip_ref: string | null;
  target: TappingTarget;
}

export type TappingResult = TappingScored | TappingFailed;

// ---------------------------------------------------------------------------- checks

type Obj = Record<string, unknown>;

const isObj = (v: unknown): v is Obj => typeof v === "object" && v !== null && !Array.isArray(v);
const isStr = (v: unknown, max = 2000): v is string => typeof v === "string" && v.length <= max;
const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const isNumOrNull = (v: unknown): v is Num => v === null || isNum(v);
const isHex8 = (v: unknown): v is string => typeof v === "string" && /^[0-9a-f]{8}$/.test(v);
const isBase64 = (v: unknown, max: number): v is string =>
  typeof v === "string" && v.length > 0 && v.length <= max && /^[A-Za-z0-9+/]+={0,2}$/.test(v);
const sameKeys = (o: Obj, keys: readonly string[]): boolean =>
  Object.keys(o).length === keys.length && keys.every((k) => k in o);

function numArray(v: unknown, max = MAX_POINTS): v is Num[] {
  return Array.isArray(v) && v.length <= max && v.every(isNumOrNull);
}

function strArray(v: unknown, max = 50): v is string[] {
  return Array.isArray(v) && v.length <= max && v.every((s) => isStr(s));
}

const isTarget = (v: unknown): v is TappingTarget => TARGETS.some((t) => t === v);

function checkScored(o: Obj): boolean {
  const { recording, quality, score, curve, signal, model } = o;
  if (!sameKeys(o, [
    "schema_version", "status", "clip_ref", "target", "recording", "quality", "score",
    "contributions", "curve", "signal", "preview_frames", "model", "pdf_base64",
  ])) return false; // prettier-ignore
  if (!isHex8(o["clip_ref"]) || !isTarget(o["target"])) return false;
  if (!isObj(recording) || !sameKeys(recording, ["duration_s", "fps", "hand_found_pct", "n_taps"])) {
    return false;
  }
  if (
    !isNumOrNull(recording["duration_s"]) || !isNumOrNull(recording["fps"]) ||
    !isNumOrNull(recording["hand_found_pct"]) || !Number.isInteger(recording["n_taps"])
  ) return false; // prettier-ignore
  if (!isObj(quality) || !sameKeys(quality, ["flags", "notes"]) || !strArray(quality["notes"])) {
    return false;
  }
  const flags = quality["flags"];
  if (!Array.isArray(flags) || flags.length > 20) return false;
  if (!flags.every((f) => isObj(f) && sameKeys(f, ["name", "text"]) && isStr(f["name"]) && isStr(f["text"]))) {
    return false;
  }
  if (!isObj(score) || !sameKeys(score, [
    "value", "band_low", "band_high", "band_basis", "threshold", "above_threshold",
    "cv_auroc", "cv_ci_low", "cv_ci_high",
  ])) return false; // prettier-ignore
  const unit = (v: unknown): boolean => isNum(v) && v >= 0 && v <= 1;
  if (!unit(score["value"]) || !unit(score["band_low"]) || !unit(score["band_high"])) return false;
  if ((score["band_low"] as number) > (score["band_high"] as number)) return false;
  if (!isStr(score["band_basis"]) || !isNum(score["threshold"])) return false;
  if (typeof score["above_threshold"] !== "boolean") return false;
  if (!isNumOrNull(score["cv_auroc"]) || !isNumOrNull(score["cv_ci_low"]) || !isNumOrNull(score["cv_ci_high"])) {
    return false;
  }
  const rows = o["contributions"];
  if (!Array.isArray(rows) || rows.length < 1 || rows.length > 10) return false;
  const rowOk = (r: unknown): boolean =>
    isObj(r) &&
    sameKeys(r, ["feature_label", "value_text", "percentile_text", "shap", "sentence"]) &&
    isStr(r["feature_label"], 80) && isStr(r["value_text"], 80) && isStr(r["percentile_text"], 120) &&
    isNumOrNull(r["shap"]) && isStr(r["sentence"], 400);
  if (!rows.every(rowOk)) return false;
  if (!isObj(curve) || !sameKeys(curve, [
    "times", "amps", "line_x", "line_y", "slope_sentence", "count_sentence", "note", "usable",
  ])) return false; // prettier-ignore
  if (!numArray(curve["times"]) || !numArray(curve["amps"]) || !numArray(curve["line_x"]) || !numArray(curve["line_y"])) {
    return false;
  }
  if (curve["times"].length !== curve["amps"].length || curve["line_x"].length !== curve["line_y"].length) {
    return false;
  }
  if (!isStr(curve["slope_sentence"], 400) || !isStr(curve["count_sentence"], 400)) return false;
  if (!isStr(curve["note"], 400)) return false;
  if (typeof curve["usable"] !== "boolean") return false;
  if (!isObj(signal) || !sameKeys(signal, ["t", "smoothed", "peak_t", "peak_v"])) return false;
  if (!numArray(signal["t"]) || !numArray(signal["smoothed"]) || !numArray(signal["peak_t"]) || !numArray(signal["peak_v"])) {
    return false;
  }
  if (signal["t"].length !== signal["smoothed"].length || signal["peak_t"].length !== signal["peak_v"].length) {
    return false;
  }
  const frames = o["preview_frames"];
  if (!Array.isArray(frames) || frames.length > MAX_PREVIEWS) return false;
  if (!frames.every((f) => isObj(f) && sameKeys(f, ["t", "jpeg_base64"]) && isNumOrNull(f["t"]) && isBase64(f["jpeg_base64"], MAX_PREVIEW_CHARS))) {
    return false;
  }
  if (!isObj(model) || !sameKeys(model, ["demo_model_sha8", "folds_sha8"])) return false;
  if (!isHex8(model["demo_model_sha8"]) || !isHex8(model["folds_sha8"])) return false;
  return isBase64(o["pdf_base64"], MAX_PDF_CHARS);
}

function checkFailed(o: Obj): boolean {
  return (
    sameKeys(o, ["schema_version", "status", "reason", "clip_ref", "target"]) &&
    isStr(o["reason"], 300) && o["reason"].length > 0 &&
    (o["clip_ref"] === null || isHex8(o["clip_ref"])) && isTarget(o["target"])
  ); // prettier-ignore
}

/** The checked result, or null when anything is off: unknown version, wrong type, wrong length. */
export function validateTappingResult(value: unknown): TappingResult | null {
  if (!isObj(value) || value["schema_version"] !== TAPPING_SCHEMA) return null;
  if (value["status"] === "scored") return checkScored(value) ? (value as unknown as TappingScored) : null;
  if (value["status"] === "failed") return checkFailed(value) ? (value as unknown as TappingFailed) : null;
  return null;
}

/** Parses the program's output text; null for oversized or invalid output. */
export function parseTappingResult(text: string): TappingResult | null {
  if (text.length > MAX_RESULT_BYTES) return null;
  try {
    return validateTappingResult(JSON.parse(text) as unknown);
  } catch {
    return null;
  }
}
