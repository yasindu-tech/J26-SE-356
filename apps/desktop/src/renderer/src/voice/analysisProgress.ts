// Progress shown while a voice recording is uploaded and analysed.
//
// Only the upload is measured (the browser reports bytes sent). The backend answers once,
// when the analysis is done, so the steps after the upload are an estimate from how long
// an analysis usually takes. The estimate slows down as it goes and never reaches 100%:
// only the real result finishes the bar.

/** What a typical analysis does, in order (models/voice/src/analyze_audio.py). */
export const ANALYSIS_STEPS = [
  "Uploading the recording",
  "Checking the recording quality",
  "Measuring the voice",
  "Scoring and explaining",
] as const;

/** Share of the bar given to the upload; the rest is the analysis. */
const UPLOAD_SHARE = 0.2;
/** Highest the estimate may reach before the real result arrives. */
export const ESTIMATE_CAP = 0.95;
/** Seconds in which the estimate covers about two thirds of the analysis part. */
const TYPICAL_ANALYSIS_S = 3;
/** Seconds after the upload at which each analysis step is assumed to start. */
const STEP_STARTS_S = [0, 1, 2.2] as const;

export type ProgressPhase =
  | { phase: "uploading"; uploaded: number } // 0 to 1, measured
  | { phase: "analysing"; elapsedS: number }; // seconds since the upload finished

export interface ProgressView {
  /** 0 to 1 for the bar. */
  fraction: number;
  /** Index into ANALYSIS_STEPS of the step shown as current. */
  step: number;
  /** True once the bar is an estimate rather than a measurement. */
  estimated: boolean;
}

export function progressView(state: ProgressPhase): ProgressView {
  if (state.phase === "uploading") {
    const uploaded = Math.min(Math.max(state.uploaded, 0), 1);
    return { fraction: uploaded * UPLOAD_SHARE, step: 0, estimated: false };
  }
  const t = Math.max(state.elapsedS, 0);
  const analysed = 1 - Math.exp(-t / TYPICAL_ANALYSIS_S); // fast at first, never 1
  const fraction = Math.min(UPLOAD_SHARE + (1 - UPLOAD_SHARE) * analysed, ESTIMATE_CAP);
  const step = STEP_STARTS_S.filter((start) => t >= start).length; // 1 to 3: step 0 is the upload
  return { fraction, step, estimated: true };
}
