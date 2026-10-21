// The gait result record (`gait-result/1`) as the backend returns it from
// GET /api/v1/gait/analyses/{id}/result. Written by models/gait/src/predict_video.py.
// A video that was not scored has the score fields null or empty.

export interface GaitCheck {
  value: number;
  limit: number;
  ok: boolean;
}

export type GaitCheckName = "frame_rate" | "leg_visibility" | "filled_share" | "facing" | "strides";

export interface GaitGate {
  passed: boolean;
  reasons: string[];
  warnings: string[];
  /** A check the pipeline did not run is missing: outside 20 to 60 fps there is no "strides". */
  checks: Partial<Record<GaitCheckName, GaitCheck>>;
}

export interface GaitWalk {
  seconds: number;
  start_s: number;
  end_s: number;
  fps: number;
}

export interface GaitFeature {
  name: string;
  value: number;
  shap: number;
  direction: "towards" | "away";
  strength: string;
  sentence: string;
}

export interface GaitResult {
  modality: "gait";
  schema_version: string;
  available: boolean;
  /** "scored", "abstained" or "failed", as the backend sets it. Absent on the mock records. */
  status?: "scored" | "abstained" | "failed";
  /** Why the analysis failed. */
  error?: string | null;
  /** A short hash of the video, not its name. Null when the analysis failed before reading it. */
  video: string | null;
  reasons: string[];
  warnings: string[];
  /** Null when the analysis failed before the video was checked. */
  gate: GaitGate | null;
  walk: GaitWalk | null;
  predicted_class: number | null;
  predicted_label: string | null;
  class_labels: string[];
  probabilities: number[] | null;
  probabilities_calibrated: boolean;
  expected_severity: number | null;
  expected_severity_note: string;
  top_features: GaitFeature[];
  feature_vector: Record<string, number> | null;
  model_version: string | null;
  one_person_confirmed: boolean;
}

/** Where a gait analysis stands, as the backend reports it (queued, running, then one final state). */
export type GaitRunState = "queued" | "running" | "scored" | "abstained" | "failed";

/** GET /api/v1/gait/analyses/{id}: the status only. It has no step list or queue position. */
export interface GaitAnalysisStatus {
  id: string;
  status: GaitRunState;
  created_at: string;
  finished_at: string | null;
  /** Seconds since the upload. */
  elapsed_s: number;
  one_person_confirmed: boolean;
  /** The pseudonymous label typed on the Patient details step, if any. */
  subject_label: string | null;
  error: string | null;
  /** Why the video could not be scored, in plain words. */
  abstain_reason: string | null;
  /** The video with the skeleton and heel strikes drawn on it can be played. */
  has_overlay: boolean;
}

export interface GaitAnalysisList {
  items: GaitAnalysisStatus[];
  total: number;
}

export interface GaitRunStatus {
  id: string;
  state: GaitRunState;
  elapsedSeconds: number;
  /** Set once the state is scored or abstained. */
  result: GaitResult | null;
  /** Set when the state is failed. */
  error: string | null;
}
