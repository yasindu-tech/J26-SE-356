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
  checks: Record<GaitCheckName, GaitCheck>;
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
  /** A short hash of the video, not its name. */
  video: string;
  reasons: string[];
  warnings: string[];
  gate: GaitGate;
  walk: GaitWalk;
  predicted_class: number | null;
  predicted_label: string | null;
  class_labels: string[];
  probabilities: number[] | null;
  probabilities_calibrated: boolean;
  expected_severity: number | null;
  expected_severity_note: string;
  top_features: GaitFeature[];
  feature_vector: Record<string, number> | null;
  model_version: string;
  one_person_confirmed: boolean;
}

/** Where a gait analysis stands, as the backend reports it (queued, running, then one final state). */
export type GaitRunState = "queued" | "running" | "scored" | "abstained" | "failed";

export interface GaitRunStatus {
  id: string;
  state: GaitRunState;
  elapsedSeconds: number;
  /** Set once the state is scored or abstained. */
  result: GaitResult | null;
  /** Set when the state is failed. */
  error: string | null;
}
