// Mirrors services/backend/app/schemas/mri.py (MriResult = predict_scan.py's
// schema "mri-result/1"). Hand-written until types are generated from the
// backend's OpenAPI schema (API-Standards.md §3); keep the two in step.

export type AnalysisStatus = "queued" | "running" | "scored" | "abstained" | "failed";
export type FinalStatus = Extract<AnalysisStatus, "scored" | "abstained" | "failed">;

export const MRI_STEPS = ["convert", "t1", "dti", "features", "score", "explain"] as const;
export type MriStep = (typeof MRI_STEPS)[number];

export interface MriAnalysisCreated {
  id: string;
  status: AnalysisStatus;
}

export interface MriAnalysisStatus {
  id: string;
  status: AnalysisStatus;
  created_at: string;
  finished_at: string | null;
  steps: string[];
  done: string[];
  current_step: string | null;
  note: string;
  error: string | null;
  abstain_reason: string | null;
}

export interface MriAnalysisList {
  items: MriAnalysisStatus[];
  total: number;
}

export interface MriScore {
  kind: "log_odds"; // raw model score; 0 is the boundary the model learned
  value: number;
  interval_95: [number, number];
  interval_method: string;
  calibrated: boolean;
  n_training: number;
  meaning: string;
}

export interface MriFeatureVector {
  names: string[];
  labels: string[];
  raw: number[];
  harmonised: number[]; // the MRI embedding the fusion model will take
  scanner_batch: Record<string, string>;
  outside_training_range: string[];
}

export interface MriDriver {
  feature: string;
  label: string;
  value: number | null;
  contribution: number; // log-odds
  direction: "towards PD" | "towards control";
  text: string;
}

export interface MriExplanation {
  kind: string;
  method: string;
  base_value: number;
  feature_attributions: Record<string, number>;
  top_drivers: MriDriver[];
  caveat: string;
}

/** One input series. Details vary by modality, so only `found` is guaranteed. */
export interface MriSeriesInput {
  found: boolean;
  series_description?: string;
  scanner_model?: string;
  manufacturer?: string;
  voxel_mm?: number[];
  status?: string;
  checks?: Record<string, number | boolean | null>;
  qc_image?: string | null; // data: URL of the QC picture
}

export interface MriModelCard {
  set?: string;
  purpose?: string;
  training_people?: { total: number; pd: number; control: number };
  cross_validated?: {
    auc_mean: number;
    auc_ci: [number, number];
    balanced_accuracy_mean: number;
    balanced_accuracy_ci: [number, number];
    folds: number | null;
    repeats: number | null;
  };
  known_limits?: string[];
}

export interface MriResult {
  modality: "mri";
  schema_version: string;
  status: FinalStatus;
  abstain_reason: string | null;
  error: string | null;
  model_version: string | null;
  model_set: string | null;
  availability: Record<string, boolean>;
  inputs: Record<string, MriSeriesInput>;
  validation: { usable: boolean; problems: string[] } | null;
  feature_vector: MriFeatureVector | null;
  score: MriScore | null;
  explanation: MriExplanation | null;
  model_card: MriModelCard | null;
  disclaimer: string;
}
