// The finger-tapping result types are shared with the main process, which checks the program's
// output against them (src/shared/tappingResult.ts). Re-exported here for the renderer.

export type {
  Num,
  TappingContribution,
  TappingFailed,
  TappingResult,
  TappingScored,
  TappingTarget,
} from "../../../shared/tappingResult";
export type { TappingSavePdfReply } from "../../../shared/tappingIpc";

/** Same states as the backend's MRI and gait analyses. */
export type TappingRunState = "queued" | "running" | "scored" | "abstained" | "failed";

/** One row of the backend's list and status endpoints (services/backend/app/schemas/tapping.py). */
export interface TappingAnalysisStatus {
  id: string;
  status: TappingRunState;
  target: "ge2" | "ge1";
  created_at: string;
  finished_at: string | null;
  /** Seconds since the upload. */
  elapsed_s: number;
  /** The pseudonymous label typed on the Patient details step, if any. */
  subject_label: string | null;
  error: string | null;
  /** Why the video could not be scored, in plain words. */
  abstain_reason: string | null;
}

export interface TappingAnalysisList {
  items: TappingAnalysisStatus[];
  total: number;
}

export interface TappingRunStatus {
  id: string;
  state: TappingRunState;
  elapsedSeconds: number;
  /** Set once the state is scored or abstained. */
  result: import("../../../shared/tappingResult").TappingResult | null;
  /** Set when the state is failed. */
  error: string | null;
}

/** How one finger-tapping analysis ended: a scored result, or a failure with a plain message. */
export type TappingOutcome =
  | { kind: "scored"; result: import("../../../shared/tappingResult").TappingScored }
  | { kind: "failed"; message: string; ref: string | null };
