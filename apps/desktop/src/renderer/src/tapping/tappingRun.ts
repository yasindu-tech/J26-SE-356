import { getTappingResult, getTappingStatus } from "../api/tapping.api";
import { validateTappingResult } from "../../../shared/tappingResult";
import type { TappingRunStatus } from "./tappingTypes";

const UNREADABLE = "The result could not be read.";

/** Where one tapping analysis stands. Once it is final this also fetches and checks the result. */
export async function getTappingRun(id: string): Promise<TappingRunStatus> {
  const status = await getTappingStatus(id);
  const base = { id, elapsedSeconds: status.elapsed_s };
  if (status.status === "queued" || status.status === "running") {
    return { ...base, state: status.status, result: null, error: null };
  }
  if (status.status === "failed") {
    return { ...base, state: "failed", result: null, error: status.error };
  }
  // Scored, or abstained: the stored result says why. Its shape is checked before it is shown.
  const result = validateTappingResult(await getTappingResult(id));
  if (result === null) return { ...base, state: "failed", result: null, error: UNREADABLE };
  return { ...base, state: status.status, result, error: null };
}
