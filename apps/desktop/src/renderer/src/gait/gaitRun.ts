import { getGaitResult, getGaitStatus } from "../api/gait.api";
import type { GaitRunStatus } from "./gaitTypes";

/** Where one gait analysis stands. Once it is final this also fetches the result record. */
export async function getGaitRun(id: string): Promise<GaitRunStatus> {
  const status = await getGaitStatus(id);
  const base = { id, elapsedSeconds: status.elapsed_s };
  if (status.status === "queued" || status.status === "running") {
    return { ...base, state: status.status, result: null, error: null };
  }
  if (status.status === "failed") {
    return { ...base, state: "failed", result: null, error: status.error };
  }
  const result = await getGaitResult(id);
  // The status row carries the plain-words reason; keep it if the record left it out.
  const reasons =
    result.reasons.length === 0 && status.abstain_reason ? [status.abstain_reason] : result.reasons;
  return { ...base, state: status.status, result: { ...result, reasons }, error: null };
}
