import type { GaitResult, GaitRunStatus } from "./gaitTypes";
import { MOCK_GAIT_NOT_SCORED, MOCK_GAIT_SCORED_CLEAR, MOCK_GAIT_SCORED_MIXED } from "./mockGaitResult";

// Stand-in for the gait service until the screen is connected to it (GAIT-31). It pretends to
// upload and to analyse for a few seconds, then returns one of the mock records, picked by the
// video's file name: "short" gives the mixed score, "bad" gives a video that was not scored,
// "fail" gives a failed analysis, anything else gives the clear score. The screen marks every
// result from here as demo data.

export const GAIT_SOURCE_IS_MOCK = true;

const RUN_SECONDS = 4;
const UPLOAD_MS = 1200;
const runs = new Map<string, { startedAt: number; name: string; confirmed: boolean }>();

export function mockResultFor(name: string, confirmed: boolean): GaitResult | null {
  const lower = name.toLowerCase();
  if (lower.includes("fail")) return null;
  const base = lower.includes("bad")
    ? MOCK_GAIT_NOT_SCORED
    : lower.includes("short")
      ? MOCK_GAIT_SCORED_MIXED
      : MOCK_GAIT_SCORED_CLEAR;
  return { ...base, one_person_confirmed: confirmed };
}

export function startGaitAnalysis(
  input: { file: File; onePersonConfirmed: boolean },
  onProgress: (fraction: number) => void,
): Promise<{ id: string }> {
  return new Promise((resolve) => {
    const started = Date.now();
    const timer = window.setInterval(() => {
      const fraction = Math.min(1, (Date.now() - started) / UPLOAD_MS);
      onProgress(fraction);
      if (fraction >= 1) {
        window.clearInterval(timer);
        const id = `mock-${started}`;
        runs.set(id, { startedAt: Date.now(), name: input.file.name, confirmed: input.onePersonConfirmed });
        resolve({ id });
      }
    }, 100);
  });
}

export function getGaitRun(id: string): Promise<GaitRunStatus> {
  const run = runs.get(id);
  if (!run) return Promise.reject(new Error("Could not open that analysis."));
  const elapsedSeconds = (Date.now() - run.startedAt) / 1000;
  if (elapsedSeconds < RUN_SECONDS) {
    return Promise.resolve({ id, state: "running", elapsedSeconds, result: null, error: null });
  }
  const result = mockResultFor(run.name, run.confirmed);
  if (result === null) {
    return Promise.resolve({
      id,
      state: "failed",
      elapsedSeconds,
      result: null,
      error: "Something went wrong while reading the video.",
    });
  }
  return Promise.resolve({
    id,
    state: result.available ? "scored" : "abstained",
    elapsedSeconds,
    result,
    error: null,
  });
}
