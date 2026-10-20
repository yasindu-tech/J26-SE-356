import { beforeEach, describe, expect, it, vi } from "vitest";
import type { GaitAnalysisStatus } from "./gaitTypes";
import { MOCK_GAIT_NOT_SCORED, MOCK_GAIT_SCORED_CLEAR } from "./mockGaitResult";

vi.mock("../api/gait.api", () => ({ getGaitStatus: vi.fn(), getGaitResult: vi.fn() }));

import { getGaitResult, getGaitStatus } from "../api/gait.api";
import { getGaitRun } from "./gaitRun";

function status(over: Partial<GaitAnalysisStatus>): GaitAnalysisStatus {
  return {
    id: "a1",
    status: "running",
    created_at: "2026-10-20T08:00:00",
    finished_at: null,
    elapsed_s: 12.5,
    one_person_confirmed: true,
    subject_label: null,
    error: null,
    abstain_reason: null,
    ...over,
  };
}

beforeEach(() => {
  vi.mocked(getGaitStatus).mockReset();
  vi.mocked(getGaitResult).mockReset();
});

describe("getGaitRun", () => {
  it("does not fetch a result while the analysis is waiting or running", async () => {
    vi.mocked(getGaitStatus).mockResolvedValue(status({ status: "queued" }));
    const run = await getGaitRun("a1");
    expect(run).toEqual({ id: "a1", state: "queued", elapsedSeconds: 12.5, result: null, error: null });
    expect(getGaitResult).not.toHaveBeenCalled();
  });

  it("fetches the record once the analysis is scored", async () => {
    vi.mocked(getGaitStatus).mockResolvedValue(status({ status: "scored" }));
    vi.mocked(getGaitResult).mockResolvedValue(MOCK_GAIT_SCORED_CLEAR);
    const run = await getGaitRun("a1");
    expect(run.state).toBe("scored");
    expect(run.result?.video).toBe("w-e40cda55");
  });

  it("keeps the plain-words reason from the status row when the record has none", async () => {
    vi.mocked(getGaitStatus).mockResolvedValue(
      status({ status: "abstained", abstain_reason: "The walk has 2 real strides." }),
    );
    vi.mocked(getGaitResult).mockResolvedValue({ ...MOCK_GAIT_NOT_SCORED, reasons: [] });
    const run = await getGaitRun("a1");
    expect(run.state).toBe("abstained");
    expect(run.result?.reasons).toEqual(["The walk has 2 real strides."]);
  });

  it("returns the error of a failed analysis without a record", async () => {
    vi.mocked(getGaitStatus).mockResolvedValue(status({ status: "failed", error: "The video could not be read." }));
    const run = await getGaitRun("a1");
    expect(run).toMatchObject({ state: "failed", result: null, error: "The video could not be read." });
    expect(getGaitResult).not.toHaveBeenCalled();
  });
});
