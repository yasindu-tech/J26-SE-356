import { beforeEach, describe, expect, it, vi } from "vitest";
import { failedFixture, scoredFixture } from "../../../shared/tappingResult.fixture";
import type { TappingAnalysisStatus } from "./tappingTypes";

// getTappingRun follows the gait one: the status first, the result only once it is final, and
// the stored result is checked before it is trusted.

const api = vi.hoisted(() => ({ getTappingStatus: vi.fn(), getTappingResult: vi.fn() }));
vi.mock("../api/tapping.api", () => api);

import { getTappingRun } from "./tappingRun";

const status = (over: Partial<TappingAnalysisStatus>): TappingAnalysisStatus => ({
  id: "a1",
  status: "scored",
  target: "ge2",
  created_at: "2026-10-22T09:00:00",
  finished_at: "2026-10-22T09:00:12",
  elapsed_s: 12.4,
  subject_label: "p-09",
  error: null,
  abstain_reason: null,
  ...over,
});

beforeEach(() => {
  api.getTappingStatus.mockReset();
  api.getTappingResult.mockReset();
});

describe("getTappingRun", () => {
  it.each(["queued", "running"] as const)("does not fetch the result while %s", async (state) => {
    api.getTappingStatus.mockResolvedValue(status({ status: state, finished_at: null }));
    expect(await getTappingRun("a1")).toEqual({
      id: "a1",
      state,
      elapsedSeconds: 12.4,
      result: null,
      error: null,
    });
    expect(api.getTappingResult).not.toHaveBeenCalled();
  });

  it("returns the error of a failed analysis without fetching a result", async () => {
    api.getTappingStatus.mockResolvedValue(status({ status: "failed", error: "the analysis stopped" }));
    expect(await getTappingRun("a1")).toMatchObject({ state: "failed", result: null, error: "the analysis stopped" });
    expect(api.getTappingResult).not.toHaveBeenCalled();
  });

  it("fetches and checks the result of a scored analysis", async () => {
    api.getTappingStatus.mockResolvedValue(status({}));
    api.getTappingResult.mockResolvedValue(scoredFixture());
    const run = await getTappingRun("a1");
    expect(run.state).toBe("scored");
    expect(run.result?.status).toBe("scored");
  });

  it("keeps the program's reason for a video it could not score", async () => {
    api.getTappingStatus.mockResolvedValue(status({ status: "abstained", abstain_reason: "x" }));
    api.getTappingResult.mockResolvedValue(failedFixture());
    const run = await getTappingRun("a1");
    expect(run.state).toBe("abstained");
    expect(run.result).toMatchObject({ status: "failed", reason: failedFixture().reason });
  });

  it("treats a stored result of the wrong shape as failed, not as a score", async () => {
    api.getTappingStatus.mockResolvedValue(status({}));
    for (const broken of [{ ...scoredFixture(), schema_version: "tapping-result/9" }, {}, null, "x"]) {
      api.getTappingResult.mockResolvedValue(broken);
      expect(await getTappingRun("a1")).toMatchObject({
        state: "failed",
        result: null,
        error: "The result could not be read.",
      });
    }
  });
});
