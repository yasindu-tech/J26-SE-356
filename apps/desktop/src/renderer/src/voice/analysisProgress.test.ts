import { describe, expect, it } from "vitest";
import { ANALYSIS_STEPS, ESTIMATE_CAP, progressView } from "./analysisProgress";

describe("progressView", () => {
  it("follows the measured upload and calls it measured", () => {
    expect(progressView({ phase: "uploading", uploaded: 0 })).toEqual({
      fraction: 0,
      step: 0,
      estimated: false,
    });
    const half = progressView({ phase: "uploading", uploaded: 0.5 });
    expect(half.fraction).toBeCloseTo(0.1);
    expect(half.estimated).toBe(false);
  });

  it("clamps bad upload values instead of jumping past the upload share", () => {
    expect(progressView({ phase: "uploading", uploaded: 3 }).fraction).toBeCloseTo(0.2);
    expect(progressView({ phase: "uploading", uploaded: -1 }).fraction).toBe(0);
  });

  it("marks the analysis part as an estimate that only moves forward", () => {
    let previous = progressView({ phase: "uploading", uploaded: 1 }).fraction;
    for (const elapsedS of [0, 0.5, 1, 2, 3, 5, 10]) {
      const view = progressView({ phase: "analysing", elapsedS });
      expect(view.estimated).toBe(true);
      expect(view.fraction).toBeGreaterThanOrEqual(previous);
      previous = view.fraction;
    }
  });

  it("never reaches 100% before the real result arrives", () => {
    for (const elapsedS of [5, 30, 120, 10_000]) {
      expect(progressView({ phase: "analysing", elapsedS }).fraction).toBeLessThanOrEqual(
        ESTIMATE_CAP,
      );
    }
  });

  it("walks through the analysis steps in order and stops at the last", () => {
    const steps = [0, 1.5, 2.5, 60].map(
      (elapsedS) => progressView({ phase: "analysing", elapsedS }).step,
    );
    expect(steps).toEqual([1, 2, 3, 3]);
    expect(ANALYSIS_STEPS).toHaveLength(4);
  });
});
