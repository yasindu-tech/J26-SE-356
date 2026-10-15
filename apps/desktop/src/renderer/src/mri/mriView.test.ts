import { describe, expect, it } from "vitest";
import type { MriScore } from "./mriTypes";
import { axisPercent, formatSigned, isFinal, readScore, statusLabel, stepState, vectorJson } from "./mriView";

function score(value: number, low: number, high: number): MriScore {
  return {
    kind: "log_odds",
    value,
    interval_95: [low, high],
    interval_method: "spread over 50 fold copies",
    calibrated: false,
    n_training: 37,
    meaning: "Raw score.",
  };
}

describe("readScore", () => {
  it("says when the interval crosses the boundary", () => {
    const r = readScore(score(0.4, -0.3, 1.1));
    expect(r.lean).toBe("towards PD");
    expect(r.intervalCrossesBoundary).toBe(true);
    expect(r.sentence).toContain("crosses 0");
  });

  it("says when every copy agrees", () => {
    const r = readScore(score(-1.2, -2.0, -0.4));
    expect(r.lean).toBe("towards control");
    expect(r.intervalCrossesBoundary).toBe(false);
    expect(r.sentence).toContain("below 0");
  });
});

describe("steps and status", () => {
  it("marks done, current and waiting steps while running", () => {
    expect(stepState("convert", "running", ["convert"], "t1")).toBe("done");
    expect(stepState("t1", "running", ["convert"], "t1")).toBe("current");
    expect(stepState("dti", "running", ["convert"], "t1")).toBe("waiting");
  });

  it("marks unfinished steps as stopped once the analysis ended", () => {
    expect(stepState("dti", "abstained", ["convert", "t1"], null)).toBe("stopped");
    expect(isFinal("abstained")).toBe(true);
    expect(isFinal("running")).toBe(false);
  });

  it("uses screening wording, not diagnostic wording", () => {
    const words = (["queued", "running", "scored", "abstained", "failed"] as const)
      .map((s) => statusLabel(s).text.toLowerCase())
      .join(" ");
    for (const banned of ["positive", "negative", "diagnos", "cleared"]) {
      expect(words).not.toContain(banned);
    }
  });
});

describe("formatting", () => {
  it("places values on the axis and clamps the ends", () => {
    expect(axisPercent(0)).toBe(50);
    expect(axisPercent(10)).toBe(100);
    expect(axisPercent(-10)).toBe(0);
  });

  it("formats signed numbers and the vector", () => {
    expect(formatSigned(0.456)).toBe("+0.46");
    expect(formatSigned(-1)).toBe("−1.00");
    expect(JSON.parse(vectorJson(["a", "b"], [1, 2]))).toEqual({ a: 1, b: 2 });
  });
});
