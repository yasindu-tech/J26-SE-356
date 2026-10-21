import { describe, expect, it } from "vitest";
import { driverText, eddyMinutes, partState, plainReading, safeImage } from "./mriPlain";

describe("plainReading", () => {
  it("reads an interval that crosses 0 as no clear lean", () => {
    expect(plainReading({ interval_95: [-0.4, 0.9] }).lean).toBe("none");
    expect(plainReading({ interval_95: [0, 1] }).lean).toBe("none");
  });

  it("reads an interval wholly above or below 0 as a lean", () => {
    expect(plainReading({ interval_95: [0.2, 1.4] }).lean).toBe("higher");
    expect(plainReading({ interval_95: [-1.4, -0.2] }).lean).toBe("lower");
  });

  it("states the result without advice, diagnosis or clearance wording", () => {
    for (const interval of [
      [-1, 1],
      [1, 2],
      [-2, -1],
    ] as [number, number][]) {
      const r = plainReading({ interval_95: interval });
      const text = `${r.headline} ${r.detail}`.toLowerCase();
      expect(text).not.toMatch(/diagnos|clear(ed|ance)|rule[sd]? out|caution|weigh|should|refer/);
    }
  });
});

describe("partState", () => {
  const result = (available: boolean, found: boolean) => ({
    availability: { t1: available },
    inputs: { t1: { found } },
  });

  it("separates used, found-but-not-used and missing", () => {
    expect(partState(result(true, true), "t1")).toBe("used");
    expect(partState(result(false, true), "t1")).toBe("not-used");
    expect(partState(result(false, false), "t1")).toBe("missing");
    expect(partState({ availability: {}, inputs: {} }, "dti")).toBe("missing");
  });
});

describe("safeImage", () => {
  it("only lets data: image URLs through", () => {
    expect(safeImage("data:image/png;base64,AAAA")).toBe("data:image/png;base64,AAAA");
    expect(safeImage("https://example.org/x.png")).toBeNull();
    expect(safeImage("javascript:alert(1)")).toBeNull();
    expect(safeImage(null)).toBeNull();
    expect(safeImage(undefined)).toBeNull();
  });
});

describe("driverText", () => {
  it("says which way a measurement pushed the reading, never a cause", () => {
    expect(driverText({ label: "Mean FA, left CST", contribution: 0.3 })).toBe(
      "Mean FA, left CST pushed the reading higher.",
    );
    expect(driverText({ label: "Putamen volume", contribution: -0.1 })).toContain("lower");
  });
});

describe("eddyMinutes", () => {
  it("rounds seconds to minutes and keeps a missing time missing", () => {
    expect(eddyMinutes(5640)).toBe(94);
    expect(eddyMinutes(null)).toBeNull();
    expect(eddyMinutes(undefined)).toBeNull();
    expect(eddyMinutes(Number.NaN)).toBeNull();
  });
});
