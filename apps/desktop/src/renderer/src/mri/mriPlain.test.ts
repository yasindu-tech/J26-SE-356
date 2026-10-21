import { describe, expect, it } from "vitest";
import { driverText, eddyMinutes, partState, plainReading, safeImage } from "./mriPlain";

describe("plainReading", () => {
  const track = {
    source: "run",
    leans_pd: { n: 16, correct: 10 },
    leans_control: { n: 21, correct: 13 },
  };

  it("always gives a direction from the score", () => {
    expect(plainReading({ value: 1.54, interval_95: [-1.16, 2.59] }).lean).toBe("higher");
    expect(plainReading({ value: -0.4, interval_95: [-1.2, 0.3] }).lean).toBe("lower");
    expect(plainReading({ value: 0, interval_95: [-1, 1] }).lean).toBe("none");
  });

  it("is consistent only when the whole interval is on the score's side", () => {
    expect(plainReading({ value: 1, interval_95: [0.2, 1.4] }).strength).toBe("consistent");
    expect(plainReading({ value: -1, interval_95: [-1.4, -0.2] }).strength).toBe("consistent");
  });

  it("uses the 80% copy agreement fixed in advance for a mostly consistent lean", () => {
    const base = { value: 1.54, interval_95: [-1.16, 2.59] as [number, number], n_copies: 50 };
    expect(plainReading({ ...base, copies_above_zero: 40 }).strength).toBe("moderate");
    expect(plainReading({ ...base, copies_above_zero: 39 }).strength).toBe("weak");
    // a control-side score counts the copies below zero
    expect(
      plainReading({ value: -1, interval_95: [-2, 0.5], n_copies: 50, copies_above_zero: 8 }).strength,
    ).toBe("moderate");
  });

  it("falls back to inconsistent when an older result has no copy counts", () => {
    expect(plainReading({ value: 1.54, interval_95: [-1.16, 2.59] }).strength).toBe("weak");
  });

  it("states the agreement and the track record for the side it leans to", () => {
    const r = plainReading(
      { value: 1.54, interval_95: [-1.16, 2.59], n_copies: 50, copies_above_zero: 44 },
      track,
    );
    expect(r.agreement).toBe("44 of 50 model copies agree with this side.");
    expect(r.trackRecord).toContain("10 of 16 people");
    const c = plainReading({ value: -1, interval_95: [-2, -0.1] }, track);
    expect(c.trackRecord).toContain("13 of 21 people");
  });

  it("uses no diagnosis, clearance, advice or risk wording", () => {
    for (const s of [
      { value: 1, interval_95: [0.5, 2] },
      { value: 1, interval_95: [-1, 2], n_copies: 50, copies_above_zero: 45 },
      { value: -1, interval_95: [-2, 1] },
    ] as const) {
      const r = plainReading({ ...s, interval_95: [...s.interval_95] as [number, number] }, track);
      const text = [r.headline, r.detail, r.agreement, r.trackRecord].join(" ").toLowerCase();
      expect(text).not.toMatch(/diagnos|clear(ed|ance)|rule[sd]? out|caution|should|refer|risk|high|low/);
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
