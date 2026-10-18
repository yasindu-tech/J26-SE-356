import { describe, expect, it } from "vitest";
import { initials } from "../components/initials";
import type { MriAnalysisStatus, MriDriver, MriResult, MriScore } from "./mriTypes";
import {
  FAST_EDDY_NOTICE,
  axisPercent,
  caveats,
  driverBars,
  driverSentence,
  eddyProfileOf,
  formatClock,
  formatElapsed,
  formatSigned,
  greeting,
  homeSummary,
  FAST_EDDY_WARNING,
  isFinal,
  isValidSubjectLabel,
  PARTS,
  missingRows,
  modelSetLabel,
  parseBackendTime,
  readScore,
  scoreBand,
  statusLabel,
  stepState,
  updateActivity,
  vectorJson,
} from "./mriView";

function withDti(eddy?: { profile: "full" | "fast" | null; seconds: number | null }) {
  return { inputs: { dti: { found: true, ...(eddy ? { eddy } : {}) } } } as Pick<
    MriResult,
    "inputs"
  >;
}

describe("eddyProfileOf", () => {
  it("reports the setting the DTI features were made with", () => {
    expect(eddyProfileOf(withDti({ profile: "fast", seconds: 1500 }))).toBe("fast");
    expect(eddyProfileOf(withDti({ profile: "full", seconds: 5000 }))).toBe("full");
  });

  it("is null when there was no DTI, or an older result did not record it", () => {
    expect(eddyProfileOf({ inputs: {} })).toBeNull();
    expect(eddyProfileOf(withDti())).toBeNull();
  });

  it("warns before a run, too, that the fast setting is not validated", () => {
    expect(FAST_EDDY_WARNING).toContain("not been checked");
  });

  it("has a notice that says the fast setting is not validated", () => {
    expect(FAST_EDDY_NOTICE).toContain("not been checked");
  });
});

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

// ------------------------------------------------------------- redesign helpers

describe("time", () => {
  it("reads the backend's zone-less UTC time as UTC", () => {
    expect(parseBackendTime("2026-10-18T12:00:00").toISOString()).toBe("2026-10-18T12:00:00.000Z");
    expect(parseBackendTime("2026-10-18T12:00:00Z").toISOString()).toBe("2026-10-18T12:00:00.000Z");
    expect(parseBackendTime("2026-10-18T12:00:00+05:30").toISOString()).toBe(
      "2026-10-18T06:30:00.000Z",
    );
  });

  it("formats elapsed time and never goes negative", () => {
    expect(formatElapsed(134)).toBe("02:14");
    expect(formatElapsed(3725)).toBe("1:02:05");
    expect(formatElapsed(-5)).toBe("00:00");
    expect(formatClock(new Date(2026, 9, 18, 9, 5, 7))).toBe("09:05:07");
  });
});

describe("updateActivity", () => {
  const status = {
    created_at: "2026-10-18T12:00:00",
    done: ["convert"],
    current_step: "t1",
    note: "T1: removing the skull",
  };

  it("lists what was reported, once", () => {
    const first = updateActivity([], status, new Date("2026-10-18T12:00:20Z"));
    expect(first.map((e) => e.text)).toEqual([
      "Analysis requested",
      "Convert DICOM: finished",
      "T1: removing the skull",
    ]);
    const again = updateActivity(first, status, new Date("2026-10-18T12:00:30Z"));
    expect(again).toHaveLength(3); // the same reports are not listed twice
  });

  it("adds a line when the note changes or a step finishes", () => {
    const first = updateActivity([], status, new Date("2026-10-18T12:00:20Z"));
    const next = updateActivity(
      first,
      { ...status, done: ["convert", "t1"], current_step: "dti", note: "DTI: correcting motion" },
      new Date("2026-10-18T12:02:00Z"),
    );
    expect(next.map((e) => e.text).slice(3)).toEqual(["T1 pipeline: finished", "DTI: correcting motion"]);
  });
});

describe("scoreBand", () => {
  it("places the zero line, score and interval on the axis", () => {
    const band = scoreBand(score(0, -2, 2));
    expect(band.zero).toBe(50);
    expect(band.point).toBe(50);
    expect(band.low).toBe(25);
    expect(band.high).toBe(75);
    expect(band.width).toBe(50);
  });

  it("keeps a score beyond the axis on the axis", () => {
    expect(scoreBand(score(9, 8, 10)).point).toBe(100);
  });
});

describe("driver bars", () => {
  const drivers: MriDriver[] = [
    { feature: "a", label: "Left putamen", value: 1, contribution: 0.4, direction: "towards PD", text: "" },
    { feature: "b", label: "Right caudate", value: 1, contribution: -0.2, direction: "towards control", text: "" },
  ];

  it("puts higher on one side and lower on the other, sized to the largest", () => {
    const bars = driverBars(drivers);
    expect(bars[0]).toMatchObject({ side: "higher", widthPct: 100 });
    expect(bars[1]).toMatchObject({ side: "lower", widthPct: 50 });
  });

  it("states a contribution, never a cause", () => {
    const text = driverSentence(drivers[0]);
    expect(text).toContain("contributed +0.40 to the score");
    expect(text.toLowerCase()).not.toMatch(/caus|because|due to/);
  });
});

describe("what is missing", () => {
  const input = (found: boolean, status?: string) => ({ found, ...(status ? { status } : {}) });

  it("names a part that was not in the upload and what the result then is", () => {
    const rows = missingRows({
      status: "scored",
      model_set: "t1",
      availability: { t1: true, dti: false },
      inputs: { t1: input(true, "ok"), dti: input(false) },
    });
    expect(rows).toHaveLength(1);
    expect(rows[0].key).toBe("dti");
    expect(rows[0].detail).toBe("Not in the upload");
    expect(rows[0].effect).toContain("T1 only model");
    expect(rows[0].action).toBe("Upload a study that includes it");
  });

  it("says a failed part was found but did not pass, and offers a re-check", () => {
    const [row] = missingRows({
      status: "scored",
      model_set: "dti",
      availability: { t1: false, dti: true },
      inputs: { t1: input(true, "failed"), dti: input(true, "ok") },
    });
    expect(row.detail).toContain("did not pass");
    expect(row.action).toBe("Check the scan and upload again");
  });

  it("lists both parts when nothing was scored, and never counts them as zero", () => {
    const rows = missingRows({
      status: "abstained",
      model_set: null,
      availability: { t1: false, dti: false },
      inputs: {},
    });
    expect(rows.map((r) => r.key)).toEqual(["t1", "dti"]);
    expect(rows[0].effect).toContain("No MRI score was produced");
    expect(rows[0].effect).toContain("nothing is counted as zero");
  });

  it("lists nothing when both parts were used", () => {
    expect(
      missingRows({
        status: "scored",
        model_set: "t1dti",
        availability: { t1: true, dti: true },
        inputs: {},
      }),
    ).toEqual([]);
    expect(modelSetLabel("t1dti")).toBe("T1 + DTI");
  });
});

describe("caveats", () => {
  it("reports the training population, the limits and what is outside the range", () => {
    const c = caveats({
      model_card: {
        training_people: { total: 37, pd: 21, control: 16 },
        known_limits: ["Small sample."],
      },
      feature_vector: {
        names: ["a", "b"],
        labels: ["Left putamen", "Right caudate"],
        raw: [1, 2],
        harmonised: [1, 2],
        scanner_batch: {},
        outside_training_range: ["b", "zz"],
      },
    });
    expect(c.population).toContain("37 people");
    expect(c.population).toContain("21 with Parkinson");
    expect(c.limits).toEqual(["Small sample."]);
    expect(c.outsideRange).toEqual([
      { name: "b", label: "Right caudate" },
      { name: "zz", label: "zz" },
    ]);
  });

  it("copes with a result that has no model card or vector", () => {
    expect(caveats({ model_card: null, feature_vector: null })).toEqual({
      population: null,
      limits: [],
      outsideRange: [],
    });
  });
});

describe("language rules", () => {
  it("never reads a low score as a clearance, nor a high one as a diagnosis", () => {
    const low = readScore(score(-2, -3, -1)).sentence;
    const high = readScore(score(2, 1, 3)).sentence;
    expect(low).toContain("never a clearance");
    expect(high).toContain("not a diagnosis");
    for (const s of [low, high]) expect(s.toLowerCase()).not.toMatch(/cleared|positive|negative/);
  });
});

describe("initials", () => {
  it("makes two letters from a username", () => {
    expect(initials("j.fernando")).toBe("JF");
    expect(initials("yasindu")).toBe("YA");
    expect(initials("Dr J Fernando")).toBe("DJ");
    expect(initials("  ")).toBe("?");
  });
});

describe("home", () => {
  const item = (status: MriAnalysisStatus["status"]) =>
    ({ id: status, status }) as unknown as MriAnalysisStatus;

  it("greets by the hour", () => {
    expect(greeting(new Date(2026, 9, 18, 9))).toBe("Good morning");
    expect(greeting(new Date(2026, 9, 18, 14))).toBe("Good afternoon");
    expect(greeting(new Date(2026, 9, 18, 20))).toBe("Good evening");
  });

  it("counts what is in progress and what finished, and counts a non-score as not scored", () => {
    const s = homeSummary([item("queued"), item("running"), item("scored"), item("abstained"), item("failed")]);
    expect(s.counted).toBe(5);
    expect(s.inProgress.map((i) => i.id)).toEqual(["queued", "running"]);
    expect([s.scored, s.notScored, s.failed]).toEqual([1, 1, 1]);
    expect(homeSummary([]).counted).toBe(0);
  });
});

describe("assessment parts", () => {
  it("accepts a pseudonymous label and refuses anything that could be a name", () => {
    expect(isValidSubjectLabel("P-8F31A2")).toBe(true);
    expect(isValidSubjectLabel("Jane Doe")).toBe(false);
    expect(isValidSubjectLabel("")).toBe(false);
    expect(isValidSubjectLabel("a".repeat(33))).toBe(false);
  });

  it("lists the four parts and marks only MRI as available in this build", () => {
    expect(PARTS.map((p) => p.key)).toEqual(["gait", "tapping", "voice", "mri"]);
    expect(PARTS.filter((p) => p.available).map((p) => p.key)).toEqual(["mri"]);
  });
});
