import { describe, expect, it } from "vitest";
import type { GaitResult } from "./gaitTypes";
import {
  MIN_MARGIN,
  RELIABILITY_LINE,
  checkRows,
  classBars,
  driverRows,
  isScored,
  auditLine,
  keepInMind,
  notices,
  timelineBar,
  videoProblem,
  videoSummary,
  GAIT_MAX_UPLOAD_MB,
  plainGaitReading,
  shapBars,
  visibleWarnings,
  walkLine,
} from "./gaitView";
import {
  MOCK_GAIT_NOT_SCORED,
  MOCK_GAIT_RESULTS,
  MOCK_GAIT_SCORED_CLEAR,
  MOCK_GAIT_SCORED_MIXED,
} from "./mockGaitResult";

function withProbs(probabilities: number[], predicted: number): GaitResult {
  return { ...MOCK_GAIT_SCORED_CLEAR, probabilities, predicted_class: predicted };
}

describe("isScored", () => {
  it("is true for a scored record and false for one that was not scored", () => {
    expect(isScored(MOCK_GAIT_SCORED_CLEAR)).toBe(true);
    expect(isScored(MOCK_GAIT_SCORED_MIXED)).toBe(true);
    expect(isScored(MOCK_GAIT_NOT_SCORED)).toBe(false);
  });
});

describe("plainGaitReading", () => {
  it("names the level the model puts clearly ahead", () => {
    const r = plainGaitReading(MOCK_GAIT_SCORED_CLEAR);
    expect(r.kind).toBe("clear");
    expect(r.headline).toBe("Walk pattern closest to: moderate-to-severe gait impairment");
  });

  it("says there is no clear pattern when the top two scores are close", () => {
    const r = plainGaitReading(MOCK_GAIT_SCORED_MIXED);
    expect(r.kind).toBe("none");
    expect(r.headline).toBe("No clear pattern");
  });

  it("uses the margin limit exactly: just under is unclear, just over is clear", () => {
    expect(plainGaitReading(withProbs([0.5, 0.5 - MIN_MARGIN + 0.01, 0.0], 0)).kind).toBe("none");
    expect(plainGaitReading(withProbs([0.5, 0.5 - MIN_MARGIN - 0.01, 0.0], 0)).kind).toBe("clear");
  });

  it("never reads a lower level as a clearance", () => {
    const r = plainGaitReading(withProbs([0.9, 0.08, 0.02], 0));
    expect(r.headline).toBe("Walk pattern closest to: no gait impairment");
    expect(r.detail).toContain("never a clearance");
  });

  it("uses no percentage, priority or diagnosis wording in any reading", () => {
    for (const p of [[0.9, 0.08, 0.02], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9], [0.4, 0.2, 0.4]]) {
      const idx = p.indexOf(Math.max(...p));
      const r = plainGaitReading(withProbs(p, idx));
      const text = `${r.headline} ${r.detail}`;
      expect(text).not.toMatch(/%|high priority|moderate priority|low priority|positive|negative/i);
    }
  });
});

describe("walkLine", () => {
  it("shows one decimal under 10 s and whole seconds above", () => {
    expect(walkLine(MOCK_GAIT_SCORED_MIXED)).toBe(
      "Scored 4.5 s of walking, from 1.3 s to 5.8 s of the video.",
    );
    expect(walkLine(MOCK_GAIT_SCORED_CLEAR)).toBe(
      "Scored 17 s of walking, from 2.6 s to 19.8 s of the video.",
    );
  });
});

describe("classBars", () => {
  it("draws the three scores to scale and marks only the model's pick", () => {
    const bars = classBars(MOCK_GAIT_SCORED_CLEAR);
    expect(bars.map((b) => b.label)).toEqual(["No impairment", "Mild", "Moderate to severe"]);
    expect(bars.map((b) => b.widthPercent)).toEqual([1, 7, 92]);
    expect(bars.map((b) => b.isPick)).toEqual([false, false, true]);
    expect(bars.map((b) => b.rating)).toEqual([
      "UPDRS gait rating 0",
      "UPDRS gait rating 1",
      "UPDRS gait rating 2 to 3",
    ]);
  });

  it("keeps a tiny score visible as a sliver, and never exceeds the full bar", () => {
    const bars = classBars(withProbs([0.0001, 0.0, 1.5], 2));
    expect(bars.map((b) => b.widthPercent)).toEqual([1, 1, 100]);
  });
});

describe("driverRows", () => {
  it("returns the model's own sentences, three at most, with a direction tag", () => {
    const rows = driverRows(MOCK_GAIT_SCORED_CLEAR);
    expect(rows).toHaveLength(3);
    expect(rows[0]?.tag).toBe("Strongly towards");
    expect(rows[0]?.towards).toBe(true);
    expect(rows[2]?.tag).toBe("Moderately away");
    expect(rows[2]?.towards).toBe(false);
    expect(rows[0]?.text).toBe(MOCK_GAIT_SCORED_CLEAR.top_features[0]?.sentence);
  });

  it("is empty for a record that was not scored", () => {
    expect(driverRows(MOCK_GAIT_NOT_SCORED)).toEqual([]);
  });
});

describe("shapBars", () => {
  it("draws the top three inputs to scale, longest at 100, with direction and plain names", () => {
    const bars = shapBars(MOCK_GAIT_SCORED_CLEAR);
    expect(bars.map((b) => b.label)).toEqual(["Hip swing range", "Knee bend range", "Step length"]);
    expect(bars.map((b) => b.widthPercent)).toEqual([100, 78, 21]);
    expect(bars.map((b) => b.towards)).toEqual([true, true, false]);
  });

  it("keeps a very small contribution visible and handles all-zero contributions", () => {
    const tiny: GaitResult = {
      ...MOCK_GAIT_SCORED_CLEAR,
      top_features: MOCK_GAIT_SCORED_CLEAR.top_features.map((f, i) => ({ ...f, shap: i === 0 ? 1 : 0.001 })),
    };
    expect(shapBars(tiny).map((b) => b.widthPercent)).toEqual([100, 3, 3]);
    const zero: GaitResult = {
      ...MOCK_GAIT_SCORED_CLEAR,
      top_features: MOCK_GAIT_SCORED_CLEAR.top_features.map((f) => ({ ...f, shap: 0 })),
    };
    expect(shapBars(zero).map((b) => b.widthPercent)).toEqual([3, 3, 3]);
  });

  it("is empty for a record that was not scored", () => {
    expect(shapBars(MOCK_GAIT_NOT_SCORED)).toEqual([]);
  });
});

describe("checkRows", () => {
  it("lists the five checks in a fixed order with what was found and what is needed", () => {
    const rows = checkRows(MOCK_GAIT_SCORED_CLEAR);
    expect(rows.map((r) => r.key)).toEqual([
      "frame_rate",
      "leg_visibility",
      "filled_share",
      "facing",
      "strides",
    ]);
    expect(rows[0]).toMatchObject({ value: "50 frames per second", need: "needs 20 to 60", ok: true });
    expect(rows[1]).toMatchObject({ value: "97% of frames", need: "needs at least 80%" });
    expect(rows[2]).toMatchObject({ value: "2%", need: "needs at most 20%" });
    expect(rows[4]).toMatchObject({ value: "18 in total, on both feet", need: "needs at least 3" });
  });

  it("flags the check that was not met", () => {
    const failed = checkRows(MOCK_GAIT_NOT_SCORED).filter((r) => !r.ok);
    expect(failed.map((r) => r.key)).toEqual(["strides"]);
    expect(failed[0]?.value).toBe("2 in total, on both feet");
  });

  it("leaves out a check the pipeline did not run", () => {
    const gate = MOCK_GAIT_NOT_SCORED.gate;
    if (!gate) throw new Error("the mock has a gate");
    const checks = { ...gate.checks, frame_rate: { value: 240, limit: 60, ok: false } };
    delete checks.strides; // outside 20 to 60 fps the pipeline does not count strides
    const slowMotion: GaitResult = { ...MOCK_GAIT_NOT_SCORED, gate: { ...gate, checks } };
    const rows = checkRows(slowMotion);
    expect(rows.map((r) => r.key)).toEqual(["frame_rate", "leg_visibility", "filled_share", "facing"]);
    expect(rows[0]).toMatchObject({ value: "240 frames per second", ok: false });
  });

  it("is empty when the analysis stopped before the checks", () => {
    expect(checkRows({ ...MOCK_GAIT_NOT_SCORED, gate: null })).toEqual([]);
  });
});

describe("visibleWarnings", () => {
  it("leaves out the one-person warnings, which Keep in mind covers", () => {
    expect(visibleWarnings(MOCK_GAIT_SCORED_CLEAR)).toEqual([]);
    expect(visibleWarnings(MOCK_GAIT_NOT_SCORED)).toEqual([]);
  });

  it("keeps the left and right stride warning", () => {
    const warnings = visibleWarnings(MOCK_GAIT_SCORED_MIXED);
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toContain("left and right stride times differ");
  });
});

describe("keepInMind", () => {
  it("says the model grades walking and not Parkinson's, and that the bars are not chances", () => {
    const text = keepInMind(MOCK_GAIT_SCORED_CLEAR).join(" ");
    expect(text).toContain("does not say whether they have Parkinson");
    expect(text).toContain("not chances");
  });

  it("uses the confirmed wording when the one-person box was ticked, and the assumed wording otherwise", () => {
    expect(keepInMind(MOCK_GAIT_SCORED_CLEAR).join(" ")).toContain("You confirmed one person");
    expect(keepInMind(MOCK_GAIT_SCORED_MIXED).join(" ")).toContain("One person walking is assumed");
  });

  it("leaves out the bars note for a video that was not scored", () => {
    const text = keepInMind(MOCK_GAIT_NOT_SCORED).join(" ");
    expect(text).not.toContain("bars");
    expect(text).toContain("One person walking is assumed");
  });
});

describe("notices", () => {
  it("is empty for a clean, long walk", () => {
    expect(notices(MOCK_GAIT_SCORED_CLEAR)).toEqual([]);
    expect(notices(MOCK_GAIT_NOT_SCORED)).toEqual([]);
  });

  it("gives the stride warning and a short-walk note, each with a title", () => {
    const items = notices(MOCK_GAIT_SCORED_MIXED);
    expect(items.map((n) => n.title)).toEqual(["Check the video", "Short walk"]);
    expect(items[0]?.text).toContain("left and right stride times differ");
    expect(items[1]?.text).toBe("Only 4.5 s of walking was scored, so the reading rests on few steps.");
  });
});

describe("auditLine", () => {
  it("names the model, the video and who confirmed one person", () => {
    expect(auditLine(MOCK_GAIT_SCORED_CLEAR)).toBe(
      "Model gait-xgb-ed1c5ad4 \u00b7 video w-e40cda55 \u00b7 one person confirmed by you",
    );
    expect(auditLine(MOCK_GAIT_SCORED_MIXED)).toContain("one person assumed");
  });
});

describe("timelineBar", () => {
  it("places the scored stretch inside the video", () => {
    const bar = timelineBar(MOCK_GAIT_SCORED_CLEAR, 21);
    expect(bar).toMatchObject({ leftPercent: 12.5, endLabel: "21 s" });
    expect(bar?.widthPercent).toBeCloseTo(81.8, 1);
    expect(bar?.text).toContain("from 2.6 s to 19.8 s");
    expect(bar?.text).not.toContain("short");
  });

  it("says a walk under five seconds is short", () => {
    expect(timelineBar(MOCK_GAIT_SCORED_MIXED, 9)?.text).toContain("That is short.");
  });

  it("is null when the video length is not known", () => {
    expect(timelineBar(MOCK_GAIT_SCORED_CLEAR, null)).toBeNull();
    expect(timelineBar(MOCK_GAIT_SCORED_CLEAR, 0)).toBeNull();
  });

  it("never runs past the end of the bar", () => {
    const bar = timelineBar(MOCK_GAIT_SCORED_CLEAR, 10); // the video is shorter than the record says
    expect((bar?.leftPercent ?? 0) + (bar?.widthPercent ?? 0)).toBeLessThanOrEqual(100.1);
  });
});

describe("videoProblem", () => {
  const mb = 1024 * 1024;

  it("accepts the four video types in any case, up to the limit", () => {
    for (const name of ["a.mp4", "a.MOV", "a.m4v", "a.AVI"]) {
      expect(videoProblem(name, 10 * mb)).toBeNull();
    }
    expect(videoProblem("a.mp4", GAIT_MAX_UPLOAD_MB * mb)).toBeNull();
  });

  it("rejects another file type with the list of types", () => {
    expect(videoProblem("notes.pdf", mb)).toEqual({
      title: "This file type cannot be read",
      text: "Choose a video: MP4, MOV, M4V or AVI.",
    });
  });

  it("rejects a file over the limit and says its size", () => {
    const problem = videoProblem("big-walk.mov", 1.2 * 1024 * mb);
    expect(problem?.title).toBe("This video is too large");
    expect(problem?.text).toContain("big-walk.mov is 1.2 GB. The limit is 500 MB.");
    expect(videoProblem("a.mp4", GAIT_MAX_UPLOAD_MB * mb + 1)?.title).toBe("This video is too large");
  });
});

describe("videoSummary", () => {
  it("shows size, length and type, leaving out a length that is not known", () => {
    expect(videoSummary("walk-side-view.mp4", 86.4 * 1024 * 1024, 21)).toBe("86.4 MB \u00b7 21 s \u00b7 MP4");
    expect(videoSummary("walk.avi", 5 * 1024 * 1024, null)).toBe("5.0 MB \u00b7 AVI");
  });
});

describe("wording", () => {
  it("states the model's test result without a number that reads as a score", () => {
    expect(RELIABILITY_LINE).toContain("about half the time");
    expect(RELIABILITY_LINE).toContain("not been tested on phone video");
  });

  it("covers every mock record", () => {
    expect(MOCK_GAIT_RESULTS.map((m) => m.key)).toEqual(["clear", "mixed", "none"]);
  });
});
