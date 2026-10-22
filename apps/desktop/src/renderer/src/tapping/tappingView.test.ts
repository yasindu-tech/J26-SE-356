import { readFileSync, readdirSync } from "fs";
import { join, resolve } from "path";
import { describe, expect, it } from "vitest";
import { failedFixture, scoredFixture } from "../../../shared/tappingResult.fixture";
import type { TappingRunStatus } from "./tappingTypes";
import {
  CHIP_AVAILABLE,
  CHIP_CHOSEN,
  CONTRIBUTION_LINE,
  NOT_COMBINED,
  SCORE_LINE,
  VIDEO_STAYS_HERE,
  auditLine,
  bandText,
  crossValidationText,
  curveChart,
  directionOf,
  directionWord,
  extensionOf,
  fixed,
  outcomeOf,
  previewAlt,
  recordingRows,
  scaleDescription,
  scaleGeometry,
  scoreHeadline,
  shapBars,
  shapDescription,
  signalChart,
  stripState,
  thresholdText,
  videoProblem,
  videoSummary,
} from "./tappingView";

// ---------------------------------------------------------------------------- the video

describe("the chosen video", () => {
  it("accepts mp4, mov and m4v in any case", () => {
    for (const name of ["a.mp4", "A.MOV", "b.M4V"]) expect(videoProblem(name, 1000)).toBeNull();
  });

  it("refuses other types, empty files and files over 200 MB, never naming the file", () => {
    const refused = [
      videoProblem("patient_jane.avi", 1000),
      videoProblem("patient_jane", 1000),
      videoProblem("patient_jane.mp4", 0),
      videoProblem("patient_jane.mp4", 200 * 1024 * 1024 + 1),
    ];
    for (const problem of refused) {
      expect(problem).not.toBeNull();
      expect(JSON.stringify(problem)).not.toMatch(/jane/i);
    }
    expect(videoProblem("a.mp4", 200 * 1024 * 1024)).toBeNull();
  });

  it("shows size, length and hashed ref, never a file name", () => {
    expect(videoSummary(12.34 * 1024 * 1024, 18.2, "f0e37b01")).toBe("12.3 MB · 18 s · clip f0e37b01");
    expect(videoSummary(1024 * 1024, null, null)).toBe("1.0 MB");
    expect(videoSummary(1024 * 1024, Number.NaN, null)).toBe("1.0 MB");
  });

  it("reads the extension in lower case", () => {
    expect(extensionOf("Clip.MP4")).toBe(".mp4");
    expect(extensionOf("noext")).toBe("");
  });
});

// ---------------------------------------------------------------------------- outcomes

const run = (over: Partial<TappingRunStatus>): TappingRunStatus => ({
  id: "a1",
  state: "scored",
  elapsedSeconds: 12,
  result: scoredFixture(),
  error: null,
  ...over,
});

describe("outcomes", () => {
  it("maps a scored run, a stored failed record, a failed run and a run without a result", () => {
    expect(outcomeOf(run({}))).toMatchObject({ kind: "scored" });
    expect(outcomeOf(run({ state: "abstained", result: failedFixture() }))).toEqual({
      kind: "failed",
      message: failedFixture().reason,
      ref: "f0e37b01",
    });
    expect(outcomeOf(run({ state: "failed", result: null, error: "the analysis stopped" }))).toEqual({
      kind: "failed",
      message: "the analysis stopped",
      ref: null,
    });
    expect(outcomeOf(run({ state: "abstained", result: null }))).toEqual({
      kind: "failed",
      message: "The video could not be scored.",
      ref: null,
    });
  });

  it("a failed status is never shown as a score", () => {
    for (const bad of [
      run({ state: "abstained", result: failedFixture() }),
      run({ state: "failed", result: null, error: "x" }),
      run({ state: "scored", result: failedFixture() }), // contradictory: still not a score
    ]) {
      const outcome = outcomeOf(bad);
      expect(outcome.kind).toBe("failed");
      expect("result" in outcome).toBe(false);
      expect(stripState(outcome)).toBe("failed");
    }
    expect(stripState(null)).toBe("running");
    expect(stripState(outcomeOf(run({})))).toBe("scored");
  });
});

// ---------------------------------------------------------------------------- numbers and texts

describe("score texts", () => {
  const r = scoredFixture();

  it("rounds the score to two places", () => {
    expect(scoreHeadline(r)).toBe("Model score 0.09");
  });

  it("shows the band as a range of models, never as a confidence interval", () => {
    const text = bandText(r);
    expect(text).toBe("Range 0.02 to 0.10 (5-95% of 25 cross-validation models).");
    expect(text).not.toMatch(/confidence|interval|\bCI\b/i);
    expect(scaleDescription(r)).not.toMatch(/confidence|\bCI\b/i);
  });

  it("states which side of the fixed threshold the score is on", () => {
    expect(thresholdText(r)).toMatch(/^Below the fixed study threshold of 0\.5\./);
    r.score.above_threshold = true;
    expect(thresholdText(r)).toMatch(/^At or above the fixed study threshold of 0\.5\./);
    r.score.above_threshold = false;
    expect(thresholdText(r)).toContain("not a clinical cut-off");
  });

  it("gives the cross-validation line only when all three figures exist", () => {
    expect(crossValidationText(r)).toBe(
      "In cross-validation on the research clips this model reached AUROC 0.87 (95% interval 0.81 to 0.92) for the primary study target.",
    );
    expect(crossValidationText({ ...r, target: "ge1" })).toContain("the secondary study target");
    expect(crossValidationText({ ...r, score: { ...r.score, cv_ci_low: null } })).toBeNull();
  });

  it("puts the marker, band and threshold on a 0 to 100 axis and clamps", () => {
    expect(scaleGeometry(r)).toEqual({
      valuePercent: expect.closeTo(8.9876, 3),
      lowPercent: expect.closeTo(1.5538, 3),
      highPercent: expect.closeTo(10.247, 3),
      thresholdPercent: 50,
    });
    expect(scaleGeometry({ ...r, score: { ...r.score, value: 1.4 } }).valuePercent).toBe(100);
  });

  it("writes a missing number as 'not available', never as zero", () => {
    expect(fixed(null, 1, " s")).toBe("not available");
    expect(fixed(Number.NaN, 1)).toBe("not available");
    expect(fixed(0, 1, " s")).toBe("0.0 s");
    const rec = recordingRows({ ...r, recording: { duration_s: null, fps: null, hand_found_pct: null, n_taps: 0 } });
    expect(rec.map((x) => x.value)).toEqual(["f0e37b01", "not available", "not available", "not available", "0"]);
  });

  it("lists the recording numbers", () => {
    expect(recordingRows(r)).toEqual([
      { label: "Clip", value: "f0e37b01" },
      { label: "Length", value: "19.5 s" },
      { label: "Frame rate", value: "60 fps" },
      { label: "Hand found in", value: "100% of frames" },
      { label: "Taps found", value: "29" },
    ]);
    expect(auditLine(r)).toBe("clip f0e37b01 · demo model 1c623df9 · fold models ae519116");
    expect(previewAlt(3.14)).toContain("at 3.1 s");
    expect(previewAlt(null)).not.toContain("null");
  });
});

// ---------------------------------------------------------------------------- SHAP bars

describe("shapBars", () => {
  it("draws a positive contribution to the right, a negative one to the left, by sign", () => {
    const bars = shapBars(scoredFixture());
    const [first, , , fourth] = bars;
    expect(first).toMatchObject({ direction: "down", width: 50, x: 0 });
    expect(fourth?.direction).toBe("up");
    expect(fourth?.x).toBe(50);
    expect(fourth?.width).toBeCloseTo((0.13 / 0.67) * 50);
    expect(directionOf(0.01)).toBe("up");
    expect(directionOf(-0.01)).toBe("down");
  });

  it("gives a zero or a missing contribution no bar, and no direction", () => {
    const r = scoredFixture();
    r.contributions[1]!.shap = null;
    const bars = shapBars(r);
    expect(bars[4]).toMatchObject({ direction: "none", width: 0 });
    expect(bars[1]).toMatchObject({ direction: "none", width: 0 });
    expect(directionOf(null)).toBe("none");
    expect(directionOf(Number.NaN)).toBe("none");
  });

  it("copes with all contributions being zero", () => {
    const r = scoredFixture();
    for (const c of r.contributions) c.shap = 0;
    expect(shapBars(r).every((b) => b.width === 0 && b.direction === "none")).toBe(true);
  });

  it("shows the program's own sentence unchanged", () => {
    const r = scoredFixture();
    expect(shapBars(r).map((b) => b.sentence)).toEqual(r.contributions.map((c) => c.sentence));
    expect(directionWord("up")).toBe("Pushed the score up");
    expect(directionWord("down")).toBe("Pushed the score down");
    expect(shapDescription(shapBars(r))).toContain("Rhythm irregularity: pushed the score down");
  });
});

// ---------------------------------------------------------------------------- charts

describe("signalChart", () => {
  it("draws a path and one marker per peak", () => {
    const chart = signalChart(scoredFixture());
    expect(chart?.path.startsWith("M")).toBe(true);
    expect(chart?.markers).toHaveLength(3);
    expect(chart?.xTicks).toHaveLength(5);
    expect(chart?.description).toContain("3 tap peaks");
  });

  it("starts a new sub-path at a gap and never plots a missing value as zero", () => {
    const r = scoredFixture();
    r.signal.smoothed[5] = null;
    const chart = signalChart(r);
    expect(chart?.path.match(/M/g)).toHaveLength(2);
    expect(chart?.markers.every((m) => m.y > 0)).toBe(true);
  });

  it("has no chart for fewer than two points", () => {
    const r = scoredFixture();
    r.signal = { t: [1], smoothed: [1], peak_t: [], peak_v: [] };
    expect(signalChart(r)).toBeNull();
  });
});

describe("curveChart", () => {
  it("draws the points and the fitted line when the program made one", () => {
    const chart = curveChart(scoredFixture());
    expect(chart?.markers).toHaveLength(4);
    expect(chart?.line).not.toBeNull();
    expect(chart?.description).toContain("rises");
  });

  it("draws no line when the curve is not usable", () => {
    const r = scoredFixture();
    r.curve.usable = false;
    expect(curveChart(r)?.line).toBeNull();
  });

  it("skips a tap without an amplitude and has no chart without any", () => {
    const r = scoredFixture();
    r.curve.amps[2] = null;
    expect(curveChart(r)?.markers).toHaveLength(3);
    r.curve.amps = [null, null, null, null];
    expect(curveChart(r)).toBeNull();
  });

  it("keeps the y positions inside the plot", () => {
    for (const m of curveChart(scoredFixture())?.markers ?? []) {
      expect(m.y).toBeGreaterThan(0);
      expect(m.y).toBeLessThan(230);
    }
  });
});

// ---------------------------------------------------------------------------- wording

const BANNED = /\b(diagnosis|positive|negative|cause)\b/i;
const SRC = resolve(__dirname, "..");

function withoutFixedSentence(text: string): string {
  return text.split(CONTRIBUTION_LINE).join("");
}

describe("banned words", () => {
  it("the two fixed sentences are exactly as agreed", () => {
    expect(SCORE_LINE).toBe("Research model score on a 0 to 1 scale. Not a calibrated probability. Does not diagnose.");
    expect(CONTRIBUTION_LINE).toBe("Contributions explain the model's score, not the cause of any condition.");
  });

  it("no text the view builds contains diagnosis, positive, negative or cause", () => {
    const r = scoredFixture();
    const texts = [
      SCORE_LINE, NOT_COMBINED, VIDEO_STAYS_HERE, CHIP_AVAILABLE, CHIP_CHOSEN,
      scoreHeadline(r), bandText(r), thresholdText(r), crossValidationText(r) ?? "",
      scaleDescription(r), auditLine(r), previewAlt(1),
      shapDescription(shapBars(r)), directionWord("up"), directionWord("down"), directionWord("none"),
      signalChart(r)?.description ?? "", curveChart(r)?.description ?? "",
      JSON.stringify(videoProblem("a.avi", 1)), JSON.stringify(videoProblem("a.mp4", 10 ** 10)),
      JSON.stringify(recordingRows(r)),
    ]; // prettier-ignore
    for (const text of texts) expect(withoutFixedSentence(text)).not.toMatch(BANNED);
  });

  it("no new source file of this feature contains one of the words", () => {
    const files = [
      ...readdirSync(__dirname).filter((f) => /\.(ts|tsx)$/.test(f) && !f.includes(".test.")).map((f) => join(__dirname, f)),
      join(SRC, "..", "..", "shared", "tappingResult.ts"),
      join(SRC, "..", "..", "main", "tappingAnalysis.ts"),
      join(SRC, "..", "..", "main", "tappingAnalysisIpc.ts"),
    ];
    expect(files.length).toBeGreaterThanOrEqual(8);
    for (const file of files) {
      const text = withoutFixedSentence(readFileSync(file, "utf8"));
      expect(text.match(BANNED), file).toBeNull();
    }
  });

  it("the banned-word pattern really catches a word", () => {
    for (const bad of ["a diagnosis", "Positive result", "negative", "the cause of it"]) {
      expect(BANNED.test(bad)).toBe(true);
    }
    expect(BANNED.test("because the diagnostic setup")).toBe(false);
  });
});
