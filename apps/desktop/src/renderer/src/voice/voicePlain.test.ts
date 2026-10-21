import { describe, expect, it } from "vitest";
import {
  OTHER_CONDITIONS_LINE,
  plainDrivers,
  plainReading,
  qualitySummary,
  reliabilityLine,
} from "./voicePlain";
import type { ExplainedReason, VoiceResult } from "./voiceResult";

// Synthetic results only.
function reason(
  feature: string,
  group: string,
  overrides: Partial<ExplainedReason> = {},
) {
  return {
    feature,
    group,
    meaning: feature,
    sentence: feature,
    effect: "raised",
    percentileAmongHealthy: 50,
    comparable: true,
    ...overrides,
  } satisfies ExplainedReason;
}

function scored(
  overrides: Partial<Extract<VoiceResult, { kind: "scored" }>> = {},
): VoiceResult {
  return {
    kind: "scored",
    person: "upload-test",
    referenceLabel: "-",
    sexFairnessNote: "",
    pipelineVersion: "voice-audio-0.1.0+l1_logistic",
    nTraining: 61,
    note: "",
    riskScore: 0.28,
    ciLower: 0.05,
    ciUpper: 0.34,
    triagePriority: "not_prioritised",
    shapTop5: [
      { feature: "mfcc9_sd", shapLogOdds: -0.46 },
      { feature: "mfcc10_sd", shapLogOdds: -0.42 },
      { feature: "jitter_local_abs", shapLogOdds: 0.3 },
    ],
    explanations: {
      summary: "",
      topReasons: [
        reason("mfcc9_sd", "mfcc"),
        reason("mfcc10_sd", "mfcc"),
        reason("jitter_local_abs", "jitter", { percentileAmongHealthy: 97 }),
      ],
      uncertainty: [],
      quality: { passed: true, problems: [], measured: {} },
      modelOverview: { sentence: "", topGroups: [] },
    },
    ...overrides,
  };
}

const abstained: VoiceResult = {
  kind: "abstained",
  person: "upload-test",
  referenceLabel: "-",
  sexFairnessNote: "",
  pipelineVersion: "voice-audio-0.1.0+l1_logistic",
  nTraining: 61,
  note: "",
  problems: ["too short"],
};

describe("plainReading", () => {
  it("never words a low result as a clearance or a diagnosis", () => {
    const low = plainReading(scored());
    expect(low.headline).toBe("Not prioritised");
    expect(low.detail).toContain("never a clearance");
    const high = plainReading(
      scored({
        riskScore: 0.7,
        ciLower: 0.6,
        ciUpper: 0.9,
        triagePriority: "prioritised_referral",
      }),
    );
    expect(high.tone).toBe("attention");
    expect(high.detail).toContain("not a diagnosis");
    for (const text of [low.headline, low.detail, high.headline, high.detail]) {
      expect(text).not.toMatch(
        /\b(positive|negative|cleared|discharged|has Parkinson)\b/i,
      );
    }
  });

  it("flags a range that crosses the referral line", () => {
    expect(plainReading(scored()).borderline).toBe(false);
    expect(
      plainReading(
        scored({
          riskScore: 0.59,
          ciLower: 0.2,
          ciUpper: 0.98,
          triagePriority: "prioritised_referral",
        }),
      ).borderline,
    ).toBe(true);
  });

  it("says an unusable recording is not a normal result", () => {
    const reading = plainReading(abstained);
    expect(reading.headline).toBe("Not scored");
    expect(reading.detail).toContain("not a normal result");
  });
});

describe("plainDrivers", () => {
  it("merges measures of one group and ranks groups by their total effect", () => {
    const drivers = plainDrivers(scored());
    expect(drivers.map((d) => d.key)).toEqual(["mfcc", "jitter"]);
    expect(drivers[0]?.text).toBe(
      "Voice tone pushed the score down: it is within the usual range for healthy voices.",
    );
    expect(drivers[1]?.text).toContain("unusual compared with healthy voices");
  });

  it("warns when a measure is not comparable with the training voices", () => {
    const result = scored({
      shapTop5: [{ feature: "locPctJitter", shapLogOdds: 0.6 }],
      explanations: {
        summary: "",
        topReasons: [reason("locPctJitter", "jitter", { comparable: false })],
        uncertainty: [],
        quality: { passed: true, problems: [], measured: {} },
        modelOverview: { sentence: "", topGroups: [] },
      },
    });
    expect(plainDrivers(result)[0]?.text).toContain("measured differently");
  });

  it("gives nothing for an abstained result", () => {
    expect(plainDrivers(abstained)).toEqual([]);
  });
});

describe("reliabilityLine and qualitySummary", () => {
  it("mentions a wide range", () => {
    expect(reliabilityLine(scored())).not.toContain("wide");
    expect(reliabilityLine(scored({ ciLower: 0.2, ciUpper: 0.98 }))).toContain(
      "wide",
    );
  });

  it("warns on every score that other voice problems can raise it", () => {
    expect(reliabilityLine(scored())).toContain(OTHER_CONDITIONS_LINE);
    expect(reliabilityLine(scored({ ciLower: 0.2, ciUpper: 0.98 }))).toContain(
      "asthma",
    );
  });

  it("summarises the recording check in one line", () => {
    expect(
      qualitySummary(true, { vowelS: 3.5, snrDb: 15, sourceFormat: "m4a" }),
    ).toBe(
      "Passed · 3.5 s of steady 'aaah' · 15 dB above background noise · M4A",
    );
    expect(qualitySummary(false, {})).toBe("Not usable");
  });
});
