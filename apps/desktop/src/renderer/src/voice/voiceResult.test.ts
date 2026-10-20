import { describe, expect, it } from "vitest";
import { parseVoiceResults, triageLabel } from "./voiceResult";

// Synthetic values in the shape demo_voice.py writes. Not real people's results.
function scored(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    person: "P-TEST01",
    reference_label: "PD",
    modality: "voice",
    risk_score: 0.7,
    ci_lower: 0.55,
    ci_upper: 0.82,
    triage_priority: "prioritised_referral",
    availability_mask: { voice: true },
    shap_top5: [
      { feature: "feature_a", shap_log_odds: 0.4, direction: "towards higher risk" },
      { feature: "feature_b", shap_log_odds: -0.2, direction: "towards lower risk" },
    ],
    sex_fairness_note: "Exploratory note.",
    pipeline_version: "voice-pp1-0.1.0+l1_logistic",
    quality_gate_passed: true,
    abstained: false,
    problems: [],
    n_training: 201,
    note: "Voice screening score for triage only.",
    ...overrides,
  };
}

function abstained(): Record<string, unknown> {
  return scored({
    person: "broken-input-example",
    reference_label: "-",
    risk_score: null,
    ci_lower: null,
    ci_upper: null,
    triage_priority: null,
    availability_mask: { voice: false },
    shap_top5: [],
    quality_gate_passed: false,
    abstained: true,
    problems: ["1 blank or non-numeric values"],
  });
}

function rejection(content: unknown): string {
  const outcome = parseVoiceResults(content);
  if (outcome.ok) throw new Error("expected the file to be rejected");
  return outcome.message;
}

describe("parseVoiceResults", () => {
  it("parses a demo_results.json list with scored and abstained results", () => {
    const outcome = parseVoiceResults([scored(), abstained()]);
    if (!outcome.ok) throw new Error(outcome.message);
    const [first, second] = outcome.results;
    expect(first).toMatchObject({ kind: "scored", riskScore: 0.7, ciLower: 0.55, ciUpper: 0.82 });
    expect(first?.kind === "scored" && first.shapTop5[0]).toEqual({
      feature: "feature_a",
      shapLogOdds: 0.4,
    });
    expect(second).toMatchObject({
      kind: "abstained",
      problems: ["1 blank or non-numeric values"],
    });
  });

  it("accepts a single result object", () => {
    const outcome = parseVoiceResults(scored());
    expect(outcome.ok && outcome.results).toHaveLength(1);
  });

  it("rejects diagnosis wording as a triage priority", () => {
    expect(rejection(scored({ triage_priority: "positive" }))).toContain("triage_priority");
    expect(rejection(scored({ triage_priority: "cleared" }))).toContain("triage_priority");
  });

  it("rejects a score without its confidence interval", () => {
    expect(rejection(scored({ ci_lower: null }))).toContain("ci_lower");
    expect(rejection(scored({ ci_upper: undefined }))).toContain("ci_upper");
  });

  it("rejects a score outside its confidence interval", () => {
    expect(rejection(scored({ risk_score: 0.9 }))).toContain("inside its confidence interval");
  });

  it("rejects a triage priority that contradicts the score", () => {
    expect(rejection(scored({ risk_score: 0.3, ci_lower: 0.2, ci_upper: 0.4 }))).toContain(
      "does not match the risk score",
    );
  });

  it("rejects scores outside 0 to 1", () => {
    expect(rejection(scored({ risk_score: 1.4, ci_upper: 1.5 }))).toContain("risk_score");
  });

  it("rejects a result from another modality", () => {
    expect(rejection(scored({ modality: "gait" }))).toContain("modality");
  });

  it("rejects an abstained result without its reasons", () => {
    const broken = { ...abstained(), problems: undefined };
    expect(rejection(broken)).toContain("problems");
  });

  it("rejects an empty file and a file that is not a result", () => {
    expect(rejection([])).toContain("no results");
    expect(rejection({ hello: "world" })).toContain("modality");
    expect(rejection("text")).toContain("object");
  });
});

describe("triageLabel", () => {
  it("uses referral language, never diagnosis language", () => {
    expect(triageLabel("prioritised_referral")).toBe("Prioritised referral");
    expect(triageLabel("not_prioritised")).toBe("Not prioritised");
  });
});

// Synthetic explanation block in the shape analyze_audio.py writes.
function explanations(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    summary: "Risk score 0.70 (95% range 0.55-0.82). Biggest single reason: ...",
    top_reasons: [
      {
        comparable: true,
        feature: "jitter_local",
        meaning: "cycle-to-cycle wobble in pitch",
        group: "jitter",
        value: 0.006,
        shap_log_odds: 0.4,
        effect: "raised",
        percentile_among_healthy: 96.5,
        percentile_among_pd: 70,
        sentence:
          "Cycle-to-cycle wobble in pitch (jitter_local): higher than 96% of healthy training voices. The model used this, and it raised the score.",
      },
    ],
    uncertainty: ["The model was trained on only 61 people (figshare)."],
    quality: {
      passed: true,
      problems: [],
      measured: { source_format: "mp3", duration_s: 4.2, vowel_s: 2.9, snr_db: null },
    },
    model_overview: {
      top_groups: [{ group: "mfcc", share: 0.6 }],
      sentence: "In general the model relies most on mfcc (60%) measures.",
    },
    ...overrides,
  };
}

describe("parseVoiceResults with explanations", () => {
  it("parses the explanation block from analyze_audio.py", () => {
    const outcome = parseVoiceResults(scored({ explanations: explanations() }));
    if (!outcome.ok) throw new Error(outcome.message);
    const parsed = outcome.results[0]?.explanations;
    expect(parsed?.topReasons[0]).toMatchObject({
      feature: "jitter_local",
      effect: "raised",
      percentileAmongHealthy: 96.5,
      comparable: true,
    });
    expect(parsed?.quality.measured).toEqual({
      sourceFormat: "mp3",
      sourceSampleRate: undefined,
      durationS: 4.2,
      vowelS: 2.9,
      snrDb: undefined,
      clippedFraction: undefined,
    });
    expect(parsed?.modelOverview.topGroups).toEqual([{ group: "mfcc", share: 0.6 }]);
  });

  it("keeps older result files without explanations working", () => {
    const outcome = parseVoiceResults(scored());
    expect(outcome.ok && outcome.results[0]?.explanations).toBeUndefined();
  });

  it("parses an abstained result with only quality reasons", () => {
    const block = explanations({
      summary: "No score: the recording did not pass the quality check.",
      top_reasons: [],
      uncertainty: [],
      quality: { passed: false, problems: ["The steady 'aaah' is only 0.4 s long."], measured: {} },
    });
    const outcome = parseVoiceResults({ ...abstained(), explanations: block });
    if (!outcome.ok) throw new Error(outcome.message);
    expect(outcome.results[0]?.explanations?.quality.passed).toBe(false);
  });

  it("rejects explanation text with diagnostic wording", () => {
    const block = explanations({ uncertainty: ["The test was positive for Parkinson's."] });
    expect(rejection(scored({ explanations: block }))).toContain("diagnostic wording");
    const summary = explanations({ summary: "This voice causes concern." });
    expect(rejection(scored({ explanations: summary }))).toContain("diagnostic wording");
  });

  it("rejects malformed explanations instead of showing part of them", () => {
    const badEffect = explanations({
      top_reasons: [{ ...(explanations().top_reasons as object[])[0], effect: "worse" }],
    });
    expect(rejection(scored({ explanations: badEffect }))).toContain("effect");
    const badPercentile = explanations({
      top_reasons: [
        { ...(explanations().top_reasons as object[])[0], percentile_among_healthy: 140 },
      ],
    });
    expect(rejection(scored({ explanations: badPercentile }))).toContain("percentile");
    expect(rejection(scored({ explanations: "text" }))).toContain("explanations");
  });

  it("treats a reason without the comparable field as comparable", () => {
    const reason = { ...(explanations().top_reasons as Record<string, unknown>[])[0] };
    delete reason["comparable"];
    const outcome = parseVoiceResults(
      scored({ explanations: explanations({ top_reasons: [reason] }) }),
    );
    expect(outcome.ok && outcome.results[0]?.explanations?.topReasons[0]?.comparable).toBe(true);
  });
});
