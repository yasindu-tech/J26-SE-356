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
