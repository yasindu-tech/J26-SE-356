import { describe, expect, it } from "vitest";
import type { MriResult } from "../mri/mriTypes";
import { plainDrivers, plainReading, reliabilityLine } from "../mri/mriPlain";
import { buildMriReportHtml, escapeHtml, reportFileName } from "./mriReport";

const result: MriResult = {
  modality: "mri",
  schema_version: "mri-result/1",
  status: "scored",
  abstain_reason: null,
  error: null,
  model_version: "20261015-1",
  model_set: "t1dti",
  availability: { t1: true, dti: true },
  inputs: {
    t1: { found: true, series_description: "MPRAGE", checks: { brain_mask_ml: 1551 } },
    dti: { found: true, eddy: { profile: "fast", seconds: 1497 }, checks: { median_fa: 0.2065 } },
  },
  validation: { usable: true, problems: [] },
  feature_vector: {
    names: ["t1_a", "dti_b"],
    labels: ["Left <thalamus>", "Right putamen"],
    raw: [1.5, 2.5],
    harmonised: [0.1, 0.2],
    scanner_batch: { t1: "TrioTim" },
    outside_training_range: ["dti_b"],
  },
  score: {
    kind: "log_odds",
    value: 1.52,
    interval_95: [-1.35, 3.17],
    interval_method: "spread over 50 fold copies",
    calibrated: false,
    n_training: 37,
    meaning: "Raw score.",
  },
  explanation: {
    kind: "attribution",
    method: "TreeSHAP",
    base_value: 0,
    feature_attributions: { t1_a: 0.7, dti_b: -0.4 },
    caveat: "Not a cause.",
    top_drivers: [
      { feature: "t1_a", label: "Left <thalamus>", value: 1, contribution: 0.7, direction: "towards PD", text: "" },
      { feature: "dti_b", label: "Right putamen", value: 1, contribution: -0.4, direction: "towards control", text: "" },
    ],
  },
  model_card: {
    training_people: { total: 37, pd: 18, control: 19 },
    cross_validated: {
      auc_mean: 0.6,
      auc_ci: [0.44, 0.81],
      balanced_accuracy_mean: 0.6,
      balanced_accuracy_ci: [0.46, 0.77],
      folds: 5,
      repeats: 10,
    },
    known_limits: ["Small sample."],
  },
  disclaimer: "Screening aid. Not a diagnosis.",
};

describe("plain reading", () => {
  it("says no clear lean when the interval crosses 0", () => {
    expect(plainReading({ interval_95: [-1.35, 3.17] }).lean).toBe("none");
  });

  it("never reads a lower score as a clearance or a higher one as a diagnosis", () => {
    const lower = plainReading({ interval_95: [-3, -1] });
    const higher = plainReading({ interval_95: [1, 3] });
    expect(lower.detail).toContain("never a clearance");
    expect(higher.detail).toContain("not a diagnosis");
    for (const r of [plainReading({ interval_95: [-1, 1] }), lower, higher]) {
      const text = `${r.headline} ${r.detail}`.toLowerCase();
      expect(text).not.toMatch(/log-odds|auc|\d%|high priority|moderate|positive|negative/);
    }
  });

  it("calls the model weak when its test interval includes chance, and still cautious when not", () => {
    expect(reliabilityLine(result.model_card)).toContain("could not be reliably told apart from chance");
    const better = { ...result.model_card!, cross_validated: { ...result.model_card!.cross_validated!, auc_ci: [0.6, 0.9] as [number, number] } };
    expect(reliabilityLine(better)).toContain("did better than chance");
    expect(reliabilityLine(null)).toContain("not reported");
  });

  it("describes drivers as nudges, not causes, with no numbers", () => {
    const [first, second] = plainDrivers(result.explanation!.top_drivers);
    expect(first.text).toBe("Left <thalamus> nudged the reading higher.");
    expect(second.text).toContain("lower");
    expect(`${first.text}${second.text}`).not.toMatch(/\d|caus/i);
  });
});

describe("technical report", () => {
  const html = buildMriReportHtml(result, { label: "P-8F31A2", generatedAt: new Date("2026-10-18T12:00:00Z") });

  it("holds the numbers the plain page leaves out", () => {
    expect(html).toContain("+1.52");
    expect(html).toContain("−1.35");
    expect(html).toContain("0.60 (95% CI 0.44–0.81)");
    expect(html).toContain("Left &lt;thalamus&gt;");
    expect(html).toContain("outside range");
    expect(html).toContain("fast"); // motion correction setting
    expect(html).toContain("P-8F31A2");
  });

  it("escapes everything it was given, so a label cannot inject markup", () => {
    const evil = buildMriReportHtml(result, { label: '<img src=x onerror="alert(1)">', generatedAt: new Date() });
    expect(evil).not.toContain("<img src=x");
    expect(evil).toContain("&lt;img src=x");
    expect(escapeHtml(`<&>"'`)).toBe("&lt;&amp;&gt;&quot;&#39;");
    expect(html).not.toContain("<script");
  });

  it("says it is not a diagnosis and shows no priority or percentage score", () => {
    expect(html).toContain("Not a diagnosis");
    expect(html.toLowerCase()).not.toMatch(/high priority|moderate priority|low priority/);
  });

  it("names the file from a safe label and the date only", () => {
    expect(reportFileName("P-8F31A2", new Date("2026-10-18T09:00:00Z"))).toBe("pdxai-mri-report-P-8F31A2-2026-10-18.html");
    expect(reportFileName("../etc passwd", new Date("2026-10-18T09:00:00Z"))).toBe("pdxai-mri-report-etcpasswd-2026-10-18.html");
    expect(reportFileName(null, new Date("2026-10-18T09:00:00Z"))).toContain("assessment");
  });
});
