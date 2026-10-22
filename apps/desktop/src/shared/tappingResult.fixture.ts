// A valid "tapping-result/1" object for tests. Not imported by the app.

import type { TappingFailed, TappingScored } from "./tappingResult";

export function scoredFixture(): TappingScored {
  const t = Array.from({ length: 40 }, (_, i) => i * 0.5);
  return {
    schema_version: "tapping-result/1",
    status: "scored",
    clip_ref: "f0e37b01",
    target: "ge2",
    recording: { duration_s: 19.5, fps: 59.9, hand_found_pct: 100, n_taps: 29 },
    quality: {
      flags: [{ name: "low_range", text: "the opening range is small, so taps may be missed" }],
      notes: ["Scored with the demo model (trained on all research clips)."],
    },
    score: {
      value: 0.089876,
      band_low: 0.015538,
      band_high: 0.10247,
      band_basis: "5-95% of 25 cross-validation models",
      threshold: 0.5,
      above_threshold: false,
      cv_auroc: 0.869589,
      cv_ci_low: 0.809327,
      cv_ci_high: 0.918123,
    },
    contributions: [
      { feature_label: "Rhythm irregularity", value_text: "0.08 CV", percentile_text: "14th percentile of the research clips", shap: -0.67, sentence: "Rhythm irregularity (0.08 CV, 14th percentile of the research clips) pushed the score down by 0.67 (log-odds)." },
      { feature_label: "Tapping rate", value_text: "1.73 taps/s", percentile_text: "70th percentile of the research clips", shap: -0.51, sentence: "Tapping rate (1.73 taps/s, 70th percentile of the research clips) pushed the score down by 0.51 (log-odds)." },
      { feature_label: "Mean amplitude", value_text: "1.15 hand sizes", percentile_text: "34th percentile of the research clips", shap: -0.15, sentence: "Mean amplitude (1.15 hand sizes, 34th percentile of the research clips) pushed the score down by 0.15 (log-odds)." },
      { feature_label: "Opening speed", value_text: "7.99 hand sizes/s", percentile_text: "31st percentile of the research clips", shap: 0.13, sentence: "Opening speed (7.99 hand sizes/s, 31st percentile of the research clips) pushed the score up by 0.13 (log-odds)." },
      { feature_label: "Closing speed", value_text: "11.62 hand sizes/s", percentile_text: "51st percentile of the research clips", shap: 0.0, sentence: "Closing speed (11.62 hand sizes/s, 51st percentile of the research clips) had almost no effect on the score (+0.00 log-odds)." },
    ],
    curve: {
      times: [1, 2, 3, 4],
      amps: [1.1, 1.15, 1.2, 1.25],
      line_x: [1, 4],
      line_y: [1.1, 1.25],
      slope_sentence: "Fitted line (Theil-Sen): amplitude changes by 9.9% of its mean per 10 s (rises).",
      count_sentence: "Taps shown: 4 of 4.",
      note: "The score above uses Set A, which has no amplitude-trend feature. This curve explains the signal; it did not change the score.",
      usable: true,
    },
    signal: {
      t,
      smoothed: t.map((x) => 0.5 + 0.3 * Math.sin(x * 3)),
      peak_t: [0.5, 1.5, 2.5],
      peak_v: [0.8, 0.8, 0.8],
    },
    preview_frames: [{ t: 0.5, jpeg_base64: "/9j/4AAQSkZJRg==" }],
    model: { demo_model_sha8: "1c623df9", folds_sha8: "ae519116" },
    pdf_base64: "JVBERi0xLjQK",
  };
}

export function failedFixture(): TappingFailed {
  return {
    schema_version: "tapping-result/1",
    status: "failed",
    reason: "Hand not found in enough frames; keep the hand in view.",
    clip_ref: "f0e37b01",
    target: "ge2",
  };
}
