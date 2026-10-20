import type { MriResult } from "../mri/mriTypes";
import {
  caveats,
  eddyProfileOf,
  formatSigned,
  missingRows,
  modelSetLabel,
  readScore,
} from "../mri/mriView";

// The technical report for one MRI result: everything the plain summary leaves out. It is a
// single self-contained HTML file (inline style, no scripts, no links) so it can be opened,
// printed to PDF or attached anywhere. Every piece of text is escaped. It contains no name
// or date of birth: only the pseudonymous label the clinician typed, if any.

export function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

const e = escapeHtml;

function table(head: string[], rows: string[][]): string {
  const th = head.map((h) => `<th>${e(h)}</th>`).join("");
  const body = rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("");
  return `<table><thead><tr>${th}</tr></thead><tbody>${body}</tbody></table>`;
}

const STYLE = `
body{font:14px/1.5 -apple-system,"Segoe UI",sans-serif;color:#1C2321;max-width:900px;margin:32px auto;padding:0 20px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px;border-bottom:1px solid #E2DDD6;padding-bottom:4px}
.note{color:#5C6663;font-size:13px}.mono,td.n{font-family:ui-monospace,Menlo,monospace}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:5px 8px;border-bottom:1px solid #EDE9E4;vertical-align:top}
th{color:#5C6663}.box{border:1px solid #E8A33D;background:#FBF3E4;padding:10px 14px;border-radius:8px}
img{max-width:100%;border:1px solid #E2DDD6;border-radius:6px}
@media print{body{margin:0}}`;

export interface ReportMeta {
  /** The pseudonymous label the clinician typed, or null. */
  label: string | null;
  generatedAt: Date;
}

export function buildMriReportHtml(result: MriResult, meta: ReportMeta): string {
  const parts: string[] = [];
  const add = (s: string) => parts.push(s);

  add(`<h1>PD-XAI · MRI module · technical report</h1>`);
  add(
    `<p class="note">Assessment: ${e(meta.label ?? "no ID given")} · generated ${e(meta.generatedAt.toISOString())} · result ${e(result.schema_version)} · model ${e(result.model_version ?? "—")} · ${e(modelSetLabel(result.model_set))} set</p>`,
  );
  add(
    `<p class="box"><strong>Screening and triage aid. Not a diagnosis.</strong> A lower score is never a clearance. This is the output of one modality; no priority is shown because the score is not calibrated and the fusion step is not built.</p>`,
  );
  if (eddyProfileOf(result) === "fast") {
    add(
      `<p class="box">DTI motion correction ran with the fast setting. It has not been checked against the standard (full) setting, so the DTI measurements and this score may differ from what the full setting would give.</p>`,
    );
  }

  // Score
  add(`<h2>Score</h2>`);
  if (result.status === "scored" && result.score) {
    const s = result.score;
    add(
      table(
        ["Item", "Value"],
        [
          ["Raw score (log-odds of the PD class)", `<span class="mono">${e(formatSigned(s.value))}</span>`],
          [
            "95% interval",
            `<span class="mono">${e(formatSigned(s.interval_95[0]))} to ${e(formatSigned(s.interval_95[1]))}</span>`,
          ],
          ["Interval method", e(s.interval_method)],
          ["Calibrated probability", s.calibrated ? "yes" : "no — raw score only"],
          ["Training people", String(s.n_training)],
        ],
      ),
    );
    add(`<p>${e(readScore(s).sentence)}</p><p class="note">${e(s.meaning)}</p>`);
  } else {
    add(
      `<p>No score was produced (${e(result.status)}).${
        result.abstain_reason ? ` ${e(result.abstain_reason)}.` : ""
      }${result.error ? ` ${e(result.error)}` : ""} This is not counted as zero and not read as normal.</p>`,
    );
  }

  // Model card
  const card = result.model_card;
  add(`<h2>The model behind it</h2>`);
  if (card) {
    const cv = card.cross_validated;
    const rows: string[][] = [];
    if (card.training_people) {
      const p = card.training_people;
      rows.push(["Trained on", `${p.total} people (${p.pd} PD, ${p.control} controls)`]);
    }
    if (cv) {
      rows.push([
        "Cross-validated AUC",
        `${cv.auc_mean.toFixed(2)} (95% CI ${cv.auc_ci[0].toFixed(2)}–${cv.auc_ci[1].toFixed(2)})`,
      ]);
      rows.push([
        "Cross-validated balanced accuracy",
        `${(cv.balanced_accuracy_mean * 100).toFixed(0)}% (95% CI ${(cv.balanced_accuracy_ci[0] * 100).toFixed(0)}–${(cv.balanced_accuracy_ci[1] * 100).toFixed(0)}%)`,
      ]);
      rows.push(["Folds × repeats", `${cv.folds ?? "—"} × ${cv.repeats ?? "—"}`]);
    }
    add(rows.length ? table(["Item", "Value"], rows.map((r) => r.map(e))) : "<p>No details reported.</p>");
    if (card.purpose) add(`<p class="note">${e(card.purpose)}</p>`);
  } else {
    add(`<p>The model card was not reported.</p>`);
  }

  // Inputs
  add(`<h2>What the result is built from</h2>`);
  for (const key of ["t1", "dti"] as const) {
    const input = result.inputs[key];
    const used = result.availability[key] ?? false;
    add(`<h3>${key === "t1" ? "T1" : "DTI"} — ${used ? "analysed" : "not used (not counted as zero)"}</h3>`);
    if (used && input) {
      add(
        `<p>${e(input.series_description ?? "—")} · ${e(input.manufacturer ?? "")} ${e(input.scanner_model ?? "")}${
          input.voxel_mm ? ` · ${e(input.voxel_mm.join(" × "))} mm` : ""
        }</p>`,
      );
      const checks = Object.entries(input.checks ?? {}).map(([k, v]) => [
        e(k.replace(/_/g, " ")),
        `<span class="mono">${e(typeof v === "number" ? v.toPrecision(4) : String(v))}</span>`,
      ]);
      if (input.eddy?.profile) {
        checks.push([
          "motion correction",
          `${e(input.eddy.profile)}${input.eddy.seconds != null ? ` (${Math.round(input.eddy.seconds)} s)` : ""}`,
        ]);
      }
      if (checks.length) add(table(["Quality check", "Value"], checks));
      if (input.qc_image && input.qc_image.startsWith("data:image/")) {
        add(`<p><img src="${e(input.qc_image)}" alt="quality-check picture"></p>`);
      }
    } else {
      add(`<p>${input?.found ? "Found, but not used." : "Not in the upload."}</p>`);
    }
  }
  const missing = missingRows(result);
  if (missing.length) {
    add(`<h3>What is missing, and the effect</h3>`);
    add(table(["Missing", "Effect on the result"], missing.map((m) => [e(`${m.what} — ${m.detail}`), e(m.effect)])));
  }

  // Explanation
  const ex = result.explanation;
  if (ex && result.status === "scored") {
    add(`<h2>Why the model reached this score</h2>`);
    add(
      `<p class="note">Method: ${e(ex.method)}, in log-odds; base value ${e(formatSigned(ex.base_value))}. Contributions add up to the score. This is a single method, so there is no second method to compare it against. ${e(ex.caveat)} These are attributions, not causes.</p>`,
    );
    add(
      table(
        ["Measurement", "Contribution", "Direction"],
        ex.top_drivers.map((d) => [
          e(d.label),
          `<span class="mono">${e(formatSigned(d.contribution, 3))}</span>`,
          d.contribution > 0 ? "pushed the score higher" : "pushed the score lower",
        ]),
      ),
    );
  }

  // Feature vector
  const fv = result.feature_vector;
  if (fv) {
    const attributions = result.explanation?.feature_attributions ?? {};
    const outside = new Set(fv.outside_training_range);
    add(`<h2>Feature vector (${fv.names.length} features, harmonised)</h2>`);
    add(
      `<p class="note">Every input by name, after the scanner adjustment (ComBat batch: ${e(Object.values(fv.scanner_batch).join(", "))}). Rows marked “outside range” fall outside what the model was trained on.</p>`,
    );
    add(
      table(
        ["Feature", "Raw", "Harmonised", "Contribution", ""],
        fv.names.map((name, i) => [
          `<span title="${e(name)}">${e(fv.labels[i] ?? name)}</span>`,
          `<span class="mono">${e(fv.raw[i].toPrecision(4))}</span>`,
          `<span class="mono">${e(fv.harmonised[i].toPrecision(4))}</span>`,
          attributions[name] !== undefined
            ? `<span class="mono">${e(formatSigned(attributions[name], 3))}</span>`
            : "–",
          outside.has(name) ? "outside range" : "",
        ]),
      ),
    );
  }

  // Caveats
  const c = caveats(result);
  add(`<h2>Caveats — read before acting on this result</h2>`);
  if (c.population) add(`<p>${e(c.population)}</p>`);
  if (c.limits.length) add(`<ul>${c.limits.map((l) => `<li>${e(l)}</li>`).join("")}</ul>`);
  if (c.outsideRange.length) {
    add(
      `<p class="box">${c.outsideRange.length} measurement(s) fall outside the training range, so the score may be less reliable than its interval suggests: ${e(c.outsideRange.map((o) => o.label).join("; "))}.</p>`,
    );
  }
  add(`<p class="note">${e(result.disclaimer)}</p>`);

  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><title>PD-XAI MRI technical report</title><style>${STYLE}</style></head><body>${parts.join("\n")}</body></html>`;
}

/** A file name that is safe on every system and carries no name: label and date only. */
export function reportFileName(label: string | null, now: Date): string {
  const safe = (label ?? "assessment").replace(/[^A-Za-z0-9_-]/g, "") || "assessment";
  return `pdxai-mri-report-${safe}-${now.toISOString().slice(0, 10)}.html`;
}

/** Saves text as a file through the browser's download, which Electron turns into a save
 *  dialog. Nothing is sent anywhere. */
export function downloadHtml(fileName: string, html: string): void {
  const url = URL.createObjectURL(new Blob([html], { type: "text/html" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}
