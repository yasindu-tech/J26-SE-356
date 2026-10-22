import type { ReactNode } from "react";
import type { MriResult } from "../mri/mriTypes";
import {
  driverBars,
  eddyProfileOf,
  formatSigned,
  modelSetLabel,
  scoreBand,
  SCORE_AXIS,
  statusLabel,
} from "../mri/mriView";
import {
  PART_CAPTIONS,
  PART_STATE_LABELS,
  PART_TITLES,
  STRENGTH_LABELS,
  driverText,
  eddyMinutes,
  partProblems,
  partState,
  plainReading,
  safeImage,
  type ScanPart,
} from "../mri/mriPlain";
import "./MriResultView.css";

// The MRI module's result: the reading, the T1 and DTI pictures, what drove the reading,
// and the numbers in "More details" sections that open and close. Everything is on the
// page; nothing is downloaded.
//
// No High/Moderate/Low priority and no percentage: the score is not calibrated and the
// step that combines parts is not built. A part that was not used is shown as "—", never 0.

const PARTS: readonly ScanPart[] = ["t1", "dti"];
const TOP_DRIVERS = 5;

export function MriResultView({ result }: { result: MriResult }) {
  const scored = result.status === "scored" && result.score !== null;
  return (
    <div className="mr">
      {scored ? <ReadingCard result={result} /> : <NotScoredCard result={result} />}
      <ScanImages result={result} />
      {scored && <ExplanationCard result={result} />}
      <MoreDetails result={result} />
    </div>
  );
}

// ------------------------------------------------------------------ reading

function ReadingCard({ result }: { result: MriResult }) {
  if (!result.score) return null;
  const reading = plainReading(result.score, result.model_card?.track_record);
  return (
    <section className="card mr__reading-card" aria-label="Reading">
      <p className="eyebrow">MRI · {modelSetLabel(result.model_set)} model · research prototype</p>
      <div className="mr__reading-row">
        <h2 className={`mr__reading-head mr__reading-head--${reading.lean}`}>{reading.headline}</h2>
        {reading.strength && (
          <span className={`mr__strength mr__strength--${reading.strength}`}>
            {STRENGTH_LABELS[reading.strength]}
          </span>
        )}
      </div>
      <p className="mr__reading-detail">{reading.detail}</p>
      {(reading.agreement || reading.trackRecord) && (
        <ul className="mr__facts">
          {reading.agreement && <li>{reading.agreement}</li>}
          {reading.trackRecord && <li>{reading.trackRecord}</li>}
        </ul>
      )}
    </section>
  );
}

function NotScoredCard({ result }: { result: MriResult }) {
  const status = statusLabel(result.status);
  return (
    <section className="card mr__reading-card mr__score--none" aria-label="Reading">
      <p className="eyebrow">MRI · research prototype</p>
      <h2 className="mr__reading-head">{status.text}</h2>
      {result.status === "abstained" && (
        <p className="mr__reading-detail">
          No score for this scan: {result.abstain_reason}. It is not counted as zero.
        </p>
      )}
      {result.status === "failed" && (
        <p className="mri__error" role="alert">
          The analysis failed: {result.error}
        </p>
      )}
      {result.validation && result.validation.problems.length > 0 && (
        <ul className="mr__list">
          {result.validation.problems.map((p) => (
            <li key={p}>{p}</li>
          ))}
        </ul>
      )}
    </section>
  );
}

// ------------------------------------------------------------------- images

function ScanImages({ result }: { result: MriResult }) {
  return (
    <section className="mr__images" aria-label="Scans">
      {PARTS.map((part) => {
        const state = partState(result, part);
        const input = result.inputs[part];
        const image = state === "used" ? safeImage(input?.qc_image) : null;
        const scanner = [input?.manufacturer, input?.scanner_model].filter(Boolean).join(" ");
        return (
          <figure key={part} className={`mr__image-card${state === "used" ? "" : " mr__part--none"}`}>
            <div className="mr__part-head">
              <span className="mr__part-title">{PART_TITLES[part]}</span>
              <span className={`mr__chip ${state === "used" ? "mr__chip--on" : "mr__chip--none"}`}>
                {PART_STATE_LABELS[state]}
              </span>
            </div>
            {image ? (
              <img className="mr__image" src={image} alt={`${PART_TITLES[part]} after processing`} />
            ) : (
              <div className="mr__image-empty">
                <span className="mono">—</span>
                <span className="note">
                  {state === "used" ? "No picture was produced." : "Not counted as zero."}
                </span>
              </div>
            )}
            {image && <figcaption className="note">{PART_CAPTIONS[part]}</figcaption>}
            {state !== "used" &&
              partProblems(result, part).map((p) => (
                <p key={p} className="mr__why">
                  {p}
                </p>
              ))}
            {state === "used" && scanner && <p className="mr__meta">{scanner}</p>}
          </figure>
        );
      })}
    </section>
  );
}

// -------------------------------------------------------------- explanation

function ExplanationCard({ result }: { result: MriResult }) {
  const drivers = result.explanation?.top_drivers ?? [];
  if (drivers.length === 0) return null;
  const bars = driverBars(drivers.slice(0, TOP_DRIVERS));
  return (
    <section className="card mr__explain" aria-label="What drove the reading">
      <div className="mr__section-head">
        <h2 className="mr__h2">What drove this reading</h2>
        <div className="mr__shap-legend note" aria-hidden="true">
          <span>
            <i className="mr__key mr__key--outline" /> lower
          </span>
          <span>
            <i className="mr__key mr__key--solid" /> higher
          </span>
        </div>
      </div>
      <ul className="mr__bars">
        {bars.map((b) => (
          <li key={b.feature} className="mr__bar-row">
            <span className="mr__bar-label">{b.label}</span>
            <span className="mr__lane mr__lane--lower">
              {b.side === "lower" && (
                <i className="mr__bar mr__bar--outline" style={{ width: `${b.widthPct}%` }} />
              )}
            </span>
            <span className="mr__zero" aria-hidden="true" />
            <span className="mr__lane">
              {b.side === "higher" && (
                <i className="mr__bar mr__bar--solid" style={{ width: `${b.widthPct}%` }} />
              )}
            </span>
            <span className="mr__sr">{driverText(b)}</span>
          </li>
        ))}
      </ul>
      <p className="note">
        Each bar is how much one measurement moved the model&rsquo;s reading (SHAP). It shows what
        the model used, not a cause.
      </p>
    </section>
  );
}

// ------------------------------------------------------------- more details

function Details({ title, hint, children }: { title: string; hint?: string; children: ReactNode }) {
  return (
    <details className="mr__acc-item">
      <summary className="mr__acc-head">
        <span className="mr__acc-title">{title}</span>
        {hint && <span className="note">{hint}</span>}
        <span className="mr__acc-chevron" aria-hidden="true" />
      </summary>
      <div className="mr__acc-body">{children}</div>
    </details>
  );
}

function Rows({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <ul className="mr__checks">
      {rows.map(([k, v]) => (
        <li key={k}>
          <span>{k}</span>
          <span className="mono">{v}</span>
        </li>
      ))}
    </ul>
  );
}

function MoreDetails({ result }: { result: MriResult }) {
  return (
    <section className="mr__acc" aria-label="More details">
      <h2 className="mr__h2">More details</h2>
      {result.score && <ScoreDetails result={result} />}
      <ModelDetails result={result} />
      <ScanDetails result={result} />
      {result.explanation && result.status === "scored" && <ExplanationDetails result={result} />}
      {result.feature_vector && <MeasurementDetails result={result} />}
    </section>
  );
}

function ScoreDetails({ result }: { result: MriResult }) {
  const s = result.score;
  if (!s) return null;
  const band = scoreBand(s);
  return (
    <Details title="Score" hint={`${formatSigned(s.value)} (${formatSigned(s.interval_95[0])} to ${formatSigned(s.interval_95[1])})`}>
      <div className="mr__band-wrap" aria-hidden="true">
        <div className="mr__band">
          <span className="mr__band-track" />
          <span className="mr__band-ci" style={{ left: `${band.low}%`, width: `${band.width}%` }} />
          <span className="mr__band-zero" style={{ left: `${band.zero}%` }} />
          <span className="mr__band-point" style={{ left: `${band.point}%` }} />
        </div>
        <div className="mr__band-labels">
          <span style={{ left: "0%" }}>{SCORE_AXIS[0]} control side</span>
          <span style={{ left: `${band.zero}%` }}>0</span>
          <span style={{ left: "100%" }}>Parkinson&rsquo;s side +{SCORE_AXIS[1]}</span>
        </div>
      </div>
      <Rows
        rows={[
          ["Raw score (log-odds)", formatSigned(s.value)],
          ["95% interval", `${formatSigned(s.interval_95[0])} to ${formatSigned(s.interval_95[1])}`],
          ["Interval method", s.interval_method],
          ...(s.n_copies != null && s.copies_above_zero != null
            ? [["Copies on the Parkinson's side", `${s.copies_above_zero} of ${s.n_copies}`] as [string, ReactNode]]
            : []),
          ["Calibrated probability", s.calibrated ? "yes" : "no (raw score only)"],
        ]}
      />
      <p className="note">{s.meaning}</p>
    </Details>
  );
}

function ModelDetails({ result }: { result: MriResult }) {
  const card = result.model_card;
  const cv = card?.cross_validated;
  const people = card?.training_people;
  const rows: [string, ReactNode][] = [
    ["Model", `${modelSetLabel(result.model_set)} · ${result.model_version ?? "—"}`],
  ];
  if (people) rows.push(["Trained on", `${people.total} people (${people.pd} PD, ${people.control} controls)`]);
  if (cv) {
    rows.push(["AUC on unseen people", `${cv.auc_mean.toFixed(2)} (95% CI ${cv.auc_ci[0].toFixed(2)}–${cv.auc_ci[1].toFixed(2)})`]);
    rows.push([
      "Balanced accuracy",
      `${(cv.balanced_accuracy_mean * 100).toFixed(0)}% (95% CI ${(cv.balanced_accuracy_ci[0] * 100).toFixed(0)}–${(cv.balanced_accuracy_ci[1] * 100).toFixed(0)}%)`,
    ]);
    rows.push(["Cross-validation", `${cv.folds ?? "—"} folds × ${cv.repeats ?? "—"} repeats, split by person`]);
  }
  return (
    <Details title="The model" hint={cv ? `AUC ${cv.auc_mean.toFixed(2)}` : undefined}>
      <Rows rows={rows} />
      {card?.known_limits && card.known_limits.length > 0 && (
        <>
          <p className="mr__sub">Known limits (model card)</p>
          <ul className="mr__list">
            {card.known_limits.map((l) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
        </>
      )}
    </Details>
  );
}

function ScanDetails({ result }: { result: MriResult }) {
  return (
    <Details title="Scans and quality checks">
      {PARTS.map((part) => {
        const input = result.inputs[part];
        const state = partState(result, part);
        const rows: [string, ReactNode][] = [["Status", PART_STATE_LABELS[state]]];
        if (input?.series_description) rows.push(["Series", input.series_description]);
        if (input?.manufacturer || input?.scanner_model) {
          rows.push(["Scanner", [input.manufacturer, input.scanner_model].filter(Boolean).join(" ")]);
        }
        if (input?.voxel_mm) rows.push(["Voxel size", `${input.voxel_mm.join(" × ")} mm`]);
        for (const [k, v] of Object.entries(input?.checks ?? {})) {
          rows.push([k.replace(/_/g, " "), typeof v === "number" ? v.toPrecision(4) : String(v)]);
        }
        if (part === "dti" && input?.eddy?.profile) {
          const minutes = eddyMinutes(input.eddy.seconds);
          const fast = eddyProfileOf(result) === "fast";
          rows.push([
            "Motion correction",
            `${input.eddy.profile}${minutes !== null ? `, ${minutes} min` : ""}${fast ? " (not validated against full)" : ""}`,
          ]);
        }
        return (
          <div key={part} className="mr__acc-group">
            <p className="mr__sub">{PART_TITLES[part]}</p>
            <Rows rows={rows} />
          </div>
        );
      })}
    </Details>
  );
}

function ExplanationDetails({ result }: { result: MriResult }) {
  const ex = result.explanation;
  const checks = result.model_card?.explanation_checks ?? null;
  if (!ex) return null;
  return (
    <Details title="Explanation" hint={ex.method}>
      <Rows
        rows={[
          ["Method", ex.method],
          ["Base value (log-odds)", formatSigned(ex.base_value)],
          ...ex.top_drivers.map((d): [string, ReactNode] => [d.label, formatSigned(d.contribution, 3)]),
        ]}
      />
      {checks && (
        <>
          <p className="mr__sub">How the explanation was checked ({checks.people ?? "—"} people, unseen by each model)</p>
          <Rows
            rows={[
              [
                "Deletion test (faithfulness)",
                `${checks.deletion_passed ? "Pass" : "Fail"}: removing the top 5 moved the score ${formatSigned(checks.deletion_mean_top_minus_random, 2)} more than 5 random (95% CI ${formatSigned(checks.deletion_ci_95[0], 2)} to ${formatSigned(checks.deletion_ci_95[1], 2)})`,
              ],
              [
                "Stability across model copies",
                `Spearman ${checks.stability_mean_spearman.toFixed(2)} (95% CI ${checks.stability_ci_95[0].toFixed(2)}–${checks.stability_ci_95[1].toFixed(2)}), target 0.70: ${checks.stability_meets_target ? "met" : "not met"}`,
              ],
              [
                "Top-5 overlap between copies",
                `${Math.round(checks.top5_overlap * 100)}%${checks.top5_overlap_by_chance != null ? ` (chance ${Math.round(checks.top5_overlap_by_chance * 100)}%)` : ""}`,
              ],
            ]}
          />
        </>
      )}
      <p className="note">
        Contributions are in log-odds and add up from the base value to the score. {ex.caveat}
      </p>
    </Details>
  );
}

function MeasurementDetails({ result }: { result: MriResult }) {
  const fv = result.feature_vector;
  if (!fv) return null;
  const attributions = result.explanation?.feature_attributions ?? {};
  const outside = new Set(fv.outside_training_range);
  return (
    <Details
      title="All measurements"
      hint={`${fv.names.length} measurements${outside.size ? ` · ${outside.size} outside training range` : ""}`}
    >
      <div className="mr__table-wrap">
        <table className="mr__table">
          <thead>
            <tr>
              <th>Measurement</th>
              <th>Raw</th>
              <th>Adjusted for scanner</th>
              <th>Contribution</th>
            </tr>
          </thead>
          <tbody>
            {fv.names.map((name, i) => (
              <tr key={name} className={outside.has(name) ? "mr__row--outside" : undefined}>
                <td title={name}>{fv.labels[i] ?? name}</td>
                <td className="mono">{fv.raw[i].toPrecision(4)}</td>
                <td className="mono">{fv.harmonised[i].toPrecision(4)}</td>
                <td className="mono">
                  {attributions[name] !== undefined ? formatSigned(attributions[name], 3) : "–"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="note">
        Scanner adjustment (ComBat) batch: {Object.values(fv.scanner_batch).join(", ") || "—"}.
      </p>
    </Details>
  );
}
