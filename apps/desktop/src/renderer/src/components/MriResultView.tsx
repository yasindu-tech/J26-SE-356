import { useState } from "react";
import type { MriModelCard, MriResult, MriScore, MriSeriesInput } from "../mri/mriTypes";
import {
  FAST_EDDY_NOTICE,
  SCORE_AXIS,
  caveats,
  driverBars,
  driverSentence,
  eddyProfileOf,
  formatSigned,
  missingRows,
  modelSetLabel,
  readScore,
  scoreBand,
  statusLabel,
  vectorJson,
} from "../mri/mriView";
import "./MriResultView.css";

// The MRI module's output for one scan, laid out like the approved result design.
//
// It deliberately shows the raw model score (log-odds) with its 95% interval instead of a
// percentage and a High/Moderate/Low priority: the score is not calibrated (37 people are
// too few), and the priority is set by the fusion step, which is not built yet. A low score
// is never shown as a clearance. The feature vector is shown because it is what fusion will
// consume.

export function MriResultView({
  result,
  onNewAnalysis,
}: {
  result: MriResult;
  onNewAnalysis?: () => void;
}) {
  const fast = eddyProfileOf(result) === "fast";
  return (
    <div className="mr">
      {fast && (
        <p className="mr__notice" role="note">
          {FAST_EDDY_NOTICE}
        </p>
      )}

      <section className="mr__row-a" aria-label="Score summary">
        {result.status === "scored" && result.score ? (
          <ScoreCard result={result} score={result.score} />
        ) : (
          <NotScoredCard result={result} />
        )}
        <TrustCard result={result} />
      </section>

      <InputsSection result={result} />
      <MissingSection result={result} onNewAnalysis={onNewAnalysis} />
      {result.explanation && result.status === "scored" && <ExplanationSection result={result} />}
      {result.feature_vector && <VectorSection result={result} />}
      <CaveatsSection result={result} />

      <footer className="mr__footer mono">
        <span>
          {result.schema_version} · model {result.model_version ?? "—"} · {modelSetLabel(result.model_set)}{" "}
          set
        </span>
      </footer>
      <p className="note">{result.disclaimer}</p>
    </div>
  );
}

// ------------------------------------------------------------------ score

function ScoreCard({ result, score }: { result: MriResult; score: MriScore }) {
  const reading = readScore(score);
  const band = scoreBand(score);
  const [low, high] = score.interval_95;
  const label = statusLabel(result.status);
  return (
    <div className="card mr__score">
      <div className="mr__head">
        <div>
          <p className="eyebrow">MRI module score · raw, not calibrated</p>
          <h2 className="mr__headline">Scored with the {modelSetLabel(result.model_set)} model</h2>
        </div>
        <span className={`mri__tag mri__tag--${label.tone}`}>{label.text}</span>
      </div>

      <div className="mr__figures">
        <div className="mr__figure">
          <span className="mr__big mono">{formatSigned(score.value)}</span>
          <span className="note">score (log-odds)</span>
        </div>
        <div className="mr__figure">
          <span className="mr__interval mono">
            {formatSigned(low)} to {formatSigned(high)}
          </span>
          <span className="note">95% interval · read the range, not the point</span>
        </div>
      </div>

      <div className="mr__band-wrap">
        <div
          className="mr__band"
          role="img"
          aria-label={`Score ${formatSigned(score.value)}, 95% interval ${formatSigned(low)} to ${formatSigned(high)}, on a scale from ${SCORE_AXIS[0]} to ${SCORE_AXIS[1]}`}
        >
          <div className="mr__band-track" />
          <div className="mr__band-zero" style={{ left: `${band.zero}%` }} />
          <div className="mr__band-ci" style={{ left: `${band.low}%`, width: `${band.width}%` }} />
          <div className="mr__band-point" style={{ left: `${band.point}%` }} />
        </div>
        <div className="mr__band-labels mono">
          <span style={{ left: "0%" }}>{SCORE_AXIS[0]}</span>
          <span style={{ left: `${band.zero}%` }}>0</span>
          <span className="mr__band-strong" style={{ left: `${band.low}%` }}>
            {formatSigned(low, 1)}
          </span>
          <span className="mr__band-strong" style={{ left: `${band.point}%` }}>
            {formatSigned(score.value, 1)}
          </span>
          <span className="mr__band-strong" style={{ left: `${band.high}%` }}>
            {formatSigned(high, 1)}
          </span>
          <span style={{ left: "100%" }}>+{SCORE_AXIS[1]}</span>
        </div>
        <div className="mr__legend note">
          <span>
            <i className="mr__swatch mr__swatch--ci" />
            95% interval
          </span>
          <span>
            <i className="mr__swatch mr__swatch--point" />
            score
          </span>
          <span>
            <i className="mr__swatch mr__swatch--zero" />0 = the boundary the model learned
          </span>
        </div>
      </div>

      <p className="mr__reading">{reading.sentence}</p>
      <p className="note">
        This is a score from one modality. It is not a diagnosis and not a referral priority:
        no priority is shown because the score is not calibrated and the fusion step that sets
        the priority is not built yet.
      </p>
    </div>
  );
}

function NotScoredCard({ result }: { result: MriResult }) {
  const label = statusLabel(result.status);
  return (
    <div className="card mr__score mr__score--none">
      <div className="mr__head">
        <div>
          <p className="eyebrow">MRI module score</p>
          <h2 className="mr__headline">{label.text}</h2>
        </div>
        <span className={`mri__tag mri__tag--${label.tone}`}>{label.text}</span>
      </div>
      <div className="mr__figures">
        <div className="mr__figure">
          <span className="mr__big mono mr__big--none">—</span>
          <span className="note">no score produced</span>
        </div>
      </div>
      {result.status === "abstained" && (
        <p className="mr__reading">
          The module did not score this scan: {result.abstain_reason}. That is deliberate. A guess
          from unusable input would be worse than no score. It is <strong>not counted as zero</strong>{" "}
          and not read as normal.
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
    </div>
  );
}

function TrustCard({ result }: { result: MriResult }) {
  const card: MriModelCard | null = result.model_card;
  const cv = card?.cross_validated;
  return (
    <div className="card">
      <p className="eyebrow">How far to trust this score</p>
      <h2 className="mr__sub">About the model behind it</h2>
      {card?.training_people && (
        <p className="mr__text">
          Trained on <strong>{card.training_people.total} people</strong> ({card.training_people.pd} PD,{" "}
          {card.training_people.control} controls). A small sample means a wide uncertainty.
        </p>
      )}
      {cv && (
        <p className="mr__text">
          Tested on people it had never seen ({cv.folds} folds × {cv.repeats} repeats): AUC{" "}
          <span className="mono">{cv.auc_mean.toFixed(2)}</span> (95% CI{" "}
          <span className="mono">
            {cv.auc_ci[0].toFixed(2)}–{cv.auc_ci[1].toFixed(2)}
          </span>
          ), balanced accuracy{" "}
          <span className="mono">{Math.round(cv.balanced_accuracy_mean * 100)}%</span> (95% CI{" "}
          <span className="mono">
            {Math.round(cv.balanced_accuracy_ci[0] * 100)}–{Math.round(cv.balanced_accuracy_ci[1] * 100)}%
          </span>
          ). Read the range, not the point.
        </p>
      )}
      {result.score && (
        <p className="note">
          {result.score.meaning} The interval is the spread over copies of the pipeline (
          {result.score.interval_method}).
        </p>
      )}
      {card?.purpose && <p className="note">{card.purpose}</p>}
    </div>
  );
}

// ------------------------------------------------------- what the score is built from

function InputsSection({ result }: { result: MriResult }) {
  return (
    <section className="mr__section" aria-label="What this result is built from">
      <div className="mr__section-head">
        <h2 className="mr__h2">What this result is built from</h2>
        <span className="note">
          Each part is shown with the quality checks it had to pass. A part that was not used is
          dashed and scored &ldquo;—&rdquo;, never zero.
        </span>
      </div>
      <div className="mr__parts">
        {(["t1", "dti"] as const).map((part) => (
          <PartCard
            key={part}
            part={part}
            used={result.availability[part] ?? false}
            input={result.inputs[part]}
          />
        ))}
      </div>
    </section>
  );
}

function PartCard({
  part,
  used,
  input,
}: {
  part: "t1" | "dti";
  used: boolean;
  input: MriSeriesInput | undefined;
}) {
  const title = part === "t1" ? "T1" : "DTI";
  const found = input?.found ?? false;
  if (!used) {
    return (
      <div className="mr__part mr__part--none">
        <div className="mr__part-head">
          <span className="mr__part-title">{title}</span>
          <span className="mr__chip mr__chip--none">{found ? "Not used" : "Not in the upload"}</span>
        </div>
        <span className="mr__big mono mr__big--none mr__big--small">—</span>
        <p className="mr__text">
          {found
            ? "Found, but not used. It did not pass processing or its quality checks, or the model that scored this scan does not take it."
            : "This study had no scan of this kind."}{" "}
          <strong>Not counted as zero.</strong>
        </p>
      </div>
    );
  }
  const eddy = input?.eddy;
  return (
    <div className="mr__part">
      <div className="mr__part-head">
        <span className="mr__part-title">{title}</span>
        <span className="mr__chip mr__chip--on">Analysed</span>
      </div>
      <p className="mr__text">
        {input?.series_description} · {input?.manufacturer} {input?.scanner_model}
        {input?.voxel_mm ? ` · ${input.voxel_mm.join(" × ")} mm` : ""}
      </p>
      {input?.checks && (
        <ul className="mr__checks mono">
          {Object.entries(input.checks).map(([k, v]) => (
            <li key={k}>
              <span>{k.replace(/_/g, " ")}</span>
              <span>{typeof v === "number" ? v.toPrecision(4) : String(v)}</span>
            </li>
          ))}
          {eddy?.profile && (
            <li>
              <span>motion correction</span>
              <span>{eddy.profile}</span>
            </li>
          )}
        </ul>
      )}
      {input?.qc_image && (
        <img className="mr__qc" src={input.qc_image} alt={`${title} quality-check picture`} />
      )}
    </div>
  );
}

// ------------------------------------------------------------------ missing

function MissingSection({
  result,
  onNewAnalysis,
}: {
  result: MriResult;
  onNewAnalysis?: () => void;
}) {
  const rows = missingRows(result);
  if (rows.length === 0) return null;
  return (
    <section className="mr__missing" aria-label="What is missing">
      <h2 className="mr__h2">What&rsquo;s missing, and how it affects this result</h2>
      <div className="mr__missing-grid">
        <span className="mr__th">Missing</span>
        <span className="mr__th">Effect on the result</span>
        <span className="mr__th">Action</span>
        {rows.map((row) => (
          <MissingRowView key={row.key} row={row} onNewAnalysis={onNewAnalysis} />
        ))}
      </div>
    </section>
  );
}

function MissingRowView({
  row,
  onNewAnalysis,
}: {
  row: ReturnType<typeof missingRows>[number];
  onNewAnalysis?: () => void;
}) {
  return (
    <>
      <div className="mr__missing-what">
        {row.what}
        <div className="note">{row.detail}</div>
      </div>
      <div className="mr__text">{row.effect}</div>
      <div>
        {row.action && onNewAnalysis && (
          <button type="button" className="btn btn--outline mr__small-btn" onClick={onNewAnalysis}>
            New MRI analysis
          </button>
        )}
        {row.action && !onNewAnalysis && <span className="note">{row.action}</span>}
      </div>
    </>
  );
}

// ------------------------------------------------------------------ explanation

function ExplanationSection({ result }: { result: MriResult }) {
  const explanation = result.explanation;
  if (!explanation) return null;
  const bars = driverBars(explanation.top_drivers);
  return (
    <section className="card" aria-label="Why the model reached this score">
      <div>
        <h2 className="mr__h2">Why the model reached this score</h2>
        <p className="note">
          A plain-language summary first; the chart shows what contributed to the model&rsquo;s
          prediction. These are attributions, not causes: a measurement that contributed to the
          prediction is not a measurement that caused disease.
        </p>
      </div>

      <ul className="mr__list mr__list--large">
        {explanation.top_drivers.map((d) => (
          <li key={d.feature}>{driverSentence(d)}</li>
        ))}
      </ul>

      <div className="mr__shap">
        <div className="mr__shap-head">
          <span className="mr__shap-title">Contribution to the score (SHAP, log-odds)</span>
          <span className="mr__shap-legend note">
            <span>
              <i className="mr__key mr__key--solid" />
              pushed the score higher (right of the line)
            </span>
            <span>
              <i className="mr__key mr__key--outline" />
              pushed it lower (left of the line)
            </span>
          </span>
        </div>
        <ul className="mr__bars">
          {bars.map((b) => (
            <li key={b.feature} className="mr__bar-row">
              <span className="mr__bar-label">{b.label}</span>
              <span className="mr__lane mr__lane--lower">
                {b.side === "lower" && (
                  <>
                    <span className="mono mr__bar-value">{formatSigned(b.contribution)}</span>
                    <span className="mr__bar mr__bar--outline" style={{ width: `${b.widthPct}%` }} />
                  </>
                )}
              </span>
              <span className="mr__zero" />
              <span className="mr__lane mr__lane--higher">
                {b.side === "higher" && (
                  <>
                    <span className="mr__bar mr__bar--solid" style={{ width: `${b.widthPct}%` }} />
                    <span className="mono mr__bar-value">{formatSigned(b.contribution)}</span>
                  </>
                )}
              </span>
            </li>
          ))}
        </ul>
        <p className="note">
          Method: {explanation.method}. The contributions add up to the score. This is a single
          method, so there is no second method to compare it against. {explanation.caveat}
        </p>
      </div>
    </section>
  );
}

// ------------------------------------------------------------------ vector

function VectorSection({ result }: { result: MriResult }) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const fv = result.feature_vector;
  if (!fv) return null;
  const attributions = result.explanation?.feature_attributions ?? {};
  const outside = new Set(fv.outside_training_range);

  async function copy(): Promise<void> {
    if (!fv) return;
    await navigator.clipboard.writeText(vectorJson(fv.names, fv.harmonised));
    setCopied(true);
  }

  return (
    <section className="card" aria-label="Feature vector">
      <div className="mr__section-head">
        <h2 className="mr__h2">Feature vector ({fv.names.length} features, harmonised)</h2>
        <span className="mr__actions">
          <button type="button" className="mr__link" onClick={() => void copy()}>
            {copied ? "Copied" : "Copy as JSON"}
          </button>
          <button type="button" className="mr__link" onClick={() => setOpen(!open)}>
            {open ? "Hide table" : "Show table"}
          </button>
        </span>
      </div>
      <p className="note">
        This is the MRI module&rsquo;s embedding for the fusion model: every input by name, after
        the scanner adjustment (ComBat batch: {Object.values(fv.scanner_batch).join(", ")}).
        {outside.size > 0 && ` ${outside.size} feature(s) fall outside the training range (marked).`}
      </p>
      {open && (
        <div className="mr__table-wrap">
          <table className="mr__table">
            <thead>
              <tr>
                <th>Feature</th>
                <th>Raw</th>
                <th>Harmonised</th>
                <th>Contribution</th>
              </tr>
            </thead>
            <tbody>
              {fv.names.map((name, i) => (
                <tr key={name} className={outside.has(name) ? "mr__row--outside" : undefined}>
                  <td title={name}>{fv.labels[i]}</td>
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
      )}
    </section>
  );
}

// ------------------------------------------------------------------ caveats

function CaveatsSection({ result }: { result: MriResult }) {
  const c = caveats(result);
  return (
    <section className="card" aria-label="Caveats">
      <h2 className="mr__h2">Caveats — read before acting on this result</h2>
      <div className="mr__caveats">
        <div className="mr__caveat">
          <p className="mr__th">Training population</p>
          <p className="mr__text">{c.population ?? "The model card did not report who it was trained on."}</p>
        </div>
        <div className="mr__caveat">
          <p className="mr__th">Known limits</p>
          {c.limits.length > 0 ? (
            <ul className="mr__list">
              {c.limits.map((l) => (
                <li key={l}>{l}</li>
              ))}
            </ul>
          ) : (
            <p className="mr__text">None reported.</p>
          )}
        </div>
        {c.outsideRange.length > 0 ? (
          <div className="mr__caveat mr__caveat--warn" role="note">
            <p className="mr__warn-title">Parts of this scan are outside what the model saw</p>
            <p className="mr__text">
              {c.outsideRange.length} measurement{c.outsideRange.length === 1 ? "" : "s"} fall
              outside the training range. The score may be less reliable than its interval
              suggests.
            </p>
            <ul className="mr__list">
              {c.outsideRange.map((o) => (
                <li key={o.name}>{o.label}</li>
              ))}
            </ul>
          </div>
        ) : (
          <div className="mr__caveat mr__caveat--calm">
            <p className="mr__th">Training range</p>
            <p className="mr__text">
              No measurement of this scan fell outside the range the model was trained on. That
              does not make the score reliable; it only means nothing was extrapolated.
            </p>
          </div>
        )}
      </div>
    </section>
  );
}
