import { useState } from "react";
import type { MriDriver, MriResult, MriScore, MriSeriesInput } from "../mri/mriTypes";
import { SCORE_AXIS, axisPercent, formatSigned, readScore, statusLabel, vectorJson } from "../mri/mriView";

// The MRI module's output for one scan. It deliberately shows the raw model score
// (log-odds) with its 95% interval rather than a percentage: the score is not
// calibrated, and the fusion model will decide how MRI is combined with the other
// modalities. The feature vector is shown because it is what fusion will consume.

export function MriResultView({ result }: { result: MriResult }) {
  const label = statusLabel(result.status);
  return (
    <div className="mri-result">
      <section className="mri__panel">
        <div className="mri-result__header">
          <div>
            <p className="mri-result__eyebrow">MRI module output</p>
            <h2 className="mri__panel-title">
              {result.status === "scored"
                ? `Scored with the ${(result.model_set ?? "").toUpperCase()} model`
                : label.text}
            </h2>
          </div>
          <span className={`mri__tag mri__tag--${label.tone}`}>{label.text}</span>
        </div>
        <Availability result={result} />
        {result.status === "abstained" && (
          <p className="mri-result__notice">
            The module did not score this scan: {result.abstain_reason}. That is deliberate. A
            guess from unusable input would be worse than no score.
          </p>
        )}
        {result.status === "failed" && (
          <p className="mri__error" role="alert">
            The analysis failed: {result.error}
          </p>
        )}
        {result.validation && result.validation.problems.length > 0 && (
          <ul className="mri-result__problems">
            {result.validation.problems.map((p) => (
              <li key={p}>{p}</li>
            ))}
          </ul>
        )}
      </section>

      {result.score && <ScorePanel score={result.score} />}
      {result.explanation && <DriversPanel drivers={result.explanation.top_drivers} caveat={result.explanation.caveat} />}
      {result.feature_vector && <VectorPanel result={result} />}
      <InputsPanel inputs={result.inputs} />
      {result.model_card && <ModelCardPanel result={result} />}
      <p className="mri__muted mri__small">{result.disclaimer}</p>
    </div>
  );
}

function Availability({ result }: { result: MriResult }) {
  return (
    <div className="mri-result__chips">
      {(["t1", "dti"] as const).map((part) => {
        const used = result.availability[part] ?? false;
        const found = result.inputs[part]?.found ?? false;
        const text = used ? "used in the score" : found ? "found, not used" : "not in the upload";
        return (
          <span key={part} className={`mri-result__chip${used ? " mri-result__chip--on" : ""}`}>
            {part.toUpperCase()}: {text}
          </span>
        );
      })}
    </div>
  );
}

function ScorePanel({ score }: { score: MriScore }) {
  const reading = readScore(score);
  const [low, high] = score.interval_95;
  return (
    <section className="mri__panel">
      <h3 className="mri-result__section-title">Model score (raw, log-odds)</h3>
      <div className="mri-result__score-row">
        <span className="mri-result__score-value">{formatSigned(score.value)}</span>
        <span className="mri__muted">
          95% interval {formatSigned(low)} to {formatSigned(high)} · leans {reading.lean}
        </span>
      </div>
      <div className="mri-result__axis" aria-label="Score with 95% interval on the model's scale">
        <div className="mri-result__axis-zero" style={{ left: `${axisPercent(0)}%` }} />
        <div
          className="mri-result__axis-interval"
          style={{ left: `${axisPercent(low)}%`, width: `${axisPercent(high) - axisPercent(low)}%` }}
        />
        <div className="mri-result__axis-dot" style={{ left: `${axisPercent(score.value)}%` }} />
      </div>
      <div className="mri-result__axis-labels">
        <span>{SCORE_AXIS[0]} · leans control</span>
        <span>0 = boundary the model learned</span>
        <span>leans PD · +{SCORE_AXIS[1]}</span>
      </div>
      <p>{reading.sentence}</p>
      <p className="mri__muted mri__small">
        {score.meaning} Interval: {score.interval_method}. Trained on {score.n_training} people.
        Not a probability{score.calibrated ? "" : " and not calibrated"}: the fusion step will
        decide how this score is combined with gait, finger tapping and voice.
      </p>
    </section>
  );
}

function DriversPanel({ drivers, caveat }: { drivers: MriDriver[]; caveat: string }) {
  const largest = Math.max(1e-9, ...drivers.map((d) => Math.abs(d.contribution)));
  return (
    <section className="mri__panel">
      <h3 className="mri-result__section-title">What moved the score most</h3>
      <ul className="mri-result__drivers">
        {drivers.map((d) => (
          <li key={d.feature} className="mri-result__driver">
            <span className="mri-result__driver-label">{d.label}</span>
            <span className="mri-result__driver-track">
              <span
                className={`mri-result__driver-bar mri-result__driver-bar--${d.contribution > 0 ? "pd" : "control"}`}
                style={{ width: `${(Math.abs(d.contribution) / largest) * 100}%` }}
              />
            </span>
            <span className="mri-result__driver-value">
              {formatSigned(d.contribution)} {d.direction}
            </span>
          </li>
        ))}
      </ul>
      <p className="mri__muted mri__small">
        Contributions are exact TreeSHAP values in log-odds; they add up to the score. {caveat}
      </p>
    </section>
  );
}

function VectorPanel({ result }: { result: MriResult }) {
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
    <section className="mri__panel">
      <div className="mri-result__header">
        <h3 className="mri-result__section-title">
          Feature vector ({fv.names.length} features, harmonised)
        </h3>
        <div className="mri-result__vector-actions">
          <button type="button" className="mri__link-button" onClick={() => void copy()}>
            {copied ? "Copied" : "Copy as JSON"}
          </button>
          <button type="button" className="mri__link-button" onClick={() => setOpen(!open)}>
            {open ? "Hide table" : "Show table"}
          </button>
        </div>
      </div>
      <p className="mri__muted mri__small">
        This is the MRI module's embedding for the fusion model: every input by name, after the
        scanner adjustment (ComBat batch: {Object.values(fv.scanner_batch).join(", ")}).
        {outside.size > 0 && ` ${outside.size} feature(s) fall outside the training range (marked).`}
      </p>
      {open && (
        <div className="mri-result__table-wrap">
          <table className="mri-result__table">
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
                <tr key={name} className={outside.has(name) ? "mri-result__row--outside" : undefined}>
                  <td title={name}>{fv.labels[i]}</td>
                  <td>{fv.raw[i].toPrecision(4)}</td>
                  <td>{fv.harmonised[i].toPrecision(4)}</td>
                  <td>{attributions[name] !== undefined ? formatSigned(attributions[name], 3) : "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function InputsPanel({ inputs }: { inputs: Record<string, MriSeriesInput> }) {
  return (
    <section className="mri__panel">
      <h3 className="mri-result__section-title">Series used and quality checks</h3>
      <div className="mri-result__inputs">
        {(["t1", "dti"] as const).map((part) => {
          const s = inputs[part];
          return (
            <div key={part} className="mri-result__input">
              <p className="mri-result__input-title">{part.toUpperCase()}</p>
              {!s?.found ? (
                <p className="mri__muted">Not found in the upload.</p>
              ) : (
                <>
                  <p className="mri__small">
                    {s.series_description} · {s.manufacturer} {s.scanner_model}
                    {s.voxel_mm ? ` · ${s.voxel_mm.join(" × ")} mm` : ""}
                  </p>
                  {s.checks && (
                    <ul className="mri-result__checks">
                      {Object.entries(s.checks).map(([k, v]) => (
                        <li key={k}>
                          {k.replace(/_/g, " ")}: {typeof v === "number" ? v.toPrecision(4) : String(v)}
                        </li>
                      ))}
                    </ul>
                  )}
                  {s.qc_image && (
                    <img className="mri-result__qc" src={s.qc_image} alt={`${part} quality-check picture`} />
                  )}
                </>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}

function ModelCardPanel({ result }: { result: MriResult }) {
  const card = result.model_card;
  const cv = card?.cross_validated;
  return (
    <section className="mri__panel">
      <h3 className="mri-result__section-title">Model card</h3>
      <p className="mri__small">
        {card?.purpose} Model version {result.model_version}.
        {card?.training_people &&
          ` Trained on ${card.training_people.total} people (${card.training_people.pd} PD, ${card.training_people.control} control).`}
      </p>
      {cv && (
        <p className="mri__small">
          Cross-validated ({cv.folds} folds × {cv.repeats} repeats, people never seen in training):
          AUC {cv.auc_mean.toFixed(2)} (95% CI {cv.auc_ci[0].toFixed(2)}–{cv.auc_ci[1].toFixed(2)}),
          balanced accuracy {Math.round(cv.balanced_accuracy_mean * 100)}% (95% CI{" "}
          {Math.round(cv.balanced_accuracy_ci[0] * 100)}–{Math.round(cv.balanced_accuracy_ci[1] * 100)}%).
        </p>
      )}
      {card?.known_limits && (
        <ul className="mri-result__problems">
          {card.known_limits.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
      )}
    </section>
  );
}
