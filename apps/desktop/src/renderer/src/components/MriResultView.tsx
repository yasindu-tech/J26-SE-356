import type { MriResult } from "../mri/mriTypes";
import {
  FAST_EDDY_NOTICE,
  caveats,
  eddyProfileOf,
  missingRows,
  modelSetLabel,
  statusLabel,
} from "../mri/mriView";
import { plainDrivers, plainReading, reliabilityLine } from "../mri/mriPlain";
import { buildMriReportHtml, downloadHtml, reportFileName } from "../reports/mriReport";
import "./MriResultView.css";

// The MRI module's result for a general clinician: a plain reading, how far to lean on the
// model, what mattered most, and what to keep in mind. The numbers (raw log-odds score and
// interval, SHAP contributions, feature vector, AUC with its confidence interval, quality
// checks) are in the technical report the clinician can download from this page.
//
// It shows no High/Moderate/Low priority and no percentage: the score is not calibrated and
// the fusion step that sets a priority is not built. A lower reading is never a clearance.

export function MriResultView({
  result,
  label = null,
  onNewAnalysis,
}: {
  result: MriResult;
  /** The pseudonymous ID of this assessment, for the report. */
  label?: string | null;
  onNewAnalysis?: () => void;
}) {
  const fast = eddyProfileOf(result) === "fast";

  function download(): void {
    const now = new Date();
    downloadHtml(reportFileName(label, now), buildMriReportHtml(result, { label, generatedAt: now }));
  }

  return (
    <div className="mr">
      {fast && (
        <p className="mr__notice" role="note">
          {FAST_EDDY_NOTICE}
        </p>
      )}

      {result.status === "scored" && result.score ? (
        <ReadingCard result={result} onDownload={download} />
      ) : (
        <NotScoredCard result={result} onDownload={download} />
      )}

      <PartsRow result={result} />
      <MissingBlock result={result} onNewAnalysis={onNewAnalysis} />
      {result.status === "scored" && result.explanation && <MatteredCard result={result} />}
      <KeepInMind result={result} />
      <p className="note">{result.disclaimer}</p>
    </div>
  );
}

// ------------------------------------------------------------------ reading

function DownloadButton({ onDownload }: { onDownload: () => void }) {
  return (
    <div className="mr__download">
      <button type="button" className="btn btn--outline" onClick={onDownload}>
        Download technical report
      </button>
      <span className="note">
        The score, intervals, model test results, measurements and quality checks, for your records
        or a colleague who wants the detail. Saved to your computer; nothing is sent.
      </span>
    </div>
  );
}

function ReadingCard({ result, onDownload }: { result: MriResult; onDownload: () => void }) {
  if (!result.score) return null;
  const reading = plainReading(result.score);
  return (
    <section className="card mr__reading-card" aria-label="Reading">
      <p className="eyebrow">MRI · {modelSetLabel(result.model_set)} · one part of the assessment</p>
      <h2 className={`mr__reading-head mr__reading-head--${reading.lean}`}>{reading.headline}</h2>
      <p className="mr__reading-detail">{reading.detail}</p>

      <div className="mr__rely">
        <p className="eyebrow">How far to lean on it</p>
        <p className="mr__text">{reliabilityLine(result.model_card)}</p>
      </div>

      <p className="note">
        This is a screening aid for prioritising specialist review. It is not a diagnosis and it
        gives no referral priority: the reading comes from one part of the platform, and the step
        that combines parts is not built yet.
      </p>
      <DownloadButton onDownload={onDownload} />
    </section>
  );
}

function NotScoredCard({ result, onDownload }: { result: MriResult; onDownload: () => void }) {
  const label = statusLabel(result.status);
  return (
    <section className="card mr__reading-card mr__score--none" aria-label="Reading">
      <p className="eyebrow">MRI · one part of the assessment</p>
      <h2 className="mr__reading-head">{label.text}</h2>
      <p className="mr__big mono mr__big--none">—</p>
      {result.status === "abstained" && (
        <p className="mr__reading-detail">
          The model did not score this scan: {result.abstain_reason}. That is deliberate, because a
          guess from unusable input would be worse than no reading. It is{" "}
          <strong>not counted as zero</strong> and not read as normal.
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
      <DownloadButton onDownload={onDownload} />
    </section>
  );
}

// -------------------------------------------------------------------- parts

function PartsRow({ result }: { result: MriResult }) {
  return (
    <section className="mr__parts" aria-label="What was used">
      {(["t1", "dti"] as const).map((part) => {
        const used = result.availability[part] ?? false;
        const found = result.inputs[part]?.found ?? false;
        return (
          <div key={part} className={`mr__part${used ? "" : " mr__part--none"}`}>
            <div className="mr__part-head">
              <span className="mr__part-title">{part === "t1" ? "T1 scan" : "DTI scan"}</span>
              <span className={`mr__chip ${used ? "mr__chip--on" : "mr__chip--none"}`}>
                {used ? "Used" : found ? "Not used" : "Not in the upload"}
              </span>
            </div>
            {!used && (
              <p className="note">
                <span className="mono">—</span> Not counted as zero.
              </p>
            )}
          </div>
        );
      })}
    </section>
  );
}

function MissingBlock({
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
      <h2 className="mr__h2">What was not used, and what that means</h2>
      <ul className="mr__list">
        {rows.map((row) => (
          <li key={row.key}>
            <strong>{row.what}</strong> — {row.detail.toLowerCase()}. {row.effect}
          </li>
        ))}
      </ul>
      {onNewAnalysis && (
        <div>
          <button type="button" className="btn btn--outline mr__small-btn" onClick={onNewAnalysis}>
            Start a new assessment
          </button>
        </div>
      )}
    </section>
  );
}

// ------------------------------------------------------------------ mattered

function MatteredCard({ result }: { result: MriResult }) {
  const drivers = plainDrivers(result.explanation?.top_drivers ?? []);
  if (drivers.length === 0) return null;
  return (
    <section className="card" aria-label="What mattered most">
      <h2 className="mr__h2">What mattered most</h2>
      <ul className="mr__list mr__list--large">
        {drivers.map((d) => (
          <li key={d.key}>{d.text}</li>
        ))}
      </ul>
      <p className="note">
        These show what moved the model&rsquo;s reading. They are not causes and not findings about
        this person&rsquo;s brain. The full list, with the numbers, is in the technical report.
      </p>
    </section>
  );
}

// ---------------------------------------------------------------- keep in mind

function KeepInMind({ result }: { result: MriResult }) {
  const c = caveats(result);
  const items = [
    ...(c.population ? [c.population] : []),
    ...c.limits,
    ...(c.outsideRange.length > 0
      ? [
          `${c.outsideRange.length} measurement${c.outsideRange.length === 1 ? " was" : "s were"} outside what the model was trained on, so the reading may be less reliable than it looks.`,
        ]
      : []),
  ];
  return (
    <section className="card" aria-label="Keep in mind">
      <h2 className="mr__h2">Keep in mind</h2>
      {items.length > 0 ? (
        <ul className="mr__list">
          {items.map((text) => (
            <li key={text}>{text}</li>
          ))}
        </ul>
      ) : (
        <p className="mr__text">The model card reported no limits for this result.</p>
      )}
    </section>
  );
}
