import {
  TRIAGE_THRESHOLD,
  triageLabel,
  type AbstainedVoiceResult,
  type ExplainedReason,
  type Explanations,
  type ScoredVoiceResult,
  type ShapDriver,
  type VoiceResult,
} from "../voice/voiceResult";
import "./VoiceResultCard.css";

// One person's voice screening result (Desktop-App-Spec.md, "Result"). The
// score is never shown without its confidence interval, what was missing, and
// why the model reached it (apps/desktop/README.md).

const OTHER_MODALITIES = ["Gait", "Finger tapping", "MRI"] as const;

// Display names for the measure groups in models/voice/src/audio_features.py and uci_praat.py.
const GROUP_LABELS: Record<string, string> = {
  mfcc: "MFCC (sound colour)",
  jitter: "Jitter (pitch wobble)",
  shimmer: "Shimmer (loudness wobble)",
  noise: "Voice clarity",
  pitch: "Pitch",
  formants: "Formants",
  spectrum: "Spectrum",
  "loudness change": "Loudness change",
};

function formatScore(value: number): string {
  return value.toFixed(2);
}

function capitalise(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function formatSigned(value: number): string {
  return `${value > 0 ? "+" : "−"}${Math.abs(value).toFixed(2)}`;
}

/** Where the model's training data came from, from its pipeline version (models/voice/README.md). */
function trainingSource(pipelineVersion: string): string {
  if (pipelineVersion.startsWith("voice-audio-")) {
    return "from the Figshare telephone voice recordings";
  }
  if (pipelineVersion.startsWith("voice-uci-praat-")) {
    return "from the UCI-470 voice dataset, using only the 16 measures Praat can recompute from audio";
  }
  return "from the UCI-470 voice dataset";
}

export function VoiceResultCard({ result }: { result: VoiceResult }) {
  return (
    <article className="voice-card" aria-label={`Voice screening result for ${result.person}`}>
      <header className="voice-card__header">
        <div>
          <p className="voice-card__eyebrow">Voice screening · triage only</p>
          <h2 className="voice-card__title">Person {result.person}</h2>
        </div>
        <PriorityBadge result={result} />
      </header>

      {result.kind === "scored" ? (
        <ScorePanel result={result} />
      ) : (
        <AbstainedPanel result={result} />
      )}

      <section className="voice-card__section">
        <h3 className="voice-card__section-title">What this result is based on</h3>
        <ul className="voice-card__modalities">
          <li>
            <span>Voice</span>
            <span
              className={
                result.kind === "scored" ? "voice-card__status--used" : "voice-card__status--absent"
              }
            >
              {result.kind === "scored" ? "Used" : "Absent (failed quality check)"}
            </span>
          </li>
          {OTHER_MODALITIES.map((name) => (
            <li key={name}>
              <span>{name}</span>
              <span className="voice-card__status--absent">Not captured</span>
            </li>
          ))}
        </ul>
        <p className="voice-card__caption">
          {result.kind === "scored"
            ? "This estimate uses voice only. Gait, finger tapping and MRI were not captured and are not part of it."
            : "No input was usable, so there is no estimate. Missing voice is never counted as a normal result."}
        </p>
      </section>

      {result.kind === "scored" && (
        <DriversSection drivers={result.shapTop5} reasons={result.explanations?.topReasons} />
      )}

      {result.explanations && result.explanations.uncertainty.length > 0 && (
        <UncertaintySection reasons={result.explanations.uncertainty} />
      )}

      {result.explanations && <QualitySection explanations={result.explanations} />}

      {result.kind === "scored" && result.explanations && (
        <OverviewSection explanations={result.explanations} />
      )}

      <section className="voice-card__section">
        <h3 className="voice-card__section-title">Limits of this result</h3>
        <ul className="voice-card__caveats">
          <li>{result.note}</li>
          <li>
            Trained on {result.nTraining} people {trainingSource(result.pipelineVersion)}: people
            already known to have Parkinson&rsquo;s versus healthy controls, not an early-stage
            screening population.
          </li>
          <li>Fairness by sex: {result.sexFairnessNote}</li>
        </ul>
        <p className="voice-card__version">Model {result.pipelineVersion}</p>
      </section>

      {result.referenceLabel !== "-" && (
        <footer className="voice-card__demo">
          Demo check only: the dataset&rsquo;s own label for this hold-out person is &ldquo;
          {result.referenceLabel}&rdquo;. A real screening would not know this.
        </footer>
      )}
    </article>
  );
}

function PriorityBadge({ result }: { result: VoiceResult }) {
  if (result.kind === "abstained") {
    return <span className="voice-card__badge voice-card__badge--none">Not scored</span>;
  }
  const modifier = result.triagePriority === "prioritised_referral" ? "attention" : "calm";
  return (
    <span className={`voice-card__badge voice-card__badge--${modifier}`}>
      {triageLabel(result.triagePriority)}
    </span>
  );
}

function ScorePanel({ result }: { result: ScoredVoiceResult }) {
  return (
    <section className="voice-card__score">
      <div className="voice-card__score-numbers">
        <span className="voice-card__score-value">{formatScore(result.riskScore)}</span>
        <span className="voice-card__score-label">
          Risk score · 95% CI {formatScore(result.ciLower)}–{formatScore(result.ciUpper)}
        </span>
      </div>

      <div
        className="voice-card__range"
        role="img"
        aria-label={`Risk score ${formatScore(result.riskScore)}, range ${formatScore(result.ciLower)} to ${formatScore(result.ciUpper)}, referral threshold ${formatScore(TRIAGE_THRESHOLD)}`}
      >
        <div className="voice-card__range-track">
          <div
            className="voice-card__range-band"
            style={{
              left: `${result.ciLower * 100}%`,
              width: `${(result.ciUpper - result.ciLower) * 100}%`,
            }}
          />
          <div
            className="voice-card__range-threshold"
            style={{ left: `${TRIAGE_THRESHOLD * 100}%` }}
          />
          <div
            className="voice-card__range-marker"
            style={{ left: `${result.riskScore * 100}%` }}
          />
        </div>
        <div className="voice-card__range-scale" aria-hidden="true">
          <span>0</span>
          <span>Referral threshold {formatScore(TRIAGE_THRESHOLD)}</span>
          <span>1</span>
        </div>
      </div>

      <p className="voice-card__caption">
        The score is calibrated as if half of the people screened had Parkinson&rsquo;s, so it is
        not the chance that this person has it.
      </p>
    </section>
  );
}

function AbstainedPanel({ result }: { result: AbstainedVoiceResult }) {
  return (
    <section className="voice-card__abstained">
      <p className="voice-card__abstained-title">
        The voice input failed the quality check, so no score was produced.
      </p>
      <ul>
        {result.problems.map((problem) => (
          <li key={problem}>{problem}</li>
        ))}
      </ul>
    </section>
  );
}

function DriversSection({
  drivers,
  reasons,
}: {
  drivers: ShapDriver[];
  reasons?: ExplainedReason[];
}) {
  const largest = Math.max(
    ...drivers.map((driver) => Math.abs(driver.shapLogOdds)),
    Number.EPSILON,
  );
  return (
    <section className="voice-card__section">
      <h3 className="voice-card__section-title">What moved this score</h3>
      <ol className="voice-card__drivers">
        {drivers.map((driver) => {
          const raises = driver.shapLogOdds > 0;
          const width = `${(Math.abs(driver.shapLogOdds) / largest) * 50}%`;
          const reason = reasons?.find((r) => r.feature === driver.feature);
          return (
            <li key={driver.feature} className="voice-card__driver">
              {reason ? (
                <span className="voice-card__driver-meaning" title={driver.feature}>
                  {capitalise(reason.meaning)}
                </span>
              ) : (
                <span className="voice-card__driver-name" title={driver.feature}>
                  {driver.feature}
                </span>
              )}
              <span className="voice-card__driver-track" aria-hidden="true">
                <span
                  className={`voice-card__driver-bar voice-card__driver-bar--${raises ? "up" : "down"}`}
                  style={raises ? { left: "50%", width } : { right: "50%", width }}
                />
              </span>
              <span className="voice-card__driver-value">
                {formatSigned(driver.shapLogOdds)} · {raises ? "raised" : "lowered"} the score
              </span>
              {reason && <ReasonDetail reason={reason} />}
            </li>
          );
        })}
      </ol>
      <p className="voice-card__caption">
        Model attributions (SHAP, in log-odds), largest first. They show what the model used, not
        what causes Parkinson&rsquo;s.{" "}
        {reasons
          ? "Each measure's technical name is in brackets."
          : "Names are the dataset's technical voice measures."}
      </p>
    </section>
  );
}

function ReasonDetail({ reason }: { reason: ExplainedReason }) {
  return (
    <div className="voice-card__reason">
      <p className="voice-card__reason-sentence">{reason.sentence}</p>
      {reason.comparable ? (
        <div
          className="voice-card__reason-scale"
          role="img"
          aria-label={`Higher than ${Math.round(reason.percentileAmongHealthy)}% of healthy training voices`}
        >
          <span className="voice-card__reason-scale-label">Healthy training voices</span>
          <span className="voice-card__reason-scale-track">
            <span
              className="voice-card__reason-scale-marker"
              style={{ left: `${reason.percentileAmongHealthy}%` }}
            />
          </span>
          <span className="voice-card__reason-scale-ends" aria-hidden="true">
            <span>lowest</span>
            <span>highest</span>
          </span>
        </div>
      ) : (
        <p className="voice-card__reason-warning">
          Not comparable with the training voices: new recordings measure this differently.
        </p>
      )}
    </div>
  );
}

function UncertaintySection({ reasons }: { reasons: string[] }) {
  return (
    <section className="voice-card__section">
      <h3 className="voice-card__section-title">How sure is this result</h3>
      <ul className="voice-card__caveats">
        {reasons.map((reason) => (
          <li key={reason}>{reason}</li>
        ))}
      </ul>
    </section>
  );
}

function formatMeasured(measured: Explanations["quality"]["measured"]): [string, string][] {
  const rows: [string, string][] = [];
  if (measured.sourceFormat) rows.push(["File format", measured.sourceFormat.toUpperCase()]);
  if (measured.sourceSampleRate !== undefined) {
    rows.push(["Sample rate", `${(measured.sourceSampleRate / 1000).toFixed(1)} kHz`]);
  }
  if (measured.durationS !== undefined) {
    rows.push(["Recording length", `${measured.durationS.toFixed(1)} s`]);
  }
  if (measured.vowelS !== undefined) {
    rows.push(["Steady 'aaah' used", `${measured.vowelS.toFixed(1)} s`]);
  }
  rows.push([
    "Voice above background noise",
    measured.snrDb !== undefined ? `${measured.snrDb.toFixed(0)} dB` : "no background to measure",
  ]);
  if (measured.clippedFraction !== undefined) {
    rows.push(["Clipped (too loud)", `${(measured.clippedFraction * 100).toFixed(1)}%`]);
  }
  return rows;
}

function QualitySection({ explanations }: { explanations: Explanations }) {
  const { quality } = explanations;
  return (
    <section className="voice-card__section">
      <h3 className="voice-card__section-title">
        Recording quality ·{" "}
        <span className={quality.passed ? "voice-card__quality--ok" : "voice-card__quality--no"}>
          {quality.passed ? "passed" : "not usable"}
        </span>
      </h3>
      <dl className="voice-card__measured">
        {formatMeasured(quality.measured).map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
      {/* Refusal reasons are already listed at the top of a "Not scored" card. */}
    </section>
  );
}

function OverviewSection({ explanations }: { explanations: Explanations }) {
  const { modelOverview } = explanations;
  return (
    <section className="voice-card__section">
      <h3 className="voice-card__section-title">What the model relies on in general</h3>
      <ul className="voice-card__groups">
        {modelOverview.topGroups.map(({ group, share }) => (
          <li key={group}>
            <span className="voice-card__group-name">
              {GROUP_LABELS[group] ?? capitalise(group)}
            </span>
            <span className="voice-card__group-track" aria-hidden="true">
              <span className="voice-card__group-bar" style={{ width: `${share * 100}%` }} />
            </span>
            <span className="voice-card__group-share">{Math.round(share * 100)}%</span>
          </li>
        ))}
      </ul>
      <p className="voice-card__caption">
        Share of the model&rsquo;s attributions (SHAP) across all training voices, by measure group.
        This describes the model, not this person.
      </p>
    </section>
  );
}
