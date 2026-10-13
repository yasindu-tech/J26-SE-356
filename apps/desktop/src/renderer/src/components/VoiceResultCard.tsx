import {
  TRIAGE_THRESHOLD,
  triageLabel,
  type AbstainedVoiceResult,
  type ScoredVoiceResult,
  type ShapDriver,
  type VoiceResult,
} from "../voice/voiceResult";
import "./VoiceResultCard.css";

// One person's voice screening result (Desktop-App-Spec.md, "Result"). The
// score is never shown without its confidence interval, what was missing, and
// why the model reached it (apps/desktop/README.md).

const OTHER_MODALITIES = ["Gait", "Finger tapping", "MRI"] as const;

function formatScore(value: number): string {
  return value.toFixed(2);
}

function formatSigned(value: number): string {
  return `${value > 0 ? "+" : "−"}${Math.abs(value).toFixed(2)}`;
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

      {result.kind === "scored" && <DriversSection drivers={result.shapTop5} />}

      <section className="voice-card__section">
        <h3 className="voice-card__section-title">Limits of this result</h3>
        <ul className="voice-card__caveats">
          <li>{result.note}</li>
          {/* The voice-pp1 model is trained on UCI-470 only (models/voice/README.md, D1). */}
          <li>
            Trained on {result.nTraining} people from the UCI-470 voice dataset: people already
            known to have Parkinson&rsquo;s versus healthy controls, not an early-stage screening
            population.
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

function DriversSection({ drivers }: { drivers: ShapDriver[] }) {
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
          return (
            <li key={driver.feature} className="voice-card__driver">
              <span className="voice-card__driver-name" title={driver.feature}>
                {driver.feature}
              </span>
              <span className="voice-card__driver-track" aria-hidden="true">
                <span
                  className={`voice-card__driver-bar voice-card__driver-bar--${raises ? "up" : "down"}`}
                  style={raises ? { left: "50%", width } : { right: "50%", width }}
                />
              </span>
              <span className="voice-card__driver-value">
                {formatSigned(driver.shapLogOdds)} · {raises ? "raised" : "lowered"} the score
              </span>
            </li>
          );
        })}
      </ol>
      <p className="voice-card__caption">
        Model attributions (SHAP, in log-odds), largest first. They show what the model used, not
        what causes Parkinson&rsquo;s. Names are the dataset&rsquo;s technical voice measures.
      </p>
    </section>
  );
}
