import type { GaitResult } from "../gait/gaitTypes";
import {
  RELIABILITY_LINE,
  auditLine,
  checkRows,
  classBars,
  driverRows,
  isScored,
  keepInMind,
  notices,
  plainGaitReading,
  shapBars,
  timelineBar,
} from "../gait/gaitView";
import "./GaitResultView.css";

// The gait result for a general clinician, from the approved screens: a plain reading, the three
// model scores drawn to scale (no numbers), what mattered most with a bar chart, and on the
// side which part of the video was scored, the video checks and what to keep in mind.
//
// It shows no High/Moderate/Low priority and no percentage: the scores are not calibrated and
// the fusion step that sets a priority is not built. A lower reading is never a clearance.

export function GaitResultView({
  result,
  videoSeconds = null,
  onNewVideo,
}: {
  result: GaitResult;
  /** Length of the video, when the picker read it; the scored-stretch bar needs it. */
  videoSeconds?: number | null;
  /** Shown on a result that was not scored, so the clinician can choose another video. */
  onNewVideo?: () => void;
}) {
  const scored = isScored(result);
  return (
    <div className="gr">
      <div className="mri__layout">
        <div className="mri__main">
          {scored ? <ReadingCard result={result} /> : <NotScoredCard result={result} onNewVideo={onNewVideo} />}
          {scored ? <MatteredCard result={result} /> : <TipsCard />}
          {scored && <p className="note mono">{auditLine(result)}</p>}
        </div>
        <aside className="mri__rail">
          {scored && <TimelineCard result={result} videoSeconds={videoSeconds} />}
          <ChecksCard result={result} />
          {scored ? <KeepInMind result={result} /> : <VideoCard />}
        </aside>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ reading

function ReadingCard({ result }: { result: GaitResult }) {
  const reading = plainGaitReading(result);
  const bars = classBars(result);
  return (
    <section className="card gr__reading-card" aria-label="Reading">
      <p className="eyebrow">Gait · one part of the assessment</p>
      <h2 className={`gr__reading-head gr__reading-head--${reading.kind}`}>{reading.headline}</h2>
      <p className="gr__reading-detail">{reading.detail}</p>

      <div className="gr__bars" role="img" aria-label={barsDescription(result)}>
        {bars.map((bar) => (
          <div key={bar.key} className="gr__bar-row">
            <span className={`gr__bar-label${bar.isPick && reading.kind === "clear" ? " gr__bar-label--pick" : ""}`}>
              {bar.label}
              <b>{bar.rating}</b>
            </span>
            <span className="gr__bar-track">
              <span
                className={`gr__bar-fill${bar.isPick && reading.kind === "clear" ? " gr__bar-fill--pick" : ""}`}
                style={{ width: `${bar.widthPercent}%` }}
              />
            </span>
          </div>
        ))}
        <p className="note">The model&rsquo;s scores for each level, drawn to scale. They are not chances.</p>
      </div>

      {notices(result).map((notice) => (
        <div key={notice.title + notice.text} className="gr__notice" role="note">
          <span className="gr__notice-title">{notice.title}</span>
          <span>{notice.text}</span>
        </div>
      ))}

      <div className="gr__rely">
        <p className="eyebrow">How far to lean on it</p>
        <p className="gr__text">{RELIABILITY_LINE}</p>
      </div>
      <p className="note">
        This is a screening aid for prioritising specialist review. It is not a diagnosis and it gives
        no referral priority: the reading comes from one part of the platform, and the step that
        combines parts is not built yet.
      </p>
    </section>
  );
}

function barsDescription(result: GaitResult): string {
  const reading = plainGaitReading(result);
  if (reading.kind === "none") return "Model scores for the three levels, drawn to scale. No level stands out.";
  const pick = classBars(result).find((b) => b.isPick);
  return `Model scores for the three levels, drawn to scale. Highest: ${pick ? pick.label.toLowerCase() : "none"}.`;
}

function NotScoredCard({ result, onNewVideo }: { result: GaitResult; onNewVideo?: () => void }) {
  return (
    <section className="card gr__reading-card gr__none" aria-label="Not scored">
      <div className="gr__none-head">
        <p className="eyebrow">Gait · one part of the assessment</p>
        <span className="mr__chip mr__chip--none">Not scored</span>
      </div>
      <h2 className="gr__reading-head">This video could not be scored</h2>
      <p className="gr__big mono">—</p>
      <p className="gr__reading-detail">
        The model gave no reading rather than guess from a walk it cannot read reliably. That is
        deliberate. It is <strong>not counted as zero</strong> and not read as normal.
      </p>
      {result.reasons.length > 0 && (
        <div className="gr__notice" role="note">
          <span className="gr__notice-title">Why</span>
          {result.reasons.map((reason) => (
            <span key={reason}>{reason}</span>
          ))}
        </div>
      )}
      {onNewVideo && (
        <div>
          <button type="button" className="btn btn--primary" onClick={onNewVideo}>
            Choose another video
          </button>
        </div>
      )}
    </section>
  );
}

function TipsCard() {
  return (
    <section className="card" aria-label="What to try next time">
      <h2 className="gr__h2">What to try next time</h2>
      <ul className="gr__list gr__list--large">
        <li>
          <strong>Film a longer walk.</strong> Let the person walk across the frame for at least 5
          seconds, ideally 10 or more, so several strides are seen.
        </li>
        <li>
          <strong>Keep the whole body in view.</strong> Head to feet, with nothing in front of the
          legs.
        </li>
        <li>
          <strong>Film from the side with the phone still.</strong> Held sideways, at the normal
          video setting, not slow motion.
        </li>
      </ul>
    </section>
  );
}

function VideoCard() {
  return (
    <section className="card" aria-label="What happens to the video">
      <h2 className="mri__rail-title">What happens to the video</h2>
      <p className="mri__body-text">
        It was deleted from the server when the analysis ended. This outcome is kept so the
        assessment shows what happened.
      </p>
    </section>
  );
}

// ------------------------------------------------------------------ mattered

function MatteredCard({ result }: { result: GaitResult }) {
  const rows = driverRows(result);
  if (rows.length === 0) return null;
  return (
    <section className="card" aria-label="What mattered most">
      <h2 className="gr__h2">What mattered most</h2>
      <ShapChart result={result} />
      <ul className="gr__drivers">
        {rows.map((row) => (
          <li key={row.key} className="gr__driver">
            <span className={`gr__tag${row.towards ? " gr__tag--towards" : " gr__tag--away"}`} aria-hidden="true" />
            <span className="gr__driver-body">
              <span className="gr__driver-tag">{row.tag}</span>
              <span className="gr__driver-text">{row.text}</span>
            </span>
          </li>
        ))}
      </ul>
      <p className="note">
        {plainGaitReading(result).kind === "none"
          ? "The scores were too close for a clear pattern, so these show what pulled towards the highest one. "
          : "These show what moved the model’s scores. "}
        They are not causes and not findings about this person&rsquo;s body.
      </p>
    </section>
  );
}

function ShapChart({ result }: { result: GaitResult }) {
  const bars = shapBars(result);
  if (bars.length === 0) return null;
  const description = bars.map((b) => `${b.label}: ${b.tag.toLowerCase()}`).join("; ");
  return (
    <div className="gr__shap" role="img" aria-label={`How much each input moved the score. ${description}.`}>
      {bars.map((bar) => (
        <div key={bar.key} className="gr__shap-row">
          <span className="gr__shap-label">{bar.label}</span>
          <span className="gr__shap-plot">
            <span
              className={`gr__shap-bar ${bar.towards ? "gr__shap-bar--towards" : "gr__shap-bar--away"}`}
              style={{ width: `${bar.widthPercent / 2}%` }}
            />
          </span>
        </div>
      ))}
      <div className="gr__shap-axis" aria-hidden="true">
        <span />
        <span className="gr__shap-axis-labels">
          <span>Away from the highest-scoring level</span>
          <span>Towards it</span>
        </span>
      </div>
      <p className="note">Bar length shows how much each input moved the score, drawn to scale.</p>
    </div>
  );
}

// ------------------------------------------------------------------- side

function TimelineCard({ result, videoSeconds }: { result: GaitResult; videoSeconds: number | null }) {
  const bar = timelineBar(result, videoSeconds);
  return (
    <section className="card" aria-label="Which part of the video was scored">
      <h2 className="mri__rail-title">Which part of the video was scored</h2>
      {bar ? (
        <>
          <div className="gr__timeline" role="img" aria-label={bar.text}>
            <span className="gr__timeline-fill" style={{ left: `${bar.leftPercent}%`, width: `${bar.widthPercent}%` }} />
          </div>
          <div className="gr__timeline-labels mono">
            <span>0 s</span>
            <span>{bar.endLabel}</span>
          </div>
          <p className="note">{bar.text} The rest was not used.</p>
        </>
      ) : result.walk ? (
        <p className="note">
          Scored {result.walk.seconds.toFixed(1)} s of walking, from {result.walk.start_s.toFixed(1)} s to{" "}
          {result.walk.end_s.toFixed(1)} s of the video.
        </p>
      ) : null}
    </section>
  );
}

function ChecksCard({ result }: { result: GaitResult }) {
  const rows = checkRows(result);
  if (rows.length === 0) return null;
  return (
    <section className="card" aria-label="Video checks">
      <h2 className="mri__rail-title">Video checks</h2>
      <ul className="gr__checks">
        {rows.map((row) => (
          <li key={row.key} className="gr__check">
            <span className="gr__check-label">{row.label}</span>
            <span className={`gr__chip ${row.ok ? "gr__chip--on" : "gr__chip--fail"}`}>
              {row.ok ? "Passed" : "Not met"}
            </span>
            <span className="gr__check-sub note">
              <span className="mono">{row.value}</span>
              <span>{row.need}</span>
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}

function KeepInMind({ result }: { result: GaitResult }) {
  return (
    <section className="card" aria-label="Keep in mind">
      <h2 className="mri__rail-title">Keep in mind</h2>
      <ul className="gr__list">
        {keepInMind(result).map((text) => (
          <li key={text}>{text}</li>
        ))}
      </ul>
    </section>
  );
}
