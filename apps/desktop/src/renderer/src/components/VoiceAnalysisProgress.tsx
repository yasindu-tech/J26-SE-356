import { useEffect, useState } from "react";
import { ANALYSIS_STEPS, progressView } from "../voice/analysisProgress";
import "./VoiceAnalysisProgress.css";

// The progress card shown while a recording is uploaded and analysed. The upload part is
// measured; the analysis steps are an estimate (see voice/analysisProgress.ts), and the
// card says so.

const TICK_MS = 100;

export function VoiceAnalysisProgress({
  fileName,
  uploaded,
  analysingSince,
}: {
  fileName: string;
  /** Measured share of the file sent, 0 to 1. */
  uploaded: number;
  /** Date.now() when the upload finished, or null while it is still uploading. */
  analysingSince: number | null;
}) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), TICK_MS);
    return () => window.clearInterval(timer);
  }, []);

  const view =
    analysingSince === null
      ? progressView({ phase: "uploading", uploaded })
      : progressView({ phase: "analysing", elapsedS: (now - analysingSince) / 1000 });
  const percent = Math.round(view.fraction * 100);
  const elapsedS = analysingSince === null ? null : Math.max(0, (now - analysingSince) / 1000);

  return (
    <section className="card vap" aria-label="Analysing the recording">
      <div className="vap__head">
        <div>
          <p className="eyebrow">Analysing</p>
          <p className="vap__file">{fileName}</p>
        </div>
        <span className="vap__percent mono">{percent}%</span>
      </div>

      <div
        className="vap__track"
        role="progressbar"
        aria-label="Analysis progress"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent}
        aria-valuetext={`${ANALYSIS_STEPS[view.step]}, about ${percent}%`}
      >
        <div className="vap__bar" style={{ width: `${percent}%` }} />
      </div>

      <ol className="vap__steps">
        {ANALYSIS_STEPS.map((label, index) => {
          const state = index < view.step ? "done" : index === view.step ? "now" : "next";
          return (
            <li key={label} className={`vap__step vap__step--${state}`}>
              <span className="vap__mark" aria-hidden="true">
                {state === "done" ? "✓" : ""}
              </span>
              <span>
                {label}
                {index === 0 && state === "now" && ` · ${Math.round(uploaded * 100)}%`}
              </span>
            </li>
          );
        })}
      </ol>

      <p className="note">
        {view.estimated
          ? `Usually takes a few seconds${elapsedS !== null ? ` · ${elapsedS.toFixed(0)} s so far` : ""}. The steps after the upload are an estimate: the backend sends the result when it is done.`
          : "Sending the recording to the backend."}
      </p>
    </section>
  );
}
