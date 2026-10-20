import { useEffect, useState } from "react";
import { listMriAnalyses } from "../api/mri.api";
import { InfoTip } from "../components/InfoTip";
import { PageHeader } from "../components/PageHeader";
import type { MriAnalysisStatus } from "../mri/mriTypes";
import {
  STEP_LABELS,
  greeting,
  homeSummary,
  parseBackendTime,
  statusLabel,
} from "../mri/mriView";
import "./Home.css";

// First screen after sign-in, from the approved design's "Home". The design's metrics
// (priorities, decisions, referrals, a monthly summary) belong to the combined result, which
// is not built. The layout stays; the numbers are the real ones for today: this clinician's
// assessments, each part separate. Gait, tapping and voice are not processed in this build,
// so they show "—". Nothing here is a placeholder number.

const RECENT_COUNT = 8;

export function Home({
  clinician,
  onNewAssessment,
  onOpenAnalysis,
  onSeeAll,
}: {
  clinician: string;
  onNewAssessment: () => void;
  onOpenAnalysis: (id: string) => void;
  onSeeAll: () => void;
}) {
  const [items, setItems] = useState<MriAnalysisStatus[] | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    listMriAnalyses(100)
      .then((list) => !cancelled && setItems(list.items))
      .catch(() => !cancelled && setFailed(true));
    return () => {
      cancelled = true;
    };
  }, []);

  const now = new Date();
  const summary = items ? homeSummary(items) : null;

  return (
    <div className="home">
      <PageHeader
        title={`${greeting(now)}, ${clinician}`}
        subtitle={now.toLocaleDateString(undefined, {
          weekday: "long",
          day: "numeric",
          month: "long",
          year: "numeric",
        })}
        actions={
          <button type="button" className="btn btn--primary" onClick={onNewAssessment}>
            + New assessment
          </button>
        }
      />

      {failed && (
        <p className="home__error" role="alert">
          Could not load your analyses. Check the backend connection in the top bar.
        </p>
      )}

      <div className="home__layout">
        <div className="home__main">
          {summary && summary.inProgress.length > 0 && (
            <section className="card home__progress" aria-label="In progress">
              <p className="eyebrow">In progress</p>
              {summary.inProgress.map((item) => (
                <div key={item.id} className="home__progress-row">
                  <div>
                    <p className="home__progress-title">
                      {item.subject_label ? `${item.subject_label} · ` : ""}MRI part running — step {Math.min(item.done.length + 1, item.steps.length)} of{" "}
                      {item.steps.length}
                    </p>
                    <p className="note">
                      {item.current_step ? (STEP_LABELS[item.current_step] ?? item.current_step) : "Starting"}
                      {item.note ? ` · ${item.note}` : ""}
                    </p>
                  </div>
                  <button type="button" className="btn btn--outline" onClick={() => onOpenAnalysis(item.id)}>
                    Open →
                  </button>
                </div>
              ))}
            </section>
          )}

          <section className="card" aria-label="Recent assessments">
            <div className="home__head">
              <h2 className="card__title">Recent assessments</h2>
              <button type="button" className="home__link" onClick={onSeeAll}>
                All assessments
              </button>
            </div>
            {items === null && !failed && <p className="note">Loading…</p>}
            {items !== null && items.length === 0 && (
              <p className="note">None yet. Start one with “New assessment”.</p>
            )}
            {items !== null && items.length > 0 && (
              <table className="home__table">
                <thead>
                  <tr>
                    <th>Person</th>
                    <th>Started</th>
                    <th>Gait</th>
                    <th>Tapping</th>
                    <th>Voice</th>
                    <th>MRI</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {items.slice(0, RECENT_COUNT).map((item) => {
                    const label = statusLabel(item.status);
                    return (
                      <tr key={item.id}>
                        <td className="mono">{item.subject_label ?? "No ID"}</td>
                        <td className="mono">{parseBackendTime(item.created_at).toLocaleString()}</td>
                        <td className="home__none" title="Not in this build yet">—</td>
                        <td className="home__none" title="Not in this build yet">—</td>
                        <td className="home__none" title="Not in this build yet">—</td>
                        <td>
                          <span className={`mri__tag mri__tag--${label.tone}`}>{label.text}</span>
                          {item.eddy_profile === "fast" && <span className="mri__tag"> Fast DTI</span>}
                        </td>
                        <td>
                          <button type="button" className="home__link" onClick={() => onOpenAnalysis(item.id)}>
                            {item.status === "queued" || item.status === "running" ? "Open →" : "View →"}
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </section>
        </div>

        <aside className="home__rail">
          <section className="card">
            <p className="eyebrow">
              Your assessments
              <InfoTip label="About these counts">
                {summary ? (summary.counted >= 100 ? "Counted over your 100 most recent." : "All of them.") : ""}{" "}
                Each counts the MRI part. A “not scored” result means the model declined to score it
                (for example, an unseen scanner model); it is not a normal result. Gait, tapping and
                voice are not processed in this build.
              </InfoTip>
            </p>
            {summary ? (
              <>
                <p className="home__big">{summary.counted}</p>
                <ul className="home__counts">
                  <li><span>Scored</span><span className="mono">{summary.scored}</span></li>
                  <li><span>Not scored</span><span className="mono">{summary.notScored}</span></li>
                  <li><span>In progress</span><span className="mono">{summary.inProgress.length}</span></li>
                  <li><span>Failed</span><span className="mono">{summary.failed}</span></li>
                </ul>
              </>
            ) : (
              <p className="note">{failed ? "Unavailable." : "Loading…"}</p>
            )}
          </section>

          <section className="card">
            <p className="eyebrow">About this tool</p>
            <p className="home__text">
              A screening and triage aid for prioritising specialist review. It does not diagnose, and a
              lower score is never a clearance. Each part you upload is processed and shown on its own;
              nothing is combined into one result.
            </p>
          </section>
        </aside>
      </div>
    </div>
  );
}
