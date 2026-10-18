import { useEffect, useState } from "react";
import { listMriAnalyses } from "../api/mri.api";
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

// First screen after sign-in, from the approved design's "Home". The design's Home belongs
// to the full assessment platform (patients, priorities, referrals); none of that exists
// yet, so this page shows only what is real today: the clinician's own MRI analyses and
// the way into a new one. Nothing here is a placeholder number.

const RECENT_COUNT = 8;

export function Home({
  clinician,
  onNewAnalysis,
  onOpenAnalysis,
  onSeeAll,
}: {
  clinician: string;
  onNewAnalysis: () => void;
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
          <button type="button" className="btn btn--primary" onClick={onNewAnalysis}>
            + New MRI analysis
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
                      MRI analysis running — step {Math.min(item.done.length + 1, item.steps.length)} of{" "}
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

          <section className="card" aria-label="Recent analyses">
            <div className="home__head">
              <h2 className="card__title">Recent analyses</h2>
              <button type="button" className="home__link" onClick={onSeeAll}>
                All analyses
              </button>
            </div>
            {items === null && !failed && <p className="note">Loading…</p>}
            {items !== null && items.length === 0 && (
              <p className="note">None yet. Start one with “New MRI analysis”.</p>
            )}
            {items !== null && items.length > 0 && (
              <table className="home__table">
                <thead>
                  <tr>
                    <th>Started</th>
                    <th>Status</th>
                    <th>DTI motion correction</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {items.slice(0, RECENT_COUNT).map((item) => {
                    const label = statusLabel(item.status);
                    return (
                      <tr key={item.id}>
                        <td className="mono">{parseBackendTime(item.created_at).toLocaleString()}</td>
                        <td>
                          <span className={`mri__tag mri__tag--${label.tone}`}>{label.text}</span>
                        </td>
                        <td>{item.eddy_profile === "fast" ? "Fast (not validated)" : "Full (standard)"}</td>
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
            <p className="eyebrow">Your analyses</p>
            {summary ? (
              <>
                <p className="home__big">{summary.counted}</p>
                <ul className="home__counts">
                  <li><span>Scored</span><span className="mono">{summary.scored}</span></li>
                  <li><span>Not scored</span><span className="mono">{summary.notScored}</span></li>
                  <li><span>In progress</span><span className="mono">{summary.inProgress.length}</span></li>
                  <li><span>Failed</span><span className="mono">{summary.failed}</span></li>
                </ul>
                <p className="note">
                  {summary.counted >= 100 ? "Counted over your 100 most recent." : "All of your analyses."} A
                  “not scored” result means the model declined to score it (for example, an unseen
                  scanner model); it is not a normal result.
                </p>
              </>
            ) : (
              <p className="note">{failed ? "Unavailable." : "Loading…"}</p>
            )}
          </section>

          <section className="card">
            <p className="eyebrow">About this tool</p>
            <p className="home__text">
              A screening and triage aid for prioritising specialist review. It does not diagnose, and a
              lower score is never a clearance.
            </p>
          </section>
        </aside>
      </div>
    </div>
  );
}
