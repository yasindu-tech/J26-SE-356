import { useEffect, useState } from "react";
import { listGaitAnalyses } from "../api/gait.api";
import { listMriAnalyses } from "../api/mri.api";
import { InfoTip } from "../components/InfoTip";
import { PageHeader } from "../components/PageHeader";
import type { GaitAnalysisStatus } from "../gait/gaitTypes";
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
// analyses, each part separate: an MRI analysis and a gait analysis are separate rows. Tapping
// and voice are not processed in this build, so they show "—". Nothing here is a placeholder number.

const RECENT_COUNT = 8;

export function Home({
  clinician,
  onNewAssessment,
  onOpenAnalysis,
  onOpenGait,
  onSeeAll,
}: {
  clinician: string;
  onNewAssessment: () => void;
  onOpenAnalysis: (id: string) => void;
  onOpenGait: (id: string) => void;
  onSeeAll: () => void;
}) {
  const [items, setItems] = useState<MriAnalysisStatus[] | null>(null);
  const [gaitItems, setGaitItems] = useState<GaitAnalysisStatus[] | null>(null);
  const [failed, setFailed] = useState(false);
  const [gaitFailed, setGaitFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    listMriAnalyses(100)
      .then((list) => !cancelled && setItems(list.items))
      .catch(() => !cancelled && setFailed(true));
    listGaitAnalyses(100)
      .then((list) => !cancelled && setGaitItems(list.items))
      .catch(() => !cancelled && setGaitFailed(true)); // the MRI rows still show
    return () => {
      cancelled = true;
    };
  }, []);

  const now = new Date();
  const mriSummary = items ? homeSummary(items) : null;
  const gaitList = gaitItems ?? [];
  const gaitRunning = gaitList.filter((g) => g.status === "queued" || g.status === "running");
  const summary =
    mriSummary === null
      ? null
      : {
          counted: mriSummary.counted + gaitList.length,
          scored: mriSummary.scored + gaitList.filter((g) => g.status === "scored").length,
          notScored: mriSummary.notScored + gaitList.filter((g) => g.status === "abstained").length,
          failed: mriSummary.failed + gaitList.filter((g) => g.status === "failed").length,
          inProgress: mriSummary.inProgress.length + gaitRunning.length,
        };
  // MRI and gait analyses are separate rows, newest first.
  const rows = [
    ...(items ?? []).map((item) => ({ kind: "mri" as const, item, at: parseBackendTime(item.created_at).getTime() })),
    ...gaitList.map((item) => ({ kind: "gait" as const, item, at: parseBackendTime(item.created_at).getTime() })),
  ].sort((a, b) => b.at - a.at);

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
      {!failed && gaitFailed && (
        <p className="home__error" role="alert">
          Could not load your gait analyses, so they are not listed or counted below.
        </p>
      )}

      <div className="home__layout">
        <div className="home__main">
          {summary && summary.inProgress > 0 && (
            <section className="card home__progress" aria-label="In progress">
              <p className="eyebrow">In progress</p>
              {gaitRunning.map((item) => (
                <div key={item.id} className="home__progress-row">
                  <div>
                    <p className="home__progress-title">
                      {item.subject_label ? `${item.subject_label} · ` : ""}Gait part running
                    </p>
                    <p className="note">
                      {item.status === "queued" ? "Waiting for its turn" : "Reading the walk"} · usually about a minute
                    </p>
                  </div>
                  <button type="button" className="btn btn--outline" onClick={() => onOpenGait(item.id)}>
                    Open →
                  </button>
                </div>
              ))}
              {(mriSummary?.inProgress ?? []).map((item) => (
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
            {items !== null && rows.length === 0 && (
              <p className="note">None yet. Start one with “New assessment”.</p>
            )}
            {items !== null && rows.length > 0 && (
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
                  {rows.slice(0, RECENT_COUNT).map((row) => {
                    const label = statusLabel(row.item.status);
                    const running = row.item.status === "queued" || row.item.status === "running";
                    const tag = <span className={`mri__tag mri__tag--${label.tone}`}>{label.text}</span>;
                    const none = (
                      <td className="home__none" title="Not in this build yet">
                        —
                      </td>
                    );
                    return (
                      <tr key={`${row.kind}-${row.item.id}`}>
                        <td className="mono">
                          {row.item.subject_label ?? "No ID"}
                        </td>
                        <td className="mono">{parseBackendTime(row.item.created_at).toLocaleString()}</td>
                        {row.kind === "gait" ? <td>{tag}</td> : <td className="home__none" title="Not added to this analysis">—</td>}
                        {none}
                        {none}
                        {row.kind === "mri" ? (
                          <td>
                            {tag}
                            {row.item.eddy_profile === "fast" && <span className="mri__tag"> Fast DTI</span>}
                          </td>
                        ) : (
                          <td className="home__none" title="Not added to this analysis">—</td>
                        )}
                        <td>
                          <button
                            type="button"
                            className="home__link"
                            onClick={() => (row.kind === "mri" ? onOpenAnalysis(row.item.id) : onOpenGait(row.item.id))}
                          >
                            {running ? "Open →" : "View →"}
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
                Each counts one analysis, an MRI scan or a walking video. A “not scored” result means
                the model declined to score it (for example, an unseen scanner model or a video it
                could not read); it is not a normal result. Tapping and voice are not processed in
                this build.
              </InfoTip>
            </p>
            {summary ? (
              <>
                <p className="home__big">{summary.counted}</p>
                <ul className="home__counts">
                  <li><span>Scored</span><span className="mono">{summary.scored}</span></li>
                  <li><span>Not scored</span><span className="mono">{summary.notScored}</span></li>
                  <li><span>In progress</span><span className="mono">{summary.inProgress}</span></li>
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
