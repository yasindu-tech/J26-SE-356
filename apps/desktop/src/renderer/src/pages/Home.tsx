import { useCallback, useEffect, useState } from "react";
import { listGaitAnalyses } from "../api/gait.api";
import { listMriAnalyses } from "../api/mri.api";
import { listTappingAnalyses } from "../api/tapping.api";
import { InfoTip } from "../components/InfoTip";
import { PageHeader } from "../components/PageHeader";
import type { GaitAnalysisStatus } from "../gait/gaitTypes";
import type { AnalysisStatus, MriAnalysisStatus } from "../mri/mriTypes";
import type { TappingAnalysisStatus } from "../tapping/tappingTypes";
import { useVoiceRuns } from "../voice/voiceHistory";
import { groupAssessments, onePart, oneVoice, openAllParts, type OpenAssessment } from "./homeRows";
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
// analyses. One row is one assessment (same pseudonymous ID, started close together) with a cell
// for each part, so parts run together show side by side, each with its own status. Voice is not
// stored by the backend, so its cell shows this session's runs only. Nothing here is a
// placeholder number. The lists refresh every few seconds while any part is still running.

const RECENT_COUNT = 8;
const REFRESH_MS = 5000;

export function Home({
  clinician,
  mode = "home",
  onNewAssessment,
  onOpenAssessment,
  onSeeAll,
  onBackHome,
}: {
  clinician: string;
  /** "home": the start page with the latest rows. "all": every assessment, nothing else. */
  mode?: "home" | "all";
  onNewAssessment: () => void;
  onOpenAssessment: (open: OpenAssessment) => void;
  onSeeAll: () => void;
  onBackHome: () => void;
}) {
  const all = mode === "all";
  const onOpenGait = (id: string, label: string | null) => onOpenAssessment(onePart(label, "gait", id));
  const onOpenTapping = (id: string, label: string | null) =>
    onOpenAssessment(onePart(label, "tapping", id));
  const onOpenAnalysis = (id: string, label: string | null) =>
    onOpenAssessment(onePart(label, "mri", id));
  const [items, setItems] = useState<MriAnalysisStatus[] | null>(null);
  const [gaitItems, setGaitItems] = useState<GaitAnalysisStatus[] | null>(null);
  const [tappingItems, setTappingItems] = useState<TappingAnalysisStatus[] | null>(null);
  const [failed, setFailed] = useState(false);
  const [gaitFailed, setGaitFailed] = useState(false);
  const [tappingFailed, setTappingFailed] = useState(false);

  const voiceRuns = useVoiceRuns();

  const refresh = useCallback((isCurrent: () => boolean) => {
    listMriAnalyses(100)
      .then((list) => isCurrent() && (setItems(list.items), setFailed(false)))
      .catch(() => isCurrent() && setFailed(true));
    listGaitAnalyses(100)
      .then((list) => isCurrent() && (setGaitItems(list.items), setGaitFailed(false)))
      .catch(() => isCurrent() && setGaitFailed(true)); // the other rows still show
    listTappingAnalyses(100)
      .then((list) => isCurrent() && (setTappingItems(list.items), setTappingFailed(false)))
      .catch(() => isCurrent() && setTappingFailed(true)); // the other rows still show
  }, []);

  const anyRunning =
    (items ?? []).some((a) => a.status === "queued" || a.status === "running") ||
    (gaitItems ?? []).some((a) => a.status === "queued" || a.status === "running") ||
    (tappingItems ?? []).some((a) => a.status === "queued" || a.status === "running");

  useEffect(() => {
    let live = true;
    refresh(() => live);
    return () => {
      live = false;
    };
  }, [refresh]);

  // While anything is running, look again every few seconds so a part that finishes changes its cell.
  useEffect(() => {
    if (!anyRunning) return;
    let live = true;
    const timer = window.setInterval(() => refresh(() => live), REFRESH_MS);
    return () => {
      live = false;
      window.clearInterval(timer);
    };
  }, [anyRunning, refresh]);

  const now = new Date();
  const mriSummary = items ? homeSummary(items) : null;
  const gaitList = gaitItems ?? [];
  const gaitRunning = gaitList.filter((g) => g.status === "queued" || g.status === "running");
  const tappingList = tappingItems ?? [];
  const tappingRunning = tappingList.filter((t) => t.status === "queued" || t.status === "running");
  const summary =
    mriSummary === null
      ? null
      : {
          counted: mriSummary.counted + gaitList.length + tappingList.length + voiceRuns.length,
          scored:
            voiceRuns.filter((v) => v.status === "scored").length +
            mriSummary.scored +
            gaitList.filter((g) => g.status === "scored").length +
            tappingList.filter((t) => t.status === "scored").length,
          notScored:
            voiceRuns.filter((v) => v.status === "abstained").length +
            mriSummary.notScored +
            gaitList.filter((g) => g.status === "abstained").length +
            tappingList.filter((t) => t.status === "abstained").length,
          failed:
            voiceRuns.filter((v) => v.status === "failed").length +
            mriSummary.failed +
            gaitList.filter((g) => g.status === "failed").length +
            tappingList.filter((t) => t.status === "failed").length,
          inProgress: mriSummary.inProgress.length + gaitRunning.length + tappingRunning.length,
        };
  // One row per assessment, newest first.
  const rows = groupAssessments(
    items ?? [],
    gaitList,
    tappingList,
    voiceRuns,
    (iso) => parseBackendTime(iso).getTime(),
  );

  return (
    <div className="home">
      <PageHeader
        title={all ? "All assessments" : `${greeting(now)}, ${clinician}`}
        subtitle={
          all
            ? "Every assessment you have started, newest first. Open a row to see all its parts."
            : now.toLocaleDateString(undefined, {
                weekday: "long",
                day: "numeric",
                month: "long",
                year: "numeric",
              })
        }
        actions={
          <span className="home__actions">
            {all && (
              <button type="button" className="btn btn--outline" onClick={onBackHome}>
                ← Home
              </button>
            )}
            <button type="button" className="btn btn--primary" onClick={onNewAssessment}>
              + New assessment
            </button>
          </span>
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
      {!failed && tappingFailed && (
        <p className="home__error" role="alert">
          Could not load your finger-tapping analyses, so they are not listed or counted below.
        </p>
      )}

      <div className={`home__layout${all ? " home__layout--all" : ""}`}>
        <div className="home__main">
          {!all && summary && summary.inProgress > 0 && (
            <section className="card home__progress" aria-label="In progress">
              <p className="eyebrow">In progress</p>
              {gaitRunning.map((item) => (
                <div key={item.id} className="home__progress-row">
                  <div>
                    <p className="home__progress-title">
                      {item.subject_label ? `${item.subject_label} · ` : ""}Gait part running
                    </p>
                    <p className="note">
                      {item.status === "queued" ? "Waiting for its turn" : "Reading the walk"} · usually about a minute or two
                    </p>
                  </div>
                  <button type="button" className="btn btn--outline" onClick={() => onOpenGait(item.id, item.subject_label)}>
                    Open →
                  </button>
                </div>
              ))}
              {tappingRunning.map((item) => (
                <div key={item.id} className="home__progress-row">
                  <div>
                    <p className="home__progress-title">
                      {item.subject_label ? `${item.subject_label} · ` : ""}Finger-tapping part running
                    </p>
                    <p className="note">
                      {item.status === "queued" ? "Waiting for its turn" : "Reading the recording"} · usually under a minute
                    </p>
                  </div>
                  <button type="button" className="btn btn--outline" onClick={() => onOpenTapping(item.id, item.subject_label)}>
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
                  <button type="button" className="btn btn--outline" onClick={() => onOpenAnalysis(item.id, item.subject_label)}>
                    Open →
                  </button>
                </div>
              ))}
            </section>
          )}

          <section className="card" aria-label={all ? "All assessments" : "Recent assessments"}>
            <div className="home__head">
              <h2 className="card__title">
                {all ? `${rows.length} assessment${rows.length === 1 ? "" : "s"}` : "Recent assessments"}
              </h2>
              {!all && (
                <button type="button" className="home__link" onClick={onSeeAll}>
                  All assessments
                </button>
              )}
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
                  {(all ? rows : rows.slice(0, RECENT_COUNT)).map((row) => (
                    <tr key={row.key}>
                      <td className="mono">{row.label ?? "No ID"}</td>
                      <td className="mono">{new Date(row.startedAt).toLocaleString()}</td>
                      <PartCell part={row.gait} onOpen={(id) => onOpenGait(id, row.label)} />
                      <PartCell part={row.tapping} onOpen={(id) => onOpenTapping(id, row.label)} />
                      <td>
                        {row.voice ? (
                          <button
                            type="button"
                            className="home__cell"
                            title="Open this voice result (kept until the app closes; the server stores none)"
                            onClick={() => row.voice && onOpenAssessment(oneVoice(row.label, row.voice.id))}
                          >
                            <span className={`mri__tag mri__tag--${statusLabel(row.voice.status).tone}`}>
                              {statusLabel(row.voice.status).text}
                            </span>
                          </button>
                        ) : (
                          <span className="home__none" title="Not added to this analysis">—</span>
                        )}
                      </td>
                      <PartCell
                        part={row.mri}
                        onOpen={(id) => onOpenAnalysis(id, row.label)}
                        fast={row.mri?.eddy_profile === "fast"}
                      />
                      <td>
                        {row.mri || row.gait || row.tapping || row.voice ? (
                          <button
                            type="button"
                            className="home__link"
                            onClick={() => onOpenAssessment(openAllParts(row))}
                          >
                            Open →
                          </button>
                        ) : (
                          <span className="home__none" title="Voice results are not stored, so they cannot be reopened">
                            —
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>
        </div>

        {!all && (
          <aside className="home__rail">
            <section className="card">
              <p className="eyebrow">
                Your assessments
                <InfoTip label="About these counts">
                  {summary ? (summary.counted >= 100 ? "Counted over your 100 most recent." : "All of them.") : ""}{" "}
                  Each counts one analysis: an MRI scan, a walking video, a finger-tapping video or a voice recording. A
                  “not scored” result means the model declined to score it (for example, an unseen
                  scanner model or a video it could not read); it is not a normal result. Voice results
                  are not stored, so only voice recordings from this session are counted.
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
        )}
      </div>
    </div>
  );
}

/** One part's cell in a row: its own status, and a click opens that analysis. */
function PartCell({
  part,
  onOpen,
  fast = false,
}: {
  part: { id: string; status: AnalysisStatus } | null;
  onOpen: (id: string) => void;
  fast?: boolean;
}) {
  if (!part) {
    return (
      <td className="home__none" title="Not added to this analysis">
        —
      </td>
    );
  }
  const label = statusLabel(part.status);
  return (
    <td>
      <button type="button" className="home__cell" onClick={() => onOpen(part.id)} title="Open this part">
        <span className={`mri__tag mri__tag--${label.tone}`}>{label.text}</span>
        {fast && <span className="mri__tag"> Fast DTI</span>}
      </button>
    </td>
  );
}
