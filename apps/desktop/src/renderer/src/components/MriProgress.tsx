import { useEffect, useRef, useState } from "react";
import { ApiError } from "../api/client";
import { getMriResult, getMriStatus } from "../api/mri.api";
import { MRI_POLL_INTERVAL_MS } from "../constants/config";
import type { MriAnalysisStatus, MriResult } from "../mri/mriTypes";
import { MRI_STEPS } from "../mri/mriTypes";
import {
  FAST_EDDY_NOTICE,
  STEP_LABELS,
  formatClock,
  formatElapsed,
  isFinal,
  parseBackendTime,
  stepState,
  updateActivity,
  type ActivityEntry,
} from "../mri/mriView";
import "../pages/MriAnalysis.css";

// The MRI part's progress view: step bar, what the analysis service last reported, and an
// activity list built from the reports this screen has seen.

// What the backend answers when there is no usable session (401 after a failed
// renewal; 403 from HTTPBearer when no token is sent at all).
const SIGNED_OUT_CODES = new Set(["NOT_AUTHENTICATED", "HTTP_401", "HTTP_403"]);

function errorText(error: unknown, fallback: string): string {
  return error instanceof ApiError ? error.message : fallback;
}

// ------------------------------------------------------------------ progress

interface ProgressPanelProps {
  id: string;
  onFinished: (result: MriResult) => void;
  onError: (message: string) => void;
}

export function ProgressPanel({ id, onFinished, onError }: ProgressPanelProps) {
  const [status, setStatus] = useState<MriAnalysisStatus | null>(null);
  const [activity, setActivity] = useState<ActivityEntry[]>([]);
  const [now, setNow] = useState(() => Date.now());
  const [lastContact, setLastContact] = useState<number | null>(null);
  // The parent passes new callback functions on every render; keeping them in refs
  // stops each render from restarting the polling loop.
  const finishedRef = useRef(onFinished);
  const errorRef = useRef(onError);
  finishedRef.current = onFinished;
  errorRef.current = onError;

  useEffect(() => {
    const tick = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(tick);
  }, []);

  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    async function poll(): Promise<void> {
      try {
        const next = await getMriStatus(id);
        if (stopped) return;
        setStatus(next);
        setLastContact(Date.now());
        setActivity((log) => updateActivity(log, next, new Date()));
        if (isFinal(next.status)) {
          finishedRef.current(await getMriResult(id));
          return;
        }
      } catch (e) {
        if (stopped) return;
        if (e instanceof ApiError && SIGNED_OUT_CODES.has(e.code)) {
          // No valid session: polling cannot succeed, so stop instead of retrying.
          errorRef.current(
            "You are signed out. Sign in again and open this analysis from the list; it keeps running.",
          );
          return;
        }
        errorRef.current(errorText(e, "Lost contact with the backend; still trying."));
      }
      if (!stopped) timer = setTimeout(() => void poll(), MRI_POLL_INTERVAL_MS);
    }

    void poll();
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
  }, [id]);

  const done = status?.done ?? [];
  const started = status ? parseBackendTime(status.created_at) : null;
  const elapsed = started ? (now - started.getTime()) / 1000 : 0;
  const sinceContact = lastContact === null ? null : Math.max(0, Math.round((now - lastContact) / 1000));

  return (
    <div className="mri__tracking">
      <section className="card">
        <div className="mri__tracking-head">
          <div className="mri__tracking-text">
            <h2 className="mri__tracking-title">MRI part: analysing the scan</h2>
            <p className="mri__body-text">
              This is normal and can take a while. You can carry on with the consultation; the
              result opens here when it is ready.
            </p>
          </div>
          <div className="mri__elapsed">
            <span className="mri__elapsed-value mono">{formatElapsed(elapsed)}</span>
            <span className="note">
              elapsed{started ? ` · started ${formatClock(started)}` : ""}
            </span>
          </div>
        </div>

        {status?.eddy_profile === "fast" && (
          <p className="mri__notice" role="note">
            {FAST_EDDY_NOTICE}
          </p>
        )}

        <ol className="mri__segments" aria-label="Analysis steps">
          {MRI_STEPS.map((step, i) => {
            const state = status
              ? stepState(step, status.status, done, status.current_step)
              : "waiting";
            return (
              <li key={step} className={`mri__segment mri__segment--${state}`}>
                <span className="mri__segment-bar" />
                <span className="mri__segment-label">
                  {i + 1} {STEP_LABELS[step]}
                </span>
              </li>
            );
          })}
        </ol>

        <div className="mri__now">
          <span className="mri__now-text">
            {status?.note ? status.note : status ? "Waiting for the next step…" : "Contacting the backend…"}
          </span>
          <span className="note mri__now-contact">
            <span className="mri__dot" />
            {sinceContact === null
              ? "Not heard from the analysis service yet"
              : `Last update from the analysis service ${sinceContact} s ago`}
          </span>
        </div>

        <p className="note mri__leave">
          Leaving this screen does not stop the analysis. It continues in the background and
          stays in <strong>Home</strong>.
        </p>
      </section>

      <section className="card">
        <h2 className="mri__rail-title">Activity</h2>
        {activity.length === 0 ? (
          <p className="note">Nothing reported yet.</p>
        ) : (
          <ol className="mri__activity mono">
            {activity.map((entry) => (
              <li key={entry.key}>
                <span className="mri__activity-time">{formatClock(new Date(entry.at))}</span>
                <span>{entry.text}</span>
              </li>
            ))}
          </ol>
        )}
        <p className="note">
          Listed as this screen sees each report. It is not a stored log, and it starts when you
          open this screen.
        </p>
      </section>
    </div>
  );
}

export function WhySlowCard() {
  return (
    <section className="card">
      <h2 className="mri__rail-title">Why does the DTI part take so long?</h2>
      <p className="mri__body-text">
        DTI holds dozens of diffusion images, and each one has to be straightened out before
        measuring. That motion correction (called eddy) is the slow step: from tens of minutes
        to over an hour on an ordinary computer. T1 takes a few minutes.
      </p>
      <p className="mri__body-text">
        Nothing is frozen. The line under the step bar changes as each stage reports in.
      </p>
    </section>
  );
}
