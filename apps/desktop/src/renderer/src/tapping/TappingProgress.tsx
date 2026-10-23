import { useEffect, useRef, useState } from "react";
import { stopTappingAnalysis } from "../api/tapping.api";
import { StopAnalysis } from "../components/StopAnalysis";
import { ApiError } from "../api/client";
import "../components/GaitProgress.css";
import { formatElapsed } from "../mri/mriView";
import { getTappingRun } from "./tappingRun";
import type { TappingRunStatus } from "./tappingTypes";
import { VIDEO_STAYS_HERE } from "./tappingView";

// The wait for a finger-tapping result, built like the gait wait: the backend reports only
// waiting and running, so there is no step-by-step bar: an indeterminate bar, the time elapsed,
// and when it last heard back. The analysis runs in the backend, so leaving this screen does not
// stop it.

const POLL_MS = 2000;
/** Polls that may fail one after another before the screen gives up. */
const MAX_FAILURES = 5;

export function TappingProgress({
  id,
  onFinished,
  onError,
}: {
  id: string;
  onFinished: (status: TappingRunStatus) => void;
  onError: (message: string) => void;
}) {
  const [status, setStatus] = useState<TappingRunStatus | null>(null);
  const [lastHeard, setLastHeard] = useState<number | null>(null);
  const [trouble, setTrouble] = useState(false);
  const [now, setNow] = useState(Date.now());
  const started = useRef(Date.now());

  useEffect(() => {
    let cancelled = false;
    let failures = 0;
    async function poll(): Promise<void> {
      try {
        const next = await getTappingRun(id);
        if (cancelled) return;
        failures = 0;
        setTrouble(false);
        setStatus(next);
        setLastHeard(Date.now());
        if (next.state === "scored" || next.state === "abstained" || next.state === "failed") {
          onFinished(next);
          return;
        }
      } catch (e) {
        if (cancelled) return;
        failures += 1;
        if (failures >= MAX_FAILURES) {
          onError(
            e instanceof ApiError && e.code !== "NETWORK_ERROR"
              ? e.message
              : "Lost contact with the analysis service. The analysis keeps running there; open it again from Home.",
          );
          return;
        }
        setTrouble(true); // keep trying; one missed poll is not a failed analysis
      }
      timer = window.setTimeout(() => void poll(), POLL_MS);
    }
    let timer = window.setTimeout(() => void poll(), 0);
    const tick = window.setInterval(() => setNow(Date.now()), 1000);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
      window.clearInterval(tick);
    };
    // onFinished and onError come from the page and change on every render; poll once per id.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  const elapsed =
    status && lastHeard !== null
      ? status.elapsedSeconds + Math.max(0, (now - lastHeard) / 1000)
      : (now - started.current) / 1000;
  const since = lastHeard === null ? null : Math.max(0, Math.round((now - lastHeard) / 1000));
  const waiting = status?.state === "queued";

  return (
    <div className="mri__tracking">
      <section className="card">
        <div className="mri__tracking-head">
          <div className="mri__tracking-text">
            <h2 className="mri__tracking-title">Finger-tapping part: analysing the recording</h2>
            <p className="mri__body-text">
              This usually takes under a minute. You can carry on with the consultation; the result
              opens here when it is ready.
            </p>
          </div>
          <div className="mri__elapsed">
            <span className="mri__elapsed-value mono">{formatElapsed(elapsed)}</span>
            <span className="note">elapsed</span>
          </div>
        </div>

        <div className="gr__wait" role="progressbar" aria-label="Analysing, time left not known">
          <div className="gr__wait-fill" />
        </div>

        <div className="mri__now">
          <span className="mri__now-text">{waiting ? "Waiting for its turn." : "Reading the recording."}</span>
          <span className="note mri__now-contact">
            <span className="mri__dot" />
            {since === null
              ? "Not heard from the analysis service yet"
              : `Last update from the analysis service ${since} s ago`}
          </span>
        </div>

        {trouble && (
          <p className="gr__err" role="status">
            Lost contact with the analysis service. Trying again. The analysis keeps running there.
          </p>
        )}

        <p className="note">
          If another tapping analysis is already running, this one waits its turn and the line above
          says &ldquo;Waiting for its turn.&rdquo;
        </p>
        <p className="note mri__leave">
          Leaving this screen does not stop the analysis. It continues in the background and stays in{" "}
          <strong>Home</strong>.
        </p>
        <StopAnalysis part="finger-tapping" onStop={() => stopTappingAnalysis(id)} />
      </section>
    </div>
  );
}

export function TappingWhatIsHappening() {
  return (
    <>
      <section className="card">
        <h2 className="mri__rail-title">What is happening</h2>
        <ol className="gr__list">
          <li>The hand is found in every frame and the thumb and index fingertips are followed.</li>
          <li>The distance between them is smoothed and each tap is found.</li>
          <li>Five measurements are taken, such as tapping rate and rhythm.</li>
          <li>The model scores the recording and shows which measurements moved the score.</li>
        </ol>
      </section>
      <section className="card">
        <h2 className="mri__rail-title">What happens to the video</h2>
        <p className="mri__body-text">{VIDEO_STAYS_HERE}</p>
      </section>
    </>
  );
}
