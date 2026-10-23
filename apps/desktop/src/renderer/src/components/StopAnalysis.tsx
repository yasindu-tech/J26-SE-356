import { useState } from "react";
import { ApiError } from "../api/client";
import "./StopAnalysis.css";

// A Stop control for a running part. It asks once before stopping, because a stopped analysis
// cannot be resumed: the upload is deleted and has to be sent again. After the backend accepts
// the stop, the progress screen's next poll sees the part as ended and moves on by itself.

type Phase =
  | { kind: "idle" }
  | { kind: "confirm" }
  | { kind: "stopping" }
  | { kind: "error"; message: string };

export function StopAnalysis({
  part,
  onStop,
}: {
  /** "MRI", "gait", "finger-tapping": used in the question. */
  part: string;
  onStop: () => Promise<unknown>;
}) {
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });

  async function stop(): Promise<void> {
    setPhase({ kind: "stopping" });
    try {
      await onStop();
      // Stay on "Stopping…": the progress screen replaces this view once its poll sees the end.
    } catch (e) {
      setPhase({
        kind: "error",
        message: e instanceof ApiError ? e.message : "Could not reach the backend to stop it.",
      });
    }
  }

  switch (phase.kind) {
    case "idle":
      return (
        <div className="stop">
          <button type="button" className="btn btn--outline stop__btn" onClick={() => setPhase({ kind: "confirm" })}>
            Stop this analysis
          </button>
        </div>
      );
    case "confirm":
      return (
        <div className="stop stop--confirm" role="alertdialog" aria-label={`Stop the ${part} analysis?`}>
          <p className="stop__question">
            Stop the {part} analysis? It ends now and cannot be resumed. To try again you will need to
            upload the file again.
          </p>
          <div className="stop__actions">
            <button type="button" className="btn btn--outline" onClick={() => setPhase({ kind: "idle" })}>
              Keep running
            </button>
            <button type="button" className="btn stop__yes" onClick={() => void stop()}>
              Yes, stop it
            </button>
          </div>
        </div>
      );
    case "stopping":
      return (
        <div className="stop">
          <button type="button" className="btn btn--outline stop__btn" disabled>
            Stopping…
          </button>
        </div>
      );
    case "error":
      return (
        <div className="stop">
          <p className="stop__error" role="alert">
            {phase.message}
          </p>
          <button type="button" className="btn btn--outline stop__btn" onClick={() => void stop()}>
            Try stopping again
          </button>
        </div>
      );
    default: {
      const exhaustive: never = phase;
      return exhaustive;
    }
  }
}
