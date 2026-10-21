import { useCallback, useEffect, useRef, useState } from "react";
import type { TappingStatus } from "../../../shared/ipc";
import { PageHeader } from "../components/PageHeader";
import "./Tapping.css";

// Finger-tapping research demo (FT-61). Streamlit runs on this computer as a child of the
// main process and is shown in a view of this window; this page only provides the box the
// view sits in (its bounds go to the main process) and the status. Leaving the page stops it.

const STARTING_TEXT = "Starting the demo. This can take up to a minute the first time.";

export function Tapping({ onBack }: { onBack: () => void }) {
  const [status, setStatus] = useState<TappingStatus>({ state: "starting" });
  const box = useRef<HTMLDivElement>(null);

  const start = useCallback(() => {
    setStatus({ state: "starting" });
    void window.electronAPI.tapping.start().then(setStatus);
  }, []);

  useEffect(() => {
    const unsubscribe = window.electronAPI.tapping.onStatus(setStatus);
    start();
    return () => {
      unsubscribe();
      void window.electronAPI.tapping.stop();
    };
  }, [start]);

  // The main process places the view where this box is: send its bounds now, when it is
  // resized and when anything scrolls.
  useEffect(() => {
    const element = box.current;
    if (!element) return;
    const send = (): void => {
      const rect = element.getBoundingClientRect();
      window.electronAPI.tapping.setBounds({
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
      });
    };
    send();
    const observer = new ResizeObserver(send);
    observer.observe(element);
    window.addEventListener("resize", send);
    window.addEventListener("scroll", send, true);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", send);
      window.removeEventListener("scroll", send, true);
    };
  }, [status.state]);

  const ready = status.state === "ready";

  return (
    <div className="tapping">
      <div className="tapping__head">
        <button type="button" className="btn btn--outline" onClick={onBack}>
          ← Back to assessment
        </button>
        <PageHeader
          title="Finger tapping"
          subtitle="Research prototype. Runs on this computer."
          actions={
            <button
              type="button"
              className="btn btn--outline"
              disabled={!ready}
              onClick={() => void window.electronAPI.tapping.openExternal()}
            >
              Open in browser
            </button>
          }
        />
      </div>
      <div className="tapping__stage" ref={box}>
        {status.state === "starting" && <p className="tapping__message">{STARTING_TEXT}</p>}
        {(status.state === "unavailable" || status.state === "error") && (
          <div className="tapping__message" role="alert">
            <p>{status.reason ?? "The demo could not be started."}</p>
            <button type="button" className="btn btn--outline" onClick={start}>
              {status.state === "error" ? "Try again" : "Check again"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
