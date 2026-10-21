import { useEffect, useState } from "react";
import { getGaitOverlay } from "../api/gait.api";
import { loadOverlay, type OverlayLoad } from "../gait/overlayLoad";
import "./GaitOverlayCard.css";

// The walking video with the skeleton and the counted heel strikes drawn on it, so the
// clinician can see what the model read. Shows nothing when the analysis has no overlay.

export function GaitOverlayCard({ analysisId }: { analysisId: string }) {
  const [load, setLoad] = useState<OverlayLoad | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    let objectUrl: string | null = null;
    setLoad(null);
    setUrl(null);
    void loadOverlay(analysisId, getGaitOverlay).then((next) => {
      if (cancelled) return;
      if (next.kind === "ready") {
        objectUrl = URL.createObjectURL(next.blob);
        setUrl(objectUrl);
      }
      setLoad(next);
    });
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [analysisId, attempt]);

  if (load?.kind === "none") return null;
  return (
    <section className="card go" aria-label="The video with the skeleton drawn on it">
      <h2 className="go__title">What the model read</h2>
      {!load && <p className="note">Loading the video…</p>}
      {load?.kind === "error" && (
        <div className="go__error">
          <p className="mri__error" role="alert">
            {load.message}
          </p>
          <button type="button" className="btn btn--outline" onClick={() => setAttempt((n) => n + 1)}>
            Try again
          </button>
        </div>
      )}
      {url && (
        <video className="go__video" src={url} controls muted preload="metadata">
          Your system cannot play this video.
        </video>
      )}
      <p className="note">
        The skeleton is the body the model found. Rings mark each step it counted: orange for the
        left foot, blue for the right. The green border is the part that was scored; the grey part
        was not used. The banner shows the number of strides counted.
      </p>
    </section>
  );
}
