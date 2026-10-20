import { useRef, useState } from "react";
import { GAIT_READ_SECONDS, videoProblem, videoSummary } from "../gait/gaitView";
import { readVideoSeconds } from "../gait/readVideoSeconds";
import "./GaitPicker.css";

// The gait card in the Modalities step: choose a walking video, tick that one person is
// walking, and see how to film. The file type and size are checked here, before anything
// is sent. The tick is the clinician's confirmation: the pipeline cannot check for other people.

export function GaitPicker({
  video,
  videoSeconds,
  confirmed,
  onVideo,
  onConfirmed,
}: {
  video: File | null;
  videoSeconds: number | null;
  confirmed: boolean;
  onVideo: (file: File | null, seconds: number | null) => void;
  onConfirmed: (value: boolean) => void;
}) {
  const [dragging, setDragging] = useState(false);
  const [rejected, setRejected] = useState<{ name: string; title: string; text: string } | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const latest = useRef(0); // guards a slow length read against a later choose or remove

  async function choose(file: File | null): Promise<void> {
    if (!file) return;
    const problem = videoProblem(file.name, file.size);
    if (problem) {
      setRejected({ name: file.name, ...problem });
      return;
    }
    setRejected(null);
    // A new video is a new confirmation: the tick is asked again for every video.
    onConfirmed(false);
    onVideo(file, null);
    const mine = ++latest.current;
    const seconds = await readVideoSeconds(file);
    if (mine === latest.current) onVideo(file, seconds);
  }

  function remove(): void {
    latest.current += 1;
    setRejected(null);
    onConfirmed(false);
    onVideo(null, null);
    if (input.current) input.current.value = "";
  }

  const hiddenInput = (
    <input
      ref={input}
      className="mri__drop-input"
      type="file"
      accept=".mp4,.mov,.m4v,.avi,video/mp4,video/quicktime,video/x-m4v,video/x-msvideo"
      onChange={(e) => void choose(e.target.files?.[0] ?? null)}
    />
  );

  return (
    <div className="gr__pick">
      {video ? (
        <div className="gr__file">
          <span className="gr__file-icon" aria-hidden="true">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="5" width="18" height="14" rx="2" />
              <path d="M10 9.5v5l4.5-2.5z" />
            </svg>
          </span>
          <span className="gr__file-body">
            <span className="gr__file-name mono">{video.name}</span>
            <span className="note mono">{videoSummary(video.name, video.size, videoSeconds)}</span>
            {videoSeconds !== null && videoSeconds > GAIT_READ_SECONDS && (
              <span className="note">Only the first {GAIT_READ_SECONDS} seconds are read.</span>
            )}
          </span>
          <span className="gr__file-actions">
            <button type="button" className="asm__link" onClick={() => input.current?.click()}>
              Choose another
            </button>
            <button type="button" className="asm__link" onClick={remove}>
              Remove
            </button>
            <span hidden>{hiddenInput}</span>
          </span>
        </div>
      ) : (
        <label
          className={`mri__drop gr__drop${dragging ? " mri__drop--over" : ""}${rejected ? " gr__drop--bad" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            void choose(e.dataTransfer.files[0] ?? null);
          }}
        >
          {hiddenInput}
          <span className="gr__drop-icon" aria-hidden="true">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="5" width="18" height="14" rx="2" />
              <path d="M10 9.5v5l4.5-2.5z" />
            </svg>
          </span>
          <span className="mri__drop-text">
            <span className="mri__drop-link">Choose a video</span> or drop it here
            <br />
            MP4, MOV, M4V or AVI · up to 500 MB
          </span>
        </label>
      )}

      {rejected && (
        <div className="gr__err" role="alert">
          <strong>{rejected.title}</strong>
          <span>
            {rejected.name} was not added. {rejected.text}
          </span>
        </div>
      )}

      {video && (
        <label className={`gr__consent${confirmed ? " gr__consent--on" : ""}`}>
          <input type="checkbox" checked={confirmed} onChange={(e) => onConfirmed(e.target.checked)} />
          <span className="gr__consent-text">
            <span className="gr__consent-title">One person is walking, seen from the side</span>
            <span className="gr__consent-sub">
              The model reads a single body. It cannot tell whether someone else is in the video, so
              this is your confirmation. It is saved with the result and asked again for every video.
            </span>
          </span>
        </label>
      )}

      <details className="gr__guide">
        <summary>How to film a walk the model can use</summary>
        <div className="gr__guide-body">
          <svg className="gr__guide-fig" width="200" height="132" viewBox="0 0 200 132" role="img" aria-label="Top view: the camera sits to the side and still, pointing at the line the person walks along.">
            <line x1="14" y1="40" x2="182" y2="40" stroke="currentColor" strokeWidth="2" />
            <path d="M176 34l8 6-8 6" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            <circle cx="70" cy="40" r="7" fill="currentColor" />
            <path d="M100 96L34 44M100 96l66-52" stroke="currentColor" strokeWidth="1.5" strokeDasharray="4 4" fill="none" />
            <rect x="88" y="96" width="24" height="14" rx="3" fill="currentColor" />
            <text x="14" y="24" fontSize="11" fill="#5C6663" fontFamily="IBM Plex Sans, sans-serif">walking direction</text>
            <text x="100" y="126" fontSize="11" fill="#5C6663" textAnchor="middle" fontFamily="IBM Plex Sans, sans-serif">camera, side-on, still</text>
          </svg>
          <ul className="gr__guide-list">
            <li>One person in view, head to feet.</li>
            <li>Film from the side, with the phone held still, sideways (landscape).</li>
            <li>Walk across the frame for at least 5 seconds.</li>
            <li>Use the normal video setting (30 or 60 frames per second), not slow motion.</li>
            <li>Good light, with the legs clear of furniture.</li>
          </ul>
        </div>
      </details>
    </div>
  );
}
