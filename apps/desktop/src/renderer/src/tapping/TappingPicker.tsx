import { useRef, useState } from "react";
import "../components/GaitPicker.css";
import { readVideoSeconds } from "../gait/readVideoSeconds";
import { videoRef } from "./tappingSource";
import { videoProblem, videoSummary } from "./tappingView";

// The finger-tapping card in the Modalities step: choose a video, see how to film. The type and
// size are checked here before anything is read. The file name is never shown: once the video is
// chosen the card shows its size, length and hashed ref only.

export function TappingPicker({
  video,
  seconds,
  videoRefText,
  onVideo,
}: {
  video: File | null;
  seconds: number | null;
  videoRefText: string | null;
  onVideo: (file: File | null, seconds: number | null, ref: string | null) => void;
}) {
  const [dragging, setDragging] = useState(false);
  const [rejected, setRejected] = useState<{ title: string; text: string } | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const latest = useRef(0); // guards a slow read against a later choose or remove

  async function choose(file: File | null): Promise<void> {
    if (!file) return;
    const problem = videoProblem(file.name, file.size);
    if (problem) {
      setRejected(problem);
      return;
    }
    setRejected(null);
    onVideo(file, null, null);
    const mine = ++latest.current;
    const [length, ref] = await Promise.all([readVideoSeconds(file), videoRef(file)]);
    if (mine === latest.current) onVideo(file, length, ref);
  }

  function remove(): void {
    latest.current += 1;
    setRejected(null);
    onVideo(null, null, null);
    if (input.current) input.current.value = "";
  }

  const hiddenInput = (
    <input
      ref={input}
      className="mri__drop-input"
      type="file"
      accept=".mp4,.mov,.m4v,video/mp4,video/quicktime,video/x-m4v"
      aria-label="Finger-tapping video"
      onChange={(e) => void choose(e.target.files?.[0] ?? null)}
    />
  );

  const icon = (size: number) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="5" width="18" height="14" rx="2" />
      <path d="M10 9.5v5l4.5-2.5z" />
    </svg>
  );

  return (
    <div className="gr__pick">
      {video ? (
        <div className="gr__file">
          <span className="gr__file-icon" aria-hidden="true">
            {icon(22)}
          </span>
          <span className="gr__file-body">
            <span className="gr__file-name">Finger-tapping video</span>
            <span className="note mono">{videoSummary(video.size, seconds, videoRefText)}</span>
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
            {icon(28)}
          </span>
          <span className="mri__drop-text">
            <span className="mri__drop-link">Choose a video</span> or drop it here
            <br />
            MP4, MOV or M4V · up to 200 MB
          </span>
        </label>
      )}

      {rejected && (
        <div className="gr__err" role="alert">
          <strong>{rejected.title}</strong>
          <span>The file was not added. {rejected.text}</span>
        </div>
      )}

      <details className="gr__guide">
        <summary>How to film finger tapping the model can use</summary>
        <div className="gr__guide-body">
          <ul className="gr__guide-list">
            <li>One hand in view, the whole hand, with the thumb and index finger clear.</li>
            <li>Tap thumb to index finger, as wide and as fast as is comfortable, for about 20 seconds.</li>
            <li>Hold the phone still. Use the normal video setting (30 or 60 frames per second), not slow motion.</li>
            <li>Keep the hand in view the whole time. Good light and a plain background help.</li>
            <li>The video can be up to 60 seconds long.</li>
          </ul>
        </div>
      </details>
    </div>
  );
}
