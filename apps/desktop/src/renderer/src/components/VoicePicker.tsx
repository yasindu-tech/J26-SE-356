import { useRef, useState } from "react";
import { VOICE_AUDIO_ACCEPT } from "../api/voice.api";
import { VOICE_MAX_UPLOAD_MB, voiceFileProblem, voiceFileSummary } from "../voice/voiceFile";
import "./GaitPicker.css";

// The voice card in the Modalities step: choose a recording of a steady "aaah" and see how
// to record one. It looks and behaves like the gait card (GaitPicker, whose styles it uses).
// The file type and size are checked here, before anything is sent.

function MicIcon({ size }: { size: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect x="9" y="3" width="6" height="11" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0M12 18v3" />
    </svg>
  );
}

export function VoicePicker({
  recording,
  onRecording,
}: {
  recording: File | null;
  onRecording: (file: File | null) => void;
}) {
  const [dragging, setDragging] = useState(false);
  const [rejected, setRejected] = useState<{ name: string; title: string; text: string } | null>(
    null,
  );
  const input = useRef<HTMLInputElement>(null);

  function choose(file: File | null): void {
    if (!file) return;
    const problem = voiceFileProblem(file.name, file.size);
    if (problem) {
      setRejected({ name: file.name, ...problem });
      return;
    }
    setRejected(null);
    onRecording(file);
  }

  function remove(): void {
    setRejected(null);
    onRecording(null);
    if (input.current) input.current.value = "";
  }

  const hiddenInput = (
    <input
      ref={input}
      className="mri__drop-input"
      type="file"
      accept={VOICE_AUDIO_ACCEPT}
      onChange={(e) => choose(e.target.files?.[0] ?? null)}
    />
  );

  return (
    <div className="gr__pick">
      {recording ? (
        <div className="gr__file">
          <span className="gr__file-icon" aria-hidden="true">
            <MicIcon size={22} />
          </span>
          <span className="gr__file-body">
            <span className="gr__file-name mono">{recording.name}</span>
            <span className="note mono">
              {voiceFileSummary(recording.name, recording.size)} · ready to send
            </span>
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
            choose(e.dataTransfer.files[0] ?? null);
          }}
        >
          {hiddenInput}
          <span className="gr__drop-icon" aria-hidden="true">
            <MicIcon size={28} />
          </span>
          <span className="mri__drop-text">
            <span className="mri__drop-link">Choose a recording</span> or drop it here
            <br />
            WAV, MP3, M4A, FLAC, OGG, AAC or WebM · up to {VOICE_MAX_UPLOAD_MB} MB
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
      <details className="gr__guide">
        <summary>How to record an &ldquo;aaah&rdquo; the model can use</summary>
        <div className="gr__guide-body">
          <ul className="gr__guide-list">
            <li>A quiet room: no fan, TV or other voices.</li>
            <li>
              Leave about one second of silence, then one steady &ldquo;aaah&rdquo; for about 5
              seconds.
            </li>
            <li>A natural voice at a comfortable pitch. Do not force or change it.</li>
            <li>Phone about 20 cm from the mouth. WAV is best; MP3 and M4A also work.</li>
            <li>
              A cold, sore throat or breathing problem can raise the score: note it, or record on
              another day.
            </li>
          </ul>
        </div>
      </details>
    </div>
  );
}
