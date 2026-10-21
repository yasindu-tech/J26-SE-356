import { useRef, useState } from "react";
import { ApiError } from "../api/client";
import { uploadVoiceRecording, VOICE_AUDIO_ACCEPT } from "../api/voice.api";
import { PageHeader } from "../components/PageHeader";
import { VoiceAnalysisProgress } from "../components/VoiceAnalysisProgress";
import { VoiceResultCard } from "../components/VoiceResultCard";
import { parseVoiceResults, triageLabel, type VoiceResult } from "../voice/voiceResult";
import "./VoiceResults.css";

// PP1 voice result screen (models/voice/README.md, definition of done item 8).
// Two ways in: upload a recording, which the backend analyses and deletes
// (POST /api/v1/voice/analyses), or open a result file the voice module wrote
// (demo_results.json, analysis_results.json), which is only read here.

type ScreenState =
  | { status: "empty" }
  | { status: "opening" }
  | {
      status: "analysing";
      fileName: string;
      uploaded: number;
      analysingSince: number | null;
    }
  | { status: "error"; message: string }
  | {
      status: "loaded";
      fileName: string;
      results: VoiceResult[];
      selected: number;
    };

function resultSummary(result: VoiceResult): string {
  return result.kind === "scored" ? triageLabel(result.triagePriority) : "Not scored";
}

export function VoiceResults() {
  const [state, setState] = useState<ScreenState>({ status: "empty" });
  const fileInput = useRef<HTMLInputElement>(null);
  const busy = state.status === "opening" || state.status === "analysing";

  async function handleUpload(file: File): Promise<void> {
    setState({
      status: "analysing",
      fileName: file.name,
      uploaded: 0,
      analysingSince: null,
    });
    try {
      const reply = await uploadVoiceRecording(file, (fraction) =>
        setState((current) =>
          current.status === "analysing" && current.analysingSince === null
            ? {
                ...current,
                uploaded: fraction,
                analysingSince: fraction >= 1 ? Date.now() : null,
              }
            : current,
        ),
      );
      const outcome = parseVoiceResults(reply);
      setState(
        outcome.ok
          ? {
              status: "loaded",
              fileName: file.name,
              results: outcome.results,
              selected: 0,
            }
          : { status: "error", message: outcome.message },
      );
    } catch (error) {
      setState({
        status: "error",
        message:
          error instanceof ApiError
            ? error.message
            : "Could not reach the backend to analyse the recording.",
      });
    }
  }

  async function handleOpen(): Promise<void> {
    const previous = state;
    setState({ status: "opening" });
    const reply = await window.electronAPI.openVoiceResultFile();
    switch (reply.status) {
      case "cancelled":
        setState(previous);
        return;
      case "error":
        setState({ status: "error", message: reply.message });
        return;
      case "loaded": {
        const outcome = parseVoiceResults(reply.content);
        setState(
          outcome.ok
            ? {
                status: "loaded",
                fileName: reply.fileName,
                results: outcome.results,
                selected: 0,
              }
            : { status: "error", message: outcome.message },
        );
        return;
      }
      default: {
        const exhaustive: never = reply;
        return exhaustive;
      }
    }
  }

  const selected = state.status === "loaded" ? state.results[state.selected] : undefined;

  return (
    <div className="voice-results">
      <div className="voice-results__body">
        <PageHeader
          title="Voice screening"
          subtitle="Upload a sustained 'aaah' recording. The backend analyses it and deletes it straight after."
          actions={
            <>
              <input
                ref={fileInput}
                type="file"
                accept={VOICE_AUDIO_ACCEPT}
                hidden
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  event.target.value = ""; // the same file can be chosen again
                  if (file) void handleUpload(file);
                }}
              />
              <button
                type="button"
                className="btn btn--outline"
                onClick={handleOpen}
                disabled={busy}
              >
                {state.status === "opening" ? "Opening…" : "Open result file"}
              </button>
              <button
                type="button"
                className="btn btn--primary"
                onClick={() => fileInput.current?.click()}
                disabled={busy}
              >
                {state.status === "analysing" ? "Analysing…" : "Upload voice recording"}
              </button>
            </>
          }
        />
        {state.status === "analysing" && (
          <VoiceAnalysisProgress
            fileName={state.fileName}
            uploaded={state.uploaded}
            analysingSince={state.analysingSince}
          />
        )}
        {state.status === "error" && (
          <p className="voice-results__error" role="alert">
            {state.message}
          </p>
        )}

        {state.status === "loaded" && selected ? (
          state.results.length === 1 ? (
            <VoiceResultCard result={selected} />
          ) : (
            <div className="voice-results__layout">
              <nav className="voice-results__list" aria-label="People in this file">
                <p className="voice-results__file">{state.fileName}</p>
                {state.results.map((result, index) => (
                  <button
                    type="button"
                    key={`${result.person}-${index}`}
                    className={`voice-results__person${index === state.selected ? " voice-results__person--selected" : ""}`}
                    aria-current={index === state.selected}
                    onClick={() => setState({ ...state, selected: index })}
                  >
                    <span className="voice-results__person-id">{result.person}</span>
                    <span className="voice-results__person-summary">{resultSummary(result)}</span>
                  </button>
                ))}
              </nav>
              <VoiceResultCard result={selected} />
            </div>
          )
        ) : (
          state.status !== "error" &&
          state.status !== "analysing" && (
            <div className="voice-results__empty">
              <span className="voice-results__mic" aria-hidden="true">
                <svg
                  width="28"
                  height="28"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                >
                  <rect x="9" y="3" width="6" height="11" rx="3" />
                  <path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21M8.5 21h7" />
                </svg>
              </span>
              <h1>Screen a voice recording</h1>
              <p className="voice-results__lead">
                Upload one person saying a steady &ldquo;aaah&rdquo;. You get a referral priority
                and the reasons behind it in a few seconds.
              </p>
              <ol className="voice-results__tips">
                <li>
                  <strong>Quiet room.</strong> No fan, TV or other voices. Leave about one second of
                  silence before the &ldquo;aaah&rdquo;, so the noise can be checked.
                </li>
                <li>
                  <strong>One long &ldquo;aaah&rdquo;</strong> at a comfortable pitch, for about 5
                  seconds.
                </li>
                <li>
                  <strong>Phone about 20 cm from the mouth.</strong> WAV is best; MP3 and M4A also
                  work.
                </li>
                <li>
                  <strong>Natural voice, on a normal day.</strong> Don&rsquo;t force or change the
                  voice. A cold, sore throat or breathing problem can raise the score, so note it or
                  record another day.
                </li>
              </ol>
              <button
                type="button"
                className="btn btn--primary voice-results__cta"
                onClick={() => fileInput.current?.click()}
              >
                Upload voice recording
              </button>
              <p className="note">
                The recording is deleted straight after the analysis and no result is stored. To
                view a result the voice module already wrote, use <strong>Open result file</strong>.
              </p>
            </div>
          )
        )}
      </div>
    </div>
  );
}
