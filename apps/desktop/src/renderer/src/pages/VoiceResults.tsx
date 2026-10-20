import { useState } from "react";
import { PageHeader } from "../components/PageHeader";
import { VoiceResultCard } from "../components/VoiceResultCard";
import { parseVoiceResults, triageLabel, type VoiceResult } from "../voice/voiceResult";
import "./VoiceResults.css";

// PP1 voice result screen (models/voice/README.md, definition of done item 8).
// There is no assessment or result API in the backend yet, so results come
// from a file the voice module wrote (demo_results.json); the file is only
// read, never stored or uploaded. Replace with the backend result endpoint
// once it exists.

type ScreenState =
  | { status: "empty" }
  | { status: "opening" }
  | { status: "error"; message: string }
  | { status: "loaded"; fileName: string; results: VoiceResult[]; selected: number };

function resultSummary(result: VoiceResult): string {
  return result.kind === "scored" ? triageLabel(result.triagePriority) : "Not scored";
}

export function VoiceResults() {
  const [state, setState] = useState<ScreenState>({ status: "empty" });

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
            ? { status: "loaded", fileName: reply.fileName, results: outcome.results, selected: 0 }
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
          title="Voice screening result"
          subtitle="Read from a file the voice module wrote. Nothing is stored or uploaded."
          actions={
            <button
              type="button"
              className="btn btn--primary"
              onClick={handleOpen}
              disabled={state.status === "opening"}
            >
              {state.status === "opening" ? "Opening…" : "Open result file"}
            </button>
          }
        />
        {state.status === "error" && (
          <p className="voice-results__error" role="alert">
            {state.message}
          </p>
        )}

        {state.status === "loaded" && selected ? (
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
        ) : (
          state.status !== "error" && (
            <div className="voice-results__empty">
              <h1>No result open</h1>
              <p>
                Open a voice result file written by the voice module, for example{" "}
                <code>models/voice/artifacts/demo_results.json</code> from{" "}
                <code>python models/voice/src/demo_voice.py</code>.
              </p>
              <p>The file is only read on this computer. It is not stored or uploaded.</p>
            </div>
          )
        )}
      </div>
    </div>
  );
}
