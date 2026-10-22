import { useSyncExternalStore } from "react";

// The backend keeps no voice result (the recording and the result are deleted straight after the
// analysis), so Home cannot list voice runs from the server. This keeps what happened in this
// session, in memory only: how each voice run ended, with the pseudonymous ID. No result, no
// recording, no name. It is gone when the app closes. Home labels these "this session".

export type VoiceRunStatus = "scored" | "abstained" | "failed";

export interface VoiceRun {
  id: number;
  /** The assessment's pseudonymous ID, or null when none was given. */
  label: string | null;
  /** Milliseconds since the epoch, when the analysis ended. */
  at: number;
  status: VoiceRunStatus;
}

const KEEP = 100;

let runs: readonly VoiceRun[] = [];
let nextId = 1;
const listeners = new Set<() => void>();

function emit(): void {
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function recordVoiceRun(label: string | null, status: VoiceRunStatus, at = Date.now()): void {
  runs = [{ id: nextId++, label, at, status }, ...runs].slice(0, KEEP);
  emit();
}

export function clearVoiceRuns(): void {
  runs = [];
  emit();
}

export function voiceRuns(): readonly VoiceRun[] {
  return runs;
}

/** The voice runs of this session, newest first; re-renders when one is added. */
export function useVoiceRuns(): readonly VoiceRun[] {
  return useSyncExternalStore(subscribe, voiceRuns, voiceRuns);
}
