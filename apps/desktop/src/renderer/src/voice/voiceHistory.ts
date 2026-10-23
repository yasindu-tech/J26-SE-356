import { useSyncExternalStore } from "react";
import type { VoiceResult } from "./voiceResult";

// The backend keeps no voice result (the recording and the result are deleted straight after the
// analysis), so Home cannot list voice runs from the server. This keeps what happened in this
// session, in memory only: how each voice run ended and the result that was already shown, with
// the pseudonymous ID, so the result can be opened again from Home. No recording, no name, nothing
// written to disk. It is gone when the app closes. Home labels these "this session".

export type VoiceRunStatus = "scored" | "abstained" | "failed";

/** How a voice analysis ended: a result (scored or not), or a failure with no result. */
export type VoiceOutcome = { kind: "result"; result: VoiceResult } | { kind: "failed"; message: string };

export interface VoiceRun {
  id: number;
  /** The assessment's pseudonymous ID, or null when none was given. */
  label: string | null;
  /** Milliseconds since the epoch, when the analysis ended. */
  at: number;
  status: VoiceRunStatus;
  outcome: VoiceOutcome;
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

export function voiceStatusOf(outcome: VoiceOutcome): VoiceRunStatus {
  if (outcome.kind === "failed") return "failed";
  return outcome.result.kind === "scored" ? "scored" : "abstained";
}

/** Records a finished voice run; returns its id (to open it again from Home). */
export function recordVoiceRun(label: string | null, outcome: VoiceOutcome, at = Date.now()): number {
  const id = nextId++;
  runs = [{ id, label, at, status: voiceStatusOf(outcome), outcome }, ...runs].slice(0, KEEP);
  emit();
  return id;
}

export function findVoiceRun(id: number): VoiceRun | undefined {
  return runs.find((r) => r.id === id);
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
