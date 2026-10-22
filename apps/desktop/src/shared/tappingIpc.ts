// The finger-tapping IPC contract (FT-61 demo, FT-63 analysis): channel names and the types
// main, preload and renderer share. Kept apart from shared/ipc.ts, which the team owns.

import type { TappingResult, TappingTarget } from "./tappingResult";

// Finger-tapping demo (Streamlit in a child process, shown in a view of the main window).
export const IPC_TAPPING_START = "tapping:start";
export const IPC_TAPPING_STOP = "tapping:stop";
export const IPC_TAPPING_STATUS = "tapping:status";
export const IPC_TAPPING_SET_BOUNDS = "tapping:set-bounds";
export const IPC_TAPPING_OPEN_EXTERNAL = "tapping:open-external";
export const IPC_TAPPING_STATUS_EVENT = "tapping:status-changed"; // main -> renderer push

// Finger-tapping analysis inside the assessment (FT-63): one video in, one checked result out.
export const IPC_TAPPING_ANALYSE = "tapping-analysis:run";
export const IPC_TAPPING_CANCEL = "tapping-analysis:cancel";
export const IPC_TAPPING_PROGRESS = "tapping-analysis:progress"; // main -> renderer push
export const IPC_TAPPING_SAVE_PDF = "tapping-analysis:save-pdf";

export type TappingState = "starting" | "ready" | "unavailable" | "error";

/** `reason` is plain English for the clinician: no file paths, no IDs. */
export interface TappingStatus {
  state: TappingState;
  reason?: string;
}

/** Where the demo view sits in the window, in CSS pixels. */
export interface TappingBounds {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface TappingAPI {
  start(): Promise<TappingStatus>;
  stop(): Promise<void>;
  getStatus(): Promise<TappingStatus>;
  setBounds(bounds: TappingBounds): void;
  /** Opens the demo's own local address in the system browser (the renderer cannot pick a URL). */
  openExternal(): Promise<boolean>;
  /** Returns the function that removes the listener. */
  onStatus(listener: (status: TappingStatus) => void): () => void;
}

export interface TappingAnalysisRequest {
  /** The video bytes: the renderer never passes a file path, so main cannot be asked to read one. */
  bytes: Uint8Array;
  /** ".mp4", ".mov" or ".m4v". */
  extension: string;
  target: TappingTarget;
}

export interface TappingProgress {
  stage: number;
  total: number;
  label: string;
}

export type TappingErrorCode = "busy" | "unavailable" | "invalid" | "timeout" | "stopped" | "cancelled";

/** `message` is plain English for the clinician: no file paths, no IDs. */
export type TappingAnalysisReply =
  | { kind: "result"; result: TappingResult }
  | { kind: "error"; code: TappingErrorCode; message: string };

export type TappingSavePdfReply =
  | { status: "saved" }
  | { status: "cancelled" }
  | { status: "error"; message: string };

export interface TappingAnalysisAPI {
  analyse(request: TappingAnalysisRequest): Promise<TappingAnalysisReply>;
  cancel(): Promise<void>;
  /** Returns the function that removes the listener. */
  onProgress(listener: (progress: TappingProgress) => void): () => void;
  /** Save dialog, then the PDF bytes; the file name is chosen from the hashed ref only. */
  savePdf(ref: string, pdfBase64: string): Promise<TappingSavePdfReply>;
}
