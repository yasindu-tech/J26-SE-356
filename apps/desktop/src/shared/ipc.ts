// The IPC contract between main, preload and renderer. Both sides import these
// types, so a renamed channel or a changed reply shape fails the typecheck
// instead of failing silently at runtime.

export const IPC_OPEN_VOICE_RESULT = "voice-result:open";

// Finger-tapping demo (Streamlit in a child process, shown in a view of the main window).
export const IPC_TAPPING_START = "tapping:start";
export const IPC_TAPPING_STOP = "tapping:stop";
export const IPC_TAPPING_STATUS = "tapping:status";
export const IPC_TAPPING_SET_BOUNDS = "tapping:set-bounds";
export const IPC_TAPPING_OPEN_EXTERNAL = "tapping:open-external";
export const IPC_TAPPING_STATUS_EVENT = "tapping:status-changed"; // main -> renderer push

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

/** Reply to "open a voice result file". `content` is parsed JSON, still
 * unvalidated: the renderer checks its shape before showing anything. */
export type OpenResultFileReply =
  | { status: "cancelled" }
  | { status: "loaded"; fileName: string; content: unknown }
  | { status: "error"; message: string };

/** Everything the preload script exposes on `window.electronAPI`. */
export interface ElectronAPI {
  openVoiceResultFile(): Promise<OpenResultFileReply>;
  tapping: TappingAPI;
}
