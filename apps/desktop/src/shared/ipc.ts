// The IPC contract between main, preload and renderer. Both sides import these
// types, so a renamed channel or a changed reply shape fails the typecheck
// instead of failing silently at runtime.

export const IPC_OPEN_VOICE_RESULT = "voice-result:open";

/** Reply to "open a voice result file". `content` is parsed JSON, still
 * unvalidated: the renderer checks its shape before showing anything. */
export type OpenResultFileReply =
  | { status: "cancelled" }
  | { status: "loaded"; fileName: string; content: unknown }
  | { status: "error"; message: string };

/** Everything the preload script exposes on `window.electronAPI`. */
export interface ElectronAPI {
  openVoiceResultFile(): Promise<OpenResultFileReply>;
}
