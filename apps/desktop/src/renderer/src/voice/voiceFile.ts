// Checks on a chosen voice recording before anything is sent, so a wrong file is caught on
// the Modalities step, not after an upload. The backend checks the same things again
// (services/backend/app/schemas/voice.py VOICE_AUDIO_TYPES, config.py voice_max_upload_mb).

/** File types the backend accepts, lower case, with the dot. */
export const VOICE_EXTENSIONS = [".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".webm"] as const;

/** The backend's upload limit (voice_max_upload_mb). */
export const VOICE_MAX_UPLOAD_MB = 50;

export interface VoiceFileProblem {
  title: string;
  text: string;
}

/** What is wrong with this file as a voice recording, or null when it can be sent. */
export function voiceFileProblem(name: string, size: number): VoiceFileProblem | null {
  const dot = name.lastIndexOf(".");
  const extension = dot >= 0 ? name.slice(dot).toLowerCase() : "";
  if (!(VOICE_EXTENSIONS as readonly string[]).includes(extension)) {
    return {
      title: "Not an audio file the model can read",
      text: "Choose a WAV, MP3, M4A, FLAC, OGG, AAC or WebM recording.",
    };
  }
  if (size <= 0) {
    return { title: "The file is empty", text: "Record the “aaah” again and choose the new file." };
  }
  if (size > VOICE_MAX_UPLOAD_MB * 1024 * 1024) {
    return {
      title: "The file is too large",
      text: `Recordings up to ${VOICE_MAX_UPLOAD_MB} MB are accepted. One “aaah” of 3 to 10 seconds is far smaller.`,
    };
  }
  return null;
}

/** "WAV · 0.4 MB" for the file card and the Review step. */
export function voiceFileSummary(name: string, size: number): string {
  const dot = name.lastIndexOf(".");
  const kind = dot >= 0 ? name.slice(dot + 1).toUpperCase() : "Audio";
  const mb = size / (1024 * 1024);
  return `${kind} · ${mb < 1 ? mb.toFixed(1) : mb.toFixed(0)} MB`;
}
