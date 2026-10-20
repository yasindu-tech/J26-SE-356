import { API_BASE_URL } from "../constants/config";
import { ApiError, refreshAccessToken, requestHeaders, toApiError } from "./client";

/** Formats the backend accepts (services/backend/app/schemas/voice.py VOICE_AUDIO_TYPES). */
export const VOICE_AUDIO_ACCEPT = ".wav,.flac,.ogg,.mp3,.m4a,.aac,.webm";

const PATH = "/api/v1/voice/analyses";
const UNAUTHORIZED = "HTTP_401_UPLOAD"; // internal: the upload hit an expired token

/**
 * Sends one sustained "aaah" recording to the backend, which runs the voice module and
 * answers with the explained result (a few seconds). The backend deletes the recording
 * straight after and keeps no result. The reply is untrusted: callers pass it through
 * parseVoiceResults before showing anything.
 *
 * `onUploaded` reports the real share of the file sent (0 to 1). Once it reaches 1 the
 * backend is analysing; it sends no progress of its own, so the caller can only estimate.
 */
export function uploadVoiceRecording(
  file: File,
  onUploaded: (fraction: number) => void = () => {},
): Promise<unknown> {
  const form = new FormData();
  form.append("file", file, file.name);
  return send(form, onUploaded).catch(async (error: unknown) => {
    // An expired access token: renew once, then send the same form again (as mri.api.ts).
    if (error instanceof ApiError && error.code === UNAUTHORIZED && (await refreshAccessToken())) {
      return send(form, onUploaded);
    }
    throw error;
  });
}

/** XHR rather than fetch: fetch cannot report upload progress. */
function send(form: FormData, onUploaded: (fraction: number) => void): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE_URL}${PATH}`);
    for (const [name, value] of Object.entries(requestHeaders(form))) {
      xhr.setRequestHeader(name, value);
    }
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onUploaded(event.loaded / event.total);
    };
    xhr.upload.onload = () => onUploaded(1);
    xhr.onload = () => {
      let body: unknown = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        body = null;
      }
      if (xhr.status === 200) resolve(body);
      else if (xhr.status === 401) reject(new ApiError(UNAUTHORIZED, "Your sign-in has expired."));
      else reject(toApiError(body, `The analysis failed with status ${xhr.status}`));
    };
    xhr.onerror = () =>
      reject(
        new ApiError("NETWORK_ERROR", "Could not reach the backend to analyse the recording."),
      );
    xhr.send(form);
  });
}
