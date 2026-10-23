import { API_BASE_URL } from "../constants/config";
import type { GaitAnalysisList, GaitAnalysisStatus, GaitResult } from "../gait/gaitTypes";
import { ApiError, apiFetch, refreshAccessToken, requestHeaders, toApiError } from "./client";

const BASE = "/api/v1/gait/analyses";
const UNAUTHORIZED = "HTTP_401_UPLOAD"; // internal: the upload hit an expired token

export interface GaitUploadInput {
  file: File; // one walking video (.mp4, .mov, .m4v or .avi)
  onePersonConfirmed: boolean; // the clinician's tick; the backend refuses an upload without it
  subjectLabel?: string; // the same pseudonymous label as MRI uploads; never a name
}

/** Uploads the video with progress (fetch cannot report upload progress, XHR can). */
export function uploadGaitVideo(
  input: GaitUploadInput,
  onProgress: (fraction: number) => void,
): Promise<{ id: string }> {
  const form = new FormData();
  form.append("file", input.file, input.file.name);
  form.append("one_person_confirmed", input.onePersonConfirmed ? "true" : "false");
  if (input.subjectLabel) form.append("subject_label", input.subjectLabel);

  return sendUpload(form, onProgress).catch(async (error: unknown) => {
    // An expired access token: renew once, then send the same form again.
    if (error instanceof ApiError && error.code === UNAUTHORIZED && (await refreshAccessToken())) {
      return sendUpload(form, onProgress);
    }
    throw error;
  });
}

function sendUpload(form: FormData, onProgress: (fraction: number) => void): Promise<{ id: string }> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE_URL}${BASE}`);
    for (const [name, value] of Object.entries(requestHeaders(form))) {
      xhr.setRequestHeader(name, value);
    }
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    };
    xhr.onload = () => {
      let body: unknown = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        body = null;
      }
      if (xhr.status === 202) resolve(body as { id: string });
      else if (xhr.status === 401) reject(new ApiError(UNAUTHORIZED, "Your sign-in has expired."));
      else reject(toApiError(body, `Upload failed with status ${xhr.status}`));
    };
    xhr.onerror = () =>
      reject(
        new ApiError(
          "NETWORK_ERROR",
          `The analysis service is not reachable. Nothing was sent. Check that the backend is running at ${API_BASE_URL}, then try again. Your video and your tick are kept.`,
        ),
      );
    xhr.send(form);
  });
}

export function getGaitStatus(id: string): Promise<GaitAnalysisStatus> {
  return apiFetch<GaitAnalysisStatus>(`${BASE}/${encodeURIComponent(id)}`);
}

export function getGaitResult(id: string): Promise<GaitResult> {
  return apiFetch<GaitResult>(`${BASE}/${encodeURIComponent(id)}/result`);
}

/** The overlay video as a blob (a video element cannot send the sign-in header itself). */
export async function getGaitOverlay(id: string): Promise<Blob> {
  const path = `${BASE}/${encodeURIComponent(id)}/overlay`;
  const send = () => fetch(`${API_BASE_URL}${path}`, { headers: requestHeaders(undefined) });
  let response = await send();
  if (response.status === 401 && (await refreshAccessToken())) response = await send();
  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null);
    throw toApiError(body, `The overlay video could not be loaded (status ${response.status})`);
  }
  return response.blob();
}

export function listGaitAnalyses(limit = 20): Promise<GaitAnalysisList> {
  return apiFetch<GaitAnalysisList>(`${BASE}?limit=${limit}`);
}

/** Stops a queued or running gait analysis. The backend records it as failed ("stopped by the
 *  clinician") at once and ends the work within about a second. An ended analysis is unchanged. */
export function stopGaitAnalysis(id: string): Promise<GaitAnalysisStatus> {
  return apiFetch<GaitAnalysisStatus>(`${BASE}/${encodeURIComponent(id)}/stop`, { method: "POST" });
}
