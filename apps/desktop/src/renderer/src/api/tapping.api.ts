import { API_BASE_URL } from "../constants/config";
import type { TappingAnalysisList, TappingAnalysisStatus, TappingTarget } from "../tapping/tappingTypes";
import { ApiError, apiFetch, refreshAccessToken, requestHeaders, toApiError } from "./client";

const BASE = "/api/v1/tapping/analyses";
const UNAUTHORIZED = "HTTP_401_UPLOAD"; // internal: the upload hit an expired token

export interface TappingUploadInput {
  file: File; // one finger-tapping video (.mp4, .mov or .m4v)
  target?: TappingTarget; // the study target; ge2 is the primary one
  subjectLabel?: string; // the same pseudonymous label as MRI and gait uploads; never a name
}

/** Uploads the video with progress (fetch cannot report upload progress, XHR can). */
export function uploadTappingVideo(
  input: TappingUploadInput,
  onProgress: (fraction: number) => void,
): Promise<{ id: string }> {
  const form = new FormData();
  form.append("file", input.file, input.file.name);
  form.append("target", input.target ?? "ge2");
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
          `The analysis service is not reachable. Nothing was sent. Check that the backend is running at ${API_BASE_URL}, then try again. Your video is kept.`,
        ),
      );
    xhr.send(form);
  });
}

export function getTappingStatus(id: string): Promise<TappingAnalysisStatus> {
  return apiFetch<TappingAnalysisStatus>(`${BASE}/${encodeURIComponent(id)}`);
}

/** The stored result, still unchecked: the caller validates its shape before showing anything. */
export function getTappingResult(id: string): Promise<unknown> {
  return apiFetch<unknown>(`${BASE}/${encodeURIComponent(id)}/result`);
}

export function listTappingAnalyses(limit = 20): Promise<TappingAnalysisList> {
  return apiFetch<TappingAnalysisList>(`${BASE}?limit=${limit}`);
}

/** Stops a queued or running finger-tapping analysis. The backend records it as failed ("stopped by the
 *  clinician") at once and ends the work within about a second. An ended analysis is unchanged. */
export function stopTappingAnalysis(id: string): Promise<TappingAnalysisStatus> {
  return apiFetch<TappingAnalysisStatus>(`${BASE}/${encodeURIComponent(id)}/stop`, { method: "POST" });
}
