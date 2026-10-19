import { API_BASE_URL } from "../constants/config";
import type {
  EddyProfile,
  MriAnalysisCreated,
  MriAnalysisList,
  MriAnalysisStatus,
  MriResult,
} from "../mri/mriTypes";
import { ApiError, apiFetch, refreshAccessToken, requestHeaders, toApiError } from "./client";

const BASE = "/api/v1/mri/analyses";
const UNAUTHORIZED = "HTTP_401_UPLOAD"; // internal: the upload hit an expired token

export interface MriUploadInput {
  file: File; // a .zip of one person's DICOM files
  age: number;
  sex: "M" | "F";
  eddyProfile: EddyProfile; // "full" unless the clinician chose "fast"
  subjectLabel?: string; // pseudonymous label (see isValidSubjectLabel); omitted when empty
}

/** Uploads the zip with progress (fetch cannot report upload progress, XHR can). */
export function uploadMriStudy(
  input: MriUploadInput,
  onProgress: (fraction: number) => void,
): Promise<MriAnalysisCreated> {
  const form = new FormData();
  form.append("file", input.file, input.file.name);
  form.append("age", String(input.age));
  form.append("sex", input.sex);
  form.append("eddy_profile", input.eddyProfile);
  if (input.subjectLabel) form.append("subject_label", input.subjectLabel);

  return sendUpload(form, onProgress).catch(async (error: unknown) => {
    // An expired access token: renew once, then send the same form again.
    if (error instanceof ApiError && error.code === UNAUTHORIZED && (await refreshAccessToken())) {
      return sendUpload(form, onProgress);
    }
    throw error;
  });
}


function sendUpload(
  form: FormData,
  onProgress: (fraction: number) => void,
): Promise<MriAnalysisCreated> {
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
      if (xhr.status === 202) resolve(body as MriAnalysisCreated);
      else if (xhr.status === 401) reject(new ApiError(UNAUTHORIZED, "Your sign-in has expired."));
      else reject(toApiError(body, `Upload failed with status ${xhr.status}`));
    };
    xhr.onerror = () =>
      reject(new ApiError("NETWORK_ERROR", "Could not reach the backend to upload the scan."));
    xhr.send(form);
  });
}

export function getMriStatus(id: string): Promise<MriAnalysisStatus> {
  return apiFetch<MriAnalysisStatus>(`${BASE}/${encodeURIComponent(id)}`);
}

export function getMriResult(id: string): Promise<MriResult> {
  return apiFetch<MriResult>(`${BASE}/${encodeURIComponent(id)}/result`);
}

export function listMriAnalyses(limit = 20): Promise<MriAnalysisList> {
  return apiFetch<MriAnalysisList>(`${BASE}?limit=${limit}`);
}
