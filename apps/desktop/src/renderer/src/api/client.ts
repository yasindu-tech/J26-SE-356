import { API_BASE_URL } from "../constants/config";
import { getAccessToken } from "./session";

// Mirrors services/backend/app/core/errors.py's envelope:
// { "error": { "code", "message", "details" } }
export class ApiError extends Error {
  constructor(
    public readonly code: string,
    message: string,
    public readonly details: unknown = null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** Headers every request sends: JSON (unless the body is a form), plus the
 * signed-in clinician's access token when there is one. */
export function requestHeaders(body: unknown, extra?: HeadersInit): Record<string, string> {
  const token = getAccessToken();
  return {
    ...(body instanceof FormData ? {} : { "Content-Type": "application/json" }),
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...(extra as Record<string, string> | undefined),
  };
}

/** The shared error envelope -> ApiError (also used by the XHR upload). */
export function toApiError(body: unknown, fallback: string): ApiError {
  const error =
    typeof body === "object" && body !== null && "error" in body
      ? (body as { error?: { code?: string; message?: string; details?: unknown } }).error
      : undefined;
  return new ApiError(
    error?.code ?? "UNKNOWN_ERROR",
    error?.message ?? fallback,
    error?.details ?? null,
  );
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: requestHeaders(init?.body, init?.headers),
  });

  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null);
    throw toApiError(body, `Request to ${path} failed with status ${response.status}`);
  }

  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}
