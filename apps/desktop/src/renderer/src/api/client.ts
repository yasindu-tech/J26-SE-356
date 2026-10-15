import { API_BASE_URL } from "../constants/config";
import { getAccessToken, getRefreshToken, replaceTokens } from "./session";
import type { TokenPair } from "./auth.api";

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

const REFRESH_PATH = "/api/v1/auth/refresh";
let refreshing: Promise<boolean> | null = null;

/**
 * Renews the access token with the refresh token (API-Standards.md §2.4).
 * Access tokens last 15 minutes; an MRI analysis is polled for much longer.
 * Requests that hit a 401 at the same moment share one renewal, because the
 * backend rotates refresh tokens and rejects a reused one.
 */
export function refreshAccessToken(): Promise<boolean> {
  refreshing ??= (async () => {
    const refreshToken = getRefreshToken();
    if (!refreshToken) return false;
    try {
      const response = await fetch(`${API_BASE_URL}${REFRESH_PATH}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: refreshToken }),
      });
      if (!response.ok) return false;
      replaceTokens((await response.json()) as TokenPair);
      return true;
    } catch {
      return false;
    }
  })().finally(() => {
    refreshing = null;
  });
  return refreshing;
}

/** Sign-in and renewal answer 401 for wrong credentials; never retried. */
function mayRetryAfterRefresh(path: string): boolean {
  return !path.startsWith("/api/v1/auth/");
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const send = () =>
    fetch(`${API_BASE_URL}${path}`, { ...init, headers: requestHeaders(init?.body, init?.headers) });
  let response = await send();
  if (response.status === 401 && mayRetryAfterRefresh(path) && (await refreshAccessToken())) {
    response = await send(); // once only
  }

  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null);
    throw toApiError(body, `Request to ${path} failed with status ${response.status}`);
  }

  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}
