import { API_BASE_URL } from "../constants/config";

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

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const error = body?.error;
    throw new ApiError(
      error?.code ?? "UNKNOWN_ERROR",
      error?.message ?? `Request to ${path} failed with status ${response.status}`,
      error?.details ?? null,
    );
  }

  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}
