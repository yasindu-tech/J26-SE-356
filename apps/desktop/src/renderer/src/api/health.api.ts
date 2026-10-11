import { apiFetch } from "./client";

interface HealthResponse {
  status: string;
}

export async function checkHealth(): Promise<{ ok: boolean; latencyMs: number }> {
  const start = performance.now();
  try {
    const body = await apiFetch<HealthResponse>("/health");
    return { ok: body.status === "ok", latencyMs: Math.round(performance.now() - start) };
  } catch {
    return { ok: false, latencyMs: Math.round(performance.now() - start) };
  }
}
