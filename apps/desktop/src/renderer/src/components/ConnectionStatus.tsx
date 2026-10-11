import { useEffect, useState } from "react";
import { checkHealth } from "../api/health.api";
import "./ConnectionStatus.css";

const POLL_INTERVAL_MS = 15_000;

export function ConnectionStatus() {
  const [state, setState] = useState<{ ok: boolean; latencyMs: number } | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function poll(): Promise<void> {
      const result = await checkHealth();
      if (!cancelled) setState(result);
    }

    poll();
    const interval = setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const label =
    state === null
      ? "Checking backend…"
      : state.ok
        ? `Backend · connected · ${state.latencyMs} ms`
        : "Backend · unreachable";

  return (
    <div className="connection-status" data-ok={state?.ok ?? false}>
      <span className="connection-status__dot" />
      {label}
    </div>
  );
}
