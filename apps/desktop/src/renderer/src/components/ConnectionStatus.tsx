import { useEffect, useState } from "react";
import { checkHealth } from "../api/health.api";
import "./ConnectionStatus.css";

const POLL_INTERVAL_MS = 15_000;

/** `onDark`: drawn on the teal top bar instead of a light background. */
export function ConnectionStatus({ onDark = false }: { onDark?: boolean }) {
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
    <div
      className={`connection-status${onDark ? " connection-status--on-dark" : ""}`}
      data-ok={state === null ? "pending" : state.ok}
    >
      <span className="connection-status__dot" />
      {label}
    </div>
  );
}
