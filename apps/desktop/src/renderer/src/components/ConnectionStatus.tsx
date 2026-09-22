import "./ConnectionStatus.css";

// Placeholder until the backend health check exists (see Backend-Spec in the
// research vault) — wire this to a real `GET /health` poll via the preload
// bridge once that endpoint ships, rather than reporting a static status.
export function ConnectionStatus() {
  return (
    <div className="connection-status">
      <span className="connection-status__dot" />
      Cloud analysis service · connected · 84 ms
    </div>
  );
}
