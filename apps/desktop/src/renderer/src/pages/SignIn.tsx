import { useState } from "react";
import logoFull from "../assets/pd-xai-logo-full.jpeg";
import logoMark from "../assets/pd-xai-logo-mark.jpeg";
import { ApiError } from "../api/client";
import { login, type TokenPair } from "../api/auth.api";
import { clearSession, startSession } from "../api/session";
import { ConnectionStatus } from "../components/ConnectionStatus";
import { PriorityRangePreview } from "../components/PriorityRangePreview";
import { SESSION_IDLE_TIMEOUT_MINUTES } from "../constants/config";
import { Workspace } from "./Workspace";
import "./SignIn.css";

export function SignIn() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [status, setStatus] = useState<"idle" | "pending" | "error">("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [tokens, setTokens] = useState<TokenPair | null>(null);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setStatus("pending");
    setErrorMessage(null);
    try {
      const issuedTokens = await login(username, password);
      startSession(issuedTokens, () => {
        setTokens(null);
        setPassword("");
      });
      setTokens(issuedTokens);
      setStatus("idle");
    } catch (error) {
      setStatus("error");
      setErrorMessage(
        error instanceof ApiError ? error.message : "Could not reach the sign-in service.",
      );
    }
  }

  function handleSignOut(): void {
    clearSession();
    setTokens(null);
    setPassword("");
  }

  if (tokens) {
    // No further screens exist yet (patient management, assessments — see
    // Desktop-App-Spec.md) — this placeholder just proves the session works
    // end to end (real tokens issued, idle sign-out armed) until they do.
    // PP1: the module screens (MRI, voice) stand in for them for now.
    return <Workspace clinician={username.trim() || "Clinician"} onSignOut={handleSignOut} />;
  }

  return (
    <div className="sign-in">
      <aside className="sign-in__brand-panel">
        <div className="sign-in__brand-mark">
          <img src={logoMark} alt="" className="sign-in__brand-mark-icon" />
          <span className="sign-in__brand-mark-text">PD-XAI</span>
        </div>

        <h1 className="sign-in__headline">
          Prioritises patients for neurology review. It does not diagnose.
        </h1>
        <p className="sign-in__lede">
          Combines a phone-captured gait, finger-tapping and voice assessment with an
          optional MRI into a calibrated referral priority — always shown with its
          confidence interval, what was missing, and why the model reached it.
        </p>

        <PriorityRangePreview />

        <footer className="sign-in__brand-footer">
          <p>For use by clinicians during a consultation, alongside their own assessment.</p>
          <p className="sign-in__build-tag">
            Build {__APP_VERSION__} · PD-XAI desktop
          </p>
        </footer>
      </aside>

      <main className="sign-in__form-panel">
        <div className="sign-in__form-panel-top">
          <ConnectionStatus />
        </div>

        <div className="sign-in__form-column">
          <img src={logoFull} alt="PD-XAI" className="sign-in__logo-full" />

          <div className="sign-in__card">
            <h2 className="sign-in__card-title">Sign in</h2>
            <p className="sign-in__card-subtitle">
              Use your clinic account. Results are attributed to the signed-in clinician.
            </p>

            <form onSubmit={handleSubmit} className="sign-in__form" noValidate>
              <label className="sign-in__field" htmlFor="clinic-username">
                <span className="sign-in__field-label">Clinic username</span>
                <input
                  id="clinic-username"
                  name="username"
                  type="text"
                  autoComplete="username"
                  value={username}
                  onChange={(event) => setUsername(event.target.value)}
                  className="sign-in__input"
                />
              </label>

              <div className="sign-in__field-row">
                <span className="sign-in__field-label">Password</span>
                <span className="sign-in__link sign-in__link--disabled" aria-disabled="true" title="Not available yet">
                  Forgotten password?
                </span>
              </div>
              <input
                id="clinic-password"
                name="password"
                type="password"
                autoComplete="current-password"
                aria-label="Password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                className="sign-in__input"
              />

              {status === "error" && errorMessage && (
                <p className="sign-in__error" role="alert">
                  {errorMessage}
                </p>
              )}

              <button type="submit" className="sign-in__submit" disabled={status === "pending"}>
                {status === "pending" ? "Signing in…" : "Sign in"}
              </button>
            </form>

            <div className="sign-in__divider">
              <span>or</span>
            </div>

            <button
              type="button"
              className="sign-in__smart-card"
              disabled
              title="Not available yet"
            >
              <SmartCardIcon />
              Sign in with smart card
            </button>

            <div className="sign-in__notice">
              <InfoIcon />
              <p>
                This is a shared consultation-room computer. You will be signed out after{" "}
                {SESSION_IDLE_TIMEOUT_MINUTES} minutes without activity, and no
                patient-identifiable data is stored on this device.
              </p>
            </div>
          </div>

          <div className="sign-in__meta">
            <span
              className="sign-in__link sign-in__link--disabled"
              aria-disabled="true"
              title="Not available yet"
            >
              Help &amp; support
            </span>
          </div>
        </div>
      </main>
    </div>
  );
}

function SmartCardIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <rect x="2" y="5" width="20" height="14" rx="2" stroke="currentColor" strokeWidth="1.6" />
      <rect x="5" y="8" width="5" height="4" rx="0.8" stroke="currentColor" strokeWidth="1.4" />
      <line x1="13" y1="9" x2="19" y2="9" stroke="currentColor" strokeWidth="1.4" />
      <line x1="13" y1="12" x2="19" y2="12" stroke="currentColor" strokeWidth="1.4" />
      <line x1="5" y1="16" x2="10" y2="16" stroke="currentColor" strokeWidth="1.4" />
    </svg>
  );
}

function InfoIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.6" />
      <line x1="12" y1="11" x2="12" y2="16" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      <circle cx="12" cy="8" r="1" fill="currentColor" />
    </svg>
  );
}
