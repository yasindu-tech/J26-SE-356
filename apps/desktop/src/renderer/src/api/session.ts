import { SESSION_IDLE_TIMEOUT_MINUTES } from "../constants/config";
import type { TokenPair } from "./auth.api";

// Access token: memory only (API-Standards.md §2.4) — never localStorage.
// The refresh token isn't persisted yet either (no OS-keychain bridge in the
// preload layer yet — see apps/desktop/README.md), so a restart currently
// requires signing in again. That's a known, deliberate gap, not an oversight.
let currentTokens: TokenPair | null = null;
let idleTimer: ReturnType<typeof setTimeout> | null = null;
let onIdleSignOut: (() => void) | null = null;

const IDLE_TIMEOUT_MS = SESSION_IDLE_TIMEOUT_MINUTES * 60 * 1000;
const ACTIVITY_EVENTS = ["mousedown", "keydown", "wheel"] as const;

function resetIdleTimer(): void {
  if (idleTimer) clearTimeout(idleTimer);
  idleTimer = setTimeout(() => {
    clearSession();
    onIdleSignOut?.();
  }, IDLE_TIMEOUT_MS);
}

/** Starts a session: holds the tokens in memory and begins the idle clock.
 * `onTimeout` is called once, when the clinician is signed out for inactivity. */
export function startSession(tokens: TokenPair, onTimeout: () => void): void {
  currentTokens = tokens;
  onIdleSignOut = onTimeout;
  resetIdleTimer();
  for (const event of ACTIVITY_EVENTS) {
    window.addEventListener(event, resetIdleTimer);
  }
}

export function clearSession(): void {
  currentTokens = null;
  onIdleSignOut = null;
  if (idleTimer) clearTimeout(idleTimer);
  idleTimer = null;
  for (const event of ACTIVITY_EVENTS) {
    window.removeEventListener(event, resetIdleTimer);
  }
}

export function getAccessToken(): string | null {
  return currentTokens?.access_token ?? null;
}

export function getRefreshToken(): string | null {
  return currentTokens?.refresh_token ?? null;
}

/** Swaps in a renewed token pair (the backend rotates the refresh token too).
 * Ignored after sign-out, so a late renewal cannot bring a session back. */
export function replaceTokens(tokens: TokenPair): void {
  if (currentTokens) currentTokens = tokens;
}
