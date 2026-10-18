import type { EddyProfile } from "../mri/mriTypes";

// Preferences kept on this computer (not on the backend, not per person). They hold no
// patient data and no credentials: the sign-in tokens stay in memory only (api/session.ts).
// Anything read back is validated, so a damaged or older value falls back to the default
// instead of breaking the app.

export type StartPage = "home" | "assessment";

export interface Settings {
  general: {
    /** Where the app opens after sign-in. */
    startPage: StartPage;
  };
  mri: {
    /** DTI motion correction for new analyses. "full" is what the model was trained on. */
    eddyProfile: EddyProfile;
    /** Ask before starting an analysis with the unvalidated fast setting. */
    confirmFast: boolean;
  };
}

export const DEFAULT_SETTINGS: Settings = {
  general: { startPage: "home" },
  mri: { eddyProfile: "full", confirmFast: true },
};

const STORAGE_KEY = "pdxai.desktop.settings.v1";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Reads a stored value, keeping every field that is valid and defaulting the rest. */
export function parseSettings(raw: unknown): Settings {
  const out: Settings = {
    general: { ...DEFAULT_SETTINGS.general },
    mri: { ...DEFAULT_SETTINGS.mri },
  };
  if (!isRecord(raw)) return out;
  const { general, mri } = raw;
  if (isRecord(general) && (general.startPage === "home" || general.startPage === "assessment")) {
    out.general.startPage = general.startPage;
  }
  if (isRecord(mri)) {
    if (mri.eddyProfile === "full" || mri.eddyProfile === "fast") out.mri.eddyProfile = mri.eddyProfile;
    if (typeof mri.confirmFast === "boolean") out.mri.confirmFast = mri.confirmFast;
  }
  return out;
}

export function loadSettings(): Settings {
  try {
    const text = window.localStorage.getItem(STORAGE_KEY);
    return parseSettings(text === null ? null : JSON.parse(text));
  } catch {
    return parseSettings(null); // storage unavailable or not JSON
  }
}

export function saveSettings(settings: Settings): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // Not saving is not an error for the clinician; the setting still applies this session.
  }
}
