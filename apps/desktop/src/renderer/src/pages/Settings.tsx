import { PageHeader } from "../components/PageHeader";
import { API_BASE_URL, MRI_MAX_UPLOAD_MB } from "../constants/config";
import type { EddyProfile } from "../mri/mriTypes";
import { EDDY_PROFILE_LABELS, FAST_EDDY_WARNING } from "../mri/mriView";
import { DEFAULT_SETTINGS, type Settings, type StartPage } from "../settings/settings";
import "./Settings.css";

// Preferences for this computer. They apply to new work only: an analysis that already ran
// keeps the setting it was run with, and its result says which one that was.

export function SettingsPage({
  clinician,
  settings,
  onChange,
}: {
  clinician: string;
  settings: Settings;
  onChange: (next: Settings) => void;
}) {
  const changed = JSON.stringify(settings) !== JSON.stringify(DEFAULT_SETTINGS);

  return (
    <div className="settings">
      <PageHeader
        title="Settings"
        subtitle="Kept on this computer only. Nothing here is sent to the backend or tied to a person."
        actions={
          <button
            type="button"
            className="btn btn--outline"
            disabled={!changed}
            onClick={() => onChange(DEFAULT_SETTINGS)}
          >
            Reset to defaults
          </button>
        }
      />

      <section className="card" aria-labelledby="settings-general">
        <p className="eyebrow">General</p>
        <h2 className="card__title" id="settings-general">
          This app
        </h2>

        <label className="settings__row">
          <span className="settings__text">
            <span className="settings__label">Open on after sign-in</span>
            <span className="note">The first screen you see.</span>
          </span>
          <select
            className="mri__input"
            value={settings.general.startPage}
            onChange={(e) =>
              onChange({
                ...settings,
                general: { startPage: (e.target.value === "assessment" ? "assessment" : "home") as StartPage },
              })
            }
          >
            <option value="home">Home</option>
            <option value="assessment">New assessment</option>
          </select>
        </label>

        <dl className="settings__facts">
          <div>
            <dt>Signed in as</dt>
            <dd className="mono">{clinician}</dd>
          </div>
          <div>
            <dt>Backend</dt>
            <dd className="mono">{API_BASE_URL}</dd>
          </div>
          <div>
            <dt>App build</dt>
            <dd className="mono">{__APP_VERSION__}</dd>
          </div>
        </dl>
        <p className="note">
          The backend address is fixed when the app is built, so it cannot be changed here. You are
          signed out automatically after a period without activity.
        </p>
      </section>

      <section className="card" aria-labelledby="settings-mri">
        <p className="eyebrow">MRI</p>
        <h2 className="card__title" id="settings-mri">
          MRI analysis
        </h2>

        <label className="settings__row">
          <span className="settings__text">
            <span className="settings__label">DTI motion correction</span>
            <span className="note">
              Used for every new MRI analysis. The standard (full) setting is the one the model was
              trained with; fast is quicker but has not been checked against it.
            </span>
          </span>
          <select
            className="mri__input"
            value={settings.mri.eddyProfile}
            onChange={(e) =>
              onChange({
                ...settings,
                mri: { ...settings.mri, eddyProfile: (e.target.value === "fast" ? "fast" : "full") as EddyProfile },
              })
            }
          >
            {(Object.keys(EDDY_PROFILE_LABELS) as EddyProfile[]).map((profile) => (
              <option key={profile} value={profile}>
                {EDDY_PROFILE_LABELS[profile]}
              </option>
            ))}
          </select>
        </label>
        {settings.mri.eddyProfile === "fast" && (
          <p className="mri__notice" role="note">
            {FAST_EDDY_WARNING}
          </p>
        )}

        <label className="settings__row settings__row--check">
          <input
            type="checkbox"
            checked={settings.mri.confirmFast}
            onChange={(e) => onChange({ ...settings, mri: { ...settings.mri, confirmFast: e.target.checked } })}
          />
          <span className="settings__text">
            <span className="settings__label">Ask me before starting a fast analysis</span>
            <span className="note">
              Shows a reminder that the fast setting is not validated, and asks you to confirm.
            </span>
          </span>
        </label>

        <dl className="settings__facts">
          <div>
            <dt>Largest upload</dt>
            <dd className="mono">{MRI_MAX_UPLOAD_MB} MB</dd>
          </div>
        </dl>
        <p className="note">
          An analysis that already ran keeps the setting it used, and its result shows which one that
          was.
        </p>
      </section>
    </div>
  );
}
