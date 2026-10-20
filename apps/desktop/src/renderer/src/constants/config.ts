// Must stay an exact origin (scheme + host [+ port]), no path — it's also
// substituted into index.html's CSP connect-src via Vite's %VITE_*% HTML
// env replacement, so the two can't drift apart.
export const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL;

// Matches the backend's access_token_expire_minutes (services/backend/app/core/config.py).
// Keep the two in sync — a longer idle timeout than the access token's own
// lifetime would just mean silent failures a few minutes before sign-out.
export const SESSION_IDLE_TIMEOUT_MINUTES = 15;

// MRI analyses (pages/MriAnalysis.tsx). DTI can run for an hour or more, so the
// screen polls the backend instead of holding a request open.
export const MRI_POLL_INTERVAL_MS = 3000;
export const MRI_MAX_UPLOAD_MB = 4096; // matches the backend's mri_max_upload_mb
