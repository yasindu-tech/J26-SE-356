# apps/desktop — Clinician workstation (Electron)

Where the non-specialist clinician creates an assessment, pairs a phone, uploads
an MRI if one exists, and reads the result.

## Stack
Electron · TypeScript · React (renderer) · Vite

## Structure
```
src/
  main/             Electron main process — windows, IPC, file access
  preload/          contextBridge — the ONLY surface the renderer can call
  renderer/src/
    api/            Backend client (client.ts, auth.api.ts, health.api.ts, session.ts)
    constants/      API_BASE_URL, session timeouts — no magic values inline
    pages/          components/   screens and UI
```
`queries/` (TanStack Query) and a router land once there's a second screen to
route to — see API-Standards.md §2.1 in the research vault for the target
layout.

## Rules
- **`contextIsolation: true`, `nodeIntegration: false`, `sandbox: true`.** The
  renderer never touches Node directly; everything goes through an explicit
  preload bridge, which itself only uses `contextBridge`/`ipcRenderer`.
- MRI upload accepts NIfTI/DICOM and stays **local** unless explicitly sent to
  the backend. No cloud upload of imaging without documented approval.
- The results view must always show, together:
  a **risk score**, a **modality breakdown**, and a **confidence interval that
  widens when inputs were missing**. Showing the score alone is not permitted —
  it is the explanation that makes the referral actionable.
- UI copy says **"screening"**, **"triage"**, **"prioritised referral"** —
  never "diagnosis", "positive", or "negative".
- Low risk renders as *not prioritised*, never *discharged* or *cleared*.

## MRI screen (`pages/MriAnalysis.tsx`)

After sign-in, the MRI tab works like this:

1. **Upload:** choose a zip of one person's DICOM study, then enter age and sex. The zip
   is sent to the backend only when the clinician presses *Upload and analyse*, which is
   the explicit send the rule above asks for. The backend deletes the DICOM files once
   the analysis ends.
2. **Progress:** the screen polls `GET /api/v1/mri/analyses/{id}` every 3 s and shows the
   steps: convert, T1, DTI, features, score, explain. DTI can take an hour or more.
   Earlier analyses are listed on the left.
3. **Result:** the screen shows the MRI module's output:
   - which parts (T1/DTI) were used;
   - the raw model score in log-odds with its 95% interval on an axis whose 0 is the
     model's boundary (deliberately not a percentage: it is not calibrated, and fusion
     decides how it is combined);
   - the features that moved the score most (TreeSHAP);
   - the harmonised feature vector, with a *Copy as JSON* button for the fusion work;
   - the series used, with their checks and QC pictures;
   - the model card.

   If the module abstains, it says why.

Backend setup for this screen is in `services/backend/README.md` (MRI analyses).

## Finger-tapping part (`tapping/`)

Finger tapping is the third part of the assessment, next to MRI and Gait, and works like Gait. In
the Modalities step the clinician chooses a phone video (MP4, MOV or M4V, up to 200 MB). It is
uploaded to the backend (`POST /api/v1/tapping/analyses`, see `services/backend/README.md`),
analysed there in the background by `models/tapping/src/predict_clip.py`, and the screen polls
the status like the gait screen does. The result is stored in PostgreSQL, so it appears under
**Recent assessments** on Home and can be reopened from there, and the analysis keeps running if
the clinician leaves the screen. The backend must run natively (not in the Docker `backend`
container) because the tapping model needs `lightgbm`, `mediapipe` and `shap`.

The result shows a model score (not a probability, no priority, no combined result), the five
measurements that moved it, the thumb-index signal, the per-tap amplitude and a one-page PDF
(`Download PDF` saves `summary_<clip ref>.pdf` through the main process). The file name of the
video is never shown; only its hashed ref is. The "Open the finger-tapping demo" link (the
Streamlit research demo, `main/tappingWindow.ts`) is kept. The tapping IPC contract is in
`shared/tappingIpc.ts`; the renderer reaches it through `tapping/tappingBridge.ts`.

## Setup
```bash
cp .env.example .env   # set VITE_API_BASE_URL if the backend isn't on localhost:8000
npm install
npm run dev --workspace apps/desktop
```

`VITE_API_BASE_URL` is also injected into `src/renderer/index.html`'s CSP
`connect-src` at build time (Vite's `%VITE_*%` HTML replacement) — the
renderer can only `fetch()` that one origin, nothing else.
