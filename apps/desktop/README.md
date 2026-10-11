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

## Setup
```bash
cp .env.example .env   # set VITE_API_BASE_URL if the backend isn't on localhost:8000
npm install
npm run dev --workspace apps/desktop
```

`VITE_API_BASE_URL` is also injected into `src/renderer/index.html`'s CSP
`connect-src` at build time (Vite's `%VITE_*%` HTML replacement) — the
renderer can only `fetch()` that one origin, nothing else.
