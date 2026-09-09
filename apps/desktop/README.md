# apps/desktop — Clinician workstation (Electron)

Where the non-specialist clinician creates an assessment, pairs a phone, uploads
an MRI if one exists, and reads the result.

## Stack
Electron · TypeScript · React (renderer) · Vite

## Structure
```
src/
  main/       Electron main process — windows, IPC, file access
  preload/    contextBridge — the ONLY surface the renderer can call
  renderer/   React UI (pages/, components/)
  api/        Backend client (types from packages/shared)
```

## Rules
- **`contextIsolation: true`, `nodeIntegration: false`.** The renderer never
  touches Node directly; everything goes through an explicit preload bridge.
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
npm install
npm run dev --workspace apps/desktop
```
