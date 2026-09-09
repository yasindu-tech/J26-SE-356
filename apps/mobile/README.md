# apps/mobile — Patient capture app (React Native)

Guides a patient through the three phone-based captures in ~10 minutes, with
**quality checked on the device** so a bad capture is caught in the room rather
than a week later.

## Stack
React Native · TypeScript · react-native-vision-camera (video) ·
react-native-audio-recorder-player (voice) · MediaPipe Tasks (on-device pose)

## Structure
```
src/
  screens/     Consent → pairing → capture → quality review → done
  capture/     gait.ts, tapping.ts, voice.ts — one module per modality
  components/  Shared UI
  api/         Backend client (types from packages/shared)
  lib/         Quality gates, device checks, permissions
```

## Rules
- **Pairing is QR-based** from the desktop app. The phone never holds a patient
  identifier beyond the session — pair, capture, upload, clear.
- Quality gates run **before** upload: lighting, framing, full-body visibility
  for gait; clipping and SNR for voice; hand visibility for tapping.
- Nothing is written to the camera roll. Nothing persists after upload.
- Never display a risk score or any interpretation in this app — it is a capture
  device. Results belong on the clinician's desktop app.

## Setup
```bash
npm install
npm run start --workspace apps/mobile
npm run android --workspace apps/mobile   # or: npm run ios
```
