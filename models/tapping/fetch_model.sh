#!/usr/bin/env bash
# Downloads the MediaPipe Hand Landmarker bundle used by the tapping module.
# ~7.5 MB; not committed (models/**/artifacts/ is gitignored).
# Run once after cloning:  ./models/tapping/fetch_model.sh
set -euo pipefail
DEST="$(cd "$(dirname "$0")" && pwd)/artifacts"
URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
mkdir -p "$DEST"
curl -fL -o "$DEST/hand_landmarker.task" "$URL"
echo "sha256:"
shasum -a 256 "$DEST/hand_landmarker.task"
