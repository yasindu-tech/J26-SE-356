#!/usr/bin/env bash
# Fetches the MediaPipe Hand Landmarker bundle used by the tapping module (FT-11).
#
# PREREQUISITE: tracking needs models/tapping/artifacts/hand_landmarker.task. The file is
# about 7.5 MB and is NOT in git (models/**/artifacts/ is gitignored), so run this once
# after cloning:
#
#     ./models/tapping/fetch_model.sh        # works from any directory
#
# What it does:
#   - the URL below pins the model version (float16, version 1);
#   - the expected SHA-256 is pinned in fetch_model.sha256 next to this script (one line:
#     "<hash>  hand_landmarker.task"); the download is COMPARED against it, not just printed;
#   - if artifacts/hand_landmarker.task exists with that hash, nothing is downloaded;
#   - if it exists with another hash, that is reported and the file is replaced, but only
#     after the new download has been verified;
#   - the download goes to a temporary file in artifacts/, is verified, then renamed into
#     place, so an interrupted or failed run never leaves a partial or unverified model
#     (the temporary file is removed on failure, Ctrl-C and kill);
#   - exit code 0 means the model at the final path has the pinned hash; anything else is
#     non-zero.
#
# Pin provenance: fetch_model.sha256 is the SHA-256 of the file this project has used so far
# (it starts fbc2a30080c3c557), and a fresh download from the pinned URL gave the same hash.
# We did not find a vendor-published checksum for this URL, so the pin is not vendor-verified.
# It guarantees that everyone gets the same bytes we used, not that Google endorses them.
#
# Updating the pin on purpose: change MODEL_URL, download the file once by hand, check it,
# put its `shasum -a 256` into fetch_model.sha256, and review both changes together. Any
# parquet made with the old model is then recomputed by the batch runner (it compares the
# model hash).
#
# Test hook: FETCH_MODEL_URL overrides MODEL_URL (for example a file:// URL). The pinned
# hash must still match; tests copy this script next to their own fetch_model.sha256.
#
# Needs bash 3.2 or newer (macOS default), curl, and shasum or sha256sum.
set -euo pipefail

MODEL_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_URL="${FETCH_MODEL_URL:-$MODEL_URL}"
MODEL_NAME="hand_landmarker.task"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="$HERE/artifacts"
DEST="$DEST_DIR/$MODEL_NAME"
PIN_FILE="$HERE/fetch_model.sha256"

die() {
  echo "fetch_model: $*" >&2
  exit 1
}

sha256_of() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | cut -d' ' -f1
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    die "neither shasum nor sha256sum is available"
  fi
}

[ -f "$PIN_FILE" ] || die "pin file not found: $PIN_FILE"
EXPECTED="$(awk 'NF { print $1; exit }' "$PIN_FILE")"
case "$EXPECTED" in
  *[!0-9a-f]* | "") die "pin file does not start with a SHA-256 hex digest" ;;
esac
[ "${#EXPECTED}" -eq 64 ] || die "pin file does not start with a 64-character SHA-256 digest"

if [ -f "$DEST" ]; then
  ACTUAL="$(sha256_of "$DEST")"
  if [ "$ACTUAL" = "$EXPECTED" ]; then
    echo "fetch_model: $MODEL_NAME already present and verified (sha256 ${ACTUAL:0:16}...); nothing to download"
    exit 0
  fi
  echo "fetch_model: existing $MODEL_NAME does not match the pin (expected ${EXPECTED:0:16}..., found ${ACTUAL:0:16}...); replacing it" >&2
fi

mkdir -p "$DEST_DIR"
TMP="$(mktemp "$DEST_DIR/.$MODEL_NAME.XXXXXX")"
cleanup() {
  rm -f "$TMP"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "fetch_model: downloading $MODEL_NAME"
curl -fsSL --retry 2 -o "$TMP" "$MODEL_URL" || die "download failed"

ACTUAL="$(sha256_of "$TMP")"
if [ "$ACTUAL" != "$EXPECTED" ]; then
  die "checksum mismatch: expected $EXPECTED, got $ACTUAL; nothing was installed"
fi

chmod 644 "$TMP"
mv -f "$TMP" "$DEST"
echo "fetch_model: verified sha256 $ACTUAL; installed $MODEL_NAME"
