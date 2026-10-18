"""Open any common audio file as one mono waveform at a fixed sample rate (VOICE-20).

Uploaded files come in many formats (WAV, FLAC, OGG, MP3, M4A, ...), sample rates
and channel counts. ``load_audio`` turns every one of them into the same shape,
so training data and uploads go through identical processing:

- mono (channels averaged);
- resampled to ``TARGET_SR`` = 8 kHz, the telephone band. One of the datasets
  (Figshare) is telephone audio at 8 kHz, and everything else must be made
  comparable to it, or the model could learn the sample rate instead of PD.
  Pitch, jitter, shimmer, HNR and formants F1-F3 all sit below 4 kHz.

WAV, FLAC, OGG and MP3 are read by libsndfile (``soundfile``). Anything else
(M4A/AAC, ...) is decoded by ``ffmpeg``, which must be installed
(``brew install ffmpeg``). The original format, sample rate and channel count
are kept in ``AudioClip`` for the quality report.

Nothing here learns from data, so it is safe to run before any split.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 8000


class AudioLoadError(ValueError):
    """The file could not be opened or decoded as audio."""


@dataclass(frozen=True)
class AudioClip:
    samples: np.ndarray  # mono float32 at TARGET_SR, original amplitude (not normalised)
    sample_rate: int  # always TARGET_SR
    source_format: str  # file extension, lower case, e.g. "wav", "mp3"
    source_sample_rate: int
    source_channels: int
    clipped_fraction: float  # share of original samples at or above full scale

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate


def _read_with_soundfile(path: Path) -> tuple[np.ndarray, int, int]:
    samples, sr = sf.read(path, dtype="float32", always_2d=True)
    return samples, int(sr), samples.shape[1]


def _audio_stream(path: Path) -> tuple[int, int]:
    """Sample rate and channel count of the first audio stream, from ffprobe."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise AudioLoadError(
            f"cannot open .{path.suffix.lstrip('.')} files without ffmpeg (brew install ffmpeg)"
        )
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        stream = json.loads(result.stdout)["streams"][0]
        return int(stream["sample_rate"]), int(stream["channels"])
    except (json.JSONDecodeError, KeyError, IndexError, ValueError) as error:
        raise AudioLoadError("the file has no readable audio stream") from error


def _read_with_ffmpeg(path: Path) -> tuple[np.ndarray, int, int]:
    sr, channels = _audio_stream(path)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AudioLoadError("ffprobe was found but ffmpeg was not")
    # Decode to raw 32-bit float at the file's own rate and channel count.
    decoded = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "f32le", "-"],
        capture_output=True,
        check=False,
    )
    if decoded.returncode != 0 or not decoded.stdout:
        raise AudioLoadError("the file could not be decoded as audio")
    samples = np.frombuffer(decoded.stdout, dtype="<f4")
    return samples[: len(samples) // channels * channels].reshape(-1, channels), sr, channels


def load_audio(path: str | Path, target_sr: int = TARGET_SR) -> AudioClip:
    """Open ``path`` as mono audio at ``target_sr``. Raises AudioLoadError if it is not audio."""
    path = Path(path)
    if not path.is_file():
        raise AudioLoadError(f"no such file: {path.name}")
    try:
        samples, sr, channels = _read_with_soundfile(path)
    except (sf.LibsndfileError, RuntimeError):
        samples, sr, channels = _read_with_ffmpeg(path)

    if samples.size == 0:
        raise AudioLoadError("the file contains no audio")
    if not np.isfinite(samples).all():
        raise AudioLoadError("the file contains invalid sample values")

    clipped_fraction = float(np.mean(np.abs(samples) >= 0.999))
    mono = samples.mean(axis=1)
    if sr != target_sr:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)
    return AudioClip(
        samples=np.ascontiguousarray(mono, dtype=np.float32),
        sample_rate=target_sr,
        source_format=path.suffix.lower().lstrip(".") or "unknown",
        source_sample_rate=sr,
        source_channels=channels,
        clipped_fraction=clipped_fraction,
    )
