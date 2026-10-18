"""Cohort counts for the raw-audio voice datasets (task VOICE-21).

Prints, per dataset on disk: recordings, people, PD and healthy counts, age by
class, sex by class, and the audio formats and sample rates. Every raw-audio
count quoted in models/voice/README.md or RESULTS.md must come from this script
(CLAUDE.md section 3.6).

Labels come from ``audio_datasets`` loaders, which raise on a blank or
conflicting label instead of counting it as healthy. No person ids are printed.

Usage (from the repo root):
    python scripts/count_voice_audio.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "voice" / "src"))
import audio_datasets  # noqa: E402

CLASS_NAMES = {1: "PD", 0: "healthy"}


def audit(recordings: list[audio_datasets.Recording]) -> dict[str, object]:
    frame = pd.DataFrame(
        {
            "person": [r.person for r in recordings],
            "label": [CLASS_NAMES[r.label] for r in recordings],
            "age": [r.age for r in recordings],
            "sex": [r.sex for r in recordings],
        }
    )
    people = frame.groupby("person").first()
    formats: Counter[str] = Counter()
    for rec in recordings:
        info = sf.info(rec.path)
        formats[f"{rec.path.suffix.lstrip('.')} {info.samplerate} Hz {info.channels} ch"] += 1
    age = people.groupby("label").age.describe()[["count", "mean", "min", "50%", "max"]]
    return {
        "recordings": len(recordings),
        "people": len(people),
        "per_class": people.label.value_counts().to_dict(),
        "age_by_class": age.round(1),
        "age_missing": int(people.age.isna().sum()),
        "sex_by_class": pd.crosstab(people.label, people.sex.fillna("missing")),
        "formats": dict(formats),
    }


def main() -> int:
    for name, loader in audio_datasets.LOADERS.items():
        try:
            recordings = loader()
        except FileNotFoundError as error:
            print(f"== {name}: not on disk ({error})\n")
            continue
        result = audit(recordings)
        print(f"== {name}")
        print(f"recordings: {result['recordings']}, people: {result['people']}")
        print(f"people per class: {result['per_class']}")
        print(f"age missing for {result['age_missing']} people; age by class:")
        print(result["age_by_class"].to_string())
        print("sex by class:")
        print(result["sex_by_class"].to_string())
        print(f"formats: {result['formats']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
