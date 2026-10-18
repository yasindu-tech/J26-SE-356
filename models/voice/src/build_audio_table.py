"""Extract voice features from every raw-audio recording into one table (VOICE-21).

Runs ``audio_features.extract_voice_features`` on each recording from
``audio_datasets`` and writes one row per recording to
``models/voice/artifacts/audio_features.csv`` (gitignored: it holds dataset ids
and local paths). The UCI-470 Praat measures (``uci_praat``) are added from
the same steady vowel.

Recordings refused by the quality gate are KEPT in the table with
``quality_passed = False`` and their reasons, and blank features. Dropping them
silently would hide abstention bias: the printed summary shows how many were
refused per dataset and class, so a gate that refuses PD voices more often than
healthy ones is visible.

Usage (from the repo root):
    python models/voice/src/build_audio_table.py
    python models/voice/src/build_audio_table.py --datasets figshare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import audio_datasets
import audio_features
import pandas as pd
import uci_praat

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
DEFAULT_OUT = ARTIFACTS / "audio_features.csv"
META_COLUMNS = ("path", "person", "label", "dataset", "age", "sex", "quality_passed", "problems")


def build_table(recordings: list[audio_datasets.Recording]) -> pd.DataFrame:
    """One row per recording: metadata, quality info and (when passed) all features."""
    rows = []
    for rec in recordings:
        result = audio_features.extract_voice_features(rec.path)
        uci = uci_praat.uci_features_from(result) if result.features else {}
        rows.append(
            {
                "path": str(rec.path.relative_to(audio_datasets.REPO_ROOT)),
                "person": rec.person,
                "label": rec.label,
                "dataset": rec.dataset,
                "age": rec.age,
                "sex": rec.sex,
                "quality_passed": result.quality.passed,
                "problems": " | ".join(result.quality.problems),
                **{f"q_{k}": v for k, v in result.quality.info.items()},
                **(result.features or {}),
                **uci,
            }
        )
    table = pd.DataFrame(rows)
    expected = (*audio_features.FEATURE_NAMES, *uci_praat.UCI_PRAAT_FEATURES)
    for name in expected:  # refused rows stay blank, never zero
        if name not in table.columns:
            table[name] = float("nan")
    return table


def abstention_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Recordings and refusals per dataset and class."""
    summary = table.groupby(["dataset", "label"]).agg(
        recordings=("person", "size"), refused=("quality_passed", lambda s: int((~s).sum()))
    )
    summary["refused_share"] = (summary.refused / summary.recordings).round(3)
    return summary.rename(index={1: "PD", 0: "healthy"}, level="label")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="*", choices=sorted(audio_datasets.LOADERS))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    recordings = audio_datasets.load_all(args.datasets)
    print(f"extracting features from {len(recordings)} recordings ...", flush=True)
    table = build_table(recordings)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    print("\nquality gate, per dataset and class:")
    print(abstention_summary(table).to_string())
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
