"""Tests for the feature subsets of the UCI-470 transfer test. No real data."""

from __future__ import annotations

import uci_transfer
from uci_praat import UCI_PRAAT_FEATURES


def _units(mismatched: set[str]) -> dict[str, dict[str, bool]]:
    return {name: {"mismatch": name in mismatched} for name in UCI_PRAAT_FEATURES}


def test_subsets_drop_the_measures_the_unit_check_flags() -> None:
    jitter = {f for f in UCI_PRAAT_FEATURES if "Jitter" in f}
    subsets = uci_transfer.feature_subsets(_units(jitter))
    assert subsets["all_16_recomputable"] == list(UCI_PRAAT_FEATURES)
    assert not set(subsets["same_scale_only"]) & jitter
    assert len(subsets["same_scale_only"]) == len(UCI_PRAAT_FEATURES) - len(jitter)


def test_shimmer_and_hnr_subset_never_includes_a_flagged_measure() -> None:
    flagged = {"locShimmer", "meanHarmToNoiseHarmonicity"}
    subset = uci_transfer.feature_subsets(_units(flagged))["shimmer_and_hnr_only"]
    assert subset and not set(subset) & flagged
    assert all("Shimmer" in f or "Harm" in f for f in subset)
