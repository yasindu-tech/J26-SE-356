"""Feature contract for the voice module (task VOICE-11).

Blocks specialist-only inputs and identifier/label columns from ever reaching a
model (CLAUDE.md section 3.1). Call ``validate_features`` on the column list
before every ``fit()`` and before every prediction.

These names may still be used as labels, targets or stratification variables;
the contract only governs what goes into the feature matrix.

This lives in models/voice for PP1. Promoting it to models/common is planned
after PP1, with the lead's review.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# Case-insensitive patterns for specialist-only inputs. Checked against all 752
# UCI-470 feature names: none match, so no real acoustic feature is blocked.
BANNED_PATTERNS: dict[str, str] = {
    r"updrs": "UPDRS / MDS-UPDRS score",
    r"hoehn|yahr": "Hoehn and Yahr stage",
    r"clinician|impression|rating": "clinician impression or rating",
    r"medic": "medication status",
    r"levodopa|ledd|dose": "levodopa / medication dose",
    r"on[_\-/ ]?off|off[_\-/ ]?state|on[_\-/ ]?state": "ON/OFF medication state",
    r"datscan|dat[_\-]?spect|sbr": "DaTscan / DaT-SPECT value",
    r"qsm": "QSM-derived value",
    r"diagnos|referral": "diagnosis or referral information",
}

# Exact names (case-insensitive) that are identifiers or the label, not features.
BANNED_EXACT: dict[str, str] = {
    "id": "person identifier",
    "subject_id": "person identifier",
    "class": "the label",
    "label": "the label",
    "target": "the label",
}


class FeatureContractError(ValueError):
    """A banned column was offered as a model input."""


def banned_reason(column: str) -> str | None:
    """Why ``column`` is not allowed as a model input, or None if it is allowed."""
    name = column.strip().lower()
    if name in BANNED_EXACT:
        return BANNED_EXACT[name]
    for pattern, reason in BANNED_PATTERNS.items():
        if re.search(pattern, name):
            return reason
    return None


def validate_features(columns: Iterable[str]) -> list[str]:
    """Return the columns unchanged if all are allowed; raise otherwise.

    Raises FeatureContractError listing every banned column and why, so one run
    shows all the problems rather than the first.
    """
    columns = [str(c) for c in columns]
    if not columns:
        raise FeatureContractError("no feature columns given")
    problems = [f"{c!r} ({reason})" for c in columns if (reason := banned_reason(c))]
    if problems:
        raise FeatureContractError("banned model inputs (CLAUDE.md 3.1): " + ", ".join(problems))
    return columns
