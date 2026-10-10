"""GAIT-10 / Check D: block specialist-only inputs before a model is fitted.

CLAUDE.md section 3.1: UPDRS / MDS-UPDRS scores, Hoehn and Yahr stage, clinician ratings,
medication status or dose, ON/OFF state, and DaTscan / QSM values must never be model input
features. They may be labels, targets or stratification variables only. Our label is UPDRS
gait, which is allowed as a label and banned as an input.

Call ``check_feature_contract`` with the names of the columns the model will train on, before
every ``fit()``:

    check_feature_contract(list(X.columns), label_name="UPDRS_GAIT")

Names that look like a label column (``score``, ``severity``, ``label``, ``target``, ``y``,
``UPDRS_GAIT``) are always rejected as inputs, because the CARE-PD loader carries the label in
``Walk.score`` and ``Walk.severity``. Pass more label names with ``label_name`` (one name or a
list).

It raises ``FeatureContractError`` listing every offending name and the rule it breaks. This is
a name check, a tripwire: it stops careless mistakes (the label or a clinical field left in the
feature table) but cannot see a banned value hidden under an innocent column name.

Matching is by whole words after splitting on punctuation and camelCase, so ``updrs-gait``,
``UPDRSGait`` and ``H&Y_stage`` are caught while ``stride_time_cv`` or ``median_filter`` are not.
It errs on the safe side: a harmless name that looks banned fails, and the fix is to rename it.

This is a gait-local stand-in. ``models/common/contracts`` is empty; when the shared validator
exists, swap the import and keep the same call.

Command line (exit code 0 = allowed, 1 = banned name found, 2 = bad arguments):

    python models/gait/src/feature_contract.py stride_time_cv cadence UPDRS_GAIT --label UPDRS_GAIT
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass

# Rule names, quoted from CLAUDE.md section 3.1.
R_UPDRS = "UPDRS / MDS-UPDRS score"
R_HY = "Hoehn and Yahr stage"
R_CLINICIAN = "clinician impression or rating"
R_MEDICATION = "medication status, levodopa dose or ON/OFF state"
R_DAT_QSM = "DaTscan / DaT-SPECT or QSM derived value"
R_LABEL = "the label column is used as an input"

# Distinctive words: banned anywhere inside a name once separators are removed.
_SUBSTRINGS: tuple[tuple[str, str], ...] = (
    ("updrs", R_UPDRS),
    ("hoehn", R_HY),
    ("yahr", R_HY),
    ("levodopa", R_MEDICATION),
    ("ldopa", R_MEDICATION),
    ("datscan", R_DAT_QSM),
    ("qsm", R_DAT_QSM),
)

# Whole words that are banned on their own.
_WORDS: dict[str, str] = {
    "hy": R_HY,
    "medication": R_MEDICATION,
    "med": R_MEDICATION,
    "medicated": R_MEDICATION,
    "medicine": R_MEDICATION,
    "medicines": R_MEDICATION,
    "drug": R_MEDICATION,
    "drugs": R_MEDICATION,
    "led": R_MEDICATION,
    "medications": R_MEDICATION,
    "meds": R_MEDICATION,
    "ledd": R_MEDICATION,
    "dose": R_MEDICATION,
    "dosage": R_MEDICATION,
    "onoff": R_MEDICATION,
    "clinician": R_CLINICIAN,
    "rater": R_CLINICIAN,
    "rating": R_CLINICIAN,
    "ratings": R_CLINICIAN,
    "impression": R_CLINICIAN,
    "spect": R_DAT_QSM,
    "sbr": R_DAT_QSM,
}
# Plain "on" and "off" are not banned on their own: gait terms such as toe_off_time use them.

# Whole names that are label columns in this module (compared after removing separators).
LABEL_LIKE_NAMES = ("score", "severity", "severity_class", "label", "target", "y", "UPDRS_GAIT")

# Word sequences that are banned only when the words are next to each other.
_SEQUENCES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("on", "off"), R_MEDICATION),
    (("on", "state"), R_MEDICATION),
    (("off", "state"), R_MEDICATION),
    (("hy", "stage"), R_HY),
    (("h", "y"), R_HY),
    (("h", "and", "y"), R_HY),
    (("dat", "scan"), R_DAT_QSM),
    (("dat", "spect"), R_DAT_QSM),
)


@dataclass(frozen=True)
class Violation:
    name: str  # the input name as it was passed in
    rule: str  # which CLAUDE.md 3.1 rule it breaks


class FeatureContractError(ValueError):
    """Raised when the model inputs break the feature contract."""

    def __init__(self, violations: list[Violation]) -> None:
        self.violations = violations
        lines = [f"  {v.name!r}: {v.rule}" for v in violations]
        super().__init__(
            "Feature contract violated (CLAUDE.md section 3.1). These may be labels, targets "
            "or stratification variables only, never model inputs:\n" + "\n".join(lines)
        )


def split_words(name: str) -> list[str]:
    """Lower-case words of a name: split on punctuation, digits' edges and camelCase."""
    text = name.replace("&", " and ")
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    return [w for w in re.split(r"[^A-Za-z0-9]+", text.lower()) if w]


def _compact(name: str) -> str:
    return "".join(split_words(name))


def _rule_for(name: str) -> str | None:
    words = split_words(name)
    compact = "".join(words)
    for sub, rule in _SUBSTRINGS:
        if sub in compact:
            return rule
    for word in words:
        if word in _WORDS:
            return _WORDS[word]
    for seq, rule in _SEQUENCES:
        n = len(seq)
        if any(tuple(words[i : i + n]) == seq for i in range(len(words) - n + 1)):
            return rule
    return None


def _label_keys(label_name: str | Iterable[str] | None) -> set[str]:
    extra = (
        []
        if label_name is None
        else [label_name]
        if isinstance(label_name, str)
        else list(label_name)
    )
    return {_compact(n) for n in (*LABEL_LIKE_NAMES, *extra) if n}


def find_violations(
    input_names: Iterable[str], label_name: str | Iterable[str] | None = None
) -> list[Violation]:
    """Every input name that breaks the contract, in the order given, each listed once."""
    names = list(input_names)
    bad_types = [n for n in names if not isinstance(n, str)]
    if bad_types:
        raise TypeError(f"input names must be strings, got {bad_types[:3]!r}")
    label_keys = _label_keys(label_name)
    found: list[Violation] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        rule = _rule_for(name)
        if rule is None and _compact(name) in label_keys:
            rule = R_LABEL
        if rule is not None:
            found.append(Violation(name, rule))
    return found


def check_feature_contract(
    input_names: Iterable[str], label_name: str | Iterable[str] | None = None
) -> None:
    """Raise ``FeatureContractError`` if any input is banned; call this before every ``fit()``.

    ``input_names`` are the columns the model will train on. ``label_name`` is the target
    column (or several); passing it, or any name in ``LABEL_LIKE_NAMES``, as an input is also
    a violation. An empty input list raises
    ``ValueError`` because there is nothing to fit.
    """
    names = list(input_names)
    if not names:
        raise ValueError("no input features were given")
    violations = find_violations(names, label_name)
    if violations:
        raise FeatureContractError(violations)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Feature contract check (GAIT-10, Check D)")
    parser.add_argument("names", nargs="+", help="model input names to check")
    parser.add_argument("--label", action="append", default=None, help="label column name")
    args = parser.parse_args(argv)
    try:
        check_feature_contract(args.names, args.label)
    except FeatureContractError as err:
        print(err)
        return 1
    print(f"Feature contract OK: {len(args.names)} inputs, none banned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
