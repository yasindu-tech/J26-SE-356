"""Feature contract for the MRI models (CLAUDE.md section 3.1). Call before every fit().

Two checks, both on column names:

1. **Deny list.** Names that point at specialist-only inputs are rejected with the rule they
   break: UPDRS / MDS-UPDRS, Hoehn and Yahr, clinician ratings, medication / levodopa /
   ON-OFF state, DaTscan / DaT-SPECT / QSM, and the label itself.
2. **Allow list.** Every remaining name must match a known MRI feature pattern. Anything
   else is rejected as "unknown". This is stricter than the deny list on its own: a clinical
   column with an innocent-looking name still cannot get in, because the MRI table has a
   fixed naming scheme. When a new feature family is added (for example DTI), its pattern is
   added here in the same PR, with a test.

``age`` and ``sex`` are accepted only when ``allow_demographics=True``, for the age + sex
rung of the baseline ladder.

This is an MRI-local stand-in: ``models/common/contracts`` is still empty. When the shared
validator exists, keep this allow list and call the shared deny list from here.

    from mri_contract import check_mri_features
    check_mri_features(list(X.columns))           # raises MriContractError if anything is wrong
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

R_UPDRS = "UPDRS / MDS-UPDRS score"
R_HY = "Hoehn and Yahr stage"
R_CLINICIAN = "clinician impression or rating"
R_MEDICATION = "medication status, levodopa dose or ON/OFF state"
R_DAT_QSM = "DaTscan / DaT-SPECT or QSM derived value"
R_LABEL = "the label is used as an input"
R_UNKNOWN = "not a known MRI feature name (add its pattern to mri_contract.py with a test)"

# Banned anywhere in the name once case and separators are removed.
_SUBSTRINGS = (
    ("updrs", R_UPDRS),
    ("hoehn", R_HY),
    ("yahr", R_HY),
    ("levodopa", R_MEDICATION),
    ("ldopa", R_MEDICATION),
    ("ledd", R_MEDICATION),
    ("medication", R_MEDICATION),
    ("onoff", R_MEDICATION),
    ("clinician", R_CLINICIAN),
    ("datscan", R_DAT_QSM),
    ("datspect", R_DAT_QSM),
    ("qsm", R_DAT_QSM),
)
# Banned as whole words.
_WORDS = {
    "hy": R_HY,
    "med": R_MEDICATION,
    "meds": R_MEDICATION,
    "dose": R_MEDICATION,
    "rating": R_CLINICIAN,
    "rater": R_CLINICIAN,
    "impression": R_CLINICIAN,
    "spect": R_DAT_QSM,
    "sbr": R_DAT_QSM,
}
LABEL_NAMES = {"label", "target", "y", "diagnosis", "dx", "group", "cohort", "is_pd", "pd"}

_REGION = r"[LR]_(caudate|putamen|pallidum|thalamus)"
ALLOWED_PATTERNS = (
    re.compile(rf"^{_REGION}_(csf|grey|white)_frac$"),
    re.compile(rf"^{_REGION}_t1_(mean|median|std|skewness|kurtosis|entropy)$"),
    re.compile(r"^(brain_volume_ml|affine_det)$"),
)
DEMOGRAPHICS = {"age", "sex"}


@dataclass(frozen=True)
class Violation:
    name: str
    rule: str


class MriContractError(ValueError):
    """The model inputs break the feature contract."""

    def __init__(self, violations: list[Violation]) -> None:
        self.violations = violations
        lines = "\n".join(f"  {v.name!r}: {v.rule}" for v in violations)
        super().__init__(f"MRI feature contract violated (CLAUDE.md 3.1):\n{lines}")


def _words(name: str) -> list[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return [w for w in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if w]


def _denied(name: str) -> str | None:
    words = _words(name)
    joined = "".join(words)
    if joined in {n.replace("_", "") for n in LABEL_NAMES}:
        return R_LABEL
    for sub, rule in _SUBSTRINGS:
        if sub in joined:
            return rule
    for w in words:
        if w in _WORDS:
            return _WORDS[w]
    pairs = set(zip(words, words[1:], strict=False))
    if pairs & {("dat", "scan"), ("dat", "spect")}:
        return R_DAT_QSM
    if pairs & {("h", "y"), ("on", "off")}:  # "H&Y", "on-off"
        return R_HY if ("h", "y") in pairs else R_MEDICATION
    return None


def violations(names: Iterable[str], allow_demographics: bool = False) -> list[Violation]:
    out = []
    for name in names:
        rule = _denied(name)
        if rule is None:
            known = any(p.match(name) for p in ALLOWED_PATTERNS) or (
                allow_demographics and name in DEMOGRAPHICS
            )
            if not known:
                rule = R_UNKNOWN
        if rule is not None:
            out.append(Violation(name, rule))
    return out


def check_mri_features(names: Iterable[str], allow_demographics: bool = False) -> None:
    """Raise MriContractError listing every bad name. Returns None when all names are allowed."""
    names = list(names)
    if not names:
        raise MriContractError([Violation("<no columns>", "a model needs at least one input")])
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise MriContractError([Violation(d, "column appears twice") for d in dupes])
    found = violations(names, allow_demographics)
    if found:
        raise MriContractError(found)
