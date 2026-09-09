# models/ — the research code

Four independent modality modules, one fusion layer, and the shared contracts
and evaluation harness they all run through.

```
common/     Feature contract, evaluation harness, shared preprocessing
mri/        T1 + DTI            owner: Yasindu     (IT23267718)
gait/       MediaPipe pose      owner: Fernando    (IT23259102)
voice/      Acoustic features   owner: Ricky       (IT23294134)
tapping/    Tap kinematics      owner: Weerasinghe (IT23247918)
fusion/     Late fusion + Level-2 cross-modal XAI
```

## Every module has the same shape

```
<module>/
  src/          Pipeline code
  configs/      YAML experiment configs — hyperparameters live here, not in code
  notebooks/    Exploration only. Outputs cleared before commit. Never the source of a reported number.
  tests/        Including rule-enforcement tests (see below)
  README.md     Dataset, verified counts, known problems
  requirements.txt
```

## The rules every module must satisfy

Not style — correctness. Full detail in [CLAUDE.md §3](../CLAUDE.md).

1. **Feature contract.** No UPDRS, Hoehn & Yahr, clinician impression,
   medication status, DaTscan or QSM as a *model input*. Call the validator in
   `common/contracts/` before `fit()`.
2. **Fit inside the fold.** Harmonisation, feature selection, scalers, imputers,
   resampling — all learned inside the training fold only.
3. **Subject-wise splitting**, asserted in code.
4. **Balanced accuracy + AUC** with bootstrap CIs. Never raw accuracy as the
   headline.
5. **Baseline ladder**: age+sex → best single feature → best single modality →
   fusion. Report all four.
6. **Two controls that fail loudly**: site-prediction gate, and a
   label-permutation control that must score at chance.

> **A rule-enforcing function without a test that proves it *fails* on a
> violation is not a validator.** Every guard needs a red test.

## Reported numbers must be reproducible

Any subject count or metric that appears in a slide, report or README must come
from a script in `scripts/` or a test — not from a notebook cell, and not from
memory. Blank fields are **not** negatives: an early PPMI audit read blank
`MRIWDTI` as "no" and found 2 paired subjects; the true figure was 410.
