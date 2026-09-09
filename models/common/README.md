# models/common — shared contracts and evaluation

Everything here is depended on by all four modality modules. Changes need the
lead's review (CODEOWNERS) — do not restructure without asking.

```
contracts/     Feature-contract validator — the runtime block on specialist-only inputs
evaluation/    Subject-wise splitters, baseline ladder, bootstrap CIs,
               faithfulness tests, site-prediction gate, label-permutation control
preprocessing/ Shared transforms that must be fold-aware by construction
```

## Design principle
These utilities exist so the rules in [CLAUDE.md §3](../../CLAUDE.md) are
**enforced by construction rather than remembered**. A splitter that cannot
produce a subject-straddling split is better than a code review that catches it.

Every guard here needs a test that proves it **fails** on a violating input.
