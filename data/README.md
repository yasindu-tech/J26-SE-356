# data/ — local only, never committed

```
raw/         Untouched downloads, exactly as received
interim/     Intermediate processing artefacts
processed/   Analysis-ready feature tables
external/    External validation cohorts (NTUA, BrainLat)
```

## Rules
- **Nothing here is ever committed.** `.gitignore` blocks it; the
  `guard-no-data` CI job blocks it again. Both are deliberate.
- PPMI, NTUA, BrainLat and UCI-470 are all under **data use agreements**. Each
  team member requests access under their own DUA. Do not redistribute
  downloads, including to each other, outside the agreement's terms.
- No PHI leaves this folder — not into logs, notebooks, issues, PRs or commit
  messages.
- **A blank field is not a negative.** Check null semantics before trusting any
  count. An early PPMI audit read blank `MRIWDTI` as "no" and reported 2 paired
  subjects; the true number was 410.

## Reproducibility
Every count quoted in a slide or report must be regenerable by a script in
`scripts/`, run against the raw files. No hand-counted numbers.
