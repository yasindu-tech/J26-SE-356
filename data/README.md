# data/ — local only, never committed

```
raw/         Untouched downloads, exactly as received
interim/     Intermediate processing artefacts
processed/   Analysis-ready feature tables
external/    External validation cohorts (NTUA, BrainLat)
```

## Gait datasets (models/gait)
Both live under `raw/` and are never committed:

```
raw/CARE-PD/      SMPL-derived h36m 17-joint .npz files + per-cohort .pkl labels
raw/KOA-PD-NM/    191 walking videos (.MOV) used as the demo input
```

- **CARE-PD** — 9 cohorts. Use the h36m world-coordinate files (metres, Y up).
  Severity labels (`UPDRS_GAIT`) are in the cohort `.pkl` files, not in the
  h36m files. fps is 30 except PD-GaM at 25. No age, sex or height is provided.
- **KOA-PD-NM** — frame rate, resolution, clip length and severity scale are not
  documented; they are confirmed in the gait Day 1 checks before any feature is
  trusted.
- Check each dataset's own licence and access terms before sharing any file.

## Rules
- **Nothing here is ever committed.** `.gitignore` blocks it. A `guard-no-data`
  CI job that blocks it again on the server side is temporarily out of CI (see
  `.github/workflows/ci.yml`) — it'll be re-added, but until then `.gitignore`
  is the only automated line of defence, so double-check `git status` before
  pushing.
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
