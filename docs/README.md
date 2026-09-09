# docs/

```
architecture/   System design, data flow, API contracts
research/       Literature notes, verified figures, dataset audits
adr/            Architecture Decision Records — why we chose/rejected things
```

## ADRs
Record any decision that a future reader (or examiner) would otherwise ask
"why did you do it that way?" about. Especially **rejections** — they are the
ones that look like oversights if undocumented.

Existing decisions worth recording: rejecting 3D transformers at n≈210,
rejecting FreeSurfer `recon-all`, rejecting end-to-end deep audio at n=252,
choosing late over early fusion, masking rather than imputing missing modalities.

Format: `adr/NNN-short-title.md` — Context, Decision, Consequences.
