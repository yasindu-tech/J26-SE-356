# packages/shared

TypeScript types and constants shared by `apps/mobile`, `apps/desktop` and the
backend client. **Single source of truth** for anything that crosses a process
boundary.

```
src/types/       Assessment, CaptureResult, RiskResult, AvailabilityMask, ...
src/constants/   Modality names, quality thresholds, API paths
```

## Rules
- Keep it in sync with `services/backend/app/schemas/` — a type that drifts from
  the Pydantic schema is worse than no shared type at all.
- **No vocabulary that implies diagnosis.** `RiskResult`, `TriagePriority`,
  `referralPriority` — never `diagnosis`, `positive`, `negative`, `cleared`.
- Any type carrying a risk score must also carry the modality breakdown and the
  availability mask that produced it. Make the honest shape the only shape.
- Owned by the lead (CODEOWNERS) — four codebases break when this changes.
