---
name: ui-consistency
description: >-
  Keep UI work in the J26-SE-356 repo (apps/mobile React Native, apps/desktop
  Electron) visually consistent with one shared color theme and design
  tokens, instead of each screen inventing its own colors/spacing. Use this
  whenever writing or reviewing any component, screen, or style file in
  apps/mobile or apps/desktop, whenever the user asks about colors, theming,
  visual consistency, or design system, and always alongside
  requirement-to-design when producing a new mockup. No design-tokens file
  exists in the repo yet — this skill creates one (in packages/shared) the
  first time it's needed, then every later UI task checks against it rather
  than picking new colors ad hoc.
---

## Why this exists

Two apps (mobile capture, desktop clinician workstation), one team, no design
system yet. Without a shared source of truth, each screen someone builds
invents its own blue, its own spacing, its own idea of what "danger" looks
like — and by the third screen nothing matches. This skill makes
`packages/shared` the single place colors and type tokens are defined, since
that package is already the contract both apps depend on (see CLAUDE.md §2).

## First time this is needed: create the tokens file

Check for `packages/shared/src/theme.ts`. If it doesn't exist yet, create it
with this starter palette — a reasonable, accessible default, not a
prescription. Tell the user you created it and that they should treat the
actual hex values as adjustable; what matters more than the specific colors
is that everyone imports from here instead of hand-picking new ones per
screen.

```ts
// packages/shared/src/theme.ts
export const colors = {
  primary: "#1D4ED8",     // primary action, navigation, brand
  primaryMuted: "#DBEAFE",
  neutral900: "#111827",  // primary text
  neutral600: "#4B5563",  // secondary text
  neutral200: "#E5E7EB",  // borders, dividers
  neutral50: "#F9FAFB",   // page background
  surface: "#FFFFFF",     // card/panel background
  attention: "#B45309",   // "needs review" / prioritised — amber, not red
  attentionMuted: "#FEF3C7",
  calm: "#047857",        // "not prioritised" — green, but muted, not a "pass" checkmark green
  calmMuted: "#D1FAE5",
  danger: "#DC2626",      // reserved for actual errors (network failure, invalid input) — NOT for screening results
} satisfies Record<string, string>;

export const spacing = {
  xs: 4, sm: 8, md: 16, lg: 24, xl: 32,
} satisfies Record<string, number>;

export const typography = {
  heading: { fontSize: 20, fontWeight: "600" },
  body: { fontSize: 16, fontWeight: "400" },
  caption: { fontSize: 13, fontWeight: "400" },
} satisfies Record<string, { fontSize: number; fontWeight: string }>;
```

(See the `typescript-conventions` skill for why `satisfies` is used here
rather than a type annotation — it validates the shape while keeping the
literal types.)

**Deliberate choice on `attention`/`calm` naming, not `danger`/`success`:**
this app reports screening priority, not a diagnosis or a pass/fail test
result. CLAUDE.md §1 is explicit that high risk means "prioritised referral"
and low risk means "not prioritised" — never "positive/negative" or
"discharged." Naming and coloring the tokens to match that language (amber
"needs attention" instead of red "positive/danger", muted green "not
prioritised" instead of a checkmark-green "you're clear") keeps the color
system from silently reintroducing the diagnostic framing that the copy
rules were written to avoid. Reserve actual `danger` red for real error
states — a failed upload, a network error — not for screening output.

## Every other time: check against the existing tokens

1. Read `packages/shared/src/theme.ts` (or wherever the user has since moved
   it — check `packages/shared/src/` if the exact filename has changed).
2. Any new component or screen should import `colors`/`spacing`/`typography`
   from there, not define new hex values or magic-number spacing inline.
3. If a screen needs a color that isn't in the palette yet, that's a signal
   to add it to the shared theme file (with a comment on what it's for), not
   to inline a one-off value — otherwise the token file silently stops being
   the source of truth within a week.
4. When reviewing existing UI code, flag inline hex colors (`#` literals
   outside `theme.ts`) and inconsistent spacing values as things to migrate,
   even if that wasn't the specific ask — visual drift compounds fast across
   two apps built by different people.
5. Mobile and desktop have different density expectations (mobile: patient-
   facing, larger touch targets, less information density; desktop:
   clinician-facing, denser data views are fine) — consistency means shared
   *colors and type scale*, not identical layouts. Don't force desktop-style
   density onto the mobile capture screens or vice versa.

## Relationship to requirement-to-design

When `requirement-to-design` produces a new mockup, it should pull from
these tokens rather than choosing its own palette — treat this skill's
tokens as a hard input to that one, not a parallel, independent choice.
