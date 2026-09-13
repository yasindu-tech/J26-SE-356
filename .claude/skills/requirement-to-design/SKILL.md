---
name: requirement-to-design
description: >-
  Turn a confirmed feature requirement into a UI design/mockup for the
  J26-SE-356 PD-XAI mobile (React Native, patient-facing capture) or desktop
  (Electron, clinician workstation) app. Use whenever the user asks to
  design a screen or flow, wants a mockup or wireframe of a feature, asks
  "what should the X screen look like", or wants to visualize a UI before
  building it. Always pairs with grill-requirements (confirm the requirement
  first) and ui-consistency (use the shared color theme, don't invent one).
---

## Before designing anything

Don't design from the feature name alone. If the requirement hasn't already
been confirmed against the docs in this conversation, run through
`grill-requirements` first — a design built on an assumed requirement has to
be redone once the real one surfaces, and by then it's harder to change
because someone's already reacted to the mockup as if it were settled.

## Pick the surface deliberately

| | Mobile (`apps/mobile`) | Desktop (`apps/desktop`) |
|---|---|---|
| User | Patient, non-specialist, one-time or occasional use | Clinician, repeat daily use |
| Density | Low — one task per screen, large touch targets | Higher — data-dense views are appropriate |
| Literacy assumptions | Minimal — short copy, visual cues, no jargon | Can use clinical terminology |
| Tone | Reassuring, simple, guided step-by-step | Efficient, scannable, trustworthy |

Don't reuse a desktop-density layout on mobile or vice versa — matching
tokens (see below) is what makes them feel like one product, not matching
layouts.

## Non-negotiable copy rules (from CLAUDE.md §1)

This is screening and triage, **never diagnosis**. Whatever the screen shows:

- High risk / flagged → "prioritised referral," "flagged for review," or
  similar — never "diagnosis," "positive," or "you have Parkinson's"
- Low risk / not flagged → "not prioritised," "no flags at this time" —
  never "negative," "clear," "discharged," or "healthy"
- Any UI string, tooltip, label, or button copy gets checked against this
  before the mockup is considered done, not just the main result display

## Use the shared theme — don't invent colors

Read `packages/shared/src/theme.ts` (created by the `ui-consistency` skill;
run that skill first if it doesn't exist yet) and build the mockup using
those tokens. In particular, use `attention`/`calm` for screening-priority
states, never `danger`/a stoplight red-green pair — see that skill for why.

## Building the mockup

Default to a self-contained HTML mockup — real layout, real (token-based)
colors, realistic sample copy — rather than a text description or a wireframe
of boxes-and-labels. A clickable, visually real mockup gets much more useful
feedback than a description does.

**Where it lives depends on how it'll be used:**

- **If it's for reviewing/iterating with the user in this conversation, or
  they'll want to reopen it later without digging through the repo** —
  persist it as a Cowork artifact (`mcp__cowork__create_artifact`) if that
  tool is available this session: write the HTML to a file, then call the
  tool. This is the default for a first-draft mockup.
- **If it's meant to live alongside the code as a durable spec** (e.g. going
  into a PR, or the team will reference it while building the real screen),
  or if no artifact tool is available — save it as a file under
  `docs/architecture/mockups/<module>-<screen>.html` in the J26-SE-356 repo
  instead, so it's versioned with the feature it describes.

Ask which the user wants if it's not obvious from context; default to the
persisted artifact for a first pass since it's faster to iterate on, then
offer to save a copy into the repo once the design is settled.

## What to include

- The primary screen/flow in its main state
- Key alternate states: empty (nothing captured yet), loading/in-progress,
  error, and the result state — a mockup that only shows the happy path
  hides most of the actual design problems
- Mobile: also show what happens on a capture failure (bad lighting, dropped
  connection) — this app runs in non-specialist settings, that path isn't an
  edge case
- A short rationale under or alongside the mockup tying each major decision
  back to the confirmed requirement — "the vault says X, so this screen
  does Y" — so the user can evaluate the design against the actual
  requirement, not just aesthetics

## After presenting

Ask specifically whether the copy avoids diagnostic framing and whether the
requirement is actually captured — not just "does this look good." Visual
polish is the easy part to get right; the framing and requirement match are
where mistakes are costly.
