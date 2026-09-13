---
name: grill-requirements
description: >-
  Before building, speccing, designing, or writing code for any PD-XAI
  feature (mobile capture flow, desktop clinician view, backend endpoint, or
  any mri/gait/voice/tapping/fusion model module), interrogate the docs in
  the research-docs folder (PD-XAI-Vault) to pin down what's actually
  documented — instead of assuming or inventing a requirement from the
  feature name alone. Use this whenever the user asks to
  build/implement/design/spec a feature, asks "what are the requirements for
  X" or "what does X need to do", references the proposal, research
  question, or scope, or when it's unclear whether a proposed feature is
  actually in scope. Trigger this proactively before writing code for a new
  feature even if the user doesn't explicitly ask for a requirements check —
  getting the requirement wrong is more expensive to fix after the code
  exists.
---

## Why "grill"

The research-docs folder (an Obsidian vault, `PD-XAI-Vault/`) is the actual
source of truth for what this system needs to do — not the feature name
someone typed in chat, and not what sounds plausible for a Parkinson's
screening app. CLAUDE.md in the J26-SE-356 repo already enforces "don't
invent numbers" for statistics; this skill applies the same discipline to
requirements. A feature built from an assumed requirement is exactly as
costly to unwind as a result built from an invented statistic — the code
gets written, reviewed, maybe even shipped, before anyone notices it solves
the wrong problem.

"Grilling" means treating the docs adversarially: don't skim until you find
something that sounds supportive, actively look for the parts that would
contradict or narrow the request, and say so when you find them.

## Where to look

The vault is organized by folder — go to the folder that matches the
question rather than grepping the whole vault blind:

| Folder | Contains |
|---|---|
| `00-MOC/` | Maps of content — index pages, good starting point to find the right note |
| `03-Datasets/` | What data actually exists, subject counts, modality availability |
| `06-My-Notes/` | The team's own synthesis — closest thing to a running spec |
| `07-Drafts/` | In-progress writing — may be ahead of or behind what's decided |
| `08-Problems/` | Open problems / gaps — what's explicitly *not* solved yet |
| `09-Architecture/` | System design decisions already made |
| `10-Team/` | Ownership, who's responsible for what |
| `scripts/` | Presentation scripts — often state scope decisions in plain language that never made it into a formal spec |

Also check `PD-XAI-Proposal-Presentation.pptx` at the vault's root and the
J26-SE-356 repo's `CLAUDE.md` (§1 for the project framing and screening-vs-
diagnosis rule, §3 for what a feature is and isn't allowed to depend on).

## Steps

1. **Identify the feature and its module** — which of mri / gait / voice /
   tapping / fusion / mobile / desktop / backend it belongs to. If that's
   unclear, that's already a finding worth surfacing before going further.

2. **Search, don't assume.** Use Glob/Grep across the vault folders above for
   the feature's name, its module, and near-synonyms (a feature called
   "risk score" might be documented as "screening output" or "triage
   result"). Read the matching notes in full, not just the snippet that
   matched.

3. **Extract the requirement in the doc's own words.** Quote or closely
   paraphrase what the docs actually say the feature must do, what inputs it
   takes, and what it must NOT do. Precision matters more than fluency here —
   don't smooth a hedge ("we might include X") into a commitment ("X is
   required").

4. **Actively look for what narrows or contradicts the request.** Specifically
   check: does this conflict with the screening-not-diagnosis framing (never
   "diagnose", "discharge" — only "prioritised referral" / "not prioritised")?
   Does it use a feature CLAUDE.md §3.1 forbids as model input (UPDRS,
   Hoehn & Yahr, medication status, DaTscan/QSM)? Does `08-Problems/` already
   flag this exact thing as unsolved or deliberately deferred?

5. **When the docs are silent or contradict each other, say so — don't fill
   the gap yourself.** "Not specified in the vault" is a valid and useful
   answer. Guessing what the team probably meant and presenting it as if it
   were documented is the exact failure mode this skill exists to prevent.

6. **Report a short requirement brief before building anything:**
   - Feature name and module
   - What it must do, with the doc(s) it came from (file name, and a short
     quote or close paraphrase — enough that the user could go verify it)
   - What's explicitly out of scope or already flagged as a problem
   - Anything unconfirmed — put this list first if it's non-empty, since it's
     usually the most decision-relevant part for the user
   - Any conflict found with CLAUDE.md's non-negotiables

7. **Only after the user confirms (or the brief has nothing unresolved)**,
   proceed to actually build, spec, or design the feature — handing off to
   `requirement-to-design` for UI work, or straight to code otherwise.

## Example

> User: "build the results screen for the desktop app"

Don't start designing a dashboard from general screening-app intuition.
Instead: check `06-My-Notes/` and `09-Architecture/` for how results are
meant to be presented, check `scripts/` for how the team has described the
results screen to reviewers already (the presentation scripts often describe
UI flows in plain language), and check CLAUDE.md for the "prioritised
referral, never diagnosis" wording constraint. Then report back: "The vault
says the desktop results view shows [X, Y] and must use referral-priority
language, not risk percentages framed as diagnosis. It doesn't say whether
per-modality contributions should be shown alongside the fused score —
that's not decided yet. Want me to design both and let you pick, or check
with the team first?"
