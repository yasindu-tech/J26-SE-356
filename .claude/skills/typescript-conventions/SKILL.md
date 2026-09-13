---
name: typescript-conventions
description: >-
  Apply strict, Matt Pocock-style TypeScript patterns whenever writing or
  reviewing TypeScript in the J26-SE-356 repo — apps/mobile (React Native),
  apps/desktop (Electron), or packages/shared. Use this whenever the user
  asks to write a component, hook, type, API contract, or any .ts/.tsx file
  in this repo, or asks for a TypeScript code review. Covers avoiding `any`,
  discriminated unions, the `satisfies` operator, and other patterns from
  Total TypeScript. This operationalizes the "strict mode on, no `any`
  without a comment" rule already in CLAUDE.md §5 — reach for it any time
  that rule is relevant, not just when the user explicitly names TypeScript
  style.
---

## Why this exists

CLAUDE.md already says: strict mode on, eslint + prettier, no `any` without a
comment explaining why. That's a rule, not a technique. This skill is the
"how" — concrete patterns (associated with Matt Pocock / Total TypeScript's
teaching, which is where most of this repo's team will have encountered
them) for actually writing code that satisfies that rule, rather than
sprinkling `as any` to make the compiler stop complaining.

The underlying idea: TypeScript's type system is there to make illegal states
unrepresentable. If a bug is possible to write, someone on a 4-person team
under deadline pressure will eventually write it. Good types make entire
categories of bug not compile, instead of relying on someone remembering a
rule.

## Core patterns

**Never `any` — reach for `unknown` and narrow it.** `any` disables type
checking on everything it touches, including downstream code that looks
type-safe but isn't. `unknown` forces a narrowing check (`typeof`,
`instanceof`, a type predicate) before use. If `any` genuinely seems
necessary, that's usually a sign the boundary needs a schema/parse step
(e.g. validating an API response) rather than a type escape hatch.

**Discriminated unions over optional-field soup.** For anything with
mutually exclusive states — a capture session that's `idle | recording |
processing | done | error`, an API result that's `success | failure` — model
it as a union with a literal discriminant field, not one object with five
optional fields where only some combinations are valid:

```ts
// ❌ any combination of these fields can be set, most are nonsensical
type CaptureState = {
  status: string;
  recordingStartedAt?: number;
  errorMessage?: string;
  resultUrl?: string;
};

// ✅ only valid states are representable
type CaptureState =
  | { status: "idle" }
  | { status: "recording"; startedAt: number }
  | { status: "error"; message: string }
  | { status: "done"; resultUrl: string };
```

Then switch on the discriminant and let `never` catch missed cases:

```ts
function describe(state: CaptureState): string {
  switch (state.status) {
    case "idle": return "Ready";
    case "recording": return "Recording…";
    case "error": return state.message;
    case "done": return "Done";
    default: {
      const _exhaustive: never = state;
      return _exhaustive;
    }
  }
}
```

Add a new state to the union and this fails to compile until every switch
handles it — that's the point.

**Prefer inference; use `satisfies` when you need to check a shape without
widening it.** Don't annotate a type when TypeScript can infer something
more precise on its own. Use `satisfies` when you want compile-time
validation against a shape but still want the narrower inferred type
preserved (useful for config objects, theme tokens — see the
`ui-consistency` skill — and route/endpoint maps):

```ts
const theme = {
  primary: "#1D4ED8",
  danger: "#DC2626",
} satisfies Record<string, string>;
// theme.primary is typed as the literal string, not widened to `string`
```

**Avoid TypeScript `enum`.** Prefer a union of string literals (optionally
with an `as const` object if you need reverse lookup or iteration). Enums
have runtime-behavior quirks (numeric enums are bidirectional, `const enum`
doesn't always work well with tooling) that string literal unions don't.

```ts
// ✅
type Modality = "mri" | "gait" | "voice" | "tapping";
const MODALITIES = ["mri", "gait", "voice", "tapping"] as const;
```

**No non-null assertions (`!`) without a comment justifying why it's safe.**
`foo!.bar` tells the compiler to trust you and disables the exact check that
would catch a null-pointer-style bug. If you're sure it's non-null, either
narrow it properly (an `if` check, a default) or leave a comment explaining
the invariant that makes the assertion safe.

**Branded types for IDs that shouldn't mix.** A `SubjectId` and a
`SessionId` are both strings, but passing one where the other belongs is a
real bug class in this codebase given the subject-wise splitting requirement
in CLAUDE.md §3.3. A brand catches it at compile time:

```ts
type SubjectId = string & { readonly __brand: "SubjectId" };
type SessionId = string & { readonly __brand: "SessionId" };
```

**Type predicates for custom narrowing**, not casts:

```ts
// ✅
function isErrorState(s: CaptureState): s is Extract<CaptureState, { status: "error" }> {
  return s.status === "error";
}
```

**Keep generics constrained and named for what they are**, not single
letters once there's more than one: `<TInput, TOutput>` reads better than
`<T, U>` once a function has two type parameters that mean different things.

## Where this applies in this repo

- `apps/mobile/src/**` — React Native. Component props should be explicit
  interfaces, not inferred from usage. Event handlers from RN APIs often
  come back loosely typed — narrow at the boundary, not throughout the
  component.
- `apps/desktop/src/**` — Electron. The `main`/`preload`/`renderer` boundary
  is a serialization boundary — anything crossing it via IPC should have a
  shared discriminated-union message type in `packages/shared`, not loose
  `any` payloads, since a typo in a string channel name or payload shape is
  otherwise a silent runtime failure.
- `packages/shared/**` — this is the contract every app depends on
  (CODEOWNERS gates it for a reason). Types here should be the most
  carefully reviewed in the repo, since a loose type here propagates
  looseness everywhere it's imported.

## When reviewing existing code

Look for: `any` without a justifying comment, non-null assertions without a
comment, boolean/optional-field combinations that should be a discriminated
union, `enum` usage, and places where a cast (`as SomeType`) papers over a
shape mismatch instead of the code actually producing that shape. Flag these
even if the user didn't ask for a style review — if you're touching a file
for an unrelated reason and it's using `any` where a real type is easy to
write, say so.

## Sources

Patterns here reflect publicly taught TypeScript practice associated with
Matt Pocock / Total TypeScript (totaltypescript.com) — discriminated unions,
`satisfies`, avoiding `any` and `enum`. These are general TypeScript best
practices, not proprietary to this repo; apply judgment about which ones are
worth the ceremony for a given piece of code rather than applying all of
them everywhere reflexively.
