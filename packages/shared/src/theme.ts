/**
 * Shared design tokens for apps/mobile and apps/desktop.
 *
 * Source of truth for color/spacing/type — screens should import from here
 * instead of hand-picking new hex values. See the `ui-consistency` skill.
 *
 * Palette sampled from the approved sign-in screen mockup and the PD-XAI
 * logo, not invented — adjust here (not per-screen) if the brand changes.
 */

export const colors = {
  primary: "#0F4C5C", // brand teal — primary actions, panels
  primaryMuted: "#DCE9EC",
  onPrimary: "#FFFFFF", // text/icons on a primary-filled surface

  accent: "#EB9532", // logo accent orange — brand mark only, not a UI action color

  neutral900: "#1C2321", // primary text
  neutral600: "#6C7572", // secondary/helper text
  neutral200: "#E2DDD6", // borders, dividers
  neutral50: "#F7F5F2", // page background (warm off-white)
  surface: "#FFFFFF", // card/panel background

  // Screening priority, not diagnosis — see CLAUDE.md §1. Named for what the
  // result means ("needs attention" / "not prioritised"), not pass/fail.
  attention: "#B45309", // amber — prioritised for review
  attentionMuted: "#FEF3C7",
  calm: "#047857", // muted green — not prioritised (never a "clear" checkmark green)
  calmMuted: "#D1FAE5",

  danger: "#DC2626", // reserved for real errors (network failure, invalid input) — never a screening result
} satisfies Record<string, string>;

export const spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
} satisfies Record<string, number>;

export const typography = {
  title: { fontSize: 34, fontWeight: "700" }, // large marketing-style headline (e.g. sign-in left panel)
  heading: { fontSize: 20, fontWeight: "600" },
  body: { fontSize: 16, fontWeight: "400" },
  caption: { fontSize: 13, fontWeight: "400" },
} satisfies Record<string, { fontSize: number; fontWeight: string }>;

export const radius = {
  sm: 6,
  md: 10,
  lg: 16,
} satisfies Record<string, number>;
