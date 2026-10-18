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
  neutral600: "#5C6663", // secondary/helper text (darkened from #6C7572 to meet 4.5:1 on white and on #F7F5F2)
  neutral300: "#C9C3BB", // dashed "not captured / abstained" borders, the zero line of a chart
  neutral200: "#E2DDD6", // borders, dividers
  neutral100: "#EDE9E4", // progress/axis tracks, row dividers
  neutral50: "#F7F5F2", // page background (warm off-white)
  surface: "#FFFFFF", // card/panel background

  // Design reference "PD-XAI clinician desktop" (the four approved artboards: home,
  // assessment setup, analysis in progress, result). See the colour rules in each comment.
  primaryHover: "#1B6E7F", // mid teal: progress fills, the active workflow step, link hover
  primaryTint: "#E6EEF0", // active navigation item background
  onPrimarySoft: "#D7E4E7", // secondary text on the teal bar
  onPrimaryFaint: "#B9D0D6", // inactive workflow steps on the teal bar
  onPrimaryBorder: "#5F95A3", // outlines of chips and buttons on the teal bar

  // Terracotta: ONLY the "High priority" heading (large text, 4.2:1 on white) and the triage
  // scale. Never small text, and never a fill under white text.
  priorityHigh: "#C75B39",
  priorityHighTint: "#F5E3DC",

  // Amber: a border or tint only, always with dark text (7.4:1). Never amber text.
  caution: "#E8A33D",
  cautionTint: "#FBF3E4",
  cautionTintStrong: "#F9EBD3",

  // Sage: the confidence-interval band only. Never text.
  sage: "#8B9D9B",
  sageTint: "#DCE3E1", // the chip behind an interval's numbers

  // Status that is not a screening result (connected, analysed, done).
  success: "#4A7C59",
  successTint: "#E3EDE6",

  // Screening priority, not diagnosis — see CLAUDE.md §1. Named for what the
  // result means ("needs attention" / "not prioritised"), not pass/fail.
  attention: "#B45309", // amber — prioritised for review
  attentionMuted: "#FEF3C7",
  calm: "#047857", // muted green — not prioritised (never a "clear" checkmark green)
  calmMuted: "#D1FAE5",

  danger: "#DC2626", // reserved for real errors (network failure, invalid input) — never a screening result
} satisfies Record<string, string>;

// IBM Plex is used when installed and falls back to the system fonts otherwise, so no font
// file has to ship with the app. Plex Mono is for numbers, IDs and codes.
export const fonts = {
  sans: '"IBM Plex Sans", system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
  mono: '"IBM Plex Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace',
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
