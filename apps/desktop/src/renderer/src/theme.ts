import { colors, radius, spacing } from "@pd-xai/shared";

/**
 * Publishes the shared design tokens as CSS custom properties so plain CSS
 * files can use `var(--color-primary)` etc. instead of a second copy of the
 * palette. Call once, before the first render.
 */
export function applyTheme(): void {
  const root = document.documentElement.style;

  for (const [name, value] of Object.entries(colors)) {
    root.setProperty(`--color-${kebabCase(name)}`, value);
  }
  for (const [name, value] of Object.entries(spacing)) {
    root.setProperty(`--spacing-${kebabCase(name)}`, `${value}px`);
  }
  for (const [name, value] of Object.entries(radius)) {
    root.setProperty(`--radius-${kebabCase(name)}`, `${value}px`);
  }
}

function kebabCase(name: string): string {
  return name
    .replace(/([a-z])([A-Z])/g, "$1-$2")
    .replace(/([a-zA-Z])([0-9])/g, "$1-$2")
    .toLowerCase();
}
