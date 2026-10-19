import { describe, expect, it } from "vitest";
import { DEFAULT_SETTINGS, parseSettings } from "./settings";

describe("parseSettings", () => {
  it("defaults to the standard (full) DTI setting, asking before fast, opening on Home", () => {
    expect(parseSettings(null)).toEqual(DEFAULT_SETTINGS);
    expect(DEFAULT_SETTINGS.mri.eddyProfile).toBe("full");
    expect(DEFAULT_SETTINGS.mri.confirmFast).toBe(true);
    expect(DEFAULT_SETTINGS.general.startPage).toBe("home");
  });

  it("keeps valid stored values", () => {
    const s = parseSettings({
      general: { startPage: "assessment" },
      mri: { eddyProfile: "fast", confirmFast: false },
    });
    expect(s.general.startPage).toBe("assessment");
    expect(s.mri).toEqual({ eddyProfile: "fast", confirmFast: false });
  });

  it("falls back per field when a value is invalid, never to fast", () => {
    const s = parseSettings({
      general: { startPage: "settings" },
      mri: { eddyProfile: "turbo", confirmFast: "yes" },
    });
    expect(s).toEqual(DEFAULT_SETTINGS);
    expect(parseSettings("garbage")).toEqual(DEFAULT_SETTINGS);
    expect(parseSettings([1, 2])).toEqual(DEFAULT_SETTINGS);
  });

  it("does not share the default object, so one change cannot alter the defaults", () => {
    const s = parseSettings(null);
    s.mri.eddyProfile = "fast";
    expect(DEFAULT_SETTINGS.mri.eddyProfile).toBe("full");
  });
});
