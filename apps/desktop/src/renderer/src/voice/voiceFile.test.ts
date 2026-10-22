import { describe, expect, it } from "vitest";
import { VOICE_MAX_UPLOAD_MB, voiceFileProblem, voiceFileSummary } from "./voiceFile";

const MB = 1024 * 1024;

describe("voiceFileProblem", () => {
  it("accepts every format the backend accepts, in any case", () => {
    for (const name of ["a.wav", "a.FLAC", "a.ogg", "a.mp3", "a.M4A", "a.aac", "a.webm"]) {
      expect(voiceFileProblem(name, MB)).toBeNull();
    }
  });

  it("refuses other files before anything is sent", () => {
    expect(voiceFileProblem("walk.mp4", MB)?.title).toMatch(/Not an audio file/);
    expect(voiceFileProblem("scan.zip", MB)?.title).toMatch(/Not an audio file/);
    expect(voiceFileProblem("no-extension", MB)?.title).toMatch(/Not an audio file/);
  });

  it("refuses empty and too large files", () => {
    expect(voiceFileProblem("a.wav", 0)?.title).toMatch(/empty/);
    expect(voiceFileProblem("a.wav", VOICE_MAX_UPLOAD_MB * MB)).toBeNull();
    expect(voiceFileProblem("a.wav", VOICE_MAX_UPLOAD_MB * MB + 1)?.title).toMatch(/too large/);
  });
});

describe("voiceFileSummary", () => {
  it("names the format and size", () => {
    expect(voiceFileSummary("aaah.m4a", 0.4 * MB)).toBe("M4A · 0.4 MB");
    expect(voiceFileSummary("aaah.wav", 12 * MB)).toBe("WAV · 12 MB");
  });
});
