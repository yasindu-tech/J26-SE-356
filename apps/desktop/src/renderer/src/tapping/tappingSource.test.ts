import { createHash } from "crypto";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { TappingAnalysisAPI } from "../../../shared/tappingIpc";
import { saveTappingPdf, videoRef } from "./tappingSource";

// The renderer's call to the main process (the PDF save dialog), with a stand-in for
// window.electronAPI. The analysis itself goes to the backend: see tappingRun.test.ts.

afterEach(() => vi.unstubAllGlobals());

const file = (bytes: number[], name = "patient_jane.MP4"): File => new File([new Uint8Array(bytes)], name);

describe("videoRef", () => {
  it("is the first 8 hex characters of the SHA-256 of the bytes, like the program's clip ref", async () => {
    const bytes = [1, 2, 3, 4, 5];
    const expected = createHash("sha256").update(Buffer.from(bytes)).digest("hex").slice(0, 8);
    expect(await videoRef(file(bytes))).toBe(expected);
    expect(await videoRef(file(bytes, "other-name.mov"))).toBe(expected);
  });
});

describe("saveTappingPdf", () => {
  it("passes the ref and the PDF to the main process and maps a failure", async () => {
    const savePdf = vi.fn(async () => ({ status: "saved" }) as const);
    vi.stubGlobal("window", { electronAPI: { tappingAnalysis: { savePdf } as Partial<TappingAnalysisAPI> } });
    expect(await saveTappingPdf("f0e37b01", "JVBERi0=")).toEqual({ status: "saved" });
    expect(savePdf).toHaveBeenCalledWith("f0e37b01", "JVBERi0=");
    vi.stubGlobal("window", {
      electronAPI: { tappingAnalysis: { savePdf: async () => Promise.reject(new Error("no")) } },
    });
    expect(await saveTappingPdf("f0e37b01", "JVBERi0=")).toEqual({
      status: "error",
      message: "The summary could not be saved.",
    });
  });
});
