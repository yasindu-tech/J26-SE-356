import { describe, expect, it } from "vitest";
import { ApiError } from "../api/client";
import { loadOverlay } from "./overlayLoad";

describe("loadOverlay", () => {
  it("returns the video when the service has one", async () => {
    const blob = new Blob(["video"]);
    expect(await loadOverlay("a1", async () => blob)).toEqual({ kind: "ready", blob });
  });

  it("asks for the analysis it was given", async () => {
    let asked = "";
    await loadOverlay("a1", async (id) => {
      asked = id;
      return new Blob([]);
    });
    expect(asked).toBe("a1");
  });

  it("shows nothing when the analysis has no overlay", async () => {
    const missing = async () => {
      throw new ApiError("GAIT_OVERLAY_NOT_FOUND", "This analysis has no overlay video");
    };
    expect(await loadOverlay("a1", missing)).toEqual({ kind: "none" });
  });

  it("reports any other failure with its message", async () => {
    const down = async () => {
      throw new ApiError("NETWORK_ERROR", "The analysis service is not reachable.");
    };
    expect(await loadOverlay("a1", down)).toEqual({
      kind: "error",
      message: "The analysis service is not reachable.",
    });
    const odd = async () => {
      throw "boom";
    };
    expect(await loadOverlay("a1", odd)).toEqual({
      kind: "error",
      message: "The overlay video could not be loaded.",
    });
  });
});
