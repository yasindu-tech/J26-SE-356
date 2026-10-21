import { ApiError } from "../api/client";

/** What the result screen can show for the overlay video of one analysis. */
export type OverlayLoad =
  | { kind: "ready"; blob: Blob }
  | { kind: "none" } // this analysis has no overlay: show nothing
  | { kind: "error"; message: string };

export const OVERLAY_NOT_FOUND = "GAIT_OVERLAY_NOT_FOUND";

export async function loadOverlay(
  id: string,
  fetchOverlay: (id: string) => Promise<Blob>,
): Promise<OverlayLoad> {
  try {
    return { kind: "ready", blob: await fetchOverlay(id) };
  } catch (error) {
    if (error instanceof ApiError && error.code === OVERLAY_NOT_FOUND) return { kind: "none" };
    const message =
      error instanceof Error ? error.message : "The overlay video could not be loaded.";
    return { kind: "error", message };
  }
}
