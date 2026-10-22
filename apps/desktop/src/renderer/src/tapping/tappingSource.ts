import { tappingBridge } from "./tappingBridge";
import type { TappingSavePdfReply } from "./tappingTypes";

/** The first 8 hex characters of the video's SHA-256: the same hashed ref the program prints. */
export async function videoRef(file: File): Promise<string | null> {
  try {
    const digest = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
    return Array.from(new Uint8Array(digest).slice(0, 4), (b) => b.toString(16).padStart(2, "0")).join("");
  } catch {
    return null;
  }
}

/** Save dialog, then the PDF bytes (the one thing that needs the main process). */
export function saveTappingPdf(ref: string, pdfBase64: string): Promise<TappingSavePdfReply> {
  return tappingBridge()
    .tappingAnalysis.savePdf(ref, pdfBase64)
    .catch(() => ({ status: "error", message: "The summary could not be saved." }) as const);
}
