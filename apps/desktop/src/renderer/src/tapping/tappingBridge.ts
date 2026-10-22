import type { TappingAPI, TappingAnalysisAPI } from "../../../shared/tappingIpc";

// The tapping members of window.electronAPI, typed here because the shared ElectronAPI type
// is owned by the team. The preload script must expose them (as it does on this branch).

export interface TappingBridge {
  tapping: TappingAPI;
  tappingAnalysis: TappingAnalysisAPI;
}

export function tappingBridge(): TappingBridge {
  return window.electronAPI as unknown as TappingBridge;
}
