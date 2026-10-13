import type { ElectronAPI } from "../../shared/ipc";

// Exposed by src/preload/index.ts through contextBridge.
declare global {
  interface Window {
    electronAPI: ElectronAPI;
  }
}

export {};
