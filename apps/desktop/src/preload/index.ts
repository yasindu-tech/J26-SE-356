import { contextBridge, ipcRenderer } from "electron";
import { IPC_OPEN_VOICE_RESULT, type ElectronAPI, type OpenResultFileReply } from "../shared/ipc";
import {
  IPC_TAPPING_ANALYSE,
  IPC_TAPPING_CANCEL,
  IPC_TAPPING_OPEN_EXTERNAL,
  IPC_TAPPING_PROGRESS,
  IPC_TAPPING_SAVE_PDF,
  IPC_TAPPING_SET_BOUNDS,
  IPC_TAPPING_START,
  IPC_TAPPING_STATUS,
  IPC_TAPPING_STATUS_EVENT,
  IPC_TAPPING_STOP,
  type TappingAnalysisReply,
  type TappingProgress,
  type TappingSavePdfReply,
  type TappingStatus,
} from "../shared/tappingIpc";

// contextIsolation is on and nodeIntegration is off (see apps/desktop/README.md)
// — this file is the ONLY bridge between the renderer and Node/Electron APIs.
// Nothing is exposed yet; add typed APIs here (backed by packages/shared
// types) as the renderer needs to talk to the backend or local disk.
// The tapping members are typed in shared/tappingIpc.ts (the team owns ElectronAPI); the
// renderer reads them through renderer/src/tapping/tappingBridge.ts.
const electronAPI: ElectronAPI = {
  // Voice result screen (PP1): pick and read a voice result JSON file.
  openVoiceResultFile: () =>
    ipcRenderer.invoke(IPC_OPEN_VOICE_RESULT) as Promise<OpenResultFileReply>,
};

const tappingApi = {
  // Finger-tapping demo: main runs Streamlit and shows it in a view of this window.
  tapping: {
    start: () => ipcRenderer.invoke(IPC_TAPPING_START) as Promise<TappingStatus>,
    stop: () => ipcRenderer.invoke(IPC_TAPPING_STOP) as Promise<void>,
    getStatus: () => ipcRenderer.invoke(IPC_TAPPING_STATUS) as Promise<TappingStatus>,
    setBounds: (bounds: unknown) => ipcRenderer.send(IPC_TAPPING_SET_BOUNDS, bounds),
    openExternal: () => ipcRenderer.invoke(IPC_TAPPING_OPEN_EXTERNAL) as Promise<boolean>,
    onStatus: (listener: (status: TappingStatus) => void) => {
      const handler = (_event: Electron.IpcRendererEvent, status: TappingStatus): void => listener(status);
      ipcRenderer.on(IPC_TAPPING_STATUS_EVENT, handler);
      return () => ipcRenderer.removeListener(IPC_TAPPING_STATUS_EVENT, handler);
    },
  },
  // Finger-tapping analysis in the assessment: main runs the Python program on a temp copy.
  tappingAnalysis: {
    analyse: (request: unknown) =>
      ipcRenderer.invoke(IPC_TAPPING_ANALYSE, request) as Promise<TappingAnalysisReply>,
    cancel: () => ipcRenderer.invoke(IPC_TAPPING_CANCEL) as Promise<void>,
    onProgress: (listener: (progress: TappingProgress) => void) => {
      const handler = (_event: Electron.IpcRendererEvent, progress: TappingProgress): void => listener(progress);
      ipcRenderer.on(IPC_TAPPING_PROGRESS, handler);
      return () => ipcRenderer.removeListener(IPC_TAPPING_PROGRESS, handler);
    },
    savePdf: (ref: string, pdfBase64: string) =>
      ipcRenderer.invoke(IPC_TAPPING_SAVE_PDF, ref, pdfBase64) as Promise<TappingSavePdfReply>,
  },
};

contextBridge.exposeInMainWorld("electronAPI", { ...electronAPI, ...tappingApi });

