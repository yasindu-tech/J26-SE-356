import { contextBridge, ipcRenderer } from "electron";
import {
  IPC_OPEN_VOICE_RESULT,
  IPC_TAPPING_OPEN_EXTERNAL,
  IPC_TAPPING_SET_BOUNDS,
  IPC_TAPPING_START,
  IPC_TAPPING_STATUS,
  IPC_TAPPING_STATUS_EVENT,
  IPC_TAPPING_STOP,
  type ElectronAPI,
  type OpenResultFileReply,
  type TappingStatus,
} from "../shared/ipc";

// contextIsolation is on and nodeIntegration is off (see apps/desktop/README.md)
// — this file is the ONLY bridge between the renderer and Node/Electron APIs.
// Nothing is exposed yet; add typed APIs here (backed by packages/shared
// types) as the renderer needs to talk to the backend or local disk.
const electronAPI: ElectronAPI = {
  // Voice result screen (PP1): pick and read a voice result JSON file.
  openVoiceResultFile: () =>
    ipcRenderer.invoke(IPC_OPEN_VOICE_RESULT) as Promise<OpenResultFileReply>,
  // Finger-tapping demo: main runs Streamlit and shows it in a view of this window.
  tapping: {
    start: () => ipcRenderer.invoke(IPC_TAPPING_START) as Promise<TappingStatus>,
    stop: () => ipcRenderer.invoke(IPC_TAPPING_STOP) as Promise<void>,
    getStatus: () => ipcRenderer.invoke(IPC_TAPPING_STATUS) as Promise<TappingStatus>,
    setBounds: (bounds) => ipcRenderer.send(IPC_TAPPING_SET_BOUNDS, bounds),
    openExternal: () => ipcRenderer.invoke(IPC_TAPPING_OPEN_EXTERNAL) as Promise<boolean>,
    onStatus: (listener) => {
      const handler = (_event: Electron.IpcRendererEvent, status: TappingStatus): void => listener(status);
      ipcRenderer.on(IPC_TAPPING_STATUS_EVENT, handler);
      return () => ipcRenderer.removeListener(IPC_TAPPING_STATUS_EVENT, handler);
    },
  },
};

contextBridge.exposeInMainWorld("electronAPI", electronAPI);
