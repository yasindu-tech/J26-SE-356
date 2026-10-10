import { contextBridge } from "electron";

// contextIsolation is on and nodeIntegration is off (see apps/desktop/README.md)
// — this file is the ONLY bridge between the renderer and Node/Electron APIs.
// Nothing is exposed yet; add typed APIs here (backed by packages/shared
// types) as the renderer needs to talk to the backend or local disk.
contextBridge.exposeInMainWorld("electronAPI", {});
