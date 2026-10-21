// Electron side of the finger-tapping demo (task FT-61): the child process, the view of the
// main window that shows Streamlit, and the IPC. Logic lives in tapping.ts. The page is
// shown in a WebContentsView of the main process: an iframe or <webview> in the renderer is
// blocked by its CSP and sandbox. The view may only open the demo's own local address.

import { execFile, spawn } from "child_process";
import { existsSync } from "fs";
import { app, BrowserWindow, ipcMain, shell, WebContentsView } from "electron";
import {
  IPC_TAPPING_OPEN_EXTERNAL,
  IPC_TAPPING_SET_BOUNDS,
  IPC_TAPPING_START,
  IPC_TAPPING_STATUS,
  IPC_TAPPING_STATUS_EVENT,
  IPC_TAPPING_STOP,
  type TappingBounds,
  type TappingStatus,
} from "../shared/ipc";
import {
  HOST,
  TappingController,
  checkPrerequisites,
  clampBounds,
  isAllowedUrl,
  pickFreePort,
  resolvePython,
  resolveRepoRoot,
  safeDownloadName,
  type ChildLike,
} from "./tapping";

const IMPORT_CHECK_TIMEOUT_MS = 30_000;
const PARTITION = "tapping-demo"; // no "persist:" prefix: nothing is kept after the app closes

export function registerTapping(): void {
  const repoRoot = resolveRepoRoot(app.getAppPath());
  const python = resolvePython({
    envPython: import.meta.env.MAIN_VITE_TAPPING_PYTHON,
    repoRoot,
    platform: process.platform,
  });

  let owner: BrowserWindow | null = null;
  let view: WebContentsView | null = null;
  let bounds: TappingBounds | null = null;
  let activePort = 0;

  function push(status: TappingStatus): void {
    if (owner && !owner.isDestroyed()) owner.webContents.send(IPC_TAPPING_STATUS_EVENT, status);
  }

  function showView(url: string): void {
    if (!owner || owner.isDestroyed()) return;
    activePort = Number(new URL(url).port);
    const guard = (details: { url: string; preventDefault: () => void }): void => {
      if (!isAllowedUrl(details.url, activePort)) details.preventDefault();
    };
    const next = new WebContentsView({
      webPreferences: { partition: PARTITION, sandbox: true, contextIsolation: true, nodeIntegration: false },
    });
    const wc = next.webContents;
    wc.setWindowOpenHandler(() => ({ action: "deny" }));
    wc.on("will-navigate", guard);
    wc.on("will-redirect", guard);
    wc.on("will-frame-navigate", guard);
    wc.session.setPermissionRequestHandler((_wc, _permission, callback) => callback(false));
    wc.session.on("will-download", (_event, item) => {
      const name = safeDownloadName(item.getFilename(), item.getMimeType());
      if (name === null) {
        item.cancel();
        return;
      }
      item.setSaveDialogOptions({ defaultPath: name, filters: [{ name: "PDF", extensions: ["pdf"] }] });
    });
    next.setBounds(bounds ?? { x: 0, y: 0, width: 0, height: 0 });
    owner.contentView.addChildView(next);
    view = next;
    void wc.loadURL(url);
  }

  function hideView(): void {
    const current = view;
    view = null;
    if (!current) return;
    if (owner && !owner.isDestroyed()) owner.contentView.removeChildView(current);
    if (!current.webContents.isDestroyed()) current.webContents.close();
  }

  const controller = new TappingController({
    repoRoot,
    python,
    spawn: (command, args, options) =>
      spawn(command, args, { cwd: options.cwd, stdio: ["ignore", "pipe", "pipe"], windowsHide: true }) as ChildLike,
    pickPort: () => pickFreePort(),
    check: () =>
      checkPrerequisites(
        {
          exists: (path) => existsSync(path),
          canImportStreamlit: (py) =>
            new Promise((done) =>
              execFile(py, ["-c", "import streamlit"], { timeout: IMPORT_CHECK_TIMEOUT_MS }, (error) =>
                done(error === null),
              ),
            ),
        },
        { python, repoRoot },
      ),
    health: {
      fetchOk: async (url) => {
        try {
          return (await fetch(url, { signal: AbortSignal.timeout(2000) })).status === 200;
        } catch {
          return false;
        }
      },
      now: () => Date.now(),
      sleep: (ms) => new Promise((done) => setTimeout(done, ms)),
    },
    showView,
    hideView,
    onStatus: push,
  });

  const fromWindow = (sender: Electron.WebContents): BrowserWindow | null => BrowserWindow.fromWebContents(sender);

  ipcMain.handle(IPC_TAPPING_START, (event) => {
    const win = fromWindow(event.sender);
    if (!win) return { state: "error", reason: "The demo could not be started." } satisfies TappingStatus;
    if (owner !== win) {
      owner = win;
      win.once("closed", () => {
        void controller.stop();
        if (owner === win) owner = null;
      });
    }
    return controller.start();
  });
  ipcMain.handle(IPC_TAPPING_STOP, async (event) => {
    if (fromWindow(event.sender) === owner) await controller.stop();
  });
  ipcMain.handle(IPC_TAPPING_STATUS, () => controller.getStatus());
  ipcMain.on(IPC_TAPPING_SET_BOUNDS, (event, raw: unknown) => {
    if (fromWindow(event.sender) !== owner) return;
    const next = clampBounds(raw);
    if (next === null) return;
    bounds = next;
    view?.setBounds(next);
  });
  ipcMain.handle(IPC_TAPPING_OPEN_EXTERNAL, async (event) => {
    if (fromWindow(event.sender) !== owner) return false;
    const url = controller.currentUrl();
    if (url === null || !isAllowedUrl(url, activePort) || new URL(url).hostname !== HOST) return false;
    await shell.openExternal(url);
    return true;
  });

  // No orphan process: wait for a clean stop, then quit; kill at once if that is not possible.
  let quitting = false;
  app.on("before-quit", (event) => {
    if (quitting || !controller.hasChild()) return;
    event.preventDefault();
    quitting = true;
    void controller.stop().finally(() => app.quit());
  });
  app.on("will-quit", () => controller.killNow());
}
