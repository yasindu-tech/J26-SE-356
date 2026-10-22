// Electron side of the finger-tapping analysis (task FT-63): the IPC handlers, the temp folder,
// the child process and the save dialog. The logic and its checks are in tappingAnalysis.ts.

import { execFile, spawn } from "child_process";
import { existsSync } from "fs";
import { mkdtemp, rm, writeFile } from "fs/promises";
import { tmpdir } from "os";
import { join } from "path";
import { app, BrowserWindow, dialog, ipcMain } from "electron";
import {
  IPC_TAPPING_ANALYSE,
  IPC_TAPPING_CANCEL,
  IPC_TAPPING_PROGRESS,
  IPC_TAPPING_SAVE_PDF,
  type TappingAnalysisReply,
  type TappingSavePdfReply,
} from "../shared/tappingIpc";
import { resolvePython, resolveRepoRoot } from "./tapping";
import { TappingAnalyser, decodePdf, pdfFileName, type AnalysisChild } from "./tappingAnalysis";

const IMPORT_CHECK_TIMEOUT_MS = 60_000;

export function registerTappingAnalysis(): void {
  const repoRoot = resolveRepoRoot(app.getAppPath());
  const python = resolvePython({
    envPython: import.meta.env.MAIN_VITE_TAPPING_PYTHON,
    repoRoot,
    platform: process.platform,
  });

  const analyser = new TappingAnalyser({
    repoRoot,
    python,
    spawn: (command, args, options) =>
      spawn(command, args, {
        cwd: options.cwd,
        stdio: ["ignore", "pipe", "pipe"],
        windowsHide: true,
      }) as AnalysisChild,
    makeTempDir: () => mkdtemp(join(tmpdir(), "pdxai-tapping-")),
    writeFile: (path, bytes) => writeFile(path, bytes, { mode: 0o600 }),
    removeDir: (path) => rm(path, { recursive: true, force: true }),
    exists: (path) => existsSync(path),
    canImport: (py, modules) =>
      new Promise((done) =>
        execFile(
          py,
          ["-c", modules.map((m) => `import ${m}`).join(";")],
          { timeout: IMPORT_CHECK_TIMEOUT_MS },
          (failure) => done(failure === null),
        ),
      ),
  });

  ipcMain.handle(IPC_TAPPING_ANALYSE, (event, request: unknown): Promise<TappingAnalysisReply> => {
    const sender = event.sender;
    return analyser.run(request, (progress) => {
      if (!sender.isDestroyed()) sender.send(IPC_TAPPING_PROGRESS, progress);
    });
  });
  ipcMain.handle(IPC_TAPPING_CANCEL, () => analyser.cancel());

  ipcMain.handle(
    IPC_TAPPING_SAVE_PDF,
    async (event, ref: unknown, base64: unknown): Promise<TappingSavePdfReply> => {
      const name = pdfFileName(ref);
      const bytes = decodePdf(base64);
      if (name === null || bytes === null) return { status: "error", message: "The summary could not be saved." };
      const options = { defaultPath: name, filters: [{ name: "PDF", extensions: ["pdf"] }] };
      const window = BrowserWindow.fromWebContents(event.sender);
      const choice = window ? await dialog.showSaveDialog(window, options) : await dialog.showSaveDialog(options);
      if (choice.canceled || choice.filePath === "") return { status: "cancelled" };
      try {
        await writeFile(choice.filePath, bytes);
        return { status: "saved" };
      } catch {
        return { status: "error", message: "The summary could not be saved." };
      }
    },
  );

  // No orphan process when the app closes: stop a running analysis first.
  let quitting = false;
  app.on("before-quit", (event) => {
    if (quitting || !analyser.isBusy()) return;
    event.preventDefault();
    quitting = true;
    void analyser.cancel().finally(() => app.quit());
  });
}
