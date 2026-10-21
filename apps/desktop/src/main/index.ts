import { app, BrowserWindow, dialog, ipcMain } from "electron";
import { readFile, stat } from "fs/promises";
import { basename, join } from "path";
import { IPC_OPEN_VOICE_RESULT, type OpenResultFileReply } from "../shared/ipc";
import { registerTapping } from "./tappingWindow";

// A voice result file is a few kB; anything far larger is the wrong file.
const MAX_RESULT_FILE_BYTES = 5 * 1024 * 1024;

/** Lets the clinician pick a voice result JSON (e.g. demo_results.json from
 * models/voice/artifacts/) and returns its parsed content. Read-only: the file
 * is never copied, kept or sent anywhere. */
async function openVoiceResultFile(): Promise<OpenResultFileReply> {
  const options = {
    title: "Open voice result file",
    properties: ["openFile" as const],
    filters: [{ name: "Voice result (JSON)", extensions: ["json"] }],
  };
  const window = BrowserWindow.getFocusedWindow();
  const choice = window
    ? await dialog.showOpenDialog(window, options)
    : await dialog.showOpenDialog(options);
  const filePath = choice.filePaths[0];
  if (choice.canceled || filePath === undefined) return { status: "cancelled" };

  try {
    if ((await stat(filePath)).size > MAX_RESULT_FILE_BYTES) {
      return { status: "error", message: "This file is too large to be a voice result file." };
    }
    const content: unknown = JSON.parse(await readFile(filePath, "utf8"));
    return { status: "loaded", fileName: basename(filePath), content };
  } catch (error) {
    const message =
      error instanceof SyntaxError ? "This file is not valid JSON." : "The file could not be read.";
    return { status: "error", message };
  }
}

function createMainWindow(): void {
  const mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1024,
    minHeight: 700,
    show: false,
    autoHideMenuBar: true,
    webPreferences: {
      preload: join(__dirname, "../preload/index.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  mainWindow.once("ready-to-show", () => mainWindow.show());

  const rendererUrl = process.env["ELECTRON_RENDERER_URL"];
  if (rendererUrl) {
    mainWindow.loadURL(rendererUrl);
  } else {
    mainWindow.loadFile(join(__dirname, "../renderer/index.html"));
  }
}

app.whenReady().then(() => {
  ipcMain.handle(IPC_OPEN_VOICE_RESULT, openVoiceResultFile);
  registerTapping();
  createMainWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createMainWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
