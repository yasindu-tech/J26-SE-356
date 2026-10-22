// Finger-tapping analysis (task FT-63): runs models/tapping/src/predict_clip.py on ONE video
// and returns its checked result. The renderer sends the video BYTES, never a path: with
// sandbox:true it has no file access, and a path from it would make main read whatever file
// it names. Main writes the bytes to a new private temp folder under a neutral name, runs the
// program with a time limit, and removes the folder in every case. Spawn, file system, timers
// and the prerequisite probes are injected, so this is tested without Python or Electron.
// Nothing here logs a path or the program's output; only "STAGE n/N label" lines are used.

import { join } from "path";
import type {
  TappingAnalysisReply,
  TappingAnalysisRequest,
  TappingErrorCode,
  TappingProgress,
} from "../shared/tappingIpc";
import { MAX_RESULT_BYTES, TARGETS, parseTappingResult } from "../shared/tappingResult";
import { REASONS, STOP_GRACE_MS, stopProcess, type Timers } from "./tapping";

export const ANALYSIS_TIMEOUT_MS = 180_000;
export const MAX_VIDEO_BYTES = 200 * 1024 * 1024;
export const EXTENSIONS = [".mp4", ".mov", ".m4v"] as const;
export const REQUIRED_MODULES = ["lightgbm", "mediapipe", "shap"] as const;
const WORK_NAME = "clip"; // the only file name that exists on disk
const STAGE_LINE = /^STAGE (\d{1,2})\/(\d{1,2}) ([A-Za-z0-9 ,.()-]{1,80})$/;

export const MESSAGES = {
  busy: "An analysis is already running. Wait for it to finish or stop it first.",
  invalidVideo: "Use an mp4, mov or m4v video of up to 200 MB.",
  invalidRequest: "The request was not understood.",
  models: "The research models for the score were not found on this computer.",
  invalidOutput: "The analysis finished but its result could not be read.",
  stopped: "The analysis stopped unexpectedly.",
  timeout: "The analysis took longer than 3 minutes and was stopped.",
  cancelled: "The analysis was stopped.",
  failedToStart: "The analysis could not be started.",
} as const;

export interface AnalysisChild {
  pid?: number;
  exitCode: number | null;
  signalCode: NodeJS.Signals | null;
  kill(signal?: NodeJS.Signals): boolean;
  once(event: "exit", listener: () => void): unknown;
  once(event: "error", listener: () => void): unknown;
  stdout?: { on(event: "data", listener: (chunk: Buffer | string) => void): unknown } | null;
  stderr?: { on(event: "data", listener: (chunk: Buffer | string) => void): unknown } | null;
}

export type AnalysisSpawn = (
  command: string,
  args: string[],
  options: { cwd: string },
) => AnalysisChild;

export interface AnalysisDeps {
  repoRoot: string;
  python: string;
  spawn: AnalysisSpawn;
  /** A new private folder; returns its path. */
  makeTempDir(): Promise<string>;
  writeFile(path: string, bytes: Uint8Array): Promise<void>;
  removeDir(path: string): Promise<void>;
  exists(path: string): boolean;
  canImport(python: string, modules: readonly string[]): Promise<boolean>;
  timers?: Timers;
  timeoutMs?: number;
  graceMs?: number;
}

const isBareCommand = (python: string): boolean => !python.includes("/") && !python.includes("\\");

/** null when everything the analysis needs is there, otherwise the plain reason (never a path). */
export async function checkAnalysisPrerequisites(
  deps: Pick<AnalysisDeps, "exists" | "canImport" | "repoRoot" | "python">,
): Promise<string | null> {
  const { python, repoRoot } = deps;
  if (!isBareCommand(python) && !deps.exists(python)) return REASONS.python;
  if (!(await deps.canImport(python, REQUIRED_MODULES))) {
    return isBareCommand(python) ? REASONS.python : MESSAGES.models;
  }
  const needs = [
    join(repoRoot, "models", "tapping", "src", "predict_clip.py"),
    join(repoRoot, "data", "processed", "features.csv"),
    join(repoRoot, "models", "tapping", "artifacts", "hand_landmarker.task"),
  ];
  return needs.every((p) => deps.exists(p)) ? null : MESSAGES.models;
}

/** The request after the checks main makes itself, or the plain reason it is refused. */
export function checkRequest(request: unknown): { ok: true; value: TappingAnalysisRequest } | { ok: false; message: string } {
  if (typeof request !== "object" || request === null) return { ok: false, message: MESSAGES.invalidRequest };
  const r = request as Record<string, unknown>;
  const bytes = r["bytes"];
  if (!(bytes instanceof Uint8Array)) return { ok: false, message: MESSAGES.invalidRequest };
  const target = TARGETS.find((t) => t === r["target"]);
  if (target === undefined) return { ok: false, message: MESSAGES.invalidRequest };
  const extension = typeof r["extension"] === "string" ? r["extension"].toLowerCase() : "";
  if (!EXTENSIONS.some((e) => e === extension) || bytes.length < 1 || bytes.length > MAX_VIDEO_BYTES) {
    return { ok: false, message: MESSAGES.invalidVideo };
  }
  return { ok: true, value: { bytes, extension, target } };
}

/** "STAGE 2/5 Tracking the hand" to a progress event; any other line is dropped. */
export function parseStageLine(line: string): TappingProgress | null {
  const m = STAGE_LINE.exec(line.trim());
  if (!m) return null;
  const stage = Number(m[1]);
  const total = Number(m[2]);
  if (stage < 1 || total < 1 || stage > total) return null;
  return { stage, total, label: m[3] ?? "" };
}

/** Pulls whole lines out of a stream that arrives in pieces. */
export class LineSplitter {
  private rest = "";
  push(chunk: Buffer | string): string[] {
    const parts = (this.rest + String(chunk)).split("\n");
    this.rest = (parts.pop() ?? "").slice(-400); // a runaway line is cut, never kept whole
    return parts;
  }
}

const error = (code: TappingErrorCode, message: string): TappingAnalysisReply => ({ kind: "error", code, message });

const realTimers: Timers = {
  setTimeout: (fn, ms) => setTimeout(fn, ms),
  clearTimeout: (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>),
};

interface Active {
  cancelled: boolean;
  child: AnalysisChild | null;
}

export class TappingAnalyser {
  private active: Active | null = null;

  constructor(private readonly deps: AnalysisDeps) {}

  isBusy(): boolean {
    return this.active !== null;
  }

  /** One analysis at a time; a second call is refused at once. Never throws. */
  async run(raw: unknown, onProgress: (p: TappingProgress) => void): Promise<TappingAnalysisReply> {
    if (this.active) return error("busy", MESSAGES.busy);
    const checked = checkRequest(raw);
    if (!checked.ok) return error("invalid", checked.message);
    const mine: Active = { cancelled: false, child: null };
    this.active = mine;
    let folder: string | null = null;
    try {
      const reason = await checkAnalysisPrerequisites(this.deps);
      if (mine.cancelled) return error("cancelled", MESSAGES.cancelled);
      if (reason) return error("unavailable", reason);
      folder = await this.deps.makeTempDir();
      const clip = join(folder, WORK_NAME + checked.value.extension);
      await this.deps.writeFile(clip, checked.value.bytes);
      if (mine.cancelled) return error("cancelled", MESSAGES.cancelled);
      return await this.execute(mine, clip, checked.value.target, onProgress);
    } catch {
      return error("stopped", MESSAGES.failedToStart);
    } finally {
      const dir = folder;
      this.active = null;
      if (dir !== null) await this.deps.removeDir(dir).catch(() => undefined);
    }
  }

  /** SIGTERM, then SIGKILL; resolves when the program has ended. Safe when nothing runs. */
  async cancel(): Promise<void> {
    const mine = this.active;
    if (!mine) return;
    mine.cancelled = true;
    if (mine.child) await stopProcess(mine.child, this.deps.graceMs ?? STOP_GRACE_MS, this.deps.timers);
  }

  private execute(
    mine: Active,
    clip: string,
    target: string,
    onProgress: (p: TappingProgress) => void,
  ): Promise<TappingAnalysisReply> {
    const { deps } = this;
    const timers = deps.timers ?? realTimers;
    const args = [join("models", "tapping", "src", "predict_clip.py"), "--video", clip, "--target", target];
    return new Promise((resolve) => {
      let child: AnalysisChild;
      try {
        child = deps.spawn(deps.python, args, { cwd: deps.repoRoot });
      } catch {
        resolve(error("stopped", MESSAGES.failedToStart));
        return;
      }
      mine.child = child;
      const out: Buffer[] = [];
      let size = 0;
      let oversized = false;
      let timedOut = false;
      let settled = false;
      const lines = new LineSplitter();
      const stop = (): void => void stopProcess(child, deps.graceMs ?? STOP_GRACE_MS, timers);

      child.stdout?.on("data", (chunk) => {
        const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
        size += buf.length;
        if (size > MAX_RESULT_BYTES) {
          oversized = true;
          stop();
          return;
        }
        out.push(buf);
      });
      child.stderr?.on("data", (chunk) => {
        for (const line of lines.push(chunk)) {
          const progress = parseStageLine(line);
          if (progress) onProgress(progress);
        }
      });
      const timer = timers.setTimeout(() => {
        timedOut = true;
        stop();
      }, deps.timeoutMs ?? ANALYSIS_TIMEOUT_MS);

      const finish = (): void => {
        if (settled) return;
        settled = true;
        timers.clearTimeout(timer);
        mine.child = null;
        if (mine.cancelled) return resolve(error("cancelled", MESSAGES.cancelled));
        if (timedOut) return resolve(error("timeout", MESSAGES.timeout));
        if (oversized) return resolve(error("invalid", MESSAGES.invalidOutput));
        if (child.exitCode !== 0) return resolve(error("stopped", MESSAGES.stopped));
        const result = parseTappingResult(Buffer.concat(out).toString("utf8"));
        resolve(result ? { kind: "result", result } : error("invalid", MESSAGES.invalidOutput));
      };
      child.once("exit", finish);
      child.once("error", finish);
      if (mine.cancelled) stop(); // a stop that arrived while the program was starting
    });
  }
}

// ---------------------------------------------------------------------------- the PDF

/** The summary_<ref>.pdf name for a hashed ref, or null for anything else. */
export function pdfFileName(ref: unknown): string | null {
  return typeof ref === "string" && /^[0-9a-f]{8}$/.test(ref) ? `summary_${ref}.pdf` : null;
}

/** The PDF bytes of a base64 text, or null when it is not a PDF of a sensible size. */
export function decodePdf(base64: unknown): Buffer | null {
  if (typeof base64 !== "string" || base64.length === 0 || base64.length > 14 * 1024 * 1024) return null;
  if (!/^[A-Za-z0-9+/]+={0,2}$/.test(base64)) return null;
  const bytes = Buffer.from(base64, "base64");
  return bytes.subarray(0, 5).toString("latin1") === "%PDF-" ? bytes : null;
}
