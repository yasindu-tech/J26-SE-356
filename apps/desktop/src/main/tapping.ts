// Finger-tapping demo: the logic that runs Streamlit as a child process of the main process
// (task FT-61). Everything with side effects is injected (spawn, port picker, health fetch,
// clock, timers), so it is tested without Electron or Streamlit. The Electron glue (view,
// IPC, downloads) is in tappingWindow.ts.

import { win32, posix, join, resolve } from "path";
import { createServer } from "net";
import type { TappingBounds, TappingStatus } from "../shared/tappingIpc";

export const HOST = "127.0.0.1";
export const HEALTH_INTERVAL_MS = 500;
export const HEALTH_TIMEOUT_MS = 60_000;
export const STOP_GRACE_MS = 5_000;
export const MAX_BOUND = 20_000;

// ---------------------------------------------------------------------------- paths

/** The repository root from Electron's app path. In development and `electron-vite preview`
 * `app.getAppPath()` is apps/desktop, so the root is two folders up. A packaged build is not
 * supported yet. */
export function resolveRepoRoot(appPath: string): string {
  return resolve(appPath, "..", "..");
}

/** MAIN_VITE_TAPPING_PYTHON wins; otherwise the repository's own virtual environment. */
export function resolvePython(opts: {
  envPython?: string;
  repoRoot: string;
  platform: NodeJS.Platform;
}): string {
  const fromEnv = opts.envPython?.trim();
  if (fromEnv) return fromEnv;
  return opts.platform === "win32"
    ? win32.join(opts.repoRoot, ".venv", "Scripts", "python.exe")
    : posix.join(opts.repoRoot, ".venv", "bin", "python");
}

// ---------------------------------------------------------------------------- port

/** A free port on 127.0.0.1: listen on port 0, read the port, and close the server first. */
export function pickFreePort(create: typeof createServer = createServer): Promise<number> {
  return new Promise((resolvePort, reject) => {
    const server = create();
    server.once("error", reject);
    server.listen(0, HOST, () => {
      const address = server.address();
      const port = typeof address === "object" && address !== null ? address.port : 0;
      server.close((error) => {
        if (error || port === 0) reject(error ?? new Error("no port was assigned"));
        else resolvePort(port);
      });
    });
  });
}

// ---------------------------------------------------------------------------- prerequisites

export interface PrerequisiteDeps {
  exists(path: string): boolean;
  canImportStreamlit(python: string): Promise<boolean>;
}

export const REASONS = {
  python: "The Python environment for the demo was not found. Set it up as described in the desktop README, or set MAIN_VITE_TAPPING_PYTHON.",
  streamlit: "Streamlit is not installed in the demo's Python environment.",
  app: "The demo's files were not found in this copy of the project.",
  data: "The research data the demo needs was not found on this computer (data/processed).",
  landmarker: "The hand-tracking model file was not found (models/tapping/artifacts).",
  timeout: "The demo did not start within 60 seconds.",
  stopped: "The demo stopped unexpectedly.",
  failedToStart: "The demo could not be started.",
} as const;

const isBareCommand = (python: string): boolean => !python.includes("/") && !python.includes("\\");

/** null when everything the demo needs is there, otherwise the plain reason (never a path). */
export async function checkPrerequisites(
  deps: PrerequisiteDeps,
  opts: { python: string; repoRoot: string },
): Promise<string | null> {
  const { python, repoRoot } = opts;
  if (!isBareCommand(python) && !deps.exists(python)) return REASONS.python;
  if (!(await deps.canImportStreamlit(python))) {
    return isBareCommand(python) ? REASONS.python : REASONS.streamlit;
  }
  if (!deps.exists(join(repoRoot, "models", "tapping", "demo", "app.py"))) return REASONS.app;
  if (!deps.exists(join(repoRoot, "data", "processed", "features.csv"))) return REASONS.data;
  if (!deps.exists(join(repoRoot, "models", "tapping", "artifacts", "hand_landmarker.task"))) {
    return REASONS.landmarker;
  }
  return null;
}

// ---------------------------------------------------------------------------- health

export interface HealthDeps {
  fetchOk(url: string): Promise<boolean>; // true only for HTTP 200
  now(): number;
  sleep(ms: number): Promise<void>;
}

/** Polls until the server answers 200. False on timeout or when `aborted` turns true. Never throws. */
export async function waitForHealth(
  url: string,
  deps: HealthDeps,
  opts: { intervalMs?: number; timeoutMs?: number; aborted?: () => boolean } = {},
): Promise<boolean> {
  const interval = opts.intervalMs ?? HEALTH_INTERVAL_MS;
  const deadline = deps.now() + (opts.timeoutMs ?? HEALTH_TIMEOUT_MS);
  while (deps.now() < deadline) {
    if (opts.aborted?.()) return false;
    let ok = false;
    try {
      ok = await deps.fetchOk(url);
    } catch {
      ok = false;
    }
    if (ok) return true;
    await deps.sleep(interval);
  }
  return false;
}

// ---------------------------------------------------------------------------- allowlist

/** Only the demo's own address: http, 127.0.0.1 (not localhost), the chosen port. */
export function isAllowedUrl(raw: string, port: number): boolean {
  try {
    const url = new URL(raw);
    return (
      url.protocol === "http:" &&
      url.hostname === HOST &&
      url.port === String(port) &&
      url.username === "" &&
      url.password === ""
    );
  } catch {
    return false;
  }
}

/** A saved download is a PDF with a plain file name; anything else is refused (null). */
export function safeDownloadName(suggested: string, mime: string): string | null {
  if (mime.toLowerCase() !== "application/pdf") return null;
  const base = suggested.split(/[\\/]/).pop() ?? "";
  return /^[A-Za-z0-9_.-]{1,64}\.pdf$/i.test(base) ? base : "summary.pdf";
}

// ---------------------------------------------------------------------------- bounds

/** A valid view rectangle, or null. Non-finite numbers and empty sizes are rejected; the
 * position is clamped to the window and every number is capped and rounded. */
export function clampBounds(input: unknown): TappingBounds | null {
  if (typeof input !== "object" || input === null) return null;
  const b = input as Record<string, unknown>;
  const values = [b["x"], b["y"], b["width"], b["height"]];
  if (!values.every((v): v is number => typeof v === "number" && Number.isFinite(v))) return null;
  const [x, y, width, height] = values as [number, number, number, number];
  if (width < 1 || height < 1) return null;
  const cap = (v: number): number => Math.min(MAX_BOUND, Math.max(0, Math.round(v)));
  return { x: cap(x), y: cap(y), width: cap(width), height: cap(height) };
}

// ---------------------------------------------------------------------------- stopping

export interface ProcessLike {
  pid?: number;
  exitCode: number | null;
  signalCode: NodeJS.Signals | null;
  kill(signal?: NodeJS.Signals): boolean;
  once(event: "exit", listener: () => void): unknown;
}

export interface Timers {
  setTimeout(fn: () => void, ms: number): unknown;
  clearTimeout(handle: unknown): void;
}

const realTimers: Timers = {
  setTimeout: (fn, ms) => setTimeout(fn, ms),
  clearTimeout: (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>),
};

const stopping = new WeakMap<object, Promise<void>>();

const hasExited = (child: ProcessLike): boolean => child.exitCode !== null || child.signalCode !== null;

/** SIGTERM, then SIGKILL if it is still alive after `graceMs`. Safe to call twice. */
export function stopProcess(
  child: ProcessLike,
  graceMs: number = STOP_GRACE_MS,
  timers: Timers = realTimers,
): Promise<void> {
  const pending = stopping.get(child);
  if (pending) return pending;
  if (hasExited(child)) return Promise.resolve();
  const done = new Promise<void>((finish) => {
    let exited = false;
    const timer = timers.setTimeout(() => {
      if (!exited) child.kill("SIGKILL");
    }, graceMs);
    child.once("exit", () => {
      exited = true;
      timers.clearTimeout(timer);
      finish();
    });
    child.kill("SIGTERM");
  });
  stopping.set(child, done);
  return done;
}

// ---------------------------------------------------------------------------- controller

export interface ChildLike extends ProcessLike {
  stdout?: { on(event: "data", listener: (chunk: Buffer | string) => void): unknown } | null;
  stderr?: { on(event: "data", listener: (chunk: Buffer | string) => void): unknown } | null;
}

export type SpawnFn = (
  command: string,
  args: string[],
  options: { cwd: string },
) => ChildLike;

export interface ControllerDeps {
  repoRoot: string;
  python: string;
  spawn: SpawnFn;
  pickPort(): Promise<number>;
  check(): Promise<string | null>;
  health: HealthDeps;
  showView(url: string): void;
  hideView(): void;
  onStatus(status: TappingStatus): void;
  timers?: Timers;
  graceMs?: number;
  healthTimeoutMs?: number;
}

/** Streamlit's command line: local only, headless, no usage statistics. */
export function streamlitArgs(port: number): string[] {
  return [
    "-m", "streamlit", "run", join("models", "tapping", "demo", "app.py"),
    "--server.headless", "true",
    "--server.address", HOST,
    "--server.port", String(port),
    "--browser.gatherUsageStats", "false",
  ]; // prettier-ignore
}

export class TappingController {
  private status: TappingStatus | null = null; // null: not started
  private child: ChildLike | null = null;
  private port = 0;
  private generation = 0;
  private queue: Promise<unknown> = Promise.resolve();
  outputLines = 0; // the child's output is only counted, never kept or logged

  constructor(private readonly deps: ControllerDeps) {}

  getStatus(): TappingStatus {
    return this.status ?? { state: "starting" };
  }

  /** The demo's own address while it is ready, else null. */
  currentUrl(): string | null {
    return this.status?.state === "ready" ? `http://${HOST}:${this.port}/` : null;
  }

  start(): Promise<TappingStatus> {
    const state = this.status?.state;
    if (state === "starting" || state === "ready") return Promise.resolve(this.getStatus());
    this.status = { state: "starting" }; // at once, so a second call cannot start another process
    const run = this.queue.then(() => this.begin());
    this.queue = run.catch(() => undefined);
    return run;
  }

  stop(): Promise<void> {
    this.generation += 1;
    this.status = null;
    this.deps.hideView();
    const child = this.child;
    this.child = null;
    const run = this.queue.then(() =>
      child ? stopProcess(child, this.deps.graceMs, this.deps.timers) : undefined,
    );
    this.queue = run.catch(() => undefined);
    return run;
  }

  /** Synchronous last resort for app quit: no waiting. */
  killNow(): void {
    this.generation += 1;
    this.child?.kill("SIGKILL");
    this.child = null;
    this.status = null;
  }

  hasChild(): boolean {
    return this.child !== null;
  }

  private set(status: TappingStatus): TappingStatus {
    this.status = status;
    this.deps.onStatus(status);
    return status;
  }

  private async begin(): Promise<TappingStatus> {
    const generation = ++this.generation;
    this.set({ state: "starting" });
    const stale = (): boolean => generation !== this.generation;

    let reason: string | null;
    try {
      reason = await this.deps.check();
    } catch {
      reason = REASONS.failedToStart;
    }
    if (stale()) return this.getStatus();
    if (reason) return this.set({ state: "unavailable", reason });

    let child: ChildLike;
    try {
      this.port = await this.deps.pickPort();
      if (stale()) return this.getStatus();
      child = this.deps.spawn(this.deps.python, streamlitArgs(this.port), { cwd: this.deps.repoRoot });
    } catch {
      return this.set({ state: "error", reason: REASONS.failedToStart });
    }
    this.child = child;
    const count = (chunk: Buffer | string): void => {
      this.outputLines += String(chunk).split("\n").length - 1;
    };
    child.stdout?.on("data", count);
    child.stderr?.on("data", count);
    child.once("exit", () => {
      if (this.child === child && !stale()) {
        this.child = null;
        this.deps.hideView();
        this.set({ state: "error", reason: REASONS.stopped });
      }
    });

    const url = `http://${HOST}:${this.port}/`;
    const ok = await waitForHealth(`${url}_stcore/health`, this.deps.health, {
      timeoutMs: this.deps.healthTimeoutMs,
      aborted: stale,
    });
    if (stale()) return this.getStatus();
    if (!ok) {
      this.child = null;
      await stopProcess(child, this.deps.graceMs, this.deps.timers);
      return this.set({ state: "error", reason: REASONS.timeout });
    }
    this.deps.showView(url);
    return this.set({ state: "ready" });
  }
}
