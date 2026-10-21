import { createServer } from "net";
import { join } from "path";
import { describe, expect, it, vi } from "vitest";
import {
  HEALTH_TIMEOUT_MS,
  MAX_BOUND,
  REASONS,
  STOP_GRACE_MS,
  TappingController,
  checkPrerequisites,
  clampBounds,
  isAllowedUrl,
  pickFreePort,
  resolvePython,
  resolveRepoRoot,
  safeDownloadName,
  stopProcess,
  streamlitArgs,
  waitForHealth,
  type ChildLike,
  type HealthDeps,
  type PrerequisiteDeps,
  type Timers,
} from "./tapping";
import type { TappingStatus } from "../shared/ipc";

// ---------------------------------------------------------------------------- helpers

/** A fake clock: sleeping moves time forward, nothing really waits. */
function fakeHealth(okAfter: number | null): HealthDeps & { calls: number; time: number } {
  const clock = {
    calls: 0,
    time: 0,
    now: () => clock.time,
    sleep: async (ms: number) => {
      clock.time += ms;
      // a loop that ignores its deadline must fail the test, not spin forever
      if (clock.time > 10 * HEALTH_TIMEOUT_MS) throw new Error("the polling did not stop at its deadline");
    },
    fetchOk: async () => {
      clock.calls += 1;
      return okAfter !== null && clock.calls > okAfter;
    },
  };
  return clock;
}

/** A fake clock for timers: run() fires everything that is due. */
function fakeTimers(): Timers & { fire(): void; pending: number } {
  const queue: { id: number; fn: () => void }[] = [];
  let next = 0;
  return {
    setTimeout(fn: () => void) {
      next += 1;
      queue.push({ id: next, fn });
      return next;
    },
    clearTimeout(handle: unknown) {
      const i = queue.findIndex((t) => t.id === handle);
      if (i >= 0) queue.splice(i, 1);
    },
    fire() {
      for (const t of queue.splice(0)) t.fn();
    },
    get pending() {
      return queue.length;
    },
  };
}

class FakeChild implements ChildLike {
  pid = 4242;
  exitCode: number | null = null;
  signalCode: NodeJS.Signals | null = null;
  signals: string[] = [];
  private listeners: (() => void)[] = [];
  private data: ((chunk: Buffer | string) => void)[] = [];
  stdout = { on: (_event: "data", l: (chunk: Buffer | string) => void) => void this.data.push(l) };
  stderr = { on: () => undefined }; // only stdout carries text in these tests
  /** Signals after which this child really exits. */
  constructor(private readonly diesOn: string[] = ["SIGTERM", "SIGKILL"]) {}
  kill(signal: NodeJS.Signals = "SIGTERM"): boolean {
    this.signals.push(signal);
    if (this.diesOn.includes(signal)) this.exit(signal);
    return true;
  }
  once(_event: "exit", listener: () => void): void {
    this.listeners.push(listener);
  }
  exit(signal: NodeJS.Signals | null = null, code: number | null = null): void {
    this.signalCode = signal;
    this.exitCode = signal ? null : (code ?? 0);
    for (const l of this.listeners.splice(0)) l();
  }
  emit(text: string): void {
    for (const l of this.data) l(text);
  }
}

const ROOT = join("/", "repo");
const ALL_PRESENT: PrerequisiteDeps = { exists: () => true, canImportStreamlit: async () => true };

// ---------------------------------------------------------------------------- port

describe("pickFreePort", () => {
  it("returns a usable port on 127.0.0.1 and releases it", async () => {
    const port = await pickFreePort();
    expect(port).toBeGreaterThan(0);
    // released: the same port can be taken again right away
    await new Promise<void>((done, fail) => {
      const server = createServer();
      server.once("error", fail);
      server.listen(port, "127.0.0.1", () => server.close(() => done()));
    });
  });

  it("closes the server before it resolves", async () => {
    const order: string[] = [];
    const fake = {
      once: () => fake,
      listen: (_port: number, _host: string, cb: () => void) => cb(),
      address: () => ({ port: 5555 }),
      close: (cb: () => void) => {
        order.push("closed");
        cb();
      },
    };
    const port = await pickFreePort((() => fake) as unknown as typeof createServer);
    order.push("resolved");
    expect(port).toBe(5555);
    expect(order).toEqual(["closed", "resolved"]);
  });
});

// ---------------------------------------------------------------------------- health

describe("waitForHealth", () => {
  it("succeeds after N failures", async () => {
    const health = fakeHealth(3);
    expect(await waitForHealth("http://127.0.0.1:1/_stcore/health", health)).toBe(true);
    expect(health.calls).toBe(4);
    expect(health.time).toBe(3 * 500); // polled every 500 ms
  });

  it("times out with a fake clock and returns false", async () => {
    const health = fakeHealth(null);
    expect(await waitForHealth("http://x", health)).toBe(false);
    expect(health.time).toBeGreaterThanOrEqual(HEALTH_TIMEOUT_MS);
    expect(health.calls).toBe(HEALTH_TIMEOUT_MS / 500);
  });

  it("never throws, even when the fetch does", async () => {
    const health = { ...fakeHealth(null), fetchOk: async () => Promise.reject(new Error("boom")) };
    await expect(waitForHealth("http://x", health, { timeoutMs: 2000 })).resolves.toBe(false);
  });

  it("stops early when aborted", async () => {
    const health = fakeHealth(null);
    expect(await waitForHealth("http://x", health, { aborted: () => true })).toBe(false);
    expect(health.calls).toBe(0);
  });
});

// ---------------------------------------------------------------------------- status

describe("checkPrerequisites", () => {
  const missing = (suffix: string): PrerequisiteDeps => ({
    exists: (path) => !path.endsWith(suffix),
    canImportStreamlit: async () => true,
  });
  const opts = { python: join(ROOT, ".venv", "bin", "python"), repoRoot: ROOT };

  it("is null when everything is there", async () => {
    expect(await checkPrerequisites(ALL_PRESENT, opts)).toBeNull();
  });

  it.each([
    ["python", join(".venv", "bin", "python"), REASONS.python],
    ["app.py", "app.py", REASONS.app],
    ["features.csv", "features.csv", REASONS.data],
    ["landmarker", "hand_landmarker.task", REASONS.landmarker],
  ])("reports a missing %s", async (_name, suffix, reason) => {
    expect(await checkPrerequisites(missing(suffix), opts)).toBe(reason);
  });

  it("reports streamlit that cannot be imported", async () => {
    const deps = { ...ALL_PRESENT, canImportStreamlit: async () => false };
    expect(await checkPrerequisites(deps, opts)).toBe(REASONS.streamlit);
  });

  it("checks in order: python, then streamlit, then app, data, landmarker", async () => {
    const none: PrerequisiteDeps = { exists: () => false, canImportStreamlit: async () => false };
    expect(await checkPrerequisites(none, opts)).toBe(REASONS.python);
    const noStreamlit = { exists: () => false, canImportStreamlit: async () => false };
    expect(await checkPrerequisites({ ...noStreamlit, exists: (p) => p === opts.python }, opts)).toBe(
      REASONS.streamlit,
    );
  });

  it("a bare command on the PATH is only checked by the import", async () => {
    const deps = { exists: () => false, canImportStreamlit: async () => true };
    expect(await checkPrerequisites(deps, { python: "python3", repoRoot: ROOT })).toBe(REASONS.app);
  });

  it("no reason contains a home-directory path or an absolute path", () => {
    for (const reason of Object.values(REASONS)) {
      expect(reason).not.toMatch(/\/Users\/|\/home\/|C:\\|~\/|\.venv/);
    }
  });
});

// ---------------------------------------------------------------------------- python path

describe("resolvePython / resolveRepoRoot", () => {
  it("the environment variable wins", () => {
    expect(resolvePython({ envPython: "/opt/py/bin/python", repoRoot: ROOT, platform: "darwin" })).toBe(
      "/opt/py/bin/python",
    );
    expect(resolvePython({ envPython: "  /x/python  ", repoRoot: ROOT, platform: "win32" })).toBe("/x/python");
  });

  it("falls back to the venv of the repository, per platform", () => {
    expect(resolvePython({ repoRoot: "/repo", platform: "darwin" })).toBe("/repo/.venv/bin/python");
    expect(resolvePython({ repoRoot: "/repo", platform: "linux", envPython: "" })).toBe("/repo/.venv/bin/python");
    expect(resolvePython({ repoRoot: "C:\\repo", platform: "win32" })).toBe(
      "C:\\repo\\.venv\\Scripts\\python.exe",
    );
  });

  it("the repository root is two folders above apps/desktop", () => {
    expect(resolveRepoRoot("/work/J26/apps/desktop")).toBe("/work/J26");
  });
});

// ---------------------------------------------------------------------------- allowlist

describe("isAllowedUrl", () => {
  const PORT = 51234;
  it("allows only 127.0.0.1 and the chosen port over http", () => {
    expect(isAllowedUrl(`http://127.0.0.1:${PORT}/`, PORT)).toBe(true);
    expect(isAllowedUrl(`http://127.0.0.1:${PORT}/_stcore/health`, PORT)).toBe(true);
    expect(isAllowedUrl(`http://127.0.0.1:${PORT}/?x=1#y`, PORT)).toBe(true);
  });

  it.each([
    ["localhost", `http://localhost:${PORT}/`],
    ["0.0.0.0", `http://0.0.0.0:${PORT}/`],
    ["another port", `http://127.0.0.1:${PORT + 1}/`],
    ["no port", "http://127.0.0.1/"],
    ["file:", "file:///etc/passwd"],
    ["https", `https://127.0.0.1:${PORT}/`],
    ["an outside site", "https://evil.example/"],
    ["a lookalike host", `http://127.0.0.1.evil.example:${PORT}/`],
    ["credentials", `http://user:pw@127.0.0.1:${PORT}/`],
    ["javascript:", "javascript:alert(1)"],
    ["garbage", "not a url"],
    ["empty", ""],
  ])("denies %s", (_name, url) => {
    expect(isAllowedUrl(url, PORT)).toBe(false);
  });
});

describe("safeDownloadName", () => {
  it("accepts a PDF with a plain name and strips folders", () => {
    expect(safeDownloadName("summary_001ee42d.pdf", "application/pdf")).toBe("summary_001ee42d.pdf");
    expect(safeDownloadName("/etc/../summary_abcd1234.pdf", "application/pdf")).toBe("summary_abcd1234.pdf");
  });
  it("falls back to a fixed name for odd names and refuses other types", () => {
    expect(safeDownloadName("we ird$.pdf", "application/pdf")).toBe("summary.pdf");
    expect(safeDownloadName("a.exe", "application/octet-stream")).toBeNull();
    expect(safeDownloadName("a.pdf", "text/html")).toBeNull();
  });
});

// ---------------------------------------------------------------------------- bounds

describe("clampBounds", () => {
  it("keeps a good rectangle (rounded)", () => {
    expect(clampBounds({ x: 10.4, y: 20.6, width: 800.2, height: 600 })).toEqual({
      x: 10, y: 21, width: 800, height: 600,
    });
  });
  it.each([
    ["NaN", { x: NaN, y: 0, width: 100, height: 100 }],
    ["Infinity", { x: 0, y: 0, width: Infinity, height: 100 }],
    ["zero width", { x: 0, y: 0, width: 0, height: 100 }],
    ["zero height", { x: 0, y: 0, width: 100, height: 0 }],
    ["negative size", { x: 0, y: 0, width: -5, height: 100 }],
    ["a string", { x: "1", y: 0, width: 100, height: 100 }],
    ["missing fields", { x: 0, y: 0 }],
    ["null", null],
    ["a number", 5],
  ])("rejects %s", (_name, input) => {
    expect(clampBounds(input)).toBeNull();
  });
  it("clamps a negative position to 0 and caps huge numbers", () => {
    expect(clampBounds({ x: -30, y: -1, width: 100, height: 100 })).toEqual({ x: 0, y: 0, width: 100, height: 100 });
    expect(clampBounds({ x: 1e9, y: 0, width: 1e9, height: 100 })).toEqual({
      x: MAX_BOUND, y: 0, width: MAX_BOUND, height: 100,
    });
  });
});

// ---------------------------------------------------------------------------- stop

describe("stopProcess", () => {
  it("sends SIGTERM and does not kill a process that exits", async () => {
    const child = new FakeChild(["SIGTERM"]);
    const timers = fakeTimers();
    await stopProcess(child, STOP_GRACE_MS, timers);
    expect(child.signals).toEqual(["SIGTERM"]);
    expect(timers.pending).toBe(0); // the kill timer was cancelled
  });

  it("sends SIGKILL after the timeout when the process is still alive", async () => {
    const child = new FakeChild(["SIGKILL"]); // ignores SIGTERM
    const timers = fakeTimers();
    const done = stopProcess(child, 100, timers);
    expect(child.signals).toEqual(["SIGTERM"]);
    timers.fire();
    await done;
    expect(child.signals).toEqual(["SIGTERM", "SIGKILL"]);
  });

  it("is safe twice, and for a process that already exited", async () => {
    const child = new FakeChild(["SIGTERM"]);
    const timers = fakeTimers();
    const first = stopProcess(child, 100, timers);
    const second = stopProcess(child, 100, timers);
    await Promise.all([first, second]);
    await stopProcess(child, 100, timers);
    expect(child.signals).toEqual(["SIGTERM"]);
    const gone = new FakeChild();
    gone.exit(null, 0);
    await stopProcess(gone, 100, timers);
    expect(gone.signals).toEqual([]);
  });
});

// ---------------------------------------------------------------------------- controller

function controller(opts: {
  reason?: string | null;
  health?: HealthDeps;
  child?: FakeChild;
  timers?: Timers;
} = {}) {
  const child = opts.child ?? new FakeChild();
  const events: string[] = [];
  const statuses: TappingStatus[] = [];
  const spawn = vi.fn(() => child);
  const c = new TappingController({
    repoRoot: ROOT,
    python: "/py",
    spawn,
    pickPort: async () => 40000,
    check: async () => (opts.reason === undefined ? null : opts.reason),
    health: opts.health ?? fakeHealth(1),
    showView: (url) => events.push(`show ${url}`),
    hideView: () => events.push("hide"),
    onStatus: (s) => statuses.push(s),
    timers: opts.timers ?? fakeTimers(),
    graceMs: 50,
  });
  return { c, child, events, statuses, spawn };
}

describe("TappingController", () => {
  it("starts Streamlit locally and shows the view when it is healthy", async () => {
    const { c, spawn, events, statuses } = controller();
    const status = await c.start();
    expect(status).toEqual({ state: "ready" });
    expect(spawn).toHaveBeenCalledWith("/py", streamlitArgs(40000), { cwd: ROOT });
    expect(events).toEqual(["show http://127.0.0.1:40000/"]);
    expect(statuses.map((s) => s.state)).toEqual(["starting", "ready"]);
    expect(c.currentUrl()).toBe("http://127.0.0.1:40000/");
  });

  it("builds the command line: headless, local address, no usage statistics", () => {
    const args = streamlitArgs(1234);
    expect(args.slice(0, 3)).toEqual(["-m", "streamlit", "run"]);
    expect(args).toContain(join("models", "tapping", "demo", "app.py"));
    const pairs = Object.fromEntries(args.slice(4).reduce<string[][]>((a, v, i, all) => (i % 2 === 0 ? [...a, [v, all[i + 1] ?? ""]] : a), []));
    expect(pairs).toEqual({
      "--server.headless": "true",
      "--server.address": "127.0.0.1",
      "--server.port": "1234",
      "--browser.gatherUsageStats": "false",
    });
    expect(args.join(" ")).not.toMatch(/enableCORS|enableXsrf/); // the defaults stay
  });

  it("is unavailable, and never spawns, when a prerequisite is missing", async () => {
    const { c, spawn, events } = controller({ reason: REASONS.data });
    expect(await c.start()).toEqual({ state: "unavailable", reason: REASONS.data });
    expect(spawn).not.toHaveBeenCalled();
    expect(events).toEqual([]);
  });

  it("is an error, and the child is stopped, when health never comes", async () => {
    const child = new FakeChild(["SIGTERM"]);
    const { c } = controller({ health: fakeHealth(null), child });
    const status = await c.start();
    expect(status).toEqual({ state: "error", reason: REASONS.timeout });
    expect(child.signals).toEqual(["SIGTERM"]);
    expect(c.hasChild()).toBe(false);
  });

  it("reports an error and hides the view when Streamlit exits by itself", async () => {
    const { c, child, events, statuses } = controller();
    await c.start();
    child.exit(null, 1);
    expect(events.at(-1)).toBe("hide");
    expect(statuses.at(-1)).toEqual({ state: "error", reason: REASONS.stopped });
    expect(c.currentUrl()).toBeNull();
  });

  it("stop hides the view, terminates the child, and can be repeated", async () => {
    const { c, child, events } = controller();
    await c.start();
    await c.stop();
    await c.stop();
    expect(events).toContain("hide");
    expect(child.signals).toEqual(["SIGTERM"]);
    expect(c.hasChild()).toBe(false);
    expect(c.getStatus()).toEqual({ state: "starting" }); // idle again
  });

  it("stop kills with SIGKILL when SIGTERM is ignored", async () => {
    const timers = fakeTimers();
    const child = new FakeChild(["SIGKILL"]);
    const { c } = controller({ child, timers });
    await c.start();
    const stopped = c.stop();
    await Promise.resolve();
    await Promise.resolve();
    timers.fire();
    await stopped;
    expect(child.signals).toEqual(["SIGTERM", "SIGKILL"]);
  });

  it("start twice gives one process; a stop during start cancels it", async () => {
    const first = controller({ health: fakeHealth(2) });
    const a = first.c.start();
    const b = first.c.start();
    await Promise.all([a, b]);
    expect(first.spawn).toHaveBeenCalledTimes(1);

    const second = controller({ health: fakeHealth(null) });
    const starting = second.c.start();
    await Promise.resolve();
    await Promise.resolve();
    await second.c.stop();
    await starting;
    expect(second.events).not.toContain("show http://127.0.0.1:40000/");
  });

  it("counts the child's output lines and keeps no text", async () => {
    const { c, child } = controller();
    await c.start();
    child.emit("one\ntwo\n");
    child.emit("three\n");
    expect(c.outputLines).toBe(3);
    expect(JSON.stringify(c)).not.toContain("three");
  });

  it("killNow kills at once without waiting", async () => {
    const { c, child } = controller();
    await c.start();
    c.killNow();
    expect(child.signals).toEqual(["SIGKILL"]);
    expect(c.hasChild()).toBe(false);
  });

  it("a failed check is an unavailable demo, never an exception", async () => {
    const c = new TappingController({
      repoRoot: ROOT, python: "/py", spawn: vi.fn(), pickPort: async () => 1,
      check: async () => Promise.reject(new Error("x")), health: fakeHealth(0),
      showView: () => {}, hideView: () => {}, onStatus: () => {},
    });
    expect((await c.start()).state).toBe("unavailable");
  });
});
