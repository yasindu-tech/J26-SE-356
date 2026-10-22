import { join } from "path";
import { describe, expect, it, vi } from "vitest";
import type { TappingProgress } from "../shared/tappingIpc";
import { MAX_RESULT_BYTES, validateTappingResult } from "../shared/tappingResult";
import { failedFixture, scoredFixture } from "../shared/tappingResult.fixture";
import {
  ANALYSIS_TIMEOUT_MS,
  EXTENSIONS,
  LineSplitter,
  MAX_VIDEO_BYTES,
  MESSAGES,
  TappingAnalyser,
  checkAnalysisPrerequisites,
  checkRequest,
  decodePdf,
  parseStageLine,
  pdfFileName,
  type AnalysisChild,
  type AnalysisDeps,
} from "./tappingAnalysis";
import type { Timers } from "./tapping";

// ---------------------------------------------------------------------------- helpers

class FakeChild implements AnalysisChild {
  pid = 777;
  exitCode: number | null = null;
  signalCode: NodeJS.Signals | null = null;
  signals: string[] = [];
  private exitListeners: (() => void)[] = [];
  private errorListeners: (() => void)[] = [];
  private out: ((chunk: Buffer | string) => void)[] = [];
  private err: ((chunk: Buffer | string) => void)[] = [];
  stdout = { on: (_e: "data", l: (chunk: Buffer | string) => void) => void this.out.push(l) };
  stderr = { on: (_e: "data", l: (chunk: Buffer | string) => void) => void this.err.push(l) };
  constructor(private readonly diesOn: string[] = ["SIGTERM", "SIGKILL"]) {}
  kill(signal: NodeJS.Signals = "SIGTERM"): boolean {
    this.signals.push(signal);
    if (this.diesOn.includes(signal)) this.end(null, signal);
    return true;
  }
  once(event: "exit" | "error", listener: () => void): void {
    (event === "exit" ? this.exitListeners : this.errorListeners).push(listener);
  }
  writeOut(chunk: Buffer | string): void {
    for (const l of this.out) l(chunk);
  }
  writeErr(chunk: Buffer | string): void {
    for (const l of this.err) l(chunk);
  }
  end(code: number | null, signal: NodeJS.Signals | null = null): void {
    this.exitCode = code;
    this.signalCode = signal;
    for (const l of this.exitListeners) l();
  }
  fail(): void {
    for (const l of this.errorListeners) l();
  }
}

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

const REPO = "/repo";
const PYTHON = "/repo/.venv/bin/python";

function setup(over: Partial<AnalysisDeps> = {}) {
  const dirs: string[] = [];
  const removed: string[] = [];
  const written: { path: string; bytes: Uint8Array }[] = [];
  const spawned: { command: string; args: string[]; cwd: string }[] = [];
  const children: FakeChild[] = [];
  const timers = fakeTimers();
  const deps: AnalysisDeps = {
    repoRoot: REPO,
    python: PYTHON,
    spawn: (command, args, options) => {
      spawned.push({ command, args, cwd: options.cwd });
      const child = new FakeChild();
      children.push(child);
      return child;
    },
    makeTempDir: async () => {
      const dir = `/tmp/pdxai-tapping-${dirs.length}`;
      dirs.push(dir);
      return dir;
    },
    writeFile: async (path, bytes) => void written.push({ path, bytes }),
    removeDir: async (path) => void removed.push(path),
    exists: () => true,
    canImport: async () => true,
    timers,
    graceMs: 10,
    ...over,
  };
  const progress: TappingProgress[] = [];
  const analyser = new TappingAnalyser(deps);
  const request = { bytes: new Uint8Array([1, 2, 3]), extension: ".MP4", target: "ge2" };
  const run = (req: unknown = request) => analyser.run(req, (p) => progress.push(p));
  const child = async (i = 0): Promise<FakeChild> => {
    await vi.waitFor(() => expect(children.length).toBeGreaterThan(i));
    return children[i] as FakeChild;
  };
  return { analyser, deps, dirs, removed, written, spawned, children, timers, progress, run, child, request };
}

const goodJson = (): string => JSON.stringify(scoredFixture());

// ---------------------------------------------------------------------------- the request

describe("checkRequest", () => {
  const ok = { bytes: new Uint8Array([1]), extension: ".mov", target: "ge1" };

  it("accepts a small mp4, mov or m4v for ge2 or ge1 and lowercases the extension", () => {
    for (const extension of EXTENSIONS) {
      expect(checkRequest({ ...ok, extension: extension.toUpperCase() })).toMatchObject({ ok: true });
    }
    expect(checkRequest({ ...ok, target: "ge2" })).toMatchObject({ ok: true });
  });

  it("refuses anything else, with a message that has no path", () => {
    const tooBig = Object.defineProperty(new Uint8Array(1), "length", { value: MAX_VIDEO_BYTES + 1 });
    const bad: unknown[] = [
      null,
      "x",
      {},
      { ...ok, bytes: [1, 2, 3] },
      { ...ok, bytes: new Uint8Array(0) },
      { ...ok, bytes: tooBig },
      { ...ok, extension: ".avi" },
      { ...ok, extension: "../../etc/passwd" },
      { ...ok, extension: 5 },
      { ...ok, target: "ge3" },
      { ...ok, target: undefined },
    ];
    for (const request of bad) {
      const checked = checkRequest(request);
      expect(checked.ok).toBe(false);
      if (!checked.ok) expect(checked.message).not.toMatch(/[\\/]/);
    }
  });
});

describe("progress lines", () => {
  it("parses a STAGE line and drops every other line", () => {
    expect(parseStageLine("STAGE 2/5 Tracking the hand and finding taps")).toEqual({
      stage: 2,
      total: 5,
      label: "Tracking the hand and finding taps",
    });
    for (const line of [
      "",
      "STAGE 0/5 x",
      "STAGE 6/5 x",
      "STAGE 2/5 /Users/someone/clip.mp4",
      "I0000 gl_context.cc:407] GL version: 2.1 /private/tmp/x.task",
      "Traceback (most recent call last):",
    ]) {
      expect(parseStageLine(line)).toBeNull();
    }
  });

  it("joins a line that arrives in pieces", () => {
    const s = new LineSplitter();
    expect(s.push("STAGE 1/5 Rea")).toEqual([]);
    expect(s.push("ding the video\nSTAGE 2/5 Tr")).toEqual(["STAGE 1/5 Reading the video"]);
    expect(s.push(Buffer.from("acking\n"))).toEqual(["STAGE 2/5 Tracking"]);
  });
});

// ---------------------------------------------------------------------------- prerequisites

describe("checkAnalysisPrerequisites", () => {
  const base = { repoRoot: REPO, python: PYTHON, exists: () => true, canImport: async () => true };

  it("is null when everything is there", async () => {
    expect(await checkAnalysisPrerequisites(base)).toBeNull();
  });

  it("asks for the three modules the program needs", async () => {
    const seen: (readonly string[])[] = [];
    await checkAnalysisPrerequisites({ ...base, canImport: async (_p, m) => (seen.push(m), true) });
    expect(seen[0]).toEqual(["lightgbm", "mediapipe", "shap"]);
  });

  it("names the missing piece without a path", async () => {
    const reasons = [
      await checkAnalysisPrerequisites({ ...base, exists: () => false }),
      await checkAnalysisPrerequisites({ ...base, canImport: async () => false }),
      await checkAnalysisPrerequisites({ ...base, exists: (p) => !p.endsWith("features.csv") }),
      await checkAnalysisPrerequisites({ ...base, exists: (p) => !p.endsWith("hand_landmarker.task") }),
      await checkAnalysisPrerequisites({ ...base, exists: (p) => !p.endsWith("predict_clip.py") }),
      await checkAnalysisPrerequisites({ ...base, python: "python3", canImport: async () => false }),
    ];
    for (const reason of reasons) {
      expect(reason).toEqual(expect.any(String));
      expect(reason).not.toMatch(/\/repo|\.venv|[\\/]/);
    }
  });
});

// ---------------------------------------------------------------------------- the run

describe("TappingAnalyser.run", () => {
  it("runs the program on a temp copy and returns the checked result", async () => {
    const t = setup();
    const done = t.run();
    const child = await t.child();
    child.writeErr("STAGE 1/5 Reading the video\nSTAGE 2/5 Tracking the hand and finding taps\n");
    child.writeOut(goodJson().slice(0, 1000));
    child.writeOut(goodJson().slice(1000));
    child.end(0);
    const reply = await done;
    expect(reply).toMatchObject({ kind: "result", result: { status: "scored", clip_ref: "f0e37b01" } });
    expect(t.progress.map((p) => p.stage)).toEqual([1, 2]);
    expect(t.spawned).toEqual([
      {
        command: PYTHON,
        cwd: REPO,
        args: [join("models", "tapping", "src", "predict_clip.py"), "--video", "/tmp/pdxai-tapping-0/clip.mp4", "--target", "ge2"],
      },
    ]);
    expect(t.written).toHaveLength(1);
    expect(Array.from(t.written[0]?.bytes ?? [])).toEqual([1, 2, 3]);
    expect(t.removed).toEqual(t.dirs);
    expect(t.timers.pending).toBe(0);
  });

  it("passes a failed record through as a result", async () => {
    const t = setup();
    const done = t.run();
    const child = await t.child();
    child.writeOut(JSON.stringify(failedFixture()));
    child.end(0);
    expect(await done).toMatchObject({ kind: "result", result: { status: "failed" } });
  });

  it("does not forward stderr lines that are not progress lines", async () => {
    const t = setup();
    const done = t.run();
    const child = await t.child();
    child.writeErr("W0000 landmark_projection_calculator.cc:81] /Users/x/models/hand_landmarker.task\n");
    child.writeErr("STAGE 3/5 Scoring the clip\n");
    child.end(0);
    await done;
    expect(t.progress).toEqual([{ stage: 3, total: 5, label: "Scoring the clip" }]);
  });

  const outputs: [string, () => string][] = [
    ["invalid JSON", () => "{not json"],
    ["a wrong schema_version", () => JSON.stringify({ ...scoredFixture(), schema_version: "tapping-result/2" })],
    ["a missing schema_version", () => JSON.stringify({ ...scoredFixture(), schema_version: undefined })],
    ["an unknown status", () => JSON.stringify({ ...scoredFixture(), status: "abstained" })],
    ["mismatched array lengths", () => JSON.stringify({ ...scoredFixture(), signal: { ...scoredFixture().signal, smoothed: [1] } })],
    ["an extra key", () => JSON.stringify({ ...scoredFixture(), label: "x" })],
    ["empty output", () => ""],
  ];

  it.each(outputs)("returns an invalid-output error for %s and removes the temp folder", async (_name, make) => {
    const t = setup();
    const done = t.run();
    const child = await t.child();
    child.writeOut(make());
    child.end(0);
    const reply = await done;
    expect(reply).toEqual({ kind: "error", code: "invalid", message: MESSAGES.invalidOutput });
    expect(t.removed).toEqual(t.dirs);
  });

  it("stops and reports invalid output when the program writes more than the limit", async () => {
    const t = setup();
    const done = t.run();
    const child = await t.child();
    child.writeOut(Buffer.alloc(MAX_RESULT_BYTES + 1, 97));
    const reply = await done;
    expect(reply).toMatchObject({ kind: "error", code: "invalid" });
    expect(child.signals).toContain("SIGTERM");
    expect(t.removed).toEqual(t.dirs);
  });

  it("reports a program that exits with an error", async () => {
    const t = setup();
    const done = t.run();
    const child = await t.child();
    child.writeOut(goodJson());
    child.end(1);
    expect(await done).toEqual({ kind: "error", code: "stopped", message: MESSAGES.stopped });
    expect(t.removed).toEqual(t.dirs);
  });

  it("reports a program that could not start (error event)", async () => {
    const t = setup();
    const done = t.run();
    (await t.child()).fail();
    expect(await done).toMatchObject({ kind: "error", code: "stopped" });
    expect(t.removed).toEqual(t.dirs);
  });

  it("times out after 180 s: SIGTERM, the timeout message, folder removed", async () => {
    const t = setup();
    expect(ANALYSIS_TIMEOUT_MS).toBe(180_000);
    const done = t.run();
    const child = await t.child();
    child.writeOut(goodJson()); // even complete-looking output does not rescue a timed-out run
    t.timers.fire();
    expect(await done).toEqual({ kind: "error", code: "timeout", message: MESSAGES.timeout });
    expect(child.signals).toEqual(["SIGTERM"]);
    expect(t.removed).toEqual(t.dirs);
  });

  it("uses the 180 s limit when none is injected", async () => {
    const seen: number[] = [];
    const timers: Timers = { setTimeout: (_fn, ms) => (seen.push(ms), 1), clearTimeout: () => undefined };
    const t = setup({ timers });
    const done = t.run();
    (await t.child()).end(0);
    await done;
    expect(seen[0]).toBe(180_000);
  });

  it("escalates to SIGKILL when SIGTERM is ignored", async () => {
    const stubborn = new FakeChild(["SIGKILL"]);
    const t = setup({ spawn: () => stubborn });
    const done = t.run();
    await vi.waitFor(() => expect(t.dirs).toHaveLength(1));
    await vi.waitFor(() => expect(t.timers.pending).toBeGreaterThan(0));
    t.timers.fire(); // the time limit
    await vi.waitFor(() => expect(stubborn.signals).toEqual(["SIGTERM"]));
    t.timers.fire(); // the grace period
    expect(await done).toMatchObject({ kind: "error", code: "timeout" });
    expect(stubborn.signals).toEqual(["SIGTERM", "SIGKILL"]);
  });

  it("cancel stops the program and reports cancelled", async () => {
    const t = setup();
    const done = t.run();
    const child = await t.child();
    await t.analyser.cancel();
    expect(await done).toEqual({ kind: "error", code: "cancelled", message: MESSAGES.cancelled });
    expect(child.signals).toEqual(["SIGTERM"]);
    expect(t.removed).toEqual(t.dirs);
    expect(t.analyser.isBusy()).toBe(false);
  });

  it("cancel with nothing running does nothing", async () => {
    const t = setup();
    await expect(t.analyser.cancel()).resolves.toBeUndefined();
  });

  it("a cancel that arrives before the program starts still ends the run", async () => {
    let release: () => void = () => undefined;
    const gate = new Promise<void>((done) => (release = done));
    const t = setup({ writeFile: async () => gate });
    const done = t.run();
    await vi.waitFor(() => expect(t.dirs).toHaveLength(1));
    const cancelled = t.analyser.cancel();
    release();
    await cancelled;
    expect(await done).toMatchObject({ kind: "error", code: "cancelled" });
    expect(t.spawned).toHaveLength(0);
    expect(t.removed).toEqual(t.dirs);
  });

  it("refuses a second analysis while one runs, and allows one afterwards", async () => {
    const t = setup();
    const first = t.run();
    const child = await t.child();
    expect(await t.run()).toEqual({ kind: "error", code: "busy", message: MESSAGES.busy });
    expect(t.spawned).toHaveLength(1);
    child.writeOut(goodJson());
    child.end(0);
    expect((await first).kind).toBe("result");
    const second = t.run();
    (await t.child(1)).end(1);
    expect((await second).kind).toBe("error");
    expect(t.spawned).toHaveLength(2);
  });

  it("two calls made at once start one program", async () => {
    const t = setup();
    const a = t.run();
    const b = t.run();
    expect((await b).kind).toBe("error");
    (await t.child()).end(1);
    await a;
    expect(t.spawned).toHaveLength(1);
  });

  it("does not create a temp folder when the request or the environment is refused", async () => {
    const bad = setup();
    expect(await bad.run({ bytes: new Uint8Array([1]), extension: ".txt", target: "ge2" })).toMatchObject({ code: "invalid" });
    const missing = setup({ canImport: async () => false });
    expect(await missing.run()).toMatchObject({ kind: "error", code: "unavailable" });
    expect(bad.dirs).toHaveLength(0);
    expect(missing.dirs).toHaveLength(0);
  });

  it("removes the folder when writing the video fails, and when spawn throws", async () => {
    const w = setup({ writeFile: async () => Promise.reject(new Error("disk full at /tmp/secret/clip.mp4")) });
    expect(await w.run()).toMatchObject({ kind: "error", code: "stopped" });
    expect(w.removed).toEqual(w.dirs);
    const s = setup({
      spawn: () => {
        throw new Error("spawn /repo/.venv/bin/python ENOENT");
      },
    });
    expect(await s.run()).toMatchObject({ kind: "error", code: "stopped" });
    expect(s.removed).toEqual(s.dirs);
    expect(s.analyser.isBusy()).toBe(false);
  });

  it("survives a failure to remove the folder", async () => {
    const t = setup({ removeDir: async () => Promise.reject(new Error("busy")) });
    const done = t.run();
    (await t.child()).end(0);
    await expect(done).resolves.toMatchObject({ kind: "error" });
    expect(t.analyser.isBusy()).toBe(false);
  });

  it("never puts a path, the interpreter or the repository in any error message", async () => {
    const cases: (() => Promise<unknown>)[] = [
      async () => {
        const t = setup({ writeFile: async () => Promise.reject(new Error(`${REPO}/x`)) });
        return t.run();
      },
      async () => {
        const t = setup();
        const done = t.run();
        const c = await t.child();
        c.writeErr(`File "${REPO}/models/tapping/src/predict_clip.py", line 1\n`);
        c.writeOut(`failed on ${PYTHON}`);
        c.end(0);
        return done;
      },
      async () => {
        const t = setup();
        const done = t.run();
        await t.child();
        t.timers.fire();
        return done;
      },
      async () => setup({ canImport: async () => false }).run(),
    ];
    for (const make of cases) {
      const reply = (await make()) as { kind: string; message?: string };
      expect(reply.kind).toBe("error");
      expect(reply.message).not.toMatch(/\/repo|\.venv|\/tmp|predict_clip|python|[\\/]/i);
    }
  });
});

// ---------------------------------------------------------------------------- the checked result

describe("validateTappingResult", () => {
  it("accepts a scored and a failed result", () => {
    expect(validateTappingResult(scoredFixture())).not.toBeNull();
    expect(validateTappingResult(failedFixture())).not.toBeNull();
  });

  it("accepts null for a number that was not available, never as zero", () => {
    const r = scoredFixture();
    r.score.cv_auroc = null;
    r.contributions[0]!.shap = null;
    r.curve.amps[1] = null;
    expect(validateTappingResult(r)).not.toBeNull();
  });

  it("rejects every kind of wrong shape", () => {
    // `any` on purpose: this helper edits a parsed copy of the fixture to build broken input.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const mutate = (edit: (r: Record<string, any>) => void): unknown => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const r = JSON.parse(JSON.stringify(scoredFixture())) as Record<string, any>;
      edit(r);
      return r;
    };
    const wrong: unknown[] = [
      null,
      [],
      "x",
      mutate((r) => (r["schema_version"] = "tapping-result/2")),
      mutate((r) => (r["status"] = "ok")),
      mutate((r) => (r["clip_ref"] = "F0E37B01")),
      mutate((r) => (r["clip_ref"] = "/Users/someone/clip.mp4")),
      mutate((r) => (r["target"] = "ge3")),
      mutate((r) => delete r["pdf_base64"]),
      mutate((r) => (r["pdf_base64"] = "not base64!")),
      mutate((r) => (r["score"]["value"] = 1.5)),
      mutate((r) => (r["score"]["value"] = null)),
      mutate((r) => (r["score"]["band_low"] = 0.9)),
      mutate((r) => (r["score"]["above_threshold"] = "no")),
      mutate((r) => (r["contributions"] = [])),
      mutate((r) => (r["contributions"][0]["shap"] = "0.5")),
      mutate((r) => (r["curve"]["amps"] = [1])),
      mutate((r) => (r["curve"]["line_y"] = [])),
      mutate((r) => (r["signal"]["peak_v"] = [])),
      mutate((r) => (r["signal"]["t"] = Array(601).fill(0))),
      mutate((r) => (r["preview_frames"] = Array(13).fill(r["preview_frames"][0]))),
      mutate((r) => (r["model"]["folds_sha8"] = "xyz")),
      mutate((r) => (r["recording"]["n_taps"] = 2.5)),
      { ...failedFixture(), reason: "" },
      { ...failedFixture(), clip_ref: "nothex!!" },
      { ...failedFixture(), extra: 1 },
    ];
    for (const value of wrong) expect(validateTappingResult(value)).toBeNull();
  });
});

describe("the PDF", () => {
  it("names the file from a hashed ref only", () => {
    expect(pdfFileName("f0e37b01")).toBe("summary_f0e37b01.pdf");
    for (const ref of ["../x", "f0e37b0", "F0E37B01", "f0e37b01.pdf", 5, null, "f0e37b01/../a"]) {
      expect(pdfFileName(ref)).toBeNull();
    }
  });

  it("decodes only a real PDF of a sensible size", () => {
    const pdf = Buffer.from("%PDF-1.4\nbody").toString("base64");
    expect(decodePdf(pdf)?.subarray(0, 5).toString()).toBe("%PDF-");
    for (const value of [Buffer.from("hello world").toString("base64"), "", "***", 5, null, "A".repeat(14 * 1024 * 1024 + 1)]) {
      expect(decodePdf(value)).toBeNull();
    }
  });
});
