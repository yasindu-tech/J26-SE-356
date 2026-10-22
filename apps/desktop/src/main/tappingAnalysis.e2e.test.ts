// The real program on a real video, through the main-process module. Skipped unless
// RUN_TAPPING_E2E=1 (it needs the repository's .venv, the research data and the hand-tracking
// model): RUN_TAPPING_E2E=1 npx vitest run src/main/tappingAnalysis.e2e
import { execFile, spawn } from "child_process";
import { existsSync } from "fs";
import { mkdtemp, readdir, readFile, rm, writeFile } from "fs/promises";
import { tmpdir } from "os";
import { join, resolve } from "path";
import { describe, expect, it } from "vitest";
import type { TappingProgress } from "../shared/tappingIpc";
import { resolvePython } from "./tapping";
import { TappingAnalyser, type AnalysisChild } from "./tappingAnalysis";

const REPO_ROOT = resolve(__dirname, "..", "..", "..", "..");
const VIDEO = join(REPO_ROOT, "data", "demo_inputs", "self_tap.mp4");
const enabled = process.env["RUN_TAPPING_E2E"] === "1";

describe.skipIf(!enabled)("finger-tapping analysis, real Python", () => {
  it("scores the demo clip, reports progress, and leaves no temp folder or process", async () => {
    const python = resolvePython({ repoRoot: REPO_ROOT, platform: process.platform });
    const made: string[] = [];
    const pids: number[] = [];
    const analyser = new TappingAnalyser({
      repoRoot: REPO_ROOT,
      python,
      spawn: (command, args, options) => {
        const child = spawn(command, args, { cwd: options.cwd, stdio: ["ignore", "pipe", "pipe"] });
        if (child.pid !== undefined) pids.push(child.pid);
        return child as AnalysisChild;
      },
      makeTempDir: async () => {
        const dir = await mkdtemp(join(tmpdir(), "pdxai-tapping-"));
        made.push(dir);
        return dir;
      },
      writeFile: (path, bytes) => writeFile(path, bytes, { mode: 0o600 }),
      removeDir: (path) => rm(path, { recursive: true, force: true }),
      exists: (path) => existsSync(path),
      canImport: (py, modules) =>
        new Promise((done) =>
          execFile(py, ["-c", modules.map((m) => `import ${m}`).join(";")], { timeout: 60_000 }, (e) => done(e === null)),
        ),
    });

    const progress: TappingProgress[] = [];
    const bytes = new Uint8Array(await readFile(VIDEO));
    const started = Date.now();
    const reply = await analyser.run({ bytes, extension: ".mp4", target: "ge2" }, (p) => progress.push(p));
    const seconds = (Date.now() - started) / 1000;
    console.log(`first run: ${seconds.toFixed(1)} s wall time`);

    expect(reply.kind).toBe("result");
    if (reply.kind !== "result" || reply.result.status !== "scored") throw new Error("not scored");
    const r = reply.result;
    expect(r.clip_ref).toMatch(/^[0-9a-f]{8}$/);
    expect(r.score.value).toBeCloseTo(0.09, 2);
    expect(r.contributions).toHaveLength(5);
    expect(r.preview_frames).toHaveLength(6);
    expect(progress.map((p) => p.stage)).toEqual([1, 2, 3, 4, 5]);

    expect(made).toHaveLength(1);
    expect(existsSync(made[0] as string)).toBe(false);
    const leftovers = (await readdir(tmpdir())).filter((n) => n.startsWith("pdxai-tapping-"));
    expect(leftovers).toEqual([]);
    for (const pid of pids) {
      expect(() => process.kill(pid, 0)).toThrow();
    }
  }, 240_000);
});
