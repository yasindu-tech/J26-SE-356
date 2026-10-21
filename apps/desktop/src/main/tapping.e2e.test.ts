// Real Streamlit, real process, real port. Skipped unless RUN_TAPPING_E2E=1 (it needs the
// repository's .venv and the research data): RUN_TAPPING_E2E=1 npx vitest run src/main/tapping.e2e
import { execFile, spawn } from "child_process";
import { existsSync } from "fs";
import { connect } from "net";
import { resolve } from "path";
import { describe, expect, it } from "vitest";
import {
  HOST,
  TappingController,
  checkPrerequisites,
  pickFreePort,
  resolvePython,
  type ChildLike,
} from "./tapping";

const REPO_ROOT = resolve(__dirname, "..", "..", "..", "..");
const enabled = process.env["RUN_TAPPING_E2E"] === "1";

const alive = (pid: number): boolean => {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
};

const portOpen = (port: number): Promise<boolean> =>
  new Promise((done) => {
    const socket = connect({ host: HOST, port });
    socket.once("connect", () => {
      socket.destroy();
      done(true);
    });
    socket.once("error", () => done(false));
  });

describe.skipIf(!enabled)("tapping demo, real Streamlit", () => {
  it("starts on a free port, answers its health check, and stops without a leftover process", async () => {
    const python = resolvePython({ repoRoot: REPO_ROOT, platform: process.platform });
    const spawned: ChildLike[] = [];
    const statuses: string[] = [];
    const controller = new TappingController({
      repoRoot: REPO_ROOT,
      python,
      spawn: (command, args, options) => {
        const child = spawn(command, args, { cwd: options.cwd, stdio: ["ignore", "pipe", "pipe"] }) as ChildLike;
        spawned.push(child);
        return child;
      },
      pickPort: () => pickFreePort(),
      check: () =>
        checkPrerequisites(
          {
            exists: (p) => existsSync(p),
            canImportStreamlit: (py) =>
              new Promise((done) => execFile(py, ["-c", "import streamlit"], { timeout: 30_000 }, (e) => done(e === null))),
          },
          { python, repoRoot: REPO_ROOT },
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
      showView: () => undefined,
      hideView: () => undefined,
      onStatus: (s) => statuses.push(s.state),
    });

    const status = await controller.start();
    expect(status.state, JSON.stringify(status)).toBe("ready");
    const url = controller.currentUrl();
    expect(url).toMatch(/^http:\/\/127\.0\.0\.1:\d+\/$/);
    const port = Number(new URL(url as string).port);
    const pid = spawned[0]?.pid as number;
    expect(alive(pid)).toBe(true);

    const health = await fetch(`${url}_stcore/health`);
    expect(health.status).toBe(200);
    expect(await health.text()).toBe("ok");
    expect(await portOpen(port)).toBe(true);

    await controller.stop();
    expect(alive(pid)).toBe(false); // no process with that PID remains
    expect(await portOpen(port)).toBe(false); // the port is closed
    expect(statuses).toEqual(["starting", "ready"]);
  }, 150_000);
});
