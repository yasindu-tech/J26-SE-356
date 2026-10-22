import { describe, expect, it } from "vitest";
import * as ipc from "./tappingIpc";

const channels = Object.entries(ipc).filter(([name]) => name.startsWith("IPC_TAPPING_"));

describe("tapping IPC channels", () => {
  it("has the ten channels the main process, preload and renderer use", () => {
    expect(channels.map(([name]) => name).sort()).toEqual([
      "IPC_TAPPING_ANALYSE",
      "IPC_TAPPING_CANCEL",
      "IPC_TAPPING_OPEN_EXTERNAL",
      "IPC_TAPPING_PROGRESS",
      "IPC_TAPPING_SAVE_PDF",
      "IPC_TAPPING_SET_BOUNDS",
      "IPC_TAPPING_START",
      "IPC_TAPPING_STATUS",
      "IPC_TAPPING_STATUS_EVENT",
      "IPC_TAPPING_STOP",
    ]);
  });

  it("gives every channel a different name", () => {
    const values = channels.map(([, value]) => value);
    expect(new Set(values).size).toBe(values.length);
  });

  it("starts every channel name with 'tapping:' or 'tapping-analysis:'", () => {
    for (const [, value] of channels) expect(value).toMatch(/^tapping(-analysis)?:[a-z-]+$/);
  });

  it("keeps the channel names the earlier builds used", () => {
    expect(ipc.IPC_TAPPING_START).toBe("tapping:start");
    expect(ipc.IPC_TAPPING_STATUS_EVENT).toBe("tapping:status-changed");
    expect(ipc.IPC_TAPPING_ANALYSE).toBe("tapping-analysis:run");
    expect(ipc.IPC_TAPPING_SAVE_PDF).toBe("tapping-analysis:save-pdf");
  });
});
