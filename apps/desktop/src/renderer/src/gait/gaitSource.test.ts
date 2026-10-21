import { describe, expect, it } from "vitest";
import { mockResultFor } from "./gaitSource";

describe("mockResultFor", () => {
  it("picks a stand-in record from the file name and carries the confirmation", () => {
    expect(mockResultFor("walk.mp4", true)?.video).toBe("w-e40cda55");
    expect(mockResultFor("short-walk.mp4", false)?.warnings.length).toBeGreaterThan(0);
    expect(mockResultFor("bad.mp4", true)?.available).toBe(false);
    expect(mockResultFor("walk.mp4", true)?.one_person_confirmed).toBe(true);
    expect(mockResultFor("walk.mp4", false)?.one_person_confirmed).toBe(false);
  });

  it("gives no record for a file that should fail", () => {
    expect(mockResultFor("fail.mp4", true)).toBeNull();
  });
});
