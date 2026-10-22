import { describe, expect, it } from "vitest";
import { GROUP_WINDOW_MS, groupAssessments } from "./homeRows";

const parse = (iso: string) => Date.parse(iso);
const base = { eddy_profile: "full", steps: [], done: [], current_step: null, note: null } as const;

function mri(id: string, label: string | null, at: string, status = "running") {
  return { ...base, id, subject_label: label, created_at: at, status } as never;
}
function other(id: string, label: string | null, at: string, status = "scored") {
  return { id, subject_label: label, created_at: at, status } as never;
}

describe("groupAssessments", () => {
  it("puts the parts of one person on one row", () => {
    const rows = groupAssessments(
      [mri("m1", "P-1", "2026-10-22T10:00:00Z")],
      [other("g1", "P-1", "2026-10-22T10:01:00Z")],
      [other("t1", "P-1", "2026-10-22T10:02:00Z", "running")],
      [{ id: 1, label: "P-1", at: Date.parse("2026-10-22T10:03:00Z"), status: "scored" }],
      parse,
    );
    expect(rows).toHaveLength(1);
    expect(rows[0]?.mri?.id).toBe("m1");
    expect(rows[0]?.gait?.id).toBe("g1");
    expect(rows[0]?.tapping?.status).toBe("running");
    expect(rows[0]?.voice?.status).toBe("scored");
  });

  it("keeps different people on different rows", () => {
    const rows = groupAssessments(
      [mri("m1", "P-1", "2026-10-22T10:00:00Z")],
      [other("g1", "P-2", "2026-10-22T10:01:00Z")],
      [],
      [],
      parse,
    );
    expect(rows).toHaveLength(2);
  });

  it("starts a new row when the part is already taken or the time is far apart", () => {
    const sameLabelTwoMri = groupAssessments(
      [mri("m1", "P-1", "2026-10-22T10:00:00Z"), mri("m2", "P-1", "2026-10-22T10:30:00Z")],
      [],
      [],
      [],
      parse,
    );
    expect(sameLabelTwoMri).toHaveLength(2);
    const days = (GROUP_WINDOW_MS / 3_600_000) * 2;
    const far = groupAssessments(
      [mri("m1", "P-1", "2026-10-22T10:00:00Z")],
      [other("g1", "P-1", new Date(Date.parse("2026-10-22T10:00:00Z") - days * 3_600_000).toISOString())],
      [],
      [],
      parse,
    );
    expect(far).toHaveLength(2);
  });

  it("never merges analyses without an ID", () => {
    const rows = groupAssessments(
      [mri("m1", null, "2026-10-22T10:00:00Z")],
      [other("g1", null, "2026-10-22T10:01:00Z")],
      [],
      [],
      parse,
    );
    expect(rows).toHaveLength(2);
  });

  it("lists the newest row first", () => {
    const rows = groupAssessments(
      [mri("m1", "P-1", "2026-10-21T10:00:00Z")],
      [other("g1", "P-2", "2026-10-22T10:00:00Z")],
      [],
      [],
      parse,
    );
    expect(rows.map((r) => r.label)).toEqual(["P-2", "P-1"]);
  });
});
