import type { GaitAnalysisStatus } from "../gait/gaitTypes";
import type { MriAnalysisStatus } from "../mri/mriTypes";
import type { TappingAnalysisStatus } from "../tapping/tappingTypes";
import type { VoiceRun } from "../voice/voiceHistory";

// Home shows one row per assessment, with a cell for each part, so two or more parts run for the
// same person sit on one line. The backend has no assessment id: MRI, gait and tapping analyses
// are separate records that share only the pseudonymous ID. Analyses with the same ID, started
// within GROUP_WINDOW_MS of each other, are one assessment; a part that is already taken starts a
// new row (a second MRI for the same ID is a second assessment). An analysis with no ID is its own row.

export const GROUP_WINDOW_MS = 12 * 60 * 60 * 1000;

export interface AssessmentRow {
  key: string;
  label: string | null;
  /** When the first part of this row was started. */
  startedAt: number;
  mri: MriAnalysisStatus | null;
  gait: GaitAnalysisStatus | null;
  tapping: TappingAnalysisStatus | null;
  voice: VoiceRun | null;
}

type Part =
  | { kind: "mri"; item: MriAnalysisStatus; at: number }
  | { kind: "gait"; item: GaitAnalysisStatus; at: number }
  | { kind: "tapping"; item: TappingAnalysisStatus; at: number }
  | { kind: "voice"; item: VoiceRun; at: number };

export function groupAssessments(
  mri: readonly MriAnalysisStatus[],
  gait: readonly GaitAnalysisStatus[],
  tapping: readonly TappingAnalysisStatus[],
  voice: readonly VoiceRun[],
  parseTime: (iso: string) => number,
): AssessmentRow[] {
  const parts: Part[] = [
    ...mri.map((item) => ({ kind: "mri" as const, item, at: parseTime(item.created_at) })),
    ...gait.map((item) => ({ kind: "gait" as const, item, at: parseTime(item.created_at) })),
    ...tapping.map((item) => ({ kind: "tapping" as const, item, at: parseTime(item.created_at) })),
    ...voice.map((item) => ({ kind: "voice" as const, item, at: item.at })),
  ].sort((a, b) => b.at - a.at);

  const rows: AssessmentRow[] = [];
  const newestAt = new Map<AssessmentRow, number>();
  for (const part of parts) {
    const label = part.kind === "voice" ? part.item.label : (part.item.subject_label ?? null);
    const row =
      label === null
        ? undefined
        : rows.find(
            (r) =>
              r.label === label &&
              r[part.kind] === null &&
              Math.abs((newestAt.get(r) ?? 0) - part.at) <= GROUP_WINDOW_MS,
          );
    const target: AssessmentRow =
      row ?? {
        key: `${part.kind}-${part.item.id}`,
        label,
        startedAt: part.at,
        mri: null,
        gait: null,
        tapping: null,
        voice: null,
      };
    if (!row) {
      rows.push(target);
      newestAt.set(target, part.at);
    }
    switch (part.kind) {
      case "mri":
        target.mri = part.item;
        break;
      case "gait":
        target.gait = part.item;
        break;
      case "tapping":
        target.tapping = part.item;
        break;
      case "voice":
        target.voice = part.item;
        break;
      default: {
        const exhaustive: never = part;
        return exhaustive;
      }
    }
    target.startedAt = Math.min(target.startedAt, part.at);
  }
  return rows.sort((a, b) => b.startedAt - a.startedAt);
}

/** The analyses of one assessment to open (any of them may be null). Voice is the id of a run
 *  kept in this session's memory: the backend stores no voice result. */
export interface OpenAssessment {
  label: string | null;
  mri: string | null;
  gait: string | null;
  tapping: string | null;
  voice: number | null;
}

/** What "Open" on a row opens: every part of that assessment. */
export function openAllParts(row: AssessmentRow): OpenAssessment {
  return {
    label: row.label,
    mri: row.mri?.id ?? null,
    gait: row.gait?.id ?? null,
    tapping: row.tapping?.id ?? null,
    voice: row.voice?.id ?? null,
  };
}

const NOTHING = { mri: null, gait: null, tapping: null, voice: null } as const;

/** What a single status cell opens: that one part. */
export function onePart(
  label: string | null,
  part: "mri" | "gait" | "tapping",
  id: string,
): OpenAssessment {
  return { label, ...NOTHING, [part]: id };
}

/** The voice cell opens that one voice run (from this session's memory). */
export function oneVoice(label: string | null, runId: number): OpenAssessment {
  return { label, ...NOTHING, voice: runId };
}
