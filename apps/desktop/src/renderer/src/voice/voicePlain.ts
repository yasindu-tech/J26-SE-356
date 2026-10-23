import {
  TRIAGE_THRESHOLD,
  type MeasuredQuality,
  type VoiceResult,
} from "./voiceResult";

// The plain-language reading of a voice result, for a general clinician (same idea as
// mri/mriPlain.ts). No log-odds and no feature codes: those stay in the technical details.
// "Not prioritised" is never presented as a clearance.

export type Tone = "attention" | "calm" | "none";

export interface PlainReading {
  tone: Tone;
  headline: string;
  detail: string;
  /** True when the 95% range crosses the referral line: the result could go either way. */
  borderline: boolean;
}

export function plainReading(result: VoiceResult): PlainReading {
  if (result.kind === "abstained") {
    return {
      tone: "none",
      headline: "Not scored",
      detail:
        "The recording could not be used, so the voice was not scored. This is not a normal result; record again using the advice below.",
      borderline: false,
    };
  }
  const borderline =
    result.ciLower < TRIAGE_THRESHOLD && result.ciUpper >= TRIAGE_THRESHOLD;
  return result.triagePriority === "prioritised_referral"
    ? {
        tone: "attention",
        headline: "Prioritised for referral",
        detail:
          "The voice leans towards the pattern the model learned from people with Parkinson's. Weigh it alongside your own assessment. It is not a diagnosis.",
        borderline,
      }
    : {
        tone: "calm",
        headline: "Not prioritised",
        detail:
          "The voice leans towards the pattern of healthy voices. This does not rule anything out: not prioritised is never a clearance.",
        borderline,
      };
}

/** How far to lean on the model, in one or two sentences. */
// The model cannot tell Parkinson's from other things that change the voice
// (models/voice/src/audio_bias.py: no dataset we hold labels them), so every score says so.
export const OTHER_CONDITIONS_LINE =
  "A cold, sore throat, asthma or another voice, throat or breathing problem can also raise the score.";

export function reliabilityLine(result: VoiceResult): string {
  const trained =
    "The model was trained on a  group of people, so treat this result with caution.";
  if (result.kind === "abstained") return trained;
  const width = result.ciUpper - result.ciLower;
  const range =
    width >= 0.5
      ? ` Its range is wide (${result.ciLower.toFixed(2)} to ${result.ciUpper.toFixed(2)}), so it is unsure about this voice.`
      : "";
  return `${trained}${range} ${OTHER_CONDITIONS_LINE}`;
}

// Plain names for the measure groups in models/voice/src/audio_features.py and uci_praat.py.
const PLAIN_GROUP: Record<string, string> = {
  mfcc: "Voice tone",
  jitter: "Pitch steadiness",
  shimmer: "Loudness steadiness",
  noise: "Voice clarity",
  pitch: "Pitch level",
  formants: "Mouth and throat resonance",
  spectrum: "Tone brightness",
  "loudness change": "Loudness changes",
};

export function plainGroup(group: string): string {
  return PLAIN_GROUP[group] ?? "Another voice measure";
}

export interface PlainDriver {
  key: string;
  text: string;
  raised: boolean;
}

/**
 * What mattered most, one sentence per measure group (several MFCC bands read as one
 * "voice tone"). Groups are ranked by their summed contribution. Not causes.
 */
export function plainDrivers(result: VoiceResult, count = 3): PlainDriver[] {
  if (result.kind !== "scored" || !result.explanations) return [];
  const reasons = result.explanations.topReasons;
  const byGroup = new Map<
    string,
    { total: number; usual: number; unusual: number; odd: number }
  >();
  for (const driver of result.shapTop5) {
    const reason = reasons.find((r) => r.feature === driver.feature);
    const group = reason?.group || "other";
    const entry = byGroup.get(group) ?? {
      total: 0,
      usual: 0,
      unusual: 0,
      odd: 0,
    };
    entry.total += driver.shapLogOdds;
    if (reason && !reason.comparable) entry.odd += 1;
    else if (
      reason &&
      (reason.percentileAmongHealthy > 90 || reason.percentileAmongHealthy < 10)
    ) {
      entry.unusual += 1;
    } else entry.usual += 1;
    byGroup.set(group, entry);
  }
  return [...byGroup.entries()]
    .sort((a, b) => Math.abs(b[1].total) - Math.abs(a[1].total))
    .slice(0, count)
    .map(([group, e]) => {
      const raised = e.total > 0;
      const context =
        e.odd > 0
          ? "it is measured differently from the training voices, so this one is less reliable"
          : e.unusual > 0
            ? "it is unusual compared with healthy voices"
            : "it is within the usual range for healthy voices";
      return {
        key: group,
        raised,
        text: `${plainGroup(group)} pushed the score ${raised ? "up" : "down"}: ${context}.`,
      };
    });
}

/** The recording check as one line, e.g. "Passed · 3.5 s of steady 'aaah' · 15 dB above noise · M4A". */
export function qualitySummary(
  passed: boolean,
  measured: MeasuredQuality,
): string {
  const parts = [passed ? "Passed" : "Not usable"];
  if (measured.vowelS !== undefined)
    parts.push(`${measured.vowelS.toFixed(1)} s of steady 'aaah'`);
  if (measured.snrDb !== undefined)
    parts.push(`${measured.snrDb.toFixed(0)} dB above background noise`);
  if (measured.sourceFormat) parts.push(measured.sourceFormat.toUpperCase());
  return parts.join(" · ");
}
