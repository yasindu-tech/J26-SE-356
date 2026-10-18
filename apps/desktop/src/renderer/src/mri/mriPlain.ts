import type { MriDriver, MriModelCard, MriScore } from "./mriTypes";

// The plain-language reading of an MRI result, for a general clinician. It carries no
// log-odds, no AUC and no feature names; those are in the downloadable technical report.
// It is a reading of one modality's raw score, not a priority: no High/Moderate/Low, no
// percentage, and a lower reading is never presented as a clearance.

export type Lean = "none" | "higher" | "lower";

export interface PlainReading {
  lean: Lean;
  headline: string;
  detail: string;
}

export function plainReading(score: Pick<MriScore, "interval_95">): PlainReading {
  const [low, high] = score.interval_95;
  if (low <= 0 && high >= 0) {
    return {
      lean: "none",
      headline: "No clear lean",
      detail:
        "The model cannot place this scan clearly on either side. It does not point towards Parkinson's disease, and it does not point away from it.",
    };
  }
  if (low > 0) {
    return {
      lean: "higher",
      headline: "Leans towards the higher-likelihood side",
      detail:
        "The model leans towards a Parkinson's-like pattern on this scan. Weigh it alongside the rest of your assessment. It is not a diagnosis.",
    };
  }
  return {
    lean: "lower",
    headline: "Leans towards the lower-likelihood side",
    detail:
      "The model leans away from a Parkinson's-like pattern on this scan. This does not rule anything out; a lower reading is never a clearance.",
  };
}

/** How much to lean on the model, in one or two sentences, from its own test results. */
export function reliabilityLine(card: MriModelCard | null): string {
  const cv = card?.cross_validated;
  const people = card?.training_people?.total;
  if (!cv) {
    return "How well this model performs on new people was not reported with this result, so treat the reading with caution.";
  }
  const trained = people === undefined ? "It was trained on a small group." : `It was trained on only ${people} people.`;
  return cv.auc_ci[0] <= 0.5
    ? `${trained} When tested on people it had not seen, it could not be reliably told apart from chance, so treat this reading as weak evidence.`
    : `${trained} When tested on people it had not seen, it did better than chance, but the sample is small, so treat this reading with caution.`;
}

export interface PlainDriver {
  key: string;
  text: string;
}

/** The few measurements that moved the reading most, as a sentence each. Not causes. */
export function plainDrivers(drivers: readonly MriDriver[], count = 3): PlainDriver[] {
  return drivers.slice(0, count).map((d) => ({
    key: d.feature,
    text: `${d.label} nudged the reading ${d.direction === "towards PD" ? "higher" : "lower"}.`,
  }));
}
