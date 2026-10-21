import { useEffect, useState } from "react";
import { ApiError } from "../api/client";
import { getMriResult, getMriStatus, uploadMriStudy } from "../api/mri.api";
import { GaitPicker } from "../components/GaitPicker";
import { GaitProgress, GaitWhatIsHappening } from "../components/GaitProgress";
import { GaitResultView } from "../components/GaitResultView";
import { MriResultView } from "../components/MriResultView";
import { InfoTip } from "../components/InfoTip";
import { ProgressPanel } from "../components/MriProgress";
import { PageHeader } from "../components/PageHeader";
import { API_BASE_URL, MRI_MAX_UPLOAD_MB } from "../constants/config";
import { GAIT_SOURCE_IS_MOCK, startGaitAnalysis } from "../gait/gaitSource";
import type { GaitResult, GaitRunStatus } from "../gait/gaitTypes";
import { gaitStateText, videoProblem, videoSummary } from "../gait/gaitView";
import type { EddyProfile, MriResult } from "../mri/mriTypes";
import {
  EDDY_PROFILE_LABELS,
  FAST_EDDY_WARNING,
  NO_COMBINED_RESULT,
  PARTS,
  PART_NOT_BUILT,
  ageFromYearOfBirth,
  isFinal,
  isValidSubjectLabel,
  parseYearOfBirth,
} from "../mri/mriView";
import "./MriAnalysis.css";
import "./Assessment.css";

// The assessment workflow from the approved design: 1 Setup, 2 Analysis, 3 Result. The
// design combines gait, tapping, voice and MRI into one result; here each part is its own
// upload, is processed on its own and gets its own result, because the step that combines
// them (fusion) is not built. MRI and gait can be uploaded in this build (one at a time for
// now); tapping and voice are shown so the whole intended workflow is visible, and are marked
// as not built.

type Screen =
  | { view: "patient" }
  | { view: "modalities" }
  | { view: "review" }
  | { view: "uploading"; fraction: number; part?: "gait" }
  | { view: "analysis"; id: string }
  | { view: "result"; id: string; result: MriResult }
  | { view: "gait-analysis"; id: string }
  | { view: "gait-result"; outcome: GaitOutcome };

/** How a gait analysis ended: a record (scored or not), or a failure with no record. */
type GaitOutcome = { kind: "record"; result: GaitResult } | { kind: "failed"; message: string };

// One screen per step, so nothing needs scrolling to find the next thing to do.
const STAGES = ["Patient details", "Modalities", "Review", "Analysis", "Result"] as const;

const STAGE_HINT: Record<Screen["view"], string> = {
  patient: "Who is this assessment for?",
  modalities: "Choose what to analyse. Each part is processed and shown separately.",
  review: "Check everything, then start.",
  uploading: "Sending the scan.",
  analysis: "Each part is processed on its own.",
  result: "Each part has its own result.",
  "gait-analysis": "Each part is processed on its own.",
  "gait-result": "Each part has its own result.",
};

function stageOf(view: Screen["view"]): number {
  switch (view) {
    case "patient":
      return 0;
    case "modalities":
      return 1;
    case "review":
    case "uploading":
      return 2;
    case "analysis":
    case "gait-analysis":
      return 3;
    case "result":
    case "gait-result":
      return 4;
  }
}

function errorText(error: unknown, fallback: string): string {
  return error instanceof ApiError ? error.message : fallback;
}

type Hand = "" | "right" | "left" | "unknown";

interface Form {
  label: string;
  yearOfBirth: string;
  sex: "" | "M" | "F";
  /** Free text, kept on this screen only. Not sent anywhere in this build. */
  reason: string;
  /** "Screening consent explained and recorded" ticked. Not stored in this build. */
  consent: boolean;
  /** Free text, kept on this screen only; no model uses it. */
  medication: string;
  hand: Hand;
  file: File | null;
  /** The walking video, and its length when the picker could read it. */
  video: File | null;
  videoSeconds: number | null;
  /** "One person is walking" ticked for this video. Saved with the result; asked again for every video. */
  onePerson: boolean;
}

const EMPTY_FORM: Form = {
  label: "",
  yearOfBirth: "",
  sex: "",
  reason: "",
  consent: false,
  medication: "",
  hand: "",
  file: null,
  video: null,
  videoSeconds: null,
  onePerson: false,
};

/** What is wrong with the patient details, or null when they are fine. */
function patientProblem(form: Form): string | null {
  const label = form.label.trim();
  if (label !== "" && !isValidSubjectLabel(label)) {
    return "The ID can use letters, digits, - and _ only (up to 32). Do not use a name.";
  }
  if (parseYearOfBirth(form.yearOfBirth, new Date()) === null) {
    return "Enter a four-digit year of birth for an adult (18 to 110 years old); the MRI part needs it.";
  }
  if (form.sex === "") return "Choose the person's sex; the MRI part needs it.";
  if (!form.consent) return "Confirm that screening consent was explained and recorded.";
  return null;
}

/** What is wrong with the chosen parts, or null when at least one valid part is chosen. */
function modalitiesProblem(form: Form): string | null {
  if (!form.file && !form.video) {
    return "Add at least one part to analyse: an MRI scan or a walking video.";
  }
  if (form.file && form.video) {
    return "MRI and gait together are not connected yet. Choose one of them for now.";
  }
  if (form.video) {
    const problem = videoProblem(form.video.name, form.video.size);
    if (problem) return problem.text;
    if (!form.onePerson) return "Tick the box to confirm that one person is walking.";
    return null;
  }
  if (!form.file) return null; // unreachable: a part is chosen
  if (!form.file.name.toLowerCase().endsWith(".zip")) {
    return "Choose a .zip file holding one person's DICOM folder.";
  }
  if (form.file.size > MRI_MAX_UPLOAD_MB * 1024 * 1024) {
    return `The zip is larger than ${MRI_MAX_UPLOAD_MB} MB.`;
  }
  return null;
}

/** `openId`: open an existing analysis (Home links to one) instead of starting at step 1. */
export function Assessment({
  openId = null,
  eddyProfile,
  confirmFast,
  clinician,
  onHome,
  onOpenVoice,
  onOpenSettings,
}: {
  openId?: string | null;
  clinician: string;
  /** From Settings: the DTI motion correction new analyses use. */
  eddyProfile: EddyProfile;
  confirmFast: boolean;
  onHome: () => void;
  onOpenVoice: () => void;
  onOpenSettings: () => void;
}) {
  const [screen, setScreen] = useState<Screen>({ view: "patient" });
  const [form, setForm] = useState<Form>(EMPTY_FORM);
  const [label, setLabel] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState(false);

  useEffect(() => {
    if (!openId) return;
    getMriStatus(openId)
      .then(async (item) => {
        setLabel(item.subject_label);
        if (isFinal(item.status)) {
          setScreen({ view: "result", id: item.id, result: await getMriResult(item.id) });
        } else {
          setScreen({ view: "analysis", id: item.id });
        }
      })
      .catch((e) => setError(errorText(e, "Could not open that analysis.")));
  }, [openId]);

  function go(next: Screen): void {
    setError(null);
    setConfirming(false);
    setScreen(next);
  }

  function next(from: "patient" | "modalities"): void {
    const problem = from === "patient" ? patientProblem(form) : modalitiesProblem(form);
    if (problem) {
      setError(problem);
    } else {
      go({ view: from === "patient" ? "modalities" : "review" });
    }
  }

  async function startGait(video: File): Promise<void> {
    const trimmed = form.label.trim();
    setError(null);
    setConfirming(false);
    setLabel(trimmed || null);
    setScreen({ view: "uploading", fraction: 0, part: "gait" });
    try {
      const created = await startGaitAnalysis(
        { file: video, onePersonConfirmed: form.onePerson },
        (fraction) => setScreen({ view: "uploading", fraction, part: "gait" }),
      );
      setScreen({ view: "gait-analysis", id: created.id });
    } catch (e) {
      setError(errorText(e, "The video could not be sent."));
      setScreen({ view: "review" });
    }
  }

  function gaitFinished(status: GaitRunStatus): void {
    setScreen({
      view: "gait-result",
      outcome:
        status.state === "failed" || !status.result
          ? { kind: "failed", message: status.error ?? "Something went wrong while reading the video." }
          : { kind: "record", result: status.result },
    });
  }

  /** Back to the video step with the video gone (or kept), and the tick cleared either way. */
  function anotherVideo(keepVideo: boolean): void {
    setForm({
      ...form,
      onePerson: false,
      ...(keepVideo ? {} : { video: null, videoSeconds: null }),
    });
    go({ view: "modalities" });
  }

  async function start(): Promise<void> {
    if (form.video && form.onePerson && !form.file) {
      await startGait(form.video);
      return;
    }
    const file = form.file;
    const year = parseYearOfBirth(form.yearOfBirth, new Date());
    if (!file || form.sex === "" || year === null) return; // already checked on the earlier steps
    if (eddyProfile === "fast" && confirmFast && !confirming) {
      setConfirming(true); // ask once; pressing the button again goes ahead
      return;
    }
    const trimmed = form.label.trim();
    setError(null);
    setConfirming(false);
    setLabel(trimmed || null);
    setScreen({ view: "uploading", fraction: 0 });
    try {
      const created = await uploadMriStudy(
        {
          file,
          age: ageFromYearOfBirth(year, new Date()),
          sex: form.sex,
          eddyProfile,
          ...(trimmed ? { subjectLabel: trimmed } : {}),
        },
        (fraction) => setScreen({ view: "uploading", fraction }),
      );
      setScreen({ view: "analysis", id: created.id });
    } catch (e) {
      setError(errorText(e, "The upload failed."));
      setScreen({ view: "review" });
    }
  }

  const stage = stageOf(screen.view);

  return (
    <div className="mri asm">
      <PageHeader
        title={label ? `Assessment ${label}` : "New assessment"}
        subtitle={
          screen.view === "uploading" && screen.part === "gait" ? "Sending the video." : STAGE_HINT[screen.view]
        }
        actions={
          <ol className="asm__stages" aria-label="Progress">
            {STAGES.map((name, i) => (
              <li
                key={name}
                className={`asm__stage${i === stage ? " asm__stage--on" : ""}${i < stage ? " asm__stage--done" : ""}`}
                aria-current={i === stage ? "step" : undefined}
              >
                <span className="asm__stage-n">{i + 1}</span>
                {name}
              </li>
            ))}
          </ol>
        }
      />

      {error && (
        <p className="mri__error" role="alert">
          {error}
        </p>
      )}

      {screen.view === "patient" && (
        <PatientStep form={form} clinician={clinician} onChange={setForm} onNext={() => next("patient")} />
      )}

      {screen.view === "modalities" && (
        <ModalitiesStep
          form={form}
          eddyProfile={eddyProfile}
          onChange={setForm}
          onOpenVoice={onOpenVoice}
          onOpenSettings={onOpenSettings}
          onBack={() => go({ view: "patient" })}
          onNext={() => next("modalities")}
        />
      )}

      {screen.view === "review" && (
        <ReviewStep
          form={form}
          eddyProfile={eddyProfile}
          confirming={confirming}
          onOpenSettings={onOpenSettings}
          onBack={() => go({ view: "modalities" })}
          onStart={() => void start()}
        />
      )}

      {screen.view === "uploading" && (
        <section className="card">
          <p className="eyebrow">Sending</p>
          <h2 className="card__title">
            {screen.part === "gait" ? "Uploading the walking video" : "Uploading the MRI scan"}
          </h2>
          <div className="mri__meter" aria-label="Upload progress">
            <div className="mri__meter-fill" style={{ width: `${screen.fraction * 100}%` }} />
          </div>
          <p className="note mono">
            {Math.round(screen.fraction * 100)}% sent
            {screen.part === "gait" && form.video ? ` · ${form.video.name}` : ""}
          </p>
        </section>
      )}

      {screen.view === "analysis" && (
        <div className="asm__stack">
          <PartsStrip mri="running" />
          <ProgressPanel
            id={screen.id}
            onFinished={(result) => setScreen({ view: "result", id: screen.id, result })}
            onError={(message) => setError(message)}
          />
        </div>
      )}

      {screen.view === "gait-analysis" && (
        <div className="asm__stack">
          <PartsStrip mri="none" gait="running" />
          <div className="mri__layout">
            <GaitProgress
              id={screen.id}
              onFinished={gaitFinished}
              onError={(message) => setError(message)}
            />
            <aside className="mri__rail">
              <GaitWhatIsHappening />
            </aside>
          </div>
        </div>
      )}

      {screen.view === "gait-result" && (
        <GaitResultPage
          outcome={screen.outcome}
          videoSeconds={form.videoSeconds}
          onAnotherVideo={() => anotherVideo(false)}
          onTryAgain={() => anotherVideo(true)}
          onHome={onHome}
          onNewAssessment={() => {
            setLabel(null);
            setForm(EMPTY_FORM);
            go({ view: "patient" });
          }}
        />
      )}

      {screen.view === "result" && (
        <div className="asm__stack">
          <PartsStrip mri={screen.result.status} />
          <p className="note asm__lead">{NO_COMBINED_RESULT}</p>
          <MriResultView
            result={screen.result}
            label={label}
            onNewAnalysis={() => {
              setLabel(null);
              setForm(EMPTY_FORM);
              go({ view: "patient" });
            }}
          />
          <section className="asm__others" aria-label="Parts that were not provided">
            {PARTS.filter((p) => p.key !== "mri").map((p) => (
              <div key={p.key} className="asm__other">
                <div className="asm__other-head">
                  <span className="asm__other-title">{p.label}</span>
                  <span className="mr__chip mr__chip--none">Not provided</span>
                </div>
                <p className="asm__dash">—</p>
                <p className="note">
                  {PART_NOT_BUILT}. Nothing is counted as zero; this part has no result.
                </p>
              </div>
            ))}
          </section>
          <div>
            <button type="button" className="btn btn--outline" onClick={onHome}>
              ← Home
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ parts strip

type MriState = "running" | "scored" | "abstained" | "failed" | "queued";

/** Where each of the four parts stands, in one row. A part that was not chosen shows "Not added". */
function PartsStrip({ mri, gait = "none" }: { mri: MriState | "none"; gait?: MriState | "none" }) {
  return (
    <ul className="asm__strip" aria-label="Parts of this assessment">
      {PARTS.map((p) => {
        const state = p.key === "mri" ? mri : p.key === "gait" ? gait : "none";
        const none = state === "none";
        return (
          <li key={p.key} className={`asm__strip-item${none ? " asm__strip-item--none" : ""}`}>
            <span className="asm__strip-name">{p.label}</span>
            <span className="note">
              {none ? (p.available ? "Not added" : PART_NOT_BUILT) : gaitStateText(state)}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

/** A gait result, as its own screen: the record (or the failure), the other parts as not provided. */
function GaitResultPage({
  outcome,
  videoSeconds,
  onAnotherVideo,
  onTryAgain,
  onHome,
  onNewAssessment,
}: {
  outcome: GaitOutcome;
  videoSeconds: number | null;
  onAnotherVideo: () => void;
  onTryAgain: () => void;
  onHome: () => void;
  onNewAssessment: () => void;
}) {
  const state =
    outcome.kind === "failed" ? "failed" : outcome.result.available ? "scored" : "abstained";
  return (
    <div className="asm__stack">
      <PartsStrip mri="none" gait={state} />
      <p className="note asm__lead">{NO_COMBINED_RESULT}</p>
      {outcome.kind === "record" ? (
        <GaitResultView
          result={outcome.result}
          videoSeconds={videoSeconds}
          demo={GAIT_SOURCE_IS_MOCK}
          onNewVideo={onAnotherVideo}
        />
      ) : (
        <section className="card" aria-label="The analysis did not finish">
          <h2 className="card__title">The analysis did not finish</h2>
          <p className="mri__error" role="alert">
            {outcome.message} No result was produced. The video was deleted from the server. Try the
            same video again, or choose another one.
          </p>
          <div className="asm__buttons">
            <button type="button" className="btn btn--primary" onClick={onTryAgain}>
              Try again
            </button>
            <button type="button" className="btn btn--outline" onClick={onAnotherVideo}>
              Choose another video
            </button>
          </div>
          <p className="note">Try again asks for the one-person confirmation once more, because it is saved per video.</p>
        </section>
      )}
      <section className="asm__others" aria-label="Parts that were not provided">
        {PARTS.filter((p) => p.key !== "gait").map((p) => (
          <div key={p.key} className="asm__other">
            <div className="asm__other-head">
              <span className="asm__other-title">{p.label}</span>
              <span className="mr__chip mr__chip--none">Not provided</span>
            </div>
            <p className="asm__dash">—</p>
            <p className="note">
              {p.available ? "Not added to this assessment" : PART_NOT_BUILT}. Nothing is counted as
              zero; this part has no result.
            </p>
          </div>
        ))}
      </section>
      <div className="asm__buttons">
        <button type="button" className="btn btn--outline" onClick={onHome}>
          ← Home
        </button>
        <button type="button" className="btn btn--outline" onClick={onNewAssessment}>
          Start a new assessment
        </button>
      </div>
    </div>
  );
}

// ----------------------------------------------------------------------- steps

function megabytes(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
}

function StepButtons({
  onBack,
  onNext,
  nextLabel,
  nextDisabled = false,
  hint,
}: {
  onBack?: () => void;
  onNext: () => void;
  nextLabel: string;
  nextDisabled?: boolean;
  /** Says why Next is off. */
  hint?: string;
}) {
  return (
    <div className="asm__buttons">
      {onBack ? (
        <button type="button" className="btn btn--outline" onClick={onBack}>
          ← Back
        </button>
      ) : (
        <span />
      )}
      <span className="asm__buttons-end">
        {hint && <span className="gr__hint">{hint}</span>}
        <button type="button" className="btn btn--primary" onClick={onNext} disabled={nextDisabled}>
          {nextLabel}
        </button>
      </span>
    </div>
  );
}

/** Step 1: who the assessment is for, as in the approved design's Patient section. */
function PatientStep({
  form,
  clinician,
  onChange,
  onNext,
}: {
  form: Form;
  clinician: string;
  onChange: (next: Form) => void;
  onNext: () => void;
}) {
  return (
    <form
      className="card asm__step"
      noValidate
      aria-label="Patient"
      onSubmit={(e) => {
        e.preventDefault();
        onNext();
      }}
    >
      <div className="mri__step-head">
        <p className="eyebrow">Step 1 of 3 · Patient</p>
        <h2 className="card__title">
          Patient details
          <InfoTip label="What happens to these details">
            In this build only the ID, age and sex go to the analysis. The MRI part takes your year of
            birth as age in years (within a year of the true age), and sex, because its scanner
            adjustment was fitted with them. The reason, medication and handedness stay on this
            screen: they are not saved or sent, and no model uses them. Do not type a name.
          </InfoTip>
        </h2>
        <p className="note">
          Pseudonymous ID only. No name or date of birth is asked for, so this screen can face the
          patient.
        </p>
      </div>

      <div className="asm__person">
        <label className="mri__field">
          <span className="mri__label">Pseudonymous ID</span>
          <input
            className="mri__input mono"
            value={form.label}
            maxLength={32}
            placeholder="e.g. P-8F31A2"
            onChange={(e) => onChange({ ...form, label: e.target.value })}
          />
        </label>
        <label className="mri__field">
          <span className="mri__label">Year of birth</span>
          <input
            className="mri__input mono"
            inputMode="numeric"
            maxLength={4}
            placeholder="e.g. 1987"
            value={form.yearOfBirth}
            onChange={(e) => onChange({ ...form, yearOfBirth: e.target.value })}
          />
        </label>
        <label className="mri__field">
          <span className="mri__label">Sex</span>
          <select
            className="mri__input"
            value={form.sex}
            onChange={(e) => onChange({ ...form, sex: e.target.value as Form["sex"] })}
          >
            <option value="">Choose…</option>
            <option value="F">Female</option>
            <option value="M">Male</option>
          </select>
        </label>
      </div>

      <dl className="asm__facts">
        <dt>Clinician</dt>
        <dd>{clinician}</dd>
      </dl>

      <label className="mri__field">
        <span className="mri__label">Reason for the assessment</span>
        <input
          className="mri__input"
          value={form.reason}
          maxLength={200}
          placeholder="e.g. slowness and right-hand tremor, 8 months"
          onChange={(e) => onChange({ ...form, reason: e.target.value })}
        />
      </label>

      <label className="asm__check">
        <input
          type="checkbox"
          checked={form.consent}
          onChange={(e) => onChange({ ...form, consent: e.target.checked })}
        />
        <span>Screening consent explained and recorded</span>
      </label>

      <label className="mri__field">
        <span className="mri__label">
          Current medication <span className="asm__muted">(affects interpretation)</span>
        </span>
        <input
          className="mri__input"
          value={form.medication}
          maxLength={200}
          placeholder="e.g. metoclopramide 10 mg"
          onChange={(e) => onChange({ ...form, medication: e.target.value })}
        />
      </label>

      <fieldset className="asm__hand">
        <legend className="mri__label">Handedness</legend>
        {(
          [
            ["right", "Right"],
            ["left", "Left"],
            ["unknown", "Unknown"],
          ] as const
        ).map(([value, text]) => (
          <label key={value}>
            <input
              type="radio"
              name="handedness"
              checked={form.hand === value}
              onChange={() => onChange({ ...form, hand: value })}
            />
            {text}
          </label>
        ))}
      </fieldset>

      <StepButtons onNext={onNext} nextLabel="Next: Modalities →" />
    </form>
  );
}

/** Step 2: which parts to analyse. MRI and gait can be uploaded in this build. */
function ModalitiesStep({
  form,
  eddyProfile,
  onChange,
  onOpenVoice,
  onOpenSettings,
  onBack,
  onNext,
}: {
  form: Form;
  eddyProfile: EddyProfile;
  onChange: (next: Form) => void;
  onOpenVoice: () => void;
  onOpenSettings: () => void;
  onBack: () => void;
  onNext: () => void;
}) {
  const [dragging, setDragging] = useState(false);
  const file = form.file;
  const setFile = (f: File | null) => onChange({ ...form, file: f });

  return (
    <section className="card asm__step asm__step--wide">
      <div className="mri__step-head">
        <p className="eyebrow">Step 2 of 3</p>
        <h2 className="card__title">
          Modalities
          <InfoTip label="How the parts are processed">{NO_COMBINED_RESULT}</InfoTip>
        </h2>
      </div>

      <div className="asm__modalities">
        <div className="asm__part asm__part--mri">
          <div className="asm__part-head">
            <h3 className="asm__part-title">MRI</h3>
            <span className="mr__chip mr__chip--on">Available</span>
          </div>
          <p className="note">{PARTS[3].provides}. Takes from minutes to over an hour.</p>
          <label
            className={`mri__drop${dragging ? " mri__drop--over" : ""}${file ? " mri__drop--has-file" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              setFile(e.dataTransfer.files[0] ?? null);
            }}
          >
            <input
              className="mri__drop-input"
              type="file"
              accept=".zip,application/zip"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
            {file ? (
              <span className="mri__drop-file">
                <span className="mono mri__drop-name">{file.name}</span>
                <span className="note mono">{megabytes(file.size)} · ready to send</span>
              </span>
            ) : (
              <span className="mri__drop-text">
                <span className="mri__drop-link">Choose a file</span> or drop it here
                <br />
                A .zip of one person&rsquo;s DICOM folder
              </span>
            )}
          </label>
          <p className="asm__setting">
            <span>
              <span className="mri__label">DTI motion correction: </span>
              {EDDY_PROFILE_LABELS[eddyProfile]}
            </span>
            <button type="button" className="asm__link" onClick={onOpenSettings}>
              Change in Settings
            </button>
          </p>
          {eddyProfile === "fast" && (
            <p className="mri__notice" role="note">
              {FAST_EDDY_WARNING}
            </p>
          )}
        </div>

        <div className="asm__part asm__part--gait">
          <div className="asm__part-head">
            <h3 className="asm__part-title">Gait</h3>
            <span className="mr__chip mr__chip--on">
              {form.video ? (form.onePerson ? "Ready to send" : "Video chosen") : "Available"}
            </span>
          </div>
          <p className="note">{PARTS[0]?.provides}. Takes about a minute.</p>
          <GaitPicker
            video={form.video}
            videoSeconds={form.videoSeconds}
            confirmed={form.onePerson}
            onVideo={(video, videoSeconds) => onChange({ ...form, video, videoSeconds })}
            onConfirmed={(onePerson) => onChange({ ...form, onePerson })}
          />
        </div>
      </div>

      <div className="asm__others-row">
        <div className="asm__others-col asm__others-col--row">
          {PARTS.filter((p) => !p.available).map((p) => (
            <div key={p.key} className="asm__part asm__part--none asm__part--small">
              <div className="asm__part-head">
                <h3 className="asm__part-title">{p.label}</h3>
                <span className="mr__chip mr__chip--none">{PART_NOT_BUILT}</span>
              </div>
              <p className="note">{p.provides}.</p>
              {p.key === "voice" && (
                <button type="button" className="asm__link" onClick={onOpenVoice}>
                  Open an existing voice result file →
                </button>
              )}
            </div>
          ))}
        </div>
      </div>

      <StepButtons
        onBack={onBack}
        onNext={onNext}
        nextLabel="Next: Review →"
        nextDisabled={form.video !== null && !form.onePerson}
        {...(form.video !== null && !form.onePerson ? { hint: "Tick the box about the walker to continue." } : {})}
      />
    </section>
  );
}

/** Step 3: a last look, then start. */
function ReviewStep({
  form,
  eddyProfile,
  confirming,
  onOpenSettings,
  onBack,
  onStart,
}: {
  form: Form;
  eddyProfile: EddyProfile;
  confirming: boolean;
  onOpenSettings: () => void;
  onBack: () => void;
  onStart: () => void;
}) {
  return (
    <section className="card asm__step">
      <div className="mri__step-head">
        <p className="eyebrow">Step 3 of 3</p>
        <h2 className="card__title">
          Review
          <InfoTip label="Where the scan goes">
            {NO_COMBINED_RESULT} {form.video ? "The video" : "The scan"} is sent to the PD-XAI backend at{" "}
            <span className="mono">{API_BASE_URL}</span>
            {form.video
              ? " only when you press Start analysis. It is deleted there when the analysis ends, and only the result is kept."
              : "; the DICOM files are deleted there when the analysis ends, and only the result is kept."}
          </InfoTip>
        </h2>
      </div>
      <dl className="asm__review">
        <div>
          <dt>Pseudonymous ID</dt>
          <dd className="mono">{form.label.trim() || "None given"}</dd>
        </div>
        <div>
          <dt>Year of birth and sex</dt>
          <dd className="mono">
            {form.yearOfBirth} · {form.sex === "F" ? "Female" : "Male"}
          </dd>
        </div>
        <div>
          <dt>Consent</dt>
          <dd>Explained and recorded (by you; not stored here)</dd>
        </div>
        {form.video && (
          <>
            <div>
              <dt>Gait video</dt>
              <dd className="mono">
                {form.video.name} · {videoSummary(form.video.name, form.video.size, form.videoSeconds)}
              </dd>
            </div>
            <div>
              <dt>Who is walking</dt>
              <dd>
                You confirmed one person, seen from the side{" "}
                <button type="button" className="asm__link" onClick={onBack}>
                  Change
                </button>
              </dd>
            </div>
          </>
        )}
        <div>
          <dt>MRI</dt>
          <dd className="mono">
            {form.file ? `${form.file.name} · ${megabytes(form.file.size)}` : "Not added"}
          </dd>
        </div>
        {form.file && (
          <div>
            <dt>DTI motion correction</dt>
            <dd>
              {EDDY_PROFILE_LABELS[eddyProfile]}{" "}
              <button type="button" className="asm__link" onClick={onOpenSettings}>
                Change in Settings
              </button>
            </dd>
          </div>
        )}
        <div>
          <dt>Tapping, voice</dt>
          <dd>{PART_NOT_BUILT}</dd>
        </div>
      </dl>
      {confirming && (
        <p className="mri__notice" role="alert">
          {FAST_EDDY_WARNING} Press &ldquo;Yes, start with Fast&rdquo; to go ahead, or go back and{" "}
          <button type="button" className="asm__link" onClick={onOpenSettings}>
            change the setting
          </button>
          .
        </p>
      )}
      <StepButtons
        onBack={onBack}
        onNext={onStart}
        nextLabel={confirming ? "Yes, start with Fast" : "Start analysis"}
      />
    </section>
  );
}
