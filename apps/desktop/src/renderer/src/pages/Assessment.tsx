import { useEffect, useState } from "react";
import { ApiError } from "../api/client";
import { getMriResult, getMriStatus, uploadMriStudy } from "../api/mri.api";
import { GaitPicker } from "../components/GaitPicker";
import { GaitProgress, GaitWhatIsHappening } from "../components/GaitProgress";
import { GaitResultView } from "../components/GaitResultView";
import { VoiceAnalysisProgress } from "../components/VoiceAnalysisProgress";
import { VoicePicker } from "../components/VoicePicker";
import { VoiceResultCard } from "../components/VoiceResultCard";
import { MriResultView } from "../components/MriResultView";
import { InfoTip } from "../components/InfoTip";
import { ProgressPanel } from "../components/MriProgress";
import { PageHeader } from "../components/PageHeader";
import { API_BASE_URL, MRI_MAX_UPLOAD_MB } from "../constants/config";
import { uploadGaitVideo } from "../api/gait.api";
import { uploadVoiceRecording } from "../api/voice.api";
import { getGaitRun } from "../gait/gaitRun";
import type { GaitResult, GaitRunStatus } from "../gait/gaitTypes";
import { asSentence, gaitStateText, videoProblem, videoSummary } from "../gait/gaitView";
import type { EddyProfile, MriResult } from "../mri/mriTypes";
import { voiceFileProblem, voiceFileSummary } from "../voice/voiceFile";
import { parseVoiceResults, type VoiceResult } from "../voice/voiceResult";
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
// them (fusion) is not built. MRI, gait and voice can be uploaded in this build, alone or
// together; tapping is shown so the whole intended workflow is visible, and is marked as not
// built. Voice differs in one way: the backend answers with the result in a few seconds and
// keeps neither the recording nor the result, so a voice part has no id and no Home row.

type Screen =
  | { view: "patient" }
  | { view: "modalities" }
  | { view: "review" }
  | { view: "uploading"; fraction: number; part?: "gait" }
  | { view: "analysis"; id: string }
  | { view: "result"; id: string; result: MriResult }
  | { view: "gait-analysis"; id: string }
  | { view: "parts" }
  | { view: "gait-result"; outcome: GaitOutcome }
  | { view: "voice-analysis"; fileName: string; uploaded: number; since: number | null }
  | { view: "voice-result"; outcome: VoiceOutcome };

/** How a gait analysis ended: a record (scored or not), or a failure with no record. */
type GaitOutcome = { kind: "record"; id: string; result: GaitResult } | { kind: "failed"; message: string };

/** How a voice analysis ended: a result (scored or not), or a failure with no result. */
type VoiceOutcome = { kind: "result"; result: VoiceResult } | { kind: "failed"; message: string };

/** The parts of one assessment that were started, and how each ended once it has. */
interface PartsState {
  mri: { id: string; result: MriResult | null } | null;
  gait: { id: string; outcome: GaitOutcome | null } | null;
  /** Voice ends before the next part starts, so it always has its outcome. */
  voice: VoiceOutcome | null;
}

/** How many parts this assessment started. */
function partCount(parts: PartsState | null): number {
  return parts ? [parts.mri, parts.gait, parts.voice].filter(Boolean).length : 0;
}

function outcomeOf(status: GaitRunStatus): GaitOutcome {
  return status.state === "failed" || !status.result
    ? { kind: "failed", message: status.error ?? "Something went wrong while reading the video." }
    : { kind: "record", id: status.id, result: status.result };
}

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
  parts: "Each part is processed on its own.",
  "gait-result": "Each part has its own result.",
  "voice-analysis": "Each part is processed on its own.",
  "voice-result": "Each part has its own result.",
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
    case "voice-analysis":
    case "parts":
      return 3;
    case "result":
    case "gait-result":
    case "voice-result":
      return 4;
  }
}

function errorText(error: unknown, fallback: string): string {
  return error instanceof ApiError ? error.message : fallback;
}

type Handedness = "" | "right" | "left" | "unknown";

/** Why the person is being assessed. Kept on this screen only; not sent, no model uses it. */
const REASONS = [
  ["tremor", "Tremor"],
  ["slowness", "Slowness or stiffness"],
  ["walking", "Walking or balance changes"],
  ["early", "Smell loss or sleep changes"],
  ["family", "Family history"],
  ["routine", "Routine screening"],
  ["other", "Other"],
] as const;
type Reason = "" | (typeof REASONS)[number][0];

interface Form {
  label: string;
  yearOfBirth: string;
  sex: "" | "M" | "F";
  reason: Reason;
  /** Free text when the reason is "Other". Kept on this screen only. */
  reasonOther: string;
  /** "Screening consent explained and recorded" ticked. Not stored in this build. */
  consent: boolean;
  /** Free text, kept on this screen only; no model uses it. */
  medication: string;
  handedness: Handedness;
  file: File | null;
  /** The walking video, and its length when the picker could read it. */
  video: File | null;
  videoSeconds: number | null;
  /** "One person is walking" ticked for this video. Saved with the result; asked again for every video. */
  onePerson: boolean;
  /** The recording of a steady "aaah". */
  voice: File | null;
}

const EMPTY_FORM: Form = {
  label: "",
  yearOfBirth: "",
  sex: "",
  reason: "",
  reasonOther: "",
  consent: false,
  medication: "",
  handedness: "",
  file: null,
  video: null,
  videoSeconds: null,
  onePerson: false,
  voice: null,
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
  if (!form.file && !form.video && !form.voice) {
    return "Add at least one part to analyse: an MRI scan, a walking video or a voice recording.";
  }
  if (form.voice) {
    const problem = voiceFileProblem(form.voice.name, form.voice.size);
    if (problem) return `${problem.title}. ${problem.text}`;
  }
  if (form.video) {
    const problem = videoProblem(form.video.name, form.video.size);
    if (problem) return problem.text;
    if (!form.onePerson) return "Tick the box to confirm that one person is walking.";
  }
  if (form.file) {
    if (!form.file.name.toLowerCase().endsWith(".zip")) {
      return "Choose a .zip file holding one person's DICOM folder.";
    }
    if (form.file.size > MRI_MAX_UPLOAD_MB * 1024 * 1024) {
      return `The zip is larger than ${MRI_MAX_UPLOAD_MB} MB.`;
    }
  }
  return null;
}

/** `openId`: open an existing analysis (Home links to one) instead of starting at step 1. */
export function Assessment({
  openId = null,
  openGaitId = null,
  eddyProfile,
  confirmFast,
  clinician,
  onHome,
  onOpenVoice,
  onOpenTapping,
  onOpenSettings,
}: {
  openId?: string | null;
  /** A gait analysis Home asked to open. */
  openGaitId?: string | null;
  clinician: string;
  /** From Settings: the DTI motion correction new analyses use. */
  eddyProfile: EddyProfile;
  confirmFast: boolean;
  onHome: () => void;
  onOpenVoice: () => void;
  onOpenTapping: () => void;
  onOpenSettings: () => void;
}) {
  const [screen, setScreen] = useState<Screen>({ view: "patient" });
  const [form, setForm] = useState<Form>(EMPTY_FORM);
  const [label, setLabel] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [sendFailed, setSendFailed] = useState(false);
  const [parts, setParts] = useState<PartsState | null>(null);

  useEffect(() => {
    if (!openGaitId) return;
    getGaitRun(openGaitId)
      .then((run) => {
        if (run.state === "queued" || run.state === "running") {
          setParts({ mri: null, gait: { id: openGaitId, outcome: null }, voice: null });
          setScreen({ view: "gait-analysis", id: openGaitId });
        } else {
          setScreen({ view: "gait-result", outcome: outcomeOf(run) });
        }
      })
      .catch((e) => setError(errorText(e, "Could not open that analysis.")));
  }, [openGaitId]);

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
    setSendFailed(false);
    if (next.view === "patient") setParts(null);
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

  function gaitFinished(status: GaitRunStatus): void {
    setScreen({ view: "gait-result", outcome: outcomeOf(status) });
  }

  /** Back to the video step with the video gone (or kept), and the tick cleared either way. */
  function anotherVideo(keepVideo: boolean): void {
    // With MRI already sent, only the video goes again: the scan is dropped from the form so it
    // is not uploaded a second time, and the running MRI part stays on the parts screen.
    const mriSent = Boolean(parts?.mri);
    setForm({
      ...form,
      onePerson: false,
      ...(keepVideo ? {} : { video: null, videoSeconds: null }),
      ...(mriSent ? { file: null } : {}),
      ...(parts?.voice ? { voice: null } : {}),
    });
    go({ view: "modalities" });
    if (mriSent) setError("The MRI part is already running, so only the video will be sent again.");
  }

  /** Back to the voice step for another recording; parts already sent are not sent again. */
  function anotherRecording(): void {
    setForm({
      ...form,
      voice: null,
      ...(parts?.mri ? { file: null } : {}),
      ...(parts?.gait ? { video: null, videoSeconds: null, onePerson: false } : {}),
    });
    go({ view: "modalities" });
    if (parts?.mri || parts?.gait) {
      setError("The other parts were already sent, so only the new recording will be sent.");
    }
  }

  /** Upload and analyse the recording. Never throws: a failure becomes the part's outcome. */
  async function analyseVoice(recording: File): Promise<VoiceOutcome> {
    setScreen({ view: "voice-analysis", fileName: recording.name, uploaded: 0, since: null });
    try {
      const reply = await uploadVoiceRecording(recording, (fraction) =>
        setScreen((s) =>
          s.view === "voice-analysis" && s.since === null
            ? { ...s, uploaded: fraction, since: fraction >= 1 ? Date.now() : null }
            : s,
        ),
      );
      const parsed = parseVoiceResults(reply);
      if (!parsed.ok) return { kind: "failed", message: parsed.message };
      const [result] = parsed.results;
      return result ? { kind: "result", result } : { kind: "failed", message: "The backend sent no result." };
    } catch (e) {
      return { kind: "failed", message: errorText(e, "Could not reach the backend to analyse the recording.") };
    }
  }

  /** The analyses that started: one part goes to its own screen, two or more to the parts screen. */
  function enter(gaitId: string | null, mriId: string | null, voiceOutcome: VoiceOutcome | null = null): void {
    // Parts sent earlier in this assessment (when one part was sent again) are kept.
    const mri = mriId ? { id: mriId, result: null } : (parts?.mri ?? null);
    const gait = gaitId ? { id: gaitId, outcome: null } : voiceOutcome ? (parts?.gait ?? null) : null;
    const voice = voiceOutcome ?? parts?.voice ?? null;
    const next: PartsState = { mri, gait, voice };
    setParts(next);
    if (partCount(next) > 1) setScreen({ view: "parts" });
    else if (gaitId) setScreen({ view: "gait-analysis", id: gaitId });
    else if (mriId) setScreen({ view: "analysis", id: mriId });
    else if (voice) setScreen({ view: "voice-result", outcome: voice });
  }

  async function start(): Promise<void> {
    const video = form.video && form.onePerson ? form.video : null;
    const file = form.file;
    const recording = form.voice;
    const year = parseYearOfBirth(form.yearOfBirth, new Date());
    if (!video && !file && !recording) return; // already checked on the earlier steps
    if (file && (form.sex === "" || year === null)) return;
    if (file && eddyProfile === "fast" && confirmFast && !confirming) {
      setConfirming(true); // ask once; pressing the button again goes ahead
      return;
    }
    const trimmed = form.label.trim();
    setError(null);
    setConfirming(false);
    setSendFailed(false);
    setLabel(trimmed || null);
    let gaitId: string | null = null;
    let mriId: string | null = null;
    let voiceOutcome: VoiceOutcome | null = null;
    try {
      // Voice first: it ends in seconds, before the longer parts are sent.
      if (recording) {
        voiceOutcome = await analyseVoice(recording);
        if (voiceOutcome.kind === "failed" && !video && !file) {
          setError(voiceOutcome.message);
          setSendFailed(true);
          setScreen({ view: "review" });
          return;
        }
      }
      if (video) {
        setScreen({ view: "uploading", fraction: 0, part: "gait" });
        gaitId = (
          await uploadGaitVideo(
            { file: video, onePersonConfirmed: form.onePerson, ...(trimmed ? { subjectLabel: trimmed } : {}) },
            (fraction) => setScreen({ view: "uploading", fraction, part: "gait" }),
          )
        ).id;
      }
      if (file && year !== null && form.sex !== "") {
        setScreen({ view: "uploading", fraction: 0 });
        mriId = (
          await uploadMriStudy(
            {
              file,
              age: ageFromYearOfBirth(year, new Date()),
              sex: form.sex,
              eddyProfile,
              ...(trimmed ? { subjectLabel: trimmed } : {}),
            },
            (fraction) => setScreen({ view: "uploading", fraction }),
          )
        ).id;
      }
    } catch (e) {
      const message = errorText(e, "The upload failed.");
      if (gaitId || mriId) {
        // One part is already running: carry on with it and say what was not sent.
        enter(gaitId, mriId, voiceOutcome);
        setError(`The MRI scan was not sent: ${message} The gait part is running and stays in Home.`);
      } else if (voiceOutcome) {
        enter(null, null, voiceOutcome);
        setError(`${video ? "The walking video" : "The MRI scan"} was not sent: ${message} The voice part is done.`);
      } else {
        setError(message);
        setSendFailed(true);
        setScreen({ view: "review" });
      }
      return;
    }
    enter(gaitId, mriId, voiceOutcome);
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
          onOpenTapping={onOpenTapping}
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
          retry={sendFailed}
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

      {screen.view === "parts" && parts && (
        <PartsScreen
          parts={parts}
          onMriDone={(result) =>
            setParts((p) => (p && p.mri ? { ...p, mri: { ...p.mri, result } } : p))
          }
          onGaitDone={(status) =>
            setParts((p) => (p && p.gait ? { ...p, gait: { ...p.gait, outcome: outcomeOf(status) } } : p))
          }
          onError={(message) => setError(message)}
          onOpenMri={(id, result) => setScreen({ view: "result", id, result })}
          onOpenGait={(outcome) => setScreen({ view: "gait-result", outcome })}
          onOpenVoice={(outcome) => setScreen({ view: "voice-result", outcome })}
          onHome={onHome}
        />
      )}

      {screen.view === "voice-analysis" && (
        <div className="asm__stack">
          <PartsStrip mri="none" voice="running" />
          <VoiceAnalysisProgress
            fileName={screen.fileName}
            uploaded={screen.uploaded}
            analysingSince={screen.since}
          />
        </div>
      )}

      {screen.view === "voice-result" && (
        <VoiceResultPage
          outcome={screen.outcome}
          mriState={mriStripState(parts?.mri ?? null)}
          gaitState={gaitStripState(parts?.gait ?? null)}
          onAnotherRecording={anotherRecording}
          onHome={onHome}
          {...(partCount(parts) > 1 ? { onBackToParts: () => go({ view: "parts" }) } : {})}
          onNewAssessment={() => {
            setLabel(null);
            setForm(EMPTY_FORM);
            go({ view: "patient" });
          }}
        />
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
          mriState={mriStripState(parts?.mri ?? null)}
          voiceState={voiceStripState(parts?.voice ?? null)}
          {...(partCount(parts) > 1 ? { onBackToParts: () => go({ view: "parts" }) } : {})}
          onNewAssessment={() => {
            setLabel(null);
            setForm(EMPTY_FORM);
            go({ view: "patient" });
          }}
        />
      )}

      {screen.view === "result" && (
        <div className="asm__stack">
          <PartsStrip
            mri={screen.result.status}
            gait={gaitStripState(parts?.gait ?? null)}
            voice={voiceStripState(parts?.voice ?? null)}
          />
          <p className="note asm__lead">{NO_COMBINED_RESULT}</p>
          <MriResultView result={screen.result} />
          <section className="asm__others" aria-label="Parts that were not provided">
            {PARTS.filter(
              (p) => p.key !== "mri" && !(p.key === "gait" && parts?.gait) && !(p.key === "voice" && parts?.voice),
            ).map((p) => (
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
          <div className="asm__result-actions">
            {partCount(parts) > 1 ? (
              <button type="button" className="btn btn--outline" onClick={() => go({ view: "parts" })}>
                ← Back to parts
              </button>
            ) : (
              <button type="button" className="btn btn--outline" onClick={onHome}>
                ← Home
              </button>
            )}
            <button
              type="button"
              className="btn btn--outline"
              onClick={() => {
                setLabel(null);
                setForm(EMPTY_FORM);
                go({ view: "patient" });
              }}
            >
              New assessment
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ parts screen

function mriStripState(m: PartsState["mri"]): MriState | "none" {
  if (!m) return "none";
  return m.result ? m.result.status : "running";
}

function voiceStripState(v: VoiceOutcome | null): MriState | "none" {
  if (!v) return "none";
  if (v.kind === "failed") return "failed";
  return v.result.kind === "scored" ? "scored" : "abstained";
}

function gaitStripState(g: PartsState["gait"]): MriState | "none" {
  if (!g) return "none";
  if (!g.outcome) return "running";
  if (g.outcome.kind === "failed") return "failed";
  return g.outcome.result.available ? "scored" : "abstained";
}

/** Two or more parts together: each part runs on its own and opens as its own result. */
function PartsScreen({
  parts,
  onMriDone,
  onGaitDone,
  onError,
  onOpenMri,
  onOpenGait,
  onOpenVoice,
  onHome,
}: {
  parts: PartsState;
  onMriDone: (result: MriResult) => void;
  onGaitDone: (status: GaitRunStatus) => void;
  onError: (message: string) => void;
  onOpenMri: (id: string, result: MriResult) => void;
  onOpenGait: (outcome: GaitOutcome) => void;
  onOpenVoice: (outcome: VoiceOutcome) => void;
  onHome: () => void;
}) {
  const { mri, gait, voice } = parts;
  const allDone = (!mri || mri.result) && (!gait || gait.outcome);
  const count = partCount(parts);
  return (
    <div className="asm__stack">
      <PartsStrip mri={mriStripState(mri)} gait={gaitStripState(gait)} voice={voiceStripState(voice)} />
      <p className="note asm__lead">
        {allDone
          ? `${count === 2 ? "Both" : "All"} parts are ready. Open each result on its own.`
          : `${count === 2 ? "Two" : "Three"} parts are being analysed. Each is processed on its own, and you can open one as soon as it is ready.`}
      </p>

      {voice && (
        <section className="card" aria-label="Voice part">
          <p className="eyebrow">Voice part</p>
          <h2 className="card__title">
            {voice.kind === "failed"
              ? "The voice analysis did not finish"
              : voice.result.kind === "scored"
                ? "The voice result is ready"
                : "The recording could not be scored"}
          </h2>
          <div>
            <button type="button" className="btn btn--primary" onClick={() => onOpenVoice(voice)}>
              Open the voice result
            </button>
          </div>
        </section>
      )}

      {mri &&
        (mri.result ? (
          <section className="card" aria-label="MRI part">
            <p className="eyebrow">MRI part</p>
            <h2 className="card__title">
              {mri.result.status === "scored" ? "The MRI result is ready" : "The MRI part ended without a score"}
            </h2>
            <div>
              <button
                type="button"
                className="btn btn--primary"
                onClick={() => mri.result && onOpenMri(mri.id, mri.result)}
              >
                Open the MRI result
              </button>
            </div>
          </section>
        ) : (
          <ProgressPanel id={mri.id} onFinished={onMriDone} onError={onError} />
        ))}

      {gait &&
        (gait.outcome ? (
          <section className="card" aria-label="Gait part">
            <p className="eyebrow">Gait part</p>
            <h2 className="card__title">
              {gait.outcome.kind === "failed"
                ? "The gait analysis did not finish"
                : gait.outcome.result.available
                  ? "The gait result is ready"
                  : "The video could not be scored"}
            </h2>
            <div>
              <button
                type="button"
                className="btn btn--primary"
                onClick={() => gait.outcome && onOpenGait(gait.outcome)}
              >
                Open the gait result
              </button>
            </div>
          </section>
        ) : (
          <div className="mri__layout">
            <GaitProgress id={gait.id} onFinished={onGaitDone} onError={onError} />
            <aside className="mri__rail">
              <GaitWhatIsHappening />
            </aside>
          </div>
        ))}

      <div>
        <button type="button" className="btn btn--outline" onClick={onHome}>
          ← Home
        </button>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ parts strip

type MriState = "running" | "scored" | "abstained" | "failed" | "queued";

/** Where each of the four parts stands, in one row. A part that was not chosen shows "Not added". */
function PartsStrip({
  mri,
  gait = "none",
  voice = "none",
}: {
  mri: MriState | "none";
  gait?: MriState | "none";
  voice?: MriState | "none";
}) {
  return (
    <ul className="asm__strip" aria-label="Parts of this assessment">
      {PARTS.map((p) => {
        const state = p.key === "mri" ? mri : p.key === "gait" ? gait : p.key === "voice" ? voice : "none";
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
  onBackToParts,
  onNewAssessment,
  mriState,
  voiceState = "none",
}: {
  /** Where the MRI part stands when it was sent with this video; "none" when it was not. */
  mriState: MriState | "none";
  /** How the voice part ended when it was sent with this video; "none" when it was not. */
  voiceState?: MriState | "none";
  outcome: GaitOutcome;
  videoSeconds: number | null;
  onAnotherVideo: () => void;
  onTryAgain: () => void;
  onHome: () => void;
  /** Set when MRI was analysed with it: the way back to the parts screen. */
  onBackToParts?: () => void;
  onNewAssessment: () => void;
}) {
  const state =
    outcome.kind === "failed" ? "failed" : outcome.result.available ? "scored" : "abstained";
  return (
    <div className="asm__stack">
      <PartsStrip mri={mriState} gait={state} voice={voiceState} />
      <p className="note asm__lead">{NO_COMBINED_RESULT}</p>
      {outcome.kind === "record" ? (
        <GaitResultView
          result={outcome.result}
          analysisId={outcome.id}
          videoSeconds={videoSeconds}
          onNewVideo={onAnotherVideo}
        />
      ) : (
        <section className="card" aria-label="The analysis did not finish">
          <h2 className="card__title">The analysis did not finish</h2>
          <p className="mri__error" role="alert">
            {asSentence(outcome.message)} No result was produced. The video was deleted from the server. Try the
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
        {PARTS.filter(
          (p) =>
            p.key !== "gait" &&
            !(p.key === "mri" && mriState !== "none") &&
            !(p.key === "voice" && voiceState !== "none"),
        ).map((p) => (
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
        {onBackToParts ? (
          <button type="button" className="btn btn--outline" onClick={onBackToParts}>
            ← Back to parts
          </button>
        ) : (
          <button type="button" className="btn btn--outline" onClick={onHome}>
            ← Home
          </button>
        )}
        <button type="button" className="btn btn--outline" onClick={onNewAssessment}>
          Start a new assessment
        </button>
      </div>
    </div>
  );
}

/** A voice result, as its own screen: the result (or the failure), the other parts as not provided. */
function VoiceResultPage({
  outcome,
  mriState,
  gaitState,
  onAnotherRecording,
  onHome,
  onBackToParts,
  onNewAssessment,
}: {
  outcome: VoiceOutcome;
  /** Where the other parts stand when they were sent with this recording; "none" when not. */
  mriState: MriState | "none";
  gaitState: MriState | "none";
  onAnotherRecording: () => void;
  onHome: () => void;
  /** Set when other parts were analysed with it: the way back to the parts screen. */
  onBackToParts?: () => void;
  onNewAssessment: () => void;
}) {
  return (
    <div className="asm__stack">
      <PartsStrip mri={mriState} gait={gaitState} voice={voiceStripState(outcome)} />
      <p className="note asm__lead">{NO_COMBINED_RESULT}</p>
      {outcome.kind === "result" ? (
        <VoiceResultCard result={outcome.result} />
      ) : (
        <section className="card" aria-label="The analysis did not finish">
          <h2 className="card__title">The voice analysis did not finish</h2>
          <p className="mri__error" role="alert">
            {asSentence(outcome.message)} No result was produced, and the recording was not kept.
          </p>
        </section>
      )}
      <section className="asm__others" aria-label="Parts that were not provided">
        {PARTS.filter(
          (p) =>
            p.key !== "voice" &&
            !(p.key === "mri" && mriState !== "none") &&
            !(p.key === "gait" && gaitState !== "none"),
        ).map((p) => (
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
        {onBackToParts ? (
          <button type="button" className="btn btn--outline" onClick={onBackToParts}>
            ← Back to parts
          </button>
        ) : (
          <button type="button" className="btn btn--outline" onClick={onHome}>
            ← Home
          </button>
        )}
        <span className="asm__buttons-end">
          <button type="button" className="btn btn--outline" onClick={onAnotherRecording}>
            Analyse another recording
          </button>
          <button type="button" className="btn btn--outline" onClick={onNewAssessment}>
            Start a new assessment
          </button>
        </span>
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

      <fieldset className="asm__reasons">
        <legend className="mri__label">Reason for the assessment</legend>
        <div className="asm__pills">
          {REASONS.map(([value, text]) => (
            <label key={value} className="asm__pill">
              <input
                type="radio"
                name="reason"
                value={value}
                checked={form.reason === value}
                onChange={() => onChange({ ...form, reason: value })}
              />
              <span>{text}</span>
            </label>
          ))}
        </div>
        {form.reason === "other" && (
          <input
            className="mri__input"
            aria-label="Other reason"
            value={form.reasonOther}
            maxLength={200}
            placeholder="Describe the reason"
            autoFocus
            onChange={(e) => onChange({ ...form, reasonOther: e.target.value })}
          />
        )}
      </fieldset>

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

      <fieldset className="asm__handedness">
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
              checked={form.handedness === value}
              onChange={() => onChange({ ...form, handedness: value })}
            />
            {text}
          </label>
        ))}
      </fieldset>

      <StepButtons onNext={onNext} nextLabel="Next: Modalities →" />
    </form>
  );
}

/** Step 2: which parts to analyse. MRI, gait and voice can be uploaded in this build. */
function ModalitiesStep({
  form,
  eddyProfile,
  onChange,
  onOpenVoice,
  onOpenTapping,
  onOpenSettings,
  onBack,
  onNext,
}: {
  form: Form;
  eddyProfile: EddyProfile;
  onChange: (next: Form) => void;
  onOpenVoice: () => void;
  onOpenTapping: () => void;
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
          <p className="note">
            {PARTS[0]?.provides}. It grades how the person walks (no, mild or moderate-to-severe
            impairment). It does not say whether they have Parkinson&rsquo;s disease. Takes about a minute.
          </p>
          <GaitPicker
            video={form.video}
            videoSeconds={form.videoSeconds}
            confirmed={form.onePerson}
            onVideo={(video, videoSeconds) => onChange({ ...form, video, videoSeconds })}
            onConfirmed={(onePerson) => onChange({ ...form, onePerson })}
          />
        </div>

        <div className="asm__part asm__part--voice">
          <div className="asm__part-head">
            <h3 className="asm__part-title">Voice</h3>
            <span className="mr__chip mr__chip--on">{form.voice ? "Ready to send" : "Available"}</span>
          </div>
          <p className="note">
            {PARTS.find((p) => p.key === "voice")?.provides} of a steady &ldquo;aaah&rdquo;. It gives a
            referral priority and the reasons behind it. It does not say whether the person has
            Parkinson&rsquo;s disease. Takes a few seconds.
          </p>
          <VoicePicker recording={form.voice} onRecording={(voice) => onChange({ ...form, voice })} />
          <div>
            <button type="button" className="asm__link" onClick={onOpenVoice}>
              Open an existing voice result file →
            </button>
          </div>
        </div>

        {PARTS.filter((p) => !p.available).map((p) => (
          <div key={p.key} className="asm__part asm__part--none asm__part--small">
            <div className="asm__part-head">
              <h3 className="asm__part-title">{p.label}</h3>
              <span className="mr__chip mr__chip--none">
                {p.key === "tapping" ? "Research demo" : PART_NOT_BUILT}
              </span>
            </div>
            <p className="note">{p.provides}.</p>
            {p.key === "tapping" && (
              <button type="button" className="asm__link" onClick={onOpenTapping}>
                Open the finger-tapping demo →
              </button>
            )}
          </div>
        ))}
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
  retry,
  onOpenSettings,
  onBack,
  onStart,
}: {
  form: Form;
  eddyProfile: EddyProfile;
  confirming: boolean;
  /** The last start did not get through: say "Retry". */
  retry: boolean;
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
          <InfoTip label={form.video ? "Where the video goes" : form.file ? "Where the scan goes" : "Where the recording goes"}>
            {NO_COMBINED_RESULT} {form.video ? "The video" : "The scan"} is sent to the PD-XAI backend at{" "}
            <span className="mono">{API_BASE_URL}</span>
            {form.video
              ? " only when you press Start analysis. The file is deleted there when the analysis ends; the result is kept, and a copy with the skeleton drawn on it when the walk can be read."
              : "; the DICOM files are deleted there when the analysis ends, and only the result is kept."}
            {form.voice &&
              " A voice recording is analysed in a few seconds and deleted straight after; its result is shown here and not stored."}
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
          <dt>Voice</dt>
          <dd className="mono">
            {form.voice ? `${form.voice.name} · ${voiceFileSummary(form.voice.name, form.voice.size)}` : "Not added"}
          </dd>
        </div>
        <div>
          <dt>Tapping</dt>
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
        nextLabel={confirming ? "Yes, start with Fast" : retry ? "Retry" : "Start analysis"}
      />
    </section>
  );
}
