import { useEffect, useState } from "react";
import { ApiError } from "../api/client";
import { getMriResult, getMriStatus, uploadMriStudy } from "../api/mri.api";
import { MriResultView } from "../components/MriResultView";
import { ProgressPanel, WhySlowCard } from "../components/MriProgress";
import { PageHeader } from "../components/PageHeader";
import { API_BASE_URL, MRI_MAX_UPLOAD_MB } from "../constants/config";
import type { EddyProfile, MriResult } from "../mri/mriTypes";
import {
  EDDY_PROFILE_LABELS,
  FAST_EDDY_WARNING,
  NO_COMBINED_RESULT,
  PARTS,
  PART_NOT_BUILT,
  isFinal,
  isValidSubjectLabel,
} from "../mri/mriView";
import "./MriAnalysis.css";
import "./Assessment.css";

// The assessment workflow from the approved design: 1 Setup, 2 Analysis, 3 Result. The
// design combines gait, tapping, voice and MRI into one result; here each part is its own
// upload, is processed on its own and gets its own result, because the step that combines
// them (fusion) is not built. Only MRI can be uploaded in this build; the other three are
// shown so the whole intended workflow is visible, and are marked as not built.

type Screen =
  | { view: "setup" }
  | { view: "uploading"; fraction: number }
  | { view: "analysis"; id: string }
  | { view: "result"; id: string; result: MriResult };

const STAGES = ["Setup", "Analysis", "Result"] as const;

function stageOf(view: Screen["view"]): number {
  return view === "setup" || view === "uploading" ? 0 : view === "analysis" ? 1 : 2;
}

function errorText(error: unknown, fallback: string): string {
  return error instanceof ApiError ? error.message : fallback;
}

/** `openId`: open an existing analysis (Home links to one) instead of starting at Setup. */
export function Assessment({
  openId = null,
  eddyProfile,
  confirmFast,
  onHome,
  onOpenVoice,
  onOpenSettings,
}: {
  openId?: string | null;
  /** From Settings: the DTI motion correction new analyses use. */
  eddyProfile: EddyProfile;
  confirmFast: boolean;
  onHome: () => void;
  onOpenVoice: () => void;
  onOpenSettings: () => void;
}) {
  const [screen, setScreen] = useState<Screen>({ view: "setup" });
  const [label, setLabel] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

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

  const stage = stageOf(screen.view);

  return (
    <div className="mri asm">
      <PageHeader
        title={label ? `Assessment ${label}` : "New assessment"}
        subtitle="Upload any of the four parts. Each is processed and shown separately."
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

      {screen.view === "setup" && (
        <Setup
          eddyProfile={eddyProfile}
          confirmFast={confirmFast}
          onOpenSettings={onOpenSettings}
          onOpenVoice={onOpenVoice}
          onStart={async (input) => {
            setError(null);
            setLabel(input.subjectLabel ?? null);
            setScreen({ view: "uploading", fraction: 0 });
            try {
              const created = await uploadMriStudy(input, (fraction) =>
                setScreen({ view: "uploading", fraction }),
              );
              setScreen({ view: "analysis", id: created.id });
            } catch (e) {
              setError(errorText(e, "The upload failed."));
              setScreen({ view: "setup" });
            }
          }}
        />
      )}

      {screen.view === "uploading" && (
        <section className="card">
          <p className="eyebrow">Sending</p>
          <h2 className="card__title">Uploading the MRI scan</h2>
          <div className="mri__meter" aria-label="Upload progress">
            <div className="mri__meter-fill" style={{ width: `${screen.fraction * 100}%` }} />
          </div>
          <p className="note mono">{Math.round(screen.fraction * 100)}% sent</p>
        </section>
      )}

      {screen.view === "analysis" && (
        <div className="asm__stack">
          <PartsStrip mri="running" />
          <div className="mri__layout">
            <ProgressPanel
              id={screen.id}
              onFinished={(result) => setScreen({ view: "result", id: screen.id, result })}
              onError={(message) => setError(message)}
            />
            <aside className="mri__rail">
              <WhySlowCard />
            </aside>
          </div>
        </div>
      )}

      {screen.view === "result" && (
        <div className="asm__stack">
          <PartsStrip mri={screen.result.status} />
          <p className="note asm__lead">{NO_COMBINED_RESULT}</p>
          <MriResultView
            result={screen.result}
            onNewAnalysis={() => {
              setError(null);
              setLabel(null);
              setScreen({ view: "setup" });
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

/** Where each of the four parts stands, in one row. Only MRI can run in this build. */
function PartsStrip({ mri }: { mri: MriState }) {
  const mriText =
    mri === "scored"
      ? "Scored"
      : mri === "abstained"
        ? "Not scored"
        : mri === "failed"
          ? "Failed"
          : "Processing";
  return (
    <ul className="asm__strip" aria-label="Parts of this assessment">
      {PARTS.map((p) => (
        <li key={p.key} className={`asm__strip-item${p.available ? "" : " asm__strip-item--none"}`}>
          <span className="asm__strip-name">{p.label}</span>
          <span className="note">{p.available ? mriText : PART_NOT_BUILT}</span>
        </li>
      ))}
    </ul>
  );
}

// ------------------------------------------------------------------------ setup

interface StartInput {
  file: File;
  age: number;
  sex: "M" | "F";
  eddyProfile: EddyProfile;
  subjectLabel?: string;
}

function megabytes(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
}

function Setup({
  eddyProfile,
  confirmFast,
  onStart,
  onOpenVoice,
  onOpenSettings,
}: {
  eddyProfile: EddyProfile;
  confirmFast: boolean;
  onStart: (input: StartInput) => Promise<void>;
  onOpenVoice: () => void;
  onOpenSettings: () => void;
}) {
  const [label, setLabel] = useState("");
  const [age, setAge] = useState("");
  const [sex, setSex] = useState<"" | "M" | "F">("");
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  function submit(event: React.FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    const years = Number(age);
    const trimmed = label.trim();
    if (trimmed !== "" && !isValidSubjectLabel(trimmed)) {
      setProblem("The ID can use letters, digits, - and _ only (up to 32). Do not use a name.");
    } else if (!file) {
      setProblem("Add at least one part to analyse. In this build that is the MRI scan.");
    } else if (!file.name.toLowerCase().endsWith(".zip")) {
      setProblem("Choose a .zip file holding one person's DICOM folder.");
    } else if (file.size > MRI_MAX_UPLOAD_MB * 1024 * 1024) {
      setProblem(`The zip is larger than ${MRI_MAX_UPLOAD_MB} MB.`);
    } else if (!Number.isFinite(years) || years < 18 || years > 110) {
      setProblem("Enter the person's age in years (18 to 110); the MRI part needs it.");
    } else if (sex === "") {
      setProblem("Choose the person's sex; the MRI part needs it.");
    } else {
      setProblem(null);
      if (eddyProfile === "fast" && confirmFast && !confirming) {
        setConfirming(true); // ask once; pressing the button again goes ahead
        return;
      }
      setConfirming(false);
      void onStart({ file, age: years, sex, eddyProfile, ...(trimmed ? { subjectLabel: trimmed } : {}) });
    }
  }

  return (
    <form className="asm__stack" onSubmit={submit} noValidate>
      <section className="card">
        <div className="mri__step-head">
          <p className="eyebrow">Person</p>
          <h2 className="card__title">Who is this for</h2>
        </div>
        <div className="asm__person">
          <label className="mri__field">
            <span className="mri__label">Pseudonymous ID</span>
            <input
              className="mri__input mono"
              value={label}
              maxLength={32}
              placeholder="e.g. P-8F31A2"
              onChange={(e) => setLabel(e.target.value)}
            />
          </label>
          <label className="mri__field">
            <span className="mri__label">Age (years)</span>
            <input
              className="mri__input mono"
              type="number"
              min={18}
              max={110}
              value={age}
              onChange={(e) => setAge(e.target.value)}
            />
          </label>
          <label className="mri__field">
            <span className="mri__label">Sex</span>
            <select
              className="mri__input"
              value={sex}
              onChange={(e) => setSex(e.target.value as "" | "M" | "F")}
            >
              <option value="">Choose…</option>
              <option value="F">Female</option>
              <option value="M">Male</option>
            </select>
          </label>
        </div>
        <p className="note">
          The ID is a label so you can find this person&rsquo;s analyses again. Use no name or date of
          birth. Age and sex are needed by the MRI part, because its scanner adjustment (ComBat) was
          fitted with them.
        </p>
      </section>

      <section className="asm__parts" aria-label="The four parts">
        {PARTS.filter((p) => !p.available).map((p) => (
          <div key={p.key} className="asm__part asm__part--none">
            <div className="asm__part-head">
              <h3 className="asm__part-title">{p.label}</h3>
              <span className="mr__chip mr__chip--none">{PART_NOT_BUILT}</span>
            </div>
            <p className="note">{p.provides}.</p>
            {p.key === "voice" ? (
              <button type="button" className="asm__link" onClick={onOpenVoice}>
                Open an existing voice result file →
              </button>
            ) : (
              <p className="note">Will be processed on its own when its module is added.</p>
            )}
          </div>
        ))}

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
      </section>

      {problem && (
        <p className="mri__error" role="alert">
          {problem}
        </p>
      )}
      {confirming && (
        <p className="mri__notice" role="alert">
          The fast DTI setting is not validated (see the note on the MRI card). Press &ldquo;Yes, start with Fast&rdquo; to go ahead, or{" "}
          <button type="button" className="asm__link" onClick={onOpenSettings}>
            change the setting
          </button>
          .
        </p>
      )}

      <section className="card asm__foot">
        <p className="note">
          {NO_COMBINED_RESULT} The scan is sent to the PD-XAI backend at{" "}
          <span className="mono">{API_BASE_URL}</span>; the DICOM files are deleted there when the
          analysis ends, and only the result is kept.
        </p>
        <button type="submit" className="btn btn--primary">
          {confirming ? "Yes, start with Fast" : "Start analysis"}
        </button>
      </section>
    </form>
  );
}
