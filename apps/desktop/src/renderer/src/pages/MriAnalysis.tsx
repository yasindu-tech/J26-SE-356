import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "../api/client";
import { getMriResult, getMriStatus, listMriAnalyses, uploadMriStudy } from "../api/mri.api";
import { MriResultView } from "../components/MriResultView";
import { PageHeader } from "../components/PageHeader";
import { API_BASE_URL, MRI_MAX_UPLOAD_MB, MRI_POLL_INTERVAL_MS } from "../constants/config";
import type { EddyProfile, MriAnalysisStatus, MriResult } from "../mri/mriTypes";
import { MRI_STEPS } from "../mri/mriTypes";
import {
  EDDY_PROFILE_LABELS,
  FAST_EDDY_NOTICE,
  STEP_LABELS,
  formatClock,
  formatElapsed,
  isFinal,
  parseBackendTime,
  statusLabel,
  stepState,
  updateActivity,
  type ActivityEntry,
} from "../mri/mriView";
import "./MriAnalysis.css";

// MRI module screen, in the approved desktop design: upload one person's DICOM study
// (zipped), watch it run, read the MRI module's output (feature vector, raw score with its
// 95% interval, explanation). The scan goes to the backend only when the clinician presses
// Upload; the backend deletes the DICOM files when the analysis ends.

// What the backend answers when there is no usable session (401 after a failed
// renewal; 403 from HTTPBearer when no token is sent at all).
const SIGNED_OUT_CODES = new Set(["NOT_AUTHENTICATED", "HTTP_401", "HTTP_403"]);

type Screen =
  | { view: "new" }
  | { view: "uploading"; fraction: number }
  | { view: "tracking"; id: string }
  | { view: "result"; id: string; result: MriResult };

function errorText(error: unknown, fallback: string): string {
  return error instanceof ApiError ? error.message : fallback;
}

/** `openId`: open this analysis straight away (the home page links to one). */
export function MriAnalysis({ openId = null }: { openId?: string | null }) {
  const [screen, setScreen] = useState<Screen>({ view: "new" });
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<MriAnalysisStatus[]>([]);

  const refreshHistory = useCallback(async () => {
    try {
      setHistory((await listMriAnalyses()).items);
    } catch {
      // The list is a convenience; the main flow reports its own errors.
    }
  }, []);

  useEffect(() => {
    void refreshHistory();
  }, [refreshHistory]);

  async function openAnalysis(item: MriAnalysisStatus): Promise<void> {
    setError(null);
    if (!isFinal(item.status)) {
      setScreen({ view: "tracking", id: item.id });
      return;
    }
    try {
      setScreen({ view: "result", id: item.id, result: await getMriResult(item.id) });
    } catch (e) {
      setError(errorText(e, "Could not load that result."));
    }
  }

  useEffect(() => {
    if (!openId) return;
    getMriStatus(openId)
      .then((item) => openAnalysis(item))
      .catch((e) => setError(errorText(e, "Could not open that analysis.")));
    // Only when the requested analysis changes; openAnalysis is recreated each render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [openId]);

  const selectedId = "id" in screen ? screen.id : null;
  const showRail = screen.view !== "result";

  return (
    <div className="mri">
      <PageHeader
        title="MRI analysis"
        subtitle="One person's T1 and DTI study in, a feature vector and a scored result out."
        actions={
          screen.view === "result" ? (
            <button
              type="button"
              className="btn btn--outline"
              onClick={() => {
                setError(null);
                setScreen({ view: "new" });
              }}
            >
              ← Your analyses
            </button>
          ) : (
            <button
              type="button"
              className="btn btn--primary"
              onClick={() => {
                setError(null);
                setScreen({ view: "new" });
              }}
              disabled={screen.view === "uploading"}
            >
              + New MRI analysis
            </button>
          )
        }
      />

      {error && (
        <p className="mri__error" role="alert">
          {error}
        </p>
      )}

      <div className={showRail ? "mri__layout" : "mri__layout mri__layout--wide"}>
        <div className="mri__main">
          {screen.view === "new" && (
            <UploadForm
              onUpload={async (input) => {
                setError(null);
                setScreen({ view: "uploading", fraction: 0 });
                try {
                  const created = await uploadMriStudy(input, (fraction) =>
                    setScreen({ view: "uploading", fraction }),
                  );
                  setScreen({ view: "tracking", id: created.id });
                  void refreshHistory();
                } catch (e) {
                  setError(errorText(e, "The upload failed."));
                  setScreen({ view: "new" });
                }
              }}
            />
          )}
          {screen.view === "uploading" && (
            <section className="card">
              <p className="eyebrow">Sending</p>
              <h2 className="card__title">Uploading the scan</h2>
              <div className="mri__meter" aria-label="Upload progress">
                <div className="mri__meter-fill" style={{ width: `${screen.fraction * 100}%` }} />
              </div>
              <p className="note mono">{Math.round(screen.fraction * 100)}% sent</p>
            </section>
          )}
          {screen.view === "tracking" && (
            <ProgressPanel
              id={screen.id}
              onFinished={(result) => {
                setScreen({ view: "result", id: screen.id, result });
                void refreshHistory();
              }}
              onError={(message) => setError(message)}
            />
          )}
          {screen.view === "result" && (
            <MriResultView
              result={screen.result}
              onNewAnalysis={() => {
                setError(null);
                setScreen({ view: "new" });
              }}
            />
          )}
        </div>

        {showRail && (
          <aside className="mri__rail">
            <HistoryCard items={history} selectedId={selectedId} onOpen={(i) => void openAnalysis(i)} />
            {screen.view === "tracking" && <WhySlowCard />}
          </aside>
        )}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ history

function HistoryCard({
  items,
  selectedId,
  onOpen,
}: {
  items: MriAnalysisStatus[];
  selectedId: string | null;
  onOpen: (item: MriAnalysisStatus) => void;
}) {
  return (
    <nav className="card mri__history" aria-label="Previous MRI analyses">
      <h2 className="mri__rail-title">Your analyses</h2>
      {items.length === 0 && <p className="note">None yet.</p>}
      {items.map((item) => {
        const label = statusLabel(item.status);
        const selected = item.id === selectedId;
        return (
          <button
            type="button"
            key={item.id}
            className={`mri__history-item${selected ? " mri__history-item--selected" : ""}`}
            aria-current={selected}
            onClick={() => onOpen(item)}
          >
            <span className="mono">{parseBackendTime(item.created_at).toLocaleString()}</span>
            <span className="mri__history-tags">
              <span className={`mri__tag mri__tag--${label.tone}`}>{label.text}</span>
              {item.eddy_profile === "fast" && <span className="mri__tag">Fast DTI</span>}
            </span>
          </button>
        );
      })}
    </nav>
  );
}

function WhySlowCard() {
  return (
    <section className="card">
      <h2 className="mri__rail-title">Why does the DTI part take so long?</h2>
      <p className="mri__body-text">
        DTI holds dozens of diffusion images, and each one has to be straightened out before
        measuring. That motion correction (called eddy) is the slow step: from tens of minutes
        to over an hour on an ordinary computer. T1 takes a few minutes.
      </p>
      <p className="mri__body-text">
        Nothing is frozen. The line under the step bar changes as each stage reports in.
      </p>
    </section>
  );
}

// ------------------------------------------------------------------ upload form

interface UploadFormProps {
  onUpload: (input: {
    file: File;
    age: number;
    sex: "M" | "F";
    eddyProfile: EddyProfile;
  }) => Promise<void>;
}

function megabytes(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
}

function UploadForm({ onUpload }: UploadFormProps) {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [age, setAge] = useState("");
  const [sex, setSex] = useState<"" | "M" | "F">("");
  const [eddyProfile, setEddyProfile] = useState<EddyProfile>("full");
  const [problem, setProblem] = useState<string | null>(null);

  function submit(event: React.FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    const years = Number(age);
    if (!file || !file.name.toLowerCase().endsWith(".zip")) {
      setProblem("Choose a .zip file holding one person's DICOM folder.");
    } else if (file.size > MRI_MAX_UPLOAD_MB * 1024 * 1024) {
      setProblem(`The zip is larger than ${MRI_MAX_UPLOAD_MB} MB.`);
    } else if (!Number.isFinite(years) || years < 18 || years > 110) {
      setProblem("Enter the person's age in years (18 to 110).");
    } else if (sex === "") {
      setProblem("Choose the person's sex.");
    } else {
      setProblem(null);
      void onUpload({ file, age: years, sex, eddyProfile });
    }
  }

  return (
    <form className="card mri__form" onSubmit={submit} noValidate>
      <div className="mri__step-head">
        <p className="eyebrow">Step 1</p>
        <h2 className="card__title">The MRI study</h2>
      </div>
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
            One person&rsquo;s DICOM folder as a .zip (T1, and DTI if it was acquired)
          </span>
        )}
      </label>
      <p className="note">
        On a Mac: right-click the folder, then Compress. Series are recognised from their
        content, so any folder layout works.
      </p>

      <div className="mri__divider" />

      <div className="mri__step-head">
        <p className="eyebrow">Step 2</p>
        <h2 className="card__title">About the person</h2>
      </div>
      <div className="mri__field-row">
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
        Age and sex are needed because the scanner adjustment (ComBat) was fitted with them.
      </p>

      <div className="mri__divider" />

      <div className="mri__step-head">
        <p className="eyebrow">Step 3 · optional</p>
        <h2 className="card__title">DTI motion correction</h2>
      </div>
      <label className="mri__field">
        <span className="mri__label">Setting</span>
        <select
          className="mri__input"
          value={eddyProfile}
          onChange={(e) => setEddyProfile(e.target.value === "fast" ? "fast" : "full")}
        >
          {(Object.keys(EDDY_PROFILE_LABELS) as EddyProfile[]).map((profile) => (
            <option key={profile} value={profile}>
              {EDDY_PROFILE_LABELS[profile]}
            </option>
          ))}
        </select>
      </label>
      <p className="note">
        {eddyProfile === "full"
          ? "The standard setting: the one the model was trained with."
          : "Quicker, but it has not been checked against the standard setting."}
      </p>
      {eddyProfile === "fast" && (
        <p className="mri__notice" role="note">
          {FAST_EDDY_NOTICE}
        </p>
      )}

      {problem && (
        <p className="mri__error" role="alert">
          {problem}
        </p>
      )}

      <div className="mri__form-foot">
        <p className="note">
          The scan is sent to the PD-XAI backend at <span className="mono">{API_BASE_URL}</span>.
          The DICOM files are deleted there when the analysis ends; only the result is kept.
        </p>
        <button type="submit" className="btn btn--primary">
          Upload and analyse
        </button>
      </div>
    </form>
  );
}

// ------------------------------------------------------------------ progress

interface ProgressPanelProps {
  id: string;
  onFinished: (result: MriResult) => void;
  onError: (message: string) => void;
}

function ProgressPanel({ id, onFinished, onError }: ProgressPanelProps) {
  const [status, setStatus] = useState<MriAnalysisStatus | null>(null);
  const [activity, setActivity] = useState<ActivityEntry[]>([]);
  const [now, setNow] = useState(() => Date.now());
  const [lastContact, setLastContact] = useState<number | null>(null);
  // The parent passes new callback functions on every render; keeping them in refs
  // stops each render from restarting the polling loop.
  const finishedRef = useRef(onFinished);
  const errorRef = useRef(onError);
  finishedRef.current = onFinished;
  errorRef.current = onError;

  useEffect(() => {
    const tick = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(tick);
  }, []);

  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    async function poll(): Promise<void> {
      try {
        const next = await getMriStatus(id);
        if (stopped) return;
        setStatus(next);
        setLastContact(Date.now());
        setActivity((log) => updateActivity(log, next, new Date()));
        if (isFinal(next.status)) {
          finishedRef.current(await getMriResult(id));
          return;
        }
      } catch (e) {
        if (stopped) return;
        if (e instanceof ApiError && SIGNED_OUT_CODES.has(e.code)) {
          // No valid session: polling cannot succeed, so stop instead of retrying.
          errorRef.current(
            "You are signed out. Sign in again and open this analysis from the list; it keeps running.",
          );
          return;
        }
        errorRef.current(errorText(e, "Lost contact with the backend; still trying."));
      }
      if (!stopped) timer = setTimeout(() => void poll(), MRI_POLL_INTERVAL_MS);
    }

    void poll();
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
  }, [id]);

  const done = status?.done ?? [];
  const started = status ? parseBackendTime(status.created_at) : null;
  const elapsed = started ? (now - started.getTime()) / 1000 : 0;
  const sinceContact = lastContact === null ? null : Math.max(0, Math.round((now - lastContact) / 1000));

  return (
    <div className="mri__tracking">
      <section className="card">
        <div className="mri__tracking-head">
          <div className="mri__tracking-text">
            <h2 className="mri__tracking-title">Analysing the scan</h2>
            <p className="mri__body-text">
              This is normal and can take a while. You can carry on with the consultation; the
              result opens here when it is ready.
            </p>
          </div>
          <div className="mri__elapsed">
            <span className="mri__elapsed-value mono">{formatElapsed(elapsed)}</span>
            <span className="note">
              elapsed{started ? ` · started ${formatClock(started)}` : ""}
            </span>
          </div>
        </div>

        {status?.eddy_profile === "fast" && (
          <p className="mri__notice" role="note">
            {FAST_EDDY_NOTICE}
          </p>
        )}

        <ol className="mri__segments" aria-label="Analysis steps">
          {MRI_STEPS.map((step, i) => {
            const state = status
              ? stepState(step, status.status, done, status.current_step)
              : "waiting";
            return (
              <li key={step} className={`mri__segment mri__segment--${state}`}>
                <span className="mri__segment-bar" />
                <span className="mri__segment-label">
                  {i + 1} {STEP_LABELS[step]}
                </span>
              </li>
            );
          })}
        </ol>

        <div className="mri__now">
          <span className="mri__now-text">
            {status?.note ? status.note : status ? "Waiting for the next step…" : "Contacting the backend…"}
          </span>
          <span className="note mri__now-contact">
            <span className="mri__dot" />
            {sinceContact === null
              ? "Not heard from the analysis service yet"
              : `Last update from the analysis service ${sinceContact} s ago`}
          </span>
        </div>

        <p className="note mri__leave">
          Leaving this screen does not stop the analysis. It continues in the background and
          stays in <strong>Your analyses</strong>.
        </p>
      </section>

      <section className="card">
        <h2 className="mri__rail-title">Activity</h2>
        {activity.length === 0 ? (
          <p className="note">Nothing reported yet.</p>
        ) : (
          <ol className="mri__activity mono">
            {activity.map((entry) => (
              <li key={entry.key}>
                <span className="mri__activity-time">{formatClock(new Date(entry.at))}</span>
                <span>{entry.text}</span>
              </li>
            ))}
          </ol>
        )}
        <p className="note">
          Listed as this screen sees each report. It is not a stored log, and it starts when you
          open this screen.
        </p>
      </section>
    </div>
  );
}
