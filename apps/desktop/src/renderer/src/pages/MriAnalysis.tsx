import { useCallback, useEffect, useRef, useState } from "react";
import logoMark from "../assets/pd-xai-logo-mark.jpeg";
import { ApiError } from "../api/client";
import { getMriResult, getMriStatus, listMriAnalyses, uploadMriStudy } from "../api/mri.api";
import { MriResultView } from "../components/MriResultView";
import { API_BASE_URL, MRI_MAX_UPLOAD_MB, MRI_POLL_INTERVAL_MS } from "../constants/config";
import type { EddyProfile, MriAnalysisStatus, MriResult } from "../mri/mriTypes";
import { MRI_STEPS } from "../mri/mriTypes";
import {
  EDDY_PROFILE_LABELS,
  FAST_EDDY_NOTICE,
  STEP_LABELS,
  isFinal,
  progressFraction,
  statusLabel,
  stepState,
} from "../mri/mriView";
import "./MriAnalysis.css";

// MRI module screen: upload one person's DICOM study (zipped), watch it run,
// read the MRI module's output (feature vector, raw score with 95% interval,
// explanation). The scan goes to the backend only when the clinician presses
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

export function MriAnalysis({ onSignOut }: { onSignOut: () => void }) {
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

  return (
    <div className="mri">
      <header className="mri__bar">
        <div className="mri__brand">
          <img src={logoMark} alt="" className="mri__brand-icon" />
          <span className="mri__brand-text">PD-XAI</span>
          <span className="mri__bar-title">MRI screening module</span>
        </div>
        <div className="mri__actions">
          <button
            type="button"
            className="mri__button mri__button--primary"
            onClick={() => {
              setError(null);
              setScreen({ view: "new" });
            }}
            disabled={screen.view === "uploading"}
          >
            New MRI analysis
          </button>
          <button type="button" className="mri__button" onClick={onSignOut}>
            Sign out
          </button>
        </div>
      </header>

      <div className="mri__layout">
        <nav className="mri__history" aria-label="Previous MRI analyses">
          <p className="mri__history-title">Your analyses</p>
          {history.length === 0 && <p className="mri__muted">None yet.</p>}
          {history.map((item) => {
            const label = statusLabel(item.status);
            const selected = "id" in screen && screen.id === item.id;
            return (
              <button
                type="button"
                key={item.id}
                className={`mri__history-item${selected ? " mri__history-item--selected" : ""}`}
                aria-current={selected}
                onClick={() => void openAnalysis(item)}
              >
                <span>{new Date(item.created_at + "Z").toLocaleString()}</span>
                <span className={`mri__tag mri__tag--${label.tone}`}>{label.text}</span>
                {item.eddy_profile === "fast" && (
                  <span className="mri__tag mri__tag--neutral">Fast DTI</span>
                )}
              </button>
            );
          })}
        </nav>

        <main className="mri__body">
          {error && (
            <p className="mri__error" role="alert">
              {error}
            </p>
          )}
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
            <section className="mri__panel">
              <h2 className="mri__panel-title">Uploading the scan</h2>
              <div className="mri__meter" aria-label="Upload progress">
                <div className="mri__meter-fill" style={{ width: `${screen.fraction * 100}%` }} />
              </div>
              <p className="mri__muted">{Math.round(screen.fraction * 100)}% sent</p>
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
          {screen.view === "result" && <MriResultView result={screen.result} />}
        </main>
      </div>
    </div>
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

function UploadForm({ onUpload }: UploadFormProps) {
  const [file, setFile] = useState<File | null>(null);
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
    <section className="mri__panel">
      <h2 className="mri__panel-title">New MRI analysis</h2>
      <p className="mri__muted">
        Upload one person's MRI study as a zip of its DICOM folder (T1, and DTI if it was
        acquired). On a Mac: right-click the folder, then Compress. Series are recognised from
        their content, so any folder layout works.
      </p>
      <form className="mri__form" onSubmit={submit} noValidate>
        <label className="mri__field">
          <span className="mri__label">DICOM study (.zip)</span>
          <input
            type="file"
            accept=".zip,application/zip"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </label>
        <div className="mri__field-row">
          <label className="mri__field">
            <span className="mri__label">Age (years)</span>
            <input
              className="mri__input"
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
        <label className="mri__field">
          <span className="mri__label">DTI motion correction</span>
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
        {eddyProfile === "fast" && (
          <p className="mri-result__notice" role="note">
            {FAST_EDDY_NOTICE}
          </p>
        )}
        <p className="mri__muted mri__small">
          Age and sex are needed because the scanner adjustment (ComBat) was fitted with them. The
          scan is sent to the PD-XAI backend at {API_BASE_URL}; the DICOM files are deleted there
          when the analysis ends. Only the result is kept.
        </p>
        {problem && (
          <p className="mri__error" role="alert">
            {problem}
          </p>
        )}
        <button type="submit" className="mri__submit">
          Upload and analyse
        </button>
      </form>
    </section>
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
  // The parent passes new callback functions on every render; keeping them in refs
  // stops each render from restarting the polling loop.
  const finishedRef = useRef(onFinished);
  const errorRef = useRef(onError);
  finishedRef.current = onFinished;
  errorRef.current = onError;

  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    async function poll(): Promise<void> {
      try {
        const next = await getMriStatus(id);
        if (stopped) return;
        setStatus(next);
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
  return (
    <section className="mri__panel">
      <h2 className="mri__panel-title">Analysing the scan</h2>
      <p className="mri__muted">
        T1 takes a few minutes. DTI is much slower: the motion correction step (eddy) can run
        for an hour or more. You can leave this screen and come back from the list on the left.
      </p>
      {status?.eddy_profile === "fast" && (
        <p className="mri-result__notice" role="note">
          {FAST_EDDY_NOTICE}
        </p>
      )}
      <div className="mri__meter" aria-label="Analysis progress">
        <div className="mri__meter-fill" style={{ width: `${progressFraction(done) * 100}%` }} />
      </div>
      <ol className="mri__steps">
        {MRI_STEPS.map((step) => {
          const state = status
            ? stepState(step, status.status, done, status.current_step)
            : "waiting";
          return (
            <li key={step} className={`mri__step mri__step--${state}`}>
              <span className="mri__step-dot" aria-hidden="true" />
              <span>{STEP_LABELS[step]}</span>
              {state === "current" && status?.note && (
                <span className="mri__step-note">{status.note}</span>
              )}
            </li>
          );
        })}
      </ol>
    </section>
  );
}
