import { useState } from "react";
import { AppShell, type NavItem } from "../components/AppShell";
import { Assessment } from "./Assessment";
import { Home } from "./Home";
import { SettingsPage } from "./Settings";
import { loadSettings, saveSettings, type Settings } from "../settings/settings";
import { VoiceResults } from "./VoiceResults";

// Signed-in area: Home, then the assessment workflow (Setup, Analysis, Result) from the
// approved design. Voice is not in the navigation: it is reached from the Voice part on the
// setup screen, because its result is read from a file rather than processed here.

type Screen = "home" | "assessment" | "voice" | "settings";
type NavKey = "home" | "assessment" | "settings";

const ICON = {
  width: 18,
  height: 18,
  viewBox: "0 0 18 18",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  "aria-hidden": true,
} as const;

const NAV: readonly NavItem<NavKey>[] = [
  {
    key: "home",
    label: "Home",
    icon: (
      <svg {...ICON}>
        <path d="M2.5 8.5 9 3l6.5 5.5M4.5 7.5V15h9V7.5" />
      </svg>
    ),
  },
  {
    key: "assessment",
    label: "New assessment",
    icon: (
      <svg {...ICON}>
        <path d="M9 3.5v11M3.5 9h11" />
      </svg>
    ),
  },
  {
    key: "settings",
    label: "Settings",
    icon: (
      <svg {...ICON}>
        <circle cx="9" cy="9" r="2.5" />
        <path d="M9 2v2M9 14v2M2 9h2M14 9h2M4 4l1.4 1.4M12.6 12.6 14 14M14 4l-1.4 1.4M5.4 12.6 4 14" />
      </svg>
    ),
  },
];

export function Workspace({ clinician, onSignOut }: { clinician: string; onSignOut: () => void }) {
  const [settings, setSettings] = useState<Settings>(loadSettings);
  const [screen, setScreen] = useState<Screen>(() =>
    settings.general.startPage === "assessment" ? "assessment" : "home",
  );
  // `openId`: an analysis Home asked to open. `session` restarts the workflow at Setup.
  const [openId, setOpenId] = useState<string | null>(null);
  const [openGaitId, setOpenGaitId] = useState<string | null>(null);
  const [session, setSession] = useState(0);
  // True once an assessment screen exists, so a visit to Settings can return to it unchanged.
  const [live, setLive] = useState(false);

  function startAssessment(): void {
    setOpenId(null);
    setOpenGaitId(null);
    setSession((n) => n + 1);
    setLive(true);
    setScreen("assessment");
  }

  function change(next: Settings): void {
    setSettings(next);
    saveSettings(next);
  }

  function navigate(key: NavKey): void {
    if (key === "home") {
      setOpenId(null);
      setOpenGaitId(null);
      setScreen("home");
    } else if (key === "settings") {
      setScreen("settings");
    } else if (live && (screen === "settings" || screen === "voice")) {
      setScreen("assessment"); // back to where you were, with the form as you left it
    } else {
      startAssessment();
    }
  }

  const active: NavKey = screen === "home" ? "home" : screen === "settings" ? "settings" : "assessment";

  return (
    <AppShell
      items={NAV}
      active={active}
      onNavigate={navigate}
      clinician={clinician}
      onSignOut={onSignOut}
    >
      {screen === "home" && (
        <Home
          clinician={clinician}
          onNewAssessment={startAssessment}
          onOpenAnalysis={(id) => {
            setOpenId(id);
            setOpenGaitId(null);
            setSession((n) => n + 1);
            setLive(true);
            setScreen("assessment");
          }}
          onOpenGait={(id) => {
            setOpenId(null);
            setOpenGaitId(id);
            setSession((n) => n + 1);
            setLive(true);
            setScreen("assessment");
          }}
          onSeeAll={startAssessment}
        />
      )}
      {/* Stays mounted while the voice file or Settings is open, so the setup form is not lost. */}
      {(screen === "assessment" || (live && (screen === "voice" || screen === "settings"))) && (
        <div hidden={screen !== "assessment"}>
          <Assessment
            key={session}
            openId={openId}
            openGaitId={openGaitId}
            clinician={clinician}
            eddyProfile={settings.mri.eddyProfile}
            confirmFast={settings.mri.confirmFast}
            onHome={() => {
              setOpenId(null);
              setOpenGaitId(null);
              setScreen("home");
            }}
            onOpenVoice={() => setScreen("voice")}
            onOpenSettings={() => setScreen("settings")}
          />
        </div>
      )}
      {screen === "settings" && (
        <SettingsPage clinician={clinician} settings={settings} onChange={change} />
      )}
      {screen === "voice" && (
        <>
          <div style={{ padding: "20px 32px 0" }}>
            <button type="button" className="btn btn--outline" onClick={() => setScreen("assessment")}>
              ← Back to the assessment
            </button>
          </div>
          <VoiceResults />
        </>
      )}
    </AppShell>
  );
}
