import { useState } from "react";
import { AppShell, type NavItem } from "../components/AppShell";
import { Assessment } from "./Assessment";
import { Home } from "./Home";
import { VoiceResults } from "./VoiceResults";

// Signed-in area: Home, then the assessment workflow (Setup, Analysis, Result) from the
// approved design. Voice is not in the navigation: it is reached from the Voice part on the
// setup screen, because its result is read from a file rather than processed here.

type Screen = "home" | "assessment" | "voice";

const ICON = {
  width: 18,
  height: 18,
  viewBox: "0 0 18 18",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  "aria-hidden": true,
} as const;

const NAV: readonly NavItem<"home" | "assessment">[] = [
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
];

export function Workspace({ clinician, onSignOut }: { clinician: string; onSignOut: () => void }) {
  const [screen, setScreen] = useState<Screen>("home");
  // `openId`: an analysis Home asked to open. `session` restarts the workflow at Setup.
  const [openId, setOpenId] = useState<string | null>(null);
  const [session, setSession] = useState(0);

  function startAssessment(): void {
    setOpenId(null);
    setSession((n) => n + 1);
    setScreen("assessment");
  }

  return (
    <AppShell
      items={NAV}
      active={screen === "home" ? "home" : "assessment"}
      onNavigate={(key) => (key === "home" ? (setOpenId(null), setScreen("home")) : startAssessment())}
      clinician={clinician}
      onSignOut={onSignOut}
    >
      {screen === "home" && (
        <Home
          clinician={clinician}
          onNewAssessment={startAssessment}
          onOpenAnalysis={(id) => {
            setOpenId(id);
            setSession((n) => n + 1);
            setScreen("assessment");
          }}
          onSeeAll={startAssessment}
        />
      )}
      {/* Stays mounted while the voice file is open, so the setup form is not lost. */}
      {(screen === "assessment" || screen === "voice") && (
        <div hidden={screen === "voice"}>
          <Assessment
            key={session}
            openId={openId}
            onHome={() => {
              setOpenId(null);
              setScreen("home");
            }}
            onOpenVoice={() => setScreen("voice")}
          />
        </div>
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
