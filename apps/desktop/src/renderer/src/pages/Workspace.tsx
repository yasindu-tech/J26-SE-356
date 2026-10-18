import { useState } from "react";
import { AppShell, type NavItem } from "../components/AppShell";
import { MriAnalysis } from "./MriAnalysis";
import { VoiceResults } from "./VoiceResults";

// Signed-in area. Until assessments exist (Desktop-App-Spec.md), each module has its own
// screen, reached from the left navigation of the shared frame.

type Module = "mri" | "voice";

const ICON = {
  width: 18,
  height: 18,
  viewBox: "0 0 18 18",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  "aria-hidden": true,
} as const;

const MODULES: readonly NavItem<Module>[] = [
  {
    key: "mri",
    label: "MRI analysis",
    icon: (
      <svg {...ICON}>
        <circle cx="9" cy="9" r="6.5" />
        <path d="M5.5 9.5c1-2.5 2-2.5 3-.5s2 2 3.5-1" />
      </svg>
    ),
  },
  {
    key: "voice",
    label: "Voice results",
    icon: (
      <svg {...ICON}>
        <rect x="7" y="2.5" width="4" height="8" rx="2" />
        <path d="M4 8.5a5 5 0 0 0 10 0M9 13.5v2" />
      </svg>
    ),
  },
];

export function Workspace({ clinician, onSignOut }: { clinician: string; onSignOut: () => void }) {
  const [module, setModule] = useState<Module>("mri");
  return (
    <AppShell
      items={MODULES}
      active={module}
      onNavigate={setModule}
      clinician={clinician}
      onSignOut={onSignOut}
    >
      {module === "mri" ? <MriAnalysis /> : <VoiceResults />}
    </AppShell>
  );
}
