import { useState } from "react";
import { MriAnalysis } from "./MriAnalysis";
import { VoiceResults } from "./VoiceResults";
import "./Workspace.css";

// Signed-in area. Until assessments exist (Desktop-App-Spec.md), each module has
// its own screen; this strip switches between them.

type Module = "mri" | "voice";
const MODULES: readonly { key: Module; label: string }[] = [
  { key: "mri", label: "MRI" },
  { key: "voice", label: "Voice" },
];

export function Workspace({ onSignOut }: { onSignOut: () => void }) {
  const [module, setModule] = useState<Module>("mri");
  return (
    <div className="workspace">
      <nav className="workspace__tabs" aria-label="Screening modules">
        {MODULES.map((m) => (
          <button
            type="button"
            key={m.key}
            className={`workspace__tab${module === m.key ? " workspace__tab--on" : ""}`}
            aria-current={module === m.key}
            onClick={() => setModule(m.key)}
          >
            {m.label}
          </button>
        ))}
      </nav>
      {module === "mri" ? <MriAnalysis onSignOut={onSignOut} /> : <VoiceResults onSignOut={onSignOut} />}
    </div>
  );
}
