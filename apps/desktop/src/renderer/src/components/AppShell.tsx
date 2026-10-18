import type { ReactNode } from "react";
import logoMark from "../assets/pd-xai-logo-mark.jpeg";
import { initials } from "./initials";
import { ConnectionStatus } from "./ConnectionStatus";
import "./AppShell.css";

// The signed-in frame from the approved desktop design: a teal top bar (brand, the
// "screening aid" reminder, connection state, who is signed in) and a left navigation.
// Only places that exist are listed. The design also shows Home, Assessments, Referrals,
// Audit log and Settings; they are added here when the screens behind them exist, so
// nothing in the navigation is a dead link.

export interface NavItem<K extends string> {
  key: K;
  label: string;
  icon: ReactNode;
}

interface AppShellProps<K extends string> {
  items: readonly NavItem<K>[];
  active: K;
  onNavigate: (key: K) => void;
  /** The signed-in user name shown in the bar. */
  clinician: string;
  onSignOut: () => void;
  children: ReactNode;
}

export function AppShell<K extends string>({
  items,
  active,
  onNavigate,
  clinician,
  onSignOut,
  children,
}: AppShellProps<K>) {
  return (
    <div className="shell">
      <header className="shell__bar">
        <div className="shell__brand">
          <span className="shell__logo">
            <img src={logoMark} alt="" />
          </span>
          <span>PD-XAI</span>
        </div>
        <div className="shell__spacer" />
        <span className="shell__chip">Screening aid — does not diagnose</span>
        <ConnectionStatus onDark />
        <div className="shell__user">
          <span className="shell__avatar" aria-hidden="true">
            {initials(clinician)}
          </span>
          <span>{clinician}</span>
          <button type="button" className="shell__signout" onClick={onSignOut}>
            Sign out
          </button>
        </div>
      </header>

      <div className="shell__body">
        <nav className="shell__nav" aria-label="Main">
          {items.map((item) => (
            <button
              type="button"
              key={item.key}
              className={`shell__nav-item${item.key === active ? " shell__nav-item--on" : ""}`}
              aria-current={item.key === active ? "page" : undefined}
              onClick={() => onNavigate(item.key)}
            >
              {item.icon}
              {item.label}
            </button>
          ))}
          <p className="shell__nav-foot mono">Build {__APP_VERSION__}</p>
        </nav>
        <main className="shell__main">{children}</main>
      </div>
    </div>
  );
}
