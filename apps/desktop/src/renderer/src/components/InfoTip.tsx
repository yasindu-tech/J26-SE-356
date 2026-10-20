import { useId, useState, type ReactNode } from "react";
import "./InfoTip.css";

/** A small (i) button that shows an explanation on hover, focus or click. The text stays in
 *  the page for screen readers. Use it for "why" and "how" notes, never for a warning the
 *  clinician must see (screening-aid wording, the fast-DTI warning, reliability). */
export function InfoTip({ label, children }: { label: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return (
    <span className={`tip${open ? " tip--open" : ""}`}>
      <button
        type="button"
        className="tip__btn"
        aria-label={label}
        aria-describedby={id}
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        onBlur={() => setOpen(false)}
        onKeyDown={(e) => {
          if (e.key === "Escape") setOpen(false);
        }}
      >
        i
      </button>
      <span role="tooltip" id={id} className="tip__pop">
        {children}
      </span>
    </span>
  );
}
