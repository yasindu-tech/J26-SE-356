import "./PriorityRangePreview.css";

interface PriorityRow {
  label: string;
  /** Illustrative range only — not a live result. 0–1. */
  rangeStart: number;
  rangeEnd: number;
  marker: number;
}

// Static illustration of "every estimate is a range with a marker", shown on
// the sign-in screen before any real assessment exists — not a result.
const ROWS: PriorityRow[] = [
  { label: "High priority", rangeStart: 0.68, rangeEnd: 0.86, marker: 0.8 },
  { label: "Moderate", rangeStart: 0.32, rangeEnd: 0.48, marker: 0.38 },
  { label: "Low priority", rangeStart: 0.04, rangeEnd: 0.2, marker: 0.1 },
];

export function PriorityRangePreview() {
  return (
    <div className="priority-preview" aria-hidden="true">
      {ROWS.map((row) => (
        <div className="priority-preview__row" key={row.label}>
          <span className="priority-preview__label">{row.label}</span>
          <div className="priority-preview__track">
            <div
              className="priority-preview__band"
              style={{
                left: `${row.rangeStart * 100}%`,
                width: `${(row.rangeEnd - row.rangeStart) * 100}%`,
              }}
            />
            <div className="priority-preview__marker" style={{ left: `${row.marker * 100}%` }} />
          </div>
        </div>
      ))}
      <p className="priority-preview__caption">
        Every estimate is shown as a range. Low priority is a routine pathway, never &ldquo;cleared&rdquo;.
      </p>
    </div>
  );
}
