import { useState } from "react";
import "../components/GaitResultView.css";
import { saveTappingPdf } from "./tappingSource";
import type { TappingScored } from "./tappingTypes";
import {
  CHART,
  CONTRIBUTION_LINE,
  NOT_COMBINED,
  SCORE_LINE,
  VIDEO_STAYS_HERE,
  auditLine,
  bandText,
  crossValidationText,
  curveChart,
  directionWord,
  previewAlt,
  recordingRows,
  scaleDescription,
  scaleGeometry,
  scoreHeadline,
  shapBars,
  shapDescription,
  signalChart,
  thresholdText,
  type LineChart,
} from "./tappingView";
import "./TappingResult.css";

// The finger-tapping result, laid out like the gait result (same card and chart classes, same
// tokens): the score first, what moved it, then the two signal charts, with the recording and
// its checks on the side. It gives no priority and no percentage; a lower score is not a clearance.

export function TappingResultView({ result }: { result: TappingScored }) {
  return (
    <div className="gr">
      <div className="mri__layout">
        <div className="mri__main">
          <ReadingCard result={result} />
          <MovedCard result={result} />
          <SignalCard result={result} />
          <CurveCard result={result} />
          <p className="note mono">{auditLine(result)}</p>
        </div>
        <aside className="mri__rail">
          <PdfCard result={result} />
          <RecordingCard result={result} />
          <ChecksCard result={result} />
          <FramesCard result={result} />
        </aside>
      </div>
    </div>
  );
}

function ReadingCard({ result }: { result: TappingScored }) {
  const g = scaleGeometry(result);
  const cv = crossValidationText(result);
  return (
    <section className="card gr__reading-card" aria-label="Reading">
      <p className="eyebrow">Finger tapping · one part of the assessment</p>
      <h2 className="gr__reading-head gr__reading-head--clear">{scoreHeadline(result)}</h2>
      <p className="gr__reading-detail">{SCORE_LINE}</p>

      <div className="gr__bars" role="img" aria-label={scaleDescription(result)}>
        <div className="tp__scale" aria-hidden="true">
          <span className="tp__scale-track" />
          <span className="tp__scale-band" style={{ left: `${g.lowPercent}%`, width: `${Math.max(0, g.highPercent - g.lowPercent)}%` }} />
          <span className="tp__scale-threshold" style={{ left: `${g.thresholdPercent}%` }} />
          <span className="tp__scale-mark" style={{ left: `${g.valuePercent}%` }} />
        </div>
        <div className="tp__scale-labels" aria-hidden="true">
          <span>0</span>
          <span>1</span>
        </div>
        <p className="note">
          {bandText(result)} The dashed line is the fixed study threshold.
        </p>
      </div>
      <p className="gr__text">{thresholdText(result)}</p>
      {cv && <p className="gr__text">{cv}</p>}

      <div className="gr__rely">
        <p className="eyebrow">How far to lean on it</p>
        <p className="gr__text">
          The model was trained on a small research set of recordings and has not been tested on
          recordings from a clinic. Treat the score as weak evidence.
        </p>
      </div>
      <p className="note">{NOT_COMBINED}</p>
    </section>
  );
}

function MovedCard({ result }: { result: TappingScored }) {
  const bars = shapBars(result);
  const row = 34;
  const labelW = 210;
  const plotW = CHART.width - labelW - CHART.right;
  const height = bars.length * row + 12;
  return (
    <section className="card" aria-label="What moved the score">
      <h2 className="gr__h2">What moved the score</h2>
      <svg className="tp__chart" viewBox={`0 0 ${CHART.width} ${height}`} role="img" aria-label={shapDescription(bars)}>
        <line className="tp__zero" x1={labelW + plotW / 2} x2={labelW + plotW / 2} y1={2} y2={height - 2} />
        {bars.map((bar, i) => (
          <g key={bar.key}>
            <text className="tp__label" x={0} y={i * row + 14}>
              {bar.label}
            </text>
            <text className="tp__sub" x={0} y={i * row + 28}>
              {bar.valueText}
            </text>
            {bar.direction !== "none" && (
              <rect
                className={bar.direction === "up" ? "tp__bar--up" : "tp__bar--down"}
                x={labelW + (bar.x / 100) * plotW}
                y={i * row + 6}
                width={(bar.width / 100) * plotW}
                height={20}
                rx={3}
              />
            )}
          </g>
        ))}
      </svg>
      <ul className="gr__drivers">
        {bars.map((bar) => (
          <li key={bar.key} className="gr__driver">
            <span className={`gr__tag${bar.direction === "up" ? " gr__tag--towards" : " gr__tag--away"}`} aria-hidden="true" />
            <span className="gr__driver-body">
              <span className="gr__driver-tag">{directionWord(bar.direction)}</span>
              <span className="gr__driver-text">{bar.sentence}</span>
            </span>
          </li>
        ))}
      </ul>
      <p className="note">
        Solid bars pushed the score up, outlined bars pushed it down; length is drawn to scale in
        log-odds. {CONTRIBUTION_LINE}
      </p>
    </section>
  );
}

function Chart({ chart, title, xLabel, yLabel, kind }: { chart: LineChart; title: string; xLabel: string; yLabel: string; kind: "signal" | "points" }) {
  const bottom = CHART.height - CHART.bottom;
  return (
    <svg className="tp__chart" viewBox={`0 0 ${CHART.width} ${CHART.height}`} role="img" aria-label={`${title}. ${chart.description}`}>
      {chart.yTicks.map((t) => (
        <g key={`y${t.at}`}>
          <line className="tp__grid" x1={CHART.left} x2={CHART.width - CHART.right} y1={t.at} y2={t.at} />
          <text className="tp__tick" x={CHART.left - 6} y={t.at + 4} textAnchor="end">
            {t.label}
          </text>
        </g>
      ))}
      <line className="tp__axis" x1={CHART.left} x2={CHART.width - CHART.right} y1={bottom} y2={bottom} />
      {chart.xTicks.map((t) => (
        <text key={`x${t.at}`} className="tp__tick" x={t.at} y={bottom + 16} textAnchor="middle">
          {t.label}
        </text>
      ))}
      <text className="tp__axis-title" x={(CHART.left + CHART.width - CHART.right) / 2} y={CHART.height - 6} textAnchor="middle">
        {xLabel}
      </text>
      <text className="tp__axis-title" x={12} y={CHART.top + 8} textAnchor="start">
        {yLabel}
      </text>
      {kind === "signal" && <path className="tp__line" d={chart.path} />}
      {chart.line && <line className="tp__fit" x1={chart.line.x1} y1={chart.line.y1} x2={chart.line.x2} y2={chart.line.y2} />}
      {chart.markers.map((m, i) => (
        <circle key={i} className={kind === "signal" ? "tp__peak" : "tp__point"} cx={m.x} cy={m.y} r={kind === "signal" ? 4 : 3.5} />
      ))}
    </svg>
  );
}

function SignalCard({ result }: { result: TappingScored }) {
  const chart = signalChart(result);
  return (
    <section className="card" aria-label="Thumb-index distance over time">
      <h2 className="gr__h2">Thumb-index distance over time</h2>
      {chart ? (
        <Chart chart={chart} title="Thumb-index distance over time" xLabel="time (s)" yLabel="distance (hand sizes)" kind="signal" />
      ) : (
        <p className="note">The distance signal is not available for this recording.</p>
      )}
      <p className="note">Circles mark the taps that were found ({result.recording.n_taps}).</p>
    </section>
  );
}

function CurveCard({ result }: { result: TappingScored }) {
  const chart = curveChart(result);
  return (
    <section className="card" aria-label="Amplitude of each tap">
      <h2 className="gr__h2">Amplitude of each tap</h2>
      {chart ? (
        <Chart chart={chart} title="Amplitude of each tap" xLabel="time (s)" yLabel="amplitude (hand sizes)" kind="points" />
      ) : (
        <p className="note">The per-tap curve is not available for this recording.</p>
      )}
      <p className="gr__text">{result.curve.count_sentence}</p>
      <p className="gr__text">{result.curve.slope_sentence}</p>
      <p className="note">{result.curve.note}</p>
    </section>
  );
}

function PdfCard({ result }: { result: TappingScored }) {
  const [state, setState] = useState<string | null>(null);
  async function save(): Promise<void> {
    setState(null);
    const reply = await saveTappingPdf(result.clip_ref, result.pdf_base64);
    setState(reply.status === "saved" ? "Saved." : reply.status === "error" ? reply.message : null);
  }
  return (
    <section className="card" aria-label="One-page summary">
      <h2 className="mri__rail-title">One-page summary</h2>
      <div>
        <button type="button" className="btn btn--outline" onClick={() => void save()}>
          Download PDF
        </button>
      </div>
      <p className="tp__save-note" role="status">
        {state ?? "The file is named from the clip ref only."}
      </p>
    </section>
  );
}

function RecordingCard({ result }: { result: TappingScored }) {
  return (
    <section className="card" aria-label="The recording">
      <h2 className="mri__rail-title">The recording</h2>
      <dl className="tp__facts">
        {recordingRows(result).map((row) => (
          <div key={row.label} style={{ display: "contents" }}>
            <dt>{row.label}</dt>
            <dd className="mono">{row.value}</dd>
          </div>
        ))}
      </dl>
      <p className="note">{VIDEO_STAYS_HERE}</p>
    </section>
  );
}

function ChecksCard({ result }: { result: TappingScored }) {
  const { flags, notes } = result.quality;
  return (
    <section className="card" aria-label="Checks and notes">
      <h2 className="mri__rail-title">Checks and notes</h2>
      <p className="note">{flags.length === 0 ? "Automatic checks: no flags." : "Automatic checks raised:"}</p>
      {(flags.length > 0 || notes.length > 0) && (
        <ul className="tp__flags">
          {flags.map((f) => (
            <li key={f.name}>{f.text}</li>
          ))}
          {notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      )}
    </section>
  );
}

function FramesCard({ result }: { result: TappingScored }) {
  if (result.preview_frames.length === 0) return null;
  return (
    <section className="card" aria-label="Tracked frames">
      <h2 className="mri__rail-title">Frames where the hand was tracked</h2>
      <ul className="tp__frames">
        {result.preview_frames.map((f, i) => (
          <li key={i} className="tp__frame">
            <img src={`data:image/jpeg;base64,${f.jpeg_base64}`} alt={previewAlt(f.t)} />
            <span className="note mono">{f.t === null ? "" : `${f.t.toFixed(1)} s`}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
