"""Live monitoring dashboard for the MRI preprocessing pipelines.

The T1 and DTI pipelines write a progress event for every person and step
(see progress.py). This server reads those events and pushes the live state to the
browser the moment it changes (Server-Sent Events, no page refresh).

Pages:
    /       overview of both pipelines
    /t1     T1 pipeline: counts, current person and step with a live timer,
            time per step, estimated time left, and every person's row
    /dti    the same for DTI

API (used by the pages, also handy for other tools):
    GET  /api/state?pipeline=t1[&run=<id>]   current state as JSON
    GET  /api/runs?pipeline=t1               list of runs
    GET  /api/stream?pipeline=t1             live updates (Server-Sent Events)
    POST /api/events                         receive events from a pipeline on another
                                             machine (pipeline flag --events-url)

Privacy: events hold hashed IDs, step names, timings and check values only, never
images or PPMI IDs. By default the server listens on this computer only.
To accept events from elsewhere, start it with --host 0.0.0.0 --token <secret>
and pass the same token in the URL the pipeline uses.

Usage (from the repo root):
    python models/mri/src/pipeline_monitor.py              # then open http://127.0.0.1:8765
    python models/mri/src/preprocess_t1.py                 # in another terminal
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from progress import RUNS_DIR, expected_step_seconds, fold_events, latest_run, read_events

PIPELINES = ("t1", "dti")
SAFE_ID = re.compile(r"^[a-z0-9_]{1,64}$")

STEP_INFO = {
    "t1": {
        "n4": ("Even out brightness", "N4 removes the scanner coil's uneven brightness."),
        "skull": ("Remove skull", "SynthStrip keeps only the brain."),
        "align": ("Align to standard brain", "ANTs fits the brain to the MNI152 template."),
        "tissue": ("Label tissue", "FSL FAST marks grey matter, white matter and fluid."),
        "qc": ("Check", "Saves a check picture."),
    },
    "dti": {
        "select": ("Pick scan", "One b~1000 scan per person, same rule for everyone."),
        "mask": ("Find brain", "BET on the average b0 picture."),
        "eddy": ("Fix motion (eddy)", "Lines up all pictures and undoes scanner warping."),
        "tensor": ("FA and MD maps", "DIPY fits the tensor at every point."),
        "align": ("Align to standard", "ANTs fits FA to the FMRIB58 template."),
        "qc": ("Check", "Saves a check picture."),
    },
}
NEXT_STAGES = ["Extract features", "Harmonise (ComBat)", "Train model", "Explain (SHAP)"]


def state_for(pipeline: str, runs_dir: Path, run: str | None = None) -> dict:
    """Live state of one run (the latest, unless a run ID is given)."""
    path = runs_dir / f"{run}.jsonl" if run else latest_run(pipeline, runs_dir)
    # typical step times from every run of this pipeline, including the current one
    history = [read_events(p) for p in sorted(runs_dir.glob(f"{pipeline}_*.jsonl"))]
    expected = expected_step_seconds(history)
    if path is None or not path.exists():
        st = fold_events([], time.time(), expected)
    else:
        st = fold_events(read_events(path), time.time(), expected)
    st["pipeline"] = pipeline
    st["step_info"] = {k: {"label": v[0], "explain": v[1]} for k, v in STEP_INFO[pipeline].items()}
    return st


def list_runs(pipeline: str, runs_dir: Path) -> list[str]:
    return [p.stem for p in sorted(runs_dir.glob(f"{pipeline}_*.jsonl"), reverse=True)]


def valid_event(event: object) -> bool:
    """Accept only well-formed events whose IDs cannot be used as file paths."""
    if not isinstance(event, dict):
        return False
    return (
        event.get("pipeline") in PIPELINES
        and isinstance(event.get("run"), str)
        and SAFE_ID.match(event["run"]) is not None
        and event["run"].startswith(event["pipeline"] + "_")
        and isinstance(event.get("type"), str)
        and isinstance(event.get("t"), int | float)
    )


# ---------------------------------------------------------------- pages

STYLE = """
:root{--bg:#0d1117;--panel:#161b22;--card:#1c2330;--line:#2a3441;--text:#e6edf3;--muted:#8b98a5;
--green:#3fb950;--yellow:#d29922;--red:#f85149;--blue:#58a6ff;--grey:#484f58}
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,Segoe UI,Roboto,sans-serif;
background:var(--bg);color:var(--text)}a{color:var(--blue);text-decoration:none}
header{display:flex;align-items:center;gap:22px;padding:14px 28px;border-bottom:1px solid var(--line);
background:var(--panel);position:sticky;top:0;z-index:2}header .brand{font-weight:700}
header nav a{color:var(--muted);padding:6px 10px;border-radius:6px}header nav a.on{color:var(--text);
background:var(--card)}.live{margin-left:auto;font-size:12px;color:var(--muted)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--grey);margin-right:6px}
.dot.on{background:var(--green);box-shadow:0 0 8px var(--green)}
main{padding:22px 28px;max-width:1280px;margin:0 auto}h1{font-size:22px;margin:0 0 4px}
.sub{color:var(--muted);font-size:13px;margin-bottom:18px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:16px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi .v{font-size:24px;font-weight:700}.kpi .l{color:var(--muted);font-size:12px;margin-top:2px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-bottom:16px}
.panel h2{font-size:15px;margin:0 0 12px;display:flex;justify-content:space-between;align-items:center}
.bar{height:10px;background:var(--card);border-radius:6px;overflow:hidden;display:flex}
.bar .g{background:var(--green)}.bar .r{background:var(--red)}.bar .y{background:var(--yellow)}
.now{display:flex;gap:18px;align-items:center;flex-wrap:wrap}.now .big{font-size:20px;font-weight:700}
.timer{font-variant-numeric:tabular-nums;color:var(--yellow);font-size:20px;font-weight:700}
.flow{display:flex;gap:6px;align-items:stretch;flex-wrap:wrap}
.step{flex:1;min-width:150px;background:var(--card);border:2px solid transparent;border-radius:10px;padding:10px 12px}
.step.active{border-color:var(--yellow)}.step .t{font-weight:600;font-size:13px}
.step .e{color:var(--muted);font-size:11px;margin:4px 0 8px;min-height:30px}
.step .s{font-size:12px;display:flex;justify-content:space-between}.arrow{align-self:center;color:var(--grey)}
.chart .row{display:grid;grid-template-columns:180px 1fr 70px;gap:10px;align-items:center;font-size:12px;margin:6px 0}
.chart .track{background:var(--card);height:12px;border-radius:6px;overflow:hidden}
.chart .fill{background:var(--blue);height:100%}
table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:500}td.num{font-variant-numeric:tabular-nums}
.badge{padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600}
.badge.done{background:#1f3a29;color:var(--green)}.badge.failed{background:#3d1f1f;color:var(--red)}
.badge.running{background:#3a2f13;color:var(--yellow)}.badge.waiting{background:var(--card);color:var(--muted)}
.tabs{display:flex;gap:6px}.tabs button{background:var(--card);color:var(--muted);border:0;padding:5px 10px;
border-radius:6px;cursor:pointer;font-size:12px}.tabs button.on{color:var(--text);background:var(--line)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}
.reason{color:var(--red)}.muted{color:var(--muted)}select{background:var(--card);color:var(--text);
border:1px solid var(--line);border-radius:6px;padding:4px 6px}.next .step{opacity:.55}
"""

COMMON_JS = """
function fmt(s){ if(s===null||s===undefined) return '–'; s=Math.max(0,Math.round(s));
  const h=Math.floor(s/3600), m=Math.floor(s%3600/60), x=s%60;
  return h? `${h}h ${m}m` : (m? `${m}m ${x}s` : `${x}s`); }
function bar(s){ const t=s.total||1;
  return `<div class="bar"><div class="g" style="width:${100*s.done/t}%"></div>
  <div class="r" style="width:${100*s.failed/t}%"></div><div class="y" style="width:${100*s.running/t}%"></div></div>`; }
function esc(t){ return String(t).replace(/[&<>"']/g, ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch])); }
function pctbar(r){ if(r.pct===null||r.pct===undefined)
    return '<span class="muted" style="font-size:12px">no estimate yet (first time this step runs)</span>';
  return `<span style="display:inline-flex;align-items:center;gap:8px;font-size:13px">
    <span style="width:120px;height:8px;background:var(--card);border-radius:5px;overflow:hidden;display:inline-block">
    <span style="display:block;height:100%;width:${r.pct}%;background:var(--yellow)"></span></span>
    ~${r.pct}% <span class="muted">of usual ${fmt(r.expected_seconds)}</span></span>`; }
function label(s,k){ return (s.step_info[k]||{label:k}).label; }
function live(ok){ document.getElementById('live').innerHTML =
  `<span class="dot ${ok?'on':''}"></span>${ok?'live':'reconnecting…'}`; }
"""


def page(title: str, active: str, body: str, script: str) -> str:
    nav = "".join(
        f'<a href="{href}" class="{"on" if key == active else ""}">{name}</a>'
        for key, href, name in (
            ("home", "/", "Overview"),
            ("t1", "/t1", "T1 pipeline"),
            ("dti", "/dti", "DTI pipeline"),
        )
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>{STYLE}</style></head>
<body><header><span class="brand">PD-XAI · MRI preprocessing</span><nav>{nav}</nav>
<span class="live" id="live"><span class="dot"></span>connecting…</span></header>
<main>{body}</main><script>{COMMON_JS}{script}</script></body></html>"""


OVERVIEW_BODY = """<h1>MRI preprocessing</h1>
<div class="sub">T1 and DTI scans are cleaned in separate pipelines, then combined for features and the model.</div>
<div class="cards" id="cards"></div>
<div class="panel next" style="margin-top:16px"><h2><span>Combined (next stages)</span><span class="muted">not started</span></h2>
<div class="flow" id="next"></div></div>"""

OVERVIEW_JS = """
const NEXT = __NEXT__;
document.getElementById('next').innerHTML = NEXT.map(n=>`<div class="step"><div class="t">${n}</div></div>`)
  .join('<div class="arrow">→</div>');
const S = {};
function card(p, s){
  const now = s.current ? `${s.current.id} · ${label(s, s.current.step||'')}` : (s.status==='finished'?'finished':'idle');
  return `<a href="/${p}" class="panel" style="display:block;color:inherit">
   <h2><span>${p.toUpperCase()} pipeline</span><span class="muted">${s.status}</span></h2>
   ${bar(s)}<div class="kpis" style="margin-top:12px">
   <div class="kpi"><div class="v">${s.done}/${s.total}</div><div class="l">done</div></div>
   <div class="kpi"><div class="v" style="color:var(--red)">${s.failed}</div><div class="l">failed</div></div>
   <div class="kpi"><div class="v">${fmt(s.elapsed_seconds)}</div><div class="l">elapsed</div></div>
   <div class="kpi"><div class="v">${fmt(s.eta_seconds)}</div><div class="l">est. time left</div></div></div>
   <div class="muted" style="font-size:12px">Now: ${now} — open details →</div></a>`;
}
function render(){ document.getElementById('cards').innerHTML =
  ['t1','dti'].map(p=> S[p] ? card(p,S[p]) : '').join(''); }
['t1','dti'].forEach(p=>{
  const es = new EventSource('/api/stream?pipeline='+p);
  es.onmessage = e => { S[p] = JSON.parse(e.data); render(); live(true); };
  es.onerror = () => live(false);
});
"""

DETAIL_BODY = """<div style="display:flex;justify-content:space-between;align-items:flex-end;gap:12px;flex-wrap:wrap">
<div><h1 id="title"></h1><div class="sub" id="runinfo"></div></div>
<div class="muted" style="font-size:12px">Run: <select id="runsel"></select></div></div>
<div class="kpis" id="kpis"></div>
<div class="panel"><h2><span>Progress</span><span class="muted" id="pct"></span></h2><div id="bar"></div></div>
<div class="panel"><h2><span>Now processing</span></h2><div class="now" id="now"></div></div>
<div class="panel"><h2><span>Pipeline steps</span><span class="muted">yellow = step running now</span></h2>
<div class="flow" id="flow"></div></div>
<div class="panel chart"><h2><span>Average time per step</span><span class="muted">people who finished the step</span></h2>
<div id="chart"></div></div>
<div class="panel"><h2><span>People</span><div class="tabs" id="tabs"></div></h2>
<table><thead id="thead"></thead><tbody id="tbody"></tbody></table></div>"""

DETAIL_JS = """
const P = __P__; let S = null, filter = 'all', run = new URLSearchParams(location.search).get('run');
document.getElementById('title').textContent = P.toUpperCase() + ' pipeline';
async function loadRuns(){ const r = await (await fetch('/api/runs?pipeline='+P)).json();
  const sel = document.getElementById('runsel');
  sel.innerHTML = '<option value="">latest</option>' + r.map(x=>`<option ${x===run?'selected':''}>${x}</option>`).join('');
  sel.onchange = () => { location.search = sel.value ? '?run='+sel.value : ''; }; }
function render(){
  if(!S) return; const s = S;
  document.getElementById('runinfo').textContent = s.run ? `${s.run} · ${s.status}` : 'No run yet. Start the pipeline script.';
  const k = [['Total',s.total],['Done',s.done,'var(--green)'],['Failed',s.failed,'var(--red)'],
    ['Running',s.running,'var(--yellow)'],['Waiting',s.waiting],['Elapsed',fmt(s.elapsed_seconds)],
    ['Avg / person',fmt(s.avg_seconds_per_subject)],['Est. time left',fmt(s.eta_seconds)]];
  document.getElementById('kpis').innerHTML = k.map(x=>`<div class="kpi"><div class="v" style="color:${x[2]||'inherit'}">${x[1]}</div><div class="l">${x[0]}</div></div>`).join('');
  document.getElementById('bar').innerHTML = bar(s);
  document.getElementById('pct').textContent = s.total ? Math.round(100*(s.done+s.failed)/s.total)+'% processed' : '';
  const c = s.current; const many = (s.running_now||[]).length > 1;
  document.getElementById('now').innerHTML = !c ? `<span class="muted">${s.status==='finished'?'Run finished.':'Nothing running.'}</span>`
    : many ? `<div style="width:100%"><div class="muted" style="margin-bottom:8px">${s.running_now.length} people at once (jobs = ${s.jobs})</div>`+
      s.running_now.map(r=>`<div style="display:flex;gap:18px;margin:4px 0"><b style="min-width:90px">${r.id}</b>
        <span style="min-width:200px">${r.step? label(s,r.step):'between steps'}</span>
        <span class="timer tick" style="font-size:15px" data-base="${r.step_seconds??''}">${fmt(r.step_seconds)}</span>
        ${pctbar(r)}</div>${r.note?`<div class="muted" style="font-size:11px;margin:-2px 0 6px 108px">eddy says: ${esc(r.note)}</div>`:''}`).join('')+`</div>`
    : `<div><div class="muted">Person</div><div class="big">${c.id}</div></div>
    <div><div class="muted">Step</div><div class="big">${c.step? label(s,c.step):'between steps'}</div></div>
    <div><div class="muted">On this step</div><div class="timer tick" data-base="${c.step_seconds??''}">${fmt(c.step_seconds)}</div></div>
    <div><div class="muted">On this person</div><div class="timer tick" data-base="${c.subject_seconds??''}">${fmt(c.subject_seconds)}</div></div>
    <div style="min-width:220px"><div class="muted">Step progress (estimate)</div>${pctbar(c)}</div>
    ${c.note?`<div class="muted" style="width:100%;font-size:12px">Tool says: ${esc(c.note)}</div>`:''}`;
  window.tick0 = Date.now();
  document.getElementById('flow').innerHTML = s.steps.map(x=>{ const i = s.step_info[x.step]||{label:x.step,explain:''};
    const act = c && c.step===x.step;
    return `<div class="step ${act?'active':''}"><div class="t">${i.label}</div><div class="e">${i.explain}</div>
     <div class="s"><span>✓ ${x.done}/${s.total}</span><span style="color:var(--red)">${x.failed?('✗ '+x.failed):''}</span>
     <span class="muted">${x.avg_seconds!==null?fmt(x.avg_seconds):''}</span></div></div>`; }).join('<div class="arrow">→</div>');
  const mx = Math.max(1, ...s.steps.map(x=>x.avg_seconds||0));
  document.getElementById('chart').innerHTML = s.steps.map(x=>`<div class="row"><span>${label(s,x.step)}</span>
    <div class="track"><div class="fill" style="width:${100*(x.avg_seconds||0)/mx}%"></div></div>
    <span class="muted">${fmt(x.avg_seconds)}</span></div>`).join('');
  const counts = {all:s.total,running:s.running,done:s.done,failed:s.failed,waiting:s.waiting};
  document.getElementById('tabs').innerHTML = Object.entries(counts).map(([n,v])=>
    `<button class="${filter===n?'on':''}" onclick="filter='${n}';render()">${n} (${v})</button>`).join('');
  document.getElementById('thead').innerHTML = `<tr><th>Person</th><th>Status</th><th>Step</th>`+
    s.steps.map(x=>`<th>${label(s,x.step)}</th>`).join('')+`<th>Total</th><th>Checks</th><th>Reason</th></tr>`;
  const rows = s.subjects.filter(p=> filter==='all' || p.state===filter);
  document.getElementById('tbody').innerHTML = rows.map(p=>{
    const total = p.started ? ((p.ended||Date.now()/1000) - p.started) : null;
    const m = Object.entries(p.metrics||{}).map(([a,b])=>`${a.replace(/_/g,' ')}: ${Math.round(b*1000)/1000}`).join('<br>');
    return `<tr><td>${p.id}</td><td><span class="badge ${p.state}">${p.state}</span></td>
     <td>${p.step? label(s,p.step) : (p.failed_step? '<span class="reason">'+label(s,p.failed_step)+'</span>' : '')}</td>`+
     s.steps.map(x=>`<td class="num">${p.durations[x.step]!==undefined?fmt(p.durations[x.step]):''}</td>`).join('')+
     `<td class="num">${total!==null?fmt(total):''}</td><td class="muted">${m}</td><td class="reason">${p.reason||''}</td></tr>`; }).join('');
}
setInterval(()=>{ const d=(Date.now()-(window.tick0||Date.now()))/1000;
  for(const el of document.querySelectorAll('.tick')){
    if(el.dataset.base!=='') el.textContent = fmt(parseFloat(el.dataset.base)+d); } }, 1000);
loadRuns();
const es = new EventSource('/api/stream?pipeline='+P+(run?'&run='+run:''));
es.onmessage = e => { S = JSON.parse(e.data); render(); live(true); };
es.onerror = () => live(false);
"""


# ---------------------------------------------------------------- server


def make_handler(
    runs_dir: Path, token: str | None, interval: float
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: object, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802 (name required by http.server)
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            pipe = q.get("pipeline", "")
            run = q.get("run") or None
            if run is not None and (not SAFE_ID.match(run) or not run.startswith(pipe + "_")):
                return self._json({"error": "bad run id"}, 400)
            if url.path == "/":
                html = page(
                    "MRI preprocessing",
                    "home",
                    OVERVIEW_BODY,
                    OVERVIEW_JS.replace("__NEXT__", json.dumps(NEXT_STAGES)),
                )
                return self._send(200, html.encode(), "text/html; charset=utf-8")
            if url.path in ("/t1", "/dti"):
                p = url.path[1:]
                html = page(
                    f"{p.upper()} pipeline",
                    p,
                    DETAIL_BODY,
                    DETAIL_JS.replace("__P__", json.dumps(p)),
                )
                return self._send(200, html.encode(), "text/html; charset=utf-8")
            if pipe not in PIPELINES and url.path.startswith("/api/"):
                return self._json({"error": "pipeline must be t1 or dti"}, 400)
            if url.path == "/api/state":
                return self._json(state_for(pipe, runs_dir, run))
            if url.path == "/api/runs":
                return self._json(list_runs(pipe, runs_dir))
            if url.path == "/api/stream":
                return self._stream(pipe, run)
            return self._send(404, b"not found", "text/plain")

        def _stream(self, pipe: str, run: str | None) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = None
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                while True:
                    body = json.dumps(state_for(pipe, runs_dir, run))
                    if body != last:  # push only when something changed; timers tick in the browser
                        self.wfile.write(f"data: {body}\n\n".encode())
                        self.wfile.flush()
                        last = body
                    else:
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                    time.sleep(interval)

        def do_POST(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            if url.path != "/api/events":
                return self._send(404, b"not found", "text/plain")
            if token is None or parse_qs(url.query).get("token", [""])[0] != token:
                return self._json({"error": "events are only accepted with --token"}, 403)
            length = int(self.headers.get("Content-Length", "0"))
            try:
                event = json.loads(self.rfile.read(min(length, 65536)))
            except json.JSONDecodeError:
                return self._json({"error": "not JSON"}, 400)
            if not valid_event(event):
                return self._json({"error": "invalid event"}, 400)
            runs_dir.mkdir(parents=True, exist_ok=True)
            with (runs_dir / f"{event['run']}.jsonl").open("a") as f:
                f.write(json.dumps(event) + "\n")
            return self._json({"ok": True})

        def log_message(self, *_: object) -> None:  # keep the terminal quiet
            return

    return Handler


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument(
        "--host", default="127.0.0.1", help="0.0.0.0 to accept other machines (use --token)"
    )
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--token", help="secret required to POST events from another machine")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between live updates")
    args = ap.parse_args(argv)
    if args.host != "127.0.0.1" and not args.token:
        ap.error("--host other than 127.0.0.1 needs --token")

    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(args.runs_dir, args.token, args.interval)
    )
    server.daemon_threads = True
    shown = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    print(
        f"MRI pipeline dashboard: http://{shown}:{args.port}   (T1: /t1, DTI: /dti, Ctrl+C to stop)"
    )
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
