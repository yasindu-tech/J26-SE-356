"""Live monitoring dashboard for the MRI pipelines, from preprocessing to model training.

The T1 and DTI pipelines write a progress event for every person and step
(see progress.py). This server reads those events and pushes the live state to the
browser the moment it changes (Server-Sent Events, no page refresh).

Pages:
    /       overview of every pipeline

Every page combines all runs by default: each person (or training set) appears once, with
the result of the most recent run that worked on them. The run menu on each page switches
to the latest run only, or to one earlier run.
    /t1     T1 pipeline: counts, current person and step with a live timer,
            time per step, estimated time left, and every person's row
    /dti    the same for DTI
    /t1features  T1 feature extraction (one row of numbers per person)
    /dtifeatures DTI feature extraction
    /predict     single uploads scored by predict_scan.py (the app's MRI path)
    /train       model training: how the people were split, every model's metrics with
                 95% CIs, and the two mandatory controls (train_mri.py)

API (used by the pages, also handy for other tools):
    GET  /api/state?pipeline=t1[&run=latest|<id>] state as JSON (default: all runs combined)
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

from progress import (
    RUNS_DIR,
    combine_runs,
    expected_step_seconds,
    fold_events,
    latest_run,
    read_events,
    training_history,
)

PIPELINES = ("t1", "dti", "t1features", "dtifeatures", "train", "predict")
NAMES = {
    "t1": "T1 pipeline",
    "dti": "DTI pipeline",
    "t1features": "T1 features",
    "dtifeatures": "DTI features",
    "train": "Model training",
    "predict": "Scan scoring",
}
# what one row of a run is: a person for the pipelines, a training set for training
UNITS = {p: ("Person", "People") for p in PIPELINES} | {
    "train": ("Set", "Training sets"),
    "predict": ("Upload", "Uploads"),
}
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
    "t1features": {
        "load": ("Load images", "Reads the aligned T1, tissue maps and atlas grid check."),
        "tissue": ("Tissue shares", "Grey, white and CSF share in each of the 8 regions."),
        "brightness": (
            "Brightness stats",
            "Six statistics of normalised T1 brightness per region.",
        ),
        "size": ("Brain size", "Native brain volume and the affine scale factor."),
    },
    "dtifeatures": {
        "load": ("Load maps", "Reads FA and MD in standard space and checks the atlas grid."),
        "tracts": ("Tracts", "Mean FA and MD in 11 white-matter tracts (JHU atlas)."),
        "deepgrey": ("Deep grey matter", "Mean MD in the 8 regions used for T1."),
        "whole": ("Whole white matter", "Median FA and MD over all labelled white matter."),
    },
    "predict": {
        "convert": ("Convert", "dcm2niix on the upload; series sorted into T1 and DTI by content."),
        "t1": ("T1 pipeline", "N4, skull removal, alignment and tissue labels, with checks."),
        "dti": ("DTI pipeline", "Brain mask, eddy, FA and MD maps, alignment, with checks."),
        "features": ("Features", "The same feature code as training."),
        "score": ("Score", "Frozen ComBat and LightGBM from the model bundle, 95% interval."),
        "explain": ("Explain", "Exact TreeSHAP contribution of every feature."),
    },
    "train": {
        "cv": (
            "Cross-validate",
            "Subject-wise folds. ComBat and every model are fitted on each fold's "
            "training people only.",
        ),
        "metrics": ("Score", "AUC, balanced accuracy, 95% bootstrap CIs, PPV at 1/2/5%."),
        "site_gate": (
            "Site gate",
            "After ComBat, the scanner must be no easier to guess than with shuffled labels.",
        ),
        "permutation": (
            "Label permutation",
            "The same pipeline on shuffled diagnoses must score at chance.",
        ),
    },
}
NEXT_STAGES = ["Fusion with gait, finger tapping and voice"]


def state_for(pipeline: str, runs_dir: Path, run: str | None = None) -> dict:
    """State shown for a pipeline.

    run=None      every run combined: each person once, with their most recent result
    run="latest"  only the newest run
    run=<id>      only that run
    """
    # typical step times from every run of this pipeline, including the current one
    history = [read_events(p) for p in sorted(runs_dir.glob(f"{pipeline}_*.jsonl"))]
    expected = expected_step_seconds(history)
    if run is None:
        st = combine_runs(history, time.time(), expected)
    else:
        path = latest_run(pipeline, runs_dir) if run == "latest" else runs_dir / f"{run}.jsonl"
        events = read_events(path) if path is not None and path.exists() else []
        st = fold_events(events, time.time(), expected)
    if pipeline == "train":
        st["history"] = training_history(history)
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
        for key, href, name in (("home", "/", "Overview"),)
        + tuple((p, f"/{p}", NAMES[p]) for p in PIPELINES)
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>{STYLE}</style></head>
<body><header><span class="brand">PD-XAI · MRI pipeline</span><nav>{nav}</nav>
<span class="live" id="live"><span class="dot"></span>connecting…</span></header>
<main>{body}</main><script>{COMMON_JS}{script}</script></body></html>"""


OVERVIEW_BODY = """<h1>MRI pipeline</h1>
<div class="sub">T1 and DTI scans are cleaned in separate pipelines, turned into features, then used to train and test the model.
Every card combines all runs: each person counts once, with their most recent result.</div>
<div class="cards" id="cards"></div>
<div class="panel next" style="margin-top:16px"><h2><span>Combined (next stages)</span><span class="muted">not started</span></h2>
<div class="flow" id="next"></div></div>"""

OVERVIEW_JS = """
const NEXT = __NEXT__, NAMES = __NAMES__;
document.getElementById('next').innerHTML = NEXT.map(n=>`<div class="step"><div class="t">${n}</div></div>`)
  .join('<div class="arrow">→</div>');
const S = {};
function card(p, s){
  const now = s.current ? `${s.current.id} · ${label(s, s.current.step||'')}` : (s.status==='finished'?'finished':'idle');
  return `<a href="/${p}" class="panel" style="display:block;color:inherit">
   <h2><span>${NAMES[p]}</span><span class="muted">${s.runs_combined||0} run${s.runs_combined===1?'':'s'} · latest ${s.status}</span></h2>
   ${bar(s)}<div class="kpis" style="margin-top:12px">
   <div class="kpi"><div class="v">${s.done}/${s.total}</div><div class="l">done</div></div>
   <div class="kpi"><div class="v" style="color:var(--red)">${s.failed}</div><div class="l">failed</div></div>
   <div class="kpi"><div class="v">${fmt(s.elapsed_seconds)}</div><div class="l">elapsed</div></div>
   <div class="kpi"><div class="v">${fmt(s.eta_seconds)}</div><div class="l">est. time left</div></div></div>
   <div class="muted" style="font-size:12px">Now: ${now} — open details →</div></a>`;
}
function render(){ document.getElementById('cards').innerHTML =
  Object.keys(NAMES).map(p=> S[p] ? card(p,S[p]) : '').join(''); }
Object.keys(NAMES).forEach(p=>{
  const es = new EventSource('/api/stream?pipeline='+p);
  es.onmessage = e => { S[p] = JSON.parse(e.data); render(); live(true); };
  es.onerror = () => live(false);
});
"""

DETAIL_BODY = """<div style="display:flex;justify-content:space-between;align-items:flex-end;gap:12px;flex-wrap:wrap">
<div><h1 id="title"></h1><div class="sub" id="runinfo"></div></div>
<div class="muted" style="font-size:12px">Run: <select id="runsel"></select></div></div>
<div class="kpis" id="kpis"></div>
<div id="extra"></div>
<div class="panel"><h2><span>Progress</span><span class="muted" id="pct"></span></h2><div id="bar"></div></div>
<div class="panel"><h2><span>Now processing</span></h2><div class="now" id="now"></div></div>
<div class="panel"><h2><span>Pipeline steps</span><span class="muted">yellow = step running now</span></h2>
<div class="flow" id="flow"></div></div>
<div class="panel chart"><h2><span>Average time per step</span><span class="muted">__UNITS_LOWER__ that finished the step</span></h2>
<div id="chart"></div></div>
<div class="panel"><h2><span>__UNITS__</span><div class="tabs" id="tabs"></div></h2>
<table><thead id="thead"></thead><tbody id="tbody"></tbody></table></div>"""

DETAIL_JS = """
const P = __P__, NAME = __NAME__, UNIT = __UNIT__; let S = null, filter = 'all', run = new URLSearchParams(location.search).get('run');
document.getElementById('title').textContent = NAME;
async function loadRuns(){ const r = await (await fetch('/api/runs?pipeline='+P)).json();
  const sel = document.getElementById('runsel');
  sel.innerHTML = `<option value="">all runs combined</option><option value="latest" ${run==='latest'?'selected':''}>latest run only</option>`
    + r.map(x=>`<option value="${x}" ${x===run?'selected':''}>${runDate(x)} (${x})</option>`).join('');
  sel.onchange = () => { location.search = sel.value ? '?run='+sel.value : ''; }; }
function runDate(id){ const m = /_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})/.exec(id||''); return m ? `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}` : (id||''); }
function render(){
  if(!S) return; const s = S;
  document.getElementById('runinfo').textContent = !s.run ? 'No run yet. Start the pipeline script.'
    : s.run==='all' ? `All ${s.runs_combined} runs combined: each ${UNIT.toLowerCase()} once, with their most recent result · latest run ${runDate(s.latest_run)} is ${s.status}`
    : `${runDate(s.run)} (${s.run}) · ${s.status}`;
  const k = [['Total',s.total],['Done',s.done,'var(--green)'],['Failed',s.failed,'var(--red)'],
    ['Running',s.running,'var(--yellow)'],['Waiting',s.waiting],['Elapsed',fmt(s.elapsed_seconds)],
    ['Avg / '+UNIT.toLowerCase(),fmt(s.avg_seconds_per_subject)],['Est. time left',fmt(s.eta_seconds)]];
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
    : `<div><div class="muted">${UNIT}</div><div class="big">${c.id}</div></div>
    <div><div class="muted">Step</div><div class="big">${c.step? label(s,c.step):'between steps'}</div></div>
    <div><div class="muted">On this step</div><div class="timer tick" data-base="${c.step_seconds??''}">${fmt(c.step_seconds)}</div></div>
    <div><div class="muted">On this ${UNIT.toLowerCase()}</div><div class="timer tick" data-base="${c.subject_seconds??''}">${fmt(c.subject_seconds)}</div></div>
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
  document.getElementById('thead').innerHTML = `<tr><th>${UNIT}</th><th>Status</th><th>Step</th>`+
    s.steps.map(x=>`<th>${label(s,x.step)}</th>`).join('')+`<th>Total</th><th>Checks</th><th>Reason</th>`+
    (s.run==='all' ? '<th>Result from run</th>' : '')+`</tr>`;
  const rows = s.subjects.filter(p=> filter==='all' || p.state===filter);
  document.getElementById('tbody').innerHTML = rows.map(p=>{
    const total = p.reused ? Object.values(p.durations).reduce((x,y)=>x+y,0)
      : (p.started ? ((p.ended||Date.now()/1000) - p.started) : null);
    const m = Object.entries(p.metrics||{}).map(([a,b])=>`${a.replace(/_/g,' ')}: ${Math.round(b*1000)/1000}`).join('<br>');
    return `<tr><td>${p.id}</td><td><span class="badge ${p.state}">${p.state}</span></td>
     <td>${p.step? label(s,p.step) : (p.failed_step? '<span class="reason">'+label(s,p.failed_step)+'</span>' : '')}</td>`+
     s.steps.map(x=>`<td class="num">${p.durations[x.step]!==undefined?fmt(p.durations[x.step]):''}</td>`).join('')+
     `<td class="num">${total!==null?fmt(total):''}</td><td class="muted">${m}</td><td class="reason">${p.reason||''}</td>`+
     (s.run==='all' ? `<td class="muted">${p.from_run?runDate(p.from_run):''}${p.reused?'<br>(reused earlier outputs)':''}</td>` : '')+`</tr>`; }).join('');
  if (typeof renderExtra === 'function') renderExtra(s);
}
setInterval(()=>{ const d=(Date.now()-(window.tick0||Date.now()))/1000;
  for(const el of document.querySelectorAll('.tick')){
    if(el.dataset.base!=='') el.textContent = fmt(parseFloat(el.dataset.base)+d); } }, 1000);
loadRuns();
const es = new EventSource('/api/stream?pipeline='+P+(run?'&run='+run:''));
es.onmessage = e => { S = JSON.parse(e.data); render(); live(true); };
es.onerror = () => live(false);
"""


TRAIN_JS = """
var MODEL_INFO = {age_sex:'Age + sex only (floor)', best_single:'Best single feature',
  lgbm_t1:'LightGBM · T1', lgbm_dti:'LightGBM · DTI', lgbm_t1dti:'LightGBM · T1 + DTI'};
function f2(x){ return x===null||x===undefined ? '–' : Number(x).toFixed(2); }
function pc(x){ return x===null||x===undefined ? '–' : Math.round(100*x)+'%'; }
function span(a){ return !a ? '–' : (a[0]===a[1] ? `${a[0]}` : `${a[0]}–${a[1]}`); }
function aucBar(m){ const lo=0.3, hi=1, X=v=>Math.max(0,Math.min(100,100*(v-lo)/(hi-lo)));
  return `<div style="position:relative;height:14px;background:var(--card);border-radius:4px;min-width:150px">
   <div style="position:absolute;left:${X(0.5)}%;top:0;bottom:0;border-left:1px dashed var(--grey)"></div>
   <div style="position:absolute;left:${X(m.auc_ci[0])}%;width:${X(m.auc_ci[1])-X(m.auc_ci[0])}%;top:6px;height:2px;background:var(--muted)"></div>
   <div style="position:absolute;left:calc(${X(m.auc_mean)}% - 5px);top:2px;width:10px;height:10px;border-radius:50%;background:var(--blue)"></div></div>`; }
function verdict(ok, yes, no){ return `<span class="badge ${ok?'done':'failed'}">${ok?yes:no}</span>`; }
function historyTable(s){
  const h = s.history||[]; if(!h.length) return '';
  return `<div class="panel"><h2><span>All training runs</span><span class="muted">newest first · main model of each set</span></h2>
   <table><thead><tr><th>Run</th><th>Set</th><th>People</th><th>Split</th><th>Model</th><th>AUC (95% CI)</th>
   <th>Balanced accuracy</th><th>Sens.</th><th>Spec.</th><th>AUC vs age + sex</th><th>Controls</th></tr></thead><tbody>`+
   h.map(r=>{ const v=r.auc_vs_age_sex, sp=r.split||{};
     return `<tr><td>${runDate(r.run)}</td><td>${esc(r.set)}</td><td class="num">${r.n} (${r.n_pd} PD)</td>
      <td class="num">${sp.folds||'–'}×${sp.repeats||'–'}</td><td>${esc(MODEL_INFO[r.model]||r.model)}</td>
      <td class="num">${f2(r.auc_mean)} (${r.auc_ci?f2(r.auc_ci[0])+'–'+f2(r.auc_ci[1]):'–'})</td>
      <td class="num">${pc(r.bal_acc_mean)}</td><td class="num">${pc(r.sensitivity)}</td><td class="num">${pc(r.specificity)}</td>
      <td class="num">${v?(v[0]>=0?'+':'')+f2(v[0]):'–'}</td><td>${verdict(r.valid,'valid','INVALID')}</td></tr>`; }).join('')+
   `</tbody></table></div>`;
}
function renderExtra(s){
  const el = document.getElementById('extra');
  const sets = s.subjects.filter(p=>p.report);
  if(!sets.length){ el.innerHTML = (s.run ? '<div class="panel muted">Results appear here when the first set finishes.</div>'
    : '<div class="panel muted">No training run yet. Start it with: python models/mri/src/train_mri.py</div>') + historyTable(s); return; }
  el.innerHTML = sets.map(p=>{ const r = p.report, sp = r.split||{};
    const main = r.models.find(m=>m.model===r.main_model) || r.models[r.models.length-1];
    const groups = Object.entries(r.scanner_groups).map(([g,c])=>`${esc(g)}: `+
      Object.entries(c).map(([k,v])=>`${esc(k)} ${v}`).join(', ')).join(' · ');
    const kp = [[`${r.n}`, `people · ${r.n_pd} PD / ${r.n-r.n_pd} control`],
      [`${r.n_features}`, 'features'],
      [`${sp.folds}-fold × ${sp.repeats}`, `subject-wise · ${sp.fits_per_model} fits per model`],
      [`${span(sp.train_people)} / ${span(sp.test_people)}`, `train / test people per fold (${span(sp.test_pd)} PD in test)`],
      [f2(main.auc_mean), `AUC · ${esc(MODEL_INFO[main.model]||main.model)} · 95% CI ${f2(main.auc_ci[0])}–${f2(main.auc_ci[1])}`],
      [pc(main.bal_acc_mean), `balanced accuracy · 95% CI ${pc(main.bal_acc_ci[0])}–${pc(main.bal_acc_ci[1])}`]];
    const rows = r.models.map(m=>{ const v = m.auc_vs_age_sex;
      const vs = !v ? '<span class="muted">baseline</span>'
        : `${v[0]>=0?'+':''}${f2(v[0])} (${f2(v[1])} to ${f2(v[2])}) ${v[1]>0?verdict(true,'better',''):''}`;
      return `<tr${m.model===r.main_model?' style="background:var(--card)"':''}>
        <td>${esc(MODEL_INFO[m.model]||m.model)}</td><td>${aucBar(m)}</td>
        <td class="num">${f2(m.auc_mean)} ± ${f2(m.auc_sd)}</td><td class="num">${f2(m.auc_ci[0])}–${f2(m.auc_ci[1])}</td>
        <td class="num">${pc(m.bal_acc_mean)} ± ${pc(m.bal_acc_sd)}</td>
        <td class="num">${pc(m.sensitivity)}</td><td class="num">${pc(m.specificity)}</td>
        <td class="num">${Object.values(m.ppv).map(x=>(100*x).toFixed(1)+'%').join(' / ')}</td><td class="num">${vs}</td></tr>`; }).join('');
    const gates = Object.entries(r.site_gate).map(([g,x])=>`<div>Site gate (${esc(g)}): the scanner is guessed with
      ${pc(x.before_harmonisation)} balanced accuracy before ComBat and ${pc(x.score)} after; shuffled-label limit
      ${pc(x.threshold)} ${verdict(x.passed,'pass','FAIL')}</div>`).join('');
    const pm = r.permutation ? `<div>Label permutation (${esc(r.permutation.model)}): shuffled diagnoses score
      ${pc(r.permutation.score)} balanced accuracy; limit ${pc(r.permutation.threshold)}
      ${verdict(r.permutation.passed,'pass','FAIL')}</div>` : '';
    return `<div class="panel"><h2><span>Results · set "${esc(r.name)}"${p.from_run?` <span class="muted" style="font-weight:400">· run ${runDate(p.from_run)}</span>`:''}</span>${verdict(r.valid,'VALID','INVALID: do not quote')}</h2>
      <div class="kpis">${kp.map(k=>`<div class="kpi"><div class="v">${k[0]}</div><div class="l">${k[1]}</div></div>`).join('')}</div>
      <div class="muted" style="font-size:12px;margin-bottom:10px">Scanner groups used as ComBat batches: ${groups}.
        Every person is scored once per repeat, by models that never saw them.
        ${Object.keys(r.dropped_features||{}).length ? 'Left out where ComBat could not estimate them (no variance inside a scanner group), fold fits out of '+sp.fits_per_model+': '+Object.entries(r.dropped_features).map(([k,v])=>esc(k)+' ('+v+')').join(', ')+'.' : ''}
        Folds spread every scanner group evenly and balance diagnosis inside it${sp.min_train_per_scanner_group!==undefined?`; fewest training people of one scanner group in any fold: ${sp.min_train_per_scanner_group}`:''}.</div>
      <table><thead><tr><th>Model</th><th>AUC · dot = mean, line = 95% CI, dashed = chance</th><th>AUC mean ± SD</th>
        <th>AUC 95% CI</th><th>Balanced accuracy</th><th>Sensitivity</th><th>Specificity</th><th>PPV at 1% / 2% / 5%</th>
        <th>AUC vs age + sex</th></tr></thead><tbody>${rows}</tbody></table>
      <div style="margin-top:12px;font-size:13px;line-height:1.9">${gates}${pm}</div>
      <div class="muted" style="font-size:11px;margin-top:6px">Balanced accuracy, sensitivity and specificity use a 0.5
        threshold. PPV is at screening prevalence, not the 50/50 research set. Screening and triage only, not diagnosis.</div></div>`;
  }).join('') + historyTable(s);
}
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
            if run not in (None, "latest") and (
                not SAFE_ID.match(run) or not run.startswith(pipe + "_")
            ):
                return self._json({"error": "bad run id"}, 400)
            if url.path == "/":
                html = page(
                    "MRI pipeline",
                    "home",
                    OVERVIEW_BODY,
                    OVERVIEW_JS.replace("__NEXT__", json.dumps(NEXT_STAGES)).replace(
                        "__NAMES__", json.dumps(NAMES)
                    ),
                )
                return self._send(200, html.encode(), "text/html; charset=utf-8")
            if url.path[1:] in PIPELINES:
                p = url.path[1:]
                html = page(
                    NAMES[p],
                    p,
                    DETAIL_BODY.replace("__UNITS__", UNITS[p][1]).replace(
                        "__UNITS_LOWER__", UNITS[p][1].lower()
                    ),
                    DETAIL_JS.replace("__P__", json.dumps(p))
                    .replace("__NAME__", json.dumps(NAMES[p]))
                    .replace("__UNIT__", json.dumps(UNITS[p][0]))
                    + (TRAIN_JS if p == "train" else ""),
                )
                return self._send(200, html.encode(), "text/html; charset=utf-8")
            if pipe not in PIPELINES and url.path.startswith("/api/"):
                return self._json(
                    {"error": "pipeline must be one of: " + ", ".join(PIPELINES)}, 400
                )
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
        f"MRI pipeline dashboard: http://{shown}:{args.port}   (pages: /t1 /dti /t1features /dtifeatures /train /predict, Ctrl+C)"
    )
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
