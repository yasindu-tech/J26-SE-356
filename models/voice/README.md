# models/voice — leakage-audited voice biomarkers

**Owner:** Ricky H.P.C. (IT23294134)

**Question.** How much of the reported performance of voice-based PD detection
survives strictly subject-wise splitting with feature selection performed inside
each CV fold — and what is honest screening performance once prevalence and sex
confounding are accounted for?

## Data — UCI-470, verified
| Metric | Value |
|---|---|
| Subjects | 252 (188 PD / 64 HC) |
| Recordings | 756 (3 per subject) |
| Features | 752 |
| Class balance | 2.9 : 1 |

## Pipeline
Sustained vowel → quality gate + VAD → noise handling + normalisation → features
(jitter, shimmer, HNR, MFCC, TQWT, wavelet) → **feature selection INSIDE the CV
fold** → LightGBM / regularised linear → SHAP → fairness by sex and age band.

## Known problems
- **p ≫ n: 752 features, 252 subjects.** Selection must run inside each fold.
  Selecting on the full set first is the same leakage class as fitting ComBat
  outside the fold.
- **2.9:1 imbalance** — predicting "PD" for everyone scores 75%. Raw accuracy is
  meaningless here; report balanced accuracy and AUC.
- **Sex correlates with class** (41 HC / 81 PD vs 23 HC / 107 PD) — belongs in
  the fairness analysis and probably as a covariate.
- **No age column in UCI-470**, so the fairness analysis is by sex only; the
  age-band breakdown cannot be done on this dataset (see RESULTS.md, VOICE-17).
- **0 of 252 voice subjects have a PPMI scan** — this module is evaluated
  independently and is *not* in the fusion. Do not claim fusion here.
- End-to-end deep audio rejected at n=252.

## The headline deliverable
The **paired difference between the audited and unaudited pipelines**, with
bootstrap CIs — a number this literature does not currently report.

## PP1 scope

PP1 delivers an honest, tested voice-risk pipeline on the precomputed UCI-470
feature table, with the leakage audit as the headline result. After review with
the supervisor and team lead (Oct 2026), PP1 also shows **uploading an audio
file of a sustained "aaah", extracting voice measures from it, scoring it and
explaining the result** (D9-D13). No audio is recorded by the app in PP1.

### Raw-audio flow (D9-D13)
Any audio file (WAV, FLAC, OGG, MP3, M4A) → `audio_io.load_audio` (mono, 8 kHz)
→ `audio_features.extract_voice_features` (quick checks first: longer than 30 s or
almost silent is refused before any pitch tracking; then find the vowel, quality
gate, 56 named Praat/librosa features) → a voice far outside every training voice
(3+ measures, `audio_explain.far_outside_training`) is refused, not extrapolated →
one of two models → `analyze_audio.py`: calibrated
score, 95% CI, triage priority, SHAP top 5 and plain-language explanations
(`audio_explain.py`).

| Model (`--model`) | Trained on | Features | Script |
|---|---|---|---|
| `uci` (default) | UCI-470, 80% of people | the 16 Baseline measures Praat recomputes from audio (`uci_praat.py`) | `uci_transfer.py` |
| `audio` | Figshare raw audio, 80% of people | the 56 audio features | `audio_model.py` |

```bash
python models/voice/src/build_audio_table.py   # features for every recording
python models/voice/src/audio_evidence.py      # evidence for the audio model
python models/voice/src/audio_bias.py          # does the audio model's score follow age or sex?
python models/voice/src/audio_model.py         # train the audio model
python models/voice/src/uci_transfer.py        # UCI model on 16 Praat measures + transfer test
python models/voice/src/analyze_audio.py recording.mp3 [--model audio]
```
Open `models/voice/artifacts/analysis_results.json` in the desktop app with
**Open result file**. Demo the `audio` model only with files in
`artifacts/holdout_audio_files.txt`; every Figshare file is unseen by the `uci`
model. Read RESULTS.md first: the `uci` model does **not** transfer to telephone
audio.

### Flow
UCI-470 feature table (`data/raw/UCI-470/pd_speech_features.csv`, gitignored)
→ feature-contract check → **person-level** split (StratifiedGroupKFold, seed 42)
→ scaler + feature selection **fitted inside each training fold** → L1 logistic
regression / LightGBM → Platt calibration + bootstrap-ensemble CI → SHAP top 5
→ result vector.

### Decisions
| # | Decision |
|---|---|
| D1 | The research evidence (leakage audit, ladder, fairness) uses the UCI-470 table. Raw audio is added for the upload demo (D9). |
| D2 | The unit is the person. A person's 3 rows get one score (mean of the 3 probabilities). Metrics and bootstraps resample people, not rows. |
| D3 | Final hold-out: 20% of people, chosen once, stratified by class and sex, seed 42. The demo model trains on the other 80%. Evidence numbers come from 5-fold person-level CV over all people. |
| D4 | Models: L1 logistic regression and LightGBM, both with in-fold selection, balanced class weights, seed 42 everywhere. |
| D5 | Result vector fields: `risk_score`, `ci_lower`, `ci_upper`, `triage_priority`, `availability_mask`, `shap_top5`, `sex_fairness_note`, `pipeline_version`, `quality_gate_passed`. |
| D6 | `triage_priority` is `prioritised_referral` or `not_prioritised`. Never diagnosis, positive, negative or cleared. |
| D7 | Gender is not an input to the main model. It is used only for the sex-only baseline and the fairness analysis. |
| D8 | Mobile recording is PP2. The desktop result card waits for the desktop branch to merge. |
| D9 | The full UCI-470 model cannot score new audio: 22 of the 24 features it uses (TQWT, MFCC, wavelet) came from the authors' own tools and cannot be recomputed. Uploads are scored either by a UCI-470 model retrained on the 16 Baseline measures Praat recomputes (`uci_praat.py`), or by an audio model trained on raw audio. One extractor serves training and uploads. |
| D10 | Task: sustained "aaah" only. All audio is resampled to 8 kHz (telephone band) so datasets and uploads are comparable. Any common format is accepted; the format is recorded and compressed files get an uncertainty note. |
| D11 | Raw-audio data: Figshare voice samples (CC BY 4.0). NeuroVoz (access pending) and Italian PVS are added if they arrive. KCL has no sustained vowels, so it is not used. Age and sex are never voice-model inputs, but an age + sex baseline is mandatory because Figshare's healthy group is much younger. |
| D12 | XAI for every output: SHAP top 5 in plain words, the value's place among healthy and PD training voices, uncertainty reasons, quality-gate reasons, and the model's overall feature use. Explanations say "the model used", never "causes". |
| D13 | A measure whose values from new audio do not match the training data's scale (`uci_transfer.unit_check`) is never compared with training voices in an explanation; the result says so instead. If the main score falls outside its bootstrap range, the range is widened, never narrowed. |

### Definition of done
1. `predict_voice(feature_row)` returns the full result vector for a held-out person.
2. Leakage audit: audited vs unaudited AUC with bootstrap 95% CIs and the paired difference.
3. Baseline ladder: sex only → best single feature → all features → LightGBM.
4. PPV at 1%, 2% and 5% prevalence.
5. Sex-fairness breakdown on the model card.
6. Red tests for the splitter, feature contract, in-fold selection and permutation control.
7. A demo command that scores five held-out people and writes the result JSON.
8. Desktop result card (or the demo command if the desktop branch is not merged).
9. One evidence slide: audit chart, ladder, sex breakdown, PPV.
10. No PHI, no data files, no secrets in commits.

### Out of scope for PP1
Audio capture in the app, the mobile app, reading and free-speech tasks, and
deployment.

### Setup
```bash
brew install python@3.11 libomp          # libomp is required by LightGBM on macOS
/opt/homebrew/bin/python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt -r models/voice/requirements.txt
```
Place the dataset at `data/raw/UCI-470/pd_speech_features.csv`. The UCI download
wraps it in `pd_speech_features.rar` inside the zip; macOS `bsdtar -xf` extracts
it. The file has **two header rows** (a group row, then column names). Never copy
it outside `data/` — a `.csv` elsewhere is not gitignored.
