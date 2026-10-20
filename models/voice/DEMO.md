# Voice module: PP1 demo runbook

A 5-minute live demo of the voice part, plus the evidence to show next to it.
Screening and triage only: say "referral priority", never "diagnosis".

## Before the session (the day before)

1. Pull `develop` with the upload feature merged (PR #108). Without it the
   desktop Voice screen can only open a result file.
2. Train or check the models (gitignored, so each laptop needs them):

   ```bash
   python models/voice/src/build_audio_table.py   # if artifacts/audio_features.csv is missing
   python models/voice/src/audio_model.py         # voice_audio_bundle.joblib, used by uploads
   ```

3. Demo files: use only people in `artifacts/holdout_audio_files.txt` (never
   seen in training). Keep one clear high score, one clear low score and one file
   that must be refused, for example a clip under 1 s of "aaah". Check them once:

   ```bash
   python models/voice/src/analyze_audio.py --model audio high.wav low.wav refused.wav
   ```

4. Make the offline backup: the same command writes
   `artifacts/analysis_results.json`, which the Voice screen opens with
   **Open result file** if the backend fails during the demo.
5. Start everything from the macOS Terminal app (Electron can exit at once
   inside the VS Code terminal):

   ```bash
   DATABASE_URL=<your local database URL> npm run dev:backend   # tab 1
   npm run dev:desktop                                           # tab 2
   ```

## The demo (about 5 minutes)

| Step | Do | Say |
|---|---|---|
| 1 | Sign in, Assessment → Voice | "Voice is one of four parts; each works on its own." |
| 2 | Upload the high-score file | "Upload is measured; the analysis takes a few seconds." |
| 3 | Read the card top to bottom | Plain reading → score with its range → how far to lean on it → what mattered most → recording check. |
| 4 | Open "Show technical details" | "Exact SHAP for every result; the reasons are checked for faithfulness." |
| 5 | Upload the low-score file | "A lower score means not prioritised, never cleared." Point at the wide range if it is wide. |
| 6 | Upload the refused file | "Bad recordings get no score and a reason. We never guess." |

## Evidence to show (one slide each, numbers from `RESULTS.md`)

1. **Leakage audit**: the common method reports AUC 0.918; the honest one 0.812.
2. **Cross-cohort drop**: the UCI-470 model, 0.73 inside its own dataset, scores
   at chance on new recordings (0.45–0.50), even with only the matching measures.
3. **Age, Figshare**: voice 0.710; age and sex alone 0.866. With normal ageing
   removed, voice keeps 0.754; trained on same-age people only, 0.556 (not yet
   above chance).
4. **XAI**: faithful (deletion curve, p = 0.010); the card's groups are fairly
   stable, the single measures are not.
5. **Safety**: quality checks, age and sex bias check, and the caution about
   other conditions that change the voice.

## Questions to be ready for

- *Why does age beat voice? Is it just detecting age?* Figshare's healthy group
  is about 20 years younger. With normal ageing removed from every voice
  measure, voice keeps AUC 0.754 (lower bound 0.64), so it hears more than age
  (`age_balanced.py`). Within each class the score does not follow age
  (`audio_bias.py`). Honest limit: trained on same-age people only (57), voice
  cannot yet be shown above chance (0.556); more older healthy voices are needed.
- *Is 0.71 good enough?* For screening support with an honest range, a start.
  More recordings (NeuroVoz, access pending) is the fix.
- *Why not use UCI-470 for uploads?* Show the transfer table.
- *Are the explanations true?* Show the deletion curve
  (`artifacts/xai_deletion_curve.png`).
- *Can a patient fake it?* Pure tones, silence and very long files are refused.
  A raised voice can pass, and a faked loudness tremor did not move the score;
  a clinician present is the main protection.
