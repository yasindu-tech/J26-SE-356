# Install the team's models

Each module owner shares their trained model as one zip on the **team drive**, never through
git. The models are derived from data under data use agreements (PPMI, HUBU-FIS, figshare,
UCI). A zip holds model parameters and a model card only: no scans, videos, recordings,
per-person features, labels or file names.

Screening and triage only. A score above a module's threshold means *prioritised referral*,
below it means *not prioritised*. Nothing here is a diagnosis.

## 1. Get the zips

From the team drive, one per module:

| Module | Zip | Owner |
|---|---|---|
| MRI | `pdxai_models_mri_<version>.zip` | Yasindu |
| Gait | not shared as a zip yet | Fernando |
| Voice | `pdxai_models_voice_<version>.zip` | Ricky |
| Tapping | `pdxai_models_tapping_<version>.zip` | Weerasinghe |

Keep the zips **outside** the repository (e.g. `~/pdxai_zips/`). `*.zip` and `pdxai_models_*/`
are git-ignored as a second guard, not as permission to keep them in the repo.

## 2. Install them

MRI and tapping zips unpack straight into place; their paths start with `data/models/`,
which git ignores:

```bash
unzip ~/pdxai_zips/pdxai_models_tapping_<version>.zip -d <repo root>
unzip ~/pdxai_zips/pdxai_models_mri_<version>.zip -d <repo root>
```

The voice zip has its own `README.txt` (copy `model.joblib` to
`models/voice/artifacts/voice_audio_bundle.joblib` after checking its SHA-256 against its
`MANIFEST.json`). It is a pickle: only load one whose hash matches.

`scripts/import_models.py` (hash check, downgrade refusal, a table of what is installed) is
planned with the shared model registry and is not in the repo yet.

## 3. Check them

Each bundle is checked against its SHA-256 hashes every time it loads; a changed or partly
copied file is refused, and that modality reports the reason instead of a score.

```bash
python models/tapping/src/tapping_bundle.py check   # "tapping bundle <version>: N files, hashes OK"
```

`scripts/check_models.py` (one synthetic input per modality, PASS/FAIL, missing system
tools) is planned with the shared model registry and is not in the repo yet.

## 4. Start the backend

As usual (`services/backend/README.md`). What each modality needs on this laptop:

| Modality | Runs where | Needs |
|---|---|---|
| MRI | host (not Docker) | FSL, ANTs, dcm2niix, SynthStrip, `models/mri` packages |
| Gait | host (not Docker) | `models/gait/requirements.txt` (mediapipe, xgboost) |
| Voice | Docker or host | ffmpeg; in Docker `models/voice/artifacts/` is mounted read-only |
| Tapping | host (not Docker) | `models/tapping/requirements.txt` (mediapipe, lightgbm, shap); lightgbm must equal the version in the bundle's `manifest.json` |

## For owners: making a zip

```bash
python models/tapping/src/tapping_bundle.py build      # tapping: writes data/models/tapping/<version>/ and CURRENT
python scripts/export_models.py --module tapping --out ~/pdxai_zips
python scripts/export_models.py --module mri --out ~/pdxai_zips
```

`export_models.py` refuses a bundle whose files differ from its manifest, that contains a data
table, raw input or pickle, or that has person-level keys (`participant`, `subject_id`,
`rating`, `labels`, ...). Build from a committed checkout: the version names the git commit.
