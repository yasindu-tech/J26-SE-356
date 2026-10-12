## Feature contract (`src/mri_contract.py`)
`check_mri_features(columns)` must run before every `fit()`. It rejects
specialist-only inputs (UPDRS, Hoehn and Yahr, clinician ratings, medication,
DaTscan/QSM) and the label, and it also rejects any name that is not a known
MRI feature pattern, so a clinical column with an innocent name cannot slip in.
`age` and `sex` are allowed only with `allow_demographics=True` (baseline ladder).
