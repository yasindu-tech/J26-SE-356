## Harmonisation (`src/harmonise.py`)
`ComBat` (parametric empirical Bayes, same steps as neuroCombat, checked against
it in the tests) with `fit` on the training fold and `transform` on the test
fold; `to_dict()` saves the frozen parameters for `predict_scan`.
`out_of_fold_harmonise()` does this per fold and refuses overlapping folds.
Diagnosis is never a ComBat covariate (unknown for a new patient).
Two controls stop the run instead of warning:
- `site_prediction_gate()` — after harmonisation the scanner must be no easier
  to guess than with shuffled scanner labels (`SiteGateError`).
- `label_permutation_control()` — the whole pipeline on shuffled diagnosis
  labels must score at chance (`PermutationControlError`).
