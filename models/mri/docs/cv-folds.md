## Cross-validation folds (`src/folds.py`)
Repeated, stratified, subject-wise K-fold (default 5 x 10). A person is never
in train and test of the same fold; the generator asserts this on every fold,
and fails if a test part would hold only one class. Pass `strata=` (e.g. the
scanner model) to balance a second variable; `unseen_test_groups()` reports a
test-fold scanner that training never saw.
