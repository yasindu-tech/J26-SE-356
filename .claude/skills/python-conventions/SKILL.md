---
name: python-conventions
description: >-
  Apply consistent Python code patterns whenever writing or reviewing Python
  in the J26-SE-356 repo — services/backend (FastAPI) or any models/* module
  (mri, gait, voice, tapping, fusion, common). Use this whenever the user
  asks to write a FastAPI endpoint, a model training/eval script, a data
  pipeline function, or any .py file in this repo, or asks for a Python code
  review. This operationalizes CLAUDE.md §3 (leakage, splitting, metrics)
  and §5 (ruff, mypy, pytest, type hints) as concrete code patterns — reach
  for it any time those rules are relevant, not just when the user
  explicitly asks about style.
---

## Why this exists

CLAUDE.md §5 sets the bar (ruff, mypy where practical, type hints on public
functions, pytest, "boring and readable over clever"). §3 sets research-
integrity rules that are really just correctness requirements with research
consequences. This skill is the concrete "how" for both — the patterns that
make the CLAUDE.md rules the natural path, not something you have to
remember to bolt on afterward.

## Structural patterns that make CLAUDE.md §3 easy to get right

**Use `sklearn.pipeline.Pipeline` for anything with a `fit` step, always.**
This is the single highest-leverage pattern in this codebase for avoiding
the leakage CLAUDE.md §3.2 warns about. A `Pipeline` composing scaler →
feature selector → model, run through `cross_val_score` or a manual CV loop
that calls `.fit()` only on the train fold, makes "fit inside the fold" the
default instead of something you have to remember on every script:

```python
# ✅ harmonisation/scaling/selection all fit only on the train fold,
# because the Pipeline is fit once per fold, not once on the full dataset
pipeline = Pipeline([
    ("harmonise", ComBatTransformer(site_col="site")),
    ("scale", StandardScaler()),
    ("select", SelectKBest(k=20)),
    ("model", LogisticRegression()),
])
for train_idx, test_idx in cv.split(X, y, groups=subject_id):
    pipeline.fit(X[train_idx], y[train_idx])
    score = pipeline.score(X[test_idx], y[test_idx])
```

```python
# ❌ ComBat and the scaler see the test fold before the split ever happens
X_scaled = StandardScaler().fit_transform(X)
X_selected = SelectKBest(k=20).fit_transform(X_scaled, y)
for train_idx, test_idx in cv.split(X_selected, y):
    ...
```

If a transform can't be expressed as a scikit-learn-compatible
`fit`/`transform` step, wrap it in one (`BaseEstimator`, `TransformerMixin`)
rather than writing it as a free function called before the split — the
wrapping is what makes the leakage structurally hard to reintroduce later
when someone edits the script.

**Always pass `groups=subject_id` to the splitter.** `GroupKFold`,
`StratifiedGroupKFold`, or `GroupShuffleSplit` — never a plain `KFold` or
`train_test_split` without `groups`, per CLAUDE.md §3.3. Assert it:

```python
def assert_subject_disjoint(train_idx, test_idx, subject_id):
    train_subjects = set(subject_id[train_idx])
    test_subjects = set(subject_id[test_idx])
    assert not (train_subjects & test_subjects), "subject leaked across split"
```

Call this assertion in the CV loop itself, not just in a test — a test can
pass on synthetic data while the real pipeline still leaks.

**Report balanced accuracy, AUC, and a bootstrap CI as a matter of course**,
not as an afterthought once someone asks for it:

```python
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.utils import resample

def bootstrap_ci(y_true, y_pred_proba, metric_fn, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    scores = [
        metric_fn(*resample(y_true, y_pred_proba, random_state=rng.integers(1e9)))
        for _ in range(n_boot)
    ]
    return np.percentile(scores, [2.5, 97.5])
```

**Missing modalities get a mask feature, never a fill value.** If a function
accepts multi-modal input, its signature should make "absent" explicit
(e.g. `Optional[np.ndarray]` plus a boolean mask array), not silently
default a missing array to zeros.

## General Python patterns

- **Type-hint every public function's signature**, including return type.
  `mypy` can't catch what isn't annotated.
- **Pydantic models for anything crossing a boundary** — FastAPI
  request/response bodies, config loaded from `.env` or a YAML file. Don't
  pass raw `dict`s across a function boundary that represents an API
  contract; a typo in a key name should fail fast, not silently return
  `None` downstream.
- **`pathlib.Path`, not `os.path`** — for the data-path handling in
  `models/*`, this also makes it easier to spot where `data/` paths are
  constructed, which matters for the "nothing from `data/` is committed"
  rule (grep for `Path("data"` is easier than grepping for arbitrary string
  concatenation).
- **No mutable default arguments** (`def f(items=[])`) — the classic Python
  footgun, and one that's easy to introduce in a config-heavy research
  codebase.
- **No bare `except:`.** Catch the specific exception, or `except Exception`
  with a comment on why it's intentionally broad (e.g. a top-level API
  error handler).
- **`logging`, not `print`, for anything in `services/backend` or a
  long-running training script.** Subject identifiers in log output must be
  hashed or pseudonymised per CLAUDE.md §4 — don't log a raw subject ID even
  at debug level.
- **Dataclasses or Pydantic models over ad-hoc dicts** for anything with a
  fixed shape used in more than one place (a training config, a feature
  vector's metadata).
- **FastAPI: use `Depends()` for shared setup** (DB session, model loader)
  rather than reconstructing it inline in every route — keeps routes
  testable in isolation.
- **pytest: use fixtures and `@pytest.mark.parametrize`** rather than
  copy-pasted near-identical test functions — and remember CLAUDE.md §5's
  rule that a validator needs a test proving it *fails* on bad input, not
  just one proving it passes on good input.

## When reviewing existing code

Look first for the leakage-shaped bugs — a `.fit()` or `.fit_transform()`
called before a CV split, a splitter without `groups=`, a metric report
that's raw accuracy with no CI. These cost the most if missed. Then check
the general patterns above. Flag issues even on an unrelated PR if you
notice them while reading nearby code — cheaper to mention now than to
discover after a leaky number makes it into a slide.
