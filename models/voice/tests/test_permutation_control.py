"""Tests for models/voice/src/permutation_control.py. Synthetic data only.

The key red test: on shuffled labels a deliberately leaky pipeline must FAIL the
gate, and the honest pipeline must pass it. A gate that never fails is not a gate.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import permutation_control as pc
import pytest

FAST = dict(score="anova_f", k_grid=(5, 10), c_grid=(0.1,), leaves_grid=(4,))
N_PEOPLE = 100


def signal_data(n_features: int = 30) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """100 people x 3 rows; f0 carries a real signal, so the unshuffled data is learnable."""
    rng = np.random.default_rng(42)
    groups = np.repeat(np.arange(9000, 9000 + N_PEOPLE), 3)
    y = np.repeat((rng.random(N_PEOPLE) < 0.75).astype(int), 3)
    X = pd.DataFrame(
        rng.normal(size=(len(groups), n_features)), columns=[f"f{i}" for i in range(n_features)]
    )
    X["f0"] += y * 1.5
    return X, y, groups


def test_shuffle_is_per_person_and_keeps_class_balance() -> None:
    _, y, groups = signal_data()
    y_perm = pc.permute_labels(y, groups, np.random.default_rng(0))
    per_person = pd.Series(y_perm).groupby(groups).nunique()
    assert (per_person == 1).all()  # all 3 rows of a person share the new label
    assert y_perm.sum() == y.sum()  # same number of PD rows
    assert not np.array_equal(y_perm, y)  # and it really was shuffled


def test_shuffle_is_reproducible() -> None:
    _, y, groups = signal_data()
    a = pc.permute_labels(y, groups, np.random.default_rng(3))
    b = pc.permute_labels(y, groups, np.random.default_rng(3))
    assert np.array_equal(a, b)


def test_honest_pipeline_passes_on_shuffled_labels() -> None:
    X, y, groups = signal_data()
    aucs = pc.run_permutations(X, y, groups, n_permutations=5, **FAST)
    pc.check_chance(aucs)  # must not raise
    assert abs(np.mean(aucs) - 0.5) < pc.TOLERANCE


def test_leaky_pipeline_fails_the_gate() -> None:
    """Red test: leaky selection on shuffled labels scores above chance and must be caught."""
    X, y, groups = signal_data(n_features=752)
    aucs = pc.run_permutations(X, y, groups, selection="leaky", n_permutations=3, **FAST)
    assert np.mean(aucs) > 0.5 + pc.TOLERANCE
    with pytest.raises(pc.PermutationControlError, match="shuffled labels"):
        pc.check_chance(aucs)


@pytest.mark.parametrize("aucs", [[0.62, 0.58, 0.60], [0.9]])
def test_gate_raises_above_chance(aucs: list[float]) -> None:
    with pytest.raises(pc.PermutationControlError):
        pc.check_chance(aucs)


@pytest.mark.parametrize("aucs", [[0.48, 0.55, 0.51], [0.5]])
def test_gate_passes_at_chance(aucs: list[float]) -> None:
    pc.check_chance(aucs)


def test_gate_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="no permutation"):
        pc.check_chance([])


def test_run_control_reports_both_models_without_ids() -> None:
    X, y, groups = signal_data()
    result = pc.run_control(X, y, groups, n_permutations=2, **FAST)
    assert set(result["models"]) == {"l1_logistic", "lightgbm"}
    assert all(r["passed"] for r in result["models"].values())
    text = json.dumps(result)
    assert not any(str(pid) in text for pid in range(9000, 9000 + N_PEOPLE))
    assert "Gate" in pc.results_markdown(result)


def test_run_control_raises_instead_of_reporting_a_leak(monkeypatch) -> None:
    """A failing model must stop the run; it must never be written up as a result."""
    monkeypatch.setattr(pc, "run_permutations", lambda *a, **k: [0.8, 0.8])
    X, y, groups = signal_data()
    with pytest.raises(pc.PermutationControlError):
        pc.run_control(X, y, groups, n_permutations=2)
