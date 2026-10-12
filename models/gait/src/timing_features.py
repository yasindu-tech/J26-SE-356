"""GAIT-14: timing features from the gait cycles of one walk.

Three features, all built from the durations of the *real* strides found by GAIT-13 (both
feet pooled, in time order):

- ``stride_time_mean`` (s): mean duration of all real strides.
- ``cadence`` (steps per minute): ``120 / stride_time_mean``. Two steps make a stride. It comes
  from stride time, never from the length of the recording, because walk duration carries the
  capture-area effect (a fixed capture area gives healthier, faster walkers fewer strides).
- ``stride_time_cv`` (%): stride-time variability, ``100 * SD / mean`` over a window of
  ``WINDOW_STRIDES`` consecutive strides from the middle of the walk (all strides if the walk
  has fewer), with the small-sample correction below.

Two choices need their reasons (measured on CARE-PD, 10 Oct 2026, see
``models/gait/docs/timing_features.md``):

1. *Small-sample correction.* The sample standard deviation of n values underestimates the true
   one by the factor ``c4(n)``: about 11% at n = 3 and 3% at n = 10 (the figures of Kroneberg
   et al. 2019). The SD is divided by ``c4(n)`` of the window.
2. *Fixed window.* The correction alone is not enough: the variability still grew with the
   number of strides in a walk (rank correlation 0.50), and the stride count differs by
   severity (median 11 strides in class 2-3, 6 in class 0). The middle ``WINDOW_STRIDES``
   strides give comparable numbers for short and long walks (correlation 0.32), skip the
   start-up and the turn, and keep the class difference.

Values are never left blank for short walks: blanks would fall mostly on BMCLab and on healthier
walkers and would leak the label. Whether short walks are reliable is for the GAIT-19 ablation
(models with and without ``stride_time_cv``). Walks that fail the cycle check get no features;
GAIT-17 only calls this for walks that pass. The stride count is returned for checks and is
never a feature (``FEATURE_NAMES`` lists the model inputs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from cycles import CycleCheck

WINDOW_STRIDES = 6  # strides used for the variability
MIN_STRIDES = 2  # a mean needs one stride, a standard deviation needs two
STEPS_PER_STRIDE = 2
FEATURE_NAMES = ("stride_time_mean", "cadence", "stride_time_cv")


def c4(n: int) -> float:
    """Factor by which the sample SD of ``n`` normal values underestimates the true SD."""
    if n < 2:
        raise ValueError("c4 needs at least 2 values")
    return math.sqrt(2.0 / (n - 1)) * math.gamma(n / 2) / math.gamma((n - 1) / 2)


def middle_window(times: np.ndarray, size: int = WINDOW_STRIDES) -> np.ndarray:
    """The ``size`` strides in the middle of the walk, or all of them if there are fewer."""
    if size < 2:
        raise ValueError("the window needs at least 2 strides")
    if times.size <= size:
        return times
    start = (times.size - size) // 2
    return times[start : start + size]


def stride_time_cv(times: np.ndarray, size: int = WINDOW_STRIDES) -> float:
    """Corrected coefficient of variation (%) over the middle window of ``times``."""
    window = middle_window(np.asarray(times, dtype=float), size)
    if window.size < 2:
        raise ValueError("need at least 2 strides for a variability")
    return float(100.0 * np.std(window, ddof=1) / c4(window.size) / np.mean(window))


@dataclass(frozen=True)
class TimingFeatures:
    stride_time_mean: float  # s
    cadence: float  # steps per minute
    stride_time_cv: float  # %
    n_strides: int  # real strides found: a CHECK, never a model input

    def as_dict(self) -> dict[str, float]:
        """The model inputs only (no stride count)."""
        return {name: getattr(self, name) for name in FEATURE_NAMES}


def timing_features(check: CycleCheck) -> TimingFeatures:
    """Features of one walk from its cycle check (real strides, both feet, in time order)."""
    times = check.stride_times()
    if times.size < MIN_STRIDES:
        raise ValueError(f"{times.size} real strides; timing features need at least {MIN_STRIDES}")
    mean = float(np.mean(times))
    return TimingFeatures(
        stride_time_mean=mean,
        cadence=60.0 * STEPS_PER_STRIDE / mean,
        stride_time_cv=stride_time_cv(times),
        n_strides=int(times.size),
    )
