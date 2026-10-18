"""Tests for models/tapping/src/tap_table.py (task FT-27): the per-tap table. Synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tap_table as tt  # noqa: E402


def make_taps(
    times: list[float],
    *,
    ref: str = "aaaa1111",
    amp: float | list[float] = 1.0,
    rise: float = 0.2,
    fall: float = 0.1,
    open_vel: float = 9.0,
    close_vel: float = 11.0,
    flags: dict[int, str] | None = None,
) -> pd.DataFrame:
    """A taps parquet content (all the columns taps.py writes) for made-up peak times."""
    n = len(times)
    amps = np.full(n, amp, dtype=float) if isinstance(amp, float | int) else np.array(amp, float)
    interval = np.concatenate([[np.nan], np.diff(times)])[:n]
    df = pd.DataFrame(
        {
            "ref": ref,
            "tap_index": np.arange(n),
            "frame": (np.array(times) * 30).round().astype(int),
            "t_peak_s": times,
            "value_peak": 1.0 + amps,
            "trough_left_s": np.array(times) - rise,
            "trough_right_s": np.array(times) + fall,
            "value_trough_left": 1.0,
            "value_trough_right": 1.0,
            "amplitude": amps,
            "amplitude_raw": amps * 1.05,
            "width_half_s": 0.1,
            "interval_s": interval,
            "rise_s": rise,
            "fall_s": fall,
            "open_vel": open_vel,
            "close_vel": close_vel,
            "open_vel_mean": amps / rise,
            "close_vel_mean": amps / fall,
            "flags": [(flags or {}).get(i, "") for i in range(n)],
        }
    )
    return df


def test_columns_and_values() -> None:
    taps = make_taps([0.5, 1.0, 1.5], amp=1.5, rise=0.25, fall=0.125)
    table = tt.build_tap_table(taps)
    assert list(table.columns) == list(tt.TABLE_COLUMNS)
    assert table["amplitude"].tolist() == [1.5] * 3
    assert table["open_vel_peak"].tolist() == [9.0] * 3  # the parquet's open_vel
    assert table["close_vel_peak"].tolist() == [11.0] * 3
    assert table["open_vel_mean"].tolist() == [6.0] * 3  # 1.5 / 0.25
    assert table["close_vel_mean"].tolist() == [12.0] * 3  # 1.5 / 0.125
    assert np.isnan(table["iti_before_s"].iloc[0])
    assert table["iti_before_s"].iloc[1:].tolist() == [0.5, 0.5]
    assert (table["open_vel_peak"] > 0).all() and (table["close_vel_peak"] > 0).all()


def test_flags_and_validity() -> None:
    taps = make_taps(
        [0.5, 1.0, 1.1, 2.0, 3.0],
        flags={0: "one_sided", 2: "narrow", 3: "gap_adjacent;gap_in_interval;long_interval"},
    )
    t = tt.build_tap_table(taps)
    assert t["one_sided"].tolist() == [True, False, False, False, False]
    assert t["gap_interval"].tolist() == [False, False, False, True, False]
    assert t["short_iti"].tolist() == [False, False, True, False, False]  # 0.1 s; first is NaN
    assert t["valid_amp"].tolist() == [False, True, True, True, True]
    assert t["valid_iti"].tolist() == [False, True, True, False, True]


def test_short_iti_boundary() -> None:
    t = tt.build_tap_table(make_taps([0.0, 0.2, 0.39]))
    assert t["short_iti"].tolist() == [False, False, True]  # exactly 0.2 is not short


def test_the_input_is_not_changed_and_nothing_is_redetected() -> None:
    taps = make_taps([0.5, 1.0])
    before = taps.copy(deep=True)
    tt.build_tap_table(taps)
    pd.testing.assert_frame_equal(taps, before)


def test_bad_tables_are_refused() -> None:
    with pytest.raises(tt.TableError):
        tt.build_tap_table(make_taps([0.5, 1.0]).drop(columns=["open_vel"]))
    with pytest.raises(tt.TableError):
        tt.build_tap_table(make_taps([0.5, 0.5]))
    assert len(tt.build_tap_table(make_taps([]))) == 0


def test_a_missing_file_is_a_table_error(tmp_path: Path) -> None:
    with pytest.raises(tt.TableError) as err:
        tt.read_taps_file(tmp_path / "nothing.parquet")
    assert str(tmp_path) not in str(err.value)
