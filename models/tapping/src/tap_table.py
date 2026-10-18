"""The per-tap table (task FT-27): one row per detected tap, built from a taps parquet (FT-23).

Reused by Set A (features_a.py), Set B (FT-28) and the synthetic-decrement tests (FT-29).
Nothing is re-detected: every column comes from the parquet. Velocities are positive magnitudes
in hand sizes per second: ``open_vel_peak`` = max slope while opening, ``close_vel_peak`` =
-(min slope) while closing, both as stored by taps.py. ``iti_before_s`` is NaN for the first tap.
Flags: ``one_sided`` (a trough sits on a window edge, so the amplitude uses one trough),
``gap_interval`` (a NaN gap lies between this tap and the previous one), ``short_iti``
(interval below 0.2 s: possible double peak). No labels are read here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from taps import META_KEY

TABLE_VERSION = "tap-table-1"
SHORT_ITI_S = 0.2  # an interval shorter than this is flagged as a possible double peak
SOURCE_COLUMNS = (
    "ref", "tap_index", "frame", "t_peak_s", "amplitude", "open_vel", "close_vel",
    "open_vel_mean", "close_vel_mean", "rise_s", "fall_s", "interval_s", "flags",
)  # fmt: skip
TABLE_COLUMNS = (
    "ref", "tap_index", "frame", "t_peak_s", "amplitude", "open_vel_peak", "close_vel_peak",
    "open_vel_mean", "close_vel_mean", "rise_s", "fall_s", "iti_before_s", "one_sided",
    "gap_interval", "short_iti", "valid_amp", "valid_iti",
)  # fmt: skip


class TableError(ValueError):
    """A taps file cannot be turned into a tap table. The message never names a path."""


def read_taps_file(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The taps parquet and its metadata dict (parameters, tool version, ...)."""
    try:
        table = pq.read_table(path)
    except (OSError, ValueError) as err:
        raise TableError("a taps file cannot be read") from err
    raw = (table.schema.metadata or {}).get(META_KEY)
    if raw is None:
        raise TableError("a taps file has no taps metadata")
    return table.to_pandas(), json.loads(raw)


def build_tap_table(taps: pd.DataFrame) -> pd.DataFrame:
    """The tap table of one clip. Pure: the input is not changed."""
    missing = [c for c in SOURCE_COLUMNS if c not in taps.columns]
    if missing:
        raise TableError(f"the taps table lacks: {', '.join(missing)}")
    t = taps.reset_index(drop=True)
    if len(t) > 1 and not (np.diff(t["t_peak_s"].to_numpy(dtype=float)) > 0).all():
        raise TableError("tap times are not strictly increasing")
    flags = t["flags"].fillna("").astype(str).str.split(";").map(set)
    out = pd.DataFrame(
        {
            "ref": t["ref"].astype(str),
            "tap_index": t["tap_index"].astype("int64"),
            "frame": t["frame"].astype("int64"),
            "t_peak_s": t["t_peak_s"].astype(float),
            "amplitude": t["amplitude"].astype(float),
            "open_vel_peak": t["open_vel"].astype(float),
            "close_vel_peak": t["close_vel"].astype(float),
            "open_vel_mean": t["open_vel_mean"].astype(float),
            "close_vel_mean": t["close_vel_mean"].astype(float),
            "rise_s": t["rise_s"].astype(float),
            "fall_s": t["fall_s"].astype(float),
            "iti_before_s": t["interval_s"].astype(float),
        }
    )
    out["one_sided"] = flags.map(lambda f: "one_sided" in f).astype(bool)
    out["gap_interval"] = flags.map(lambda f: "gap_in_interval" in f).astype(bool)
    out["short_iti"] = (out["iti_before_s"] < SHORT_ITI_S).astype(bool)
    out["valid_amp"] = ~out["one_sided"]
    out["valid_iti"] = ~out["gap_interval"] & np.isfinite(out["iti_before_s"])
    return out[list(TABLE_COLUMNS)]
