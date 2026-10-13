"""Tests for scripts/evaluate_hand_scale.py (task FT-13 follow-up).

Synthetic landmark tables only (made-up clip names such as CONTROL9001_DCHA); no real clip.
No conftest.py on purpose; ``src`` and the script are put on ``sys.path`` / loaded here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hand_landmarks as hl  # noqa: E402
from hubu_fis import pseudonym  # noqa: E402
from test_hand_scale import make_table  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "evaluate_hand_scale.py"
spec = importlib.util.spec_from_file_location("evaluate_hand_scale", SCRIPT)
assert spec and spec.loader
ev = importlib.util.module_from_spec(spec)
sys.modules["evaluate_hand_scale"] = ev
spec.loader.exec_module(ev)

FPS = 30.0


def tap(freq: float, n: int = 600) -> np.ndarray:
    """Open/close signal of ``freq`` cycles per second, 15 to 80 px."""
    return np.asarray(15.0 + 65.0 * (0.5 + 0.5 * np.sin(2 * np.pi * freq * np.arange(n) / FPS)))


def write_clips(folder: Path) -> dict[str, str]:
    """Four synthetic clips; returns {made-up clip name: hash}."""
    n = 600
    specs = {
        "CONTROL9001_DCHA": (1.0, np.linspace(0.8, 1.2, n)),  # approaches
        "CONTROL9001_IZDA": (1.5, np.ones(n)),  # stable
        "ID9101_DCHA": (0.6, np.linspace(1.2, 0.85, n)),  # recedes, slow tapper
        "ID9101_IZDA": (2.0, np.ones(n)),
    }
    for name, (freq, zoom) in specs.items():
        hl.write_landmarks(make_table(tap(freq, n), zoom=zoom), folder / f"{name}.parquet")
    return {name: pseudonym(name) for name in specs}


def load(folder: Path) -> dict[str, pd.DataFrame]:
    return ev.load_clips(folder)


# --------------------------------------------------------------------------- metrics


def test_a_scale_that_moves_with_the_tap_is_caught_per_frame_not_rolling() -> None:
    d = tap(1.0)
    coupled_zoom = 1.0 + 0.2 * (d - d.mean()) / (d.max() - d.min())  # scale follows the tap
    df = make_table(d, zoom=coupled_zoom)
    per_frame = ev.clip_metrics(df, "w9", "per_frame", None)
    rolling = ev.clip_metrics(df, "w9", "rolling_median", 3.0)
    assert per_frame["r"] > 0.9  # the check does fail when the rule is violated
    assert abs(rolling["r"]) < 0.3
    clean = ev.clip_metrics(make_table(d), "w9", "per_frame", None)
    assert np.isnan(clean["r"])  # a constant scale has no correlation: NaN, not 0


def test_zoom_clip_scales_the_hand_by_the_ramp() -> None:
    df = make_table(tap(1.0))
    zoomed = ev.zoom_clip(df, 0.5, 2.0)
    before, after = df["y_px09"] - df["y_px00"], zoomed["y_px09"] - zoomed["y_px00"]
    assert (after / before).iloc[[0, -1]].tolist() == pytest.approx([0.5, 2.0])


def test_helpers_on_known_numbers() -> None:
    assert ev.correlation(np.arange(30.0), 2 * np.arange(30.0)) == pytest.approx(1.0)
    assert np.isnan(ev.correlation(np.ones(30), np.arange(30.0)))
    assert np.isnan(ev.correlation(np.arange(5.0), np.arange(5.0)))  # too few values
    a = np.sin(np.arange(200) / 5.0)
    assert ev.lagged_max_abs_correlation(a, np.roll(a, 3)) > 0.99  # found at a lag


def test_rate_distribution_on_known_values() -> None:
    out = ev.rate_distribution(pd.Series([0.5, 1.0, 1.5, 2.5, 3.5, np.nan]))
    assert out["n"] == 5 and out["min"] == 0.5 and out["max"] == 3.5 and out["median"] == 1.5
    assert out["above_2"] == 2 and out["above_3"] == 1


def test_coupling_summary_flags_only_clips_above_the_limit() -> None:
    table = pd.DataFrame(
        {"rate": [0.5, 1.0, 1.5, 2.0], "abs_r_3s": [0.1, 0.7, 0.4, 0.51]},
        index=["aaaaaaaa", "bbbbbbbb", "cccccccc", "dddddddd"],
    )
    summary = ev.coupling_summary(table)
    assert sorted(summary["flagged"]) == ["bbbbbbbb", "dddddddd"]
    assert summary["slow_clips"] == 1 and summary["slow_clips_flagged"] == 0
    assert summary["two_cycle_window_s_for_slowest"] == pytest.approx(4.0)


# --------------------------------------------------------------------------- tables on synthetic clips


def test_tap_rate_and_approach_table_on_known_clips(tmp_path: Path) -> None:
    names = write_clips(tmp_path)
    clips = load(tmp_path)
    assert set(clips) == set(names.values())
    rates = {name: ev.tap_rate(clips[h]) for name, h in names.items()}
    assert rates["CONTROL9001_DCHA"] == pytest.approx(1.0, abs=0.1)
    assert rates["ID9101_DCHA"] == pytest.approx(0.6, abs=0.1)
    assert rates["ID9101_IZDA"] == pytest.approx(2.0, abs=0.15)

    table = ev.approach_table(clips)
    approach = table.loc[names["CONTROL9001_DCHA"]]
    assert (
        approach["hand_change"] == pytest.approx(0.4, abs=0.05)
        and approach["spearman_vs_time"] > 0.95
    )
    assert approach["open_trend_raw"] > 0.2 and abs(approach["open_trend_normalised"]) < 0.08
    assert approach["camera_trend_removed"] > 0.8
    assert table.loc[names["ID9101_DCHA"], "hand_change"] < -0.2  # recedes
    stable = table.loc[names["CONTROL9001_IZDA"]]
    assert abs(stable["hand_change"]) < 1e-6 and np.isnan(stable["camera_trend_removed"])
    assert table.index[-1] in (names["CONTROL9001_IZDA"], names["ID9101_IZDA"])  # sorted by size


def test_zoom_table_shows_clip_median_failing_and_per_frame_exact(tmp_path: Path) -> None:
    names = write_clips(tmp_path)
    clips = load(tmp_path)
    table = ev.zoom_table(clips, [names["CONTROL9001_DCHA"], names["CONTROL9001_IZDA"]])
    one = table.xs("0.7x to 1.4x", level="zoom").xs("w9", level="scale")
    assert one.loc["per_frame", "max_err"] == pytest.approx(0.0, abs=1e-9)
    assert one.loc["roll3s", "p95_err"] < 0.05
    assert one.loc["clip_median", "median_err"] > 0.05


def test_candidate_table_has_every_scale_and_mode(tmp_path: Path) -> None:
    write_clips(tmp_path)
    table, per_clip = ev.candidate_table(load(tmp_path))
    assert len(table) == len(ev.hs.SCALES) * len(ev.MODE_ROWS)
    assert set(per_clip["ref"]) == set(load(tmp_path))
    assert (table["jitter_median"] >= 0).all()


# --------------------------------------------------------------------------- command line


def run(tmp_path: Path, *extra: str) -> int:
    return int(
        ev.main(
            [
                "--landmarks-dir",
                str(tmp_path / "landmarks"),
                "--out-dir",
                str(tmp_path / "interim" / "out"),
                "--interim-root",
                str(tmp_path / "interim"),
                *extra,
            ]
        )
    )


def test_full_run_writes_only_inside_interim_and_shows_hashes_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    names = write_clips(tmp_path / "landmarks")
    assert run(tmp_path) == 0
    captured = capsys.readouterr()
    out = tmp_path / "interim" / "out"
    files = sorted(p.name for p in out.iterdir())
    assert files == [
        "approach_recede.csv",
        "candidates.csv",
        "candidates_per_clip.csv",
        "coupling_by_rate.csv",
        "summary.json",
        "world_aperture.csv",
    ]
    everything = captured.out + captured.err + " ".join(files)
    everything += " ".join(p.read_text() for p in out.iterdir())
    assert all(raw not in everything for raw in ("CONTROL9001", "CONTROL", "ID9101"))
    assert all(h in captured.out for h in names.values())
    written = {p.relative_to(tmp_path).parts[0] for p in tmp_path.rglob("*") if p.is_file()}
    assert written == {"landmarks", "interim"}


def test_missing_or_empty_landmark_folder_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(tmp_path) == 2
    (tmp_path / "landmarks").mkdir()
    assert run(tmp_path) == 2
    assert "no landmark parquet" in capsys.readouterr().err
    assert not (tmp_path / "interim").exists()


def test_output_folder_outside_the_interim_root_is_refused(tmp_path: Path) -> None:
    write_clips(tmp_path / "landmarks")
    code = ev.main(
        [
            "--landmarks-dir",
            str(tmp_path / "landmarks"),
            "--out-dir",
            str(tmp_path / "elsewhere"),
            "--interim-root",
            str(tmp_path / "interim"),
        ]
    )
    assert code == 1 and not (tmp_path / "elsewhere").exists()


def test_corrupt_parquet_exits_1_without_the_raw_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "landmarks"
    folder.mkdir()
    (folder / "CONTROL9002_DCHA.parquet").write_bytes(b"not a parquet file")
    assert run(tmp_path) == 1
    err = capsys.readouterr().err
    assert pseudonym("CONTROL9002_DCHA") in err and "CONTROL9002" not in err
    assert not (tmp_path / "interim").exists()


def test_world_aperture_on_a_known_geometry() -> None:
    df = make_table(np.full(60, 50.0))  # 50 mm open, knuckle 9 at 100 mm, palm facing the camera
    table = ev.world_aperture_table({"aaaaaaaa": df})
    row = table.loc["aaaaaaaa"]
    assert row["open_cm"] == pytest.approx(5.0) and row["w9_cm"] == pytest.approx(10.0)
    assert row["open_3d_hand_sizes"] == pytest.approx(0.5)
    assert row["open_2d_hand_sizes"] == pytest.approx(0.5, abs=1e-6)
    assert row["w9_tilt_deg"] == pytest.approx(0.0)
    tilted = df.copy()
    tilted["wz09"] = 0.1  # knuckle 9 moves 100 mm toward the camera: a 45 degree tilt
    assert ev.tilt_deg(tilted, 0, 9)[0] > 40
