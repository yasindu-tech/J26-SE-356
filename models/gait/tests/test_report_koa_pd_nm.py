"""Tests for the KOA-PD-NM report (GAIT-33), on a small made-up results table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import report_koa_pd_nm as rp
from run_koa_pd_nm import COLUMNS


def results(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "results.csv"
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
    return path


def clip(group: str, level: str, person: str, es: float, **extra: object) -> dict:
    return {
        "clip": f"w-{group}{level}{person}{es}",
        "group": group,
        "level": level,
        "person": person,
        "trial": "01",
        "known_two_people": 0,
        "status": "scored",
        "walk_seconds": 5.0 + es,
        "frame_rate": 30.0,
        "expected_severity": es,
        "predicted_class": int(round(es)),
        "step_length_norm": 0.4 - es / 10,
        "cadence": 110 - es,
        "run_seconds": 10.0,
    } | extra


def made_up() -> list[dict]:
    rows = [clip("NM", "", f"p-n{i}", 0.2 + i / 20) for i in range(6)]
    rows += [clip("NM", "", "p-n0", 0.3)]  # a second clip of the same person
    rows += [
        clip("PD", "ML", "p-1", 1.0),
        clip("PD", "MD", "p-1", 1.5),
        clip("PD", "SV", "p-2", 1.9),
    ]
    rows += [clip("PD", "SV", "p-3", 1.8, known_two_people=1, frame_rate=50.0)]
    rows += [
        clip("KOA", lv, f"p-k{i}", 0.6 + r / 4 + i / 50)
        for r, lv in enumerate(("EL", "MD", "SV"))
        for i in range(3)
    ]
    rows += [
        {
            **clip("NM", "", "p-n9", 0.0),
            "status": "not_scored",
            "reasons": "The walk has 2 real strides on both feet; at least 3 are needed.",
            "expected_severity": None,
        }
    ]
    return rows


def test_people_are_group_folder_and_person_and_cut_rows_are_dropped(tmp_path: Path) -> None:
    path = results(tmp_path, made_up())
    with path.open("a") as f:
        f.write("w-cut,NM,,p-x\n")  # a row cut short by a stopped run
    table = rp.load(path)
    assert "w-cut" not in set(table["clip"])
    pd_people = table[table["group"] == "PD"]["who"].nunique()
    assert pd_people == 4  # p-1 in ML and in MD are two people
    people = rp.per_person(table[table["status"] == "scored"])
    n0 = people[people["who"] == "NM||p-n0"]["expected_severity"].iloc[0]
    assert n0 == pytest.approx((0.2 + 0.3) / 2)  # clips averaged per person


def test_auc_is_by_person_and_the_sign_says_which_way_is_impaired(tmp_path: Path) -> None:
    table = rp.load(results(tmp_path, made_up()))
    people = rp.per_person(table[table["status"] == "scored"])
    assert rp.auc(people, "PD", "NM", "expected_severity", 1) == 1.0
    assert rp.pair_auc(np.array([1.0, 2.0]), np.array([2.0, 0.0])) == pytest.approx(0.625)
    assert rp.auc(people, "PD", "NM", "step_length_norm", -1) == 1.0  # shorter steps: impaired
    assert rp.auc(people, "PD", "NM", "step_length_norm", 1) == 0.0
    assert np.isnan(rp.auc(people, "PD", "XX", "expected_severity", 1))
    low, high = rp.auc_interval(people, "KOA", "NM", "expected_severity", 1)
    assert 0 <= low <= high <= 1
    assert (low, high) == rp.auc_interval(people, "KOA", "NM", "expected_severity", 1)  # seeded


def test_severity_order_follows_the_folders(tmp_path: Path) -> None:
    table = rp.load(results(tmp_path, made_up()))
    people = rp.per_person(table[table["status"] == "scored"])
    koa = rp.severity_order(people, "KOA")
    assert koa["spearman"] > 0.9 and koa["people"] == 9 and koa["ci_low"] <= koa["ci_high"]
    flat = people.assign(expected_severity=1.0)
    assert np.isnan(rp.severity_order(flat, "KOA")["spearman"])


def test_group_table_counts_people_clips_and_classes(tmp_path: Path) -> None:
    groups = rp.group_table(rp.load(results(tmp_path, made_up())))
    nm = groups[groups["group"] == "NM"].iloc[0]
    assert (nm["people"], nm["clips"], nm["scored"], nm["not_scored"]) == (7, 8, 7, 1)
    assert nm["level"] == "-" and nm["class_0"] == 7


def test_subsets_are_reported_and_the_helper_clip_is_left_out_of_one(tmp_path: Path) -> None:
    table = rp.load(results(tmp_path, made_up()))
    comps = rp.comparisons(table[table["status"] == "scored"])
    pd_nm = comps[(comps["comparison"] == "PD vs NM") & (comps["score"].str.startswith("model"))]
    by_subset = dict(zip(pd_nm["subset"], pd_nm["people"], strict=True))
    assert by_subset["all scored clips"] == "4 vs 6"
    assert by_subset["without the known helper clips"] == "3 vs 6"
    assert by_subset["50 fps clips only"] == "1 vs 0"
    baselines = comps[comps["subset"] == "all scored clips"]["score"].unique()
    assert len(baselines) == len(rp.SCORES)


def test_cli_writes_the_tables_and_the_figure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "report"
    assert rp.main(["--results", str(results(tmp_path, made_up())), "--out-dir", str(out)]) == 0
    for name in ("groups.csv", "comparisons.csv", "severity_order.csv", "not_scored.csv"):
        assert (out / name).is_file()
    assert (out / "severity_by_group.png").stat().st_size > 1000
    printed = capsys.readouterr().out
    assert "21 clips, 20 scored" in printed and "The walk has N real strides" in printed


def test_cli_without_results_says_what_to_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert rp.main(["--results", str(tmp_path / "none.csv")]) == 2
    assert "run_koa_pd_nm.py" in capsys.readouterr().out
