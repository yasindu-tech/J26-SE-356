"""Tests for models/tapping/src/detection_audit.py (task FT-17).

Synthetic only: made-up manifests and labels with clip IDs such as ``CONTROL9001_DCHA`` and
``ID9101_IZDA``. No real data, no conftest.py (voice and gait have one).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import detection_audit as da  # noqa: E402
from hubu_fis import pseudonym  # noqa: E402

# clip, detection rate, flags, resolution, rotation metadata, UPDRS
SPEC: list[tuple[str, float, str, str, int, int]] = [
    ("CONTROL9001_DCHA", 1.00, "", "1080x1920", 90, 0),
    ("CONTROL9001_IZDA", 0.97, "", "1080x1920", 90, 1),
    ("CONTROL9002_DCHA", 0.90, "", "1920x1080", 0, 0),
    ("CONTROL9002_IZDA", 1.00, "", "1920x1080", 0, 0),
    ("CONTROL9003_IZDA", 0.80, "low_detection", "1920x1080", 0, 1),  # one hand only
    ("ID9101_DCHA", 0.50, "low_detection;gaps_not_fillable", "1080x1920", 90, 2),
    ("ID9101_IZDA", 0.96, "scale_tap_coupling", "1080x1920", 90, 3),
    ("ID9102_DCHA", 0.95, "", "1080x1920", 90, 1),
    ("ID9102_IZDA", 0.99, "info_irregular_timing", "1080x1920", 90, 1),
    ("ID9103_DCHA", 0.40, "low_detection", "1080x1920", 90, 3),
    ("ID9103_IZDA", 0.45, "low_detection", "1080x1920", 90, 3),
]
RAW_PARTS = ("CONTROL9001", "CONTROL9002", "CONTROL9003", "ID9101", "ID9102", "ID9103")


def manifest(spec: list[tuple[str, float, str, str, int, int]] | None = None) -> pd.DataFrame:
    rows = []
    for clip, rate, flags, resolution, rotation, _ in spec or SPEC:
        group = "CONTROL" if clip.startswith("CONTROL") else "ID"
        width, height = (int(v) for v in resolution.split("x"))
        rows.append(
            {
                "clip_id": clip,
                "ref": pseudonym(clip),
                "participant_ref": pseudonym(clip.split("_")[0]),
                "group": group,
                "hand": "right" if clip.endswith("DCHA") else "left",
                "resolution": resolution,
                "orientation": "portrait" if height > width else "landscape",
                "orientation_meta": rotation,
                "fps": 30.0,
                "frames_decoded": 600,
                "frames_detected": int(round(600 * rate)),
                "detection_rate": rate,
                "longest_missing_run": int(round(600 * (1 - rate) / 3)),
                "n_gaps": 3,
                "flags": flags if flags else np.nan,  # an empty cell is NaN in the CSV
                "status": "ok",
            }
        )
    return pd.DataFrame(rows)


def labels(spec: list[tuple[str, float, str, str, int, int]] | None = None) -> pd.DataFrame:
    rows = [(clip, updrs) for clip, *_, updrs in spec or SPEC]
    table = pd.DataFrame(rows, columns=["ID", "UPDRS"])
    return table


def frame(spec: list[tuple[str, float, str, str, int, int]] | None = None) -> pd.DataFrame:
    return da.build_log(manifest(spec), labels(spec))


def audit_at(threshold: float, spec: Any = None) -> dict[str, Any]:
    result = da.audit(frame(spec), [threshold])
    entry: dict[str, Any] = result["by_threshold"][f"{threshold:g}"]
    return entry


# --------------------------------------------------------------------------- the retention rule


def test_retained_means_at_or_above_the_threshold_and_nan_is_not_retained() -> None:
    rate = pd.Series([0.89999, 0.9, 0.90001, np.nan])
    assert da.retained_mask(rate, 0.9).tolist() == [False, True, True, False]


def test_a_clip_exactly_at_the_threshold_is_retained() -> None:
    entry = audit_at(0.90)
    assert entry["overall"]["clips"] == 11
    # rates 0.50, 0.40, 0.45, 0.80 fall below 0.90; 0.90, 0.95, 0.96, 0.97, ... are kept
    assert entry["overall"]["retained"] == 7 and entry["overall"]["dropped"] == 4
    assert audit_at(0.9000001)["overall"]["retained"] == 6  # the clip at exactly 0.90 now drops


def test_overall_counts_and_percentages_use_all_clips_as_the_denominator() -> None:
    o = audit_at(0.95)["overall"]
    assert (o["clips"], o["retained"], o["dropped"]) == (11, 6, 5)
    assert o["retained_pct"] == pytest.approx(54.5)  # 6 / 11, not 6 / 5 or 5 / 11


def test_all_retained_and_none_retained() -> None:
    everyone = audit_at(0.30)
    assert everyone["overall"]["dropped"] == 0 and everyone["overall"]["retained_pct"] == 100.0
    assert everyone["drop_causes"]["flag_counts"] == {}
    assert everyone["participants"]["all"]["two_hands_lose_none"] == 5
    nobody = audit_at(1.0)
    assert nobody["overall"]["retained"] == 2  # only the two clips at exactly 1.0
    spec = [(c, 0.5, "", r, m, u) for c, _, _, r, m, u in SPEC]
    none = audit_at(0.9, spec)
    assert none["overall"]["retained"] == 0 and none["overall"]["retained_pct"] == 0.0
    assert none["participants"]["all"]["two_hands_lose_both"] == 5
    assert none["participants"]["all"]["one_hand_lost"] == 1


# --------------------------------------------------------------------------- strata


def test_each_stratum_uses_its_own_column() -> None:
    by = audit_at(0.95)["by"]
    assert by["group"]["CONTROL"] == {
        "clips": 5, "retained": 3, "dropped": 2, "retained_pct": 60.0,
        "small_cell": False, "few_dropped": True,
    }  # fmt: skip
    assert by["group"]["ID"]["clips"] == 6 and by["group"]["ID"]["retained"] == 3
    assert by["resolution"]["1920x1080"]["clips"] == 3
    assert (
        by["orientation"]["landscape"]["clips"] == 3 and by["orientation"]["portrait"]["clips"] == 8
    )
    assert (
        by["stored_rotated"]["True"]["clips"] == 8 and by["stored_rotated"]["False"]["clips"] == 3
    )
    assert {k: v["clips"] for k, v in by["updrs"].items()} == {"0": 3, "1": 4, "2": 1, "3": 3}
    for column, levels in by.items():  # every stratum partitions the same 11 clips
        assert sum(v["clips"] for v in levels.values()) == 11, column
        assert sum(v["retained"] for v in levels.values()) == 6, column


def test_updrs_comes_from_the_labels_and_the_levels_are_correct() -> None:
    by = audit_at(0.95)["by"]["updrs"]
    assert by["3"]["retained"] == 1 and by["3"]["dropped"] == 2  # only ID9101_IZDA (0.96) stays
    assert by["2"]["retained"] == 0


def test_a_cell_under_five_clips_is_marked_and_few_dropped_is_marked_separately() -> None:
    by = audit_at(0.95)["by"]
    assert by["updrs"]["2"]["small_cell"] is True and by["updrs"]["2"]["clips"] == 1
    assert by["updrs"]["0"]["small_cell"] is True  # 3 clips
    assert by["group"]["ID"]["small_cell"] is False  # 6 clips, exactly 5 or more is fine
    five = [(f"CONTROL90{i:02d}_DCHA", 1.0, "", "1080x1920", 90, 0) for i in range(5)]
    assert audit_at(0.9, five)["by"]["group"]["CONTROL"]["small_cell"] is False  # exactly 5
    assert audit_at(0.9, five[:4])["by"]["group"]["CONTROL"]["small_cell"] is True  # 4
    assert by["group"]["ID"]["few_dropped"] is True and by["group"]["ID"]["dropped"] == 3
    assert audit_at(0.3)["by"]["group"]["ID"]["few_dropped"] is False  # nothing dropped


# --------------------------------------------------------------------------- causes and participants


def test_drop_causes_list_the_flags_of_the_dropped_clips() -> None:
    causes = audit_at(0.90)["drop_causes"]
    assert causes["dropped_clips"] == 4
    assert causes["flag_counts"] == {"gaps_not_fillable": 1, "low_detection": 4}
    assert causes["longest_missing_run_of_dropped"] == sorted(
        causes["longest_missing_run_of_dropped"]
    )
    rate_only = audit_at(0.98)["drop_causes"]["flag_counts"]
    assert rate_only["no_flag_rate_only"] == 3  # 0.97, 0.95, 0.90 carry no flag; 0.96 has one
    assert rate_only["scale_tap_coupling"] == 1


def test_a_failed_clip_without_a_rate_is_reported_and_never_counted_as_zero() -> None:
    m, lab = manifest(), labels()
    m.loc[0, "detection_rate"] = np.nan
    m.loc[0, "status"] = "failed"
    f = da.build_log(m, lab)
    result = da.audit(f, [0.5])
    entry = result["by_threshold"]["0.5"]
    assert entry["drop_causes"]["flag_counts"]["no_rate_failed_clip"] == 1
    assert result["rate_distribution"]["no_rate"] == 1
    assert result["rate_distribution"]["min"] == 0.4  # the NaN did not become the minimum


def test_participant_view_counts_hands_lost() -> None:
    p = audit_at(0.96)["participants"]
    # CONTROL9001: 1.00 + 0.97 keep both; CONTROL9002: 0.90 lost, 1.00 kept; CONTROL9003: one hand 0.80 lost
    assert p["CONTROL"] == {
        "participants": 3, "with_two_hands": 2, "two_hands_lose_none": 1,
        "two_hands_lose_one": 1, "two_hands_lose_both": 0,
        "with_one_hand": 1, "one_hand_kept": 0, "one_hand_lost": 1,
    }  # fmt: skip
    # ID9101: 0.50 lost, 0.96 kept; ID9102: 0.95 lost, 0.99 kept; ID9103: both lost
    assert p["ID"]["two_hands_lose_one"] == 2 and p["ID"]["two_hands_lose_both"] == 1
    assert p["all"]["participants"] == 6 and p["all"]["with_two_hands"] == 5


# --------------------------------------------------------------------------- inputs and outputs


def test_manifest_and_labels_must_hold_the_same_clips() -> None:
    lab = labels().iloc[:-1]
    with pytest.raises(da.AuditError, match="1 clips only in the manifest"):
        da.build_log(manifest(), lab)
    extra = pd.concat([labels(), pd.DataFrame({"ID": ["ID9199_DCHA"], "UPDRS": [0]})])
    with pytest.raises(da.AuditError, match="1 only in the labels"):
        da.build_log(manifest(), extra)
    both = manifest().iloc[1:]
    with pytest.raises(da.AuditError) as err:
        da.build_log(both, lab)
    assert "CONTROL" not in str(err.value) and "ID9" not in str(err.value)  # hashes only


def test_the_log_has_hashes_only_and_no_updrs() -> None:
    log = da.detection_log(frame())
    assert list(log.columns) == list(da.LOG_COLUMNS) and len(log) == 11
    assert "updrs" not in log.columns and "clip_id" not in log.columns
    text = log.to_csv(index=False)
    assert not any(raw in text for raw in RAW_PARTS)
    assert log["flags"].tolist().count("") == 5  # empty flags are empty strings, not "nan"
    assert "nan" not in text.lower().replace("scale_tap", "")


def test_the_aggregate_check_refuses_a_raw_id_and_accepts_normal_output() -> None:
    da.check_no_raw_ids(json.dumps(da.audit(frame())))
    with pytest.raises(da.AuditError):
        da.check_no_raw_ids('{"x": "CONTROL9001_DCHA"}')
    with pytest.raises(da.AuditError):
        da.check_no_raw_ids('{"x": "ID9104"}')


def test_bad_thresholds_are_rejected() -> None:
    for bad in ([0.0], [1.5], [float("nan")], [-0.1]):
        with pytest.raises(da.AuditError):
            da.audit(frame(), bad)
    assert da.parse_thresholds("0.9, 0.8,0.9") == (0.8, 0.9)
    with pytest.raises(da.AuditError):
        da.parse_thresholds("high")


# --------------------------------------------------------------------------- command line


def write_inputs(tmp_path: Path) -> dict[str, Path]:
    raw = tmp_path / "raw"
    interim = tmp_path / "interim"
    raw.mkdir()
    interim.mkdir()
    m = interim / "tracking_manifest.csv"
    manifest().to_csv(m, index=False)
    lab = raw / "fis_diagnostic.csv"
    labels().to_csv(lab, index=False)
    return {"manifest": m, "labels": lab, "log": interim / "detection_log.csv",
            "json": tmp_path / "docs" / "detection_audit.json", "interim": interim}  # fmt: skip


def run(box: dict[str, Path], *extra: str) -> int:
    return da.main(
        [
            "--manifest", str(box["manifest"]),
            "--labels", str(box["labels"]),
            "--log", str(box["log"]),
            "--json", str(box["json"]),
            "--interim-root", str(box["interim"]),
            *extra,
        ]
    )  # fmt: skip


def test_cli_writes_the_log_and_an_aggregate_json_without_raw_ids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    box = write_inputs(tmp_path)
    assert run(box) == 0
    out = capsys.readouterr().out
    data = json.loads(box["json"].read_text())
    assert set(data["by_threshold"]) == {"0.8", "0.9", "0.95"} and data["clips"] == 11
    assert "still to be decided with the supervisor" in out and "NOT chosen" in out
    text = out + box["json"].read_text() + box["log"].read_text()
    assert not any(raw in text for raw in RAW_PARTS)
    assert len(pd.read_csv(box["log"])) == 11
    assert "retained" in out and "* = small cell" in out


def test_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    box = write_inputs(tmp_path)
    assert run(box, "--thresholds", "nonsense") == 1
    labels().iloc[:-1].to_csv(box["labels"], index=False)
    assert run(box) == 1 and "differ" in capsys.readouterr().err
    assert not box["log"].exists() and not box["json"].exists()  # nothing written on failure
    box["labels"].unlink()
    assert run(box) == 2
    (tmp_path / "again").mkdir()
    box2 = write_inputs(tmp_path / "again")
    assert da.main(
        ["--manifest", str(box2["manifest"]), "--labels", str(box2["labels"]),
         "--log", str(tmp_path / "elsewhere.csv"), "--json", str(box2["json"]),
         "--interim-root", str(box2["interim"])]
    ) == 2  # fmt: skip
    assert not (tmp_path / "elsewhere.csv").exists()


def test_an_empty_labels_file_or_manifest_is_a_clean_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for which in ("labels", "manifest"):
        (tmp_path / which).mkdir()
        box = write_inputs(tmp_path / which)
        box[which].write_text("")  # zero bytes: pandas has no columns to parse
        assert run(box) == 1
        err = capsys.readouterr().err
        assert err.startswith("FAILED:") and "Traceback" not in err
        assert not any(raw in err for raw in RAW_PARTS)
        assert not box["log"].exists() and not box["json"].exists()  # nothing written
