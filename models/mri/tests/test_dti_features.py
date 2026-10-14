"""Tests for models/mri/src/dti_features.py, on tiny synthetic maps (no real scans).

Each guard is shown to fail on bad input as well as pass on good input.
"""

from __future__ import annotations

from pathlib import Path

import dti_features as f
import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from progress import RunRecorder, read_events
from t1_features import REGIONS, pseudonym

SHAPE = (40, 30, 30)
AFFINE = np.eye(4)


def save(path: Path, data: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), AFFINE), str(path))


def write_label_atlas(
    tmp: Path, name: str, labels: list[str], kind: str, first_index: int, z0: int
) -> tuple[Path, Path]:
    """Cubes of 3x3x3 voxels, one per label, with the value FSL would use for this kind."""
    data = np.zeros(SHAPE)
    xml = [f"<atlas><header><type>{kind}</type></header><data>"]
    for i, lab in enumerate(labels):
        index = first_index + i
        value = index + 1 if kind == "Probabilistic" else index
        x = 1 + (i % 9) * 4
        y = 1 + (i // 9) * 4
        data[x : x + 3, y : y + 3, z0 : z0 + 3] = value
        xml.append(f'<label index="{index}" x="0" y="0" z="0">{lab}</label>')
    xml.append("</data></atlas>")
    save(tmp / f"{name}.nii.gz", data)
    (tmp / f"{name}.xml").write_text("".join(xml))
    return tmp / f"{name}.nii.gz", tmp / f"{name}.xml"


@pytest.fixture
def atlas_files(tmp_path: Path):
    # JHU-style: a label atlas whose index 0 is "Unclassified" and values equal the index
    jhu = write_label_atlas(tmp_path, "jhu", ["Unclassified", *f.TRACTS.values()], "Label", 0, z0=2)
    ho = write_label_atlas(tmp_path, "ho", list(REGIONS.values()), "Probabilistic", 0, z0=10)
    return jhu, ho


@pytest.fixture
def atlases(atlas_files) -> f.DtiAtlases:
    (jhu, jhu_xml), (ho, ho_xml) = atlas_files
    return f.load_atlases(jhu, jhu_xml, ho, ho_xml)


def write_subject(tmp: Path, sid: str = "S0001", fa: float = 0.45, md: float = 0.0008) -> Path:
    d = tmp / sid
    d.mkdir(parents=True)
    save(d / f"{sid}_FA_mni.nii.gz", np.full(SHAPE, fa))
    save(d / f"{sid}_MD_mni.nii.gz", np.full(SHAPE, md))
    return d


def test_feature_names() -> None:
    names = f.feature_names()
    assert len(names) == 32 == len(set(names))
    assert names[0] == "dti_CC_genu_fa_mean" and "dti_L_putamen_md_mean" in names


# --- atlas label values -----------------------------------------------------------


def test_label_atlas_uses_index_and_probabilistic_uses_index_plus_one(atlas_files) -> None:
    (_, jhu_xml), (_, ho_xml) = atlas_files
    assert f.label_values(jhu_xml)["Genu of corpus callosum"] == 1
    assert f.label_values(ho_xml)["Left Caudate"] == 1  # index 0 + 1


def test_tract_masks_hit_the_right_cubes(atlases: f.DtiAtlases) -> None:
    # "Genu" is the second label (index 1) -> cube 1 in the synthetic atlas
    mask = atlases.tracts.masks["CC_genu"]
    assert mask.sum() == 27 and mask[5, 1, 2]
    assert not atlases.tracts.masks["CC_body"][5, 1, 2]


def test_missing_tract_name_fails(tmp_path: Path) -> None:
    jhu, jhu_xml = write_label_atlas(tmp_path, "jhu", ["Unclassified"], "Label", 0, z0=2)
    with pytest.raises(f.FeatureError, match="Genu of corpus callosum"):
        f.load_label_atlas(jhu, jhu_xml, f.TRACTS)


def test_atlases_on_different_grids_fail(tmp_path: Path, atlas_files) -> None:
    (jhu, jhu_xml), (_, ho_xml) = atlas_files
    nib.save(
        nib.Nifti1Image(np.zeros((10, 10, 10), np.float32), AFFINE), str(tmp_path / "x.nii.gz")
    )
    with pytest.raises(f.FeatureError):
        f.load_atlases(jhu, jhu_xml, tmp_path / "x.nii.gz", ho_xml)


# --- one person -------------------------------------------------------------------


def test_extract_values_and_md_units(tmp_path: Path, atlases: f.DtiAtlases) -> None:
    r = f.extract_subject(write_subject(tmp_path), "S0001", atlases)
    assert r.status == "ok", r.reason
    assert set(r.features) == set(f.feature_names())
    assert r.features["dti_L_plic_fa_mean"] == pytest.approx(0.45)
    assert r.features["dti_L_plic_md_mean"] == pytest.approx(0.8)  # 10^-3 mm^2/s
    assert r.features["dti_R_thalamus_md_mean"] == pytest.approx(0.8)
    assert r.features["dti_wm_fa_median"] == pytest.approx(0.45)


def test_uncovered_voxels_are_left_out(tmp_path: Path, atlases: f.DtiAtlases) -> None:
    d = write_subject(tmp_path)
    fa = np.full(SHAPE, 0.45)
    genu = atlases.tracts.masks["CC_genu"]
    idx = np.argwhere(genu)[:9]  # a third of the region outside the scan
    fa[tuple(idx.T)] = 0.0
    save(d / "S0001_FA_mni.nii.gz", fa)
    r = f.extract_subject(d, "S0001", atlases)
    assert r.status == "ok" and r.features["dti_CC_genu_fa_mean"] == pytest.approx(0.45)


def test_mostly_uncovered_region_fails(tmp_path: Path, atlases: f.DtiAtlases) -> None:
    d = write_subject(tmp_path)
    fa = np.full(SHAPE, 0.45)
    fa[atlases.tracts.masks["L_cst"]] = 0.0
    save(d / "S0001_FA_mni.nii.gz", fa)
    r = f.extract_subject(d, "S0001", atlases)
    assert r.status == "failed" and "L_cst only 0% covered" in r.reason


def test_missing_map_gives_failed_result(tmp_path: Path, atlases: f.DtiAtlases) -> None:
    d = write_subject(tmp_path)
    (d / "S0001_MD_mni.nii.gz").unlink()
    r = f.extract_subject(d, "S0001", atlases)
    assert r.status == "failed" and "MD_mni" in r.reason and r.features == {}


def test_wrong_grid_gives_failed_result(tmp_path: Path, atlases: f.DtiAtlases) -> None:
    d = tmp_path / "S0001"
    d.mkdir()
    for m in ("FA", "MD"):
        save(d / f"S0001_{m}_mni.nii.gz", np.full((20, 20, 20), 0.4))
    r = f.extract_subject(d, "S0001", atlases)
    assert r.status == "failed" and "shape" in r.reason


def test_progress_events(tmp_path: Path, atlases: f.DtiAtlases) -> None:
    rec = RunRecorder("dtifeatures", ["S0001"], f.FEATURE_STEPS, runs_dir=tmp_path / "runs")
    f.extract_subject(write_subject(tmp_path), "S0001", atlases, recorder=rec)
    ends = [e["step"] for e in read_events(rec.path) if e["type"] == "step_end"]
    assert ends == f.FEATURE_STEPS
    assert "S0001" not in rec.path.read_text()


# --- inputs and table ---------------------------------------------------------------


def test_finished_people_later_table_wins(tmp_path: Path) -> None:
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    pd.DataFrame(
        {"subject_id": ["1", "2"], "image_id": ["10", "20"], "status": ["ok", "failed"]}
    ).to_csv(a, index=False)
    pd.DataFrame(
        {"subject_id": ["1", "2"], "image_id": ["11", "21"], "status": ["ok", "ok"]}
    ).to_csv(b, index=False)
    assert f.finished_people([a, b]) == {"1": "11", "2": "21"}
    assert f.finished_people([a]) == {"1": "10"}  # a failed person is not "finished"


def test_scanner_comes_from_the_series_actually_used(tmp_path: Path) -> None:
    conv = tmp_path / "dti_x_conversion.csv"
    pd.DataFrame(
        {
            "subject_id": ["1", "1"],
            "image_id": ["10", "11"],
            "model": ["Prisma", "Verio"],
            "manufacturer": ["SIEMENS", "SIEMENS"],
            "site_key": ["", ""],
        }
    ).to_csv(conv, index=False)
    meta = f.scanner_metadata([conv], {"1": "11"})
    assert meta[pseudonym("1")]["scanner_model"] == "Verio"


def test_table_has_pseudonyms_and_rejects_duplicates(tmp_path: Path, atlases) -> None:
    r = f.extract_subject(write_subject(tmp_path), "S0001", atlases)
    table = f.build_table([r], {})
    assert table.loc[0, "subject"] == pseudonym("S0001") and "S0001" not in table.to_csv()
    with pytest.raises(f.FeatureError, match="twice"):
        f.build_table([r, r], {})
