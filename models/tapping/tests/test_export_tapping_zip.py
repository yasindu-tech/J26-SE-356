"""Tests for scripts/export_models.py with a tapping bundle: the zip unpacks to a bundle that
loads, and the exporter refuses a changed file, an unlisted file, a data table and
person-level keys. Synthetic world, as in test_tapping_bundle."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent / "demo"))
sys.path.insert(0, str(HERE))

import tapping_bundle as tb  # noqa: E402
from test_tapping_bundle import COMMIT, T0, processed, rehash  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "export_models", HERE.parents[2] / "scripts" / "export_models.py"
)
assert spec and spec.loader
em = importlib.util.module_from_spec(spec)
spec.loader.exec_module(em)


def bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    models = tmp_path / "data" / "models"
    monkeypatch.setattr(em, "MODELS", models)
    return tb.build(processed(tmp_path, monkeypatch), models / "tapping", now=T0, commit=COMMIT)


def test_the_zip_unpacks_to_a_bundle_that_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = bundle(tmp_path, monkeypatch)
    zip_path = em.export("tapping", None, tmp_path / "out")
    assert zip_path.name == f"pdxai_models_tapping_{folder.name}.zip"
    with zipfile.ZipFile(zip_path) as z:
        top = json.loads(z.read("MANIFEST.json"))
        assert set(z.namelist()) - {"MANIFEST.json"} == set(top["files"])
        for name, digest in top["files"].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
        assert all(n.startswith("data/models/tapping/") or n in {"README.txt", "MANIFEST.json"}
                   for n in z.namelist())  # fmt: skip
        assert not [n for n in z.namelist() if n.endswith((".csv", ".parquet", ".joblib", ".mp4"))]
        z.extractall(tmp_path / "teammate")
    b = tb.load(tmp_path / "teammate" / "data" / "models" / "tapping")
    assert b.version == folder.name and top["git_commit"] == COMMIT


def test_a_changed_file_is_not_exported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = bundle(tmp_path, monkeypatch)
    (folder / tb.REFERENCE).write_text("{}")
    with pytest.raises(SystemExit, match="does not match its manifest hash"):
        em.export("tapping", None, tmp_path / "out")
    assert not list(tmp_path.rglob("*.zip"))


def test_an_unlisted_file_is_not_exported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = bundle(tmp_path, monkeypatch)
    (folder / "features.csv").write_text("ref,participant,rating\n0000000a,aaaaaaaa,3\n")
    with pytest.raises(SystemExit, match="files differ from the manifest"):
        em.export("tapping", None, tmp_path / "out")


def test_a_listed_data_table_is_not_exported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = bundle(tmp_path, monkeypatch)
    (folder / "features.csv").write_text("tap_rate_hz\n1.0\n")
    rehash(folder)
    with pytest.raises(SystemExit, match="data table, raw input or pickle"):
        em.export("tapping", None, tmp_path / "out")


def test_person_level_keys_are_not_exported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = bundle(tmp_path, monkeypatch)
    card = json.loads((folder / tb.CARD).read_text())
    card["examples"] = [{"participant": "aaaaaaaa"}]
    (folder / tb.CARD).write_text(json.dumps(card))
    rehash(folder)
    with pytest.raises(SystemExit, match="person-level keys"):
        em.export("tapping", None, tmp_path / "out")
