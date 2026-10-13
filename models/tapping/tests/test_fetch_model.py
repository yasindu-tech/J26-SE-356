"""Tests for models/tapping/fetch_model.sh.

No network and no real model: the script is copied into a temporary folder next to its own
``fetch_model.sha256`` (it reads paths relative to itself) and pointed at a ``file://`` URL
through ``FETCH_MODEL_URL``. The content is made up. No conftest.py on purpose.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

TAPPING = Path(__file__).resolve().parents[1]
SCRIPT = TAPPING / "fetch_model.sh"
PIN_FILE = TAPPING / "fetch_model.sha256"
NAME = "hand_landmarker.task"
# The pin is the hash of the file this project has used, not a vendor checksum. Changing the
# pin must be a deliberate act: update fetch_model.sha256 and this constant together.
PINNED_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
GOOD = b"made-up model bytes, version one"
OTHER = b"made-up model bytes, version two"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def setup(tmp_path: Path, pinned: bytes = GOOD, source: bytes | None = GOOD) -> dict[str, Path]:
    """A sandbox: script + pin file for ``pinned``, and a source file with ``source``."""
    home = tmp_path / "tapping"
    home.mkdir()
    shutil.copy(SCRIPT, home / "fetch_model.sh")
    (home / "fetch_model.sha256").write_text(f"{sha(pinned)}  {NAME}\n")
    src = tmp_path / "source.bin"
    if source is not None:
        src.write_bytes(source)
    return {"home": home, "src": src, "dest": home / "artifacts" / NAME}


def run(
    box: dict[str, Path], cwd: Path | None = None, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    full = {**os.environ, "FETCH_MODEL_URL": box["src"].resolve().as_uri(), **(env or {})}
    return subprocess.run(
        ["bash", str(box["home"] / "fetch_model.sh")],
        cwd=cwd or box["home"],
        env=full,
        capture_output=True,
        text=True,
        timeout=60,
    )


def leftovers(box: dict[str, Path]) -> list[str]:
    folder = box["home"] / "artifacts"
    return sorted(p.name for p in folder.iterdir()) if folder.exists() else []


def test_the_real_pin_file_is_a_single_valid_sha256_line() -> None:
    lines = PIN_FILE.read_text().splitlines()
    assert len(lines) == 1
    digest, name = lines[0].split()
    assert len(digest) == 64 and set(digest) <= set("0123456789abcdef") and name == NAME


def test_the_pin_file_holds_the_deliberately_pinned_hash() -> None:
    assert PIN_FILE.read_text().split()[0] == PINNED_SHA256


def test_a_model_already_on_disk_matches_the_pin() -> None:
    model = TAPPING / "artifacts" / NAME
    if not model.is_file():
        pytest.skip("the model is not downloaded here (run models/tapping/fetch_model.sh)")
    assert sha(model.read_bytes()) == PINNED_SHA256


def test_the_real_script_is_executable_and_has_a_pinned_https_url() -> None:
    assert os.access(SCRIPT, os.X_OK)
    text = SCRIPT.read_text()
    assert 'MODEL_URL="https://storage.googleapis.com/mediapipe-models/' in text
    assert "/float16/1/" in text  # the model version is pinned in the URL


def test_good_download_installs_the_verified_file(tmp_path: Path) -> None:
    box = setup(tmp_path)
    result = run(box)
    assert result.returncode == 0, result.stderr
    assert box["dest"].read_bytes() == GOOD
    assert sha(GOOD) in result.stdout and leftovers(box) == [NAME]


def test_wrong_checksum_is_rejected_and_nothing_is_left_behind(tmp_path: Path) -> None:
    box = setup(tmp_path, pinned=GOOD, source=OTHER)
    result = run(box)
    assert result.returncode != 0
    assert "checksum mismatch" in result.stderr
    assert sha(GOOD) in result.stderr and sha(OTHER) in result.stderr  # expected and actual
    assert leftovers(box) == []  # no final file, no temporary file


def test_failed_download_leaves_no_temporary_file(tmp_path: Path) -> None:
    box = setup(tmp_path, source=None)  # the file:// URL points at nothing
    result = run(box)
    assert result.returncode != 0 and "download failed" in result.stderr
    assert leftovers(box) == []


def test_a_failed_download_keeps_the_previous_valid_file(tmp_path: Path) -> None:
    box = setup(tmp_path, source=OTHER)  # download will not verify
    box["dest"].parent.mkdir()
    box["dest"].write_bytes(b"corrupt")  # existing, wrong
    assert run(box).returncode != 0
    assert box["dest"].read_bytes() == b"corrupt"  # replaced only after a verified download
    box["src"].unlink()
    assert run(box).returncode != 0
    assert box["dest"].read_bytes() == b"corrupt" and leftovers(box) == [NAME]


def test_existing_valid_file_is_reused_without_any_download(tmp_path: Path) -> None:
    box = setup(tmp_path, source=None)  # a download attempt would fail
    box["dest"].parent.mkdir()
    box["dest"].write_bytes(GOOD)
    before = box["dest"].stat().st_mtime_ns
    result = run(box)
    assert result.returncode == 0 and "nothing to download" in result.stdout
    assert box["dest"].stat().st_mtime_ns == before and leftovers(box) == [NAME]


def test_existing_corrupt_file_is_reported_and_replaced(tmp_path: Path) -> None:
    box = setup(tmp_path)
    box["dest"].parent.mkdir()
    box["dest"].write_bytes(b"corrupt")
    result = run(box)
    assert result.returncode == 0
    assert "does not match the pin" in result.stderr
    assert box["dest"].read_bytes() == GOOD and leftovers(box) == [NAME]


def test_it_works_from_any_working_directory(tmp_path: Path) -> None:
    box = setup(tmp_path)
    elsewhere = tmp_path / "somewhere" / "else"
    elsewhere.mkdir(parents=True)
    assert run(box, cwd=elsewhere).returncode == 0
    assert box["dest"].read_bytes() == GOOD  # next to the script, not in the working directory
    assert list(elsewhere.iterdir()) == []


@pytest.mark.parametrize("pin", ["", "not-a-hash  x\n", "ABCDEF" * 11 + "\n", "ab12  x\n"])
def test_a_bad_pin_file_is_refused(tmp_path: Path, pin: str) -> None:
    box = setup(tmp_path)
    (box["home"] / "fetch_model.sha256").write_text(pin)
    result = run(box)
    assert result.returncode != 0 and "pin file" in result.stderr
    assert leftovers(box) == []


def test_missing_pin_file_is_refused(tmp_path: Path) -> None:
    box = setup(tmp_path)
    (box["home"] / "fetch_model.sha256").unlink()
    assert run(box).returncode != 0 and leftovers(box) == []


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_an_interrupted_download_leaves_no_temporary_file(tmp_path: Path, sig: int) -> None:
    box = setup(tmp_path, source=None)
    fake = tmp_path / "bin"
    fake.mkdir()
    curl = fake / "curl"
    curl.write_text(  # a curl that writes half a file to the -o target, then stalls
        "#!/usr/bin/env bash\n"
        'while [ $# -gt 0 ]; do [ "$1" = "-o" ] && out="$2"; shift; done\n'
        'printf partial > "$out"\n'
        "sleep 30\n"
    )
    curl.chmod(0o755)
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}", "FETCH_MODEL_URL": "file:///x"}
    proc = subprocess.Popen(
        ["bash", str(box["home"] / "fetch_model.sh")],
        env=env,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    folder = box["home"] / "artifacts"
    for _ in range(100):  # wait until the partial file exists
        if folder.exists() and any(p.read_bytes() == b"partial" for p in folder.iterdir()):
            break
        time.sleep(0.05)
    else:
        os.killpg(proc.pid, signal.SIGKILL)
        pytest.fail("the fake download never started")
    os.killpg(proc.pid, sig)
    proc.communicate(timeout=30)
    assert proc.returncode != 0
    assert leftovers(box) == []  # no partial file at the final path, none left over


def test_shellcheck_if_installed() -> None:
    if shutil.which("shellcheck") is None:
        pytest.skip("shellcheck is not installed")
    assert subprocess.run(["shellcheck", str(SCRIPT)], capture_output=True).returncode == 0
