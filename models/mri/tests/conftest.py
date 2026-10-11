"""Make the scripts in models/mri/src importable by name (e.g. ``import progress``)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
