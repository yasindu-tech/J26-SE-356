import sys
from pathlib import Path

# The gait modules live in models/gait/src (a scripts folder, not a package).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
