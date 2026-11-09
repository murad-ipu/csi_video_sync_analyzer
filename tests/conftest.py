"""Make the project and the ESPectre clone importable from the tests."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import espectre_path  # noqa: E402

espectre_path.setup()
