"""Find the ESPectre clone and make tools/lib/csi_io.py importable.

Nothing ESPectre-specific is copied into this project: the V8 parser and the
Direct HTTP client are always imported from the clone.
"""

import os
import sys
from pathlib import Path

# Default layout: ...\ESPectre\csi_video_sync next to ...\ESPectre\espectre
DEFAULT_REPO = Path(__file__).resolve().parent.parent / "espectre"


def setup(repo=None) -> Path:
    """Put the clone on sys.path. Order: argument, ESPECTRE_REPO, ../espectre."""
    root = Path(repo or os.environ.get("ESPECTRE_REPO") or DEFAULT_REPO).resolve()
    if not (root / "tools" / "lib" / "csi_io.py").is_file():
        raise SystemExit(
            f"ESPectre repository not found at {root}.\n"
            "Pass --repo <path to espectre clone> or set ESPECTRE_REPO."
        )
    # The root serves `tools.lib...` and `src.python...`; src/python serves
    # `micro_espectre`, which ESPectre's CLI modules import as a top-level package.
    for path in (str(root / "src" / "python"), str(root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return root
