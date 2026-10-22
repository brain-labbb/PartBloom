"""External resources; no imports from the former experimental implementation."""
import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
PIPELINE = PACKAGE / "pipeline"
PROJECT = PACKAGE.parent
VENDOR = Path(os.environ.get("PARTBLOOM_VENDOR", str(PROJECT / "vendor"))).resolve()
WEIGHTS_ROOT = Path(os.environ.get("PARTBLOOM_WEIGHTS", str(PROJECT / "weights"))).resolve()
