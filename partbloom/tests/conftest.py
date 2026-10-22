import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PACKAGE), str(PACKAGE / "pipeline")]
