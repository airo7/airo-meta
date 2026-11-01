import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("AIRO_DATA") or ROOT / "data")
WEB = ROOT / "web"
