"""Moved to landerpi/detector.py (2026-10-12: navigation, manipulation and scene_understanding use it; repo rule).
This name stays so `import detector` / `from detector import Detector` keep working in all three projects."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from landerpi.detector import *  # noqa: E402,F401,F403  (torch loads first there, as before)
from landerpi.detector import MODELS, Detector, outline, use_models_dir  # noqa: E402,F401
