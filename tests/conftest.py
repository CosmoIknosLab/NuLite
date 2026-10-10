# -*- coding: utf-8 -*-
# Make the repository root importable for the tests (run with `python -m pytest tests` from the repo root)
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
