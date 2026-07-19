"""Ensure the repository root is importable as `src` during test collection.

Placing this at the repo root makes pytest add the root to ``sys.path``, so
``from src import ...`` resolves no matter the invocation directory.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
