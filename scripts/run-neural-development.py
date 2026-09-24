#!/usr/bin/env python3
"""Entry point for the current-metric compact neural development study."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.neural_development import main

if __name__ == "__main__":
    raise SystemExit(main())
