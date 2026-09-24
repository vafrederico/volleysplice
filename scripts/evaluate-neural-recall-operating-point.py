#!/usr/bin/env python3
"""CPU-only strict99% selection from saved neural study predictions."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.neural_recall_operating_point import main

if __name__ == "__main__":
    main()
