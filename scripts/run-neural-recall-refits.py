#!/usr/bin/env python3
"""Run explicitly authorized missing99% checkpoints or finalize their scores."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.neural_recall_refits import main

if __name__ == "__main__":
    main()
