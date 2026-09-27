#!/usr/bin/env python3
"""Run one frozen generalization fit or explicitly scoped inference task."""
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.neural_generalization_experiment import main

if __name__ == '__main__':
    main()
