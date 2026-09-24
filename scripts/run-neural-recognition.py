#!/usr/bin/env python3
"""Launch only the explicitly selected registered recognition arm."""
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.neural_recognition_development import main

if __name__ == '__main__':
    main()
