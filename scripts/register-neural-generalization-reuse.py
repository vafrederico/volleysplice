#!/usr/bin/env python3
"""Freeze first-owner mappings before executing duplicate logical training tasks."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.neural_generalization_reuse import register

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration-dir', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--qualification', type=Path, required=True)
    args = parser.parse_args()
    result = register(args.registration_dir, args.protocol, args.qualification)
    print(json.dumps(result['counts']))
