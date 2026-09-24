#!/usr/bin/env python3
"""Rebind an audited owner's raw all42 inference archives to a duplicate recipe."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.neural_generalization_reuse import materialize_inference

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--precision', choices=('fp32', 'fp16', 'int8'), default='fp32')
    args = parser.parse_args()
    result = materialize_inference(args.plan, args.task, args.output, args.panel, args.precision)
    print(json.dumps({'copiedRecordingCount': len(result['scoreCopies']), 'precision': result['precision']}))
