#!/usr/bin/env python3
"""Freeze calibration choices, or evaluate those choices on independent videos."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.neural_generalization_results import select_task, evaluate_task


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('select', 'evaluate'))
    p.add_argument('--task', type=Path, required=True)
    p.add_argument('--fit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--historical', type=Path)
    p.add_argument('--selection', type=Path)
    p.add_argument('--selection-audit', type=Path)
    p.add_argument('--panel', type=Path)
    p.add_argument('--inventory', type=Path)
    p.add_argument('--original-manifest', type=Path)
    p.add_argument('--precision', choices=('fp32', 'fp16', 'int8'), default='fp32')
    a = p.parse_args()
    if a.action == 'select':
        result = select_task(a.task, a.fit, a.output, a.historical)
    else:
        if not all((a.selection, a.selection_audit, a.panel, a.inventory, a.original_manifest)):
            p.error('Evaluation requires selection, selection-audit, panel, inventory and original-manifest')
        result = evaluate_task(a.task, a.fit, a.selection, a.panel, a.inventory, a.original_manifest, a.output, a.precision, a.selection_audit)
    print(json.dumps({'written': str(a.output), 'taskId': result['taskId']}))


if __name__ == '__main__':
    main()
