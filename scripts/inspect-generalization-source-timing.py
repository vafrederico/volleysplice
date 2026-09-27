#!/usr/bin/env python3
"""Inspect source presentation gaps without rally labels or learned outputs."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import neural_generalization_feature_stage as stage
from analysis.neural_context_development import identity, read, write_immutable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id', required=True)
    args = parser.parse_args()
    root = stage.environment(stage.ROOT)
    plan = stage.verify_plan(root/'stage-plan.json')
    source = next(r for r in plan['records'] if r['id'] == args.id)
    pts, inventory = stage.mobile.presentation_inventory(source['videoIdentity']['path'])
    duration = float(inventory['stream']['duration'])
    times = np.arange(0., duration, .25, dtype=np.float64)
    right = np.clip(np.searchsorted(pts, times), 0, len(pts)-1)
    left = np.maximum(right-1, 0)
    selected = np.where(abs(pts[left]-times) <= abs(pts[right]-times), left, right)
    errors = pts[selected]-times
    bad = np.flatnonzero(abs(errors) > .125+1e-9)
    gaps = np.flatnonzero(np.diff(pts) > .25+1e-9)
    folder = root/'timing-diagnostics-v1'/source['id']
    folder.mkdir(parents=True, exist_ok=True)
    arrays = folder/'selection.npz'
    stage.npz_new(arrays, pts=pts, times=times, ordinals=selected, errors=errors)
    report = {'source': identity(__file__), 'stagePlan': identity(root/'stage-plan.json'),
              'id': source['id'], 'video': source['videoIdentity'], 'inventory': inventory,
              'arrays': identity(arrays), 'duration': duration, 'frameCount': len(pts),
              'firstPts': float(pts[0]), 'lastPts': float(pts[-1]),
              'maximumSelectionErrorSeconds': float(abs(errors).max()),
              'badTicks': [{'time': float(times[i]), 'ordinal': int(selected[i]), 'pts': float(pts[selected[i]]),
                            'errorSeconds': float(errors[i])} for i in bad],
              'longGaps': [{'leftOrdinal': int(i), 'leftPts': float(pts[i]), 'rightPts': float(pts[i+1]),
                            'gapSeconds': float(pts[i+1]-pts[i])} for i in gaps],
              'labelsUsed': False, 'outputsUsed': False}
    write_immutable(folder/'report.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'inventory'}), flush=True)


if __name__ == '__main__':
    main()
