#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.neural_generalization_report import build_report

p = argparse.ArgumentParser(description='Build explicit-scope source-composition report data')
p.add_argument('--index', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--audit', type=Path)
a = p.parse_args()
print(json.dumps(build_report(a.index, a.output, a.audit)))
