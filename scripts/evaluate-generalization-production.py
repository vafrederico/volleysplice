#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.neural_generalization_results import evaluate_production

p = argparse.ArgumentParser(description='Score fixed production on the identical independent label policies')
p.add_argument('--production-dir', type=Path, required=True)
p.add_argument('--parity-audit', type=Path, required=True)
p.add_argument('--panel', type=Path, required=True)
p.add_argument('--inventory', type=Path, required=True)
p.add_argument('--original-manifest', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
evaluate_production(a.production_dir, a.parity_audit, a.panel, a.inventory, a.original_manifest, a.output)
print(str(a.output))
