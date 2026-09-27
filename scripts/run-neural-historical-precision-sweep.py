#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.private_ledger import private_value
from analysis import neural_recall_sweep_precision as study

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=study.__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--historical', type=Path)
    parser.add_argument('--stage-plan', type=Path)
    args = parser.parse_args()
    study.sweep.require(str(args.output.resolve()).startswith(private_value('private-reference-0060')), 'Precision artifacts must use NAS')
    for name in ('TMPDIR', 'TMP', 'TEMP', 'TORCH_HOME', 'HF_HOME', 'XDG_CACHE_HOME', 'CUDA_CACHE_PATH'):
        path = args.output / 'runtime' / name.lower(); path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    if args.action == 'prepare':
        if not args.historical or not args.stage_plan:
            parser.error('Prepare requires historical study and exact cache stage plan')
        study.prepare(args.historical, args.stage_plan, args.output)
    else:
        study.run(args.output)
