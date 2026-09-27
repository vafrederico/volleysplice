#!/usr/bin/env python3
from pathlib import Path
import argparse
import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.private_ledger import private_value
from analysis import neural_recall_sweep_historical as study

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=study.__doc__)
    parser.add_argument('action', choices=('prepare', 'select', 'fit', 'finalize'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--model', choices=study.MODELS)
    parser.add_argument('--seed', type=int, choices=study.SEEDS)
    parser.add_argument('--outer', type=int, choices=range(4))
    args = parser.parse_args()
    study.sweep.require(str(args.output.resolve()).startswith(private_value('private-reference-0060')), 'Historical output must use NAS')
    for name in ('TMPDIR', 'TMP', 'TEMP', 'TORCH_HOME', 'HF_HOME', 'XDG_CACHE_HOME', 'CUDA_CACHE_PATH'):
        location = args.output / 'runtime' / name.lower()
        location.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(location)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    if args.action == 'prepare':
        study.prepare(args.output)
    elif args.action == 'fit':
        study.fit_refits(args.output, args.device, model=args.model, seed=args.seed, outer=args.outer)
    else:
        study.verify(args.output)
        if args.action == 'finalize':
            study.make_bundle(args.output, complete=True)
        suffix = 'complete' if args.action == 'finalize' else 'selection'
        study.load_script('evaluate-neural-recall-sweep.py').run_bundle(args.output / f'bundle-{suffix}.json', args.output / f'report-{suffix}.json')
