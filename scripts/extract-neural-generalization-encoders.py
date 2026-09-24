#!/usr/bin/env python3
"""Execute only the registered image-encoder feature phases."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import neural_generalization_feature_inference as encoders


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'qualify', 'gpu', 'int8', 'index-av', 'index-core', 'index-complete'))
    parser.add_argument('--id', action='append')
    parser.add_argument('--scope', choices=('all', 'fit'), default='all')
    args = parser.parse_args()
    if args.phase == 'register':
        result = encoders.register()
        print(json.dumps({'registered': True, 'kind': result['kind']}), flush=True)
    elif args.phase == 'qualify':
        result = encoders.qualify()
        print(json.dumps({'passed': result['passed'], 'kind': result['kind']}), flush=True)
    elif args.phase in ('gpu', 'int8'):
        encoders.run(args.phase, args.id)
    else:
        print(json.dumps(encoders.publish_index(args.phase.removeprefix('index-'), scope=args.scope)), flush=True)


if __name__ == '__main__':
    main()
