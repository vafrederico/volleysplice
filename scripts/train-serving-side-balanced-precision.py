#!/usr/bin/env python3
"""Register or train the matched Balanced-encoder serving-side candidate."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import serving_side_balanced_precision as study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.action == "prepare":
        print(json.dumps(study.prepare(Path(__file__).resolve().parents[1])), flush=True)
    else:
        if args.report is None:
            parser.error("run requires --report")
        study.run(args.report, args.device)


if __name__ == "__main__":
    main()
