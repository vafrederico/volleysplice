#!/usr/bin/env python3
"""Register, fit, or summarize the declared temporal serving-side experiment."""
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import serving_side_temporal as study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "fit", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.phase == "prepare":
        parser.error("--inputs is required for prepare") if args.inputs is None else None
        print(study.prepare(args.inputs, args.root, Path(__file__).resolve().parents[1]))
    elif args.phase == "fit":
        study.run(args.root, args.device)
    else:
        parser.error("--report is required for summarize") if args.report is None else None
        study.summarize(args.root, args.report, args.device)


if __name__ == "__main__":
    main()
