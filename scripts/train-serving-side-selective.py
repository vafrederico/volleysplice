#!/usr/bin/env python3
"""Run the registered precision-first serving-side study."""
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import serving_side_selective as study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "fit"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.phase == "prepare":
        print(study.prepare(Path(__file__).resolve().parents[1]), flush=True)
    else:
        study.run(args.device)


if __name__ == "__main__":
    main()
