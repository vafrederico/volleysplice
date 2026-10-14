#!/usr/bin/env python3
"""Register and train the frozen-protocol hybrid serving-side follow-up."""
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import serving_side_hybrid as study
from analysis.private_ledger import private_value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "fit", "summarize"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = Path(private_value(study.STUDY_INDEX))
    if args.phase == "prepare":
        print(study.prepare(Path(__file__).resolve().parents[1]), flush=True)
    elif args.phase == "fit":
        study.run(root, args.device)
    else:
        if args.report is None:
            parser.error("--report is required for summarize")
        study.summarize(root, args.report, args.device)


if __name__ == "__main__":
    main()
