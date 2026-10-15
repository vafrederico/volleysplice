#!/usr/bin/env python3
"""Select and audit review policies without exposing any outer labels to fitting."""
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.serving_side_selective_eval import evaluate_and_refit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    evaluate_and_refit(args.report, args.device)
