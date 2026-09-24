#!/usr/bin/env python3
"""Write a new immutable inventory; refuses to replace an existing output."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.private_ledger import private_value
from analysis.source_exposure_inventory import build_inventory, identity

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", type=Path, default=Path(private_value('private-reference-0059')))
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--filename", default="inventory.json")
args = parser.parse_args()
if Path(args.filename).name != args.filename:
    raise ValueError("filename must be a basename")
target = args.output_root / args.filename
if target.exists():
    raise FileExistsError(target)
result = build_inventory(args.data_root, Path(__file__).resolve().parents[1], args.output_root)
result["wrapper"] = identity(Path(__file__).resolve())
args.output_root.mkdir(parents=True, exist_ok=True)
with target.open("x", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2, allow_nan=False)
    handle.write("\n")
print(json.dumps({"inventory": identity(target), "counts": {k: v["recordings"] for k, v in result["summary"].items() if isinstance(v, dict) and "recordings" in v}}, indent=2))
