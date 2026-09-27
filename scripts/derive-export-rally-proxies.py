#!/usr/bin/env python3
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import export_rally_proxy_inventory as proxy
from analysis.source_exposure_inventory import identity, read, sha256

parser = argparse.ArgumentParser()
parser.add_argument("--inventory", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
base = read(args.inventory)
rows = []
for row in base["records"]:
    if row["sourceGroup"] not in proxy.ALLOWED_GROUPS:
        continue
    feedback_path = Path(row["feedback"]["path"])
    if sha256(feedback_path) != row["feedback"]["sha256"]:
        raise ValueError("feedback changed after inventory")
    rows.append(proxy.derive_proxy_record(row, read(feedback_path)))
result = {"kind": "reviewed-export-rally-proxy-manifest-v1", "createdAt": datetime.now(timezone.utc).isoformat(),
          "inventory": identity(args.inventory), "sourceCode": identity(Path(proxy.__file__)),
          "wrapper": identity(Path(__file__).resolve()), "records": rows,
          "policy": {"authorization": "user explicitly requested training/selection experiments using approximately corrected export rallies",
                     "protectedAndSeptember17Excluded": True, "individualSavedCoresOnly": True,
                     "minimumCoreDurationSeconds": .25, "approximateBoundaries": True,
                     "preserveIndividualCoreEndpointsUnderIgnoredMask": True, "finalEvaluationGoldTierUnchanged": True},
          "summary": {"recordings": len(rows), "rallies": sum(len(r["rallies"]) for r in rows),
                      "byGroup": {g: {"recordings": sum(r["sourceGroup"] == g for r in rows),
                                      "rallies": sum(len(r["rallies"]) for r in rows if r["sourceGroup"] == g)} for g in sorted(proxy.ALLOWED_GROUPS)},
                      "exclusionReasons": dict(Counter(e["reason"] for r in rows for e in r["derivation"]["excluded"]))}}
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2, allow_nan=False); handle.write("\n")
print(json.dumps({"output": identity(args.output), "summary": result["summary"]}, indent=2))
