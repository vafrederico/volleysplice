"""Compare retained benchmark values without publishing private media or outputs.

Analysis stage and model profiling timings are excluded. All other analysis fields and every cached
embedding/feature value must match exactly; counts alone cannot pass this check.
Raw input directories must remain external to the repository.
"""
import argparse
import json
import math
from pathlib import Path
import plistlib

MISSING = object()

def compare(left, right):
    stats = {"equal": True, "numericValues": 0, "numericDifferences": 0,
             "maxAbsoluteError": 0.0, "structuralDifferences": 0}

    def visit(a, b):
        if isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)) and not isinstance(b, bool):
            stats["numericValues"] += 1
            if not math.isfinite(a) or not math.isfinite(b):
                stats["structuralDifferences"] += 1
            elif a != b:
                stats["numericDifferences"] += 1
                stats["maxAbsoluteError"] = max(stats["maxAbsoluteError"], abs(a - b))
        elif type(a) is not type(b):
            stats["structuralDifferences"] += 1
        elif isinstance(a, dict):
            stats["structuralDifferences"] += len(a.keys() ^ b.keys())
            for key in a.keys() & b.keys(): visit(a[key], b[key])
        elif isinstance(a, list):
            stats["structuralDifferences"] += abs(len(a) - len(b))
            for x, y in zip(a, b): visit(x, y)
        elif a != b:
            stats["structuralDifferences"] += 1

    visit(left, right)
    stats["equal"] = not (stats["numericDifferences"] or stats["structuralDifferences"])
    return stats


def compare_runs(left, right):
    models = sorted(p.name for p in left.iterdir() if p.is_dir())
    if len(models) != 3 or models != sorted(p.name for p in right.iterdir() if p.is_dir()):
        raise ValueError("Expected the same three model directories")
    results = {}
    for model in models:
        a = json.loads((left / model / "analysis.json").read_text())
        b = json.loads((right / model / "analysis.json").read_text())
        a.pop("stageSeconds", None); b.pop("stageSeconds", None)
        for result in (a, b):
            for head in ("allLabels", "previous"):
                result.get(head, {}).pop("profileMilliseconds", None)
        fields = {key: compare(a.get(key, MISSING), b.get(key, MISSING)) for key in sorted(a.keys() | b.keys())}
        caches = []
        # Content names can contain private cache identities; keep them out of
        # the summary. Decode plists before comparison (dictionary order varies).
        af = sorted((left / model / "cache").glob("*.plist"))
        bf = sorted((right / model / "cache").glob("*.plist"))
        if not af or [p.name for p in af] != [p.name for p in bf]:
            raise ValueError("Cache file sets differ or are missing")
        for x, y in zip(af, bf):
            caches.append(compare(plistlib.loads(x.read_bytes()), plistlib.loads(y.read_bytes())))
        results[model] = {"fields": fields, "caches": caches,
                          "equal": all(v["equal"] for v in [*fields.values(), *caches])}
    return {"schemaVersion": 1, "exactMatch": all(r["equal"] for r in results.values()),
            "excludedAnalysisFields": ["stageSeconds", "allLabels.profileMilliseconds", "previous.profileMilliseconds"],
            "models": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_runs(args.left, args.right)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"exactMatch": result["exactMatch"],
                      "models": {k: v["equal"] for k, v in result["models"].items()}}))
    raise SystemExit(0 if result["exactMatch"] else 1)


if __name__ == "__main__":
    main()
