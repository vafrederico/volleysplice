#!/usr/bin/env python3
"""Describe compact/production coverage against an immutable reviewed import."""
from analysis.private_ledger import private_value
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def identity(path):
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


def union(rows):
    result = []
    for row in sorted(rows, key=lambda r: (r["start"], r["end"])):
        start, end = row["start"], row["end"]
        if end <= start:
            continue
        if result and start <= result[-1]["end"]:
            result[-1]["end"] = max(result[-1]["end"], end)
        else:
            result.append({"start": start, "end": end})
    return result


def subtract(rows, ignored):
    result = union(rows)
    for barrier in ignored:
        kept = []
        for row in result:
            if barrier["end"] <= row["start"] or barrier["start"] >= row["end"]:
                kept.append(row)
            else:
                if row["start"] < barrier["start"]:
                    kept.append({"start": row["start"], "end": barrier["start"]})
                if row["end"] > barrier["end"]:
                    kept.append({"start": barrier["end"], "end": row["end"]})
        result = kept
    return result


def seconds(rows):
    return sum(row["end"] - row["start"] for row in rows)


def intersection(left, right):
    return sum(max(0, min(a["end"], b["end"]) - max(a["start"], b["start"])) for a in left for b in right)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.study_root
    source_path = REPO / "scripts/prepare-neural-production-comparison.py"
    spec = importlib.util.spec_from_file_location("production_comparison", source_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    paths = [root / "production-replay.json", root / "compact-events.json",
        root / private_value('private-reference-0097')]
    production, compact, label = [json.loads(p.read_text()) for p in paths]
    ignored = label["ignoredIntervals"]
    duration = label["recording"]["durationSeconds"]
    model_rows = {"productionEnsemble": production["unionWithConfidence"],
        "productionAggressive": production["variants"]["aggressive"]["core"], "compactStandalone": compact["events"]}
    cores = {name: subtract(rows, ignored) for name, rows in model_rows.items()}
    human_core = subtract(label["rallies"], ignored)
    exports = {name: {str(pad): helper.canonical_exports(rows, ignored, duration, pad)
        for pad in [0, 1, 2, 3]} for name, rows in model_rows.items()}
    human_exports = {str(pad): helper.canonical_exports(label["rallies"], ignored, duration, pad) for pad in [0, 1, 2, 3]}
    metrics = []
    for name in model_rows:
        for pad in [0, 1, 2, 3]:
            model_export, human_export = exports[name][str(pad)], human_exports[str(pad)]
            model_seconds, human_seconds = seconds(model_export), seconds(human_export)
            matched_padding = intersection(model_export, human_export)
            retained_core = intersection(model_export, human_core)
            precision = matched_padding / model_seconds if model_seconds else 0
            recall = retained_core / seconds(human_core)
            metrics.append({"model": name, "beforeSeconds": pad, "afterSeconds": pad,
                "joinGapSeconds": 3, "P_pad": precision, "R_core": recall,
                "F1_padP_coreR": 2 * precision * recall / (precision + recall) if precision + recall else 0,
                "modelExportSeconds": model_seconds, "humanExportSeconds": human_seconds,
                "exportDifferenceSeconds": model_seconds - human_seconds,
                "humanCoreSecondsOmitted": seconds(human_core) - retained_core})
    humans = []
    for row in label["rallies"]:
        scoped = subtract([row], ignored)
        size = seconds(scoped)
        if not size:
            continue
        humans.append({"id": row["id"], "start": row["start"], "end": row["end"], "coreSeconds": size,
            "coreCoverage": {name: intersection(scoped, core) / size for name, core in cores.items()},
            "exportCoverageByPadding": {name: {pad: intersection(scoped, ranges) / size for pad, ranges in values.items()}
                for name, values in exports.items()}})
    candidates = []
    for row in production["unionWithConfidence"]:
        scoped = subtract([row], ignored)
        size = seconds(scoped)
        if not size:
            continue
        compact_seconds = intersection(scoped, cores["compactStandalone"])
        candidates.append({"id": row["id"], "start": row["start"], "end": row["end"], "agreement": row["agreement"],
            "coreSeconds": size, "compactCoreOverlapSeconds": compact_seconds, "compactCoreCoverage": compact_seconds / size,
            "keptByAggressive": intersection(scoped, cores["productionAggressive"]) > 0,
            "humanCoreOverlapSeconds": intersection(scoped, human_core),
            "overlappingHumanIds": [h["id"] for h in label["rallies"] if intersection(scoped, subtract([h], ignored)) > 0],
            "overlappingCompactIds": [c["id"] for c in compact["events"] if intersection(scoped, subtract([c], ignored)) > 0]})
    result = {"recordingId": label["recording"]["id"], "targetPaddingSeconds": 2,
        "paddingCases": [0, 1, 2, 3], "joinGapSeconds": 3, "ignoredIntervals": ignored,
        "fullyMissedDefinition": "Zero raw compact core overlap; event IoU and padded export coverage are separate.",
        "selectionPerformed": False, "labelSemantics": "Manually reviewed retained export core ranges, not independently exact serve/dead-ball annotations.",
        "sources": [identity(p) for p in paths] + [identity(Path(__file__)), identity(source_path)],
        "counts": {"productionValidCandidates": len(candidates), "compactCandidates": len(compact["events"]), "humanRetainedRanges": len(humans),
            "productionCandidatesWithZeroCompactOverlap": sum(c["compactCoreOverlapSeconds"] == 0 for c in candidates),
            "humanRangesWithZeroCompactOverlap": sum(h["coreCoverage"]["compactStandalone"] == 0 for h in humans)},
        "productionCandidates": candidates, "humanRanges": humans, "paddingMetrics": metrics}
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"output": str(args.output), "counts": result["counts"],
        "humanExportOmissionsAt2s": [{"id": h["id"], "start": h["start"], "end": h["end"],
            "compactRetained": h["exportCoverageByPadding"]["compactStandalone"]["2"],
            "productionRetained": h["exportCoverageByPadding"]["productionEnsemble"]["2"]}
            for h in humans if h["exportCoverageByPadding"]["compactStandalone"]["2"] < .999999]}, indent=2))


if __name__ == "__main__":
    main()
