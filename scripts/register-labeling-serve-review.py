#!/usr/bin/env python3
"""Publish serving inference and explanatory boundary metadata without relabeling."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def identity(path):
    path = Path(path)
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "sizeBytes": len(data)}


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--research", type=Path, required=True)
    args = parser.parse_args()
    root = args.study_root
    base_path = root / "labeling-catalog-production-v1.json"
    catalog = json.loads(base_path.read_text())
    updated = deepcopy(catalog)
    evaluation = json.loads(args.evaluation.read_text())
    if evaluation.get("labelsUsedAsInferenceInputs") is not False or evaluation.get("llmLabelingUsed") is not False:
        raise ValueError("Serving inference must be label independent")
    rid = evaluation["recordingId"]
    matches = [row for row in updated["records"] if row["recordingId"] == rid]
    if len(matches) != 1:
        raise ValueError("Expected one matching recording")
    record = matches[0]
    old_metadata = record["candidateSource"]["modelEvalInference"]
    old_evaluation = json.loads(Path(old_metadata["metadataPath"]).read_text())
    for key in ("predictedEnsembleRanges", "coreInput", "suppression"):
        if evaluation[key] != old_evaluation[key]:
            raise ValueError(f"Serving publication changes prior {key}")
    if len(evaluation["servingSide"]["candidates"]) != len(evaluation["predictedEnsembleRanges"]):
        raise ValueError("Serving candidates do not cover the production ensemble")
    refs = json.loads(args.research.read_text())
    old_refs = json.loads((root / "research-references.json").read_text())
    stripped = deepcopy(refs)
    stripped.pop("boundaryDetailsProvenance", None)
    for entry in stripped["recordings"]:
        for reference in entry["references"]:
            for flag in reference.get("research", {}).get("boundaryFlags", []):
                flag.pop("details", None)
    if stripped != old_refs:
        raise ValueError("Enriched boundary manifest changes original predictions or queue")
    for entry in refs["recordings"]:
        if entry["recordingId"] == rid:
            flags = [flag for ref in entry["references"] for flag in ref.get("research", {}).get("boundaryFlags", [])]
            if not flags or any("details" not in flag for flag in flags):
                raise ValueError("Boundary details are incomplete")

    previous = json.loads((root / "production-labeling-registration-v1.json").read_text())
    preserved = [identity(base_path), identity(previous["humanDraft"]["path"]), identity(root / "research-references.json")]
    record["candidateSource"]["modelEvalInference"] = {"metadataPath": str(args.evaluation),
        "metadataSha256": identity(args.evaluation)["sha256"], "labelsUsedAsInferenceInputs": False}
    restored = deepcopy(updated)
    next(row for row in restored["records"] if row["recordingId"] == rid)["candidateSource"]["modelEvalInference"] = old_metadata
    if restored != catalog:
        raise ValueError("Unrelated catalog content changed")
    output = root / "labeling-catalog-serve-review-v1.json"
    receipt_path = root / "serve-review-registration-v1.json"
    if output.exists() or receipt_path.exists():
        raise FileExistsError("Preserve existing publication history")
    write_new(output, updated)
    if any(identity(item["path"]) != item for item in preserved):
        raise RuntimeError("An input changed during publication")
    receipt = {"passed": True, "createdAt": datetime.now(timezone.utc).isoformat(), "recordingId": rid,
        "catalog": identity(output), "evaluation": identity(args.evaluation), "researchReferences": identity(args.research),
        "preserved": preserved, "humanDraftUnchanged": True, "rallyPredictionsUnchanged": True,
        "reviewQueueAndOriginalSignalsUnchanged": True, "boundaryFlagsWithDetails": len(flags),
        "servingCandidates": len(evaluation["servingSide"]["candidates"]), "source": identity(__file__)}
    write_new(receipt_path, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
