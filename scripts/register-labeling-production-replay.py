#!/usr/bin/env python3
"""Add a label-independent production replay to a derived labeling catalog."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def identity(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "sizeBytes": len(data)}


def write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    args = parser.parse_args()
    root = args.study_root.resolve()
    old_path = root / "labeling-catalog.json"
    output = root / "labeling-catalog-production-v1.json"
    receipt = root / "production-labeling-registration-v1.json"
    if output.exists() or receipt.exists():
        raise FileExistsError("Production publication already exists; preserve its history")

    old_identity = identity(old_path)
    references_identity = identity(root / "research-references.json")
    evaluation_identity = identity(args.evaluation)
    evaluation = json.loads(args.evaluation.read_text())
    if evaluation.get("labelsUsedAsInferenceInputs") is not False or evaluation.get("llmLabelingUsed") is not False:
        raise ValueError("Production replay must declare label-independent inference")
    recording_id = evaluation["recordingId"]
    core_path = Path(evaluation["coreInput"]["metadataPath"])
    core = json.loads(core_path.read_text())
    if core["recordingId"] != recording_id or not evaluation.get("predictedEnsembleRanges"):
        raise ValueError("Production/core metadata identity mismatch or empty ensemble")
    publication = json.loads((root / "labeling-registration-v2.json").read_text())
    draft_path = Path(publication["humanDraft"]["path"])
    draft_identity = identity(draft_path)
    if json.loads(draft_path.read_text())["recording"]["id"] != recording_id:
        raise ValueError("Human draft is for another recording")

    original = json.loads(old_path.read_text())
    updated = deepcopy(original)
    matches = [row for row in updated["records"] if row["recordingId"] == recording_id]
    if len(matches) != 1:
        raise ValueError("Expected exactly one existing recording")
    record = matches[0]
    if record["candidateSource"].get("modelEvalInference"):
        raise ValueError("An existing production reference must not be replaced")
    record["candidateSource"]["modelEvalInference"] = {
        "metadataPath": str(args.evaluation.resolve()),
        "metadataSha256": evaluation_identity["sha256"],
        "labelsUsedAsInferenceInputs": False,
    }
    restored = deepcopy(updated)
    next(row for row in restored["records"] if row["recordingId"] == recording_id)["candidateSource"].pop("modelEvalInference")
    if restored != original:
        raise ValueError("Registration unexpectedly changed other catalog content")
    write_new(output, updated)
    for old in (old_identity, references_identity, draft_identity, evaluation_identity):
        if identity(Path(old["path"])) != old:
            raise RuntimeError("An input changed during registration")
    result = {
        "passed": True, "createdAt": datetime.now(timezone.utc).isoformat(),
        "recordingId": recording_id, "sourceCatalog": old_identity,
        "catalog": identity(output), "evaluation": evaluation_identity,
        "core": identity(core_path), "humanDraft": draft_identity,
        "researchReferences": references_identity,
        "onlyChangedField": "target record candidateSource.modelEvalInference",
        "humanDraftUnchanged": True, "existingCatalogPreserved": True,
        "researchReferencesUnchanged": True,
        "registeredRecords": len(updated["records"]),
    }
    write_new(receipt, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
