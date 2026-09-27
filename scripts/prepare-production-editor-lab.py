"""Prepare an editor sandbox from frozen predictions, never human review answers."""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(root: Path, output: Path):
    names = ["production-replay.json", "boundary-adviser.json", "compact-events.json",
             "research-references-compact-standalone-v1.json", "serving-side-v2/ensemble-inference.json",
             "inference-registration.json"]
    contents = {name: (root / name).read_bytes() for name in names}
    sources = {name: json.loads(raw) for name, raw in contents.items()}
    production = sources[names[0]]
    adviser = sources[names[1]]
    compact = sources[names[2]]
    references = sources[names[3]]["recordings"]
    serving = sources[names[4]]
    registration = sources[names[5]]
    assert adviser["record"]["rallies"] == [], "Adviser must have been created without human answers"
    assert serving["labelsUsedAsInferenceInputs"] is False
    assert registration["humanRalliesUsedForInference"] is False
    assert registration["humanCorrectionsUsedForInference"] is False
    recording_id = adviser["record"]["id"]
    reference = next(row for row in references if row["recordingId"] == recording_id)
    assert compact["recordingId"] == serving["recordingId"] == recording_id
    standalone = next(row for row in reference["references"] if row["modelId"] == "compact-standalone-3407-outer0")
    # Select model-only fields. Existing evaluation/oracle directories are never opened.
    result = {
        "schemaVersion": 1,
        "kind": "volleycut-production-editor-lab",
        "recording": {"id": recording_id, "videoFilename": reference["videoFilename"],
                      "durationSeconds": reference["durationSeconds"], "contentSha256": registration["rawContentSha256"]},
        "ignoredIntervals": adviser["record"]["ignoredIntervals"],
        "productionEvents": production["unionWithConfidence"],
        "suppression": production["suggestions"],
        "policies": {key: {"core": row["core"], "barriersByPadding": row["barriersByPadding"]}
                     for key, row in production["variants"].items()},
        "boundaryEvents": adviser["plan"]["events"],
        "guidanceReviewParents": [{"parentId": job["parentId"], "priority": job["priority"],
                                  "recommended": job["parentId"] in adviser["queue"]["selectedParentIds"]}
                                 for job in adviser["jobs"]],
        "compactEvents": compact["events"],
        "signals": standalone["research"]["signals"],
        "serving": {"candidates": serving["servingSide"]["candidates"]},
        "provenance": {"labelBlind": True, "sourceHashes": {
            name: hashlib.sha256(raw).hexdigest() for name, raw in contents.items()}},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(result, indent=2) + "\n").encode()
    if output.exists():
        if output.read_bytes() != raw:
            raise FileExistsError("Refusing to overwrite a different frozen lab manifest")
    else:
        output.write_bytes(raw)
    print(json.dumps({"output": str(output), "sha256": hashlib.sha256(raw).hexdigest(),
                      "modelEvents": {"production": len(result["productionEvents"]),
                                      "boundary": len(result["boundaryEvents"]), "compact": len(result["compactEvents"])},
                      "labelsUsed": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root, args.output)
