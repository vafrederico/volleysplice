#!/usr/bin/env python3
"""Expose a previously audited compact run as an independent labeling rail.

This does not perform inference, fitting, or label edits. It verifies the saved
probabilities and decoder output, then appends a read-only standalone reference.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
HEADS = ("live", "serve", "end", "keep")
MODEL_ID = "compact-standalone-3407-outer0"
WEIGHT_SHA = "b386bb0834c0a3663d2fbefb89377b1300ca9437aeb0485762444747c2c13311"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def identity(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "sizeBytes": path.stat().st_size}


def semantic_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run(args: argparse.Namespace) -> dict:
    from analysis.neural_development import Example, decode

    root = args.study_root
    inputs = {name: identity(root / name) for name in (
        "compact-events.json", "compact-probabilities.npz", "inference-registration.json",
        "inference-receipt.json", "inference-audit-v1.json")}
    inputs[args.base_manifest.name] = identity(args.base_manifest)
    registration = read(root / "inference-registration.json")
    audit = read(root / "inference-audit-v1.json")
    receipt = read(root / "inference-receipt.json")
    compact = read(root / "compact-events.json")
    manifest = read(args.base_manifest)
    require(not args.output.exists() and not args.receipt.exists(), "Refusing to overwrite an artifact")
    require(audit["passed"] is True, "Original inference audit did not pass")
    for name in ("compact-events.json", "compact-probabilities.npz", "inference-registration.json", "inference-receipt.json"):
        require(inputs[name]["sha256"] == audit["artifacts"][name]["sha256"], f"Audited input changed: {name}")
    require((registration["seed"], registration["outerFoldIndex"], registration["epoch"]) == (3407, 0, 60),
            "Unexpected frozen checkpoint")
    require(registration["weights"]["sha256"] == WEIGHT_SHA == identity(Path(registration["weights"]["path"]))["sha256"],
            "Frozen checkpoint hash changed")
    require(registration["heads"] == list(HEADS) and registration["modelParameters"] == 29700,
            "Unexpected compact architecture")
    require(not any(registration[key] for key in ("trainingPerformed", "humanRalliesUsedForInference",
                                                "humanCorrectionsUsedForInference", "humanOracleUsed")),
            "Inference must remain independent of human annotations")
    source = next(row for row in registration["sources"] if Path(row["path"]).name == "neural_development.py")
    require(identity(REPO / "analysis/neural_development.py")["sha256"] == source["sha256"], "Decoder source changed")
    require(compact["decoder"] == registration["decoder"], "Decoder settings differ from inference provenance")
    recording_id = compact["recordingId"]
    records = [row for row in manifest["recordings"] if row["recordingId"] == recording_id]
    require(len(records) == 1 and receipt["recordingId"] == recording_id, "Recording identity mismatch")
    record = records[0]
    require(record["durationSeconds"] == receipt["durationSeconds"], "Duration mismatch")
    require(not any(ref["modelId"] == MODEL_ID for ref in record["references"]), "Standalone reference already exists")
    signal_source = next(ref for ref in record["references"] if ref["modelId"] == "production-compact-head-refined-review-3407-outer0")
    require(signal_source["research"]["provenance"] == registration, "Signal inference provenance changed")
    signals = signal_source["research"]["signals"]
    with np.load(root / "compact-probabilities.npz", allow_pickle=False) as saved:
        times, scores = saved["times"].copy(), saved["scores"].copy()
    require(times.shape == (receipt["featureRows"],) and scores.shape == (len(times), len(HEADS)), "Invalid probability shapes")
    require(np.isfinite(times).all() and np.all(np.diff(times) > 0) and np.isfinite(scores).all()
            and np.all((scores >= 0) & (scores <= 1)), "Invalid native timestamps or probabilities")
    require(np.array_equal(np.asarray(signals["times"]), times), "Signal timestamps differ from saved native timestamps")
    for index, head in enumerate(HEADS):
        require(np.array_equal(np.asarray(signals[head]), scores[:, index].astype(np.float64)), f"Signal values changed: {head}")
    # Re-decode saved scores only; dummy features/targets are never consumed by decode.
    example = Example(recording_id, "standalone-display", record["durationSeconds"], times,
                      np.empty((len(times), 0), np.float32), np.zeros((len(times), 4), np.float32),
                      np.ones(len(times), bool), (), (), "unspecified")
    decoded = [{"id": f"compact:{index + 1}", "start": event.start, "end": event.end}
               for index, event in enumerate(decode(example, scores, registration["decoder"]))]
    require(decoded == compact["events"] and len(decoded) == receipt["compactRallies"], "Saved compact events do not match decoder output")
    rallies = [{"start": event["start"], "end": event["end"], "tags": ["ai-reference", "compact-standalone"],
                "notes": f"{event['id']}: read-only compact prediction; not human ground truth or a validated serve contact."}
               for event in decoded]
    provenance = {key: copy.deepcopy(registration[key]) for key in (
        "weights", "seed", "outerFoldIndex", "epoch", "decoder", "modelParameters", "inputColumns", "heads",
        "checkpointChoice", "checkpointStatus", "rawContentSha256", "sampledFingerprint", "featureInput")}
    provenance.update({"inferenceRegistration": inputs["inference-registration.json"],
                       "compactEvents": inputs["compact-events.json"], "compactProbabilities": inputs["compact-probabilities.npz"],
                       "policy": "standalone-decoder", "trainingPerformed": False, "inferenceRerun": False,
                       "humanLabelsUsed": False, "serveAssignmentsProduced": False})
    reference = {
        "modelId": MODEL_ID, "modelLabel": "Compact standalone",
        "description": "Independent compact short-boost TCN predictions. Rally borders use the saved decoder output; the export strip uses these compact rallies with the selected padding and gap joining.",
        "rallies": rallies, "exportRallies": copy.deepcopy(rallies), "exportPolicy": "model-predictions",
        "research": {
            "recommendation": "Inspect the compact model's own decoded rallies and four signals. This standalone layer has no production boundary review queue. Serve/start is a rally-start cue, not serving side or a validated serve contact.",
            "signals": copy.deepcopy(signals), "boundaryFlags": [], "reviewRegions": [],
            "queue": {"budgetFraction": 0, "reviewSeconds": 0, "selectedParentCount": 0}, "provenance": provenance,
        },
    }
    result = copy.deepcopy(manifest)
    target = next(row for row in result["recordings"] if row["recordingId"] == recording_id)
    target["references"].append(reference)
    restoration = copy.deepcopy(result)
    next(row for row in restoration["recordings"] if row["recordingId"] == recording_id)["references"].pop()
    require(restoration == manifest, "Existing reference contents changed")
    write_new(args.output, result)
    prior_hashes = {ref["modelId"]: semantic_sha(ref) for ref in record["references"]}
    final = read(args.output)
    final_refs = next(row for row in final["recordings"] if row["recordingId"] == recording_id)["references"]
    require(all(semantic_sha(ref) == prior_hashes[ref["modelId"]] for ref in final_refs[:-1]), "Prior reference readback mismatch")
    registration_receipt = {
        "schemaVersion": 1, "createdAt": datetime.now(timezone.utc).isoformat(), "generator": identity(Path(__file__)),
        "recordingId": recording_id, "modelId": MODEL_ID, "inputs": inputs, "output": identity(args.output),
        "counts": {"previousReferences": len(record["references"]), "references": len(final_refs), "rallies": len(rallies),
                   "exportRallies": len(rallies), "signalSamples": len(times), "signalHeads": len(HEADS)},
        "exportPolicy": "model-predictions", "rawDecodedBoundariesPreserved": True, "ownExportRalliesVerified": True,
        "previousReferencesPreserved": True, "previousReferenceSemanticSha256": prior_hashes,
        "savedDecoderReplayExact": True, "displaySignalsExact": True, "humanLabelsReadOrWritten": False,
        "inferenceRerun": False, "trainingPerformed": False, "serveAssignmentsProduced": False,
        "notes": "Only adds a read-only display reference. Production, serving references and editable human labels are not changed.",
    }
    write_new(args.receipt, registration_receipt)
    return registration_receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--base-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    args.base_manifest = args.base_manifest or args.study_root / "research-references-boundary-details-v1.json"
    args.output = args.output or args.study_root / "research-references-compact-standalone-v1.json"
    args.receipt = args.receipt or args.study_root / "compact-standalone-registration-v1.json"
    print(json.dumps(run(args), indent=2, allow_nan=False))
