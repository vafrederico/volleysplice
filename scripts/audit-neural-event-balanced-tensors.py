#!/usr/bin/env python3
"""Audit event-balanced tensors with the frozen independent scaler verifier.

The reused verifier computes normalization and scalers independently of training.
This wrapper binds its immutable bytes, the new contract and historical reference
hashes, and checks the 144-fit/468-checkpoint TCN-only completion inventory.
"""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(private_value('private-reference-0088'))
MANIFEST = ROOT.parent/"2026-09-19-expanded/manifest.json"
EXPECTED_CONTRACT = "9f1ce53e50c5bd21a4365e0c30d62bb43dd7e72962d2d7a3341f720489aaebda"
VERIFIER_SHA256 = "c7460cf0dc1cea7f54031405a0c78713edb54ebb941be18fd0bd8ad82b4c500b"


def verifier():
    import hashlib

    path = REPO/"scripts/audit-neural-expanded-tensors.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != VERIFIER_SHA256:
        raise ValueError("independent numerical verifier changed")
    spec = importlib.util.spec_from_file_location("frozen_independent_tensor_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def self_test(audit):
    import numpy as np

    absolute = next(iter(audit.ABSOLUTE_FEATURE_NAMES))
    raw = np.array([[2, 4], [2, 3], [4, 2], [8, 1]], dtype=np.float32)
    normalized = audit.independent_percentile_normalization(raw, ("rank_test", absolute))
    np.testing.assert_array_equal(normalized[:, 0], np.array([1/6, 1/6, 2/3, 1], np.float32))
    np.testing.assert_array_equal(normalized[:, 1], raw[:, 1])
    inputs = {"synthetic": {"tier": "exact", "valid": np.array([True, False, True]),
                            "normalized": np.array([[0, 2, 7], [99, 99, 99], [2, 6, 7]], np.float32)}}
    mean, scale, count = audit.independent_scaler(["synthetic"], inputs)
    np.testing.assert_array_equal(mean, np.array([1, 4, 7], np.float32))
    np.testing.assert_array_equal(scale, np.array([1, 2, 1e-4], np.float32))
    assert count == 2
    inputs["synthetic"]["tier"] = "draft"
    try:
        audit.independent_scaler(["synthetic"], inputs)
    except ValueError:
        pass
    else:
        raise AssertionError("auxiliary scaler population was not rejected")
    architecture = audit.parameter_contract(["tcn"])["tcn"]
    assert architecture["parameterCount"] == 29700
    print(json.dumps({"selfTestPassed": True, "checks": ["tied average ranks", "absolute channels preserved",
          "ignored ticks excluded", "centered variance and scale floor", "auxiliary scaler rejection", "29700-parameter TCN contract"]}))


def run(manifest, study, progress, audit):
    registration = audit.read(study/"preregistration.json")
    contract = registration["contract"]
    audit.require(registration["sha256"] == EXPECTED_CONTRACT, "wrong event-balanced frozen contract")
    audit.require(contract["kinds"] == ["tcn"] and len(contract["cohorts"]) == 3
                  and len(contract["groups"]) == 4 and len(contract["seeds"]) == 3
                  and contract["checkpointEpochs"] == [5, 15, 30, 60], "unexpected completion scope")
    audit.require(len(contract["code"]) == 13, "unexpected historical/new source scope")
    for name, digest in contract["code"].items():
        audit.require(audit.sha256(REPO/"analysis"/name) == digest, f"registered source changed: {name}")
    reference = contract["referenceStudy"]
    reference_root = Path(reference["path"])
    references = []
    for name, field in (("preregistration.json", "preregistrationFileSha256"),
                        ("report.json", "reportSha256"), ("summary.json", "summarySha256")):
        item = audit.identity(reference_root/name)
        audit.require(item["sha256"] == reference[field], f"frozen reference changed: {name}")
        references.append(item)
    old_registration = audit.read(reference_root/"preregistration.json")
    audit.require(old_registration["sha256"] == reference["contractSha256"], "reference contract mismatch")
    audit.require(all(contract["code"][name] == digest for name, digest in old_registration["contract"]["code"].items()),
                  "historical source binding differs from reference")
    result = audit.audit(manifest, study, progress)
    result.update({"kind": "neural-event-balanced-independent-tensor-audit-v1",
                   "auditScript": audit.identity(Path(__file__).resolve()),
                   "independentVerifier": audit.identity(REPO/"scripts/audit-neural-expanded-tensors.py"),
                   "registeredSourceHashesVerified": contract["code"], "frozenReferencesVerified": references,
                   "contractSha256": registration["sha256"],
                   "weightingMetadataAudit": "Separate summary auditor independently reconstructs live multipliers, mask semantics, sampling exposure and metrics."})
    audit.require(result["expectedFitCount"] == 144, "incorrect expected fit inventory")
    audit.require(result["architectures"]["tcn"]["parameterCount"] == 29700, "unexpected TCN parameter count")
    if result["studyComplete"]:
        weights = list((study/"fits").rglob("weights-*.npz"))
        predictions = list((study/"fits").rglob("predictions-*.npz"))
        all_npz = list((study/"fits").rglob("*.npz"))
        result["completeArtifactInventory"] = {"completedFits": result["completedFitCountAtStart"],
                                                "checkpoints": result["checkedCheckpointCount"],
                                                "weightNPZ": len(weights), "predictionNPZ": len(predictions),
                                                "totalNPZ": len(all_npz)}
        if result["checkedCheckpointCount"] != 468 or len(weights) != 468 or len(predictions) != 468 or len(all_npz) != 936:
            result["passed"] = False
            result["failures"].append({"error": "completion inventory differs from 468 checkpoint pairs / 936 NPZ"})
    for name, digest in contract["code"].items():
        audit.require(audit.sha256(REPO/"analysis"/name) == digest, f"source changed during audit: {name}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    audit = verifier()
    if args.self_test:
        self_test(audit)
        return
    audit.require(not (args.progress and args.output), "progress mode must not write an output")
    output = args.output or args.root/"tensor-scaler-audit-v1.json"
    if not args.progress:
        audit.require(not output.exists(), "refusing to overwrite audit artifact")
    result = run(args.manifest, args.study or args.root/"study", args.progress, audit)
    if not args.progress:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as f:
            json.dump(result, f, indent=2, allow_nan=False)
            f.write("\n")
    summary = {key: result[key] for key in ("passed", "studyComplete", "completedFitCountAtStart", "expectedFitCount",
              "passedFitCount", "failedFitCount", "checkedCheckpointCount", "checkedPredictionArrayCount",
              "scalerExactlyEqualCheckpointCount", "maximumMeanAbsoluteError", "maximumScaleAbsoluteError", "failures")}
    if not args.progress:
        summary["artifact"] = audit.identity(output)
        summary["completeArtifactInventory"] = result["completeArtifactInventory"]
    print(json.dumps(summary, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
