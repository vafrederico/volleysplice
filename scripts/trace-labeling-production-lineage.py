#!/usr/bin/env python3
"""Publish exact decoder/serve-composition lineage without changing study inputs."""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def identity(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "sizeBytes": path.stat().st_size}


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run(args):
    root = args.study_root
    output = root / "compact-standalone-misses-v1"
    output.mkdir(exist_ok=True)
    destination = output / "production-head-lineage.json"
    if destination.exists():
        raise FileExistsError(destination)
    replay = read(root / "production-replay.json")
    latest = read(root / "production-ensemble-v1/production-replay.json")
    assert replay == latest, "Production replays differ"
    registration = read(root / "inference-registration.json")
    audit = read(root / "inference-audit-v1.json")
    native_path = root / "native-features.npz"
    assert audit["passed"] and identity(native_path)["sha256"] == audit["artifacts"][native_path.name]["sha256"]
    assert identity(root / "production-replay.json")["sha256"] == audit["artifacts"]["production-replay.json"]["sha256"]
    for key in ("previous", "v2"):
        expected = registration["browserRuntimes"][key]
        assert identity(Path(expected["path"])) == expected, "Runtime asset changed"
    with np.load(native_path, allow_pickle=False) as native:
        times = native["times"].tolist()
    receipt = read(root / "inference-receipt.json")
    ignored = registration["operationalIgnoredIntervalsForAdviserOnly"]
    assert len(ignored) == 1 and ignored[0]["start"] == 0
    payload = {"repoUrl": REPO.as_uri(), "id": receipt["recordingId"], "duration": receipt["durationSeconds"],
               "times": times, "probabilities": replay["probabilities"], "components": replay["components"],
               "unionWithConfidence": replay["unionWithConfidence"], "aggressiveCore": replay["variants"]["aggressive"]["core"],
               "ignoredEnd": ignored[0]["end"]}
    traced = subprocess.run([str(args.node), str(REPO / "scripts/trace-labeling-production-lineage.mjs")],
                            input=json.dumps(payload), text=True, capture_output=True, check=True)
    result = json.loads(traced.stdout)
    write(destination, result)
    scope = [row for row in result["unionLineage"] if row["intersectsEvaluationTime"]]
    summary = {"schemaVersion": 1, "createdAt": datetime.now(timezone.utc).isoformat(),
               "inputs": [identity(root / name) for name in ("production-replay.json", "native-features.npz", "inference-registration.json", "inference-audit-v1.json")],
               "sources": [identity(REPO / name) for name in ("scripts/trace-labeling-production-lineage.py", "scripts/trace-labeling-production-lineage.mjs", "prod/src/lib/on-device/model.ts", "prod/src/lib/on-device/ensemble.ts")],
               "runtimeAssets": {key: registration["browserRuntimes"][key] for key in ("previous", "v2")},
               "output": identity(destination), "counts": {"fullFileEnsemble": len(result["unionLineage"]), "evaluationEnsemble": len(scope),
                   "primaryRallyBacked": sum(row["mechanism"] == "rally-head-backed" for row in scope),
                   "serveCreatedWithoutPrimaryRally": sum(row["mechanism"] == "serve-created-without-primary-rally" for row in scope)},
               "verification": result["verification"], "labelsUsed": False, "servingSideUsed": False,
               "interpretation": "Attribution is about production candidate creation, not correctness. Serve recovery means no primary decoded rally contributed. Existing-rally start refinement is not recovery. Serving-side near/far inference is separate and does not create exports."}
    write(output / "production-head-lineage-receipt.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--node", type=Path, default=Path(private_value('private-reference-0101')))
    run(parser.parse_args())
