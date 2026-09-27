#!/usr/bin/env python3
"""Verify frozen inputs and replay three uniform controls against stored epoch5.

Only the new fitter executes. Historical training and candidate evaluation never
run here. All output paths are newly created, and mismatches fail immediately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
ROOT = Path(private_value('private-reference-0057'))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(path):
    return {"path": str(path), "sha256": sha256(path), "sizeBytes": path.stat().st_size}


def read(path):
    return json.loads(path.read_text())


def write_new(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def declared_references(value):
    result = {}

    def walk(item):
        if isinstance(item, dict):
            if isinstance(item.get("path"), str) and "sha256" in item:
                old = result.setdefault(item["path"], item["sha256"])
                require(old == item["sha256"], "contradictory declared input hashes")
            for child in item.values():
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)
    walk(value)
    return result


def compare_npz(reference, candidate):
    import numpy as np

    result = {"reference": identity(reference), "candidate": identity(candidate), "arrays": []}
    with np.load(reference, allow_pickle=False) as a, np.load(candidate, allow_pickle=False) as b:
        require(set(a.files) == set(b.files), f"checkpoint array names differ: {candidate}")
        for name in a.files:
            left, right = a[name], b[name]
            equal = (left.dtype == right.dtype and left.shape == right.shape
                     and left.tobytes(order="C") == right.tobytes(order="C"))
            result["arrays"].append({"name": name, "shape": list(left.shape), "dtype": str(left.dtype),
                                     "bitExact": equal,
                                     "maximumAbsoluteError": float(np.max(np.abs(left.astype(np.float64)-right.astype(np.float64))))
                                     if left.shape == right.shape and left.size else None})
        result["bitExact"] = all(row["bitExact"] for row in result["arrays"])
    return result


def chunk_checks(data, mean, scale, expanded, weighting):
    import numpy as np

    result = []
    for tier, rows in data.items():
        before = [(row.mask.copy(), row.example.targets.copy()) for row in rows]
        original = expanded.make_chunks(rows, mean, scale, "tcn")
        modified = weighting.make_weighted_chunks(rows, mean, scale, "tcn")
        require(len(original) == len(modified), "chunk counts changed")
        for a, b in zip(original, modified):
            require(all(a[i].dtype == b[i].dtype and a[i].shape == b[i].shape
                        and a[i].tobytes() == b[i].tobytes() for i in range(3))
                    and a[3] == b[3], "frozen first four chunk fields changed")
        require(np.array_equal(expanded.sampling_weights(original), expanded.sampling_weights(modified)),
                "sampling probabilities changed")
        seeds = []
        for seed in expanded.base.SEEDS:
            left = expanded.epoch_batches(original, np.random.default_rng(seed))
            right = expanded.epoch_batches(modified, np.random.default_rng(seed))
            require(left == right, "sampled batch identities changed")
            seeds.append({"seed": seed, "batchCount": len(left), "bitExact": True,
                          "batchIdentitySha256": hashlib.sha256(json.dumps(left, separators=(",", ":")).encode()).hexdigest()})
        for row, (mask, targets) in zip(rows, before):
            require(np.array_equal(row.mask, mask) and np.array_equal(row.example.targets, targets),
                    "weight construction mutated supervision")
        result.append({"tier": tier, "recordings": len(rows), "chunks": len(original),
                       "firstFourFieldsBitExact": True, "samplingProbabilitiesBitExact": True, "seeds": seeds})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT/"2026-09-19-expanded/manifest.json")
    parser.add_argument("--reference-study", type=Path, default=ROOT/"2026-09-19-expanded/study")
    parser.add_argument("--output", type=Path, default=ROOT/"2026-09-19-event-balanced/preflight")
    args = parser.parse_args()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import numpy as np
    import torch
    from analysis import neural_expanded_development as expanded
    from analysis import neural_event_weighting as weighting
    from analysis import neural_event_balanced_development as runner

    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve()
    with (args.output/"preflight-source.py").open("xb") as f:
        f.write(source.read_bytes())
    report = {"kind": "event-balanced-uniform-control-preflight-v1", "passed": False,
              "createdAt": datetime.now(timezone.utc).isoformat(), "script": identity(source),
              "scriptSnapshot": identity(args.output/"preflight-source.py"), "comparisons": [],
              "limits": ["No candidate outcomes or decoder selection are evaluated.",
                         "Historical training is never rerun; compare new uniform fitter to immutable stored checkpoints.",
                         "Raw video content is not reread; frozen source/proxy hashes and current small metadata/cache hashes are bound."]}
    try:
        registration_path = args.reference_study/"preregistration.json"
        registration = read(registration_path)
        contract = registration["contract"]
        require(runner.canonical_hash(contract) == registration["sha256"], "historical contract changed")
        require(len(contract["code"]) == 11, "unexpected historical code scope")
        require(sha256(args.manifest) == contract["manifestSha256"], "historical expanded manifest changed")
        current_code = {name: sha256(REPO/"analysis"/name) for name in
                        (*contract["code"], "neural_event_weighting.py", "neural_event_balanced_development.py")}
        require(all(current_code[name] == digest for name, digest in contract["code"].items()),
                "historical implementation bytes changed")
        environment = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                       "device": "cuda", "gpu": torch.cuda.get_device_name()}
        require(environment == contract["environment"], "uniform replay environment differs")
        report.update({"historicalRegistration": identity(registration_path), "historicalContractSha256": registration["sha256"],
                       "manifest": identity(args.manifest), "code": current_code, "historicalCodeUnchangedCount": 11,
                       "environment": environment, "torchThreads": torch.get_num_threads(),
                       "cublasWorkspaceConfig": os.environ["CUBLAS_WORKSPACE_CONFIG"]})
        manifest = read(args.manifest)
        references = []
        for name, digest in declared_references(manifest).items():
            path = Path(name)
            require(path.suffix in (".json", ".npz"), f"unexpected nonmetadata/cache reference: {path}")
            actual = identity(path)
            require(actual["sha256"] == digest, f"declared dataset/cache input changed: {path}")
            references.append(actual)
        report["verifiedInputReferences"] = references
        data = expanded.load_data(args.manifest)
        groups = sorted({row.example.group for row in data["exact"]})
        require(groups == contract["groups"] and [len(data[t]) for t in ("exact", "draft", "coverage")] == [8, 3, 7],
                "frozen population changed")
        diagnostics = {tier: [weighting.live_event_weights(row)[1] for row in rows] for tier, rows in data.items()}
        report["weightDiagnostics"] = diagnostics
        report["warnings"] = [
            "Uncapped per-rally normalization amplifies one supervised draft tick up to 29.575758; this was accepted prospectively.",
            "18 draft events have no supervised live ticks after existing endpoint masking and remain unknown, not negative.",
            "Float32 multipliers preserve positive mass within rounding tolerance, not exact integer arithmetic."]
        report["sourceLineageBindings"] = [{"id": row["id"], "sourceGroup": row["sourceGroup"],
                                          "contentSha256": row["contentSha256"],
                                          "sourceContentSha256": row.get("sourceContentSha256"),
                                          "audiovisualCache": row["featureCaches"]["audiovisual"]["sha256"]}
                                         for key in ("exactRows", "draftRows", "coverageRows") for row in manifest[key]]
        outer, inner = groups[:2]
        train = [row for row in data["exact"] if row.example.group not in {outer, inner}]
        validation = [row.example for row in data["exact"] if row.example.group == inner]
        mean, scale = expanded.base.fit_scaler([row.example for row in train])
        report["chunkChecks"] = chunk_checks(data, mean, scale, expanded, weighting)
        replay_contract = {"purpose": "uniform implementation-equivalence preflight only",
                           "historicalContractSha256": registration["sha256"], "manifestSha256": contract["manifestSha256"],
                           "code": current_code, "environment": environment, "weighting": "uniform",
                           "outerIndex": 0, "innerIndex": 0, "seed": 3407, "epochs": [5]}
        replay_hash = runner.canonical_hash(replay_contract)
        report["controlContract"] = replay_contract
        report["controlContractSha256"] = replay_hash
        write_new(args.output/"control-contract.json", {"sha256": replay_hash, "contract": replay_contract})
        source_dir = args.output/"sources"
        source_dir.mkdir()
        for name in current_code:
            with (source_dir/name).open("xb") as f:
                f.write((REPO/"analysis"/name).read_bytes())
        report["codeSnapshots"] = [identity(source_dir/name) for name in current_code]
        for cohort in expanded.COHORTS:
            print(f"UNIFORM REPLAY {cohort} seed3407 outer0 inner0 epoch5", flush=True)
            auxiliary = expanded.auxiliary_for_fold(data, cohort, {outer, inner})
            reference = args.reference_study/"fits"/cohort/"tcn/3407/outer-0/inner-0"
            meta = read(reference/"completed.json")
            expected = {"trainIds": [r.example.id for r in train],
                        "auxiliaryIds": {tier: [r.example.id for r in rows] for tier, rows in auxiliary.items()},
                        "validationIds": [e.id for e in validation]}
            require(meta["contractSha256"] == registration["sha256"] and meta["kind"] == "tcn"
                    and meta["seed"] == 3407 and meta["epochs"] == list(expanded.EPOCHS)
                    and all(meta[k] == v for k, v in expected.items()), "reference fit identity changed")
            for name in ("weights-5.npz", "predictions-5.npz"):
                require(sha256(reference/name) == meta["artifacts"][name], "stored reference epoch5 artifact changed")
            destination = args.output/"control-replay"/cohort
            runner.fit_model(train, auxiliary, validation, "tcn", 3407, (5,), destination, "cuda", replay_hash, weighting="uniform")
            replay_meta = read(destination/"completed.json")
            comparison = {"cohort": cohort, "referenceCompleted": identity(reference/"completed.json"),
                          "replayCompleted": identity(destination/"completed.json"), "membership": expected,
                          "weights": compare_npz(reference/"weights-5.npz", destination/"weights-5.npz"),
                          "predictions": compare_npz(reference/"predictions-5.npz", destination/"predictions-5.npz"),
                          "historyBitExact": replay_meta["history"] == meta["history"][:5],
                          "exposureBitExact": replay_meta["exposureSha256"] == meta["history"][4]["exposureSha256"],
                          "optimizerStepsIdentical": replay_meta["optimizerSteps"] == meta["history"][4]["optimizerSteps"]}
            report["comparisons"].append(comparison)
            require(comparison["weights"]["bitExact"] and comparison["predictions"]["bitExact"]
                    and comparison["historyBitExact"] and comparison["exposureBitExact"]
                    and comparison["optimizerStepsIdentical"], f"uniform replay mismatch: {cohort}")
            print(f"PASS {cohort}: weights, scalers, predictions, history and sampling exposure bit-exact", flush=True)
        require(all(sha256(REPO/"analysis"/name) == digest for name, digest in current_code.items()),
                "source changed during preflight")
        require(sha256(args.manifest) == contract["manifestSha256"] and sha256(source) == report["script"]["sha256"],
                "manifest or preflight script changed during replay")
        report["passed"] = True
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["completedAt"] = datetime.now(timezone.utc).isoformat()
        report_path = args.output/"preflight-report.json"
        write_new(report_path, report)
        print(json.dumps({"passed": report["passed"], "report": identity(report_path),
                          "completedComparisons": len(report["comparisons"])}), flush=True)


if __name__ == "__main__":
    main()
