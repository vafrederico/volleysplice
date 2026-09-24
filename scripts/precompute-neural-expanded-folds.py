#!/usr/bin/env python3
"""Schedule one reviewed-export architecture/seed's twelve frozen inner fits.

This worker performs no checkpoint/decoder selection, refits, or evaluation.
The main study later verifies and resumes the completed fit artifacts normally.
It stops before another fit if the main runner approaches the same arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
COHORT, KIND, SEED = "reviewed_export", "tcn", 20260918


def read(path):
    return json.loads(path.read_text())


def write_new(path, payload):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main_near_target(study):
    """The main runner's first reviewed-export linear seed precedes all TCNs.

    A separate worker may own the last linear seed's inner folds. Its directory
    is therefore not evidence that the sequential main runner reached this arm.
    """
    if KIND == "linear":
        # Another worker owns this draft seed's inner fits, but only the main
        # process creates its refits. Stop with the final draft seed as buffer.
        markers = [study / "report.json"]
        markers.extend((study / "fits" / "draft" / "tcn" / "20260918").glob("outer-*/refit"))
    else:
        markers = [study / "report.json", study / "fits" / COHORT / "linear" / "3407"]
    return [str(path) for path in markers if path.exists()]


def main():
    global SEED, KIND
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--record-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(20260918, 1729, 3407), default=20260918)
    parser.add_argument("--kind", choices=("linear", "tcn"), default="tcn")
    args = parser.parse_args()
    SEED = args.seed
    KIND = args.kind
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import numpy as np
    import torch
    from analysis import neural_expanded_development as runner

    torch.set_num_threads(2)
    manifest, study = args.manifest.resolve(), args.study.resolve()
    registration = read(study / "preregistration.json")
    contract, contract_hash = registration["contract"], registration["sha256"]
    if hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest() != contract_hash:
        raise ValueError("invalid preregistration digest")
    if runner.base.file_sha256(manifest) != contract["manifestSha256"]:
        raise ValueError("manifest bytes changed")
    for name, expected in contract["code"].items():
        if runner.base.file_sha256(ROOT / "analysis" / name) != expected:
            raise ValueError(f"frozen implementation changed: {name}")
    environment = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                   "device": "cuda", "gpu": torch.cuda.get_device_name()}
    if environment != contract["environment"]:
        raise ValueError("worker environment differs from registered study")
    if (contract["cohorts"] != list(runner.COHORTS) or contract["kinds"] != list(runner.KINDS)
            or contract["seeds"] != list(runner.base.SEEDS) or SEED not in contract["seeds"]
            or contract["checkpointEpochs"] != list(runner.EPOCHS)):
        raise ValueError("unexpected frozen arm ordering or epoch grid")
    markers = main_near_target(study)
    if markers:
        raise RuntimeError(f"main runner already approaches target arm: {markers}")
    data = runner.load_data(manifest)
    groups = sorted({r.example.group for r in data["exact"]})
    if groups != contract["groups"] or len(groups) != 4:
        raise ValueError("exact group order changed")
    if len(data["exact"]) != contract["evaluationPopulation"]["records"]:
        raise ValueError("exact recording count changed")
    folds = []
    for outer_index, outer in enumerate(groups):
        fitting = [r for r in data["exact"] if r.example.group != outer]
        for inner_index, inner in enumerate(g for g in groups if g != outer):
            train = [r for r in fitting if r.example.group != inner]
            validation = [r.example for r in fitting if r.example.group == inner]
            auxiliary = runner.auxiliary_for_fold(data, COHORT, {outer, inner})
            all_training = train + [r for rows in auxiliary.values() for r in rows]
            if {r.example.group for r in all_training} & {outer, inner}:
                raise ValueError("auxiliary fold contamination")
            destination = study / "fits" / COHORT / KIND / str(SEED) / f"outer-{outer_index}" / f"inner-{inner_index}"
            membership = {"outer": outer, "inner": inner, "path": str(destination),
                          "trainIds": [r.example.id for r in train],
                          "auxiliaryIds": {tier: [r.example.id for r in rows] for tier, rows in auxiliary.items()},
                          "validationIds": [r.id for r in validation]}
            folds.append((destination, train, auxiliary, validation, membership))
    if len(folds) != 12:
        raise ValueError("unexpected inner-fit count")
    args.record_output.mkdir(parents=True, exist_ok=False)
    # Preserve exactly the scheduling code whose hash is recorded, even if a
    # later scheduling-only version is introduced while this process runs.
    with (args.record_output / "worker-source.py").open("xb") as handle:
        handle.write(Path(__file__).read_bytes())
    write_new(args.record_output / "execution.json", {
        "schemaVersion": 1, "purpose": "Scheduling only: frozen last-arm inner fits, no selection or refits",
        "createdAt": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
        "worker": {"path": str(Path(__file__).resolve()), "sha256": runner.base.file_sha256(Path(__file__))},
        "workerSnapshot": str(args.record_output / "worker-source.py"),
        "manifestSha256": contract["manifestSha256"], "contractSha256": contract_hash,
        "environment": environment, "torchThreads": torch.get_num_threads(),
        "cublasWorkspaceConfig": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "cohort": COHORT, "kind": KIND, "seed": SEED, "epochs": runner.EPOCHS,
        "folds": [item[4] for item in folds],
    })
    completed, status = [], "running"
    started = time.perf_counter()
    with (args.record_output / "worker.log").open("x", encoding="utf-8") as log_handle:
        def log(message):
            line = f"{datetime.now(timezone.utc).isoformat()} {message}"
            print(line, flush=True)
            log_handle.write(line + "\n")
            log_handle.flush()

        try:
            for destination, train, auxiliary, validation, membership in folds:
                markers = main_near_target(study)
                if markers:
                    status = "stopped-main-near-target"
                    log(f"STOP main progress markers: {markers}")
                    break
                if destination.exists() and not (destination / "completed.json").exists():
                    raise FileExistsError(f"refusing existing incomplete destination: {destination}")
                action = "verify-resume" if destination.exists() else "fit"
                log(f"START {action} {destination}")
                before = time.perf_counter()
                runner.fit_model(train, auxiliary, validation, KIND, SEED, runner.EPOCHS,
                                 destination, "cuda", contract_hash)
                metadata = read(destination / "completed.json")
                if any(metadata[key] != membership[key] for key in ("trainIds", "auxiliaryIds", "validationIds")):
                    raise ValueError("completed fit membership differs from planned fold")
                completed.append({"path": str(destination), "action": action,
                                  "completedSha256": runner.base.file_sha256(destination / "completed.json"),
                                  "wallSeconds": time.perf_counter()-before})
                log(f"DONE {len(completed)}/12 wallSeconds={completed[-1]['wallSeconds']:.3f}")
            else:
                status = "completed-all-twelve-inner-fits"
        except BaseException as error:
            status = "failed"
            log(f"FAIL {type(error).__name__}: {error}")
            raise
        finally:
            write_new(args.record_output / "outcome.json", {
                "status": status, "completedAt": datetime.now(timezone.utc).isoformat(),
                "wallSeconds": time.perf_counter()-started, "completed": completed,
                "contractSha256": contract_hash,
            })
            log(f"EXIT {status} completed={len(completed)}")


if __name__ == "__main__":
    main()
