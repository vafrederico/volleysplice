#!/usr/bin/env python3
"""Schedule only draft/tcn/20260918 frozen inner fits.

Calls unchanged fit_model on twelve canonical inner paths, stopping before
main begins the previous TCN seed. No selection, refits or bound-code edits.
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
COHORT, KIND, SEED = "draft", "tcn", 20260918


def read(path):
    return json.loads(path.read_text())


def write_new(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def main_progress_markers(study):
    # The previous draft TCN seed is owned solely by main. Our own last-seed
    # inner directories do not constitute evidence that main is approaching.
    paths = [study / "report.json", study / "fits" / COHORT / KIND / "1729"]
    return [str(p) for p in paths if p.exists()]


def main():
    global SEED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--record-output", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--fit-count", type=int, default=12)
    parser.add_argument("--seed", type=int, choices=(20260918,), default=20260918)
    args = parser.parse_args()
    SEED = args.seed
    if args.start_index < 0 or args.fit_count < 1 or args.start_index + args.fit_count > 12:
        raise ValueError("Requested slice must stay within the twelve inner fits")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import numpy as np
    import torch
    from analysis import neural_expanded_development as runner
    torch.set_num_threads(2)
    manifest, study = args.manifest.resolve(), args.study.resolve()
    registration = read(study / "preregistration.json")
    contract, contract_hash = registration["contract"], registration["sha256"]
    if hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest() != contract_hash:
        raise ValueError("Invalid registered contract hash")
    if runner.base.file_sha256(manifest) != contract["manifestSha256"]:
        raise ValueError("Changed manifest")
    for name, expected in contract["code"].items():
        if runner.base.file_sha256(REPO / "analysis" / name) != expected:
            raise ValueError(f"Changed bound implementation: {name}")
    environment = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                   "device": "cuda", "gpu": torch.cuda.get_device_name()}
    if environment != contract["environment"]:
        raise ValueError("Worker environment differs from preregistration")
    if (contract["cohorts"] != list(runner.COHORTS) or contract["kinds"] != list(runner.KINDS)
            or contract["seeds"] != list(runner.base.SEEDS)
            or contract["checkpointEpochs"] != list(runner.EPOCHS)):
        raise ValueError("Changed arm order or epoch grid")
    markers = main_progress_markers(study)
    if markers:
        raise RuntimeError(f"Main already approaches reserved arm: {markers}")
    data = runner.load_data(manifest)
    groups = sorted({r.example.group for r in data["exact"]})
    if groups != contract["groups"] or len(groups) != 4:
        raise ValueError("Changed exact group order")
    planned = []
    for outer_index, outer in enumerate(groups):
        fitting = [r for r in data["exact"] if r.example.group != outer]
        for inner_index, inner in enumerate(g for g in groups if g != outer):
            train = [r for r in fitting if r.example.group != inner]
            validation = [r.example for r in fitting if r.example.group == inner]
            auxiliary = runner.auxiliary_for_fold(data, COHORT, {outer, inner})
            all_training = train + [r for rows in auxiliary.values() for r in rows]
            if {r.example.group for r in all_training} & {outer, inner}:
                raise ValueError("Group leakage in scheduled fit")
            destination = study / "fits" / COHORT / KIND / str(SEED) / f"outer-{outer_index}" / f"inner-{inner_index}"
            membership = {"index": len(planned), "outerGroup": outer, "innerGroup": inner,
                          "destination": str(destination), "trainIds": [r.example.id for r in train],
                          "auxiliaryIds": {tier: [r.example.id for r in rows] for tier, rows in auxiliary.items()},
                          "validationIds": [r.id for r in validation]}
            planned.append((destination, train, auxiliary, validation, membership))
    if len(planned) != 12:
        raise ValueError("Expected twelve inner fits")
    selected = planned[args.start_index:args.start_index + args.fit_count]
    for destination, *_ in selected:
        if destination.exists() and not (destination / "completed.json").exists():
            raise FileExistsError(f"Refusing incomplete destination: {destination}")
    args.record_output.mkdir(parents=True, exist_ok=False)
    snapshot = args.record_output / "worker-source.py"
    with snapshot.open("xb") as f:
        f.write(Path(__file__).read_bytes())
    write_new(args.record_output / "execution.json", {
        "kind": "frozen-expanded-draft-tcn-inner-fit-scheduling-v1", "createdAt": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(), "worker": {"path": str(Path(__file__).resolve()), "sha256": runner.base.file_sha256(Path(__file__))},
        "workerSnapshot": {"path": str(snapshot), "sha256": runner.base.file_sha256(snapshot)},
        "manifestSha256": contract["manifestSha256"], "contractSha256": contract_hash,
        "environment": environment, "torchThreads": torch.get_num_threads(),
        "cublasWorkspaceConfig": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "cohort": COHORT, "kindName": KIND, "seed": SEED, "epochs": list(runner.EPOCHS),
        "scheduledFitCount": len(selected), "startIndex": args.start_index,
        "allTwelveMemberships": [item[4] for item in planned], "scheduledMemberships": [item[4] for item in selected],
        "exclusions": "No outer refits, checkpoint/decoder selection, result creation, GPU setting changes, or bound-code changes.",
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
            for destination, train, auxiliary, validation, membership in selected:
                markers = main_progress_markers(study)
                if markers:
                    status = "stopped-main-near-target"
                    log(f"STOP main progress markers: {markers}")
                    break
                if destination.exists() and not (destination / "completed.json").exists():
                    raise FileExistsError(f"Refusing incomplete destination: {destination}")
                action = "verify-resume" if destination.exists() else "fit"
                log(f"START {action} index={membership['index']} {destination}")
                before = time.perf_counter()
                runner.fit_model(train, auxiliary, validation, KIND, SEED, runner.EPOCHS,
                                 destination, "cuda", contract_hash)
                metadata = read(destination / "completed.json")
                if any(metadata[key] != membership[key] for key in ("trainIds", "auxiliaryIds", "validationIds")):
                    raise ValueError("Completed membership differs from frozen plan")
                completed.append({"path": str(destination), "index": membership["index"], "action": action,
                                  "completedSha256": runner.base.file_sha256(destination / "completed.json"),
                                  "wallSeconds": time.perf_counter() - before,
                                  "fitWallSeconds": metadata["wallSeconds"],
                                  "peakAllocatedCudaBytes": metadata.get("peakAllocatedCudaBytes")})
                log(f"DONE {len(completed)}/{len(selected)} wallSeconds={completed[-1]['wallSeconds']:.3f}")
            else:
                status = "completed-requested-inner-fits"
        except BaseException as error:
            status = "failed"
            log(f"FAIL {type(error).__name__}: {error}")
            raise
        finally:
            write_new(args.record_output / "outcome.json", {
                "status": status, "completedAt": datetime.now(timezone.utc).isoformat(),
                "wallSeconds": time.perf_counter() - started, "completed": completed,
                "contractSha256": contract_hash,
            })
            log(f"EXIT {status} completed={len(completed)}")


if __name__ == "__main__":
    main()
