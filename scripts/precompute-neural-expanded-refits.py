#!/usr/bin/env python3
"""Schedule registered reviewed-export refits from completed inner predictions.

Uses the frozen study's exact inner selection and fitting functions. This helper
never writes pooled result files, ranks outer outcomes, or changes the recipe.
Each selection is recorded for exact comparison with the main runner later.
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
COHORT = "reviewed_export"


def read(path):
    return json.loads(path.read_text())


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_new(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main_progress_markers(study, kind):
    del kind  # Both workers use the same conservative upstream main marker.
    markers = [study / "report.json"]
    # Auxiliary workers create only draft inner fits, never these refits.
    markers.extend((study / "fits" / "draft" / "tcn" / "20260918").glob("outer-*/refit"))
    return [str(path) for path in markers if path.exists()]


def outer_plan(data, groups, kind, seed, outer_index, study):
    from analysis import neural_expanded_development as runner

    outer = groups[outer_index]
    fitting = [row for row in data["exact"] if row.example.group != outer]
    held = [row.example for row in data["exact"] if row.example.group == outer]
    inners = []
    for inner_index, inner in enumerate(group for group in groups if group != outer):
        training = [row for row in fitting if row.example.group != inner]
        validation = [row.example for row in fitting if row.example.group == inner]
        auxiliary = runner.auxiliary_for_fold(data, COHORT, {outer, inner})
        path = study / "fits" / COHORT / kind / str(seed) / f"outer-{outer_index}" / f"inner-{inner_index}"
        inners.append((path, training, auxiliary, validation))
    auxiliary = runner.auxiliary_for_fold(data, COHORT, {outer})
    destination = study / "fits" / COHORT / kind / str(seed) / f"outer-{outer_index}" / "refit"
    return outer, fitting, held, auxiliary, inners, destination


def inner_predictions(inners, kind, seed, contract_hash, epochs):
    import numpy as np
    from analysis import neural_expanded_development as runner

    predictions = {epoch: {} for epoch in epochs}
    bindings = []
    for path, train, auxiliary, validation in inners:
        if not (path / "completed.json").exists():
            raise FileNotFoundError(f"required inner fit is incomplete: {path}")
        metadata = read(path / "completed.json")
        expected_artifacts = {f"{stem}-{epoch}.npz" for stem in ("weights", "predictions") for epoch in epochs}
        if set(metadata["artifacts"]) != expected_artifacts:
            raise ValueError("inner artifact inventory differs from checkpoint grid")
        # This path is required complete: fit_model can only verify and resume.
        outputs = runner.fit_model(train, auxiliary, validation, kind, seed, epochs,
                                   path, "cuda", contract_hash)
        expected_ids = {row.id for row in validation}
        for epoch in epochs:
            if set(outputs[epoch]) != expected_ids or predictions[epoch].keys() & expected_ids:
                raise ValueError("inner OOF predictions duplicate or omit an exact recording")
            for row in validation:
                values = outputs[epoch][row.id]
                if (values.shape != (len(row.times), 4) or values.dtype != np.float32
                        or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1):
                    raise ValueError("invalid four-head inner probabilities")
            predictions[epoch].update(outputs[epoch])
        bindings.append({"path": str(path), "completedSha256": runner.base.file_sha256(path / "completed.json"),
                         "artifacts": metadata["artifacts"]})
    return predictions, bindings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--record-output", type=Path, required=True)
    parser.add_argument("--kind", choices=("linear", "tcn"), required=True)
    args = parser.parse_args()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import numpy as np
    import torch
    from analysis import neural_expanded_development as runner

    torch.set_num_threads(2)
    manifest, study = args.manifest.resolve(), args.study.resolve()
    registration = read(study / "preregistration.json")
    contract, contract_hash = registration["contract"], registration["sha256"]
    if canonical_hash(contract) != contract_hash:
        raise ValueError("invalid registered contract hash")
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
            or contract["seeds"] != list(runner.base.SEEDS) or contract["checkpointEpochs"] != list(runner.EPOCHS)
            or contract["decoderCandidates"] != runner.decoder_candidates()):
        raise ValueError("registered fitting or selection grid differs")
    if markers := main_progress_markers(study, args.kind):
        raise RuntimeError(f"main runner already approaches refit arm: {markers}")
    data = runner.load_data(manifest)
    groups = sorted({row.example.group for row in data["exact"]})
    if groups != contract["groups"] or len(groups) != 4:
        raise ValueError("exact source-group order changed")
    plans = [(seed, index, outer_plan(data, groups, args.kind, seed, index, study))
             for seed in runner.base.SEEDS for index in range(len(groups))]
    # Fail before any selection/fitting if required inputs or destination state
    # are incomplete. Workers never repair another process's in-progress fit.
    for _, _, (_, _, _, _, inners, destination) in plans:
        for path, *_ in inners:
            if not (path / "completed.json").exists():
                raise FileNotFoundError(f"missing completed inner fit: {path}")
        if destination.exists() and not (destination / "completed.json").exists():
            raise FileExistsError(f"refusing incomplete outer destination: {destination}")
    args.record_output.mkdir(parents=True, exist_ok=False)
    with (args.record_output / "worker-source.py").open("xb") as handle:
        handle.write(Path(__file__).read_bytes())
    write_new(args.record_output / "execution.json", {
        "schemaVersion": 1, "purpose": "Scheduling registered refits; select on exact inner OOF only",
        "createdAt": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
        "worker": {"path": str(Path(__file__).resolve()), "sha256": runner.base.file_sha256(Path(__file__))},
        "workerSnapshot": str(args.record_output / "worker-source.py"),
        "manifestSha256": contract["manifestSha256"], "contractSha256": contract_hash,
        "environment": environment, "torchThreads": torch.get_num_threads(), "cohort": COHORT,
        "kind": args.kind, "seeds": runner.base.SEEDS, "groups": groups,
        "selectionSource": "unchanged runner.choose_settings over hash-verified completed inner OOF",
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
            for seed, index, (outer, fitting, held, auxiliary, inners, destination) in plans:
                if markers := main_progress_markers(study, args.kind):
                    status = "stopped-main-near-target"
                    log(f"STOP main progress markers: {markers}")
                    break
                log(f"SELECT kind={args.kind} seed={seed} outer={outer}")
                predictions, bindings = inner_predictions(inners, args.kind, seed, contract_hash, runner.EPOCHS)
                if any(set(values) != {row.example.id for row in fitting} for values in predictions.values()):
                    raise ValueError("OOF selection population differs from outer-fitting exact records")
                selected = runner.choose_settings([row.example for row in fitting], predictions)
                selection = {"heldSourceGroup": outer, **selected}
                selection_hash = canonical_hash(selection)
                selection_path = args.record_output / f"selection-{args.kind}-{seed}-outer-{index}.json"
                write_new(selection_path, {"contractSha256": contract_hash, "kind": args.kind, "seed": seed,
                                          "outerIndex": index, "selection": selection,
                                          "selectionSha256": selection_hash, "innerFits": bindings})
                if markers := main_progress_markers(study, args.kind):
                    status = "stopped-main-near-target"
                    log(f"STOP after selection; main progress markers: {markers}")
                    break
                if destination.exists() and not (destination / "completed.json").exists():
                    raise FileExistsError(f"refusing incomplete outer destination: {destination}")
                log(f"REFIT kind={args.kind} seed={seed} outer={outer} epoch={selected['epoch']} selectionSha256={selection_hash}")
                before = time.perf_counter()
                runner.fit_model(fitting, auxiliary, held, args.kind, seed, (selected["epoch"],),
                                 destination, "cuda", contract_hash)
                completed.append({"path": str(destination), "selectionPath": str(selection_path),
                                  "selectionSha256": selection_hash,
                                  "completedSha256": runner.base.file_sha256(destination / "completed.json"),
                                  "wallSeconds": time.perf_counter()-before})
                log(f"DONE {len(completed)}/12 fitSeconds={completed[-1]['wallSeconds']:.3f}")
            else:
                status = "completed-all-twelve-refits"
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
