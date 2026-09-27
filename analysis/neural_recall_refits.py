"""Execute only preregistered missing outer checkpoints and finalize99% scores.

Invoked explicitly after CPU selection and a resource grant. Original study
directories are never written. Selected and original epochs are both saved;
the original epoch must exactly reproduce its old weights and predictions.
"""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import numpy as np

from . import neural_development as base
from . import neural_expanded_development as expanded
from . import neural_short_boost_transfer as historical
from .neural_recall_operating_point import (
    Evidence, canonical, identity, read, require, serial_rows, write_new,
)
from .neural_evaluation import evaluate_predictions


def task_name(task):
    return f"{task['model']}/seed-{task['seed']}/outer-{task['outerIndex']}"


def original_refit(task):
    root = Path(task["study"])/"fits"
    if task["layout"] == "legacy":
        return root/"reviewed_export"/task["kind"]/"short_boost"/str(task["seed"])/f"outer-{task['outerIndex']}"/"refit"
    return root/str(task["seed"])/f"outer-{task['outerIndex']}"


def identical_npz(left, right):
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        require(set(a.files) == set(b.files), "Checkpoint archive keys differ")
        for name in a.files:
            require(a[name].dtype == b[name].dtype and a[name].shape == b[name].shape
                    and np.array_equal(a[name], b[name]), "Original-epoch exact replay differs: "+name)


def verify_plan(plan_path):
    evidence = Evidence()
    plan = read(evidence.bind(plan_path))
    require(plan["kind"] == "strict-recall-missing-outer-refits-v1"
            and plan["newSettingsSelectedFromOuter"] is False, "Not a frozen missing-checkpoint plan")
    protocol = read(evidence.bind(plan["protocol"]["path"], plan["protocol"]["sha256"]))
    for path, sha in protocol["code"].items():
        evidence.bind(path, sha)
    for task in plan["tasks"]:
        registration = read(evidence.bind(Path(task["study"])/"preregistration.json"))
        require(registration["sha256"] == task["originalContractSha256"]
                and canonical(registration["contract"]) == registration["sha256"], "Original registration differs")
        for name, sha in registration["contract"]["code"].items():
            evidence.bind(Path(__file__).parent/name, sha)
    return plan, evidence


def fit_missing(plan_path, destination, device="cuda"):
    import torch
    from .neural_recognition_fit import fit_model
    from .recognition_temporal_model import RecognitionConfig
    from .neural_recognition_inputs import attach_features

    plan, evidence = verify_plan(plan_path)
    require(not device.startswith("cuda") or os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
            "Set CUBLAS_WORKSPACE_CONFIG=:4096:8")
    torch.set_num_threads(4)
    data_cache = {}
    for task in plan["tasks"]:
        print("REFIT99 "+task_name(task), flush=True)
        registration = read(Path(task["study"])/"preregistration.json")
        contract = registration["contract"]
        cache_key = task["model"]
        if cache_key not in data_cache:
            # One model's data in memory at a time; do not retain all visual caches.
            data_cache.clear()
            manifest = Path(contract["manifest"]["path"]) if task["layout"] == "recognition" else historical.MANIFEST
            dino = contract.get("dinoManifest")
            dino_path = Path(dino["path"]) if dino else None
            with_dino = task.get("kind") == "dino_tcn" or contract.get("config", {}).get("family") == "dino"
            data = historical.load_data(manifest, dino_path, with_dino=with_dino)
            config = RecognitionConfig(**contract["config"]) if task["layout"] == "recognition" else None
            if config and config.family in ("mobile", "player"):
                feature = contract["featureManifest"]
                evidence.bind(feature["path"], feature["sha256"])
                data = attach_features(data, Path(feature["path"]), config, manifest)
            data_cache[cache_key] = data, config
        data, config = data_cache[cache_key]
        group = task["heldSourceGroup"]
        train = [row for row in data["exact"] if row.example.group != group]
        held = [row.example for row in data["exact"] if row.example.group == group]
        auxiliary = expanded.auxiliary_for_fold(data, "reviewed_export", {group})
        folder = destination/task_name(task)
        epochs = tuple(task["checkpointEpochs"])
        if task["layout"] == "legacy":
            historical.fit_model(train, auxiliary, held, task["kind"], task["seed"], epochs,
                                 folder, device, task["originalContractSha256"], "short_boost")
        else:
            fit_model(train, auxiliary, held, config, task["seed"], epochs,
                      folder, device, task["originalContractSha256"], "short_boost")
        old = original_refit(task)
        old_metadata = read(evidence.bind(old/"completed.json"))
        for stem in ("weights", "predictions"):
            name = f"{stem}-{task['originalEpoch']}.npz"
            evidence.bind(old/name, old_metadata["artifacts"][name])
            identical_npz(folder/name, old/name)
        metadata = read(folder/"completed.json")
        # Sampling/exposure histories also reproduce at the overlapping epoch.
        shared = task["originalEpoch"]
        require(metadata["history"][shared-1] == old_metadata["history"][shared-1], "Original-epoch history/exposure differs")
        parity_path = folder/"original-parity.json"
        parity = {"passed": True, "originalEpoch": shared, "plan": identity(plan_path),
                  "task": task, "completed": identity(folder/"completed.json"),
                  "originalCompleted": identity(old/"completed.json"),
                  "weightsAndPredictionsExactlyEqual": True, "samplingHistoryExactlyEqual": True}
        if parity_path.exists():
            require(read(parity_path) == parity, "Existing parity proof differs")
        else:
            write_new(parity_path, parity)
    evidence.verify_final()
    completion = {"kind": "strict-recall-missing-outer-refits-completion-v1", "plan": identity(plan_path),
                  "tasks": len(plan["tasks"]), "references": list(evidence.files.values()),
                  "newSelectionPerformed": False, "protectedTestOpened": False}
    write_new(destination/"completion.json", completion)


def finalize(selection_path, plan_path, refit_root, output):
    plan, evidence = verify_plan(plan_path)
    selection = read(evidence.bind(selection_path))
    require(selection["protocol"] == plan["protocol"], "Selection/plan protocol differs")
    for binding in selection["references"]:
        evidence.bind(binding["path"], binding["sha256"])
    manifest = read(historical.MANIFEST)
    examples = base.load_examples(Path(manifest["exactManifest"]["path"]), False)
    report = copy.deepcopy(selection)
    for task in plan["tasks"]:
        result = next(row for row in report["results"] if row["model"] == task["model"] and row["seed"] == task["seed"])
        fold = next(row for row in result["folds"] if row["heldSourceGroup"] == task["heldSourceGroup"])
        decision = fold["decisions"]["joint"]
        require(decision["feasible"] and decision["selected"] == task["selected"]
                and decision["outerStatus"] == "missing-selected-outer-checkpoint", "Refit was not the frozen missing selection")
        folder = refit_root/task_name(task)
        parity = read(evidence.bind(folder/"original-parity.json"))
        require(parity["passed"] and parity["task"] == task and parity["plan"] == identity(plan_path), "Missing exact original-checkpoint parity")
        metadata = read(evidence.bind(folder/"completed.json", parity["completed"]["sha256"]))
        name = f"predictions-{task['selected']['epoch']}.npz"
        path = evidence.bind(folder/name, metadata["artifacts"][name])
        held = [e for e in examples if e.group == task["heldSourceGroup"]]
        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == {e.id for e in held}, "Supplemental outer prediction population differs")
            rows = serial_rows([e.row(base.decode(e, archive[e.id], task["selected"]["decoder"])) for e in held])
        decision["outerStatus"] = "available-from-parity-verified-refit"
        decision["heldEvaluation"] = evaluate_predictions(rows)
        decision["supplementalCheckpoint"] = identity(path)
        result["modes"]["joint"]["predictions"].extend(rows)
    for result in report["results"]:
        mode = result["modes"]["joint"]
        order = {e.id: index for index, e in enumerate(examples)}
        mode["predictions"].sort(key=lambda row: order[row["id"]])
        mode["scopeRecordingIds"] = [row["id"] for row in mode["predictions"]]
        mode["completeEvaluationScope"] = set(mode["scopeRecordingIds"]) == set(order)
        mode["evaluation"] = evaluate_predictions(mode["predictions"]) if mode["completeEvaluationScope"] else None
    evidence.verify_final()
    report["status"] = "completed-strict-recall-with-available-refits"
    report["newTrainingPerformed"] = bool(plan["tasks"])
    report["selectionReport"] = identity(selection_path)
    report["refitPlan"] = identity(plan_path)
    report["references"] = list(evidence.files.values())
    write_new(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fit", "finalize"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--refit-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "fit":
        fit_missing(args.plan, args.refit_root, args.device)
    else:
        require(args.selection and args.output, "Finalize requires --selection and --output")
        finalize(args.selection, args.plan, args.refit_root, args.output)


if __name__ == "__main__":
    main()
