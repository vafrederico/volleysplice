#!/usr/bin/env python3
"""Audit supplemental checkpoints and final coverage after a passing grid audit.

The unchanged 11,520-candidate search is bound to its independent prior audit.
This extension checks every supplemental artifact, exact old-epoch numerical
parity, source membership, saved-score decoding, and independent interval math.
It does not perform a second neural forward pass at the new epoch.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis import neural_development as base
from analysis.neural_recall_operating_point import Evidence, digest, identity, read, require, serial_rows, write_new
from analysis.neural_recall_refits import original_refit, task_name


def helper():
    path = REPO/"scripts/audit-neural-short-boost-intervals.py"
    require(digest(path) == "8ed1ed75ebc2511c9603642c0b09ba5888554910072a6e1d1c5490a65864b81b",
            "Independent interval helper changed")
    spec = importlib.util.spec_from_file_location("refit_interval_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def audit(report_path, selection_audit_path, refit_root, output):
    interval, interval_path = helper()
    evidence = Evidence()
    report = read(evidence.bind(report_path))
    prior_audit = read(evidence.bind(selection_audit_path))
    require(prior_audit["passed"] is True and prior_audit["counts"]["innerCandidateMetrics"] == 11520,
            "Complete independent grid audit required")
    prior = read(evidence.bind(prior_audit["report"]["path"], prior_audit["report"]["sha256"]))
    require(report["selectionReport"] == prior_audit["report"], "Final report changes selection parent")
    plan = read(evidence.bind(report["refitPlan"]["path"], report["refitPlan"]["sha256"]))
    for binding in report["references"]:
        evidence.bind(binding["path"], binding["sha256"])
    manifest_binding = next(row for row in prior["references"] if row["path"].endswith("/manifest-pts-v1.json"))
    manifest = read(manifest_binding["path"])
    examples = base.load_examples(Path(manifest["exactManifest"]["path"]), False)
    checks = []
    require(report["production"] == prior["production"] and report["protocol"] == prior["protocol"],
            "Production reference or frozen protocol changed")
    for previous, current in zip(prior["results"], report["results"], strict=True):
        require((previous["model"], previous["seed"]) == (current["model"], current["seed"])
                and previous["original95"] == current["original95"]
                and previous["modes"]["fixed_epoch"] == current["modes"]["fixed_epoch"], "Original or secondary results changed")
        for old_fold, fold in zip(previous["folds"], current["folds"], strict=True):
            for key in ("heldSourceGroup", "oldSelection", "outerCheckpointEpochs", "candidates"):
                require(old_fold[key] == fold[key], "Frozen selection field changed: "+key)
            require(old_fold["decisions"]["fixed_epoch"] == fold["decisions"]["fixed_epoch"], "Secondary fold changed")
            old_decision, decision = old_fold["decisions"]["joint"], fold["decisions"]["joint"]
            for key in ("recallEligibilityFloor", "feasible", "candidateCount", "eligibleCandidateCount", "maximumInnerRecall", "selected"):
                require(old_decision[key] == decision[key], "Joint choice changed: "+key)
            task = next((task for task in plan["tasks"] if task["model"] == current["model"]
                         and task["seed"] == current["seed"] and task["heldSourceGroup"] == fold["heldSourceGroup"]), None)
            if task is None:
                require(old_decision == decision, "An unplanned joint fold changed")
                continue
            require(decision["outerStatus"] == "available-from-parity-verified-refit", "Planned refit was not completed")
            folder, old_folder = refit_root/task_name(task), original_refit(task)
            metadata = read(evidence.bind(folder/"completed.json"))
            old_metadata = read(evidence.bind(old_folder/"completed.json"))
            for key in ("contractSha256", "seed", "trainIds", "auxiliaryIds", "validationIds", "scalerTrainIds", "trainGroups", "auxiliaryGroups", "validationGroups", "supervisedCounts", "positiveWeight", "liveLossWeighting"):
                require(metadata[key] == old_metadata[key], "Refit membership/objective differs: "+key)
            require(metadata["epochs"] == task["checkpointEpochs"], "Refit checkpoint epochs differ")
            require(metadata["validationGroups"] == [task["heldSourceGroup"]]
                    and task["heldSourceGroup"] not in metadata["trainGroups"]
                    and all(task["heldSourceGroup"] not in groups for groups in metadata["auxiliaryGroups"].values()),
                    "Held source entered fitting")
            prefix = min(len(metadata["history"]), len(old_metadata["history"]))
            require(metadata["history"][:prefix] == old_metadata["history"][:prefix], "Original training prefix is not exact")
            for name, sha in metadata["artifacts"].items():
                evidence.bind(folder/name, sha)
            for stem in ("weights", "predictions"):
                name = f"{stem}-{task['originalEpoch']}.npz"
                evidence.bind(old_folder/name, old_metadata["artifacts"][name])
                with np.load(folder/name, allow_pickle=False) as a, np.load(old_folder/name, allow_pickle=False) as b:
                    require(set(a.files) == set(b.files), "Original checkpoint tensor inventory differs")
                    for key in a.files:
                        require(a[key].dtype == b[key].dtype and a[key].shape == b[key].shape
                                and np.array_equal(a[key], b[key]), "Original checkpoint differs: "+key)
            selected = task["selected"]
            held = [e for e in examples if e.group == task["heldSourceGroup"]]
            checkpoint = folder/f"predictions-{selected['epoch']}.npz"
            require(decision["supplementalCheckpoint"] == identity(checkpoint), "Selected score artifact differs")
            with np.load(checkpoint, allow_pickle=False) as scores:
                require(set(scores.files) == {e.id for e in held}, "Supplemental score population differs")
                for e in held:
                    require(scores[e.id].dtype == np.float32 and scores[e.id].shape == (len(e.times), 4)
                            and np.isfinite(scores[e.id]).all() and np.all((scores[e.id] >= 0) & (scores[e.id] <= 1))
                            and np.all(scores[e.id][~e.valid] == 0), "Invalid supplemental predictions")
                rows = serial_rows([e.row(base.decode(e, scores[e.id], selected["decoder"])) for e in held])
            stored = [row for row in current["modes"]["joint"]["predictions"] if row["sourceGroup"] == task["heldSourceGroup"]]
            require(rows == stored, "Supplemental decoder output differs")
            interval.compare_scope([interval.parse_record(row) for row in rows], decision["heldEvaluation"], task_name(task))
            checks.append({"task": task, "completed": identity(folder/"completed.json"),
                           "originalNumericalPrefixEpochs": prefix, "allFourPaddingMetricsChecked": True})
        mode = current["modes"]["joint"]
        complete = {row["id"] for row in mode["predictions"]} == {e.id for e in examples}
        require(mode["completeEvaluationScope"] == complete, "Final scope flag differs")
        if complete:
            interval.compare_scope([interval.parse_record(row) for row in mode["predictions"]], mode["evaluation"], "final pooled scope")
        else:
            require(mode["evaluation"] is None, "Infeasible partial population received a full result")
    require(len(checks) == len(plan["tasks"]), "Not every planned supplemental fit was checked")
    evidence.verify_final()
    write_new(output, {"kind": "independent-strict-recall-supplemental-audit-v1", "passed": True,
                       "report": identity(report_path), "priorSelectionAudit": identity(selection_audit_path),
                       "supplementalFits": checks, "references": list(evidence.files.values()),
                       "independentArithmetic": identity(interval_path), "code": identity(Path(__file__)),
                       "neuralReplayScope": "Exact full old-epoch weights, saved probabilities and overlapping numerical history; new-epoch score shape/hash/decoder/metrics checked, no second new-epoch neural forward pass.",
                       "protectedTestOpened": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--selection-audit", type=Path, required=True)
    parser.add_argument("--refit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.report, args.selection_audit, args.refit_root, args.output)
