#!/usr/bin/env python3
"""Independent endpoint-sweep audit of strict99% saved-score selection.

Replays every candidate's inner metric and all available held-fold padding and
coverage metrics. Only the pre-existing neural decoder is shared with selection.
No training, new candidate search, model promotion or GPU access occurs.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value

from analysis import neural_development as base
from analysis.neural_recall_operating_point import (
    Evidence, collect_fold_scores, digest, identity, load_study, model_specs,
    read, require, serial_rows, write_new,
)

HELPER_SHA = "8ed1ed75ebc2511c9603642c0b09ba5888554910072a6e1d1c5490a65864b81b"


def independent_helper():
    path = REPO/"scripts/audit-neural-short-boost-intervals.py"
    require(digest(path) == HELPER_SHA, "Independent endpoint-sweep implementation changed")
    spec = importlib.util.spec_from_file_location("independent_recall_intervals", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def audit(report_path, output, root):
    report = read(report_path)
    helper, helper_path = independent_helper()
    evidence = Evidence()
    evidence.bind(report_path)
    evidence.bind(helper_path, HELPER_SHA)
    for binding in report["references"]:
        evidence.bind(binding["path"], binding["sha256"])
    manifest = read(root/"2026-09-19-short-boost-transfer/manifest-pts-v1.json")
    examples = base.load_examples(Path(manifest["exactManifest"]["path"]), False)
    groups = sorted({e.group for e in examples})
    specs = {spec["name"]: spec for spec in model_specs(root)}
    counts = {"modelSeedCells": 0, "outerSelections": 0, "innerCandidateMetrics": 0,
              "availableHeldModeScopes": 0, "completeModeCells": 0}
    for result in report["results"]:
        spec = specs[result["model"]]
        registration, original_results = load_study(spec, evidence)
        original = next(row for row in original_results if row["seed"] == result["seed"])
        counts["modelSeedCells"] += 1
        for outer_index, fold in enumerate(result["folds"]):
            outer = groups[outer_index]
            require(fold["heldSourceGroup"] == outer, "Fold source ordering differs")
            print(f"AUDIT99 {result['model']} seed={result['seed']} outer={outer}", flush=True)
            inner, _ = collect_fold_scores(spec, result["seed"], outer_index, examples, groups,
                                          registration["sha256"], evidence)
            fitting = [e for e in examples if e.group != outer]
            checked = []
            for candidate in fold["candidates"]:
                rows = serial_rows([e.row(base.decode(e, inner[candidate["epoch"]][e.id], candidate["decoder"]))
                                    for e in fitting])
                parsed = [helper.parse_record(row) for row in rows]
                metric = helper.pooled_metric(parsed, 2.)
                helper.close(candidate["innerR_core"], metric["R_core"], "candidate core recall")
                helper.close(candidate["innerF1_padP_coreR"], metric["F1_padP_coreR"], "candidate primary F1")
                checked.append({**candidate, "innerR_core": metric["R_core"],
                                "innerF1_padP_coreR": metric["F1_padP_coreR"]})
                counts["innerCandidateMetrics"] += 1
            require(len(checked) == 192, "Candidate grid count differs")
            for mode, decision in fold["decisions"].items():
                pool = checked if mode == "joint" else [row for row in checked if row["epoch"] == fold["oldSelection"]["epoch"]]
                eligible = [row for row in pool if row["innerR_core"] >= .99]
                require(decision["feasible"] == bool(eligible)
                        and decision["eligibleCandidateCount"] == len(eligible)
                        and decision["candidateCount"] == len(pool), "Strict-floor eligibility differs")
                if eligible:
                    best = max(eligible, key=lambda row: row["innerF1_padP_coreR"])
                    require(decision["selected"]["epoch"] == best["epoch"]
                            and decision["selected"]["decoder"] == best["decoder"], "Selected candidate differs")
                else:
                    require(decision["selected"] is None and decision["outerStatus"] == "infeasible-inner-recall",
                            "Infeasible floor was relaxed")
                if decision["outerStatus"].startswith("available"):
                    rows = [row for row in result["modes"][mode]["predictions"] if row["sourceGroup"] == outer]
                    helper.compare_scope([helper.parse_record(row) for row in rows], decision["heldEvaluation"], mode+" held")
                    counts["availableHeldModeScopes"] += 1
            old = next(row for row in original["selections"] if row["heldSourceGroup"] == outer)
            require(fold["oldSelection"] == old, "Original95% comparison was changed")
            counts["outerSelections"] += 1
        for mode in result["modes"].values():
            complete = {row["id"] for row in mode["predictions"]} == {e.id for e in examples}
            require(mode["completeEvaluationScope"] == complete, "Mode scope declaration differs")
            if complete:
                helper.compare_scope([helper.parse_record(row) for row in mode["predictions"]], mode["evaluation"], "complete mode")
                counts["completeModeCells"] += 1
            else:
                require(mode["evaluation"] is None, "Partial population was ranked as a complete model")
    evidence.verify_final()
    write_new(output, {"kind": "independent-strict-recall-operating-point-audit-v1", "passed": True,
                       "report": identity(report_path), "counts": counts,
                       "code": identity(Path(__file__)), "independentArithmetic": identity(helper_path),
                       "references": list(evidence.files.values()), "newTrainingPerformed": False,
                       "protectedTestOpened": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(private_value('private-reference-0057')))
    args = parser.parse_args()
    audit(args.report, args.output, args.root)


if __name__ == "__main__":
    main()
