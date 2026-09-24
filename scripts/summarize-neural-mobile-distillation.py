#!/usr/bin/env python3
"""Describe audited student and frozen-Mobile strict99 results without partial ranks.

This reads small saved evaluations only. It does not load features, fit models,
select candidates, or run an encoder. Input reports must have passing bound audits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value

from analysis.crop_evaluation import subtract_intervals
from analysis.neural_evaluation import evaluate_predictions
from analysis.schema import Interval

SEEDS = (3407, 1729, 20260918)
PADDINGS = (0, 1, 2, 3)


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def identity(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def signature(rows):
    require(len({r["id"] for r in rows}) == len(rows), "Duplicate recording in evaluation")
    return {r["id"]: {k: r[k] for k in ("sourceGroup", "durationSeconds", "rallies", "ignoredIntervals")}
            for r in rows}


def audited_report(report_path, audit_path, kind):
    report_ref, audit_ref = identity(report_path), identity(audit_path)
    audit = read(audit_path)
    require(audit.get("kind") == kind and audit.get("passed") is True,
            "Expected passing independent audit: " + str(audit_path))
    require(audit["report"] == report_ref, "Audit does not bind current report")
    return read(report_path), audit, {"report": report_ref, "audit": audit_ref}


def universe_seconds(gold):
    return sum(sum(i.end - i.start for i in subtract_intervals(
        (Interval(0, row["durationSeconds"]),),
        tuple(Interval(i["start"], i["end"]) for i in row["ignoredIntervals"]))) for row in gold.values())


def coverage_metrics(coverage):
    def subset(rows):
        seconds = sum(r["evaluableCoreSeconds"] for r in rows)
        retained = sum(r["retainedCoreSeconds"] for r in rows)
        return {"evaluableRallies": len(rows), "completeRallyLosses": sum(r["completelyLost"] for r in rows),
                "partialRallyLosses": sum(r["partiallyLost"] for r in rows),
                "fullyCoveredRallies": sum(r["fullyCovered"] for r in rows),
                "evaluableCoreSeconds": seconds, "retainedCoreSeconds": retained,
                "coreRecall": retained / seconds if seconds else None}
    return {**{k: coverage[k] for k in ("originalRallies", "evaluableRallies", "fullyIgnoredRallies",
            "completeRallyLosses", "partialRallyLosses", "fullyCoveredRallies", "evaluableCoreSeconds",
            "retainedCoreSeconds", "coreRecall")},
            "shortOriginalRalliesAtMost3Seconds": subset([r for r in coverage["rallies"] if r["end"] - r["start"] <= 3]),
            "longOriginalRalliesOver3Seconds": subset([r for r in coverage["rallies"] if r["end"] - r["start"] > 3])}


def padding_metrics(row, universe):
    incorrect = row["paddedModelExportSeconds"] - row["paddedPrecisionIntersectionSeconds"]
    return {**{k: row[k] for k in ("paddingSecondsBeforeAndAfter", "joinGapSeconds", "P_pad", "R_core",
                "F1_padP_coreR", "paddedModelExportSeconds", "paddedHumanExportSeconds", "exportDurationDifferenceSeconds")},
            "incorrectExportSeconds": incorrect,
            "wantedHumanExportOmittedSeconds": row["paddedHumanExportSeconds"] - row["paddedPrecisionIntersectionSeconds"],
            "correctlyRemovedSeconds": universe - row["paddedHumanExportSeconds"] - incorrect,
            "missedCoreSeconds": row["coreHumanSeconds"] - row["coreRecallIntersectionSeconds"]}


def describe_evaluation(evaluation, universe):
    require([r["paddingSecondsBeforeAndAfter"] for r in evaluation["padding"]] == list(PADDINGS),
            "All four registered padding cases required")
    require(evaluation["primary"]["paddingSecondsBeforeAndAfter"] == 2
            and all(r["joinGapSeconds"] == 3 for r in evaluation["padding"]), "Padding/join contract differs")
    g = evaluation["guardrails"]
    return {"primary": padding_metrics(evaluation["primary"], universe),
            "padding": [padding_metrics(p, universe) for p in evaluation["padding"]],
            "event": {k: g[k] for k in ("trueRallies", "predictedRallies", "matchedRallies", "eventPrecision",
                "eventRecall", "eventF1", "ignoredTouchedRalliesExcludedFromEvents")},
            "primaryExportCoverage": coverage_metrics(g["primaryExportCoverage"]),
            "rawCoreCoverage": coverage_metrics(g["coreCoverage"])}


def average(values):
    first = values[0]
    if isinstance(first, dict):
        require(all(v.keys() == first.keys() for v in values), "Mean metric keys differ")
        return {k: average([v[k] for v in values]) for k in first}
    if isinstance(first, list):
        require(all(len(v) == len(first) for v in values), "Mean metric dimensions differ")
        return [average([v[i] for v in values]) for i in range(len(first))]
    if first is None:
        require(all(v is None for v in values), "Inconsistent empty population")
        return None
    return statistics.mean(values)


def describe_model(cells, gold, universe):
    require([r["seed"] for r in cells] == list(SEEDS), "All three registered seeds required in fixed order")
    groups = {r["sourceGroup"] for r in gold.values()}
    require(len(gold) == 8 and len(groups) == 4, "Expected full eight-recording/four-source evaluation")
    seeds = []
    for cell in cells:
        selections = cell["selections"]
        require(len(selections) == 4 and {s["heldSourceGroup"] for s in selections} == groups,
                "Four distinct held source groups required")
        for s in selections:
            require(s["recallEligibilityFloor"] == .99 and s["candidateCount"] == 192,
                    "Strict99 candidate contract differs")
            require(s["feasible"] == (s["eligibleCandidateCount"] > 0)
                    and (s["selected"] is not None) == s["feasible"], "Infeasible selection has fallback")
        expected_groups = {s["heldSourceGroup"] for s in selections if s["feasible"]}
        expected_gold = {key: row for key, row in gold.items() if row["sourceGroup"] in expected_groups}
        require(signature(cell["predictions"]) == expected_gold, "Gold revision or evaluated source membership differs")
        complete = all(s["feasible"] for s in selections)
        require(cell["complete"] == complete and (cell["evaluation"] is not None) == complete,
                "Incomplete source scope must have no pooled evaluation")
        if complete:
            require(evaluate_predictions(cell["predictions"]) == cell["evaluation"], "Saved metric replay differs")
        seeds.append({"seed": cell["seed"], "completeEvaluationScope": complete,
            "status": "complete" if complete else "infeasible-inner-recall",
            "metrics": describe_evaluation(cell["evaluation"], universe) if complete else None,
            "folds": [{"heldSourceGroup": s["heldSourceGroup"], "feasible": s["feasible"],
                "eligibleCandidateCount": s["eligibleCandidateCount"], "candidateCount": s["candidateCount"],
                "maximumInnerRecall": s["maximumInnerRecall"], "selected": s["selected"]} for s in selections]})
    all_complete = all(s["completeEvaluationScope"] for s in seeds)
    return {"allThreeSeedsComplete": all_complete,
            "modelRankEligible": all_complete, "completeSeeds": [s["seed"] for s in seeds if s["completeEvaluationScope"]],
            "feasibleFolds": sum(s["feasible"] for r in seeds for s in r["folds"]), "totalFolds": 12,
            "meanAcrossAllThreeSeeds": average([s["metrics"] for s in seeds]) if all_complete else None,
            "seeds": seeds}


def paired_differences(student, baseline):
    rows = []
    for a, b in zip(student["seeds"], baseline["seeds"], strict=True):
        require(a["seed"] == b["seed"], "Paired seed order differs")
        eligible = a["completeEvaluationScope"] and b["completeEvaluationScope"]
        rows.append({"seed": a["seed"], "bothCompleteEvaluationScopes": eligible,
            "studentMinusFrozenMobile": [{k: x[k] - y[k] for k in x if k not in ("paddingSecondsBeforeAndAfter", "joinGapSeconds")}
                | {"paddingSecondsBeforeAndAfter": x["paddingSecondsBeforeAndAfter"], "joinGapSeconds": 3}
                for x, y in zip(a["metrics"]["padding"], b["metrics"]["padding"], strict=True)] if eligible else None})
    complete = all(r["bothCompleteEvaluationScopes"] for r in rows)
    return {"allThreePairedSeedsComplete": complete, "seeds": rows,
            "meanAcrossAllThreeSeeds": average([r["studentMinusFrozenMobile"] for r in rows]) if complete else None,
            "interpretation": "A complete seed is descriptive. No favorable feasible-seed subset is averaged or ranked."}


def percent(value):
    return "n/a" if value is None else f"{100 * value:.2f}%"


def markdown(summary):
    lines = ["# Distilled MobileNet versus frozen MobileNet at strict 99% inner recall", "",
        "Target padding is 2 seconds before and after; positive gaps strictly below 3 seconds are retained. Ignored time is outside the evaluation universe. Inner eligibility does not guarantee 99% recall on an unseen source.", "",
        "Each full seed pools recording durations before computing precision, recall, and F1. A model mean requires all three seeds to have all four held source groups. Infeasible seeds have no relaxed fallback or partial-scope rank.", "",
        "| Model | Feasible folds | Complete seeds | Mean P / R / F1 across all three seeds |",
        "|---|---:|---|---|"]
    for name, model in summary["models"].items():
        mean = model["meanAcrossAllThreeSeeds"]
        value = " / ".join(percent(mean["primary"][k]) for k in ("P_pad", "R_core", "F1_padP_coreR")) if mean else "Unavailable: incomplete three-seed scope"
        lines.append(f"| {name} | {model['feasibleFolds']}/12 | {', '.join(map(str, model['completeSeeds'])) or 'None'} | {value} |")
    lines += ["", "## Complete seed results at the target padding", "",
        "These are full-scope descriptive seed results, including any complete seeds from a model whose three-seed result is unavailable. Times are seconds. Correctly removed time is nonignored time outside the wanted human export that was removed. Wanted human-export time omitted is an error.", "",
        "| Model | Seed | P_pad | R_core | F1_padP_coreR | Export | Correctly removed | Incorrect export | Wanted export omitted | Missed core |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, model in summary["models"].items():
        for seed in model["seeds"]:
            if not seed["completeEvaluationScope"]:
                lines.append(f"| {name} | {seed['seed']} | Infeasible | — | — | — | — | — | — | — |")
                continue
            p = seed["metrics"]["primary"]
            lines.append(f"| {name} | {seed['seed']} | " + " | ".join(percent(p[k]) for k in ("P_pad", "R_core", "F1_padP_coreR")) + " | "
                + " | ".join(f"{p[k]:.1f}" for k in ("paddedModelExportSeconds", "correctlyRemovedSeconds", "incorrectExportSeconds", "wantedHumanExportOmittedSeconds", "missedCoreSeconds")) + " |")
    lines += ["", "## Original-rally and event guardrails", "",
        "Complete loss retains no evaluable core; partial loss retains some but not all. Short/long uses the original rally duration at most/over 3 seconds, not fragments after ignored-time subtraction. Event P/R/F1 uses unpadded original, uncensored events at IoU ≥ 0.5; it is distinct from retained play recall and does not establish accurate serve boundaries.", "",
        "| Model | Seed | Fully retained / partial / complete loss | Short partial / complete loss | Long partial / complete loss | Long core recall | Event P / R / F1 |",
        "|---|---:|---|---|---|---:|---|"]
    for name, model in summary["models"].items():
        for seed in model["seeds"]:
            if not seed["completeEvaluationScope"]:
                continue
            m = seed["metrics"]
            c, e = m["primaryExportCoverage"], m["event"]
            short, long = c["shortOriginalRalliesAtMost3Seconds"], c["longOriginalRalliesOver3Seconds"]
            lines.append(f"| {name} | {seed['seed']} | {c['fullyCoveredRallies']} / {c['partialRallyLosses']} / {c['completeRallyLosses']} | {short['partialRallyLosses']} / {short['completeRallyLosses']} | {long['partialRallyLosses']} / {long['completeRallyLosses']} | {percent(long['coreRecall'])} | "
                + " / ".join(percent(e[k]) for k in ("eventPrecision", "eventRecall", "eventF1")) + " |")
    lines += ["", "## All four padding cases", "", "| Model | Seed | Padding per side | P_pad | R_core | F1_padP_coreR | Model export | Human export | Difference |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, model in summary["models"].items():
        for seed in model["seeds"]:
            if seed["completeEvaluationScope"]:
                for p in seed["metrics"]["padding"]:
                    lines.append(f"| {name} | {seed['seed']} | {p['paddingSecondsBeforeAndAfter']:g}s | " + " | ".join(percent(p[k]) for k in ("P_pad", "R_core", "F1_padP_coreR")) + " | "
                        + " | ".join(f"{p[k]:.1f}" for k in ("paddedModelExportSeconds", "paddedHumanExportSeconds", "exportDurationDifferenceSeconds")) + " |")
    lines += ["", "## Fold eligibility", "", "| Model | Seed | Held source | Eligible candidates / 192 | Maximum inner recall | Selected epoch |",
        "|---|---:|---|---:|---:|---:|"]
    for name, model in summary["models"].items():
        for seed in model["seeds"]:
            for fold in seed["folds"]:
                chosen = fold["selected"]
                lines.append(f"| {name} | {seed['seed']} | {fold['heldSourceGroup']} | {fold['eligibleCandidateCount']} | {percent(fold['maximumInnerRecall'])} | {chosen['epoch'] if chosen else 'Infeasible'} |")
    lines += ["", "The adjacent JSON includes hashes of both independent audits and reports, all duration errors at every padding, raw-core coverage, original short/long counts, and per-seed paired differences. No phone runtime or accuracy claim follows from these desktop training results.", ""]
    return "\n".join(lines)


def summarize(student_report, student_audit, registration_path, baseline_report, baseline_audit):
    student, audit, student_refs = audited_report(student_report, student_audit, "independent-mobile-distillation-audit-v1")
    registration = read(registration_path)
    registration_ref = identity(registration_path)
    require(audit["registration"] == registration_ref and student["contractSha256"] == registration["sha256"]
            == audit["contractSha256"], "Student registration binding differs")
    baseline, baseline_check, baseline_refs = audited_report(baseline_report, baseline_audit,
        "independent-strict-recall-supplemental-audit-v1")
    prior_ref = baseline_check["priorSelectionAudit"]
    require(identity(prior_ref["path"]) == prior_ref and read(prior_ref["path"])["passed"] is True,
            "Baseline selection audit changed")
    require(baseline["recallFloor"] == .99 and baseline["targetPaddingSeconds"] == 2
            and baseline["joinGapSeconds"] == 3, "Baseline strict99 contract differs")
    require(student["protectedTestOpened"] is False and baseline["protectedTestOpened"] is False,
            "Protected test must stay unopened")
    baseline_cells = [c for c in baseline["results"] if c["model"] == "mobile_tcn"]
    require(len(baseline_cells) == 3, "Three frozen Mobile baseline seeds required")
    complete_baseline = [c for c in baseline_cells if c["modes"]["joint"]["completeEvaluationScope"]]
    require(complete_baseline, "A complete audited baseline seed is needed to bind the full gold scope")
    gold = signature(complete_baseline[0]["modes"]["joint"]["predictions"])
    universe = universe_seconds(gold)
    normalized_baseline = [{"seed": c["seed"], "selections": [{**f["decisions"]["joint"],
        "heldSourceGroup": f["heldSourceGroup"]} for f in c["folds"]],
        "complete": c["modes"]["joint"]["completeEvaluationScope"],
        "evaluation": c["modes"]["joint"]["evaluation"], "predictions": c["modes"]["joint"]["predictions"]} for c in baseline_cells]
    normalized_student = [{**c, "complete": c["completeEvaluation"]} for c in student["results"]]
    models = {"Distilled MobileNet + TCN": describe_model(normalized_student, gold, universe),
              "Frozen MobileNet + TCN": describe_model(normalized_baseline, gold, universe)}
    return {"kind": "audited-mobile-distillation-descriptive-summary-v1", "source": identity(Path(__file__)),
        "inputs": {"student": {**student_refs, "registration": registration_ref}, "baseline": baseline_refs},
        "targetPaddingSeconds": 2, "paddingCases": list(PADDINGS), "joinGapSeconds": 3,
        "strictPositiveJoinGap": True, "recallEligibilityFloor": .99,
        "evaluableVideoSeconds": universe, "goldScopeSha256": hashlib.sha256(json.dumps(gold, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest(), "recordingCount": len(gold), "sourceGroupCount": 4,
        "aggregation": "Duration-pooled metrics within each complete seed. Means require all three full-scope seeds; never mean an eligible subset.",
        "trainingStudy": {"registeredMaximumStudentFits": 30, "maximumIsUpperBound": True,
            "infeasibleOuterFitsSkipped": True, "auditedFitCounts": audit["counts"]},
        "models": models, "pairedComparison": paired_differences(*models.values()),
        "protectedTestOpened": False, "trainingPerformed": False, "gpuUsed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("student-report", "student-audit", "registration", "baseline-report", "baseline-audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    require(str(args.output.resolve()).startswith(private_value('private-reference-0060')), "Summary output must use direct NAS")
    require(not args.output.exists() and not args.output.with_suffix(".md").exists(), "Summary output already exists")
    result = summarize(args.student_report, args.student_audit, args.registration, args.baseline_report, args.baseline_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with args.output.with_suffix(".md").open("x", encoding="utf-8") as stream:
        stream.write(markdown(result))
    print(json.dumps({"output": str(args.output), "allThreeStudentSeedsComplete": result["models"]["Distilled MobileNet + TCN"]["allThreeSeedsComplete"]}))


if __name__ == "__main__":
    main()
