#!/usr/bin/env python3
"""Summarize strict99% feasibility without ranking incomplete seed populations."""
from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis import neural_development as base
from analysis.crop_evaluation import subtract_intervals
from analysis.schema import Interval
from analysis.neural_recall_operating_point import identity, metric_summary, read, require, write_new


def mean(rows):
    return {key: statistics.mean(row[key] for row in rows) for key in rows[0]}


def summarize(report_path, output):
    report = read(report_path)
    # Bind the same exact label universe, not an independently estimated duration.
    manifest_binding = next(row for row in report["references"] if row["path"].endswith("/manifest-pts-v1.json"))
    require(identity(manifest_binding["path"]) == manifest_binding, "Manifest revision changed")
    manifest = read(manifest_binding["path"])
    exact = manifest["exactManifest"]
    require(identity(exact["path"]) == exact, "Exact label universe changed")
    examples = base.load_examples(Path(exact["path"]), False)
    universe = sum(sum(row.end-row.start for row in subtract_intervals((Interval(0, e.duration),), e.ignored)) for e in examples)

    def metrics(evaluation):
        row = metric_summary(evaluation)
        row["correctlyRemovedSeconds"] = universe-row["paddedHumanExportSeconds"]-row["incorrectExportSeconds"]
        return row

    models = {}
    detail = []
    for name in dict.fromkeys(row["model"] for row in report["results"]):
        cells = [row for row in report["results"] if row["model"] == name]
        all_folds = [fold for cell in cells for fold in cell["folds"]]
        modes = {}
        for mode in ("joint", "fixed_epoch"):
            full = [cell for cell in cells if cell["modes"][mode]["completeEvaluationScope"]]
            complete = len(full) == 3
            modes[mode] = {
                "feasibleFolds": sum(fold["decisions"][mode]["feasible"] for fold in all_folds),
                "totalFolds": len(all_folds), "completeSeeds": [cell["seed"] for cell in full],
                "allThreeSeedsComplete": complete,
                "mean": mean([metrics(cell["modes"][mode]["evaluation"]) for cell in full]) if complete else None,
                "seedMetrics": [{"seed": cell["seed"], **metrics(cell["modes"][mode]["evaluation"])} for cell in full],
                "padding": [{key: statistics.mean(cell["modes"][mode]["evaluation"]["padding"][i][key] for cell in full)
                             for key in ("paddingSecondsBeforeAndAfter", "joinGapSeconds", "P_pad", "R_core", "F1_padP_coreR",
                                         "paddedModelExportSeconds", "paddedHumanExportSeconds", "exportDurationDifferenceSeconds")}
                            for i in range(4)] if complete else None,
            }
        models[name] = {"original95": mean([metrics(cell["original95"]["evaluation"]) for cell in cells]), "modes": modes}
        for cell in cells:
            for fold in cell["folds"]:
                for mode in ("joint", "fixed_epoch"):
                    decision = fold["decisions"][mode]
                    chosen = decision["selected"]
                    detail.append({"model": name, "seed": cell["seed"], "sourceGroup": fold["heldSourceGroup"],
                                   "mode": mode, "status": decision["outerStatus"], "selected": chosen,
                                   "maximumInnerRecall": decision["maximumInnerRecall"],
                                   "heldMetrics": metric_summary(decision["heldEvaluation"]) if "heldEvaluation" in decision else None})
    summary = {"kind": "strict-recall-operating-point-summary-v1", "report": identity(report_path),
               "evaluableVideoSeconds": universe, "targetPaddingSeconds": 2, "joinGapSeconds": 3,
               "interpretation": "Means require all three complete source-held seeds. Individual complete seeds are descriptive, not a substitute ranking population.",
               "production": metrics(report["production"]["evaluation"]), "models": models, "folds": detail}
    write_new(output, summary)
    lines = ["# Strict 99% recall operating-point comparison", "",
             "Inner eligibility uses retained human play with 2 seconds of padding per side and joins strictly below 3 seconds. It is not an unseen-source guarantee. No infeasible fold is given a relaxed fallback. Production has historical label exposure.", "",
             "| Model | Joint feasible folds /12 | Decoder-only feasible /12 | Joint complete seeds | Joint mean P / R / F1 |", "|---|---:|---:|---|---|"]
    for name, row in models.items():
        joint, fixed = row["modes"]["joint"], row["modes"]["fixed_epoch"]
        measured = joint["mean"]
        value = " / ".join(f"{100*measured[k]:.2f}%" for k in ("P_pad", "R_core", "F1_padP_coreR")) if measured else "Not ranked: incomplete three-seed scope"
        lines.append(f"| {name} | {joint['feasibleFolds']} | {fixed['feasibleFolds']} | {', '.join(map(str,joint['completeSeeds'])) or 'None'} | {value} |")
    lines += ["", "No averaging only the feasible/favorable subset is used for a model ranking. Full fold decisions, seed metrics, padding sensitivity, export durations and omission/error time are in the adjacent JSON.", "",
              "| Model | Seed | Held source | Joint status | Max inner R | Selected epoch | Held P / R / F1 |", "|---|---:|---|---|---:|---:|---|"]
    for row in detail:
        if row["mode"] != "joint":
            continue
        m = row["heldMetrics"]
        value = " / ".join(f"{100*m[k]:.2f}%" for k in ("P_pad", "R_core", "F1_padP_coreR")) if m else "Unavailable"
        lines.append(f"| {row['model']} | {row['seed']} | {row['sourceGroup']} | {row['status']} | {100*row['maximumInnerRecall']:.3f}% | {row['selected']['epoch'] if row['selected'] else '—'} | {value} |")
    with output.with_suffix(".md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.report, args.output)
