"""Strict-recall operating-point selection from immutable saved predictions.

This module never trains a model or accesses a GPU. The original registrations,
weights, scores and metrics remain read-only. Infeasible recall constraints and
missing outer checkpoints are represented explicitly, not relaxed or inferred.
"""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

from . import neural_development as base
from . import neural_expanded_development as expanded
from .crop_evaluation import RecordingIntervals, evaluate_f1_pad_p_core_r
from .neural_evaluation import evaluate_predictions

SEEDS = (3407, 1729, 20260918)
EPOCHS = (5, 15, 30, 60)
RECALL_FLOOR = .99
PROTOCOL_KIND = "strict-recall-operating-point-v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return base.file_sha256(Path(path))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def identity(path):
    return {"path": str(path), "sha256": digest(path)}


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def strict_selection(candidates, recall_floor=RECALL_FLOOR):
    """Maximize registered F1 among feasible candidates, preserving tie order."""
    require(0 < recall_floor <= 1, "Recall floor must lie in (0,1]")
    require(bool(candidates), "Empty candidate grid")
    for row in candidates:
        require(all(np.isfinite(row[key]) and 0 <= row[key] <= 1
                    for key in ("innerR_core", "innerF1_padP_coreR")), "Invalid candidate metric")
    eligible = [row for row in candidates if row["innerR_core"] >= recall_floor]
    return {"recallEligibilityFloor": recall_floor, "feasible": bool(eligible),
            "candidateCount": len(candidates), "eligibleCandidateCount": len(eligible),
            "maximumInnerRecall": max(row["innerR_core"] for row in candidates),
            "selected": max(eligible, key=lambda row: row["innerF1_padP_coreR"]) if eligible else None}


def evaluate_candidates(examples, predictions):
    candidates = []
    require(bool(examples), "Empty selection population")
    expected = {e.id for e in examples}
    for epoch, scores in sorted(predictions.items()):
        require(set(scores) == expected, "Selection score population differs")
        for decoder in expanded.decoder_candidates():
            rows = [RecordingIntervals(e.id, "development", e.duration, e.truth,
                    tuple(base.decode(e, scores[e.id], decoder)), e.ignored) for e in examples]
            metric = evaluate_f1_pad_p_core_r(rows, [2.], 3.)[0]
            candidates.append({"epoch": epoch, "decoder": decoder,
                               "innerF1_padP_coreR": metric["F1_padP_coreR"],
                               "innerR_core": metric["R_core"]})
    return candidates


def outer_status(selection, available_epochs):
    if not selection["feasible"]:
        return "infeasible-inner-recall"
    if selection["selected"]["epoch"] not in available_epochs:
        return "missing-selected-outer-checkpoint"
    return "available"


def serial_rows(rows):
    values = [{**row, **{key: [interval.to_dict() if hasattr(interval, "start") else interval
                           for interval in row[key]] for key in ("rallies", "ignoredIntervals", "predictions")}}
            for row in rows]
    return json.loads(json.dumps(values, allow_nan=False))


def gold_signature(rows):
    return {row["id"]: {key: row[key] for key in ("sourceGroup", "durationSeconds", "rallies", "ignoredIntervals")}
            for row in serial_rows(rows)}


def metric_summary(evaluation):
    p, g = evaluation["primary"], evaluation["guardrails"]
    coverage = g["primaryExportCoverage"]
    short = [r for r in coverage["rallies"] if r["end"]-r["start"] <= 3]
    long = [r for r in coverage["rallies"] if r["end"]-r["start"] > 3]
    return {**{key: p[key] for key in ("P_pad", "R_core", "F1_padP_coreR", "paddedModelExportSeconds",
                                      "paddedHumanExportSeconds", "exportDurationDifferenceSeconds")},
            "eventPrecision": g["eventPrecision"], "eventRecall": g["eventRecall"], "eventF1": g["eventF1"],
            "completeLosses": coverage["completeRallyLosses"],
            "incompleteLosses": coverage["completeRallyLosses"]+coverage["partialRallyLosses"],
            "shortCompleteLosses": sum(row["completelyLost"] for row in short),
            "longR_core": sum(row["retainedCoreSeconds"] for row in long)/sum(row["evaluableCoreSeconds"] for row in long),
            "incorrectExportSeconds": p["paddedModelExportSeconds"]-p["paddedPrecisionIntersectionSeconds"],
            "wantedExportOmittedSeconds": p["paddedHumanExportSeconds"]-p["paddedPrecisionIntersectionSeconds"],
            "missedCoreSeconds": p["coreHumanSeconds"]-p["coreRecallIntersectionSeconds"]}


def mean_metrics(results):
    rows = [metric_summary(row["evaluation"]) for row in results]
    return {key: statistics.mean(row[key] for row in rows) for key in rows[0]}


class Evidence:
    def __init__(self):
        self.files = {}

    def bind(self, path, expected=None):
        path = Path(path)
        value = identity(path)
        if expected is not None:
            require(value["sha256"] == expected, "Changed artifact: "+str(path))
        self.files[str(path)] = value
        return path

    def verify_final(self):
        for value in self.files.values():
            require(digest(value["path"]) == value["sha256"], "Input changed during run: "+value["path"])


def load_scores(folder, epochs, examples, expected_train_ids, contract_sha, evidence):
    """Check saved fit exclusion and score bytes; no checkpoint execution."""
    metadata = read(evidence.bind(folder/"completed.json"))
    require(metadata["contractSha256"] == contract_sha, "Saved fit contract differs")
    require(set(metadata["trainIds"]) == set(expected_train_ids), "Training source exclusion differs")
    require(set(metadata["validationIds"]) == {e.id for e in examples}, "Saved validation population differs")
    require(set(epochs) <= set(metadata["epochs"]), "Requested checkpoint is unavailable")
    output = {}
    for epoch in epochs:
        name = f"predictions-{epoch}.npz"
        path = evidence.bind(folder/name, metadata["artifacts"][name])
        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == {e.id for e in examples}, "Score archive population differs")
            scores = {e.id: archive[e.id].copy() for e in examples}
        for e in examples:
            values = scores[e.id]
            require(values.shape == (len(e.times), 4) and np.isfinite(values).all()
                    and np.all((values >= 0) & (values <= 1)), "Invalid scores: "+e.id)
        output[epoch] = scores
    return output, metadata


def model_specs(root):
    return [
        {"name": "av_tcn_short_boost", "layout": "legacy", "kind": "tcn",
         "study": root/"2026-09-19-short-boost-transfer/study"},
        {"name": "dino_tcn_short_boost", "layout": "legacy", "kind": "dino_tcn",
         "study": root/"2026-09-19-short-boost-transfer/study"},
        *[{"name": name, "layout": "recognition", "study": root/"2026-09-22-recognition"/folder}
          for name, folder in (("av_transformer", "av-transformer-v1"),
                               ("dino_transformer", "dino-transformer-v1"),
                               ("mobile_tcn", "mobile-tcn-v1"))]]


def load_study(spec, evidence):
    folder = spec["study"]
    registration = read(evidence.bind(folder/"preregistration.json"))
    require(canonical(registration["contract"]) == registration["sha256"], "Registration hash differs")
    report = read(evidence.bind(folder/"report.json"))
    require(report["contractSha256"] == registration["sha256"], "Report registration differs")
    contract = registration["contract"]
    for name, sha in contract["code"].items():
        evidence.bind(Path(__file__).parent/name, sha)
    require(tuple(contract["seeds"]) == SEEDS and tuple(contract["checkpointEpochs"]) == EPOCHS,
            "Seed/epoch grid differs")
    require(contract["decoderCandidates"] == expanded.decoder_candidates(), "Decoder grid differs")
    if spec["layout"] == "recognition":
        audit = read(evidence.bind(folder/"audit.json"))
        require(audit.get("passed") is True and audit["report"] == identity(folder/"report.json")
                and audit["registration"] == identity(folder/"preregistration.json"), "Recognition audit differs")
        results = report["results"]
    else:
        results = [row for row in report["results"] if row["cohort"] == "reviewed_export"
                   and row["kind"] == spec["kind"] and row["lossArm"] == "short_boost"]
    require(sorted(row["seed"] for row in results) == sorted(SEEDS), "Study result cells differ")
    return registration, results


def collect_fold_scores(spec, seed, outer_index, examples, groups, contract_sha, evidence):
    outer = groups[outer_index]
    fit_root = spec["study"]/"fits"
    if spec["layout"] == "legacy":
        fit_root = fit_root/"reviewed_export"/spec["kind"]/"short_boost"/str(seed)
        outer_folder = fit_root/f"outer-{outer_index}"/"refit"
    else:
        fit_root = fit_root/str(seed)
        outer_folder = fit_root/f"outer-{outer_index}"
    inner_scores = {epoch: {} for epoch in EPOCHS}
    for inner_index, inner in enumerate(groups):
        if inner == outer:
            continue
        excluded = {outer, inner}
        train_ids = [e.id for e in examples if e.group not in excluded]
        if spec["layout"] == "legacy":
            legacy_index = [group for group in groups if group != outer].index(inner)
            inner_folder = fit_root/f"outer-{outer_index}"/f"inner-{legacy_index}"
            validation = [e for e in examples if e.group == inner]
        else:
            left, right = sorted((outer_index, inner_index))
            inner_folder = fit_root/f"inner-{left}-{right}"
            validation = [e for e in examples if e.group in excluded]
        scores, _ = load_scores(inner_folder, EPOCHS, validation, train_ids, contract_sha, evidence)
        for epoch in EPOCHS:
            inner_scores[epoch].update({e.id: scores[epoch][e.id] for e in examples if e.group == inner})
    metadata = read(evidence.bind(outer_folder/"completed.json"))
    held = [e for e in examples if e.group == outer]
    outer_scores, _ = load_scores(outer_folder, tuple(metadata["epochs"]), held,
                                 [e.id for e in examples if e.group != outer], contract_sha, evidence)
    return inner_scores, outer_scores


def run(root, output, protocol_path):
    evidence = Evidence()
    protocol = read(evidence.bind(protocol_path))
    require(protocol.get("kind") == PROTOCOL_KIND and protocol.get("recallFloor") == RECALL_FLOOR,
            "Expected frozen strict99% protocol")
    require(protocol.get("jointSelection") == "192-candidates-no-infeasible-fallback"
            and protocol.get("fixedEpochSensitivity") == "48-decoders-old-epoch-no-infeasible-fallback",
            "Protocol selection contract differs")
    require(protocol.get("infeasibleDeploymentPolicy") == "not-evaluated",
            "Protocol must explicitly declare infeasible deployment policy")
    combined = protocol["combinedProtocol"]
    evidence.bind(combined["path"], combined["sha256"])
    for path, expected in protocol.get("code", {}).items():
        evidence.bind(path, expected)
    require(protocol.get("code"), "Protocol needs frozen code identities")
    manifest_path = root/"2026-09-19-short-boost-transfer/manifest-pts-v1.json"
    manifest = read(evidence.bind(manifest_path))
    exact = manifest["exactManifest"]
    examples = base.load_examples(evidence.bind(exact["path"], exact["sha256"]), False)
    for row in manifest["exactRows"]:
        cache = row["featureCaches"]["audiovisual"]
        evidence.bind(cache["path"], cache["sha256"])
    groups = sorted({e.group for e in examples})
    require(len(groups) == 4 and len(examples) == 8, "Frozen exact evaluation population differs")
    gold = serial_rows([e.row([]) for e in examples])
    gold_id = gold_signature(gold)
    results = []
    refits = []
    for spec in model_specs(root):
        registration, original_results = load_study(spec, evidence)
        require(registration["contract"]["groups"] == groups, "Group order differs")
        for original in original_results:
            seed = original["seed"]
            require(gold_signature(original["predictions"]) == gold_id, "Original gold/source/ignored revision differs")
            require(evaluate_predictions(original["predictions"]) == original["evaluation"], "Original metric replay differs")
            folds, all_predictions = [], {"joint": [], "fixed_epoch": []}
            for outer_index, outer in enumerate(groups):
                print(f"SELECT {spec['name']} seed={seed} outer={outer}", flush=True)
                inner, outer_scores = collect_fold_scores(spec, seed, outer_index, examples, groups,
                                                          registration["sha256"], evidence)
                fitting = [e for e in examples if e.group != outer]
                held = [e for e in examples if e.group == outer]
                candidates = evaluate_candidates(fitting, inner)
                old = next(row for row in original["selections"] if row["heldSourceGroup"] == outer)
                old_replay = expanded.select_candidate(candidates)
                require(all(old[key] == value for key, value in old_replay.items()), "Original95% selection replay differs")
                baseline_rows = serial_rows([e.row(base.decode(e, outer_scores[old["epoch"]][e.id], old["decoder"])) for e in held])
                expected_rows = [row for row in original["predictions"] if row["sourceGroup"] == outer]
                require(baseline_rows == expected_rows, "Original held predictions replay differs")
                decisions = {"joint": strict_selection(candidates),
                             "fixed_epoch": strict_selection([row for row in candidates if row["epoch"] == old["epoch"]])}
                fold = {"heldSourceGroup": outer, "oldSelection": old, "outerCheckpointEpochs": sorted(outer_scores),
                        "candidates": candidates, "decisions": decisions}
                for mode, selection in decisions.items():
                    status = outer_status(selection, outer_scores)
                    selection["outerStatus"] = status
                    if status == "available":
                        chosen = selection["selected"]
                        rows = serial_rows([e.row(base.decode(e, outer_scores[chosen["epoch"]][e.id], chosen["decoder"])) for e in held])
                        selection["heldEvaluation"] = evaluate_predictions(rows)
                        all_predictions[mode].extend(rows)
                    elif status == "missing-selected-outer-checkpoint":
                        refits.append({"model": spec["name"], "layout": spec["layout"], "kind": spec.get("kind"),
                                       "study": str(spec["study"]), "seed": seed, "outerIndex": outer_index,
                                       "heldSourceGroup": outer, "selected": selection["selected"],
                                       "originalEpoch": old["epoch"],
                                       "checkpointEpochs": sorted({old["epoch"], selection["selected"]["epoch"]}),
                                       "originalContractSha256": registration["sha256"]})
                folds.append(fold)
            modes = {}
            for mode, rows in all_predictions.items():
                complete = {row["id"] for row in rows} == {e.id for e in examples}
                modes[mode] = {"completeEvaluationScope": complete,
                               "scopeRecordingIds": [row["id"] for row in rows],
                               "evaluation": evaluate_predictions(rows) if complete else None,
                               "predictions": rows,
                               "interpretation": "Strict inner99% eligibility; held recall is measured, never guaranteed."}
            results.append({"model": spec["name"], "seed": seed, "folds": folds, "modes": modes,
                            "original95": {"evaluation": original["evaluation"]}})
    production_path = root/"2026-09-19-production-combinations/results/productionDefault--fixed.json"
    production = read(evidence.bind(production_path))
    production_rows = [{**row, "predictions": production["predictions"][row["id"]]} for row in gold]
    production_eval = evaluate_predictions(production_rows)
    require(production_eval == production["evaluation"], "Production scope/evaluation differs")
    evidence.verify_final()
    report = {"kind": PROTOCOL_KIND, "status": "completed-cpu-reselection", "protocol": identity(protocol_path),
              "recallFloor": RECALL_FLOOR, "targetPaddingSeconds": 2, "paddingCases": [0, 1, 2, 3], "joinGapSeconds": 3,
              "aggregation": "Pool recordings per seed; report seed means only on complete identical evaluation scopes.",
              "newTrainingPerformed": False, "protectedTestOpened": False, "productionChanged": False,
              "production": {"evaluation": production_eval, "historicalLabelExposure": True},
              "results": results, "references": list(evidence.files.values())}
    write_new(output/"refit-plan.json", {"kind": "strict-recall-missing-outer-refits-v1",
              "protocol": identity(protocol_path), "newSettingsSelectedFromOuter": False,
              "trainingRecipe": "Identical original registered fit routine, seed, source exclusions and sample exposure; save selected and old epochs to verify original checkpoint parity.",
              "tasks": refits})
    write_new(output/"report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(private_value('private-reference-0057')))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output, args.protocol)


if __name__ == "__main__":
    main()
