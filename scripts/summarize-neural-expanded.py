#!/usr/bin/env python3
"""Audit and summarize the frozen expanded neural study, without selection.

Progress mode reads completed result files and writes nothing. Final mode
requires the full report, verifies exact target revisions and replays canonical
metrics, audits fit membership/artifacts and paired exposure, then exclusively
creates summary.json, summary.md and loss-identities.json.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
ROOT = Path(private_value('private-reference-0070'))
COHORTS = ("exact", "draft", "reviewed_export")
KINDS = ("linear", "tcn")
SCOPES = ("coreCoverage", "primaryExportCoverage")
HELPER_PATH = REPO / "scripts/summarize-neural-development.py"
spec = importlib.util.spec_from_file_location("expanded_summary_helpers", HELPER_PATH)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
from analysis.neural_evaluation import evaluate_predictions


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def identity(path: Path) -> dict:
    return {"path": str(path), "sha256": helpers.digest(path)}


def mean(values) -> float:
    return float(statistics.fmean(values))


def mean_available(values) -> float | None:
    available = [value for value in values if value is not None]
    return mean(available) if available else None


def metric_means(rows: list[dict]) -> dict:
    return {key: mean(row[key] for row in rows) for key in helpers.METRICS}


def compact_evaluation(evaluation: dict) -> dict:
    """Empty outcome slices are unavailable, never zero-recall failures."""
    require([row["paddingSecondsBeforeAndAfter"] for row in evaluation["padding"]] == [0, 1, 2, 3],
            "Missing ordered padding sensitivity cases")
    guard = evaluation["guardrails"]
    return {"primary": {key: evaluation["primary"][key] for key in helpers.METRICS},
            "padding": [{"paddingSecondsBeforeAndAfter": row["paddingSecondsBeforeAndAfter"],
                         **{key: row[key] for key in helpers.METRICS}} for row in evaluation["padding"]],
            "eventF1": guard.get("eventF1"),
            "boundaryErrors": {key: guard.get(key) for key in ("startBoundaryMaeSeconds", "endBoundaryMaeSeconds",
                                                               "startBoundaryP90Seconds", "endBoundaryP90Seconds")},
            "outcomeRecall": {name: {key: guard["outcomeSlices"][name].get(key)
                                     for key in ("rallies", "strictMatchRecall", "anyOverlapRecall", "coverageAtLeast95Rate")}
                              for name in helpers.SLICES},
            "coverage": {scope: {key: guard[scope][key] for key in helpers.COVERAGE_KEYS} for scope in SCOPES}}


def result_key(row: dict) -> tuple:
    return row["cohort"], row["kind"], row["seed"]


def selection_feasible(selection: dict, floor: float) -> bool:
    """Require both declared feasibility and the actual selected inner recall."""
    feasible = selection.get("recallEligibilityPassed")
    require(isinstance(feasible, bool), "Selection needs an explicit recallEligibilityPassed boolean")
    recall = selection.get("innerR_core")
    require(isinstance(recall, (int, float)) and math.isfinite(recall), "Selection needs finite innerR_core")
    require(selection["recallEligibilityFloor"] == floor, "Selection recall floor differs from contract")
    require(feasible == (recall >= floor), "Inner feasibility flag contradicts selected recall")
    return feasible


def feasibility(candidate_rows: list[dict], paired_rows: list[dict], floor: float) -> dict:
    """Descriptive development screen; never a deployment authorization."""
    seed_deltas = [row["primaryDelta"]["F1_padP_coreR"] for row in paired_rows]
    groups = sorted(paired_rows[0]["sourceGroups"])
    group_deltas = {group: mean(row["sourceGroups"][group]["F1_padP_coreRDelta"] for row in paired_rows)
                    for group in groups}
    checks = {
        "meanF1GainAtLeast002": mean(seed_deltas) >= .02 - 1e-12,
        "atLeastTwoThirdsSeedsPositive": sum(value > 1e-12 for value in seed_deltas) >= math.ceil(2 * len(seed_deltas) / 3),
        "majoritySourceGroupsPositive": sum(value > 1e-12 for value in group_deltas.values()) > len(groups) / 2,
        "meanCoreRecallRegressionAtMost0005": mean(row["primaryDelta"]["R_core"] for row in paired_rows) >= -.005 - 1e-12,
    }
    inner_feasible = all(selection_feasible(selection, floor)
                         for row in candidate_rows for selection in row["selections"])
    outer_floor = all(row["evaluation"]["primary"]["R_core"] >= floor - 1e-12 for row in candidate_rows)
    return {"passed": all(checks.values()), "checks": checks, "innerRecallEligibilityFloor": floor,
            "allCandidateInnerSelectionsFeasible": inner_feasible,
            "screenPassedAndInnerFeasible": all(checks.values()) and inner_feasible,
            "outerRecallFloorDiagnostic": {"allCandidateSeedsAtLeastInnerFloor": outer_floor,
                                           "role": "descriptive only; not an additional frozen comparison gate"},
            "role": "four frozen development screen criteria; inner eligibility reported separately; no production promotion",
            "productionPromotionAllowed": False,
            "positiveSeeds": sum(value > 1e-12 for value in seed_deltas), "seedCount": len(seed_deltas),
            "positiveSourceGroups": sum(value > 1e-12 for value in group_deltas.values()), "sourceGroupCount": len(groups)}


def paired_evaluation(candidate: dict, baseline: dict) -> dict:
    helpers.assert_comparable(candidate, baseline)
    return {"primaryDelta": {key: candidate["primary"][key] - baseline["primary"][key] for key in helpers.METRICS},
            "sourceGroups": helpers.paired_groups(candidate, baseline),
            "eventF1Delta": candidate["guardrails"]["eventF1"] - baseline["guardrails"]["eventF1"],
            "lossCounts": {scope: helpers.coverage_comparison(candidate, baseline, scope)["counts"] for scope in SCOPES}}


def comparison(name: str, candidate: tuple[str, str], baseline: tuple[str, str],
               by_key: dict, seeds: list[int], floor: float) -> dict:
    candidate_rows = [by_key[(*candidate, seed)] for seed in seeds]
    pairs = [{"seed": seed, **paired_evaluation(by_key[(*candidate, seed)]["evaluation"],
                                                by_key[(*baseline, seed)]["evaluation"])} for seed in seeds]
    groups = sorted(pairs[0]["sourceGroups"])
    return {"name": name, "candidate": {"cohort": candidate[0], "kind": candidate[1]},
            "baseline": {"cohort": baseline[0], "kind": baseline[1]},
            "meanSeedPrimaryDelta": metric_means([row["primaryDelta"] for row in pairs]),
            "meanSeedEventF1Delta": mean(row["eventF1Delta"] for row in pairs),
            "sourceGroupMeanPairedDelta": {group: {
                key: mean(row["sourceGroups"][group][key] for row in pairs)
                for key in ("F1_padP_coreRDelta", "R_coreDelta", "paddedModelExportSecondsDelta")}
                for group in groups},
            "perSeed": pairs, "developmentScreen": feasibility(candidate_rows, pairs, floor)}


def summarize_results(results: list[dict], contract: dict, floor: float) -> tuple[list, list, list, list]:
    by_key = {result_key(row): row for row in results}
    seeds = contract["seeds"]
    per_seed, aggregates, losses = [], [], []
    for cohort in COHORTS:
        for kind in KINDS:
            these = []
            for seed in seeds:
                raw = by_key[(cohort, kind, seed)]
                evaluation = raw["evaluation"]
                refs = {"sameCohortLinear": by_key[(cohort, "linear", seed)]["evaluation"],
                        "exactTCN": by_key[("exact", "tcn", seed)]["evaluation"]}
                if cohort == "reviewed_export" and kind == "tcn":
                    refs["draftTCN"] = by_key[("draft", "tcn", seed)]["evaluation"]
                details = {name: {scope: helpers.coverage_comparison(evaluation, reference, scope) for scope in SCOPES}
                           for name, reference in refs.items()}
                row = {"cohort": cohort, "kind": kind, "seed": seed,
                       **compact_evaluation(evaluation),
                       "sourceGroups": {group: compact_evaluation(value) for group, value in evaluation["sourceGroups"].items()},
                       "selections": raw["selections"],
                       "infeasibleInnerSelections": sum(not selection_feasible(choice, floor) for choice in raw["selections"]),
                       "outerCoreRecallFloorSatisfied": evaluation["primary"]["R_core"] >= floor - 1e-12,
                       "pairedReferences": {name: paired_evaluation(evaluation, reference) for name, reference in refs.items()}}
                per_seed.append(row)
                these.append(row)
                losses.append({"cohort": cohort, "kind": kind, "seed": seed,
                               "ownLosses": {scope: [rally for rally in evaluation["guardrails"][scope]["rallies"]
                                                     if not rally["fullyCovered"]] for scope in SCOPES},
                               "pairedReferences": details})
            aggregates.append({"cohort": cohort, "kind": kind, "seedCount": len(these),
                               "meanSeedPrimary": metric_means([row["primary"] for row in these]),
                               "minSeedF1_padP_coreR": min(row["primary"]["F1_padP_coreR"] for row in these),
                               "maxSeedF1_padP_coreR": max(row["primary"]["F1_padP_coreR"] for row in these),
                               "meanSeedPadding": [{"paddingSecondsBeforeAndAfter": pad,
                                                    **metric_means([row["padding"][pad] for row in these])} for pad in range(4)],
                               "meanSeedEventF1": mean(row["eventF1"] for row in these),
                               "meanSeedOutcomeStrictRecall": {name: mean_available(row["outcomeRecall"][name]["strictMatchRecall"] for row in these)
                                                               for name in helpers.SLICES},
                               "meanSeedCoverage": {scope: {key: mean(row["coverage"][scope][key] for row in these)
                                                            for key in helpers.COVERAGE_KEYS} for scope in SCOPES},
                               "infeasibleInnerSelections": sum(row["infeasibleInnerSelections"] for row in these),
                               "allSeedOuterCoreRecallFloorSatisfied": all(row["outerCoreRecallFloorSatisfied"] for row in these)})
    comparisons = []
    for kind in KINDS:
        for label, candidate, baseline in (("B-minus-A", "draft", "exact"), ("C-minus-B", "reviewed_export", "draft"),
                                            ("C-minus-A", "reviewed_export", "exact")):
            comparisons.append(comparison(f"{kind}/{label}", (candidate, kind), (baseline, kind), by_key, seeds, floor))
    for cohort in COHORTS:
        comparisons.append(comparison(f"{cohort}/TCN-minus-linear", (cohort, "tcn"), (cohort, "linear"), by_key, seeds, floor))
    return per_seed, aggregates, comparisons, losses


def read_inputs(study: Path, manifest_path: Path, *, progress: bool) -> tuple[dict, dict, dict, dict, dict]:
    registration_path = study / "preregistration.json"
    registration = helpers.load(registration_path)
    contract = registration["contract"]
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    require(hashlib.sha256(canonical.encode()).hexdigest() == registration["sha256"], "Invalid preregistration hash")
    require(set(contract["cohorts"]) == set(COHORTS) and set(contract["kinds"]) == set(KINDS), "Unexpected cohort/model contract")
    require(contract["primaryMetric"] == "F1_padP_coreR" and contract["targetPaddingSeconds"] == 2
            and contract["joinGapSeconds"] == 3 and contract["paddingSweep"] == [0, 1, 2, 3], "Unexpected ranking contract")
    require(contract["screen"] == {"meanF1Gain": .02, "meanRCoreMinimumDelta": -.005,
                                    "positiveSeedCount": 2, "positiveGroupMajority": True}
            and len(contract["seeds"]) == 3, "Summary screen differs from frozen study contract")
    require(helpers.digest(manifest_path) == contract["manifestSha256"], "Expanded manifest hash mismatch")
    manifest = helpers.load(manifest_path)
    exact_path = Path(manifest["exactManifest"]["path"])
    require(helpers.digest(exact_path) == manifest["exactManifest"]["sha256"], "Exact manifest hash mismatch")
    exact = helpers.load(exact_path)
    require(exact["recordings"] == manifest["exactRows"], "Exact manifest rows differ from expanded exact tier")
    groups = sorted({row["sourceGroup"] for row in exact["recordings"]})
    require(groups == sorted(contract["groups"]), "Exact groups differ from preregistration")
    all_rows = manifest["exactRows"] + manifest["draftRows"] + manifest["coverageRows"]
    require(len({row["id"] for row in all_rows}) == len(all_rows), "Duplicate cross-tier recording IDs")
    require(not {row["sourceGroup"] for row in all_rows}.intersection(manifest["protectedSourceGroups"]), "Protected group in manifest")
    require(all(row["environment"] in ("grass", "indoor") and row["consent"]["train"] for row in all_rows), "Invalid environment/consent")
    # Re-encoded proxies can have distinct content hashes but the same raw master.
    source_groups: dict[str, set] = {}
    for row in all_rows:
        for key in ("sourceContentSha256", "contentSha256"):
            if row.get(key):
                source_groups.setdefault(row[key], set()).add(row["sourceGroup"])
    require(all(len(value) == 1 for value in source_groups.values()), "A shared raw/proxy identity crosses source groups")
    report_path = study / "report.json"
    if report_path.exists():
        report = helpers.load(report_path)
        require(report["contractSha256"] == registration["sha256"], "Report contract hash mismatch")
        require(not report["protectedTestOpened"] and not report["productionPromotionAllowed"], "Unexpected test/promotion status")
        require(report["records"] == len(exact["recordings"]) and sorted(report["sourceGroups"]) == groups, "Report evaluation scope mismatch")
    elif progress:
        report = {"results": [helpers.load(path) for path in sorted(study.glob("result-*.json"))]}
    else:
        raise FileNotFoundError("Final mode requires completed report.json; use --progress while training runs")
    expected = {(cohort, kind, seed) for cohort in COHORTS for kind in KINDS for seed in contract["seeds"]}
    observed = {result_key(row) for row in report["results"]}
    require(len(observed) == len(report["results"]) and observed <= expected, "Duplicate/unexpected result identity")
    if not progress:
        require(observed == expected, "Completed report is missing results")
    inputs = {"registration": identity(registration_path), "expandedManifest": identity(manifest_path),
              "exactManifest": identity(exact_path), "summaryScript": identity(Path(__file__).resolve()),
              "summaryHelpers": identity(HELPER_PATH)}
    if report_path.exists():
        inputs["report"] = identity(report_path)
    return registration, manifest, exact, report, inputs


def expected_supervision(row: dict, tier: str) -> dict:
    """Independent reconstruction using only frozen times and annotation layers."""
    import numpy as np
    from analysis.crop_evaluation import pad_and_merge_intervals, subtract_intervals
    from analysis.schema import Interval, labels_for_times, mask_for_times
    cache = row["featureCaches"]["audiovisual"]
    with np.load(cache["path"], allow_pickle=False) as data:
        times = data["times"].astype(np.float64)
        duration = float(json.loads(str(data["metadata_json"].item()))["duration"])
    convert = lambda rows: tuple(Interval(float(r["start"]), float(r["end"])) for r in rows)
    truth, ignored = convert(row.get("rallies", [])), convert(row.get("ignoredIntervals", []))
    valid = mask_for_times(times, ignored)
    targets = np.zeros((len(times), 4), np.float32)
    masks = np.zeros_like(targets)
    if tier == "exact":
        targets[:, 0] = labels_for_times(times, truth)
        for head, attr in ((1, "start"), (2, "end")):
            for interval in truth:
                distances = np.abs(times - getattr(interval, attr))
                pulse = np.exp(-.5 * (distances / .35) ** 2)
                pulse[distances > 1.] = 0
                targets[:, head] = np.maximum(targets[:, head], pulse.astype(np.float32))
        keep = subtract_intervals(pad_and_merge_intervals(truth, duration, 2., 3.), ignored)
        targets[:, 3] = labels_for_times(times, keep)
        masks[:] = valid[:, None]
        for interval in ignored:
            masks[(times >= interval.start-1.) & (times <= interval.end+1.), 1:3] = 0
    elif tier == "draft":
        targets[:, 0] = labels_for_times(times, truth)
        masks[:, 0] = valid
        for interval in (*truth, *ignored):
            for edge in (interval.start, interval.end):
                masks[np.abs(times-edge) <= 1., 0] = 0
    else:
        require(tier == "coverage", "Unknown supervision tier")
        window = row["gameWindow"]
        valid &= (times >= window["start"]) & (times < window["end"])
        targets[:, 3] = labels_for_times(times, convert(row["keepTargets"]))
        masks[:, 3] = valid
    return {"valid": masks.sum(axis=0).astype(int).tolist(),
            "positiveMass": (targets*masks).sum(axis=0).tolist()}


def audit_exposure_pair(left: dict, right: dict, *, compare_draft: bool) -> int:
    """Compare the entire common epoch prefix, including unequal refit lengths."""
    require(left["trainIds"] == right["trainIds"] and left["scalerTrainIds"] == right["scalerTrainIds"],
            "Paired cohorts differ in exact/scaler fitting population")
    require(left["positiveWeight"] == right["positiveWeight"], "Paired cohorts differ in class weighting")
    first, second = left["history"], right["history"]
    require(first and second, "Missing paired training history")
    for a, b in zip(first, second):
        require(a["epoch"] == b["epoch"] and a["optimizerSteps"] == b["optimizerSteps"], "Paired optimizer-step exposure differs")
        streams = ("exact", "draft") if compare_draft else ("exact",)
        for stream in streams:
            require(a["exposureSha256"].get(stream) == b["exposureSha256"].get(stream)
                    and a["exposureSha256"].get(stream) is not None, f"Paired {stream} sample exposure differs")
    return min(len(first), len(second))


def audit_fits(study: Path, manifest: dict, contract: dict, contract_hash: str, results: list[dict]) -> dict:
    import numpy as np
    groups, seeds = contract["groups"], contract["seeds"]
    tiers = {"exact": manifest["exactRows"], "draft": manifest["draftRows"], "coverage": manifest["coverageRows"]}
    by_id = {row["id"]: row for rows in tiers.values() for row in rows}
    counts_by_id = {row["id"]: expected_supervision(row, tier) for tier, rows in tiers.items() for row in rows}
    selected = {(row["cohort"], row["kind"], row["seed"], choice["heldSourceGroup"]): choice["epoch"]
                for row in results for choice in row["selections"]}
    metadata, fit_paths, completed_inputs, runtime, parameter_sets = {}, {}, [], {}, {}
    artifact_count = 0
    for cohort in COHORTS:
        active_tiers = () if cohort == "exact" else ("draft",) if cohort == "draft" else ("draft", "coverage")
        for kind in KINDS:
            for seed in seeds:
                for outer_index, outer in enumerate(groups):
                    folds = [(f"inner-{index}", inner) for index, inner in enumerate(group for group in groups if group != outer)]
                    folds.append(("refit", outer))
                    for fold, validation_group in folds:
                        excluded = {outer, validation_group}
                        path = study / "fits" / cohort / kind / str(seed) / f"outer-{outer_index}" / fold / "completed.json"
                        meta = helpers.load(path)
                        completed_inputs.append(identity(path))
                        require(meta["contractSha256"] == contract_hash and meta["kind"] == kind and meta["seed"] == seed, f"Fit identity mismatch: {path}")
                        train_ids = [row["id"] for row in tiers["exact"] if row["sourceGroup"] not in excluded]
                        validation_ids = [row["id"] for row in tiers["exact"] if row["sourceGroup"] == validation_group]
                        auxiliary = {tier: [row["id"] for row in tiers[tier] if row["sourceGroup"] not in excluded] for tier in active_tiers}
                        require(meta["trainIds"] == train_ids and meta["scalerTrainIds"] == train_ids, f"Exact/scaler fold leakage: {path}")
                        require(meta["validationIds"] == validation_ids and meta["auxiliaryIds"] == auxiliary, f"Validation/auxiliary fold leakage: {path}")
                        require(meta["trainGroups"] == sorted({by_id[i]["sourceGroup"] for i in train_ids})
                                and meta["validationGroups"] == [validation_group], f"Fit group metadata mismatch: {path}")
                        require(meta["auxiliaryGroups"] == {tier: sorted({by_id[i]["sourceGroup"] for i in ids}) for tier, ids in auxiliary.items()}, f"Auxiliary group metadata mismatch: {path}")
                        expected_epochs = [selected[(cohort, kind, seed, outer)]] if fold == "refit" else contract["checkpointEpochs"]
                        require(meta["epochs"] == expected_epochs, f"Checkpoint selection/refit mismatch: {path}")
                        supervised_ids = {"exact": train_ids, **auxiliary}
                        require(set(meta["supervisedCounts"]) == set(supervised_ids), f"Unexpected supervision tier: {path}")
                        for tier, ids in supervised_ids.items():
                            expected = {name: np.sum([counts_by_id[i][name] for i in ids], axis=0) if ids else np.zeros(4)
                                        for name in ("valid", "positiveMass")}
                            actual = meta["supervisedCounts"][tier]
                            require(np.array_equal(actual["valid"], expected["valid"])
                                    and np.allclose(actual["positiveMass"], expected["positiveMass"], rtol=1e-6, atol=1e-4),
                                    f"Supervision masks/targets differ from frozen labels: {path}/{tier}")
                        history = meta["history"]
                        require([row["epoch"] for row in history] == list(range(1, max(expected_epochs)+1)), f"Missing epoch history: {path}")
                        require(meta["optimizerSteps"] == history[-1]["optimizerSteps"]
                                and meta["exposureSha256"] == history[-1]["exposureSha256"], f"Final exposure metadata mismatch: {path}")
                        expected_artifacts = {f"{stem}-{epoch}.npz" for epoch in expected_epochs for stem in ("weights", "predictions")}
                        require(set(meta["artifacts"]) == expected_artifacts, f"Missing/unexpected fit artifacts: {path}")
                        for name, digest in meta["artifacts"].items():
                            require(helpers.digest(path.parent / name) == digest, f"Changed fit artifact: {path.parent/name}")
                            artifact_count += 1
                        metadata[(cohort, kind, seed, outer_index, fold)] = meta
                        fit_paths[(cohort, kind, seed, outer_index, fold)] = path.parent
                        key = f"{cohort}/{kind}"
                        one = runtime.setdefault(key, {"fitCount": 0, "wallSeconds": 0., "optimizerSteps": 0,
                                                       "peakAllocatedCudaBytes": 0, "parameterCounts": []})
                        one["fitCount"] += 1
                        one["wallSeconds"] += meta["wallSeconds"]
                        one["optimizerSteps"] += meta["optimizerSteps"]
                        one["peakAllocatedCudaBytes"] = max(one["peakAllocatedCudaBytes"], meta["peakAllocatedCudaBytes"] or 0)
                        parameter_sets.setdefault(key, set()).add(meta["parameters"])
    pair_count = epoch_prefix_count = scaler_pairs = linear_trace_pairs = 0
    linear_max_error = 0.
    for kind in KINDS:
        for seed in seeds:
            for outer_index, outer in enumerate(groups):
                for fold in [*(f"inner-{i}" for i in range(len(groups)-1)), "refit"]:
                    for first, second in (("exact", "draft"), ("draft", "reviewed_export")):
                        a, b = metadata[(first, kind, seed, outer_index, fold)], metadata[(second, kind, seed, outer_index, fold)]
                        epoch_prefix_count += audit_exposure_pair(a, b, compare_draft=first == "draft")
                        pair_count += 1
                        left = fit_paths[(first, kind, seed, outer_index, fold)]
                        right = fit_paths[(second, kind, seed, outer_index, fold)]
                        with np.load(left/f"weights-{a['epochs'][0]}.npz", allow_pickle=False) as wa, np.load(right/f"weights-{b['epochs'][0]}.npz", allow_pickle=False) as wb:
                            require(np.array_equal(wa["mean"], wb["mean"]) and np.array_equal(wa["scale"], wb["scale"]), "Paired fitted scaler tensors differ")
                        scaler_pairs += 1
                        if kind == "linear" and first == "draft":
                            for epoch in sorted(set(a["epochs"]).intersection(b["epochs"])):
                                with np.load(left/f"predictions-{epoch}.npz", allow_pickle=False) as pa, np.load(right/f"predictions-{epoch}.npz", allow_pickle=False) as pb:
                                    require(set(pa.files) == set(pb.files), "Paired linear validation traces differ in scope")
                                    for identifier in pa.files:
                                        error = float(np.max(np.abs(pa[identifier][:, :3] - pb[identifier][:, :3])))
                                        require(error <= 1e-7, "Coverage-only auxiliary changes independent primary linear outputs")
                                        linear_max_error = max(linear_max_error, error)
                                        linear_trace_pairs += 1
    for key, values in parameter_sets.items():
        require(len(values) == 1, f"Architecture capacity changed within {key}")
        runtime[key]["parameterCounts"] = sorted(values)
    for kind in KINDS:
        require(len({next(iter(parameter_sets[f"{cohort}/{kind}"])) for cohort in COHORTS}) == 1, f"Architecture changed across cohorts: {kind}")
    return {"counts": {"fitMembershipsVerified": len(metadata), "artifactHashesVerified": artifact_count,
                       "pairedCohortFitPrefixesVerified": pair_count, "pairedEpochPrefixesVerified": epoch_prefix_count,
                       "pairedScalerTensorsVerified": scaler_pairs, "linearCoverageOnlyTracePairsVerified": linear_trace_pairs,
                       "linearCoverageOnlyPrimaryMaxAbsoluteError": linear_max_error,
                       "supervisionRowsIndependentlyReconstructed": len(counts_by_id)},
            "completedMetadata": completed_inputs, "trainingRuntime": runtime,
            "runtimeScope": "Summed fit durations, not elapsed study wall time: separately scheduled fits may overlap. Peak CUDA allocation is per process, not combined concurrent GPU use. These are training measurements, not phone/browser inference latency."}


def audit_selected_inner_candidates(study: Path, exact: dict, contract: dict, results: list[dict]) -> int:
    """Replay selected candidates from inner artifacts, never outer predictions."""
    import numpy as np
    from types import SimpleNamespace
    from analysis.crop_evaluation import RecordingIntervals, evaluate_f1_pad_p_core_r
    from analysis.neural_development import decode
    from analysis.schema import Interval, mask_for_times
    examples = {}
    for row in exact["recordings"]:
        with np.load(row["featureCaches"]["audiovisual"]["path"], allow_pickle=False) as data:
            times = data["times"].astype(np.float64)
            duration = float(json.loads(str(data["metadata_json"].item()))["duration"])
        convert = lambda rows: tuple(Interval(r["start"], r["end"], tuple(r.get("tags", ()))) for r in rows)
        ignored = convert(row.get("ignoredIntervals", []))
        examples[row["id"]] = SimpleNamespace(id=row["id"], group=row["sourceGroup"], times=times, duration=duration,
                                             truth=convert(row["rallies"]), ignored=ignored, valid=mask_for_times(times, ignored))
    replayed = 0
    for result in results:
        for selection in result["selections"]:
            outer = selection["heldSourceGroup"]
            outer_index = contract["groups"].index(outer)
            epoch = selection["epoch"]
            probabilities = {}
            for inner_index, inner in enumerate(group for group in contract["groups"] if group != outer):
                path = study/"fits"/result["cohort"]/result["kind"]/str(result["seed"])/f"outer-{outer_index}"/f"inner-{inner_index}"/f"predictions-{epoch}.npz"
                expected_ids = {e.id for e in examples.values() if e.group == inner}
                with np.load(path, allow_pickle=False) as predictions:
                    require(set(predictions.files) == expected_ids, "Selected inner trace has unexpected validation recordings")
                    for identifier in predictions.files:
                        value = predictions[identifier]
                        require(value.shape == (len(examples[identifier].times), 4) and np.isfinite(value).all()
                                and (value >= 0).all() and (value <= 1).all(), "Invalid selected inner probabilities")
                        probabilities[identifier] = value
            fitting = [e for e in examples.values() if e.group != outer]
            rows = [RecordingIntervals(e.id, "development", e.duration, e.truth,
                                       tuple(decode(e, probabilities[e.id], selection["decoder"])), e.ignored) for e in fitting]
            score = evaluate_f1_pad_p_core_r(rows, [2.], 3.)[0]
            require(math.isclose(score["F1_padP_coreR"], selection["innerF1_padP_coreR"], abs_tol=1e-12, rel_tol=0)
                    and math.isclose(score["R_core"], selection["innerR_core"], abs_tol=1e-12, rel_tol=0),
                    "Selected inner metrics differ from held-source-group artifact replay")
            replayed += 1
    return replayed


def summarize(study: Path, manifest_path: Path, *, progress: bool = False) -> tuple[dict, dict | None]:
    registration, manifest, exact, report, inputs = read_inputs(study, manifest_path, progress=progress)
    contract, results = registration["contract"], report["results"]
    expected = {(cohort, kind, seed) for cohort in COHORTS for kind in KINDS for seed in contract["seeds"]}
    observed = {result_key(row) for row in results}
    if progress:
        return {"status": "complete" if observed == expected and (study/"report.json").exists() else "in-progress",
                "completedResults": len(results), "expectedResults": len(expected),
                "missing": [list(key) for key in sorted(expected-observed)],
                "results": [{"cohort": row["cohort"], "kind": row["kind"], "seed": row["seed"],
                             "F1_padP_coreR": row["evaluation"]["objective"], "R_core": row["evaluation"]["primary"]["R_core"]}
                            for row in results], "note": "Progress only; no completed audit, comparison gate or selection."}, None
    require(report["manifestSha256"] == contract["manifestSha256"], "Report manifest hash mismatch")
    inputs["frozenModules"] = {}
    for name, digest in contract["code"].items():
        path = REPO / "analysis" / name
        require(helpers.digest(path) == digest, f"Frozen module hash mismatch: {name}")
        inputs["frozenModules"][name] = identity(path)
    inputs["featureCaches"] = []
    for row in manifest["exactRows"] + manifest["draftRows"] + manifest["coverageRows"]:
        cache = row["featureCaches"]["audiovisual"]
        require(helpers.digest(Path(cache["path"])) == cache["sha256"], f"Changed feature cache: {row['id']}")
        inputs["featureCaches"].append(identity(Path(cache["path"])))
    floor = contract["selection"]["recallEligibilityFloor"]
    inputs["resultFiles"] = []
    for row in results:
        path = study / f"result-{row['cohort']}-{row['kind']}-{row['seed']}.json"
        require(helpers.load(path) == row, f"Result file differs from report: {path}")
        inputs["resultFiles"].append(identity(path))
        helpers.assert_result_revision(exact, row["predictions"], str(result_key(row)))
        replay = evaluate_predictions(row["predictions"], primary_padding_seconds=2., join_gap_seconds=3.)
        require(replay == row["evaluation"], f"Canonical metric replay differs: {result_key(row)}")
        require(sorted(choice["heldSourceGroup"] for choice in row["selections"]) == sorted(contract["groups"]), "Missing/duplicate outer selections")
        for choice in row["selections"]:
            require(choice["epoch"] in contract["checkpointEpochs"] and choice["decoder"] in contract["decoderCandidates"], "Selection outside frozen candidate grid")
            selection_feasible(choice, floor)
    audit = audit_fits(study, manifest, contract, registration["sha256"], results)
    audit["counts"].update({"exactResultTargetRevisionsVerified": len(results), "canonicalEvaluationsReplayedExactly": len(results),
                            "frozenModuleHashesVerified": len(contract["code"]), "featureCacheHashesVerified": len(inputs["featureCaches"])})
    audit["counts"]["selectedInnerCandidatesReplayed"] = audit_selected_inner_candidates(study, exact, contract, results)
    per_seed, aggregates, comparisons, losses = summarize_results(results, contract, floor)
    selections = [choice for row in results for choice in row["selections"]]
    audit["selectionCounts"] = {"epochs": dict(sorted(Counter(choice["epoch"] for choice in selections).items())),
                                "enterThresholds": dict(sorted(Counter(choice["decoder"]["enter"] for choice in selections).items())),
                                "infeasible": sum(not choice["recallEligibilityPassed"] for choice in selections)}
    summary = {"schemaVersion": 1, "kind": "expanded-neural-development-summary-v1",
               "createdAt": datetime.now(timezone.utc).isoformat(), "status": "completed-expanded-development-audit",
               "protectedTestOpened": False, "productionPromotionAllowed": False,
               "manifestSha256": contract["manifestSha256"], "contractSha256": registration["sha256"],
               "primaryMetric": "F1_padP_coreR", "primaryPaddingSeconds": 2, "joinGapSeconds": 3,
               "scope": {"exactRecordingCount": len(exact["recordings"]), "exactSourceGroupCount": len(contract["groups"]),
                         "exactSourceGroups": contract["groups"], "exactRallyCount": sum(len(row["rallies"]) for row in exact["recordings"]),
                         "auxiliaryDraftRecordings": len(manifest["draftRows"]), "auxiliaryCoverageRecordings": len(manifest["coverageRows"]),
                         "seedCount": len(contract["seeds"])},
               "aggregation": "Each seed pools metric numerators/denominators over exact recordings; aggregate summaries average separate seed runs. No per-video F1 averaging or seed duplication in pooled metrics.",
               "limitations": [
                   "This is adaptive development evidence on already available source groups, not an independent protected test or production promotion.",
                   "Seed variation does not quantify uncertainty across new capture sessions; source-group deltas are descriptive and no significance claim is made.",
                   "A/B/C use paired exact batch exposure and exact-only scaler/class weights. Auxiliary exposure is additional compute; selected epoch lengths may differ, so common refit prefixes are audited.",
                   "Draft intervals supervise live state only away from uncertain endpoints. Reviewed exports supervise actual retained coverage only, including owner padding choices; neither supplies exact boundary targets or evaluation truth.",
                   "Inner recall eligibility does not guarantee outer recall or short-rally retention. Outer attainment of the inner .95 floor is a descriptive diagnostic, not an additional frozen screen. Inner infeasibility is reported separately and prevents a feasible-candidate claim.",
                   "Linear output heads are independent; coverage-only supervision cannot directly transfer a learned representation into primary linear logits. Linear clipping is per head, avoiding gradient-norm coupling.",
                   "The keep head is auxiliary; primary prediction intervals decode only live/serve/end. All four padding cases use the same fixed predictions.",
                   "Loss identities refer to original zero-based rally indexes. Complete and partial losses must be read together; a partial loss becoming complete is not an improvement.",
                   "Runtime summaries measure desktop training, not actual phone/browser inference performance. Summed fit durations may overlap in wall time, and peak CUDA allocations are per process rather than combined concurrent GPU use.",
               ], "inputs": inputs, "audit": audit, "aggregates": aggregates, "perSeed": per_seed, "comparisons": comparisons}
    return summary, {"scope": summary["scope"], "perSeed": losses}


def markdown(summary: dict) -> str:
    scope = summary["scope"]
    lines = ["# Expanded neural development summary", "",
             f"All results evaluate the same {scope['exactRecordingCount']} exact recordings / {scope['exactSourceGroupCount']} source groups. A = exact; B = exact + reviewed drafts; C = B + reviewed export coverage. Auxiliary data never contribute evaluation targets.", "",
             "Primary: pooled `F1_padP_coreR` at 2-second symmetric padding and strictly less than 3-second gap joining. Kind/cohort means average separate seed runs. No production promotion.", "",
             "| Cohort | Kind | Mean F1 | Mean R_core | Mean P_pad | Event F1 | Infeasible inner selections | All seed recalls >= floor |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for row in summary["aggregates"]:
        p = row["meanSeedPrimary"]
        lines.append(f"| {row['cohort']} | {row['kind']} | {p['F1_padP_coreR']:.4f} | {p['R_core']:.4f} | {p['P_pad']:.4f} | {row['meanSeedEventF1']:.4f} | {row['infeasibleInnerSelections']} | {row['allSeedOuterCoreRecallFloorSatisfied']} |")
    lines += ["", "## Paired comparisons", "",
              "The frozen development screen requires mean F1 gain >=.02, at least two thirds of seeds positive, a majority of source groups positive, and mean recall regression no worse than .005. Inner selection feasibility is reported separately; outer attainment of .95 recall is descriptive only. Passing does not authorize deployment.", "",
              "| Comparison | F1 delta | R_core delta | Export delta s | Positive seeds | Positive groups | Screen |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for row in summary["comparisons"]:
        d, g = row["meanSeedPrimaryDelta"], row["developmentScreen"]
        lines.append(f"| {row['name']} | {d['F1_padP_coreR']:+.4f} | {d['R_core']:+.4f} | {d['paddedModelExportSeconds']:+.1f} | {g['positiveSeeds']}/{g['seedCount']} | {g['positiveSourceGroups']}/{g['sourceGroupCount']} | {'PASS' if g['passed'] else 'FAIL'} |")
    lines += ["", "## Padding sensitivity", "",
              "Means of separately pooled seed results; padding 2 is fixed for every comparison.", "",
              "| Cohort | Kind | Pad s | P_pad | R_core | F1_padP_coreR | Model s | Human s | Difference s |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary["aggregates"]:
        for p in row["meanSeedPadding"]:
            lines.append(f"| {row['cohort']} | {row['kind']} | {p['paddingSecondsBeforeAndAfter']:g} | {p['P_pad']:.4f} | {p['R_core']:.4f} | {p['F1_padP_coreR']:.4f} | {p['paddedModelExportSeconds']:.1f} | {p['paddedHumanExportSeconds']:.1f} | {p['exportDurationDifferenceSeconds']:+.1f} |")
    lines += ["", "## Per-seed rally retention", "",
              "Complete/partial losses refer to original gold rallies after fixed 2-second export padding. Exact identities and paired regressions/recoveries are in `loss-identities.json`; core, outcome and boundary guardrails are in `summary.json`.", "",
              "| Cohort | Kind | Seed | F1 | R_core | Complete / partial export losses |",
              "|---|---|---:|---:|---:|---:|"]
    for row in summary["perSeed"]:
        p, losses = row["primary"], row["coverage"]["primaryExportCoverage"]
        lines.append(f"| {row['cohort']} | {row['kind']} | {row['seed']} | {p['F1_padP_coreR']:.4f} | {p['R_core']:.4f} | {losses['completeRallyLosses']} / {losses['partialRallyLosses']} |")
    lines += ["", "## Training runtime", "",
              "Desktop fitting measurements across all inner fits and selected refits; not phone/browser latency. Fit durations may overlap because of separate scheduling, so their sum is not elapsed study wall time. CUDA peaks are per process, not combined concurrent use.", "",
              "| Cohort / kind | Fits | Summed fit hours | Optimizer steps | Per-process peak CUDA MiB | Parameters |",
              "|---|---:|---:|---:|---:|---:|"]
    for key, row in summary["audit"]["trainingRuntime"].items():
        lines.append(f"| {key} | {row['fitCount']} | {row['wallSeconds']/3600:.3f} | {row['optimizerSteps']} | {row['peakAllocatedCudaBytes']/1048576:.1f} | {row['parameterCounts'][0]} |")
    lines += ["", "## Integrity and limitations", "", json.dumps(summary["audit"]["counts"], sort_keys=True), ""]
    lines.extend(f"- {item}" for item in summary["limitations"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "study")
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifest.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    summary, losses = summarize(args.study, args.manifest, progress=args.progress)
    if args.progress:
        print(json.dumps(summary, indent=2, allow_nan=False))
        return
    output = args.output or args.study
    names = ("summary.json", "summary.md", "loss-identities.json")
    require(not any((output / name).exists() for name in names), "Refusing to overwrite existing summary artifacts")
    output.mkdir(parents=True, exist_ok=True)
    loss_path = output / "loss-identities.json"
    with loss_path.open("x", encoding="utf-8") as handle:
        json.dump(losses, handle, indent=2, allow_nan=False)
        handle.write("\n")
    summary["lossIdentitiesArtifact"] = identity(loss_path)
    with (output / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    with (output / "summary.md").open("x", encoding="utf-8") as handle:
        handle.write(markdown(summary))
    print(json.dumps({"status": summary["status"], "output": str(output), "audit": summary["audit"]["counts"]}, indent=2))


if __name__ == "__main__":
    main()
