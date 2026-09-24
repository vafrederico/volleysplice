#!/usr/bin/env python3
"""Summarize completed neural development results without fitting or selection.

--progress reads only completed result files and prints status without writing.
Without that flag, a complete report is required and new summary.json/summary.md
are exclusively created. Existing study/checkpoint/report files are never changed.
"""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(private_value('private-reference-0096'))
METRICS = ("P_pad", "R_core", "F1_padP_coreR", "paddedModelExportSeconds",
           "paddedHumanExportSeconds", "exportDurationDifferenceSeconds")
COVERAGE_KEYS = ("evaluableRallies", "completeRallyLosses", "partialRallyLosses", "fullyCoveredRallies")
SLICES = ("shortAtMost3Seconds", "ace", "serviceFault")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values) -> float:
    return float(statistics.fmean(values))


def interval_identity(rows: list[dict]) -> tuple:
    """Exact evaluation-relevant boundaries/tags, without rounding or unioning."""
    return tuple((row["start"], row["end"], tuple(row.get("tags", ()))) for row in rows)


def assert_result_revision(manifest: dict, predictions: list[dict], name: str) -> None:
    """Bind raw result targets to every original gold and ignored interval.

    Remaining core durations alone cannot identify an ignored revision: ignored
    spans outside core rallies and equal-duration shifted holes must also match.
    """
    expected = {row["id"]: row for row in manifest["recordings"]}
    observed = {row["id"]: row for row in predictions}
    if len(observed) != len(predictions) or set(observed) != set(expected):
        raise ValueError(f"{name}: duplicate/missing/unexpected recording identity")
    for identifier, truth in expected.items():
        row = observed[identifier]
        duration = truth.get("featureCaches", {}).get("audiovisual", {}).get("metadata", {}).get(
            "duration", truth["durationSeconds"])
        if row["sourceGroup"] != truth["sourceGroup"] or row["durationSeconds"] != duration:
            raise ValueError(f"{name}/{identifier}: source group or duration differs from frozen manifest")
        for key in ("rallies", "ignoredIntervals"):
            if interval_identity(row.get(key, [])) != interval_identity(truth.get(key, [])):
                raise ValueError(f"{name}/{identifier}: exact {key} revision differs from frozen manifest")


def verify_frozen_inputs(contract: dict, baseline: dict, results: list[dict]) -> dict:
    """Verify the manifest bytes and replay baseline metrics from that revision."""
    manifest_path = Path(baseline["manifestPath"])
    manifest_hash = digest(manifest_path)
    if manifest_hash != baseline["manifestSha256"] or manifest_hash != contract["manifestSha256"]:
        raise ValueError("frozen manifest bytes differ from study or production baseline hash")
    manifest = load(manifest_path)
    for row in results:
        assert_result_revision(manifest, row["predictions"], f"{row['kind']}/{row['seed']}")
    # Baseline artifacts store predictions plus a manifest binding, not duplicate
    # raw targets. Reconstruct from verified manifest targets to validate that binding.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis.neural_evaluation import evaluate_predictions
    for name, predictions in baseline["predictions"].items():
        if set(predictions) != {row["id"] for row in manifest["recordings"]}:
            raise ValueError(f"baseline {name}: recording scope differs from frozen manifest")
        rows = []
        for row in manifest["recordings"]:
            duration = row.get("featureCaches", {}).get("audiovisual", {}).get("metadata", {}).get(
                "duration", row["durationSeconds"])
            rows.append({"id": row["id"], "sourceGroup": row["sourceGroup"],
                         "durationSeconds": duration, "rallies": row["rallies"],
                         "ignoredIntervals": row.get("ignoredIntervals", []),
                         "predictions": predictions[row["id"]]})
        replay = evaluate_predictions(rows, primary_padding_seconds=contract["targetPaddingSeconds"],
                                      join_gap_seconds=contract["joinGapSeconds"])
        if replay != baseline["results"][name]:
            raise ValueError(f"baseline {name}: stored evaluation differs from frozen-manifest replay")
    return {"path": str(manifest_path), "sha256": manifest_hash,
            "recordingCount": len(manifest["recordings"]),
            "sourceGroups": sorted({row["sourceGroup"] for row in manifest["recordings"]}),
            "neuralResultTargetRevisionsVerified": len(results),
            "baselineVariantsReplayed": len(baseline["predictions"])}


def study_scope(contract: dict, report: dict, verified_manifest: dict) -> dict:
    """Use the verified data scope, refusing contradictory report prose inputs."""
    groups = verified_manifest["sourceGroups"]
    if (sorted(contract["groups"]) != groups or sorted(report["sourceGroups"]) != groups
            or report["records"] != verified_manifest["recordingCount"]):
        raise ValueError("study report/contract counts differ from the frozen manifest")
    return {"recordingCount": verified_manifest["recordingCount"],
            "sourceGroupCount": len(groups), "sourceGroups": groups,
            "seedCount": len(contract["seeds"])}


def compact_evaluation(evaluation: dict) -> dict:
    padding = evaluation["padding"]
    if [row["paddingSecondsBeforeAndAfter"] for row in padding] != [0, 1, 2, 3]:
        raise ValueError("every result must contain the four ordered symmetric padding cases")
    guard = evaluation["guardrails"]
    return {
        "primary": {key: evaluation["primary"][key] for key in METRICS},
        "padding": [{"paddingSecondsBeforeAndAfter": row["paddingSecondsBeforeAndAfter"],
                     **{key: row[key] for key in METRICS}} for row in padding],
        "eventF1": guard["eventF1"],
        "boundaryErrors": {key: guard[key] for key in ("startBoundaryMaeSeconds", "endBoundaryMaeSeconds",
                                                        "startBoundaryP90Seconds", "endBoundaryP90Seconds")},
        "outcomeRecall": {name: {key: guard["outcomeSlices"][name][key]
                                  for key in ("rallies", "strictMatchRecall", "anyOverlapRecall", "coverageAtLeast95Rate")}
                          for name in SLICES},
        "coverage": {scope: {key: guard[scope][key] for key in COVERAGE_KEYS}
                     for scope in ("coreCoverage", "primaryExportCoverage")},
    }


def assert_comparable(left: dict, right: dict) -> None:
    left_scope = {(r["id"], r["sourceGroup"]) for r in left["recordings"]}
    right_scope = {(r["id"], r["sourceGroup"]) for r in right["recordings"]}
    if left_scope != right_scope or left["metricContract"] != right["metricContract"]:
        raise ValueError("results differ in recording/source-group scope or metric contract")
    a = left["guardrails"]["primaryExportCoverage"]["rallies"]
    b = right["guardrails"]["primaryExportCoverage"]["rallies"]
    identity = lambda row: (row["recordingId"], row["truthIndex"], row["start"], row["end"], row["evaluableCoreSeconds"])
    if {identity(row) for row in a} != {identity(row) for row in b}:
        raise ValueError("results differ in exact gold boundaries or ignored-time revision")


def coverage_comparison(candidate: dict, baseline: dict, scope: str) -> dict:
    reference = {(row["recordingId"], row["truthIndex"]): row
                 for row in baseline["guardrails"][scope]["rallies"]}
    changed = []
    counts = {"newCompleteLosses": 0, "newPartialLossesFromFullyCovered": 0,
              "recoveredCompleteLosses": 0, "coverageRegressions": 0, "coverageImprovements": 0}
    for row in candidate["guardrails"][scope]["rallies"]:
        other = reference[(row["recordingId"], row["truthIndex"])]
        delta = row["retainedCoreSeconds"] - other["retainedCoreSeconds"]
        if abs(delta) <= 1e-9:
            continue
        flags = {
            "newCompleteLosses": row["completelyLost"] and not other["completelyLost"],
            "newPartialLossesFromFullyCovered": row["partiallyLost"] and other["fullyCovered"],
            "recoveredCompleteLosses": other["completelyLost"] and not row["completelyLost"],
            "coverageRegressions": delta < -1e-9, "coverageImprovements": delta > 1e-9,
        }
        for name, present in flags.items():
            counts[name] += int(present)
        changed.append({
            **{key: row[key] for key in ("recordingId", "truthIndex", "start", "end", "tags")},
            "baselineCoverage": other["coverage"], "candidateCoverage": row["coverage"],
            "retainedCoreSecondsDelta": delta,
            "baselineState": "complete-loss" if other["completelyLost"] else "partial-loss" if other["partiallyLost"] else "fully-covered",
            "candidateState": "complete-loss" if row["completelyLost"] else "partial-loss" if row["partiallyLost"] else "fully-covered",
            "changes": [name for name, present in flags.items() if present],
        })
    return {"counts": counts, "changedRallies": changed}


def paired_groups(candidate: dict, baseline: dict) -> dict:
    return {group: {
        "F1_padP_coreRDelta": values["objective"] - baseline["sourceGroups"][group]["objective"],
        "R_coreDelta": values["primary"]["R_core"] - baseline["sourceGroups"][group]["primary"]["R_core"],
        "paddedModelExportSecondsDelta": values["primary"]["paddedModelExportSeconds"] - baseline["sourceGroups"][group]["primary"]["paddedModelExportSeconds"],
        "direction": "positive" if values["objective"] > baseline["sourceGroups"][group]["objective"] + 1e-12 else
                     "negative" if values["objective"] < baseline["sourceGroups"][group]["objective"] - 1e-12 else "tie",
    } for group, values in candidate["sourceGroups"].items()}


def proposed_gate(mean_delta: float, seed_deltas: list[float], group_deltas: dict, recall_delta: float) -> dict:
    positive_seeds = sum(value > 1e-12 for value in seed_deltas)
    positive_groups = sum(value > 1e-12 for value in group_deltas.values())
    tests = {
        "meanF1GainAtLeast002": mean_delta >= 0.02 - 1e-12,
        "atLeastTwoThirdsSeedsPositive": positive_seeds >= math.ceil(2 * len(seed_deltas) / 3),
        "majoritySourceGroupsPositive": positive_groups > len(group_deltas) / 2,
        "meanCoreRecallRegressionAtMost0005": recall_delta >= -0.005 - 1e-12,
    }
    return {"scope": "development feasibility only; no production promotion",
            "passed": all(tests.values()), "checks": tests,
            "positiveSeeds": positive_seeds, "seedCount": len(seed_deltas),
            "positiveSourceGroups": positive_groups, "sourceGroupCount": len(group_deltas)}


def summarize(study: Path, baseline_path: Path, *, progress: bool = False) -> dict:
    registration = load(study / "preregistration.json")
    contract = registration["contract"]
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(canonical.encode()).hexdigest() != registration["sha256"]:
        raise ValueError("preregistration contract hash is invalid")
    complete_path = study / "report.json"
    if complete_path.exists():
        report = load(complete_path)
        if report["contractSha256"] != registration["sha256"]:
            raise ValueError("study report and preregistration identities differ")
        results = report["results"]
    elif progress:
        results = [load(path) for path in sorted(study.glob("result-*.json"))]
    else:
        raise FileNotFoundError("completed study report.json is required; use --progress for a read-only partial status")
    expected = {(kind, seed) for kind in contract["kinds"] for seed in contract["seeds"]}
    observed = {(row["kind"], row["seed"]) for row in results}
    if len(observed) != len(results) or not observed <= expected:
        raise ValueError("duplicate or unexpected result kind/seed")
    if not progress and observed != expected:
        raise ValueError("completed study is missing preregistered kind/seed results")
    if progress:
        return {"status": "complete" if observed == expected and complete_path.exists() else "in-progress",
                "completedResults": len(results), "expectedResults": len(expected),
                "missing": [{"kind": kind, "seed": seed} for kind, seed in sorted(expected - observed)],
                "results": [{"kind": row["kind"], "seed": row["seed"],
                             "F1_padP_coreR": row["evaluation"]["primary"]["F1_padP_coreR"],
                             "R_core": row["evaluation"]["primary"]["R_core"]} for row in results],
                "note": "Partial scores are status only, not model selection or completed gate decisions."}
    baseline = load(baseline_path)
    if baseline["manifestSha256"] != contract["manifestSha256"]:
        raise ValueError("production baseline used a different frozen manifest")
    verified_manifest = verify_frozen_inputs(contract, baseline, results)
    scope = study_scope(contract, report, verified_manifest)
    by_pair = {(row["kind"], row["seed"]): row for row in results}
    shipped = baseline["results"]["shipped-union"]
    summaries = []
    kinds = []
    for kind in contract["kinds"]:
        seed_rows = []
        for seed in contract["seeds"]:
            raw = by_pair[(kind, seed)]
            evaluation = raw["evaluation"]
            linear = by_pair[("linear", seed)]["evaluation"]
            assert_comparable(evaluation, linear)
            assert_comparable(evaluation, shipped)
            one = {"kind": kind, "seed": seed, **compact_evaluation(evaluation),
                   "pairedVsLinear": {
                       "primaryDelta": {key: evaluation["primary"][key] - linear["primary"][key] for key in METRICS},
                       "sourceGroups": paired_groups(evaluation, linear),
                   },
                   "lossIdentities": {name: {scope: coverage_comparison(evaluation, reference, scope)
                                              for scope in ("coreCoverage", "primaryExportCoverage")}
                                      for name, reference in (("sameSeedLinear", linear), ("shippedUnionRetrospective", shipped))},
                   "selections": raw["selections"]}
            summaries.append(one)
            seed_rows.append(one)
        group_names = sorted(seed_rows[0]["pairedVsLinear"]["sourceGroups"])
        group_mean = {group: {
            key: mean(row["pairedVsLinear"]["sourceGroups"][group][key] for row in seed_rows)
            for key in ("F1_padP_coreRDelta", "R_coreDelta", "paddedModelExportSecondsDelta")}
                      for group in group_names}
        for row in group_mean.values():
            row["direction"] = "positive" if row["F1_padP_coreRDelta"] > 1e-12 else "negative" if row["F1_padP_coreRDelta"] < -1e-12 else "tie"
        primary_mean = {key: mean(row["primary"][key] for row in seed_rows) for key in METRICS}
        delta_mean = {key: mean(row["pairedVsLinear"]["primaryDelta"][key] for row in seed_rows) for key in METRICS}
        gate = None if kind == "linear" else proposed_gate(
            delta_mean["F1_padP_coreR"], [row["pairedVsLinear"]["primaryDelta"]["F1_padP_coreR"] for row in seed_rows],
            {group: value["F1_padP_coreRDelta"] for group, value in group_mean.items()}, delta_mean["R_core"])
        kinds.append({
            "kind": kind, "seedCount": len(seed_rows), "meanSeedPrimary": primary_mean,
            "meanSeedPrimaryDeltaVsLinear": delta_mean, "sourceGroupMeanPairedDeltaVsLinear": group_mean,
            "meanSeedPadding": [{"paddingSecondsBeforeAndAfter": padding,
                                 **{key: mean(row["padding"][padding][key] for row in seed_rows) for key in METRICS}}
                                for padding in range(4)],
            "meanSeedEventF1": mean(row["eventF1"] for row in seed_rows),
            "meanSeedOutcomeStrictRecall": {name: mean(row["outcomeRecall"][name]["strictMatchRecall"] for row in seed_rows) for name in SLICES},
            "meanSeedPrimaryExportLossCounts": {key: mean(row["coverage"]["primaryExportCoverage"][key] for row in seed_rows) for key in COVERAGE_KEYS},
            "proposedFeasibilityGate": gate,
        })
    return {
        "schemaVersion": 1, "kind": "volleycut-neural-development-summary-v1",
        "createdAt": datetime.now(timezone.utc).isoformat(), "status": "completed-development-feasibility",
        "manifestSha256": contract["manifestSha256"], "contractSha256": registration["sha256"],
        "scope": scope,
        "primaryMetric": "F1_padP_coreR", "primaryPaddingSeconds": 2.0, "joinGapSeconds": 3.0,
        "aggregation": "Each seed first pools duration numerators/denominators across recordings; kind summaries average those separate seed runs, never per-video F1.",
        "proposedGate": {"minimumMeanSeedF1DeltaVsLinear": 0.02, "minimumPositiveSeedFraction": 2/3,
                         "sourceGroupRequirement": "strict majority of groups have positive mean paired delta",
                         "maximumMeanSeedRCoreRegression": 0.005,
                         "role": "summary feasibility screen; not used for training, checkpoint/decoder selection, or promotion"},
        "limitations": [
            f"Only {scope['sourceGroupCount']} independent source groups; seed variation does not measure uncertainty across new capture sessions.",
            "This is a development feasibility comparison, not a protected-test result or deployment decision.",
            "Same-seed linear controls use the same fitting population and nested selection as the neural models.",
            "Production refits use historical no-beach allowlists and target/training/decoder settings; inspect their population audit for scope differences. The same-seed linear model remains the capacity control.",
            "Shipped union results are training/selection-exposed retrospective product references.",
            "Loss identities refer to original gold rally indexes within each recording, not predicted fragment indexes.",
            "Strict event recall uses IoU0.5 matching on uncensored original events; any-overlap recall is separately reported.",
        ],
        "inputs": {"report": {"path": str(complete_path), "sha256": digest(complete_path)},
                   "baseline": {"path": str(baseline_path), "sha256": digest(baseline_path)},
                   "verifiedManifest": verified_manifest,
                   "script": {"path": str(Path(__file__).resolve()), "sha256": digest(Path(__file__).resolve())}},
        "kinds": kinds, "perSeed": summaries,
        "productionReferences": {name: compact_evaluation(evaluation) for name, evaluation in baseline["results"].items()},
    }


def markdown(summary: dict) -> str:
    scope = summary["scope"]
    required_positive_seeds = math.ceil(2 * scope["seedCount"] / 3)
    lines = ["# Neural development feasibility summary", "",
             "Primary: pooled `F1_padP_coreR` at ±2 seconds; strictly less than 3-second export gap joining. "
             f"{scope['sourceGroupCount']} source groups, {scope['recordingCount']} recordings. No production promotion.", "",
             f"Kind means average {scope['seedCount']} separately pooled seed results. Comparisons to the same-seed linear control use identical fitting populations.", "",
             "| Kind | Mean F1 | Δ vs linear | Mean R_core | Δ R_core | Mean export s | Event F1 | Gate |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in summary["kinds"]:
        primary, delta = row["meanSeedPrimary"], row["meanSeedPrimaryDeltaVsLinear"]
        gate = row["proposedFeasibilityGate"]
        lines.append(f"| {row['kind']} | {primary['F1_padP_coreR']:.4f} | {delta['F1_padP_coreR']:+.4f} | {primary['R_core']:.4f} | {delta['R_core']:+.4f} | {primary['paddedModelExportSeconds']:.1f} | {row['meanSeedEventF1']:.4f} | {'control' if gate is None else 'pass' if gate['passed'] else 'fail'} |")
    lines += ["", f"The proposed feasibility screen requires mean F1 improvement ≥0.02, positive improvement in at least {required_positive_seeds} of {scope['seedCount']} seeds, a positive mean paired improvement in a strict majority of source groups, and mean R_core regression no worse than 0.005. It is not a deployment gate.", "",
              "## Required padding sensitivity by kind and seed", "",
              "| Kind | Seed | Padding | P_pad | R_core | F1_padP_coreR | Model export s | Human export s | Difference s |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary["perSeed"]:
        for pad in row["padding"]:
            lines.append(f"| {row['kind']} | {row['seed']} | {pad['paddingSecondsBeforeAndAfter']:g} | {pad['P_pad']:.4f} | {pad['R_core']:.4f} | {pad['F1_padP_coreR']:.4f} | {pad['paddedModelExportSeconds']:.1f} | {pad['paddedHumanExportSeconds']:.1f} | {pad['exportDurationDifferenceSeconds']:+.1f} |")
    lines += ["", "## Paired source-group direction versus linear", "",
              "| Kind | Seed | Source group | Δ F1 | Δ R_core | Δ export s | Direction |",
              "|---|---|---|---:|---:|---:|---|"]
    group_rows = [(row["kind"], row["seed"], row["pairedVsLinear"]["sourceGroups"]) for row in summary["perSeed"]]
    group_rows += [(row["kind"], "mean", row["sourceGroupMeanPairedDeltaVsLinear"]) for row in summary["kinds"]]
    for kind, seed, groups in group_rows:
        if kind == "linear":
            continue
        for group, values in groups.items():
            lines.append(f"| {kind} | {seed} | {group} | {values['F1_padP_coreRDelta']:+.4f} | {values['R_coreDelta']:+.4f} | {values['paddedModelExportSecondsDelta']:+.1f} | {values['direction']} |")
    lines += ["", "## Event and original-rally coverage guardrails", "",
              "Strict recall is IoU0.5 event recall. Loss counts below use the final ±2-second export union; unpadded counts and all identities are in JSON.", "",
              "| Kind | Seed | Event F1 | Short recall | Ace recall | Fault recall | Complete losses | Partial losses | Fully covered | New complete vs linear | New complete vs shipped |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary["perSeed"]:
        outcome, coverage = row["outcomeRecall"], row["coverage"]["primaryExportCoverage"]
        new = lambda name: row["lossIdentities"][name]["primaryExportCoverage"]["counts"]["newCompleteLosses"]
        lines.append(f"| {row['kind']} | {row['seed']} | {row['eventF1']:.4f} | {outcome['shortAtMost3Seconds']['strictMatchRecall']:.4f} | {outcome['ace']['strictMatchRecall']:.4f} | {outcome['serviceFault']['strictMatchRecall']:.4f} | {coverage['completeRallyLosses']} | {coverage['partialRallyLosses']} | {coverage['fullyCoveredRallies']} | {new('sameSeedLinear')} | {new('shippedUnionRetrospective')} |")
    lines += ["", "## Production references", "",
              "Shipped results are retrospective. Production refits hold out current fitting groups and retain historical allowlists and target/training/decoder settings. Inspect their population audit: fitting populations may match or differ on this scope. The same-seed linear model remains the capacity control.", "",
              "| Reference | Padding | P_pad | R_core | F1_padP_coreR | Model export s | Human export s | Difference s |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in summary["productionReferences"].items():
        for pad in row["padding"]:
            lines.append(f"| {name} | {pad['paddingSecondsBeforeAndAfter']:g} | {pad['P_pad']:.4f} | {pad['R_core']:.4f} | {pad['F1_padP_coreR']:.4f} | {pad['paddedModelExportSeconds']:.1f} | {pad['paddedHumanExportSeconds']:.1f} | {pad['exportDurationDifferenceSeconds']:+.1f} |")
    lines += ["", "## Interpretation limits", ""] + [f"- {note}" for note in summary["limitations"]]
    lines += ["", "Exact changed-rally identities are in `summary.json → perSeed → lossIdentities`, separately for unpadded core and padded export coverage against linear and shipped union. Each records recording ID, truth index, timestamps, tags, before/after coverage and change type.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=ROOT / "nested-study-v1")
    parser.add_argument("--baseline", type=Path, default=ROOT / "production-baseline/baseline.json")
    parser.add_argument("--output-dir", type=Path, help="default: study directory; outputs summary.json and summary.md")
    parser.add_argument("--progress", action="store_true", help="print completed-result status only; do not write")
    args = parser.parse_args()
    if args.progress:
        print(json.dumps(summarize(args.study_dir, args.baseline, progress=True), indent=2))
        return
    output = args.output_dir or args.study_dir
    targets = [output / "summary.json", output / "summary.md"]
    if any(path.exists() for path in targets):
        raise FileExistsError("refusing to overwrite summary artifacts; choose a new --output-dir")
    summary = summarize(args.study_dir, args.baseline)
    payloads = [json.dumps(summary, indent=2, allow_nan=False) + "\n", markdown(summary)]
    output.mkdir(parents=True, exist_ok=True)
    for path, payload in zip(targets, payloads):
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
    print(json.dumps({"outputs": [{"path": str(path), "sha256": digest(path)} for path in targets],
                      "status": summary["status"]}))


if __name__ == "__main__":
    main()
