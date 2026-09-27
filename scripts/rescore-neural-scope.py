#!/usr/bin/env python3
"""Re-pool frozen phase-one predictions on the user-requested non-beach scope.

This is a retrospective diagnostic, not a beach-free refit. Original training
and inner selection included beach. No prediction, decoder, epoch, or seed is
changed or selected. Outputs are exclusively created in a new leaf directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
BASE = Path(private_value('private-reference-0057'))
ORIGINAL = BASE / "2026-09-18-phase1"
NONBEACH = BASE / "2026-09-19-nonbeach"
EXPECTED_REPORT_SHA256 = "7b0e308f3df74dba6bdec448dbc1a4e02c66a1b72e490d4a7099afe4c1389280"
EXPECTED_SCOPE_SHA256 = "04365c5fe26a3186acad90cba5b4f595588d17d24240251c26a2f9b2aab60532"
SCOPES = ("coreCoverage", "primaryExportCoverage")

from analysis.neural_evaluation import evaluate_predictions

HELPER_PATH = REPO / "scripts/summarize-neural-development.py"
spec = importlib.util.spec_from_file_location("neural_summary_helpers", HELPER_PATH)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def identify(path: Path) -> dict:
    return {"path": str(path), "sha256": helpers.digest(path)}


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def mean_metrics(rows: list[dict]) -> dict:
    return {key: helpers.mean(row[key] for row in rows) for key in helpers.METRICS}


def own_losses(evaluation: dict) -> dict:
    """Preserve original recording/rally identities for all partially/lost core."""
    return {scope: [row for row in evaluation["guardrails"][scope]["rallies"]
                    if not row["fullyCovered"]] for scope in SCOPES}


def rescore(original: Path, scope_path: Path) -> tuple[dict, dict, dict]:
    study = original / "nested-study-v1"
    paths = {"registration": study / "preregistration.json", "report": study / "report.json",
             "originalSummary": study / "summary.json", "originalManifest": original / "manifest.json",
             "scopeManifest": scope_path, "script": Path(__file__).resolve(), "summaryHelpers": HELPER_PATH}
    inputs = {name: identify(path) for name, path in paths.items()}
    registration = helpers.load(paths["registration"])
    contract = registration["contract"]
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    ensure(hashlib.sha256(canonical.encode()).hexdigest() == registration["sha256"],
           "Original contract hash mismatch")
    manifest, scope = helpers.load(paths["originalManifest"]), helpers.load(scope_path)
    report, previous_summary = helpers.load(paths["report"]), helpers.load(paths["originalSummary"])
    ensure(inputs["report"]["sha256"] == EXPECTED_REPORT_SHA256 == previous_summary["inputs"]["report"]["sha256"],
           "Original report differs from the previously verified completed study")
    ensure(inputs["scopeManifest"]["sha256"] == EXPECTED_SCOPE_SHA256,
           "Non-beach scope manifest differs from the agreed frozen revision")
    ensure(inputs["originalManifest"]["sha256"] == contract["manifestSha256"] == scope["parentManifest"]["sha256"],
           "Manifest identity differs from original contract or non-beach parent binding")
    ensure(report["contractSha256"] == registration["sha256"], "Report contract identity mismatch")
    ensure(report["status"] == "completed-development-screen" and not report["protectedTestOpened"]
           and not report["productionPromotionAllowed"], "Original study is not an eligible development screen")
    ensure(contract["primaryMetric"] == "F1_padP_coreR" and contract["targetPaddingSeconds"] == 2
           and contract["joinGapSeconds"] == 3 and contract["paddingSweep"] == [0, 1, 2, 3],
           "Unexpected original evaluation contract")
    inputs["frozenModules"] = {}
    for name, expected in contract["code"].items():
        item = identify(REPO / "analysis" / name)
        ensure(item["sha256"] == expected, f"Frozen module changed: {name}")
        inputs["frozenModules"][name] = item

    retained = [row for row in manifest["recordings"] if row["environment"] in ("grass", "indoor")]
    ensure(scope["recordings"] == retained, "Scope must preserve exact original six recording rows and target revisions")
    ids = {row["id"] for row in retained}
    groups = sorted({row["sourceGroup"] for row in retained})
    excluded = [row for row in manifest["recordings"] if row["id"] not in ids]
    ensure(len(ids) == 6 and len(groups) == 3 and sum(len(row["rallies"]) for row in retained) == 235,
           "Expected six non-beach recordings, three source groups, and 235 rallies")
    ensure(all(row["environment"] == "beach" for row in excluded) and len(excluded) == 2,
           "Unexpected excluded recording scope")
    ensure(not set(groups).intersection(manifest["protectedSourceGroups"]), "Protected source group present")
    ensure(all(row["consent"]["train"] and row["consent"]["analyze"] for row in retained), "Consent missing")
    expected_pairs = {(kind, seed) for kind in contract["kinds"] for seed in contract["seeds"]}
    actual_pairs = {(row["kind"], row["seed"]) for row in report["results"]}
    ensure(len(report["results"]) == 12 and actual_pairs == expected_pairs, "Missing/duplicate original result")

    inputs["originalResultFiles"] = []
    evaluations = {}
    originals = {}
    for row in report["results"]:
        key = (row["kind"], row["seed"])
        path = study / f"result-{key[0]}-{key[1]}.json"
        inputs["originalResultFiles"].append(identify(path))
        ensure(helpers.load(path) == row, f"Result file differs from completed report: {key}")
        helpers.assert_result_revision(manifest, row["predictions"], str(key))
        replay = evaluate_predictions(row["predictions"], primary_padding_seconds=2, join_gap_seconds=3)
        ensure(replay == row["evaluation"], f"Original canonical evaluation did not replay exactly: {key}")
        subset = [prediction for prediction in row["predictions"] if prediction["id"] in ids]
        helpers.assert_result_revision(scope, subset, f"non-beach/{key}")
        evaluations[key] = evaluate_predictions(subset, primary_padding_seconds=2, join_gap_seconds=3)
        originals[key] = row

    per_seed, detailed_losses, canonical_evaluations = [], [], []
    for kind in contract["kinds"]:
        for seed in contract["seeds"]:
            evaluation, linear = evaluations[(kind, seed)], evaluations[("linear", seed)]
            helpers.assert_comparable(evaluation, linear)
            loss_comparison = {name: helpers.coverage_comparison(evaluation, linear, name) for name in SCOPES}
            original = originals[(kind, seed)]
            per_seed.append({
                "kind": kind, "seed": seed, **helpers.compact_evaluation(evaluation),
                "sourceGroups": {name: helpers.compact_evaluation(value)
                                 for name, value in evaluation["sourceGroups"].items()},
                "originalEightRecordingPrimary": {key: original["evaluation"]["primary"][key] for key in helpers.METRICS},
                "pairedVsSameSeedLinear": {
                    "primaryDelta": {key: evaluation["primary"][key] - linear["primary"][key] for key in helpers.METRICS},
                    "sourceGroups": helpers.paired_groups(evaluation, linear),
                    "lossCounts": {name: value["counts"] for name, value in loss_comparison.items()},
                },
                "unchangedOriginalSelectionsForRetainedGroups": [choice for choice in original["selections"]
                                                                  if choice["heldSourceGroup"] in groups],
            })
            detailed_losses.append({"kind": kind, "seed": seed, "losses": own_losses(evaluation),
                                    "pairedVsSameSeedLinear": loss_comparison})
            canonical_evaluations.append({"kind": kind, "seed": seed, "evaluation": evaluation})

    kinds = []
    for kind in contract["kinds"]:
        rows = [row for row in per_seed if row["kind"] == kind]
        kinds.append({
            "kind": kind, "seedCount": len(rows),
            "meanSeedPrimary": mean_metrics([row["primary"] for row in rows]),
            "meanSeedPrimaryDeltaVsSameSeedLinear": mean_metrics([row["pairedVsSameSeedLinear"]["primaryDelta"] for row in rows]),
            "meanSeedPadding": [{"paddingSecondsBeforeAndAfter": padding,
                                 **mean_metrics([row["padding"][padding] for row in rows])} for padding in range(4)],
            "meanSeedSourceGroups": {group: mean_metrics([row["sourceGroups"][group]["primary"] for row in rows]) for group in groups},
            "sourceGroupMeanPairedDeltaVsSameSeedLinear": {group: {
                key: helpers.mean(row["pairedVsSameSeedLinear"]["sourceGroups"][group][key] for row in rows)
                for key in ("F1_padP_coreRDelta", "R_coreDelta", "paddedModelExportSecondsDelta")}
                for group in groups},
            "meanSeedEventF1": helpers.mean(row["eventF1"] for row in rows),
            "meanSeedOutcomeStrictRecall": {name: helpers.mean(row["outcomeRecall"][name]["strictMatchRecall"] for row in rows)
                                            for name in helpers.SLICES},
            "meanSeedLossCounts": {name: {key: helpers.mean(row["coverage"][name][key] for row in rows)
                                          for key in helpers.COVERAGE_KEYS} for name in SCOPES},
        })
    # Detect any concurrent mutation of inputs before creating artifacts.
    def verify_unchanged(node):
        if isinstance(node, dict):
            if set(node) == {"path", "sha256"}:
                ensure(helpers.digest(Path(node["path"])) == node["sha256"], f"Input changed during rescore: {node['path']}")
            else:
                for value in node.values():
                    verify_unchanged(value)
        elif isinstance(node, list):
            for value in node:
                verify_unchanged(value)
    verify_unchanged(inputs)
    summary = {
        "schemaVersion": 1, "kind": "retrospective-nonbeach-scope-only-rescore",
        "createdAt": datetime.now(timezone.utc).isoformat(), "status": "completed-diagnostic-rescore",
        "protectedTestOpened": False, "productionPromotionAllowed": False, "refitPerformed": False,
        "primaryMetric": "F1_padP_coreR", "primaryPaddingSeconds": 2, "joinGapSeconds": 3,
        "joinRule": "Join strictly positive gaps less than 3 seconds; exactly 3 remains cut. Subtract ignored intervals after joining; never rejoin across them.",
        "scope": {"recordingCount": len(ids), "sourceGroupCount": len(groups), "sourceGroups": groups,
                  "rallyCount": 235, "recordingIds": sorted(ids), "excludedRecordingIds": [row["id"] for row in excluded],
                  "environments": ["grass", "indoor"], "ignoredIntervalsPresent": any(row.get("ignoredIntervals") for row in retained)},
        "originalContractSha256": registration["sha256"],
        "aggregation": "Each seed pools metric numerators and denominators over six recordings. Kind means average separate seed runs; no per-video F1 averaging and no pooling duplicate videos across seeds.",
        "limitations": [
            "Retrospective scope-only rescore. Original models and inner checkpoint/decoder selection included beach; this is not a beach-free refit.",
            "Retained predictions remain original outer-source-group-held-out predictions. No prediction, checkpoint, decoder, epoch, or seed was changed or selected.",
            "Same-seed linear controls use the same original fitting and selection population, including beach, and the same six-recording evaluation scope.",
            "The exclusion follows a user-directed scope change after the original results were inspected; it is an adaptive development diagnostic, not independent confirmatory evidence.",
            "Only three retained source groups. Seed variation does not estimate uncertainty on new capture sessions.",
            "Uses the exact original label/tag and ignored revision; all ignored lists here are empty. No newer labels or draft ignored spans are substituted.",
            "Full-scope versus non-beach scores describe different evaluation populations and are not treatment effects of removing beach from training.",
            "Loss identities use the original zero-based truthIndex within each recording; coreCoverage precedes product padding, primaryExportCoverage uses fixed 2-second product padding.",
            "No feasibility gate or model promotion is applied to this retrospective diagnostic.",
        ],
        "verification": {"originalResultFilesMatchedReport": 12, "originalEvaluationsReplayedExactly": 12,
                         "rawTargetRevisionsVerified": 12, "retainedTargetsExactlyMatchNewScopeManifest": True,
                         "frozenModuleHashesMatched": len(contract["code"]), "inputHashesRecheckedBeforeWriting": True},
        "inputs": inputs, "kinds": kinds, "perSeed": per_seed,
    }
    return summary, {"scope": summary["scope"], "perSeed": detailed_losses}, {"scope": summary["scope"], "perSeed": canonical_evaluations}


def markdown(summary: dict) -> str:
    lines = ["# Retrospective non-beach scope-only rescore", "",
             "Original models and inner selection included beach. **This is not a beach-free refit.** Fixed original held-out predictions are re-pooled over six grass/indoor recordings, three source groups, and 235 rallies. No model or decoder is selected here.", "",
             "Primary: pooled `F1_padP_coreR` at symmetric 2-second padding; positive export gaps strictly below 3 seconds are joined. Exact original labels and empty ignored lists are retained. Kind means average three separately pooled seed runs; seed variation is not new-session uncertainty.", "",
             "| Kind | Mean F1 | Delta vs same-seed linear | Mean R_core | Delta R_core | Mean export s | Event F1 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary["kinds"]:
        p, d = row["meanSeedPrimary"], row["meanSeedPrimaryDeltaVsSameSeedLinear"]
        lines.append(f"| {row['kind']} | {p['F1_padP_coreR']:.4f} | {d['F1_padP_coreR']:+.4f} | {p['R_core']:.4f} | {d['R_core']:+.4f} | {p['paddedModelExportSeconds']:.1f} | {row['meanSeedEventF1']:.4f} |")
    lines += ["", "## Required padding sensitivity", "",
              "Each cell is a mean over separately pooled seed metrics; padding 2 remains the declared target for every model.", "",
              "| Kind | Pad s | P_pad | R_core | F1_padP_coreR | Model export s | Human export s | Difference s |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary["kinds"]:
        for p in row["meanSeedPadding"]:
            lines.append(f"| {row['kind']} | {p['paddingSecondsBeforeAndAfter']:g} | {p['P_pad']:.4f} | {p['R_core']:.4f} | {p['F1_padP_coreR']:.4f} | {p['paddedModelExportSeconds']:.1f} | {p['paddedHumanExportSeconds']:.1f} | {p['exportDurationDifferenceSeconds']:+.1f} |")
    lines += ["", "## Per-seed recall and loss guardrails", "",
              "Loss counts are complete / partial rally losses. Export loss counts use 2-second padding. Core loss counts precede product padding. Exact identities and paired new/recovered losses are in `loss-identities.json`.", "",
              "| Kind | Seed | F1 | R_core | Core losses | Export losses | New complete export losses vs linear | Short / ace / fault strict recall |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in summary["perSeed"]:
        p, c, e = row["primary"], row["coverage"]["coreCoverage"], row["coverage"]["primaryExportCoverage"]
        short = " / ".join(f"{row['outcomeRecall'][name]['strictMatchRecall']:.3f}" for name in helpers.SLICES)
        new = row["pairedVsSameSeedLinear"]["lossCounts"]["primaryExportCoverage"]["newCompleteLosses"]
        lines.append(f"| {row['kind']} | {row['seed']} | {p['F1_padP_coreR']:.4f} | {p['R_core']:.4f} | {c['completeRallyLosses']} / {c['partialRallyLosses']} | {e['completeRallyLosses']} / {e['partialRallyLosses']} | {new} | {short} |")
    lines += ["", "## Source-group paired deltas", "",
              "Mean of paired candidate-minus-same-seed-linear deltas at padding 2.", "",
              "| Kind | Source group | F1 delta | R_core delta | Export delta s |", "|---|---|---:|---:|---:|"]
    for row in summary["kinds"]:
        for name, p in row["sourceGroupMeanPairedDeltaVsSameSeedLinear"].items():
            lines.append(f"| {row['kind']} | {name} | {p['F1_padP_coreRDelta']:+.4f} | {p['R_coreDelta']:+.4f} | {p['paddedModelExportSecondsDelta']:+.1f} |")
    lines += ["", "## Verification and interpretation", "",
              "All 12 original result files match the hash-bound report; all 12 original evaluations replay exactly; every original target revision matches the frozen manifest; all eight frozen module hashes match. The six retained manifest rows exactly match the new non-beach manifest. Protected test data were not opened.", "",
              "`summary.json` includes every per-seed and per-source-group padding case, boundary and outcome guardrails, paired deltas, preserved original selections, and input hashes. `evaluations.json` preserves complete canonical reports with metric numerators and denominators.", ""]
    lines.extend(f"- {item}" for item in summary["limitations"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=ORIGINAL)
    parser.add_argument("--scope-manifest", type=Path, default=NONBEACH / "manifest.json")
    parser.add_argument("--output", type=Path, default=NONBEACH / "prior-run-rescore")
    args = parser.parse_args()
    ensure(not args.output.exists(), f"Refusing to overwrite output directory: {args.output}")
    summary, losses, evaluations = rescore(args.original, args.scope_manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (("loss-identities.json", losses), ("evaluations.json", evaluations)):
        path = args.output / name
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        summary.setdefault("artifacts", {})[name] = identify(path)
    with (args.output / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    with (args.output / "summary.md").open("x", encoding="utf-8") as handle:
        handle.write(markdown(summary))
    print(json.dumps({"output": str(args.output), "scope": summary["scope"],
                      "kinds": [{"kind": row["kind"], "primary": row["meanSeedPrimary"],
                                 "deltaVsLinear": row["meanSeedPrimaryDeltaVsSameSeedLinear"]}
                                for row in summary["kinds"]]}, indent=2))


if __name__ == "__main__":
    main()
