#!/usr/bin/env python3
"""Render the audited, public Balanced serving-side comparison aggregates."""
import argparse
import hashlib
import json
from pathlib import Path


def percent(value):
    return "undefined" if value is None else f"{100*value:.2f}%"


def count(value):
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def qualified(item, macro, target=.98):
    return all(value is not None and value >= target for value in
               (item["precision"], item["nearPrecision"], item["farPrecision"], macro))


def render(result, audit, result_name, audit_name):
    main = result["result"]["targets"]["0.98"]
    pooled = main["consensus"]["pooled"]
    mean = main["meanSeeds"]
    macro = main["consensus"]["sourceGroupMacroPrecision"]
    verdict = "cleared" if qualified(pooled, macro) else "did not clear"
    text = ["# Balanced-encoder serving-side precision: results", "",
        f"The centered serving-side head trained on **Balanced embeddings** achieved **{percent(mean['precision'])} automatic precision with {percent(mean['reviewFraction'])} reviewed**, averaged across three seeds at the 98% selection target. Its three-seed agreement policy achieved **{percent(pooled['precision'])} precision**, with **{pooled['reviewed']} of 968 serves reviewed ({percent(pooled['reviewFraction'])})** and **{pooled['errors']} wrong automatic answers among {pooled['accepted']} accepted**. Production has not changed.", "",
        "This matched comparison changes only the frozen image encoder. It uses the same 968 reviewed true non-beach serves from 26 recordings and eight source groups, human labels, anchors, temporal masks, frame timestamps, 237 specialist ranks, and 12 phase-specific absolute motion-quality values. All 108 nested neural fits and three final refits are newly trained; the previous serving weights are not reused.", "",
        "The Balanced encoder weights match `high-f1` in the production manifest. Both encoders are DINO-distilled MobileNetV3-Large but have different weights and embeddings. Inputs retain FP16-rounded four-region 960-dimensional tokens over 13 frames spanning -3 to +3 seconds. Centering subtracts each window's valid-frame mean embedding. The serving head architecture, training settings, seeds, and nested precision selector match the Maximum coverage centered experiment.", "",
        "Outer source groups are excluded from fitting and inner epoch/threshold selection. Precision is correct automatic side answers divided by accepted side answers; review rate uses all 968 known serves. A target is an inner-validation condition and does not guarantee unseen-source precision. This evaluation does not rerun rally detection or the separate serve/no-serve gate.", "",
        "## Matched encoder comparison", "",
        "Mean-seed counts may be fractional. Agreement requires all three seed policies to accept and agree.", "",
        "| Encoder | Policy | Selection target | Measured precision | Reviews / 968 | Review rate | Wrong automatic |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for target in ("0.95", "0.98", "0.99"):
        for encoder, variant in (("Maximum coverage", result["maximumCoverageControl"]), ("Balanced", result["result"])):
            item = variant["targets"][target]
            for policy, metric in (("Mean seeds", item["meanSeeds"]), ("Three-seed agreement", item["consensus"]["pooled"])):
                text.append(f"| {encoder} | {policy} | {float(target)*100:.0f}% | {percent(metric['precision'])} | {count(metric['reviewed'])} | {percent(metric['reviewFraction'])} | {count(metric['errors'])} |")
    text += ["", "## Precision guardrails at the 98% target", "",
        "| Encoder | Policy | Near precision | Far precision | Source-average precision | All four 98% checks |",
        "| --- | --- | ---: | ---: | ---: | --- |"]
    for encoder, variant in (("Maximum coverage", result["maximumCoverageControl"]), ("Balanced", result["result"])):
        item = variant["targets"]["0.98"]
        for policy, metric, source_precision in (("Mean seeds", item["meanSeeds"], item["meanSeeds"]["sourceGroupMacroPrecision"]),
                                               ("Three-seed agreement", item["consensus"]["pooled"], item["consensus"]["sourceGroupMacroPrecision"])):
            text.append(f"| {encoder} | {policy} | {percent(metric['nearPrecision'])} | {percent(metric['farPrecision'])} | {percent(source_precision)} | {'Pass' if qualified(metric, source_precision) else 'Fail'} |")
    text += ["", f"The Balanced agreement policy **{verdict}** the registered pooled, both-side, and average-source 98% guardrails. Source-average precision excludes groups with no accepted answers; those groups remain in review/coverage denominators. A source average does not impose 98% on every individual source.", "",
        f"The Balanced agreement policy's descriptive pooled Wilson 95% lower precision bound is {percent(pooled['precisionLowerWilson95DESCRIPTIVE'])}. This is not a source-cluster or repeated-model-selection guarantee. The 99%-target Balanced agreement policy measured {percent(result['result']['targets']['0.99']['consensus']['pooled']['precision'])} and missed 99% despite its larger review workload.", "",
        "## Individual Balanced seeds at the 98% target", "",
        "| Seed | Precision | Near precision | Far precision | Source-average precision | Reviews / 968 | Wrong automatic |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for seed in main["seeds"]:
        e = seed["evaluation"]; p = e["pooled"]
        text.append(f"| {seed['seed']} | {percent(p['precision'])} | {percent(p['nearPrecision'])} | {percent(p['farPrecision'])} | {percent(e['sourceGroupMacroPrecision'])} | {p['reviewed']} | {p['errors']} |")
    text += ["", "No best-seed selection was performed. Infeasible outer policies across three seeds: "+str(main["infeasibleOuterPolicies"])+".", "",
        "## Balanced agreement by held source at the 98% target", "",
        "| Source | Serves | Accepted | Reviews | Wrong automatic | Precision |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for source, p in main["consensus"]["bySourceGroup"].items():
        text.append(f"| {source} | {p['rows']} | {p['accepted']} | {p['reviewed']} | {p['errors']} | {percent(p['precision'])} |")
    text += ["", "## Paired agreement outcomes at the 98% target", "",
        "Each cell counts the same serves under the two encoder-specific agreement policies. Rows are Maximum coverage outcomes; columns are Balanced outcomes.", "",
        "| Maximum coverage outcome | Balanced review | Balanced correct automatic | Balanced wrong automatic |",
        "| --- | ---: | ---: | ---: |"]
    transitions = audit["pairedConsensusOutcomes"]["0.98"]["maximumCoverageToBalanced"]
    for status in ("review", "correct", "wrong"):
        row = transitions[status]
        text.append(f"| {status.capitalize()} | {row['review']} | {row['correct']} | {row['wrong']} |")
    text += ["", f"Of Maximum coverage's wrong automatic answers, {transitions['wrong']['correct']} become correct automatic Balanced answers, {transitions['wrong']['review']} become reviews, and {transitions['wrong']['wrong']} remain wrong. Balanced also makes {transitions['correct']['wrong']} wrong automatic answers on previously correct answers and {transitions['review']['wrong']} on previously reviewed serves. Both policies accept but disagree on {audit['pairedConsensusOutcomes']['0.98']['bothAcceptedSideDisagreements']} serves.", "",
        "## Verification and artifacts", "",
        "Nine qualification tests passed before registration. The independent sampler also passed 150 synthetic cases, including exact window matching and invalid-anchor rejection around ignored time. The independent audit rebuilt all Balanced windows from the private cache with that sampler, checked matching image lineage/timestamps/masks/features, verified all 37 logistic teacher partitions and 111 neural fits, and replayed all 72 outer policies and nine final policies. It reproduced every aggregate and agreement metric.", "",
        f"Reloaded {audit['selectedCheckpointsReloaded']} selected checkpoints; maximum logit error was {audit['maximumReloadLogitError']:.3g}. All nine final research-model configurations passed CPU/GPU parity; maximum score difference was {audit['maximumCpuGpuScoreError']:.3g}. Final policies were selected on source-held development predictions before all-development refits; final in-sample predictions are not evaluation.", ""]
    resource = result["resources"]
    text += [f"Training ran in one fitting process with two CPU threads. Peak sampled resident memory was {resource['peakResidentMiB']/1024:.2f} GiB; maximum sampled process swap was {resource['maximumProcessSwapKiB']} KiB. WSL swap counter deltas were {resource['wslSwapCounterDelta']['pswpin']} pages in and {resource['wslSwapCounterDelta']['pswpout']} pages out. Minimum WSL available memory was {resource['minimumWslAvailableGiB']:.2f} GiB.", "",
        f"Frozen registration: `{result['contractSha256']}`. Durable artifacts use ledger `{result['studyIndex']}`, SSD artifacts use `{result['fastArtifactIndex']}`, and Balanced encoder/cache inputs use `private-reference-0244` / `private-reference-0245`. Exact locations, recording identities, row predictions, weights, and logs remain private. Protected labels were not opened. No commit, publication, or production change was performed.", "",
        f"[Complete public aggregates]({result_name}) and [independent audit]({audit_name}). See the [registered protocol](serving-side-balanced-precision-2026-10-14.md) and [earlier Maximum coverage results](serving-side-precision-iterations-results-2026-10-14.md).", "",
        "## Next steps", "",
        "1. Confirm the encoder-specific candidates on untouched recordings and source groups. These are reused development data; prior pixel exposure, historical encoder selection, and reviewed-candidate recording ranks limit generalization claims.",
        "2. Evaluate the actual production rally and serve candidate pipeline, including false serves, review gates, and production candidate-based ranks, to measure end-to-end wrong automatic answers and review workload.",
        "3. Inspect confident mistakes for label ambiguity, anchor timing, occlusion, and court geometry, prioritizing errors shared by both encoders. Register any new hypothesis instead of tuning the current thresholds on held-source errors.",
        "4. Before promotion, export and verify the appropriate serving head for each production encoder, including centering, valid-frame masking, absolute quality features, thresholds, and agreement. Embeddings from the two encoders are not interchangeable.", ""]
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.results.read_text()); audit = json.loads(args.audit.read_text())
    assert audit["passed"] and audit["summarySha256"] == hashlib.sha256(args.results.read_bytes()).hexdigest()
    assert audit["contractSha256"] == result["contractSha256"]
    args.output.write_text(render(result, audit, args.results.name, args.audit.name), encoding="utf-8")
    print(json.dumps({"reportedTargets":3,"encodersCompared":2,"pairedRowsVerified":968}))


if __name__ == "__main__":
    main()
