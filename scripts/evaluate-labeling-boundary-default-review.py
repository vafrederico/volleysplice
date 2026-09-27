#!/usr/bin/env python3
"""Evaluate automatic boundary edits and strictly scoped removal review on PXL.

Uses frozen automatic proposals; human labels enter only evaluation and explicit
perfect-review simulations. Never edits a label, existing study artifact or UI.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
from analysis import neural_production_combinations as iv
from analysis.neural_boundary_metrics import evaluate_boundary_proposals
from analysis.neural_rally_identity_metrics import evaluate_rally_identities


def identity(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "sizeBytes": path.stat().st_size}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def touched(gold, windows):
    return [index for index, row in enumerate(gold) if iv.duration(iv.intersection([row], windows)) > 0]


def workload(record, windows, parents):
    windows = iv.difference(windows, record["ignoredIntervals"])
    playback = iv.export(windows, record, 2)
    return {"reviewSeconds": iv.duration(windows), "reviewRegions": len(windows), "reviewWindows": iv.serial(windows),
            "contextSecondsBeforeAndAfter": 2, "contextJoinGapSeconds": 3,
            "playbackSecondsWithContext": iv.duration(playback), "playbackRegionsWithContext": len(playback),
            "playbackWindowsWithContext": iv.serial(playback),
            "affectedParentIds": [p["id"] for p in parents if iv.intersection([p], windows)],
            "reviewedHumanRallyIndexes": touched(record["rallies"], windows),
            "playbackHumanRallyIndexes": touched(record["rallies"], playback)}


def observed_restored_boundary(time, windows, gold, field):
    return any(w.start < time < w.end for w in windows) and any(g[field] == time for g in gold)


def restore_core_fragments(record, events, restore, windows, *, boundary_mode="gold-inside-queue"):
    """Attach reviewed positive fragments by geometric contact, never gold ID.

    Automatic identities remain separate unless a restored positive fragment
    physically connects them. Isolated fragments remain permission-limited new
    events. No endpoint outside review windows is corrected from gold.
    """
    output, changes = [], []
    for parent in record["productionEvents"]:
        children = [copy.deepcopy(event) for event in events if event.get("parentId", event["id"]) == parent["id"]]
        positive = iv.intersection(restore, [parent])
        nodes = [{"event": event, "restored": False} for event in children]
        for index, fragment in enumerate(positive):
            if boundary_mode == "known-parent-only":
                start_observed = fragment.start == parent["start"] and parent.get("startObserved", True)
                end_observed = fragment.end == parent["end"] and parent.get("endObserved", True)
                restored_id = parent["id"] if fragment.start == parent["start"] and fragment.end == parent["end"] else f"{parent['id']}::restored-component:{index}"
                inherited = copy.deepcopy(parent)
            else:
                assert boundary_mode == "gold-inside-queue"
                start_observed = observed_restored_boundary(fragment.start, windows, record["rallies"], "start")
                end_observed = observed_restored_boundary(fragment.end, windows, record["rallies"], "end")
                restored_id, inherited = f"reviewed-fragment:{parent['id']}:{index}", {}
            nodes.append({"restored": True, "event": {**inherited, "id": restored_id,
                "parentId": parent["id"], "start": fragment.start, "end": fragment.end,
                "startObserved": start_observed, "endObserved": end_observed,
                "startSource": "known-parent-undo" if boundary_mode == "known-parent-only" and start_observed else "reviewed-core-or-permission-clip",
                "endSource": "known-parent-undo" if boundary_mode == "known-parent-only" and end_observed else "reviewed-core-or-permission-clip"}})
        groups = list(range(len(nodes)))
        def find(i):
            while groups[i] != i:
                groups[i] = groups[groups[i]]
                i = groups[i]
            return i
        for i, left in enumerate(nodes):
            for j in range(i):
                right = nodes[j]
                if not left["restored"] and not right["restored"]:
                    continue
                a, b = left["event"], right["event"]
                if a["start"] <= b["end"] and b["start"] <= a["end"]:
                    groups[find(i)] = find(j)
        clusters = {}
        for index, node in enumerate(nodes):
            clusters.setdefault(find(index), []).append(node)
        for cluster in clusters.values():
            automatic = [n["event"] for n in cluster if not n["restored"]]
            restored = [n["event"] for n in cluster if n["restored"]]
            if not restored:
                assert len(automatic) == 1
                output.extend(automatic)
                continue
            all_events = [n["event"] for n in cluster]
            first = min(all_events, key=lambda x: (x["start"], x not in automatic))
            last = max(all_events, key=lambda x: (x["end"], x in automatic))
            new = copy.deepcopy(automatic[0] if len(automatic) == 1 else first)
            if len(automatic) > 1:
                new["id"] = "review-joined:" + "+".join(x["id"] for x in automatic)
            new.update(start=first["start"], end=last["end"], parentId=parent["id"],
                       startObserved=first.get("startObserved", True), endObserved=last.get("endObserved", True),
                       startSource=first.get("startSource", "automatic"), endSource=last.get("endSource", "automatic"))
            output.append(new)
            changes.append({"parentId": parent["id"], "automaticIds": [x["id"] for x in automatic],
                            "restoredFragments": restored, "resultEvent": new,
                            "operation": "merge-through-restored-play" if len(automatic) > 1 else "extend-existing" if automatic else "isolated-reviewed-fragment"})
    expected = iv.union(events, restore)
    assert iv.duration(iv.difference(output, expected)) < 1e-8 and iv.duration(iv.difference(expected, output)) < 1e-8
    return sorted(output, key=lambda e: (e["start"], e["end"], e["id"])), changes


def evaluate(record, predictions, export_overrides=None):
    row = {**record, "predictions": predictions}
    return {"duration": iv.duration_rows([row], export_overrides), "identity": evaluate_rally_identities([row])}


def self_check():
    record = {"productionEvents": [{"id": "p", "start": 0, "end": 10}], "rallies": [{"start": 0, "end": 10}]}
    events = [{"id": "a", "parentId": "p", "start": 1, "end": 4}, {"id": "b", "parentId": "p", "start": 6, "end": 9}]
    restored, changes = restore_core_fragments(record, events, iv.intervals([(4, 6)]), iv.intervals([(4, 6)]))
    assert len(restored) == 1 and restored[0]["start"] == 1 and restored[0]["end"] == 9
    assert changes[0]["operation"] == "merge-through-restored-play"
    restored, changes = restore_core_fragments(record, events, iv.intervals([(0, .5)]), iv.intervals([(0, 1)]))
    assert len(restored) == 3 and not restored[0]["startObserved"]
    touching = [{"id": "a", "parentId": "p", "start": 1, "end": 4}, {"id": "b", "parentId": "p", "start": 4, "end": 9}]
    restored, changes = restore_core_fragments(record, touching, iv.intervals([(0, 1)]), iv.intervals([(0, 1)]))
    assert len(restored) == 2, "Restoration must not collapse untouched touching identities"
    no_gold = {"productionEvents": record["productionEvents"]}
    restored, _ = restore_core_fragments(no_gold, [], iv.intervals([(0, 10)]), iv.intervals([(0, 10)]), boundary_mode="known-parent-only")
    assert len(restored) == 1 and restored[0]["id"] == "p" and restored[0]["startObserved"] and restored[0]["endObserved"]


def run(args):
    self_check()
    root = args.study_root
    destination = root / "boundary-default-review-v1"
    destination.mkdir(exist_ok=True)
    result_path, summary_path = destination / "results-v2.json", destination / "summary-v2.json"
    if result_path.exists() or summary_path.exists():
        raise FileExistsError("Refusing to overwrite a registered revision")
    boundary_path = root / "boundary-adviser.json"
    label_path = root / private_value('private-reference-0097')
    draft_path = Path(private_value('private-reference-0098'))
    labels, draft, adviser = read(label_path), read(draft_path), read(boundary_path)
    assert labels["rallies"] == draft["rallies"] and labels["ignoredIntervals"] == draft["ignoredIntervals"], "Human draft changed since import"
    audit = read(root / "inference-audit-v1.json")
    assert identity(boundary_path)["sha256"] == audit["artifacts"]["boundary-adviser.json"]["sha256"]
    record = {**adviser["record"], "rallies": labels["rallies"], "ignoredIntervals": labels["ignoredIntervals"]}
    parents, automatic = record["productionEvents"], adviser["plan"]["events"]
    primary = read(root / "production-replay.json")["variants"]["aggressive"]["core"]
    assert [(p["start"], p["end"]) for p in parents] == [(p["start"], p["end"]) for p in primary]
    valid = iv.difference([(0, record["durationSeconds"])], record["ignoredIntervals"])
    p_core, a_core, human = (iv.intersection(rows, valid) for rows in (parents, automatic, record["rallies"]))
    assert not iv.difference(a_core, p_core), "This experiment assumes proposal cores remain inside their production parents"
    removed_core = iv.difference(p_core, a_core)
    restore_core = iv.intersection(human, removed_core)
    unsafe = [p["id"] for p in parents if iv.intersection([p], restore_core)]
    vetoed = [copy.deepcopy(p) for p in parents if p["id"] in unsafe]
    vetoed += [copy.deepcopy(e) for e in automatic if e.get("parentId", e["id"]) not in unsafe]
    vetoed.sort(key=lambda e: (e["start"], e["end"], e["id"]))
    fragment_events, fragment_changes = restore_core_fragments(record, automatic, restore_core, removed_core)
    restore_whole_fragments = tuple(region for region in removed_core if iv.intersection([region], human))
    granular_events, granular_changes = restore_core_fragments(record, automatic, restore_whole_fragments, removed_core,
                                                               boundary_mode="known-parent-only")
    core_work = workload(record, removed_core, parents)
    export_overrides, export_work = {record["id"]: {}}, []
    for pad in iv.PADS:
        pe, ae, he = (iv.export(rows, record, pad) for rows in (parents, automatic, record["rallies"]))
        assert not iv.difference(ae, pe)
        removed, restored = iv.difference(pe, ae), iv.intersection(he, iv.difference(pe, ae))
        result = iv.union(ae, restored)  # No re-padding/rejoining after exact review.
        export_overrides[record["id"]][str(pad)] = iv.serial(result)
        linked_parents = []
        for parent in parents:
            children = [e for e in automatic if e.get("parentId", e["id"]) == parent["id"]]
            parent_removed = iv.difference(iv.export([parent], record, pad), iv.export(children, record, pad))
            if iv.intersection(parent_removed, removed):
                linked_parents.append(parent["id"])
        export_work.append({"paddingSecondsBeforeAndAfter": pad, "joinGapSeconds": 3,
            **workload(record, removed, parents), "automaticRemovedExportSeconds": iv.duration(removed),
            "affectedParentIds": linked_parents,
            "parentAssociationRule": "A parent's own padded export loss intersects the actual global removal queue.",
            "humanCoreLostByAutomaticRemovalSeconds": iv.duration(iv.intersection(human, removed)),
            "wantedHumanExportLostByAutomaticRemovalSeconds": iv.duration(restored),
            "restoredExportRegions": len(restored), "restoredExportSeconds": iv.duration(restored),
            "restoredCoreSeconds": iv.duration(iv.intersection(restored, human)), "restoredExportWindows": iv.serial(restored),
            "wantedExportAffectedHumanRallyIndexes": [i for i, h in enumerate(record["rallies"])
                if iv.intersection(iv.export([h], record, pad), removed)],
            "unreviewedIncorrectExportSeconds": iv.duration(iv.difference(result, he)),
            "boundaryFlagsWithoutRemovedExportAtParent": [f["id"] for f in adviser["plan"]["proposals"]
                if f["parentId"] not in linked_parents]})
    scenarios = {
        "production": {"events": parents, **evaluate(record, parents)},
        "automatic_all_boundaries": {"events": automatic, **evaluate(record, automatic)},
        "removed_core_review_parent_veto": {"events": vetoed, **evaluate(record, vetoed)},
        "removed_core_review_granular_veto": {"events": granular_events, **evaluate(record, granular_events)},
        "removed_core_review_fragment_restoration": {"events": fragment_events, **evaluate(record, fragment_events)},
        "removed_export_review_keep_mask": {"events": automatic, "exportOverrides": export_overrides,
                                             **evaluate(record, automatic, export_overrides)},
    }
    # Perfect review must not invent coverage beyond production or lose reviewed play.
    for name in ("removed_core_review_parent_veto", "removed_core_review_granular_veto", "removed_core_review_fragment_restoration"):
        result_core = iv.intersection(scenarios[name]["events"], valid)
        assert iv.duration(iv.intersection(human, iv.difference(p_core, result_core))) < 1e-8
        assert not iv.difference(result_core, p_core)
    proposal_metrics = evaluate_boundary_proposals([{**record, "predictions": automatic,
                                                    "boundaryProposals": adviser["plan"]["eventCandidates"]}])
    split_diagnostics = []
    for parent in parents:
        children = [e for e in automatic if e.get("parentId", e["id"]) == parent["id"]]
        if len(children) <= 1:
            continue
        gold_indexes = touched(record["rallies"], [parent])
        gaps = []
        for left, right in zip(children, children[1:]):
            if left["end"] >= right["start"]:
                continue
            gap = [(left["end"], right["start"])]
            gaps.append({"start": left["end"], "end": right["start"],
                "humanCoreSeconds": iv.duration(iv.intersection(gap, human)),
                "granularVetoRestoredSeconds": iv.duration(iv.intersection(gap, restore_whole_fragments)),
                "exportQueueOverlapSecondsByPadding": {str(pad): iv.duration(iv.intersection(gap, export_work[pad]["reviewWindows"])) for pad in iv.PADS},
                "exportPlaybackOverlapSecondsByPadding": {str(pad): iv.duration(iv.intersection(gap, export_work[pad]["playbackWindowsWithContext"])) for pad in iv.PADS}})
        split_diagnostics.append({"parentId": parent["id"], "children": [{k: e[k] for k in ("id", "start", "end", "type")} for e in children],
            "internalSplitGaps": gaps,
            "overlappingHumanRallyIndexes": gold_indexes, "overlappingHumanRallyCount": len(gold_indexes),
            "removalCoreRegions": iv.serial(iv.intersection(removed_core, [parent])),
            "removedHumanCoreSeconds": iv.duration(iv.intersection(restore_core, [parent])),
            "parentVetoTriggered": parent["id"] in unsafe,
            "exportReviewRemovalSecondsByPadding": {str(pad): iv.duration(iv.intersection(
                iv.difference(iv.export(parents, record, pad), iv.export(automatic, record, pad)), [parent])) for pad in iv.PADS}})
    result = {"schemaVersion": 1, "createdAt": datetime.now(timezone.utc).isoformat(), "recordingId": record["id"],
        "targetPaddingSecondsBeforeAndAfter": 2, "paddingCases": list(iv.PADS), "joinGapSeconds": 3,
        "sourceGroup": record["sourceGroup"], "durationSeconds": record["durationSeconds"],
        "ignoredIntervals": record["ignoredIntervals"],
        "scope": "Single challenge recording, existing manually reviewed export-coverage labels. Descriptive; neither threshold selection nor independent frame-exact serve/dead-ball validation.",
        "revision": 2, "supersedes": "results.json (v1); adds practical granular veto and fixes attribution of padded removal windows to parents.",
        "definitions": {
            "automatic": "Apply all frozen head_refined event boundaries. Materialize exports from those changed cores; the current UI instead preserves production export.",
            "parentVeto": "Review only removed core P\\A. If any wanted human core is inside a parent's removed portion, reject that entire parent edit and restore the known original parent. Other parent edits remain automatic. No new gold boundary is read outside the queue.",
            "granularVeto": "Review each connected P_core\\A_core region. Restore that WHOLE region iff it contains any human core; otherwise leave it removed. Use only existing geometric endpoints, never gold-derived timing. Extend candidates touching restored prefixes/suffixes; merge candidates only when a restored region physically links them. Preserve unrelated automatic identities.",
            "fragmentRestoration": "Optimistic fine-edit reference: restore only H_core intersect (P_core\\A_core). Attach to existing candidates by contact; merge only if restored play connects candidates; isolated reviewed fragments become permission-clipped events. Unreviewed false positives persist.",
            "exportMaskRestoration": "Optimistic export-only reference per padding: A_export union [H_padded intersect (P_export\\A_export)]. No repadding/rejoining follows. Automatic event identities remain unchanged; export keeps are not rally labels.",
            "playback": "Exact removed portions plus separate +/-2 s context workload with strict <3 s joining. Context does not expand label-edit permissions.",
            "humanAssumption": "Perfect recognition of wanted play only within queued portions. Measured human accuracy or review completion time is not established.",
        },
        "inputs": [identity(p) for p in (boundary_path, label_path, draft_path, root / "production-replay.json", root / "inference-audit-v1.json")],
        "sources": [identity(REPO / p) for p in ("scripts/evaluate-labeling-boundary-default-review.py", "analysis/neural_rally_identity_metrics.py", "analysis/neural_boundary_metrics.py", "analysis/neural_production_combinations.py", "analysis/crop_evaluation.py")],
        "coreReview": {**core_work, "automaticRemovedCoreSeconds": iv.duration(removed_core),
            "removedHumanCoreSeconds": iv.duration(restore_core), "restoredCoreRegions": len(restore_core), "restoredCoreWindows": iv.serial(restore_core),
            "revertedParentIds": unsafe, "revertedParents": len(unsafe),
            "remainingAutomaticEditedParentIds": [p["id"] for p in parents if p["id"] not in unsafe and iv.intersection([p], removed_core)],
            "fragmentChanges": fragment_changes},
        "granularVeto": {"restoredRegions": len(restore_whole_fragments), "acceptedRemovalRegions": len(removed_core)-len(restore_whole_fragments),
            "restoredSeconds": iv.duration(restore_whole_fragments), "restoredHumanCoreSeconds": iv.duration(iv.intersection(restore_whole_fragments, human)),
            "extraNonHumanCoreRestoredSeconds": iv.duration(iv.difference(restore_whole_fragments, human)),
            "restoredWindows": iv.serial(restore_whole_fragments), "changes": granular_changes,
            "noGoldDerivedEndpointCoordinates": True},
        "exportReviewByPadding": export_work, "automaticSplitDiagnostics": split_diagnostics,
        "boundaryProposalMetrics": proposal_metrics, "scenarios": scenarios,
        "qualificationChecks": {"fourFragmentRestorationSyntheticCasesPassed": True, "granularGeometryHelperDoesNotReadGold": True, "humanDraftEqualsImportedCoverage": True,
            "frozenAutomaticPlanHashVerified": True, "productionParentReplayExact": True,
            "allReviewRestorationConfinedToProductionCoverage": True, "parentVetoAndCoreRestorationPreventAddedCoreLoss": True},
    }
    write(result_path, result)
    report = []
    for name, scenario in scenarios.items():
        duration = next(row for row in scenario["duration"] if row["paddingSecondsBeforeAndAfter"] == 2)
        events = scenario["identity"]["pooled"]
        report.append({"name": name, **{k: duration[k] for k in ("P_pad", "R_core", "F1_padP_coreR", "paddedModelExportSeconds", "incorrectExportSeconds", "incorrectlyRemovedSeconds", "missedCoreSeconds")},
            **{k: events[k] for k in ("predictedRallies", "matchedRallies", "eventPrecision", "eventRecall", "eventF1", "completeMisses", "splitTrueRalliesMaterial", "mergedPredictionsMaterial")}})
    brief = {"targetPadding": 2, "rows": report, "coreReview": {k: v for k, v in result["coreReview"].items() if k not in ("fragmentChanges", "reviewWindows", "playbackWindowsWithContext", "restoredCoreWindows")},
             "granularVeto": {k: v for k, v in result["granularVeto"].items() if k not in ("changes", "restoredWindows")},
             "exportReviewTarget": export_work[2], "splitDiagnostics": split_diagnostics, "results": identity(result_path)}
    write(summary_path, brief)
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    run(parser.parse_args())
