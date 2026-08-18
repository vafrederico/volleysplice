#!/usr/bin/env python3
"""Prepare UI data for suppression limited to one-model-only production support."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TRAIN_SCRIPT = ROOT / "scripts" / "train-feedback-suppression-v3.py"
DEFAULT_OUTPUT = ROOT / "data" / "single-model-suppression-review.json"
ANY_OVERLAP_OUTPUT = ROOT / "data" / "any-overlap-suppression-review.json"

POLICIES = {
    "pointwise": {
        "label": "Pointwise overlap",
        "strategy": "Suppress only production-positive time supported by exactly one production component at that instant; preserve only the time where both components overlap.",
    },
    "any-overlap": {
        "label": "Any-overlap protects padded/joined span",
        "strategy": "Build the final production export components after 2-second padding and joins below 3 seconds. If a component contains any previous-production and all-labels-v2 raw predictions, protect every production interval in that component, including non-overlapping heads, tails, and intervals connected only through padding or joining; suppress only export components supported by one model.",
    },
    "any-overlap-raw": {
        "label": "Legacy any-overlap protects raw-connected span",
        "strategy": "Build overlap-connected components from raw previous-production and all-labels-v2 intervals before padding. Protect a component when both models occur in that raw component, including its non-overlapping heads and tails; padding and joining do not extend cross-model protection to another raw component.",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=tuple(POLICIES), default="pointwise")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        help="Experiment directory containing split-policy.json and inference/.",
    )
    parser.add_argument(
        "--suppression-model",
        type=Path,
        help="Suppression model directory; defaults to the original specialist.",
    )
    parser.add_argument(
        "--model-variant-id",
        default="original",
        help="Stable identifier exposed in the review dataset.",
    )
    parser.add_argument(
        "--model-variant-label",
        default="Original suppression specialist",
        help="Human-readable specialist label exposed in the review UI.",
    )
    parser.add_argument(
        "--agreement-padding",
        type=float,
        default=2.0,
        help="Padding used only to group raw predictions for any-overlap protection.",
    )
    parser.add_argument(
        "--agreement-join-gap",
        type=float,
        default=3.0,
        help="Strict join threshold used only to group any-overlap protection.",
    )
    parser.add_argument(
        "--minimum-per-model-support",
        type=float,
        default=0.0,
        help="Minimum raw seconds from each model required to protect a component.",
    )
    args = parser.parse_args()
    output = args.output or (
        ANY_OVERLAP_OUTPUT if args.policy == "any-overlap" else DEFAULT_OUTPUT
    )
    ns = runpy.run_path(str(TRAIN_SCRIPT))
    experiment_root = (args.experiment_root or ns["DEFAULT_OUTPUT"]).resolve()
    interval = ns["Interval"]
    merge = ns["_merge_intervals"]
    subtract = ns["subtract_intervals"]
    total_duration = ns["_duration"]
    intersection_duration = ns["_intersection_duration"]
    pad_and_merge = ns["pad_and_merge_intervals"]
    item_type = ns["Item"]

    def intersect(left: Iterable[Any], right: Iterable[Any]) -> tuple[Any, ...]:
        first = list(merge(left))
        second = list(merge(right))
        result = []
        first_index = second_index = 0
        while first_index < len(first) and second_index < len(second):
            start = max(first[first_index].start, second[second_index].start)
            end = min(first[first_index].end, second[second_index].end)
            if end > start:
                result.append(interval(start, end))
            if first[first_index].end <= second[second_index].end:
                first_index += 1
            else:
                second_index += 1
        return merge(result)

    def one_model_support_for(
        old: Iterable[Any], v2: Iterable[Any], duration: float
    ) -> tuple[Any, ...]:
        if args.policy == "pointwise":
            return ns["union_intervals"](
                subtract(old, v2), subtract(v2, old)
            )

        old_rows = tuple(old)
        v2_rows = tuple(v2)
        if args.policy == "any-overlap-raw":
            tagged = [
                (float(row.start), float(row.end), "old")
                for row in old_rows
            ]
            tagged.extend(
                (float(row.start), float(row.end), "v2")
                for row in v2_rows
            )
            tagged.sort(key=lambda row: (row[0], row[1], row[2]))
            components: list[tuple[float, float, set[str]]] = []
            for start, end, source in tagged:
                if not components or start >= components[-1][1]:
                    components.append((start, end, {source}))
                    continue
                component_start, component_end, sources = components[-1]
                sources.add(source)
                components[-1] = (
                    component_start,
                    max(component_end, end),
                    sources,
                )
            return merge(
                interval(start, end)
                for start, end, sources in components
                if len(sources) == 1
            )

        raw_union = ns["union_intervals"](old_rows, v2_rows)
        export_components = pad_and_merge(
            raw_union,
            duration,
            args.agreement_padding,
            args.agreement_join_gap,
        )
        eligible_components = tuple(
            component
            for component in export_components
            if not (
                intersection_duration((component,), old_rows)
                >= max(args.minimum_per_model_support, 1e-9)
                and intersection_duration((component,), v2_rows)
                >= max(args.minimum_per_model_support, 1e-9)
            )
        )
        return intersect(raw_union, eligible_components)

    def rows(intervals: Iterable[Any]) -> list[dict[str, float]]:
        return [{"start": item.start, "end": item.end} for item in intervals]

    def scored_rows(
        intervals: Iterable[Any],
        scored: Iterable[Any],
        prepared: Any,
    ) -> list[dict[str, float]]:
        """Split target ranges by decoded predictions while retaining their score."""
        targets = subtract(intervals, prepared.recording.ignored_intervals)
        result = []
        for target in targets:
            for prediction in scored:
                start = max(target.start, float(prediction.start))
                end = min(target.end, float(prediction.end))
                if end <= start:
                    continue
                result.append({
                    "start": start,
                    "end": end,
                    "confidence": max(
                        0.0, min(1.0, float(prediction.confidence))
                    ),
                })
        return result

    def read_ranges(path: Path) -> tuple[Any, ...]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return tuple(
            interval(float(item["start"]), float(item["end"]))
            for item in payload["ranges"]
        )

    def track(
        raw: Iterable[Any],
        prepared: Any,
        scored: Iterable[Any] | None = None,
    ) -> dict[str, Any]:
        raw_union = subtract(raw, prepared.recording.ignored_intervals)
        padded_without_join = subtract(
            pad_and_merge(raw, prepared.sequence.metadata.duration, 2.0, 0.0),
            prepared.recording.ignored_intervals,
        )
        padded = subtract(
            pad_and_merge(raw, prepared.sequence.metadata.duration, 2.0, 3.0),
            prepared.recording.ignored_intervals,
        )
        return {
            "raw": scored_rows(raw_union, scored, prepared) if scored else rows(raw_union),
            "padding": rows(subtract(padded_without_join, raw_union)),
            "joinedGaps": rows(subtract(padded, padded_without_join)),
            "padded": rows(padded),
        }

    def predict_scored(item: Any, heads: Any) -> tuple[Any, ...]:
        serve_decoder = ns["ServeDecoderConfig"].from_dict(
            heads.serve.training_summary["serveDecoder"]
        )
        composition = ns["ServeCompositionConfig"].from_dict(
            heads.serve.training_summary["composition"]
        )
        dead_decoder = ns["DeadStateDecoderConfig"].from_dict(
            heads.dead.training_summary["selectedDeadStateDecoder"]
        )
        refinement = ns["DeadStateRefinementConfig"].from_dict(
            heads.dead.training_summary["selectedRefinement"]
        )
        inputs = ns["_prediction_inputs"](
            item,
            heads.rally,
            heads.serve,
            heads.dead,
            serve_decoder,
            composition,
        )
        prediction = ns["_predictions_for"](
            [inputs], dead_decoder, refinement
        )[0]
        return tuple(prediction.candidate)

    old_heads, v2_heads = ns["load_heads"]()
    suppression_model_path = args.suppression_model or (
        experiment_root
        / "models"
        / "v3-candidate2-four-head"
        / "suppression"
    )
    suppression_model = ns["load_model"](suppression_model_path)
    suppression_config = ns["DecoderConfig"].from_dict(
        suppression_model.training_summary["productionEnsembleDecoderSelection"][
            "selectedConfig"
        ]
    )
    training_ids = {
        recording.id
        for recording in ns["load_manifest"](ns["ALL_LABELS_MANIFEST"]).recordings
    }
    split = json.loads(
        (experiment_root / "split-policy.json").read_text(encoding="utf-8")
    )
    feedback_fit = set(split["fitProjectIds"])
    feedback_imports = {}
    for path in sorted(ns["FEEDBACK_ROOT"].glob("*/import.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        feedback_imports[value["projectId"]] = value["id"]

    entries = []
    for recording in ns["load_labeled_records"]().values():
        prepared = ns["prepare_recording"](
            recording, v2_heads.rally.feature_config, ns["FEATURE_CACHE"]
        )
        provenance = (
            "training-dataset"
            if recording.id in training_ids
            else "evaluation-validation-test-only"
        )
        entries.append(item_type(prepared, provenance, None, True))
    for path in sorted(ns["FEEDBACK_ROOT"].glob("*/bundle.json")):
        prepared, _ = ns["feedback_prepared"](path, v2_heads.rally.feature_config)
        partition = "fit" if prepared.recording.id in feedback_fit else "held-out"
        entries.append(item_type(prepared, "export-feedback", partition, True))

    inference_root = experiment_root / "inference"
    baseline_predictions = {}
    candidate_predictions = {}
    video_payloads = []
    for index, entry in enumerate(entries, start=1):
        prepared = entry.prepared
        recording_id = prepared.recording.id
        baseline = read_ranges(
            inference_root / "current-production-ensemble" / f"{recording_id}.json"
        )
        old_scored = predict_scored(prepared, old_heads)
        v2_scored = predict_scored(prepared, v2_heads)
        old = tuple(interval(float(row.start), float(row.end)) for row in old_scored)
        v2 = tuple(interval(float(row.start), float(row.end)) for row in v2_scored)
        suppression_scored, _ = ns["decode_probabilities"](
            prepared.sequence.times,
            suppression_model.predict(prepared.contextual_values),
            prepared.sequence.metadata.duration,
            suppression_config,
            4.0,
        )
        suppression_decoded = tuple(
            interval(float(row.start), float(row.end))
            for row in suppression_scored
        )
        suppression_cuts = intersect(baseline, suppression_decoded)
        one_model_support = one_model_support_for(
            old, v2, prepared.sequence.metadata.duration
        )
        applied_cuts = intersect(suppression_cuts, one_model_support)
        candidate = subtract(baseline, applied_cuts)
        baseline_predictions[recording_id] = baseline
        candidate_predictions[recording_id] = candidate

        baseline_padded = subtract(
            pad_and_merge(baseline, prepared.sequence.metadata.duration, 2.0, 3.0),
            prepared.recording.ignored_intervals,
        )
        candidate_padded = subtract(
            pad_and_merge(candidate, prepared.sequence.metadata.duration, 2.0, 3.0),
            prepared.recording.ignored_intervals,
        )
        affected = []
        affected_core = []
        for rally_number, core in enumerate(prepared.recording.rallies, start=1):
            evaluable_core = subtract((core,), prepared.recording.ignored_intervals)
            evaluable_duration = total_duration(evaluable_core)
            if evaluable_duration <= 1e-9:
                continue
            production_covered = intersection_duration(evaluable_core, baseline_padded)
            candidate_covered = intersection_duration(evaluable_core, candidate_padded)
            lost = max(0.0, production_covered - candidate_covered)
            if lost <= 1e-6:
                continue
            affected_core.extend(evaluable_core)
            affected.append({
                "rallyNumber": rally_number,
                "start": core.start,
                "end": core.end,
                "duration": evaluable_duration,
                "productionCoveredSeconds": production_covered,
                "candidateCoveredSeconds": candidate_covered,
                "lostCoreSeconds": lost,
                "completeMiss": candidate_covered <= 1e-6,
            })
        if affected:
            focus_seed = pad_and_merge(
                affected_core, prepared.sequence.metadata.duration, 2.0, 3.0
            )
            related = []
            for group in (
                pad_and_merge(prepared.recording.rallies, prepared.sequence.metadata.duration, 2.0, 3.0),
                pad_and_merge(old, prepared.sequence.metadata.duration, 2.0, 3.0),
                pad_and_merge(v2, prepared.sequence.metadata.duration, 2.0, 3.0),
                baseline_padded,
                candidate_padded,
            ):
                related.extend(
                    item
                    for item in group
                    if intersection_duration((item,), focus_seed) > 0
                )
            focus_ranges = subtract(merge((*focus_seed, *related)), prepared.recording.ignored_intervals)
            affected.sort(key=lambda row: (not row["completeMiss"], row["start"]))
            filename = prepared.recording.raw.get(
                "sourceFilename", prepared.recording.video.name
            )
            video_payloads.append({
                "recordingId": recording_id,
                "file": filename,
                "duration": prepared.sequence.metadata.duration,
                "environment": prepared.recording.environment,
                "provenance": entry.provenance,
                "feedbackPartition": entry.feedback_partition,
                "videoUrl": (
                    f"/api/model-feedback/{feedback_imports[recording_id]}/source"
                    if entry.provenance == "export-feedback"
                    else f"/api/labeling/tasks/{recording_id}/video"
                ),
                "completeMisses": sum(row["completeMiss"] for row in affected),
                "partialMisses": sum(not row["completeMiss"] for row in affected),
                "lostCoreSeconds": sum(row["lostCoreSeconds"] for row in affected),
                "affectedRallies": affected,
                "focusRanges": rows(focus_ranges),
                "tracks": {
                    "affectedRallies": track(affected_core, prepared),
                    "human": track(prepared.recording.rallies, prepared),
                    "previousProduction": track(old, prepared, old_scored),
                    "allLabelsV2": track(v2, prepared, v2_scored),
                    "ensemble": track(baseline, prepared),
                    "ensembleSuppressed": track(candidate, prepared),
                    "suppressionApplied": {
                        "raw": scored_rows(
                            applied_cuts, suppression_scored, prepared
                        ),
                        "padding": [],
                        "joinedGaps": [],
                        "padded": rows(subtract(applied_cuts, prepared.recording.ignored_intervals)),
                    },
                },
            })
        print(f"Prepared {index}/{len(entries)}: {recording_id}", flush=True)

    video_payloads.sort(
        key=lambda video: (
            video["completeMisses"] == 0,
            -video["completeMisses"],
            -video["lostCoreSeconds"],
            video["file"],
        )
    )
    all_metric = ns["metric_for"](entries, candidate_predictions, 2.0)
    baseline_metric = ns["metric_for"](entries, baseline_predictions, 2.0)

    def metric_summary(value: dict[str, Any]) -> dict[str, Any]:
        return {
            "core": {
                key: value["core"][key]
                for key in ("precision", "recall", "f1")
            },
            "padded": {
                key: value["padded"][key]
                for key in ("precision", "recall", "f1")
            },
            **{
                key: value[key]
                for key in (
                    "P_pad",
                    "R_core",
                    "F1_padP_coreR",
                    "paddedModelExportSeconds",
                )
            },
        }

    correctly_removed_predictions = 0
    correctly_removed_raw_seconds = 0.0
    correctly_removed_rows = []
    for entry in entries:
        prepared = entry.prepared
        recording_id = prepared.recording.id
        baseline = subtract(
            baseline_predictions[recording_id],
            prepared.recording.ignored_intervals,
        )
        candidate = subtract(
            candidate_predictions[recording_id],
            prepared.recording.ignored_intervals,
        )
        human_export = subtract(
            pad_and_merge(
                prepared.recording.rallies,
                prepared.sequence.metadata.duration,
                2.0,
                3.0,
            ),
            prepared.recording.ignored_intervals,
        )
        for prediction in baseline:
            if (
                intersection_duration((prediction,), human_export) <= 1e-9
                and intersection_duration((prediction,), candidate) <= 1e-9
            ):
                correctly_removed_predictions += 1
                correctly_removed_raw_seconds += prediction.end - prediction.start
                correctly_removed_rows.append({
                    "recordingId": recording_id,
                    "file": prepared.recording.raw.get(
                        "sourceFilename", prepared.recording.video.name
                    ),
                    "environment": prepared.recording.environment,
                    "provenance": entry.provenance,
                    "feedbackPartition": entry.feedback_partition,
                    "start": prediction.start,
                    "end": prediction.end,
                    "duration": prediction.end - prediction.start,
                })

    baseline_summary = metric_summary(baseline_metric)
    candidate_summary = metric_summary(all_metric)
    scopes = {
        "all-evaluable": entries,
        "development": [
            entry
            for entry in entries
            if entry.prepared.recording.id in ns["DEVELOPMENT_IDS"]
        ],
        "protected-test": [
            entry
            for entry in entries
            if entry.prepared.recording.id in ns["PROTECTED_TEST_IDS"]
        ],
        "training-dataset": [
            entry for entry in entries if entry.provenance == "training-dataset"
        ],
        "evaluation-validation-test-only": [
            entry
            for entry in entries
            if entry.provenance == "evaluation-validation-test-only"
        ],
        "export-feedback": [
            entry for entry in entries if entry.provenance == "export-feedback"
        ],
        "feedback-fit": [
            entry for entry in entries if entry.feedback_partition == "fit"
        ],
        "feedback-held-out": [
            entry for entry in entries if entry.feedback_partition == "held-out"
        ],
    }
    scope_metrics = {
        scope: {
            "recordings": len(scoped_entries),
            "production": {
                f"{padding:g}": metric_summary(
                    ns["metric_for"](
                        scoped_entries, baseline_predictions, padding
                    )
                )
                for padding in ns["PADDING_CASES"]
            },
            "candidate": {
                f"{padding:g}": metric_summary(
                    ns["metric_for"](
                        scoped_entries, candidate_predictions, padding
                    )
                )
                for padding in ns["PADDING_CASES"]
            },
        }
        for scope, scoped_entries in scopes.items()
        if scoped_entries
    }
    agreement_join_description = (
        "with no agreement-component gap joining"
        if args.agreement_join_gap <= 0
        else f"and joins below {args.agreement_join_gap:g} seconds"
    )
    agreement_support_description = (
        "any raw prediction"
        if args.minimum_per_model_support <= 0
        else f"at least {args.minimum_per_model_support:g} raw seconds"
    )
    agreement_strategy = (
        "Group raw previous-production and all-labels-v2 predictions using "
        f"{args.agreement_padding:g}-second padding {agreement_join_description}. "
        "Protect a component when each model contributes "
        f"{agreement_support_description}; suppress only one-model components. "
        "The actual export still uses 2-second padding and joins below 3 seconds."
    )
    payload = {
        "schemaVersion": 4,
        "experiment": split.get("experiment", ns["EXPERIMENT_ID"]),
        "policyId": args.policy,
        "policyLabel": (
            f"Any-overlap: {args.agreement_padding:g}s grouping padding, "
            f"<{args.agreement_join_gap:g}s grouping join"
            if args.policy == "any-overlap"
            else POLICIES[args.policy]["label"]
        ),
        "modelVariantId": args.model_variant_id,
        "modelVariantLabel": args.model_variant_label,
        "suppressionModelPath": str(suppression_model_path.resolve()),
        "suppressionModelSha256": suppression_model.artifact_sha256,
        "strategy": (
            agreement_strategy
            if args.policy == "any-overlap"
            else POLICIES[args.policy]["strategy"]
        ),
        "paddingSecondsBeforeAndAfter": 2.0,
        "joinGapSecondsStrictlyLessThan": 3.0,
        "agreementGrouping": {
            "paddingSecondsBeforeAndAfter": args.agreement_padding,
            "joinGapSecondsStrictlyLessThan": args.agreement_join_gap,
            "minimumRawSupportSecondsPerModel": args.minimum_per_model_support,
            "effectiveRawGapSeconds": (
                2 * args.agreement_padding + args.agreement_join_gap
            ),
        },
        "confidenceNotes": {
            "componentPredictions": "Uncalibrated mean smoothed rally score over each decoded raw interval.",
            "suppressionApplied": "Uncalibrated mean smoothed suppression score over each decoded veto interval.",
            "derivedRanges": "Human labels, padding, joined gaps, production unions, and post-veto unions do not have a standalone confidence score.",
        },
        "summary": {
            "videos": len(video_payloads),
            "affectedRallies": sum(
                len(video["affectedRallies"]) for video in video_payloads
            ),
            "completeMisses": sum(video["completeMisses"] for video in video_payloads),
            "partialMisses": sum(video["partialMisses"] for video in video_payloads),
            "lostCoreSeconds": sum(video["lostCoreSeconds"] for video in video_payloads),
            "exportTimeSavedSeconds": (
                baseline_metric["paddedModelExportSeconds"]
                - all_metric["paddedModelExportSeconds"]
            ),
            "correctlyRemovedFalsePositivePredictions": correctly_removed_predictions,
            "correctlyRemovedRawSeconds": correctly_removed_raw_seconds,
            "production": baseline_summary,
            "candidate": candidate_summary,
        },
        "scopeMetrics": scope_metrics,
        "correctlyRemovedPredictions": correctly_removed_rows,
        "videos": video_payloads,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        delete=False,
    ) as handle:
        handle.write(rendered)
        temporary = Path(handle.name)
    os.replace(temporary, output)
    print(output)


if __name__ == "__main__":
    main()
