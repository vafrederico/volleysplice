#!/usr/bin/env python3
"""Sweep one-model-only agreement rules for the retrained suppression specialist.

The production export contract remains fixed at +/-2 seconds of padding and joins
for positive gaps strictly below 3 seconds.  This sweep changes only how nearby
old-model and all-labels-v2 raw predictions are grouped when deciding whether an
export region is protected as having support from both production components.
"""

from __future__ import annotations

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
OUTPUT = ROOT / "data" / "single-model-agreement-policy-sweep.json"
RETRAINED_MODEL = Path(
    "/mnt/freenas/volleycut/intake-2026-08-13/experiments/"
    "feedback-suppression-v3-2026-08-16/models/"
    "suppression-overlap-exclusion-retrained"
)
ROTATED_CAMERA_EXCEPTION = "project-15ljci6"

# Predeclared before opening protected-test or held-feedback policy results.
AGREEMENT_PADDINGS = (0.0, 0.5, 1.0, 1.5, 2.0)
AGREEMENT_JOIN_GAPS = (0.0, 0.5, 1.0, 2.0, 3.0)
MINIMUM_PER_MODEL_SUPPORT_SECONDS = (0.0, 0.5, 1.0)


def main() -> None:
    ns = runpy.run_path(str(TRAIN_SCRIPT))
    interval = ns["Interval"]
    merge = ns["_merge_intervals"]
    subtract = ns["subtract_intervals"]
    duration_of = ns["_duration"]
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

    def read_ranges(path: Path) -> tuple[Any, ...]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return tuple(
            interval(float(item["start"]), float(item["end"]))
            for item in payload["ranges"]
        )

    def one_model_support(
        old: tuple[Any, ...],
        v2: tuple[Any, ...],
        video_duration: float,
        agreement_padding: float,
        agreement_join_gap: float,
        minimum_support: float,
    ) -> tuple[Any, ...]:
        raw_union = ns["union_intervals"](old, v2)
        components = pad_and_merge(
            raw_union,
            video_duration,
            agreement_padding,
            agreement_join_gap,
        )
        one_model_components = []
        for component in components:
            old_support = intersection_duration((component,), old)
            v2_support = intersection_duration((component,), v2)
            threshold = max(minimum_support, 1e-9)
            if old_support < threshold or v2_support < threshold:
                one_model_components.append(component)
        return intersect(raw_union, one_model_components)

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
                    "paddedHumanExportSeconds",
                    "paddedDurationDifferenceSeconds",
                )
            },
        }

    old_heads, v2_heads = ns["load_heads"]()
    suppression_model = ns["load_model"](RETRAINED_MODEL)
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
        (ns["DEFAULT_OUTPUT"] / "split-policy.json").read_text(encoding="utf-8")
    )
    feedback_fit = set(split["fitProjectIds"])

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

    inference_root = ns["DEFAULT_OUTPUT"] / "inference"
    baseline_predictions: dict[str, tuple[Any, ...]] = {}
    component_predictions: dict[str, tuple[tuple[Any, ...], tuple[Any, ...]]] = {}
    suppression_cuts: dict[str, tuple[Any, ...]] = {}
    for index, entry in enumerate(entries, start=1):
        prepared = entry.prepared
        recording_id = prepared.recording.id
        baseline = read_ranges(
            inference_root / "current-production-ensemble" / f"{recording_id}.json"
        )
        old = tuple(ns["predict_heads"](prepared, old_heads))
        v2 = tuple(ns["predict_heads"](prepared, v2_heads))
        decoded = ns["decode_suppression"](
            prepared, suppression_model, suppression_config
        )
        baseline_predictions[recording_id] = baseline
        component_predictions[recording_id] = (old, v2)
        suppression_cuts[recording_id] = intersect(baseline, decoded)
        print(f"Prepared {index}/{len(entries)}: {recording_id}", flush=True)

    development = [
        entry
        for entry in entries
        if entry.prepared.recording.id in ns["DEVELOPMENT_IDS"]
    ]
    protected_test = [
        entry
        for entry in entries
        if entry.prepared.recording.id in ns["PROTECTED_TEST_IDS"]
    ]
    non_exception = [
        entry
        for entry in entries
        if entry.prepared.recording.id != ROTATED_CAMERA_EXCEPTION
    ]

    def miss_summary(
        scoped_entries: list[Any], candidate: dict[str, tuple[Any, ...]]
    ) -> dict[str, Any]:
        videos: set[str] = set()
        affected = complete = partial = 0
        lost_seconds = 0.0
        rows = []
        for entry in scoped_entries:
            prepared = entry.prepared
            recording_id = prepared.recording.id
            baseline_padded = subtract(
                pad_and_merge(
                    baseline_predictions[recording_id],
                    prepared.sequence.metadata.duration,
                    2.0,
                    3.0,
                ),
                prepared.recording.ignored_intervals,
            )
            candidate_padded = subtract(
                pad_and_merge(
                    candidate[recording_id],
                    prepared.sequence.metadata.duration,
                    2.0,
                    3.0,
                ),
                prepared.recording.ignored_intervals,
            )
            for rally_number, core in enumerate(prepared.recording.rallies, start=1):
                evaluable = subtract((core,), prepared.recording.ignored_intervals)
                if duration_of(evaluable) <= 1e-9:
                    continue
                before = intersection_duration(evaluable, baseline_padded)
                after = intersection_duration(evaluable, candidate_padded)
                lost = max(0.0, before - after)
                if lost <= 1e-6:
                    continue
                is_complete = after <= 1e-6
                videos.add(recording_id)
                affected += 1
                complete += int(is_complete)
                partial += int(not is_complete)
                lost_seconds += lost
                rows.append({
                    "recordingId": recording_id,
                    "file": prepared.recording.raw.get(
                        "sourceFilename", prepared.recording.video.name
                    ),
                    "rallyNumber": rally_number,
                    "start": core.start,
                    "end": core.end,
                    "lostCoreSeconds": lost,
                    "completeMiss": is_complete,
                })
        return {
            "videos": len(videos),
            "affectedRallies": affected,
            "completeMisses": complete,
            "partialMisses": partial,
            "lostCoreSeconds": lost_seconds,
            "rows": rows,
        }

    baseline_all = ns["metric_for"](entries, baseline_predictions, 2.0)
    baseline_development = ns["metric_for"](
        development, baseline_predictions, 2.0
    )
    baseline_test = ns["metric_for"](protected_test, baseline_predictions, 2.0)
    results = []
    for agreement_padding in AGREEMENT_PADDINGS:
        for agreement_join_gap in AGREEMENT_JOIN_GAPS:
            for minimum_support in MINIMUM_PER_MODEL_SUPPORT_SECONDS:
                candidate: dict[str, tuple[Any, ...]] = {}
                for entry in entries:
                    prepared = entry.prepared
                    recording_id = prepared.recording.id
                    old, v2 = component_predictions[recording_id]
                    eligible = one_model_support(
                        old,
                        v2,
                        prepared.sequence.metadata.duration,
                        agreement_padding,
                        agreement_join_gap,
                        minimum_support,
                    )
                    applied = intersect(suppression_cuts[recording_id], eligible)
                    candidate[recording_id] = subtract(
                        baseline_predictions[recording_id], applied
                    )

                all_metric = ns["metric_for"](entries, candidate, 2.0)
                development_metric = ns["metric_for"](
                    development, candidate, 2.0
                )
                test_metric = ns["metric_for"](protected_test, candidate, 2.0)
                correctly_removed = 0
                correctly_removed_raw_seconds = 0.0
                for entry in entries:
                    prepared = entry.prepared
                    recording_id = prepared.recording.id
                    baseline = subtract(
                        baseline_predictions[recording_id],
                        prepared.recording.ignored_intervals,
                    )
                    candidate_ranges = subtract(
                        candidate[recording_id],
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
                            and intersection_duration((prediction,), candidate_ranges) <= 1e-9
                        ):
                            correctly_removed += 1
                            correctly_removed_raw_seconds += prediction.end - prediction.start

                all_misses = miss_summary(entries, candidate)
                non_exception_misses = miss_summary(non_exception, candidate)
                development_misses = miss_summary(development, candidate)
                test_misses = miss_summary(protected_test, candidate)
                results.append({
                    "id": (
                        f"pad-{agreement_padding:g}-join-{agreement_join_gap:g}"
                        f"-support-{minimum_support:g}"
                    ),
                    "agreementPaddingSeconds": agreement_padding,
                    "agreementJoinGapSecondsStrictlyLessThan": agreement_join_gap,
                    "effectiveRawGapSeconds": 2 * agreement_padding + agreement_join_gap,
                    "minimumRawSupportSecondsPerModel": minimum_support,
                    "correctlyRemovedFalsePositivePredictions": correctly_removed,
                    "correctlyRemovedRawSeconds": correctly_removed_raw_seconds,
                    "exportTimeSavedSeconds": (
                        baseline_all["paddedModelExportSeconds"]
                        - all_metric["paddedModelExportSeconds"]
                    ),
                    "allEvaluable": {
                        "metric": metric_summary(all_metric),
                        "misses": all_misses,
                        "paddingSensitivity": {
                            f"{padding:g}": metric_summary(
                                ns["metric_for"](entries, candidate, padding)
                            )
                            for padding in ns["PADDING_CASES"]
                        },
                    },
                    "allEvaluableExceptRotatedCamera": {
                        "misses": non_exception_misses,
                    },
                    "development": {
                        "metric": metric_summary(development_metric),
                        "misses": development_misses,
                    },
                    "protectedTest": {
                        "metric": metric_summary(test_metric),
                        "misses": test_misses,
                    },
                })

    # Official ordering is based only on predeclared development data.  Full-set
    # no-miss rows are a post-hoc diagnostic and are never used to tune the model.
    development_ranking = sorted(
        (row["id"] for row in results),
        key=lambda policy_id: next(
            (
                -row["development"]["metric"]["F1_padP_coreR"],
                -row["development"]["metric"]["P_pad"],
                -row["development"]["metric"]["R_core"],
                row["development"]["metric"]["paddedModelExportSeconds"],
                row["id"],
            )
            for row in results
            if row["id"] == policy_id
        ),
    )
    zero_non_exception_miss_ids = [
        row["id"]
        for row in sorted(
            results,
            key=lambda row: (
                -row["correctlyRemovedFalsePositivePredictions"],
                -row["correctlyRemovedRawSeconds"],
                -row["exportTimeSavedSeconds"],
                row["id"],
            ),
        )
        if row["allEvaluableExceptRotatedCamera"]["misses"]["affectedRallies"] == 0
    ]

    payload = {
        "schemaVersion": 1,
        "experiment": ns["EXPERIMENT_ID"],
        "modelVariantId": "overlap-exclusion-retrained",
        "suppressionModelPath": str(RETRAINED_MODEL),
        "suppressionModelSha256": suppression_model.artifact_sha256,
        "fixedProductExportRule": {
            "paddingSecondsBeforeAndAfter": 2.0,
            "joinGapSecondsStrictlyLessThan": 3.0,
        },
        "predeclaredSweep": {
            "agreementPaddingSeconds": list(AGREEMENT_PADDINGS),
            "agreementJoinGapSecondsStrictlyLessThan": list(AGREEMENT_JOIN_GAPS),
            "minimumRawSupportSecondsPerModel": list(MINIMUM_PER_MODEL_SUPPORT_SECONDS),
        },
        "rotatedCameraDiagnosticException": {
            "recordingId": ROTATED_CAMERA_EXCEPTION,
            "file": "PXL_20260816_164327879.mp4",
            "reason": "Camera rotates and no longer keeps the full court in frame.",
        },
        "selectionContract": {
            "officialRankingScope": sorted(ns["DEVELOPMENT_IDS"]),
            "protectedTestIds": sorted(ns["PROTECTED_TEST_IDS"]),
            "rankingMetric": "F1_padP_coreR",
            "note": "Protected test and held-feedback outcomes are report-only and were not used for official ranking.",
        },
        "baseline": {
            "allEvaluable": metric_summary(baseline_all),
            "allEvaluablePaddingSensitivity": {
                f"{padding:g}": metric_summary(
                    ns["metric_for"](entries, baseline_predictions, padding)
                )
                for padding in ns["PADDING_CASES"]
            },
            "development": metric_summary(baseline_development),
            "protectedTest": metric_summary(baseline_test),
        },
        "developmentRanking": development_ranking,
        "zeroNonExceptionMissDiagnosticRanking": zero_non_exception_miss_ids,
        "results": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=OUTPUT.parent,
        prefix=f".{OUTPUT.name}.",
        delete=False,
    ) as handle:
        handle.write(rendered)
        temporary = Path(handle.name)
    os.replace(temporary, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
