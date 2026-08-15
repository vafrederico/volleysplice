#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from analysis.annotations import load_label_document
from analysis.crop_evaluation import (
    DEFAULT_JOIN_GAP_SECONDS,
    RecordingIntervals,
    evaluate_f1_pad_p_core_r,
    pad_and_merge_intervals,
    subtract_intervals,
)
from analysis.metrics import (
    aggregate_evaluations,
    evaluate_intervals,
    truth_slice_metrics,
)
from analysis.schema import Interval, Recording, load_manifest


BASE_VARIANTS = {
    "offline": "model-9c92b8e9333f",
    "browserOnDevice": "model-browser-on-device-9c92b8e9333f",
}
LIBSWRESAMPLE_WASM_VARIANT = "browserOnDeviceLibswresampleWasm"
LIBSWRESAMPLE_WASM_PREFIX = (
    "model-browser-on-device-libswresample-wasm-9c92b8e9333f"
)
REQUIRED_PADDING_SECONDS = (0.0, 1.0, 2.0, 3.0)
LIFT_PADDING_SECONDS = (2.0, 3.0)
MODEL_SHA256 = "9c92b8e9333f6247336639409acbe063da68dea4cc74735c7b2a8791f8dda2a7"
MODEL_BUNDLE_SHA256 = (
    "d8cc42f70bc10576a5e03251b05981ceeee1a61a15c61cc5dfb68dd631e6f90d"
)
LIBSWRESAMPLE_WASM_SHA256 = (
    "c7ed95ed8b6f5e11ea9e86262214b978bd145a6ec1f648dd56616af298449f90"
)
LIBSWRESAMPLE_GLUE_SHA256 = (
    "022782d1e08e483d8c67de30f68177eff5904998c410df028f26e342c793ae48"
)
MODEL_VERSION = "dead-state-transition-audio-normalized-v5-no-legacy-final"
MODEL_COMPONENTS = {
    "full-audiovisual-audio-normalized-v3": (
        "ca004bff50fb36652142a861fcc9334bab0ad50d3aca68cd3bb30d1c53e725ba"
    ),
    "serve-specialist-audio-normalized-v5": (
        "5ff951b60ee838a1c51e8677ede1c9c9ed5a48b252028b22b2ea8783f4964bbb"
    ),
    MODEL_VERSION: MODEL_SHA256,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def load_predictions(
    path: Path,
    expected_analysis_id: str,
    expected_recording_id: str,
) -> tuple[float, tuple[Interval, ...], dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read inference output {path}: {error}") from error
    source = payload.get("source")
    analysis = payload.get("analysis")
    rallies = payload.get("rallies")
    if (
        payload.get("schemaVersion") != 1
        or payload.get("id") != expected_analysis_id
        or payload.get("recordingId") != expected_recording_id
    ):
        raise ValueError(f"unexpected analysis identity in {path}")
    if (
        not isinstance(source, dict)
        or not _number(source.get("duration"))
        or not isinstance(analysis, dict)
        or not isinstance(rallies, list)
    ):
        raise ValueError(
            f"analysis metadata, duration, or rallies are missing in {path}"
        )
    duration = float(source["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"analysis source duration is invalid in {path}")
    predictions: list[Interval] = []
    previous_end = -1.0
    excluded = 0
    for index, row in enumerate(rallies):
        if (
            not isinstance(row, dict)
            or not _number(row.get("start"))
            or not _number(row.get("end"))
        ):
            raise ValueError(f"invalid rally {index} in {path}")
        start = float(row["start"])
        end = float(row["end"])
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < previous_end
            or start < 0
            or end <= start
            or end > duration + 1e-6
        ):
            raise ValueError(f"out-of-range or unordered rally {index} in {path}")
        included = row.get("included", True)
        if not isinstance(included, bool):
            raise ValueError(f"rally {index} included flag is not boolean in {path}")
        if included:
            predictions.append(Interval(start, min(end, duration)))
        else:
            excluded += 1
        previous_end = end
    model_sha256 = analysis.get("modelSha256")
    if (
        not isinstance(model_sha256, str)
        or len(model_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in model_sha256.lower()
        )
    ):
        raise ValueError(f"analysis model SHA-256 is missing or invalid in {path}")
    if model_sha256.lower() != MODEL_SHA256:
        raise ValueError(f"analysis references the wrong model artifact in {path}")
    if analysis.get("modelVersion") != MODEL_VERSION:
        raise ValueError(f"analysis references the wrong model version in {path}")
    raw_models = analysis.get("models")
    if not isinstance(raw_models, dict):
        raise ValueError(f"analysis component models are missing in {path}")
    components: dict[str, str] = {}
    for component in raw_models.values():
        if not isinstance(component, dict):
            raise ValueError(f"analysis component model is invalid in {path}")
        version = component.get("version")
        digest = component.get("sha256")
        if not isinstance(version, str) or not isinstance(digest, str):
            raise ValueError(f"analysis component model is invalid in {path}")
        components[version] = digest.lower()
    if components != MODEL_COMPONENTS:
        raise ValueError(f"analysis references the wrong model components in {path}")
    model_bundle_sha256 = analysis.get("modelBundleSha256")
    if model_bundle_sha256 is not None and model_bundle_sha256 != MODEL_BUNDLE_SHA256:
        raise ValueError(f"analysis references the wrong browser model bundle in {path}")
    runtime_provenance = analysis.get("provenance")
    if runtime_provenance is not None and not isinstance(runtime_provenance, dict):
        raise ValueError(f"analysis runtime provenance is invalid in {path}")
    source_sha256 = source.get("contentSha256")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in source_sha256.lower()
        )
    ):
        raise ValueError(f"analysis source SHA-256 is missing or invalid in {path}")
    return duration, tuple(predictions), {
        "path": str(path.resolve()),
        "fileSha256": sha256_file(path),
        "modelSha256": model_sha256,
        "modelVersion": analysis.get("modelVersion"),
        "modelComponents": components,
        "modelBundleSha256": model_bundle_sha256,
        "sourceContentSha256": source_sha256.lower(),
        "sourceFilename": source.get("filename"),
        "sourceDuration": duration,
        "method": analysis.get("method"),
        "variantLabel": analysis.get("variantLabel"),
        "runtimeVariant": (
            runtime_provenance.get("runtimeVariant")
            if isinstance(runtime_provenance, dict)
            else None
        ),
        "audioResampler": (
            runtime_provenance.get("audioResampler")
            if isinstance(runtime_provenance, dict)
            else None
        ),
        "includedRallies": len(predictions),
        "excludedRallies": excluded,
    }


def harmonic_mean(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def interval_duration(intervals: tuple[Interval, ...]) -> float:
    return sum(item.end - item.start for item in intervals)


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _adjusted_metrics(row: dict[str, Any]) -> dict[str, float]:
    padded_precision = _ratio(
        float(row["paddedPrecisionIntersectionSeconds"]),
        float(row["paddedModelExportSeconds"]),
    )
    core_recall = _ratio(
        float(row["coreRecallIntersectionSeconds"]),
        float(row["coreHumanSeconds"]),
    )
    model_seconds = float(row["paddedModelExportSeconds"])
    human_seconds = float(row["paddedHumanExportSeconds"])
    return {
        "P_pad": padded_precision,
        "R_core": core_recall,
        "F1_padP_coreR": harmonic_mean(padded_precision, core_recall),
        "paddedModelExportSeconds": model_seconds,
        "paddedHumanExportSeconds": human_seconds,
        "exportDurationDifferenceSeconds": model_seconds - human_seconds,
    }


def evaluate_scope(
    recordings: list[RecordingIntervals],
    paddings: list[float],
) -> dict[str, Any]:
    adjusted_rows = {
        float(row["paddingSecondsBeforeAndAfter"]): row
        for row in evaluate_f1_pad_p_core_r(recordings, paddings)
    }
    results: dict[str, Any] = {}
    for padding in paddings:
        adjusted = adjusted_rows[padding]
        adjusted_by_id = {
            str(row["id"]): _adjusted_metrics(row)
            for row in adjusted["recordings"]
        }
        per_recording: list[dict[str, Any]] = []
        contained_total = 0
        actual_export_sections = 0
        actual_export_seconds = 0.0
        for recording in recordings:
            truth = subtract_intervals(recording.truth, recording.ignored_intervals)
            actual_export = pad_and_merge_intervals(
                recording.predictions,
                recording.duration,
                padding,
            )
            export = subtract_intervals(
                actual_export,
                recording.ignored_intervals,
            )
            event = evaluate_intervals(truth, export)
            containment = truth_slice_metrics(truth, export, range(len(truth)))
            fully_contained = round(
                float(containment["fullyContainedRate"]) * len(truth)
            )
            contained_total += fully_contained
            actual_export_sections += len(actual_export)
            actual_export_seconds += interval_duration(actual_export)
            per_recording.append(
                {
                    "id": recording.id,
                    "eventMetricsAtIou05": {
                        "truePositives": event["matchedRallies"],
                        "evaluablePredictedFragments": event["predictedRallies"],
                        "expectedCoreRallies": event["trueRallies"],
                        "precision": event["eventPrecision"],
                        "recall": event["eventRecall"],
                        "F1": event["eventF1"],
                    },
                    "adjustedMetrics": adjusted_by_id[recording.id],
                    "expectedCoreRallies": len(truth),
                    "fullyContainedCoreRallies": fully_contained,
                    "eventTruePositivesAtIou05": event["matchedRallies"],
                    "inputPredictionCount": len(recording.predictions),
                    "actualMergedExportSections": len(actual_export),
                    "actualPaddedExportSeconds": interval_duration(actual_export),
                    "evaluableExportFragments": len(export),
                    "evaluablePaddedModelSeconds": interval_duration(export),
                }
            )
        aggregate = aggregate_evaluations(
            [
                evaluate_intervals(
                    subtract_intervals(item.truth, item.ignored_intervals),
                    subtract_intervals(
                        pad_and_merge_intervals(
                            item.predictions,
                            item.duration,
                            padding,
                        ),
                        item.ignored_intervals,
                    ),
                )
                for item in recordings
            ]
        )
        live_precision = float(aggregate["liveTimePrecision"])
        live_recall = float(aggregate["liveTimeRecall"])
        results[f"pad-{padding:g}s"] = {
            "paddingSecondsBeforeAndAfter": padding,
            "eventMetricsAtIou05": {
                "truePositives": aggregate["matchedRallies"],
                "evaluablePredictedFragments": aggregate["predictedRallies"],
                "expectedCoreRallies": aggregate["trueRallies"],
                "precision": aggregate["eventPrecision"],
                "recall": aggregate["eventRecall"],
                "F1": aggregate["eventF1"],
            },
            "coreLiveTimeMetrics": {
                "precision": live_precision,
                "recall": live_recall,
                "F1": harmonic_mean(live_precision, live_recall),
            },
            "adjustedMetrics": {
                "P_pad": adjusted["P_pad"],
                "R_core": adjusted["R_core"],
                "F1_padP_coreR": adjusted["F1_padP_coreR"],
                "paddedModelExportSeconds": adjusted[
                    "paddedModelExportSeconds"
                ],
                "paddedHumanExportSeconds": adjusted[
                    "paddedHumanExportSeconds"
                ],
                "exportDurationDifferenceSeconds": adjusted[
                    "exportDurationDifferenceSeconds"
                ],
            },
            "fullyContainedCoreRallies": contained_total,
            "expectedCoreRallies": aggregate["trueRallies"],
            "inputPredictionCount": adjusted["inputCropCount"],
            "actualMergedExportSections": actual_export_sections,
            "actualPaddedExportSeconds": actual_export_seconds,
            "evaluableExportFragments": aggregate["predictedRallies"],
            "evaluablePaddedModelSeconds": adjusted["paddedModelExportSeconds"],
            "evaluablePaddedHumanSeconds": adjusted["paddedHumanExportSeconds"],
            "evaluableExportDurationDifferenceSeconds": adjusted[
                "exportDurationDifferenceSeconds"
            ],
            "cropCounts": {
                "inputPredictionRanges": adjusted["inputCropCount"],
                "actualMergedExportSections": actual_export_sections,
                "evaluableExportFragments": aggregate["predictedRallies"],
            },
            "perRecording": per_recording,
        }
    return results


def _point_lift(candidate: float, reference: float) -> float:
    return 100.0 * (candidate - reference)


def _aggregate_lift(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> dict[str, float | int]:
    if candidate["expectedCoreRallies"] != reference["expectedCoreRallies"]:
        raise ValueError("cannot compare variants with different expected rally counts")
    candidate_adjusted = candidate["adjustedMetrics"]
    reference_adjusted = reference["adjustedMetrics"]
    candidate_event = candidate["eventMetricsAtIou05"]
    reference_event = reference["eventMetricsAtIou05"]
    return {
        "P_padPoints": _point_lift(
            float(candidate_adjusted["P_pad"]),
            float(reference_adjusted["P_pad"]),
        ),
        "R_corePoints": _point_lift(
            float(candidate_adjusted["R_core"]),
            float(reference_adjusted["R_core"]),
        ),
        "F1_padP_coreRPoints": _point_lift(
            float(candidate_adjusted["F1_padP_coreR"]),
            float(reference_adjusted["F1_padP_coreR"]),
        ),
        "eventPrecisionPoints": _point_lift(
            float(candidate_event["precision"]),
            float(reference_event["precision"]),
        ),
        "eventRecallPoints": _point_lift(
            float(candidate_event["recall"]),
            float(reference_event["recall"]),
        ),
        "eventF1Points": _point_lift(
            float(candidate_event["F1"]),
            float(reference_event["F1"]),
        ),
        "eventTruePositives": int(candidate_event["truePositives"])
        - int(reference_event["truePositives"]),
        "fullyContainedCoreRallies": int(
            candidate["fullyContainedCoreRallies"]
        )
        - int(reference["fullyContainedCoreRallies"]),
        "inputPredictionRanges": int(candidate["inputPredictionCount"])
        - int(reference["inputPredictionCount"]),
        "actualMergedExportSections": int(
            candidate["actualMergedExportSections"]
        )
        - int(reference["actualMergedExportSections"]),
        "evaluableExportFragments": int(candidate["evaluableExportFragments"])
        - int(reference["evaluableExportFragments"]),
        "actualPaddedExportSeconds": float(
            candidate["actualPaddedExportSeconds"]
        )
        - float(reference["actualPaddedExportSeconds"]),
        "evaluablePaddedModelSeconds": float(
            candidate_adjusted["paddedModelExportSeconds"]
        )
        - float(reference_adjusted["paddedModelExportSeconds"]),
        "exportDurationDifferenceSeconds": float(
            candidate_adjusted["exportDurationDifferenceSeconds"]
        )
        - float(reference_adjusted["exportDurationDifferenceSeconds"]),
    }


def _recording_lift(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> dict[str, float | int]:
    if (
        candidate["id"] != reference["id"]
        or candidate["expectedCoreRallies"] != reference["expectedCoreRallies"]
    ):
        raise ValueError("cannot compare mismatched per-recording rows")
    return _aggregate_lift(candidate, reference)


def build_lift_summaries(
    variant_reports: dict[str, Any],
    *,
    candidate_variant: str,
    reference_variants: tuple[str, ...],
) -> dict[str, Any]:
    """Summarize SWR lifts at product-relevant padding without ranking test data."""
    if candidate_variant not in variant_reports:
        raise ValueError(f"candidate variant is missing: {candidate_variant}")
    for variant in reference_variants:
        if variant not in variant_reports:
            raise ValueError(f"reference variant is missing: {variant}")
    candidate_scopes = variant_reports[candidate_variant]
    by_scope: dict[str, Any] = {}
    for scope_name, candidate_scope in candidate_scopes.items():
        scope_result: dict[str, Any] = {}
        for padding in LIFT_PADDING_SECONDS:
            padding_key = f"pad-{padding:g}s"
            if padding_key not in candidate_scope:
                raise ValueError(f"candidate report is missing {padding_key}")
            candidate = candidate_scope[padding_key]
            candidate_recordings = {
                row["id"]: row for row in candidate["perRecording"]
            }
            reference_rows: dict[str, Any] = {}
            per_recording = {
                recording_id: {
                    "id": recording_id,
                    "expectedCoreRallies": row["expectedCoreRallies"],
                    "candidateFullyContainedCoreRallies": row[
                        "fullyContainedCoreRallies"
                    ],
                    "versus": {},
                }
                for recording_id, row in candidate_recordings.items()
            }
            for reference_variant in reference_variants:
                try:
                    reference = variant_reports[reference_variant][scope_name][
                        padding_key
                    ]
                except KeyError as error:
                    raise ValueError(
                        f"reference report {reference_variant} is missing "
                        f"{scope_name}/{padding_key}"
                    ) from error
                reference_rows[reference_variant] = _aggregate_lift(
                    candidate, reference
                )
                reference_recordings = {
                    row["id"]: row for row in reference["perRecording"]
                }
                if candidate_recordings.keys() != reference_recordings.keys():
                    raise ValueError(
                        "candidate and reference per-recording sets differ for "
                        f"{scope_name}/{padding_key}"
                    )
                for recording_id, row in candidate_recordings.items():
                    per_recording[recording_id]["versus"][reference_variant] = (
                        _recording_lift(
                            row,
                            reference_recordings[recording_id],
                        )
                    )
            scope_result[padding_key] = {
                "paddingSecondsBeforeAndAfter": padding,
                "candidate": {
                    "fullyContainedCoreRallies": candidate[
                        "fullyContainedCoreRallies"
                    ],
                    "expectedCoreRallies": candidate["expectedCoreRallies"],
                    "adjustedMetrics": candidate["adjustedMetrics"],
                    "eventMetricsAtIou05": candidate["eventMetricsAtIou05"],
                    "cropCounts": candidate["cropCounts"],
                },
                "versus": reference_rows,
                "perRecording": list(per_recording.values()),
            }
        by_scope[scope_name] = scope_result
    return {
        "candidateVariant": candidate_variant,
        "referenceVariants": list(reference_variants),
        "paddingSecondsBeforeAndAfter": list(LIFT_PADDING_SECONDS),
        "interpretation": (
            "Point fields are candidate minus reference in percentage points; "
            "count and duration fields are candidate minus reference in their "
            "named units. Protected-test rows are descriptive guardrails only."
        ),
        "byScope": by_scope,
    }


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    data_root = Path(os.environ.get("VOLLEYCUT_DATA_ROOT", "/mnt/freenas/volleycut"))
    gold_workspace = data_root / "labeling-v1-2026-08-09"
    model_workspace = data_root / "labeling-v1-2026-08-09-no-beach-2026-08-12"
    analyses_root = model_workspace / "analyses"
    parser = argparse.ArgumentParser(
        description=(
            "Compare offline, browser-linear, and optionally browser-libswresample "
            "model-9c92 export padding."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=gold_workspace / "manifests/full-gold-v1.json",
    )
    parser.add_argument(
        "--model-manifest",
        type=Path,
        default=model_workspace / "manifests/full-gold-v1-no-beach.json",
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=gold_workspace / "labels/full",
    )
    parser.add_argument("--analyses-root", type=Path, default=analyses_root)
    parser.add_argument(
        "--padding-seconds",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Padding sweep. The SWR experiment requires exactly 0 1 2 3; "
            "legacy two-runtime reproduction defaults to 2 3."
        ),
    )
    parser.add_argument(
        "--include-libswresample-wasm",
        action="store_true",
        help=(
            "Include ranges produced by the browser libswresample-WASM runtime "
            "and emit its pad-2s/pad-3s lifts."
        ),
    )
    parser.add_argument(
        "--libswresample-wasm-prefix",
        default=LIBSWRESAMPLE_WASM_PREFIX,
        help="Analysis artifact prefix for the libswresample-WASM runtime.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    destination = args.output.expanduser().resolve()
    if destination.exists():
        raise ValueError(f"refusing to overwrite existing report: {destination}")
    requested_paddings = args.padding_seconds
    if requested_paddings is None:
        requested_paddings = (
            list(REQUIRED_PADDING_SECONDS)
            if args.include_libswresample_wasm
            else [2.0, 3.0]
        )
    paddings = sorted(set(float(value) for value in requested_paddings))
    if not paddings or paddings[0] < 0:
        raise ValueError("padding values must be non-negative")
    if (
        args.include_libswresample_wasm
        and tuple(paddings) != REQUIRED_PADDING_SECONDS
    ):
        raise ValueError(
            "the libswresample-WASM comparison requires padding values "
            "0, 1, 2, and 3 seconds"
        )
    variants = dict(BASE_VARIANTS)
    if args.include_libswresample_wasm:
        prefix = args.libswresample_wasm_prefix.strip()
        if not prefix:
            raise ValueError("libswresample-WASM artifact prefix cannot be empty")
        variants[LIBSWRESAMPLE_WASM_VARIANT] = prefix
    manifest_path = args.manifest.expanduser().resolve()
    model_manifest_path = args.model_manifest.expanduser().resolve()
    labels_root = args.labels_root.expanduser().resolve()
    analyses_root = args.analyses_root.expanduser().resolve()
    manifest = load_manifest(manifest_path, require_videos=False)
    model_manifest = load_manifest(model_manifest_path, require_videos=False)
    in_domain_ids = {item.id for item in model_manifest.recordings}
    all_ids = {item.id for item in manifest.recordings}
    if not in_domain_ids < all_ids:
        raise ValueError(
            "model manifest must be a strict subset of the evaluation manifest"
        )
    ood_ids = all_ids - in_domain_ids

    ignored_by_id: dict[str, tuple[Interval, ...]] = {}
    label_provenance: dict[str, Any] = {}
    for recording in manifest.recordings:
        path = labels_root / f"{recording.id}.labels.json"
        document = load_label_document(
            path,
            require_complete=False,
            require_video=False,
        )
        if (
            document.recording_id != recording.id
            or document.split != recording.split
            or document.source_group != recording.source_group
        ):
            raise ValueError(f"label identity differs from manifest for {recording.id}")
        ignored_by_id[recording.id] = document.ignored_intervals
        label_provenance[recording.id] = {
            "path": str(path.resolve()),
            "fileSha256": sha256_file(path),
            "ignoredIntervals": [item.to_dict() for item in document.ignored_intervals],
            "ralliesMatchFrozenManifest": document.rallies == recording.rallies,
        }

    scopes: dict[str, list[Recording]] = {
        "all9Descriptive": list(manifest.recordings),
        "inDomainNoBeach7Descriptive": [
            item for item in manifest.recordings if item.id in in_domain_ids
        ],
        "training4InDomainResubstitution": [
            item
            for item in manifest.recordings
            if item.split == "train" and item.id in in_domain_ids
        ],
        "validation2": [
            item for item in manifest.recordings if item.split == "validation"
        ],
        "protectedTest1Descriptive": [
            item for item in manifest.recordings if item.split == "test"
        ],
        "beachOod2Qualitative": [
            item for item in manifest.recordings if item.id in ood_ids
        ],
    }
    variant_reports: dict[str, Any] = {}
    analysis_provenance: dict[str, dict[str, Any]] = {}
    for variant_label, prefix in variants.items():
        loaded: dict[str, RecordingIntervals] = {}
        provenance: dict[str, Any] = {}
        for recording in manifest.recordings:
            analysis_id = f"{prefix}--{recording.id}"
            path = analyses_root / analysis_id / "analysis.json"
            duration, predictions, metadata = load_predictions(
                path,
                analysis_id,
                recording.id,
            )
            if variant_label == LIBSWRESAMPLE_WASM_VARIANT and (
                metadata["method"]
                != "browser-on-device-webcodecs-opencv-libswresample-wasm-v1"
                or metadata["runtimeVariant"] != "libswresample-wasm-v1"
                or metadata["audioResampler"] != "ffmpeg-libswresample-wasm"
                or metadata["modelBundleSha256"] != MODEL_BUNDLE_SHA256
            ):
                raise ValueError(
                    f"libswresample-WASM provenance is invalid in {path}"
                )
            if recording.rallies and recording.rallies[-1].end > duration + 1e-6:
                raise ValueError(
                    f"gold labels exceed inference duration for {recording.id}"
                )
            loaded[recording.id] = RecordingIntervals(
                id=recording.id,
                split=recording.split,
                duration=duration,
                truth=recording.rallies,
                predictions=predictions,
                ignored_intervals=ignored_by_id[recording.id],
            )
            provenance[recording.id] = metadata
        analysis_provenance[variant_label] = provenance
        variant_reports[variant_label] = {
            scope_name: evaluate_scope(
                [loaded[item.id] for item in scope_recordings],
                paddings,
            )
            for scope_name, scope_recordings in scopes.items()
        }
    for recording in manifest.recordings:
        source_hashes = {
            analysis_provenance[variant][recording.id]["sourceContentSha256"]
            for variant in variants
        }
        if len(source_hashes) != 1:
            raise ValueError(
                f"runtime variants reference different source media for {recording.id}"
            )

    subject = (
        "offline-versus-browser-audio-resampler-model-9c92b8e9333f"
        if args.include_libswresample_wasm
        else "offline-versus-browser-on-device-model-9c92b8e9333f"
    )
    report = {
        "schemaVersion": 1,
        "createdAt": datetime.now(UTC).isoformat(),
        "subject": subject,
        "expectedModelSha256": MODEL_SHA256,
        "expectedModelVersion": MODEL_VERSION,
        "expectedModelComponents": MODEL_COMPONENTS,
        "manifest": {
            "path": str(manifest_path),
            "fileSha256": sha256_file(manifest_path),
            "expectedRallies": sum(len(item.rallies) for item in manifest.recordings),
        },
        "modelCorpusManifest": {
            "path": str(model_manifest_path),
            "fileSha256": sha256_file(model_manifest_path),
            "recordingIds": [item.id for item in model_manifest.recordings],
        },
        "labels": {
            "source": (
                "frozen manifest rallies with current label-document ignoredIntervals"
            ),
            "recordings": label_provenance,
        },
        "analysesRoot": str(analyses_root),
        "evaluator": {
            "path": str(Path(__file__).resolve()),
            "fileSha256": sha256_file(Path(__file__).resolve()),
            "sourceFiles": {
                str(path.relative_to(repository_root)): sha256_file(path)
                for path in (
                    repository_root / "analysis/crop_evaluation.py",
                    repository_root / "analysis/metrics.py",
                    repository_root / "analysis/schema.py",
                )
            },
        },
        "analysisArtifacts": analysis_provenance,
        "metricDefinitions": {
            "eventMetricsAtIou05": (
                "Micro-pooled chronological one-to-one rally matches at IoU >= 0.5 "
                "after padding, clipping, merging, and ignored-time subtraction."
            ),
            "coreLiveTimeMetrics": (
                "Duration overlap with unpadded human core rallies; both precision "
                "and recall use the core-human intersection."
            ),
            "F1_padP_coreR": (
                "Harmonic mean of P_pad and R_core. P_pad compares padded model "
                "export with equally padded human export; R_core measures unpadded "
                "human core time recovered."
            ),
            "fullyContainedCoreRallies": (
                "A human core rally counts only when one merged padded export "
                "interval contains it from start through end."
            ),
            "actualExportVersusEvaluable": (
                "Actual export counts and seconds are measured before gold-label ignored "
                "ranges are masked. Event and duration metrics use evaluable fragments "
                "after ignored-time subtraction."
            ),
        },
        "paddingPolicy": {
            "symmetricSecondsBeforeAndAfter": paddings,
            "declaredTargetSecondsBeforeAndAfter": 3.0,
            "secondarySensitivitySecondsBeforeAndAfter": 2.0,
            "selectionScope": "validation2",
            "clipToVideoBounds": True,
            "mergeTouchingOrOverlappingRanges": True,
            "joinPositiveGapsStrictlyUnderSeconds": DEFAULT_JOIN_GAP_SECONDS,
            "subtractIgnoredIntervals": True,
        },
        "scopeRecordingIds": {
            key: [item.id for item in value] for key, value in scopes.items()
        },
        "variants": variant_reports,
        "caveats": [
            (
                "The model was trained without beach footage; beach and all-nine "
                "results are not generalization estimates."
            ),
            (
                "Training and validation scopes are descriptive because those "
                "recordings informed model fitting or selection."
            ),
            (
                "The protected one-video test result is reported only as a post-hoc "
                "regression check and was not used to choose this model or padding."
            ),
            (
                "The existing browser reference uses a non-parity linear audio "
                "resampler and differs from canonical FFmpeg/OpenCV extraction."
            ),
        ],
    }
    if args.include_libswresample_wasm:
        wasm_path = (
            repository_root
            / "vendor/libswresample-wasm/dist/libswresample.wasm"
        )
        glue_path = (
            repository_root
            / "vendor/libswresample-wasm/dist/libswresample.mjs"
        )
        if sha256_file(wasm_path) != LIBSWRESAMPLE_WASM_SHA256:
            raise ValueError("checked-in libswresample WASM has the wrong SHA-256")
        if sha256_file(glue_path) != LIBSWRESAMPLE_GLUE_SHA256:
            raise ValueError("checked-in libswresample loader has the wrong SHA-256")
        report["runtimeAssets"] = {
            "libswresampleWasm": {
                "version": "FFmpeg 7.1.5 / libswresample 5.3.100",
                "path": str(wasm_path),
                "fileSha256": LIBSWRESAMPLE_WASM_SHA256,
            },
            "libswresampleLoader": {
                "path": str(glue_path),
                "fileSha256": LIBSWRESAMPLE_GLUE_SHA256,
            },
        }
        report["liftSummaries"] = build_lift_summaries(
            variant_reports,
            candidate_variant=LIBSWRESAMPLE_WASM_VARIANT,
            reference_variants=("browserOnDevice", "offline"),
        )
        report["caveats"].append(
            (
                "The libswresample-WASM run changes only the browser audio resampling "
                "path; remaining browser-versus-offline visual extraction differences "
                "are still present."
            )
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {"report": str(destination), "sha256": sha256_file(destination)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
