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


VARIANTS = {
    "offline": "model-9c92b8e9333f",
    "browserOnDevice": "model-browser-on-device-9c92b8e9333f",
}
MODEL_SHA256 = "9c92b8e9333f6247336639409acbe063da68dea4cc74735c7b2a8791f8dda2a7"
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
        "modelBundleSha256": analysis.get("modelBundleSha256"),
        "sourceContentSha256": source_sha256.lower(),
        "sourceFilename": source.get("filename"),
        "sourceDuration": duration,
        "method": analysis.get("method"),
        "variantLabel": analysis.get("variantLabel"),
        "includedRallies": len(predictions),
        "excludedRallies": excluded,
    }


def harmonic_mean(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def interval_duration(intervals: tuple[Interval, ...]) -> float:
    return sum(item.end - item.start for item in intervals)


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
                    "expectedCoreRallies": len(truth),
                    "fullyContainedCoreRallies": fully_contained,
                    "eventTruePositivesAtIou05": event["matchedRallies"],
                    "actualMergedExportSections": len(actual_export),
                    "actualPaddedExportSeconds": interval_duration(actual_export),
                    "evaluableExportFragments": len(export),
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
        adjusted = adjusted_rows[padding]
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
            },
            "fullyContainedCoreRallies": contained_total,
            "expectedCoreRallies": aggregate["trueRallies"],
            "inputPredictionCount": adjusted["inputCropCount"],
            "actualMergedExportSections": actual_export_sections,
            "actualPaddedExportSeconds": actual_export_seconds,
            "evaluableExportFragments": aggregate["predictedRallies"],
            "evaluablePaddedModelSeconds": adjusted["paddedModelExportSeconds"],
            "evaluablePaddedHumanSeconds": adjusted["paddedHumanExportSeconds"],
            "perRecording": per_recording,
        }
    return results


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    data_root = Path(os.environ.get("VOLLEYCUT_DATA_ROOT", "/mnt/freenas/volleycut"))
    gold_workspace = data_root / "labeling-v1-2026-08-09"
    model_workspace = data_root / "labeling-v1-2026-08-09-no-beach-2026-08-12"
    analyses_root = model_workspace / "analyses"
    parser = argparse.ArgumentParser(
        description="Compare offline and browser model-9c92 export padding."
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
    parser.add_argument("--padding-seconds", type=float, nargs="+", default=[2, 3])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    destination = args.output.expanduser().resolve()
    if destination.exists():
        raise ValueError(f"refusing to overwrite existing report: {destination}")
    paddings = sorted(set(float(value) for value in args.padding_seconds))
    if not paddings or paddings[0] < 0:
        raise ValueError("padding values must be non-negative")
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
    for variant_label, prefix in VARIANTS.items():
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
            for variant in VARIANTS
        }
        if len(source_hashes) != 1:
            raise ValueError(
                f"runtime variants reference different source media for {recording.id}"
            )

    report = {
        "schemaVersion": 1,
        "createdAt": datetime.now(UTC).isoformat(),
        "subject": "offline-versus-browser-on-device-model-9c92b8e9333f",
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
            "clipToVideoBounds": True,
            "mergeTouchingOrOverlappingRanges": True,
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
                "The browser path uses a non-parity linear audio resampler and "
                "currently differs from canonical FFmpeg/OpenCV feature extraction."
            ),
        ],
    }
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
