#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from analysis.annotations import load_label_document
from analysis.crop_evaluation import (
    DEFAULT_JOIN_GAP_SECONDS,
    RecordingIntervals,
    evaluate_f1_pad_p_core_r,
    subtract_intervals,
)
from analysis.metrics import (
    aggregate_evaluations,
    aggregate_outcome_slices,
    evaluate_intervals,
    outcome_slice_metrics,
)
from analysis.schema import DatasetManifest, Interval, Recording, load_manifest


SAFE_VARIANT = re.compile(r"^[A-Za-z0-9_-]+$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def discover_model_variants(
    analyses_root: Path,
    recording_ids: list[str],
) -> list[str]:
    common: set[str] | None = None
    for recording_id in recording_ids:
        suffix = f"--{recording_id}"
        variants = {
            path.name[len("model-") : -len(suffix)]
            for path in analyses_root.glob(f"model-*{suffix}")
            if path.is_dir()
            and path.name.startswith("model-")
            and path.name.endswith(suffix)
            and (path / "analysis.json").is_file()
        }
        common = variants if common is None else common & variants
    return sorted(common or set())


def load_ignored_revision(
    labels_root: Path,
    recordings: list[Recording],
) -> tuple[dict[str, tuple[Interval, ...]], dict[str, Any]]:
    ignored_by_id: dict[str, tuple[Interval, ...]] = {}
    provenance: dict[str, Any] = {}
    for recording in recordings:
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
            raise ValueError(f"label document identity differs from manifest for {recording.id}")
        ignored_by_id[recording.id] = document.ignored_intervals
        annotation = document.payload.get("annotation", {})
        provenance[recording.id] = {
            "path": str(path.resolve()),
            "fileSha256": sha256_file(path),
            "annotationStatus": annotation.get("status"),
            "reviewedAt": annotation.get("reviewedAt"),
            "ignoredIntervalCount": len(document.ignored_intervals),
            "ignoredIntervals": [item.to_dict() for item in document.ignored_intervals],
            "ralliesMatchFrozenManifest": document.rallies == recording.rallies,
        }
    return ignored_by_id, provenance


def load_analysis_predictions(
    path: Path,
    expected_analysis_id: str,
    expected_recording_id: str,
) -> tuple[float, list[tuple[Interval, float]], dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read inference output {path}: {error}") from error
    source = payload.get("source")
    analysis = payload.get("analysis")
    rows = payload.get("rallies")
    recorded_id = payload.get("recordingId")
    if (
        payload.get("schemaVersion") != 1
        or payload.get("id") != expected_analysis_id
        or (
            recorded_id is not None
            and recorded_id != expected_recording_id
        )
        or (
            recorded_id is None
            and payload.get("title") != expected_recording_id
        )
    ):
        raise ValueError(f"unexpected analysis identity in {path}")
    if not isinstance(source, dict) or not _number(source.get("duration")):
        raise ValueError(f"analysis source duration is missing in {path}")
    if not isinstance(analysis, dict) or not isinstance(rows, list):
        raise ValueError(f"analysis metadata or rallies are missing in {path}")
    model_sha256 = analysis.get("modelSha256")
    if (
        not isinstance(model_sha256, str)
        or len(model_sha256) != 64
        or any(character not in "0123456789abcdef" for character in model_sha256.lower())
    ):
        raise ValueError(f"analysis model SHA-256 is missing or invalid in {path}")
    duration = float(source["duration"])
    predictions: list[tuple[Interval, float]] = []
    previous_end = -1.0
    excluded_count = 0
    for index, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or not _number(row.get("start"))
            or not _number(row.get("end"))
            or not _number(row.get("confidence"))
        ):
            raise ValueError(f"invalid scored rally {index} in {path}")
        start = float(row["start"])
        end = float(row["end"])
        confidence = float(row["confidence"])
        if (
            not all(math.isfinite(value) for value in (start, end, confidence))
            or start < previous_end
            or start < 0
            or end <= start
            or end > duration + 1e-6
            or not 0 <= confidence <= 1
        ):
            raise ValueError(f"out-of-range or unordered scored rally {index} in {path}")
        included = row.get("included", True)
        if not isinstance(included, bool):
            raise ValueError(f"rally {index} included flag is not boolean in {path}")
        if included:
            predictions.append((Interval(start, min(end, duration)), confidence))
        else:
            excluded_count += 1
        previous_end = end
    metadata = {
        "modelVersion": analysis.get("modelVersion"),
        "variantLabel": analysis.get("variantLabel"),
        "variantDescription": analysis.get("variantDescription"),
        "method": analysis.get("method"),
        "modelArtifactSha256": model_sha256.lower(),
        "analysisFileSha256": sha256_file(path),
        "alreadyExcludedRallyCount": excluded_count,
    }
    return duration, predictions, metadata


def core_guardrails(recordings: list[RecordingIntervals]) -> dict[str, Any]:
    per_recording: list[dict[str, Any]] = []
    slices: list[dict[str, Any]] = []
    for recording in recordings:
        truth = subtract_intervals(recording.truth, recording.ignored_intervals)
        predictions = subtract_intervals(
            recording.predictions,
            recording.ignored_intervals,
        )
        evaluation = evaluate_intervals(truth, predictions)
        per_recording.append(evaluation)
        slices.append(outcome_slice_metrics(truth, predictions))
    aggregate = aggregate_evaluations(per_recording)
    aggregate["outcomeSlices"] = aggregate_outcome_slices(slices)
    return aggregate


def _padding_key(padding: float) -> str:
    return f"pad-{padding:g}s"


def _threshold_key(threshold: float | None) -> str:
    return "unfiltered" if threshold is None else f"score-{threshold:.2f}"


def _metric_delta(current: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
    keys = (
        "P_pad",
        "R_core",
        "F1_padP_coreR",
        "paddedModelExportSeconds",
        "paddedHumanExportSeconds",
        "exportDurationDifferenceSeconds",
    )
    return {key: float(current[key]) - float(reference[key]) for key in keys}


def evaluate_model(
    variant: str,
    analyses_root: Path,
    recordings: list[Recording],
    ignored_by_id: dict[str, tuple[Interval, ...]],
    paddings: list[float],
    thresholds: list[float],
    target_padding: float,
    join_gap_seconds: float,
) -> dict[str, Any]:
    loaded: list[
        tuple[Recording, float, list[tuple[Interval, float]], dict[str, Any]]
    ] = []
    metadata_rows: list[dict[str, Any]] = []
    analysis_hashes: dict[str, str] = {}
    for recording in recordings:
        analysis_id = f"model-{variant}--{recording.id}"
        path = analyses_root / analysis_id / "analysis.json"
        duration, scored, metadata = load_analysis_predictions(
            path,
            analysis_id,
            recording.id,
        )
        if recording.rallies and recording.rallies[-1].end > duration + 1e-6:
            raise ValueError(f"gold labels exceed inference duration for {recording.id}")
        loaded.append((recording, duration, scored, metadata))
        metadata_rows.append(metadata)
        analysis_hashes[recording.id] = metadata["analysisFileSha256"]

    stable_fields = (
        "modelVersion",
        "variantLabel",
        "variantDescription",
        "method",
        "modelArtifactSha256",
    )
    stable_metadata: dict[str, Any] = {}
    for field in stable_fields:
        values = {json.dumps(item[field], sort_keys=True) for item in metadata_rows}
        if len(values) != 1:
            raise ValueError(f"{variant} has inconsistent {field} metadata")
        stable_metadata[field] = metadata_rows[0][field]

    threshold_results: dict[str, Any] = {}
    core_prediction_payload = [
        {
            "recordingId": recording.id,
            "intervals": [[interval.start, interval.end] for interval, _ in scored],
        }
        for recording, _, scored, _ in loaded
    ]
    scored_prediction_payload = [
        {
            "recordingId": recording.id,
            "intervals": [
                [interval.start, interval.end, confidence]
                for interval, confidence in scored
            ],
        }
        for recording, _, scored, _ in loaded
    ]
    all_thresholds: list[float | None] = [None, *thresholds]
    for threshold in all_thresholds:
        evaluated_recordings: list[RecordingIntervals] = []
        available_count = 0
        selected_count = 0
        for recording, duration, scored, _ in loaded:
            available_count += len(scored)
            predictions = tuple(
                interval
                for interval, confidence in scored
                if threshold is None or confidence >= threshold
            )
            selected_count += len(predictions)
            evaluated_recordings.append(
                RecordingIntervals(
                    id=recording.id,
                    split=recording.split,
                    duration=duration,
                    truth=recording.rallies,
                    predictions=predictions,
                    ignored_intervals=ignored_by_id[recording.id],
                )
            )
        padding_rows = evaluate_f1_pad_p_core_r(
            evaluated_recordings,
            paddings,
            join_gap_seconds,
        )
        cases = {_padding_key(float(row["paddingSecondsBeforeAndAfter"])): row for row in padding_rows}
        target = cases[_padding_key(target_padding)]
        for row in cases.values():
            row["deltaFromTargetPaddingSameThreshold"] = _metric_delta(row, target)
        threshold_results[_threshold_key(threshold)] = {
            "minimumConfidenceInclusive": threshold,
            "availableRallyCount": available_count,
            "selectedRallyCount": selected_count,
            "filteredOutRallyCount": available_count - selected_count,
            "selectedRallyRate": selected_count / available_count if available_count else 0.0,
            "coreGuardrails": core_guardrails(evaluated_recordings),
            "paddingCases": cases,
        }

    baseline = threshold_results["unfiltered"]["paddingCases"]
    for result in threshold_results.values():
        for padding_key, row in result["paddingCases"].items():
            row["deltaFromUnfilteredSamePadding"] = _metric_delta(
                row,
                baseline[padding_key],
            )
    return {
        "inferenceVariant": variant,
        **stable_metadata,
        "analysisFileSha256": analysis_hashes,
        "corePredictionSetSha256": sha256_json(core_prediction_payload),
        "scoredPredictionSetSha256": sha256_json(scored_prediction_payload),
        "alreadyExcludedRallyCount": sum(
            item["alreadyExcludedRallyCount"] for item in metadata_rows
        ),
        "scoreSemantics": (
            "The analysis.json rally confidence field is an uncalibrated model score; "
            "thresholds are inclusive and are not probability claims."
        ),
        "thresholds": threshold_results,
    }


def build_rankings(
    models: dict[str, Any],
    paddings: list[float],
    thresholds: list[float],
) -> dict[str, list[dict[str, Any]]]:
    rankings: dict[str, list[dict[str, Any]]] = {}
    for threshold in [None, *thresholds]:
        threshold_key = _threshold_key(threshold)
        for padding in paddings:
            padding_key = _padding_key(padding)
            key = f"{threshold_key}--{padding_key}"
            rows = []
            for variant, model in models.items():
                metric = model["thresholds"][threshold_key]["paddingCases"][padding_key]
                rows.append(
                    {
                        "inferenceVariant": variant,
                        "modelVersion": model["modelVersion"],
                        "variantLabel": model["variantLabel"],
                        "F1_padP_coreR": metric["F1_padP_coreR"],
                        "P_pad": metric["P_pad"],
                        "R_core": metric["R_core"],
                        "paddedModelExportSeconds": metric["paddedModelExportSeconds"],
                        "paddedHumanExportSeconds": metric["paddedHumanExportSeconds"],
                        "exportDurationDifferenceSeconds": metric[
                            "exportDurationDifferenceSeconds"
                        ],
                    }
                )
            rows.sort(
                key=lambda row: (
                    -row["F1_padP_coreR"],
                    -row["R_core"],
                    -row["P_pad"],
                    row["inferenceVariant"],
                )
            )
            for rank, row in enumerate(rows, start=1):
                row["rank"] = rank
            rankings[key] = rows
    return rankings


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    data_root = Path(os.environ.get("VOLLEYCUT_DATA_ROOT", "data")).expanduser().resolve()
    workspace = Path(
        os.environ.get(
            "VOLLEYCUT_LABELING_WORKSPACE",
            data_root / "labeling-v1-2026-08-09",
        )
    ).expanduser().resolve()
    parser = argparse.ArgumentParser(
        description=(
            "Rank complete model inference variants with pooled F1_padP_coreR and "
            "sweep export score thresholds."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=workspace / "manifests" / "full-gold-v1.json",
    )
    parser.add_argument("--labels-root", type=Path, default=workspace / "labels" / "full")
    parser.add_argument("--analyses-root", type=Path, default=data_root / "analyses")
    parser.add_argument("--split", action="append")
    parser.add_argument("--model-version", action="append", dest="model_versions")
    parser.add_argument("--padding-seconds", type=float, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--target-padding-seconds", type=float, default=1.0)
    parser.add_argument(
        "--join-gap-seconds",
        type=float,
        default=DEFAULT_JOIN_GAP_SECONDS,
        help="Join padded export ranges separated by less than this many seconds.",
    )
    parser.add_argument(
        "--score-thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    destination = args.output.expanduser().resolve()
    if destination.exists():
        raise ValueError(f"refusing to overwrite existing report: {destination}")
    paddings = sorted(set(float(value) for value in args.padding_seconds))
    target_padding = float(args.target_padding_seconds)
    join_gap_seconds = float(args.join_gap_seconds)
    if paddings != [0.0, 1.0, 2.0, 3.0]:
        raise ValueError("the canonical report requires padding values 0, 1, 2, and 3")
    if target_padding not in paddings:
        raise ValueError("target padding must be one of the reported padding values")
    if not math.isfinite(join_gap_seconds) or join_gap_seconds < 0:
        raise ValueError("join-gap-seconds must be finite and non-negative")
    thresholds = sorted(set(float(value) for value in args.score_thresholds))
    if thresholds != [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
        raise ValueError(
            "the requested report requires score thresholds .50, .60, .70, .75, "
            ".80, .85, and .90"
        )

    manifest_path = args.manifest.expanduser().resolve()
    labels_root = args.labels_root.expanduser().resolve()
    analyses_root = args.analyses_root.expanduser().resolve()
    manifest: DatasetManifest = load_manifest(manifest_path, require_videos=False)
    splits = list(dict.fromkeys(args.split or ["validation"]))
    recordings = [item for item in manifest.recordings if item.split in splits]
    if not recordings:
        raise ValueError(f"no manifest recordings found for splits {splits}")
    variants = args.model_versions or discover_model_variants(
        analyses_root,
        [item.id for item in recordings],
    )
    if not variants:
        raise ValueError("no complete model inference variants were found")
    if any(not SAFE_VARIANT.fullmatch(variant) for variant in variants):
        raise ValueError("model variants must contain only letters, digits, underscores, or hyphens")
    ignored_by_id, ignored_provenance = load_ignored_revision(labels_root, recordings)

    models = {
        variant: evaluate_model(
            variant,
            analyses_root,
            recordings,
            ignored_by_id,
            paddings,
            thresholds,
            target_padding,
            join_gap_seconds,
        )
        for variant in variants
    }
    rankings = build_rankings(models, paddings, thresholds)
    report = {
        "schemaVersion": 1,
        "metric": "F1_padP_coreR",
        "createdAt": datetime.now(UTC).isoformat(),
        "evaluator": {
            "path": str(Path(__file__).resolve()),
            "fileSha256": sha256_file(Path(__file__).resolve()),
            "sourceFiles": {
                str(path.relative_to(repository_root)): sha256_file(path)
                for path in (
                    repository_root / "analysis" / "crop_evaluation.py",
                    repository_root / "analysis" / "metrics.py",
                    repository_root / "analysis" / "schema.py",
                )
            },
        },
        "selectionScope": {
            "splits": splits,
            "recordingIds": [item.id for item in recordings],
            "sourceGroups": sorted({item.source_group for item in recordings}),
            "protectedTestUsed": "test" in splits,
        },
        "manifest": {
            "path": str(manifest_path),
            "fileSha256": sha256_file(manifest_path),
            "ralliesSource": "frozen manifest",
        },
        "ignoredIntervalRevision": {
            "source": "current human label documents (drafts allowed), overlaid on frozen manifest rallies",
            "labelsRoot": str(labels_root),
            "recordings": ignored_provenance,
        },
        "analysesRoot": str(analyses_root),
        "modelCount": len(models),
        "modelVersions": variants,
        "targetProductPaddingSecondsBeforeAndAfter": target_padding,
        "joinGapSeconds": join_gap_seconds,
        "requiredPaddingSecondsBeforeAndAfter": paddings,
        "scoreThresholdsInclusive": thresholds,
        "scoreThresholdCaveat": (
            "Rally confidence values are model-specific uncalibrated scores. A .80 cutoff "
            "does not mean an 80% probability that a rally is correct."
        ),
        "rankingPolicy": (
            "Primary model ranking is the unfiltered target-product-padding result. "
            "Threshold and padding sweeps are validation-only sensitivity analyses."
        ),
        "models": models,
        "rankings": rankings,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    primary_key = f"unfiltered--{_padding_key(target_padding)}"
    print(
        json.dumps(
            {
                "report": str(destination),
                "models": len(models),
                "scope": splits,
                "primaryRanking": rankings[primary_key],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
