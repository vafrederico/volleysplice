#!/usr/bin/env python3
"""Evaluate every deployed classifier head independently.

The report deliberately keeps the protected test closed.  It evaluates the two
three-head production bundles and the suppression specialist on the three declared
development recordings and the six file-level held-out feedback recordings.

Raw sample metrics answer whether a head's score can be used as a binary criterion.
Operational metrics additionally apply the production decoder that belongs to that
head.  Targets and evaluation universes differ by role and are recorded explicitly;
metrics from unlike heads must not be ranked against each other.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import runpy
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.artifacts import atomic_write_text
from analysis.crop_evaluation import (
    _duration,
    _intersection_duration,
    subtract_intervals,
)
from analysis.dead_state import DeadStateDecoderConfig
from analysis.dead_state_experiment import (
    DeadStateRefinementConfig,
    END_TRANSITION_TARGET_MODE,
    _decode_near_end,
    _eligible_target_mask,
    _target_for_mode,
)
from analysis.decoder import decode_probabilities
from analysis.model import LogisticModel, load_model
from analysis.pipeline import PreparedRecording, prepare_recording
from analysis.schema import Interval, labels_for_times
from analysis.serve import (
    ServeDecoderConfig,
    decode_serve_probabilities,
    match_serve_contacts,
    serve_labels_for_times,
)


TRAIN_FEEDBACK_SCRIPT = REPO_ROOT / "scripts" / "train-feedback-suppression-v3.py"
EXPERIMENT_ROOT = Path(
    "/mnt/freenas/volleycut/intake-2026-08-13/experiments/"
    "feedback-suppression-v3-2026-08-16"
)
SPLIT_POLICY = EXPERIMENT_ROOT / "split-policy.json"
FEATURE_CACHE = Path(
    "/mnt/freenas/volleycut/intake-2026-08-13/experiments/"
    "environment-specialists-v2/features/audiovisual-audio-normalized-v3"
)
LABEL_ROOT = Path("/mnt/freenas/volleycut/intake-2026-08-13/labels/full")
FEEDBACK_ROOT = Path("/mnt/freenas/volleycut/model-feedback")
V2_BUNDLE = Path(
    "/mnt/freenas/volleycut/intake-2026-08-13/experiments/"
    "environment-specialists-v2/models/all-labels/bundle.json"
)
OLD_MODEL_ROOT = Path(
    "/mnt/freenas/volleycut/labeling-v1-2026-08-09-no-beach-2026-08-12/models"
)
OLD_HEAD_PATHS = {
    "rally": OLD_MODEL_ROOT / "full-audiovisual-audio-normalized-v3",
    "serve": OLD_MODEL_ROOT / "serve-specialist-audio-normalized-v5",
    "deadState": OLD_MODEL_ROOT
    / "dead-state-transition-audio-normalized-v5-no-legacy-final",
}
SUPPRESSION_PATH = EXPERIMENT_ROOT / "models/suppression-overlap-exclusion-retrained"
DEFAULT_JSON = (
    REPO_ROOT
    / "docs/research/production-head-independent-evaluation-2026-08-20.json"
)
DEFAULT_MARKDOWN = (
    REPO_ROOT
    / "docs/research/production-head-independent-evaluation-2026-08-20.md"
)
DEVELOPMENT_IDS = (
    "grass-source-06",
    "grass-source-08",
    "indoor-source-03",
)
THRESHOLD_GRID = tuple(round(value, 2) for value in np.arange(0.05, 1.0, 0.05))


@dataclass(frozen=True)
class EvaluationItem:
    prepared: PreparedRecording
    partition: str
    source_path: Path
    boundary_reference: str


@dataclass(frozen=True)
class ModelHeads:
    rally: LogisticModel
    serve: LogisticModel
    dead_state: LogisticModel


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def ratio(numerator: int | float, denominator: int | float, *, empty: float = 0.0) -> float:
    return float(numerator / denominator) if denominator else empty


def f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def binary_counts(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("binary labels and scores must be aligned vectors")
    positive = labels > 0.5
    predicted = scores >= threshold
    tp = int(np.sum(positive & predicted))
    fp = int(np.sum(~positive & predicted))
    fn = int(np.sum(positive & ~predicted))
    tn = int(np.sum(~positive & ~predicted))
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn, empty=1.0)
    return {
        "threshold": threshold,
        "samples": len(labels),
        "positiveSamples": int(np.sum(positive)),
        "predictedPositiveSamples": int(np.sum(predicted)),
        "truePositive": tp,
        "falsePositive": fp,
        "falseNegative": fn,
        "trueNegative": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1(precision, recall),
        "falsePositiveRate": ratio(fp, fp + tn),
        "specificity": ratio(tn, tn + fp, empty=1.0),
    }


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive = labels > 0.5
    count = int(np.sum(positive))
    if count == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    ranked = positive[order]
    precision_at_rank = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision_at_rank * ranked) / count)


def threshold_profile(
    labels: np.ndarray,
    scores: np.ndarray,
    operational_threshold: float,
) -> dict[str, Any]:
    thresholds = sorted({*THRESHOLD_GRID, float(operational_threshold)})
    rows = [binary_counts(labels, scores, threshold) for threshold in thresholds]
    recall_floors: dict[str, Any] = {}
    for floor in (1.0, 0.99, 0.95, 0.90):
        eligible = [row for row in rows if row["recall"] + 1e-12 >= floor]
        recall_floors[f"{floor:.2f}"] = max(
            eligible,
            key=lambda row: (row["threshold"], row["precision"]),
            default=None,
        )
    return {
        "operational": binary_counts(labels, scores, operational_threshold),
        "averagePrecision": average_precision(labels, scores),
        "recallFloorOperatingPoints": recall_floors,
        "thresholds": rows,
    }


def pooled_arrays(
    rows: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([labels for labels, _ in rows]).astype(np.float32),
        np.concatenate([scores for _, scores in rows]).astype(np.float32),
    )


def sample_mask_from_intervals(times: np.ndarray, intervals: Sequence[Interval]) -> np.ndarray:
    return labels_for_times(times, intervals) > 0.5


def rally_operational_metrics(
    items: Sequence[EvaluationItem],
    model: LogisticModel,
    probabilities: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    intersection = predicted_seconds = truth_seconds = 0.0
    per_recording: list[dict[str, Any]] = []
    for entry in items:
        item = entry.prepared
        fps = (
            1.0 / float(np.median(np.diff(item.sequence.times)))
            if len(item.sequence.times) > 1
            else model.feature_config.analysis_fps
        )
        decoded, _ = decode_probabilities(
            item.sequence.times,
            probabilities[item.recording.id],
            item.sequence.metadata.duration,
            model.decoder,
            fps,
        )
        predicted = subtract_intervals(
            tuple(Interval(float(row.start), float(row.end)) for row in decoded),
            item.recording.ignored_intervals,
        )
        truth = subtract_intervals(item.recording.rallies, item.recording.ignored_intervals)
        local_intersection = _intersection_duration(predicted, truth)
        local_predicted = _duration(predicted)
        local_truth = _duration(truth)
        intersection += local_intersection
        predicted_seconds += local_predicted
        truth_seconds += local_truth
        precision = ratio(local_intersection, local_predicted)
        recall = ratio(local_intersection, local_truth, empty=1.0)
        per_recording.append(
            {
                "id": item.recording.id,
                "intersectionSeconds": local_intersection,
                "predictedSeconds": local_predicted,
                "truthSeconds": local_truth,
                "precision": precision,
                "recall": recall,
                "f1": f1(precision, recall),
            }
        )
    precision = ratio(intersection, predicted_seconds)
    recall = ratio(intersection, truth_seconds, empty=1.0)
    return {
        "metric": "decoded unpadded live-time overlap",
        "decoder": model.decoder.to_dict(),
        "intersectionSeconds": intersection,
        "predictedSeconds": predicted_seconds,
        "truthSeconds": truth_seconds,
        "precision": precision,
        "recall": recall,
        "f1": f1(precision, recall),
        "perRecording": per_recording,
    }


def serve_event_metrics(
    items: Sequence[EvaluationItem],
    model: LogisticModel,
    probabilities: Mapping[str, np.ndarray],
    *,
    threshold: float,
    tolerance: float,
) -> dict[str, Any]:
    config = replace(
        ServeDecoderConfig.from_dict(model.training_summary["serveDecoder"]),
        threshold=threshold,
    )
    truth_count = predicted_count = matched_count = 0
    errors: list[float] = []
    per_recording: list[dict[str, Any]] = []
    for entry in items:
        item = entry.prepared
        detections = decode_serve_probabilities(
            item.sequence.times,
            probabilities[item.recording.id],
            config,
            duration=item.sequence.metadata.duration,
        )
        detections = [
            detection
            for detection in detections
            if not any(
                ignored.start <= detection.time < ignored.end
                for ignored in item.recording.ignored_intervals
            )
        ]
        truth = [float(rally.start) for rally in item.recording.rallies]
        matches = match_serve_contacts(truth, detections, tolerance)
        truth_count += len(truth)
        predicted_count += len(detections)
        matched_count += len(matches)
        errors.extend(error for _, _, error in matches)
        precision = ratio(len(matches), len(detections))
        recall = ratio(len(matches), len(truth), empty=1.0)
        per_recording.append(
            {
                "id": item.recording.id,
                "trueServes": len(truth),
                "predictedServes": len(detections),
                "matchedServes": len(matches),
                "precision": precision,
                "recall": recall,
                "f1": f1(precision, recall),
            }
        )
    precision = ratio(matched_count, predicted_count)
    recall = ratio(matched_count, truth_count, empty=1.0)
    absolute = np.abs(np.asarray(errors, dtype=np.float64))
    return {
        "metric": "decoded serve-contact events",
        "threshold": threshold,
        "toleranceSeconds": tolerance,
        "trueServes": truth_count,
        "predictedServes": predicted_count,
        "matchedServes": matched_count,
        "precision": precision,
        "recall": recall,
        "f1": f1(precision, recall),
        "timingMaeSeconds": float(np.mean(absolute)) if len(absolute) else None,
        "timingP90Seconds": float(np.percentile(absolute, 90)) if len(absolute) else None,
        "perRecording": per_recording,
    }


def serve_threshold_profile(
    items: Sequence[EvaluationItem],
    model: LogisticModel,
    probabilities: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    operational = float(model.training_summary["serveDecoder"]["threshold"])
    thresholds = sorted({*THRESHOLD_GRID, operational})
    full_rows = [
        serve_event_metrics(
            items,
            model,
            probabilities,
            threshold=threshold,
            tolerance=1.0,
        )
        for threshold in thresholds
    ]
    rows = [
        {key: value for key, value in row.items() if key != "perRecording"}
        for row in full_rows
    ]
    floors: dict[str, Any] = {}
    for floor in (1.0, 0.99, 0.95, 0.90):
        eligible = [row for row in rows if row["recall"] + 1e-12 >= floor]
        floors[f"{floor:.2f}"] = max(
            eligible,
            key=lambda row: (row["threshold"], row["precision"]),
            default=None,
        )
    return {
        "operational": next(
            row for row in full_rows if row["threshold"] == operational
        ),
        "toleranceSensitivity": {
            str(tolerance): serve_event_metrics(
                items,
                model,
                probabilities,
                threshold=operational,
                tolerance=tolerance,
            )
            for tolerance in (0.5, 1.0, 2.0)
        },
        "recallFloorOperatingPoints": floors,
        "thresholds": rows,
    }


def combined_serve_path_metrics(
    items: Sequence[EvaluationItem],
    heads: Mapping[str, ModelHeads],
    probabilities: Mapping[str, Mapping[str, np.ndarray]],
    tolerance: float,
) -> dict[str, Any]:
    model_ids = ("model-1ca43e38eefc", "model-9c92b8e9333f")
    truth_count = alert_count = matched_alert_count = covered_truth_count = 0
    coverage = {"both": 0, "allLabelsV2Only": 0, "previousProductionOnly": 0, "neither": 0}
    per_recording: list[dict[str, Any]] = []
    for entry in items:
        item = entry.prepared
        detections_by_model: dict[str, list[Any]] = {}
        for model_id in model_ids:
            model = heads[model_id].serve
            config = ServeDecoderConfig.from_dict(
                model.training_summary["serveDecoder"]
            )
            detections_by_model[model_id] = [
                detection
                for detection in decode_serve_probabilities(
                    item.sequence.times,
                    probabilities[f"{model_id}:serve"][item.recording.id],
                    config,
                    duration=item.sequence.metadata.duration,
                )
                if not any(
                    ignored.start <= detection.time < ignored.end
                    for ignored in item.recording.ignored_intervals
                )
            ]
        truth = [float(rally.start) for rally in item.recording.rallies]
        local_coverage = {
            "both": 0,
            "allLabelsV2Only": 0,
            "previousProductionOnly": 0,
            "neither": 0,
        }
        for contact in truth:
            v2 = any(
                abs(detection.time - contact) <= tolerance + 1e-12
                for detection in detections_by_model[model_ids[0]]
            )
            old = any(
                abs(detection.time - contact) <= tolerance + 1e-12
                for detection in detections_by_model[model_ids[1]]
            )
            key = (
                "both"
                if v2 and old
                else "allLabelsV2Only"
                if v2
                else "previousProductionOnly"
                if old
                else "neither"
            )
            local_coverage[key] += 1
            coverage[key] += 1
        alerts = [
            detection
            for model_id in model_ids
            for detection in detections_by_model[model_id]
        ]
        local_matched_alerts = sum(
            any(abs(detection.time - contact) <= tolerance + 1e-12 for contact in truth)
            for detection in alerts
        )
        local_covered = len(truth) - local_coverage["neither"]
        truth_count += len(truth)
        alert_count += len(alerts)
        matched_alert_count += local_matched_alerts
        covered_truth_count += local_covered
        per_recording.append(
            {
                "id": item.recording.id,
                "trueServes": len(truth),
                "allLabelsV2Alerts": len(detections_by_model[model_ids[0]]),
                "previousProductionAlerts": len(detections_by_model[model_ids[1]]),
                "matchedPathAlerts": local_matched_alerts,
                "coveredServes": local_covered,
                "precision": ratio(local_matched_alerts, len(alerts)),
                "recall": ratio(local_covered, len(truth), empty=1.0),
                "coverageAttribution": local_coverage,
            }
        )
    precision = ratio(matched_alert_count, alert_count)
    recall = ratio(covered_truth_count, truth_count, empty=1.0)
    return {
        "metric": "parallel production serve paths",
        "toleranceSeconds": tolerance,
        "trueServes": truth_count,
        "pathAlerts": alert_count,
        "matchedPathAlerts": matched_alert_count,
        "coveredServes": covered_truth_count,
        "precision": precision,
        "recall": recall,
        "f1": f1(precision, recall),
        "coverageAttribution": coverage,
        "perRecording": per_recording,
    }


def dead_oracle_coverage(
    items: Sequence[EvaluationItem],
    model: LogisticModel,
    probabilities: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    decoder = DeadStateDecoderConfig.from_dict(
        model.training_summary["selectedDeadStateDecoder"]
    )
    refinement = DeadStateRefinementConfig.from_dict(
        model.training_summary["selectedRefinement"]
    )
    truth_count = detection_count = 0
    errors: list[float] = []
    per_recording: list[dict[str, Any]] = []
    for entry in items:
        item = entry.prepared
        local_detections = 0
        local_errors: list[float] = []
        for index, rally in enumerate(item.recording.rallies):
            next_start = (
                float(item.recording.rallies[index + 1].start)
                if index + 1 < len(item.recording.rallies)
                else None
            )
            detection = _decode_near_end(
                item.sequence.times,
                probabilities[item.recording.id],
                float(rally.end),
                next_start,
                decoder,
                refinement,
                item.sequence.metadata.duration,
            )
            if detection is not None:
                local_detections += 1
                local_errors.append(float(detection.time - rally.end))
        truth_count += len(item.recording.rallies)
        detection_count += local_detections
        errors.extend(local_errors)
        per_recording.append(
            {
                "id": item.recording.id,
                "trueEnds": len(item.recording.rallies),
                "detectedTransitions": local_detections,
                "coverageRecall": ratio(
                    local_detections, len(item.recording.rallies), empty=1.0
                ),
                "within0.5SecondsRecall": ratio(
                    sum(abs(error) <= 0.5 + 1e-12 for error in local_errors),
                    len(item.recording.rallies),
                    empty=1.0,
                ),
            }
        )
    absolute = np.abs(np.asarray(errors, dtype=np.float64))
    return {
        "metric": "oracle-end-anchored transition coverage",
        "note": (
            "Each human end supplies the local anchor required by this refinement head; "
            "this measures end evidence coverage, not standalone global precision."
        ),
        "decoder": decoder.to_dict(),
        "refinement": refinement.to_dict(),
        "trueEnds": truth_count,
        "detectedTransitions": detection_count,
        "coverageRecall": ratio(detection_count, truth_count, empty=1.0),
        "within0.5SecondsRecall": ratio(
            int(np.sum(absolute <= 0.5 + 1e-12)), truth_count, empty=1.0
        ),
        "timingMaeSeconds": float(np.mean(absolute)) if len(absolute) else None,
        "timingP90Seconds": float(np.percentile(absolute, 90)) if len(absolute) else None,
        "perRecording": per_recording,
    }


def intervals_from_decoder(
    item: PreparedRecording,
    scores: np.ndarray,
    model: LogisticModel,
    decoder: Any,
) -> tuple[Interval, ...]:
    fps = (
        1.0 / float(np.median(np.diff(item.sequence.times)))
        if len(item.sequence.times) > 1
        else model.feature_config.analysis_fps
    )
    decoded, _ = decode_probabilities(
        item.sequence.times,
        scores,
        item.sequence.metadata.duration,
        decoder,
        fps,
    )
    return tuple(Interval(float(row.start), float(row.end)) for row in decoded)


def feedback_exclusion_mask(item: PreparedRecording, helpers: Mapping[str, Any]) -> np.ndarray:
    if not item.recording.raw.get("feedbackPath"):
        return np.zeros(len(item.sequence.times), dtype=bool)
    false_positives = helpers["as_intervals"](
        item.recording.raw.get("hardNegatives", [])
    )
    excluded = tuple(
        false_positive
        for false_positive in false_positives
        if any(
            false_positive.start < rally.end and rally.start < false_positive.end
            for rally in item.recording.rallies
        )
    )
    return sample_mask_from_intervals(item.sequence.times, excluded)


def suppression_targets(
    entry: EvaluationItem,
    baseline: Sequence[Interval],
    helpers: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    item = entry.prepared
    truth = item.labels > 0.5
    selected_by_production = sample_mask_from_intervals(item.sequence.times, baseline)
    explicit = helpers["hard_negative_mask"](item)
    positive = (
        item.sample_mask
        & ~truth
        & (selected_by_production | explicit)
        & ~feedback_exclusion_mask(item, helpers)
    )
    negative = item.sample_mask & truth
    universe = positive | negative
    return positive[universe].astype(np.float32), universe, positive


def suppression_operational_metrics(
    items: Sequence[EvaluationItem],
    model: LogisticModel,
    probabilities: Mapping[str, np.ndarray],
    targets: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    decoder: Any,
) -> dict[str, Any]:
    rows: list[tuple[np.ndarray, np.ndarray]] = []
    per_recording: list[dict[str, Any]] = []
    for entry in items:
        item = entry.prepared
        labels, universe, _ = targets[item.recording.id]
        intervals = intervals_from_decoder(
            item, probabilities[item.recording.id], model, decoder
        )
        decoded = sample_mask_from_intervals(item.sequence.times, intervals)[universe]
        metric = binary_counts(labels, decoded.astype(np.float32), 0.5)
        metric["id"] = item.recording.id
        per_recording.append(metric)
        rows.append((labels, decoded.astype(np.float32)))
    labels, decoded_scores = pooled_arrays(rows)
    metric = binary_counts(labels, decoded_scores, 0.5)
    return {
        "metric": "production-decoded suppression samples in selected target universe",
        "decoder": decoder.to_dict(),
        **metric,
        "rallySampleSurvival": metric["specificity"],
        "perRecording": per_recording,
    }


def load_items(helpers: Mapping[str, Any], feature_config: Any) -> list[EvaluationItem]:
    items: list[EvaluationItem] = []
    for recording_id in DEVELOPMENT_IDS:
        path = LABEL_ROOT / f"{recording_id}.labels.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        annotation = payload.get("annotation", {})
        boundary_reference = (
            "blind-ai-unvalidated"
            if "unvalidated" in str(annotation.get("annotator", "")).lower()
            else "human-reviewed-label-document-in-progress"
        )
        recording = helpers["record_from_label"](path)
        items.append(
            EvaluationItem(
                prepare_recording(recording, feature_config, FEATURE_CACHE),
                "development",
                path,
                boundary_reference,
            )
        )
    policy = json.loads(SPLIT_POLICY.read_text(encoding="utf-8"))
    for project_id in policy["holdoutProjectIds"]:
        matches = sorted(FEEDBACK_ROOT.glob(f"{project_id}-*/bundle.json"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected one held-feedback bundle for {project_id}, found {len(matches)}"
            )
        prepared, _ = helpers["feedback_prepared"](matches[0], feature_config)
        items.append(
            EvaluationItem(
                prepared,
                "held-out-feedback",
                matches[0],
                "model-origin-feedback-boundaries-not-user-adjusted",
            )
        )
    return items


def load_production_heads() -> dict[str, ModelHeads]:
    bundle = json.loads(V2_BUNDLE.read_text(encoding="utf-8"))["heads"]
    return {
        "model-1ca43e38eefc": ModelHeads(
            load_model(bundle["rally"]["path"]),
            load_model(bundle["serve"]["path"]),
            load_model(bundle["deadState"]["path"]),
        ),
        "model-9c92b8e9333f": ModelHeads(
            load_model(OLD_HEAD_PATHS["rally"]),
            load_model(OLD_HEAD_PATHS["serve"]),
            load_model(OLD_HEAD_PATHS["deadState"]),
        ),
    }


def head_metadata(model: LogisticModel) -> dict[str, Any]:
    return {
        "artifactSha256": model.artifact_sha256,
        "predictionTask": model.prediction_task,
        "featureVersion": model.feature_version,
        "featureCount": len(model.feature_names),
        "trainingRecordingIds": model.training_summary.get("trainingRecordingIds", []),
    }


def evaluation_for_scope(
    items: Sequence[EvaluationItem],
    heads: Mapping[str, ModelHeads],
    suppression: LogisticModel,
    probabilities: Mapping[str, Mapping[str, np.ndarray]],
    suppression_probabilities: Mapping[str, np.ndarray],
    suppression_target_rows: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    suppression_decoder: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {"recordings": [entry.prepared.recording.id for entry in items]}
    for model_id, bundle in heads.items():
        model_result: dict[str, Any] = {}
        for role, model, threshold, target_builder in (
            (
                "rally",
                bundle.rally,
                float(bundle.rally.decoder.enter_threshold),
                lambda item: item.labels,
            ),
            (
                "serve",
                bundle.serve,
                float(bundle.serve.training_summary["serveDecoder"]["threshold"]),
                lambda item, radius=float(
                    bundle.serve.training_summary["serveTarget"]["radiusSeconds"]
                ): serve_labels_for_times(
                    item.sequence.times, item.recording.rallies, radius
                ),
            ),
        ):
            sample_rows = []
            per_recording_raw = []
            for entry in items:
                item = entry.prepared
                eligible = item.sample_mask
                local_labels = target_builder(item)[eligible]
                local_scores = probabilities[f"{model_id}:{role}"][item.recording.id][
                    eligible
                ]
                sample_rows.append((local_labels, local_scores))
                per_recording_raw.append(
                    {
                        "id": item.recording.id,
                        **binary_counts(local_labels, local_scores, threshold),
                    }
                )
            labels, scores = pooled_arrays(sample_rows)
            raw_profile = threshold_profile(labels, scores, threshold)
            raw_profile["perRecordingOperational"] = per_recording_raw
            model_result[role] = {
                "metadata": head_metadata(model),
                "target": (
                    "valid rally-live samples"
                    if role == "rally"
                    else "valid samples within ±1.0s of human serve contact"
                ),
                "rawSample": raw_profile,
            }
        dead_rows = []
        dead_per_recording = []
        dead_target = bundle.dead_state.training_summary["deadStateTarget"]
        for entry in items:
            item = entry.prepared
            labels, target_mask = _target_for_mode(
                item.sequence.times,
                item.recording.rallies,
                target_mode=END_TRANSITION_TARGET_MODE,
                before_end_seconds=float(dead_target["beforeEndSeconds"]),
                after_end_seconds=float(dead_target["afterEndSeconds"]),
                pre_serve_setup_seconds=float(dead_target["preServeSetupSeconds"]),
            )
            eligible = _eligible_target_mask(item.sample_mask, target_mask)
            local_labels = labels[eligible]
            local_scores = probabilities[f"{model_id}:deadState"][item.recording.id][
                eligible
            ]
            dead_rows.append((local_labels, local_scores))
            dead_per_recording.append(
                {
                    "id": item.recording.id,
                    **binary_counts(
                        local_labels,
                        local_scores,
                        float(
                            bundle.dead_state.training_summary[
                                "selectedDeadStateDecoder"
                            ]["deadThreshold"]
                        ),
                    ),
                }
            )
        dead_labels, dead_scores = pooled_arrays(dead_rows)
        dead_threshold = float(
            bundle.dead_state.training_summary["selectedDeadStateDecoder"][
                "deadThreshold"
            ]
        )
        dead_raw_profile = threshold_profile(dead_labels, dead_scores, dead_threshold)
        dead_raw_profile["perRecordingOperational"] = dead_per_recording
        model_result["deadState"] = {
            "metadata": head_metadata(bundle.dead_state),
            "target": (
                "valid local end-transition samples: post-end positive; pre-end and "
                "pre-serve controls negative"
            ),
            "rawSample": dead_raw_profile,
            "oracleAnchoredCoverage": dead_oracle_coverage(
                items,
                bundle.dead_state,
                probabilities[f"{model_id}:deadState"],
            ),
        }
        model_result["rally"]["operational"] = rally_operational_metrics(
            items, bundle.rally, probabilities[f"{model_id}:rally"]
        )
        model_result["serve"]["operationalEvent"] = serve_threshold_profile(
            items, bundle.serve, probabilities[f"{model_id}:serve"]
        )
        result[model_id] = model_result

    suppression_rows = []
    suppression_per_recording = []
    for entry in items:
        item = entry.prepared
        labels, universe, _ = suppression_target_rows[item.recording.id]
        local_scores = suppression_probabilities[item.recording.id][universe]
        suppression_rows.append((labels, local_scores))
        suppression_per_recording.append(
            {
                "id": item.recording.id,
                **binary_counts(
                    labels, local_scores, float(suppression_decoder.enter_threshold)
                ),
            }
        )
    suppression_labels, suppression_scores = pooled_arrays(suppression_rows)
    suppression_raw_profile = threshold_profile(
        suppression_labels,
        suppression_scores,
        float(suppression_decoder.enter_threshold),
    )
    suppression_raw_profile["perRecordingOperational"] = suppression_per_recording
    result["suppression-overlap-exclusion-retrained"] = {
        "metadata": head_metadata(suppression),
        "target": (
            "within the selected target universe: production-selected or explicit "
            "valid non-rally samples positive; human rally samples negative"
        ),
        "rawSample": suppression_raw_profile,
        "operational": suppression_operational_metrics(
            items,
            suppression,
            suppression_probabilities,
            suppression_target_rows,
            suppression_decoder,
        ),
    }
    result["combinedServeParallelPaths"] = {
        "note": (
            "Production does not create one merged serve detector. Each serve head is "
            "decoded inside its own three-head model path, then the resulting rally "
            "intervals are unioned. These diagnostics preserve both path alerts, count "
            "duplicate alerts in precision, and count each reference serve once in recall."
        ),
        "toleranceSensitivity": {
            str(tolerance): combined_serve_path_metrics(
                items, heads, probabilities, tolerance
            )
            for tolerance in (0.5, 1.0, 2.0)
        },
    }
    return result


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def headline_rows(report: Mapping[str, Any], scope: str) -> list[str]:
    data = report["scopes"][scope]
    rows: list[str] = []
    labels = {
        "model-1ca43e38eefc": "All-labels v2",
        "model-9c92b8e9333f": "Previous production",
    }
    for model_id in ("model-1ca43e38eefc", "model-9c92b8e9333f"):
        for role in ("rally", "serve", "deadState"):
            raw = data[model_id][role]["rawSample"]["operational"]
            rows.append(
                f"| {labels[model_id]} | {role} | {raw['threshold']:.2f} | "
                f"{raw['positiveSamples']} samples | {raw['predictedPositiveSamples']} samples | "
                f"{raw['precision']:.4f} | {raw['recall']:.4f} | {raw['f1']:.4f} |"
            )
    raw = data["suppression-overlap-exclusion-retrained"]["rawSample"]["operational"]
    rows.append(
        f"| Suppression | suppression | {raw['threshold']:.2f} | "
        f"{raw['positiveSamples']} samples | {raw['predictedPositiveSamples']} samples | "
        f"{raw['precision']:.4f} | {raw['recall']:.4f} | {raw['f1']:.4f} |"
    )
    return rows


def markdown_report(report: Mapping[str, Any]) -> str:
    held = report["scopes"]["held-out-feedback"]
    boundary = report["scopes"]["development-human-reviewed"]
    v2_serve = boundary["model-1ca43e38eefc"]["serve"]["operationalEvent"]
    old_serve = boundary["model-9c92b8e9333f"]["serve"]["operationalEvent"]
    lines = [
        "# Independent precision/recall of production model heads",
        "",
        f"Generated: `{report['generatedAt']}`.",
        "",
        "## Outcome",
        "",
        "No production head should yet be treated as a universally complete standalone "
        "gate. The serve heads are the closest match for a region proposal criterion, but "
        f"their decoded 1-second recall on the two human-reviewed development recordings is "
        f"only {v2_serve['operational']['recall']:.1%} (all-labels v2) and "
        f"{old_serve['operational']['recall']:.1%} (previous production). Even a 2-second "
        f"matching region reaches {v2_serve['toleranceSensitivity']['2.0']['recall']:.1%} "
        f"and {old_serve['toleranceSensitivity']['2.0']['recall']:.1%}, not 100%. The "
        "dead-state head is explicitly local and needs an existing candidate end anchor. "
        "The suppression head is a veto specialist with a restricted evaluation universe, "
        "not a general dead-time detector.",
        "",
        "The protected test (`indoor-source-05`) was not opened. There is currently "
        "no fully held-out, completed human boundary-gold scope available without opening it. "
        "The two human-reviewed development documents remain `in-progress` and share source "
        "groups with model development, so serve/dead results are provisional. Held-out "
        "feedback is valid for reviewed inclusion/veto analysis, but all retained boundaries "
        "are untouched model-origin ranges and are not boundary-gold evidence.",
        "",
        "## Raw classifier metrics at production thresholds",
        "",
        "These are pooled sample counts. Each row uses that head's own target and universe, "
        "so rows are useful for understanding a head but are not a cross-head leaderboard.",
        "",
    ]
    for scope in (
        "held-out-feedback",
        "development-human-reviewed",
        "development-all-diagnostic",
        "combined-diagnostic",
    ):
        scope_note = (
            "Serve and dead-state rows in this scope use untouched model-origin "
            "boundaries and are diagnostic only."
            if scope == "held-out-feedback"
            else ""
        )
        lines.extend([f"### {scope}", ""])
        if scope_note:
            lines.extend([scope_note, ""])
        lines.extend(
            [
                "| Bundle | Head | Threshold | Positives | Predicted positives | Precision | Recall | F1 |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
                *headline_rows(report, scope),
                "",
            ]
        )

    lines.extend(
        [
            "## Serve heads on held-out export feedback",
            "",
            "Production does not merge the two serve peaks into one detector. It decodes "
            "each serve head inside its own three-head model path and later unions the "
            "resulting rally intervals. The combined rows below mirror those parallel paths: "
            "duplicate path alerts remain in the precision denominator, while each feedback "
            "start is counted only once for recall if either path covers it.",
            "",
            "| Tolerance | Path | Precision | Recall | Matched / predicted | Covered / true |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for tolerance in (0.5, 1.0, 2.0):
        key = str(tolerance)
        v2 = held["model-1ca43e38eefc"]["serve"]["operationalEvent"][
            "toleranceSensitivity"
        ][key]
        old = held["model-9c92b8e9333f"]["serve"]["operationalEvent"][
            "toleranceSensitivity"
        ][key]
        combined = held["combinedServeParallelPaths"]["toleranceSensitivity"][key]
        lines.extend(
            [
                f"| {tolerance:.1f}s | All-labels v2 | {v2['precision']:.4f} | {v2['recall']:.4f} | {v2['matchedServes']} / {v2['predictedServes']} | {v2['matchedServes']} / {v2['trueServes']} |",
                f"| {tolerance:.1f}s | Previous production | {old['precision']:.4f} | {old['recall']:.4f} | {old['matchedServes']} / {old['predictedServes']} | {old['matchedServes']} / {old['trueServes']} |",
                f"| {tolerance:.1f}s | Both production paths | {combined['precision']:.4f} | {combined['recall']:.4f} | {combined['matchedPathAlerts']} / {combined['pathAlerts']} | {combined['coveredServes']} / {combined['trueServes']} |",
            ]
        )
    lines.extend(
        [
            "",
            "These feedback boundaries are untouched production-model starts, not verified "
            "serve-contact annotations. This section measures agreement with export feedback "
            "starts; it cannot establish semantic serve recall.",
            "",
        ]
    )

    lines.extend(
        [
            "## Operational interpretation on the strongest applicable scope",
            "",
            "| Bundle/head | Operational unit | Precision | Recall/coverage | Counts |",
            "|---|---|---:|---:|---|",
        ]
    )
    for model_id, label in (
        ("model-1ca43e38eefc", "All-labels v2"),
        ("model-9c92b8e9333f", "Previous production"),
    ):
        rally = held[model_id]["rally"]["operational"]
        serve = boundary[model_id]["serve"]["operationalEvent"]["operational"]
        dead = boundary[model_id]["deadState"]["oracleAnchoredCoverage"]
        lines.extend(
            [
                f"| {label} rally | decoded live seconds against held-feedback retained ranges | {rally['precision']:.4f} | {rally['recall']:.4f} | {rally['predictedSeconds']:.1f}s predicted / {rally['truthSeconds']:.1f}s retained |",
                f"| {label} serve | decoded contacts within 1.0s on human-reviewed development | {serve['precision']:.4f} | {serve['recall']:.4f} | {serve['matchedServes']}/{serve['trueServes']} serves; {serve['predictedServes']} detections |",
                f"| {label} dead-state | oracle-end-anchored transitions on human-reviewed development | — | {dead['coverageRecall']:.4f} | {dead['detectedTransitions']}/{dead['trueEnds']} ends emitted |",
            ]
        )
    suppression = held["suppression-overlap-exclusion-retrained"]["operational"]
    lines.append(
        f"| Suppression | decoded selected-universe samples | {suppression['precision']:.4f} | {suppression['recall']:.4f} | rally-sample survival {suppression['rallySampleSurvival']:.4f} |"
    )

    lines.extend(
        [
            "",
            "## Can a head be used by itself?",
            "",
            "- **Serve-region proposals:** useful, but not a complete gate. A consumer must "
            "keep a fallback path for rallies whose serve head does not emit a matched peak. "
            "The JSON artifact includes event precision/recall at thresholds 0.05–0.95 and "
            "0.5/1.0/2.0-second tolerance sensitivity.",
            "- **Rally inclusion:** the rally heads can independently propose live regions, "
            "but neither should be interpreted as complete coverage. Production unions them "
            "and also uses serve rescue specifically because one head alone misses live time. "
            "Held-feedback precision is optimistic as an absolute truth estimate because the "
            "accepted reference ranges themselves originated from the production ensemble.",
            "- **End refinement:** dead-state scores are meaningful only in local windows "
            "around an existing proposed end. Oracle-anchored coverage is reported to isolate "
            "the head; it does not establish global standalone precision.",
            "- **Suppression/veto:** use only inside the current one-model-only eligibility "
            "gate. Its precision/recall universe intentionally excludes arbitrary dead time, "
            "so it cannot justify whole-video dead-time classification.",
            "",
            "## Metric contracts",
            "",
            "- Rally raw target: human live-play samples outside `ignoredIntervals`; "
            "operational metrics apply only that rally head's production temporal decoder "
            "and measure unpadded duration overlap.",
            "- Serve raw target: samples within ±1.0 seconds of every human serve contact. "
            "Operational metrics apply the production peak threshold, 10-second NMS, and "
            "a primary 1.0-second one-to-one matching tolerance.",
            "- Dead-state raw target: the production local end-transition target (2 seconds "
            "before/after each end plus pre-serve negative controls), at the production 0.90 "
            "dead threshold.",
            "- Suppression raw target: valid non-rally samples selected by the production "
            "ensemble or explicit hard-negative labels versus valid human rally samples. "
            "The production 1.0-second smoothing and 0.75/0.65 decoder is used for the "
            "operational row.",
            "- All sample metrics pool confusion counts across recordings; they do not "
            "average per-recording precision or recall.",
            "- Held-feedback serve/dead rows remain in JSON for auditability, but they are "
            "diagnostic only: `userTouchedCutIds` is empty for all six held files, so those "
            "model-origin starts and ends are not independent semantic boundary labels.",
            "",
            "Full threshold profiles, per-recording diagnostics, model hashes, source hashes, "
            "and exact counts are in the companion JSON artifact.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace the two exact generated report paths if they already exist",
    )
    args = parser.parse_args()

    helpers = runpy.run_path(str(TRAIN_FEEDBACK_SCRIPT))
    heads = load_production_heads()
    suppression = load_model(SUPPRESSION_PATH)
    items = load_items(helpers, heads["model-1ca43e38eefc"].rally.feature_config)

    probabilities: dict[str, dict[str, np.ndarray]] = {}
    for model_id, bundle in heads.items():
        for role, model in (
            ("rally", bundle.rally),
            ("serve", bundle.serve),
            ("deadState", bundle.dead_state),
        ):
            probabilities[f"{model_id}:{role}"] = {
                entry.prepared.recording.id: model.predict(
                    entry.prepared.contextual_values
                )
                for entry in items
            }
    suppression_probabilities = {
        entry.prepared.recording.id: suppression.predict(
            entry.prepared.contextual_values
        )
        for entry in items
    }

    helper_old, helper_v2 = helpers["load_heads"]()
    baseline: dict[str, tuple[Interval, ...]] = {}
    suppression_target_rows: dict[
        str, tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    for entry in items:
        item = entry.prepared
        old_intervals = helpers["predict_heads"](item, helper_old)
        v2_intervals = helpers["predict_heads"](item, helper_v2)
        baseline[item.recording.id] = helpers["union_intervals"](
            old_intervals, v2_intervals
        )
        suppression_target_rows[item.recording.id] = suppression_targets(
            entry, baseline[item.recording.id], helpers
        )

    suppression_decoder = helpers["HELD_PRODUCTION_SUPPRESSION_DECODER"]
    scopes = {
        "development-human-reviewed": [
            entry
            for entry in items
            if entry.boundary_reference
            == "human-reviewed-label-document-in-progress"
        ],
        "development-all-diagnostic": [
            entry for entry in items if entry.partition == "development"
        ],
        "held-out-feedback": [
            entry for entry in items if entry.partition == "held-out-feedback"
        ],
        "combined-diagnostic": list(items),
    }
    report = {
        "schemaVersion": 1,
        "analysis": "production-head-independent-precision-recall-v1",
        "generatedAt": datetime.now(UTC).isoformat(),
        "selectionPolicy": {
            "protectedTestOpened": False,
            "protectedTestIds": ["indoor-source-05"],
            "developmentRole": "tuning-only; suppression decoder was selected here",
            "humanReviewedDevelopmentRole": (
                "provisional boundary diagnostics; documents remain in-progress and "
                "source groups overlap model development"
            ),
            "heldFeedbackRole": (
                "file-level held-out inclusion/veto evaluation; model-origin boundaries "
                "were not user-adjusted and are not boundary gold"
            ),
            "combinedRole": "diagnostic only; not a model-selection score",
        },
        "sources": [
            {
                "id": entry.prepared.recording.id,
                "partition": entry.partition,
                "sourceGroup": entry.prepared.recording.source_group,
                "environment": entry.prepared.recording.environment,
                "rallies": len(entry.prepared.recording.rallies),
                "ignoredIntervals": len(entry.prepared.recording.ignored_intervals),
                "sourcePath": str(entry.source_path),
                "sourceSha256": sha256_file(entry.source_path),
                "boundaryReference": entry.boundary_reference,
            }
            for entry in items
        ],
        "modelArtifacts": {
            model_id: {
                "rally": head_metadata(bundle.rally),
                "serve": head_metadata(bundle.serve),
                "deadState": head_metadata(bundle.dead_state),
            }
            for model_id, bundle in heads.items()
        }
        | {"suppression-overlap-exclusion-retrained": head_metadata(suppression)},
        "scopes": {
            name: evaluation_for_scope(
                scope_items,
                heads,
                suppression,
                probabilities,
                suppression_probabilities,
                suppression_target_rows,
                suppression_decoder,
            )
            for name, scope_items in scopes.items()
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    if args.force:
        args.json.unlink(missing_ok=True)
        args.markdown.unlink(missing_ok=True)
    atomic_write_text(args.json, json.dumps(report, indent=2, allow_nan=False) + "\n")
    atomic_write_text(args.markdown, markdown_report(report))
    print(args.json)
    print(args.markdown)


if __name__ == "__main__":
    main()
