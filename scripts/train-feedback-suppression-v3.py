#!/usr/bin/env python3
"""Train and evaluate feedback-augmented v3 and false-positive suppression models.

This experiment keeps the production feature substrate and three-head decoding
architecture fixed.  Candidate 1 refits rally, serve, and end-transition heads on
the all-labels-v2 corpus plus a seeded file-level half of the model-feedback store.
Candidate 2 adds a fourth head that detects currently-selected dead/setup activity
and subtracts those spans from the candidate-1 output.

All feedback features are decoded from the feedback JSON.  No feedback source video
is opened for feature extraction.  Suppression thresholds are selected only on the
three predeclared challenge-development recordings; protected test and held-feedback
labels are opened only for final reporting.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from analysis.annotations import load_label_document
from analysis.artifacts import atomic_write_text
from analysis.config import DecoderConfig, TrainingConfig
from analysis.crop_evaluation import (
    _duration,
    _intersection_duration,
    _merge_intervals,
    pad_and_merge_intervals,
    subtract_intervals,
)
from analysis.dead_ball_experiment import (
    _dead_ball_input_mask,
    _masked_dead_ball_values,
    _prediction_inputs,
)
from analysis.dead_state import DeadStateDecoderConfig
from analysis.dead_state_experiment import (
    DEAD_STATE_TARGET_ID,
    END_TRANSITION_TARGET_MODE,
    DeadStateRefinementConfig,
    _eligible_target_mask,
    _predictions_for,
    _target_for_mode,
)
from analysis.decoder import DecodedInterval, decode_probabilities
from analysis.features import FeatureSequence, VideoMetadata, contextualize
from analysis.model import (
    DEAD_STATE_TASK,
    RALLY_LIVE_TASK,
    SERVE_CONTACT_TASK,
    LogisticModel,
    load_model,
    train_logistic_model,
)
from analysis.pipeline import PreparedRecording, prepare_recording
from analysis.schema import Interval, Recording, labels_for_times, load_manifest, mask_for_times
from analysis.serve import ServeCompositionConfig, ServeDecoderConfig, serve_labels_for_times


EXPERIMENT_ID = "feedback-suppression-v3-2026-08-16"
TARGET_PADDING_SECONDS = 2.0
PADDING_CASES = (0.0, 1.0, 2.0, 3.0)
JOIN_GAP_SECONDS = 3.0
FEEDBACK_SPLIT_SEED = 20260816
HARD_NEGATIVE_MULTIPLIER = 4

ALL_LABELS_MANIFEST = Path(
    "/mnt/freenas/volleycut/intake-2026-08-13/experiments/"
    "environment-specialists-v2/manifests/all-labels-train.json"
)
INFERENCE_MANIFEST = Path(
    "/mnt/freenas/volleycut/intake-2026-08-13/manifests/inference-only.json"
)
FEATURE_CACHE = Path(
    "/mnt/freenas/volleycut/intake-2026-08-13/experiments/"
    "environment-specialists-v2/features/audiovisual-audio-normalized-v3"
)
FEEDBACK_ROOT = Path("/mnt/freenas/volleycut/model-feedback")
LABEL_ROOTS = (
    Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09/labels/full"),
    Path("/mnt/freenas/volleycut/intake-2026-08-13/labels/full"),
    Path("/mnt/freenas/volleycut/intake-2026-08-13/completed/full-v1"),
)
OLD_MODEL_ROOT = Path(
    "/mnt/freenas/volleycut/labeling-v1-2026-08-09-no-beach-2026-08-12/models"
)
OLD_HEADS = {
    "rally": "full-audiovisual-audio-normalized-v3",
    "serve": "serve-specialist-audio-normalized-v5",
    "dead": "dead-state-transition-audio-normalized-v5-no-legacy-final",
}
V2_BUNDLE = Path(
    "/mnt/freenas/volleycut/intake-2026-08-13/experiments/"
    "environment-specialists-v2/models/all-labels/bundle.json"
)
DEFAULT_OUTPUT = Path(
    "/mnt/freenas/volleycut/intake-2026-08-13/experiments/"
    "feedback-suppression-v3-2026-08-16"
)
DEVELOPMENT_IDS = {
    "grass-source-06",
    "grass-source-08",
    "indoor-source-03",
}
PROTECTED_TEST_IDS = {"indoor-source-05"}


@dataclass(frozen=True)
class Heads:
    rally: LogisticModel
    serve: LogisticModel
    dead: LogisticModel


@dataclass(frozen=True)
class Item:
    prepared: PreparedRecording
    provenance: str
    feedback_partition: str | None
    labeled: bool


def write_new_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(payload, indent=2, allow_nan=False) + "\n")


def json_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def interval_rows(intervals: Iterable[Interval]) -> list[dict[str, float]]:
    return [{"start": item.start, "end": item.end} for item in intervals]


def as_intervals(rows: Iterable[Mapping[str, Any]], *, core: bool = False) -> tuple[Interval, ...]:
    start_key, end_key = ("coreStart", "coreEnd") if core else ("start", "end")
    return tuple(
        Interval(float(row[start_key]), float(row[end_key]))
        for row in rows
        if float(row[end_key]) > float(row[start_key])
    )


def numeric_array(payload: Mapping[str, Any], dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    if payload.get("encoding") != "base64" or payload.get("byteOrder") != "little-endian":
        raise ValueError("feedback numeric payload has unsupported encoding")
    if payload.get("dataType") != dtype or tuple(payload.get("shape", ())) != shape:
        raise ValueError(f"feedback numeric payload does not match {dtype} {shape}")
    raw = base64.b64decode(str(payload["data"]), validate=True)
    numpy_dtype = np.dtype("<f4" if dtype == "float32" else "<f8")
    result = np.frombuffer(raw, dtype=numpy_dtype).copy()
    if result.size != math.prod(shape) or not np.isfinite(result).all():
        raise ValueError("feedback numeric payload is malformed")
    return result.reshape(shape)


def intervals_from_mask(times: np.ndarray, mask: np.ndarray, duration: float) -> tuple[Interval, ...]:
    if len(times) == 0:
        return ()
    width = 1.0 / 4.0
    rows: list[Interval] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        if not value and start is not None:
            rows.append(Interval(max(0.0, float(times[start] - width / 2)), min(duration, float(times[index - 1] + width / 2))))
            start = None
    if start is not None:
        rows.append(Interval(max(0.0, float(times[start] - width / 2)), min(duration, float(times[-1] + width / 2))))
    return _merge_intervals(rows)


def record_from_label(path: Path) -> Recording:
    document = load_label_document(path, require_complete=False, require_video=True)
    raw_recording = document.payload["recording"]
    roi = raw_recording.get("roi")
    roi_tuple = (
        (float(roi["x"]), float(roi["y"]), float(roi["width"]), float(roi["height"]))
        if isinstance(roi, Mapping)
        else None
    )
    return Recording(
        id=document.recording_id,
        video=document.video,
        split="challenge",
        source_group=document.source_group,
        environment=document.environment,
        game=dict(raw_recording.get("game", {})),
        rallies=tuple(document.rallies),
        ignored_intervals=tuple(document.ignored_intervals),
        roi=roi_tuple,
        capture=dict(raw_recording.get("capture", {})),
        consent={"analyze": True, "train": False},
        content_sha256=raw_recording.get("contentSha256"),
        raw={
            **dict(raw_recording),
            "hardNegatives": document.payload.get("hardNegatives", []),
            "sideSwitches": document.payload.get("sideSwitches", []),
        },
    )


def load_labeled_records() -> dict[str, Recording]:
    paths: dict[str, Path] = {}
    for root in LABEL_ROOTS:
        for path in sorted(root.glob("*.labels.json")):
            recording_id = path.name.removesuffix(".labels.json")
            if recording_id == "indoor-source-08" and "completed" not in str(path):
                continue
            paths[recording_id] = path
    return {recording_id: record_from_label(path) for recording_id, path in paths.items()}


def feedback_prepared(path: Path, feature_config: Any) -> tuple[PreparedRecording, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload["source"]
    features = payload["features"]
    rows, columns = int(features["rows"]), int(features["columns"])
    times = numeric_array(features["timestamps"], "float64", (rows,)).astype(np.float64)
    values = numeric_array(features["values"], "float32", (rows, columns)).astype(np.float32)
    names = tuple(str(item) for item in features["names"])
    media = source["media"]
    duration = float(media["duration"])
    sequence = FeatureSequence(
        times=times,
        values=values,
        names=names,
        metadata=VideoMetadata(
            duration=duration,
            width=int(media["width"]),
            height=int(media["height"]),
            fps=float(features["analysisFps"]),
            frame_count=max(1, round(duration * float(features["analysisFps"]))),
            has_audio=bool(media.get("hasAudio")),
        ),
    )
    contextual_values, contextual_names = contextualize(sequence, feature_config)
    corrections = payload["corrections"]
    truth = as_intervals(
        (row for row in corrections["correctedRanges"] if row.get("included") is True),
        core=True,
    )
    game_window = source["gameWindow"]
    ignored: list[Interval] = []
    if float(game_window["start"]) > 0:
        ignored.append(Interval(0.0, float(game_window["start"])))
    if float(game_window["end"]) < duration:
        ignored.append(Interval(float(game_window["end"]), duration))
    ignored.extend(as_intervals(corrections.get("ignoredIntervals", [])))
    false_positives = as_intervals(corrections["labels"].get("falsePositives", []))
    source_path = json.loads((path.parent / "import.json").read_text(encoding="utf-8")).get("sourcePath")
    roi = source.get("featureRoi")
    recording = Recording(
        id=str(source["projectId"]),
        video=Path(source_path) if source_path else Path(source["file"]["name"]),
        split="train",
        source_group="feedback-2026-08-16",
        environment="unknown",
        game={},
        rallies=truth,
        ignored_intervals=tuple(ignored),
        roi=(float(roi["x"]), float(roi["y"]), float(roi["width"]), float(roi["height"])),
        capture={},
        consent={"analyze": True, "train": True},
        content_sha256=None,
        raw={
            "hardNegatives": [row.to_dict() for row in false_positives],
            "feedbackPath": str(path),
            "sourceFilename": source["file"]["name"],
        },
    )
    prepared = PreparedRecording(
        recording=recording,
        sequence=sequence,
        contextual_values=contextual_values,
        contextual_names=contextual_names,
        labels=labels_for_times(times, truth),
        sample_mask=mask_for_times(times, ignored),
    )
    return prepared, payload


def load_heads() -> tuple[Heads, Heads]:
    old = Heads(*(load_model(OLD_MODEL_ROOT / OLD_HEADS[key]) for key in ("rally", "serve", "dead")))
    bundle = json.loads(V2_BUNDLE.read_text(encoding="utf-8"))["heads"]
    v2 = Heads(
        load_model(bundle["rally"]["path"]),
        load_model(bundle["serve"]["path"]),
        load_model(bundle["deadState"]["path"]),
    )
    return old, v2


def predict_heads(item: PreparedRecording, heads: Heads) -> tuple[Interval, ...]:
    serve_decoder = ServeDecoderConfig.from_dict(heads.serve.training_summary["serveDecoder"])
    composition = ServeCompositionConfig.from_dict(heads.serve.training_summary["composition"])
    dead_decoder = DeadStateDecoderConfig.from_dict(heads.dead.training_summary["selectedDeadStateDecoder"])
    refinement = DeadStateRefinementConfig.from_dict(heads.dead.training_summary["selectedRefinement"])
    inputs = _prediction_inputs(item, heads.rally, heads.serve, heads.dead, serve_decoder, composition)
    prediction = _predictions_for([inputs], dead_decoder, refinement)[0]
    return tuple(Interval(float(row.start), float(row.end)) for row in prediction.candidate)


def union_intervals(*groups: Sequence[Interval]) -> tuple[Interval, ...]:
    return _merge_intervals(item for group in groups for item in group)


def subtract_predictions(source: Sequence[Interval], cuts: Sequence[Interval]) -> tuple[Interval, ...]:
    return subtract_intervals(source, cuts)


def hard_negative_mask(item: PreparedRecording) -> np.ndarray:
    mask = np.zeros(len(item.sequence.times), dtype=bool)
    for field in ("hardNegatives", "sideSwitches"):
        for row in item.recording.raw.get(field, []):
            if not isinstance(row, Mapping):
                continue
            start = row.get("start", row.get("time"))
            end = row.get("end", row.get("time"))
            if start is None or end is None:
                continue
            start_f, end_f = float(start), float(end)
            if end_f <= start_f:
                end_f = start_f + 1.0
            mask |= (item.sequence.times >= start_f) & (item.sequence.times < end_f)
    return mask & item.sample_mask & (item.labels < 0.5)


def augmented(values: np.ndarray, labels: np.ndarray, hard: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    indexes = np.flatnonzero(hard)
    if len(indexes) == 0:
        return values, labels, 0
    repeats = HARD_NEGATIVE_MULTIPLIER - 1
    return (
        np.concatenate([values, *([values[indexes]] * repeats)]),
        np.concatenate([labels, *([labels[indexes]] * repeats)]),
        len(indexes),
    )


def fixed_training(epochs: int) -> TrainingConfig:
    return TrainingConfig(epochs=epochs, batch_size=2048, learning_rate=0.02, l2=0.0001, patience=epochs, seed=7)


def train_three_heads(training: Sequence[PreparedRecording], template: Heads, destination: Path) -> Heads:
    if destination.exists():
        raise FileExistsError(destination)
    values: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    hard_count = 0
    for item in training:
        eligible = item.sample_mask
        base_values = item.contextual_values[eligible]
        base_labels = item.labels[eligible]
        local_hard = hard_negative_mask(item)[eligible]
        fitted_values, fitted_labels, count = augmented(base_values, base_labels, local_hard)
        values.append(fitted_values)
        labels.append(fitted_labels)
        hard_count += count
    rally = train_logistic_model(
        values, labels, [], [], template.rally.feature_config, template.rally.feature_names,
        template.rally.decoder, fixed_training(int(template.rally.training_summary["bestEpoch"])),
        prediction_task=RALLY_LIVE_TASK,
    )
    rally.feature_version = template.rally.feature_version
    lineage = {
        "experiment": EXPERIMENT_ID,
        "candidate": "v3-three-head-feedback-half",
        "trainingRecordingIds": [item.recording.id for item in training],
        "validationRecordingIds": [],
        "feedbackFeaturesReusedFromBundles": True,
        "hardNegativeMultiplier": HARD_NEGATIVE_MULTIPLIER,
        "hardNegativeUniqueSamples": hard_count,
        "selectionPolicy": "Frozen v2 epoch cap and decoders; no evaluation, protected-test, or held-feedback labels used.",
    }
    rally.training_summary.update(lineage)
    rally.save(destination / "rally")

    radius = float(template.serve.training_summary["serveTarget"]["radiusSeconds"])
    serve_values: list[np.ndarray] = []
    serve_labels: list[np.ndarray] = []
    for item in training:
        eligible = item.sample_mask
        targets = serve_labels_for_times(item.sequence.times, item.recording.rallies, radius)[eligible]
        fitted_values, fitted_labels, _ = augmented(
            item.contextual_values[eligible], targets, hard_negative_mask(item)[eligible]
        )
        serve_values.append(fitted_values)
        serve_labels.append(fitted_labels)
    serve = train_logistic_model(
        serve_values, serve_labels, [], [], rally.feature_config, rally.feature_names,
        rally.decoder, fixed_training(int(template.serve.training_summary["bestEpoch"])),
        prediction_task=SERVE_CONTACT_TASK,
    )
    serve.feature_version = rally.feature_version
    serve.training_summary.update({
        **lineage,
        "rallyModelSha256": rally.artifact_sha256,
        "serveTarget": template.serve.training_summary["serveTarget"],
        "serveInputProfile": template.serve.training_summary["serveInputProfile"],
        "serveDecoder": template.serve.training_summary["serveDecoder"],
        "composition": template.serve.training_summary["composition"],
    })
    serve.save(destination / "serve")

    target = template.dead.training_summary["deadStateTarget"]
    retained = _dead_ball_input_mask(
        rally.feature_names, str(template.dead.training_summary["deadStateInputProfile"]["id"])
    )
    dead_values: list[np.ndarray] = []
    dead_labels: list[np.ndarray] = []
    for item in training:
        targets, target_mask = _target_for_mode(
            item.sequence.times,
            item.recording.rallies,
            target_mode=END_TRANSITION_TARGET_MODE,
            before_end_seconds=float(target["beforeEndSeconds"]),
            after_end_seconds=float(target["afterEndSeconds"]),
            pre_serve_setup_seconds=float(target["preServeSetupSeconds"]),
        )
        eligible = _eligible_target_mask(item.sample_mask, target_mask)
        dead_values.append(_masked_dead_ball_values(item.contextual_values[eligible], retained))
        dead_labels.append(targets[eligible])
    dead = train_logistic_model(
        dead_values, dead_labels, [], [], rally.feature_config, rally.feature_names,
        rally.decoder, fixed_training(int(template.dead.training_summary["bestEpoch"])),
        prediction_task=DEAD_STATE_TASK,
    )
    dead.feature_version = rally.feature_version
    dead.training_summary.update({
        **lineage,
        "rallyModelSha256": rally.artifact_sha256,
        "serveModelSha256": serve.artifact_sha256,
        "deadStateTarget": {**template.dead.training_summary["deadStateTarget"], "id": DEAD_STATE_TARGET_ID},
        "deadStateInputProfile": template.dead.training_summary["deadStateInputProfile"],
        "selectedDeadStateDecoder": template.dead.training_summary["selectedDeadStateDecoder"],
        "selectedRefinement": template.dead.training_summary["selectedRefinement"],
    })
    dead.save(destination / "dead-state")
    return Heads(rally, serve, dead)


def train_suppression(
    training: Sequence[PreparedRecording],
    baseline: Mapping[str, Sequence[Interval]],
    template: LogisticModel,
) -> tuple[LogisticModel, dict[str, Any]]:
    train_values: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    stats: list[dict[str, Any]] = []
    for item in training:
        times = item.sequence.times
        truth = item.labels > 0.5
        predicted = labels_for_times(times, baseline[item.recording.id]) > 0.5
        explicit = hard_negative_mask(item)
        positive = item.sample_mask & ~truth & (predicted | explicit)
        negative = item.sample_mask & truth
        selected = positive | negative
        if not np.any(positive) or not np.any(negative):
            raise ValueError(f"suppression target lacks a class for {item.recording.id}")
        train_values.append(item.contextual_values[selected])
        train_labels.append(positive[selected].astype(np.float32))
        stats.append({
            "id": item.recording.id,
            "positiveSuppressionSamples": int(np.sum(positive)),
            "negativeRallySamples": int(np.sum(negative)),
            "explicitNegativeSamples": int(np.sum(explicit)),
        })
    model = train_logistic_model(
        train_values, train_labels, [], [], template.feature_config, template.feature_names,
        DecoderConfig(
            smoothing_seconds=0.5,
            enter_threshold=0.7,
            exit_threshold=0.6,
            min_live_seconds=0.5,
            bridge_gap_seconds=0.5,
            short_event_min_seconds=0.25,
            short_event_threshold=0.9,
        ),
        fixed_training(int(template.training_summary["bestEpoch"])),
        prediction_task=DEAD_STATE_TASK,
    )
    model.feature_version = template.feature_version
    model.training_summary.update({
        "experiment": EXPERIMENT_ID,
        "candidate": "false-positive-suppression-head",
        "specialistRole": "veto selected dead/setup/side-switch activity",
        "positiveDefinition": "valid non-rally samples selected by current production, plus explicit hard-negative/side-switch samples",
        "negativeDefinition": "valid human rally samples",
        "trainingRecordingIds": [item.recording.id for item in training],
        "feedbackFeaturesReusedFromBundles": True,
        "perRecordingTargets": stats,
        "selectionPolicy": "Fit labels exclude protected test and held-feedback files; decoder selected only on predeclared development recordings.",
    })
    return model, {"perRecording": stats}


def decode_suppression(item: PreparedRecording, model: LogisticModel, config: DecoderConfig) -> tuple[Interval, ...]:
    probabilities = model.predict(item.contextual_values)
    decoded, _ = decode_probabilities(
        item.sequence.times, probabilities, item.sequence.metadata.duration, config, 4.0
    )
    return tuple(Interval(row.start, row.end) for row in decoded)


def metric_for(
    items: Sequence[Item], predictions: Mapping[str, Sequence[Interval]], padding: float
) -> dict[str, Any]:
    core_intersection = core_predicted = core_truth = 0.0
    padded_intersection = padded_predicted = padded_truth = 0.0
    hybrid_core_intersection = 0.0
    crop_count = 0
    per_recording: list[dict[str, Any]] = []
    for entry in items:
        item = entry.prepared
        truth_core = subtract_intervals(item.recording.rallies, item.recording.ignored_intervals)
        pred_core = subtract_intervals(predictions[item.recording.id], item.recording.ignored_intervals)
        truth_pad = subtract_intervals(
            pad_and_merge_intervals(item.recording.rallies, item.sequence.metadata.duration, padding, JOIN_GAP_SECONDS),
            item.recording.ignored_intervals,
        )
        pred_pad = subtract_intervals(
            pad_and_merge_intervals(predictions[item.recording.id], item.sequence.metadata.duration, padding, JOIN_GAP_SECONDS),
            item.recording.ignored_intervals,
        )
        ci = _intersection_duration(pred_core, truth_core)
        pi = _intersection_duration(pred_pad, truth_pad)
        hi = _intersection_duration(pred_pad, truth_core)
        cp, ct = _duration(pred_core), _duration(truth_core)
        pp, pt = _duration(pred_pad), _duration(truth_pad)
        core_intersection += ci
        core_predicted += cp
        core_truth += ct
        padded_intersection += pi
        padded_predicted += pp
        padded_truth += pt
        hybrid_core_intersection += hi
        crop_count += len(pred_pad)
        per_recording.append({
            "id": item.recording.id,
            "coreIntersectionSeconds": ci,
            "corePredictedSeconds": cp,
            "coreTruthSeconds": ct,
            "paddedIntersectionSeconds": pi,
            "paddedPredictedSeconds": pp,
            "paddedTruthSeconds": pt,
            "paddedVsCoreIntersectionSeconds": hi,
        })

    def scores(intersection: float, predicted: float, truth: float) -> dict[str, float]:
        precision = intersection / predicted if predicted else 0.0
        recall = intersection / truth if truth else 1.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }

    core = scores(core_intersection, core_predicted, core_truth)
    padded = scores(padded_intersection, padded_predicted, padded_truth)
    p_pad = padded["precision"]
    r_core = hybrid_core_intersection / core_truth if core_truth else 1.0
    hybrid_f1 = 2 * p_pad * r_core / (p_pad + r_core) if p_pad + r_core else 0.0
    return {
        "paddingSecondsBeforeAndAfter": padding,
        "joinGapSeconds": JOIN_GAP_SECONDS,
        "recordings": len(items),
        "core": {**core, "intersectionSeconds": core_intersection, "predictedSeconds": core_predicted, "truthSeconds": core_truth},
        "padded": {**padded, "intersectionSeconds": padded_intersection, "predictedSeconds": padded_predicted, "truthSeconds": padded_truth},
        "P_pad": p_pad,
        "R_core": r_core,
        "F1_padP_coreR": hybrid_f1,
        "paddedModelExportSeconds": padded_predicted,
        "paddedHumanExportSeconds": padded_truth,
        "paddedDurationDifferenceSeconds": padded_predicted - padded_truth,
        "outputCropCount": crop_count,
        "perRecording": per_recording,
    }


def config_candidates() -> list[DecoderConfig]:
    result: list[DecoderConfig] = []
    for enter in (0.55, 0.65, 0.75, 0.85):
        for smoothing in (0.5, 1.0):
            for minimum in (0.5, 1.0, 2.0):
                result.append(DecoderConfig(
                    smoothing_seconds=smoothing,
                    enter_threshold=enter,
                    exit_threshold=max(0.05, enter - 0.1),
                    min_live_seconds=minimum,
                    bridge_gap_seconds=0.5,
                    short_event_min_seconds=0.25,
                    short_event_threshold=max(0.9, enter),
                ))
    return result


def tune_suppression(
    development: Sequence[Item],
    base: Mapping[str, Sequence[Interval]],
    model: LogisticModel,
) -> tuple[DecoderConfig, dict[str, Any]]:
    best: tuple[tuple[float, ...], DecoderConfig, dict[str, Any]] | None = None
    base_metric = metric_for(development, base, TARGET_PADDING_SECONDS)
    for config in config_candidates():
        candidate = {
            entry.prepared.recording.id: subtract_predictions(
                base[entry.prepared.recording.id], decode_suppression(entry.prepared, model, config)
            )
            for entry in development
        }
        metric = metric_for(development, candidate, TARGET_PADDING_SECONDS)
        key = (
            metric["F1_padP_coreR"],
            metric["P_pad"],
            metric["R_core"],
            -metric["paddedModelExportSeconds"],
        )
        if best is None or key > best[0]:
            best = (key, config, metric)
    assert best is not None
    return best[1], {
        "scope": sorted(entry.prepared.recording.id for entry in development),
        "assessmentRole": "tuning-only",
        "targetPaddingSeconds": TARGET_PADDING_SECONDS,
        "rankingMetric": "F1_padP_coreR",
        "candidateCount": len(config_candidates()),
        "baseline": base_metric,
        "selected": best[2],
        "selectedConfig": best[1].to_dict(),
    }


def group_items(items: Sequence[Item]) -> dict[str, list[Item]]:
    groups: dict[str, list[Item]] = {"all-evaluable": list(items)}
    for entry in items:
        environment = entry.prepared.recording.environment
        groups.setdefault(f"environment:{environment}", []).append(entry)
        groups.setdefault(f"provenance:{entry.provenance}", []).append(entry)
        groups.setdefault(f"environment:{environment}|provenance:{entry.provenance}", []).append(entry)
        if entry.feedback_partition:
            groups.setdefault(f"feedback-partition:{entry.feedback_partition}", []).append(entry)
    return groups


def delta(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "corePrecision": candidate["core"]["precision"] - baseline["core"]["precision"],
        "coreRecall": candidate["core"]["recall"] - baseline["core"]["recall"],
        "coreF1": candidate["core"]["f1"] - baseline["core"]["f1"],
        "paddedPrecision": candidate["padded"]["precision"] - baseline["padded"]["precision"],
        "paddedRecall": candidate["padded"]["recall"] - baseline["padded"]["recall"],
        "paddedF1": candidate["padded"]["f1"] - baseline["padded"]["f1"],
        "P_pad": candidate["P_pad"] - baseline["P_pad"],
        "R_core": candidate["R_core"] - baseline["R_core"],
        "F1_padP_coreR": candidate["F1_padP_coreR"] - baseline["F1_padP_coreR"],
        "paddedModelExportSeconds": candidate["paddedModelExportSeconds"] - baseline["paddedModelExportSeconds"],
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    names = report["variantLabels"]
    lines = [
        "# Feedback suppression v3 experiment",
        "",
        f"Target product padding: **{TARGET_PADDING_SECONDS:g}s before / {TARGET_PADDING_SECONDS:g}s after**. ",
        f"Ranking metric: `F1_padP_coreR`; gaps strictly below {JOIN_GAP_SECONDS:g}s are joined.",
        "",
        "Feedback was split at file level: 5 fit files and 6 held-out files. Feedback environment is reported as `unknown` because the bundle schema does not encode it.",
        "",
        "## Target-padding summary",
        "",
        "| Scope | Variant | Core P | Core R | Core F1 | Padded P | Padded R | Padded F1 | P_pad | R_core | F1_padP_coreR | Export s | Δ export s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scope in report["displayScopes"]:
        rows = report["metrics"][scope]
        baseline = rows["current-production-ensemble"]["padding"]["2"]
        for key in report["variantOrder"]:
            row = rows[key]["padding"]["2"]
            lines.append(
                f"| {scope} | {names[key]} | {row['core']['precision']:.4f} | {row['core']['recall']:.4f} | {row['core']['f1']:.4f} | "
                f"{row['padded']['precision']:.4f} | {row['padded']['recall']:.4f} | {row['padded']['f1']:.4f} | "
                f"{row['P_pad']:.4f} | {row['R_core']:.4f} | {row['F1_padP_coreR']:.4f} | "
                f"{row['paddedModelExportSeconds']:.1f} | {row['paddedModelExportSeconds'] - baseline['paddedModelExportSeconds']:+.1f} |"
            )
    lines.extend(["", "## Four required padding cases", ""])
    for scope in report["displayScopes"]:
        lines.extend([
            f"### {scope}",
            "",
            "| Variant | Pad | P_pad | R_core | F1_padP_coreR | Model export s | Human export s | Difference s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for key in report["variantOrder"]:
            for pad in ("0", "1", "2", "3"):
                row = report["metrics"][scope][key]["padding"][pad]
                lines.append(
                    f"| {names[key]} | {pad}s | {row['P_pad']:.4f} | {row['R_core']:.4f} | {row['F1_padP_coreR']:.4f} | "
                    f"{row['paddedModelExportSeconds']:.1f} | {row['paddedHumanExportSeconds']:.1f} | {row['paddedDurationDifferenceSeconds']:+.1f} |"
                )
        lines.append("")
    lines.extend([
        "## Interpretation constraints",
        "",
        "- Development recordings are tuning-only, not held-out evidence.",
        "- The protected test recording and six held-feedback files were not used for fitting or decoder selection.",
        "- Historical label documents are still marked in-progress; results are experiment evidence, not a production promotion decision.",
        "- Two challenge videos received inference but have no reviewed label document, so they are excluded from metrics.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment directory {output}")
    output.mkdir(parents=True)

    old_heads, v2_heads = load_heads()
    if old_heads.rally.feature_names != v2_heads.rally.feature_names:
        raise ValueError("production and v2 feature signatures differ")

    all_labels = load_manifest(ALL_LABELS_MANIFEST)
    historical_training = [
        prepare_recording(recording, v2_heads.rally.feature_config, FEATURE_CACHE)
        for recording in all_labels.recordings
    ]

    feedback_rows: list[tuple[PreparedRecording, dict[str, Any], Path]] = []
    for path in sorted(FEEDBACK_ROOT.glob("*/bundle.json")):
        prepared, payload = feedback_prepared(path, v2_heads.rally.feature_config)
        if prepared.contextual_names != v2_heads.rally.feature_names:
            raise ValueError(f"feedback feature signature differs: {path}")
        feedback_rows.append((prepared, payload, path))
    rng = np.random.default_rng(FEEDBACK_SPLIT_SEED)
    ordered = sorted(feedback_rows, key=lambda row: row[0].recording.id)
    permutation = list(rng.permutation(len(ordered)))
    train_indexes = set(permutation[: len(ordered) // 2])
    feedback_train = [row[0] for index, row in enumerate(ordered) if index in train_indexes]
    feedback_hold = [row[0] for index, row in enumerate(ordered) if index not in train_indexes]
    training = [*historical_training, *feedback_train]

    # Baseline predictions used only to define selected false-positive suppression targets.
    baseline_training: dict[str, tuple[Interval, ...]] = {}
    for item in training:
        old = predict_heads(item, old_heads)
        v2 = predict_heads(item, v2_heads)
        baseline_training[item.recording.id] = union_intervals(old, v2)

    split_payload = {
        "schemaVersion": 1,
        "experiment": EXPERIMENT_ID,
        "seed": FEEDBACK_SPLIT_SEED,
        "method": "numpy-PCG64 permutation of project IDs sorted lexicographically; floor(n/2) fit files",
        "feedbackCount": len(ordered),
        "fitCount": len(feedback_train),
        "holdoutCount": len(feedback_hold),
        "fitProjectIds": sorted(item.recording.id for item in feedback_train),
        "holdoutProjectIds": sorted(item.recording.id for item in feedback_hold),
        "historicalTrainingIds": [item.recording.id for item in historical_training],
        "developmentIds": sorted(DEVELOPMENT_IDS),
        "protectedTestIds": sorted(PROTECTED_TEST_IDS),
        "targetPaddingSeconds": TARGET_PADDING_SECONDS,
        "joinGapSecondsStrictlyLessThan": JOIN_GAP_SECONDS,
    }
    write_new_json(output / "split-policy.json", split_payload)

    candidate1 = train_three_heads(training, v2_heads, output / "models" / "v3-candidate1-three-head")
    suppression, suppression_targets = train_suppression(training, baseline_training, v2_heads.rally)

    labeled_records = load_labeled_records()
    inference_manifest = load_manifest(INFERENCE_MANIFEST)
    inference_records = dict(labeled_records)
    for recording in inference_manifest.recordings:
        inference_records.setdefault(recording.id, recording)
    all_items: list[Item] = []
    training_ids = {item.recording.id for item in historical_training}
    feedback_train_ids = {item.recording.id for item in feedback_train}
    feedback_hold_ids = {item.recording.id for item in feedback_hold}
    for recording_id, recording in sorted(inference_records.items()):
        prepared = prepare_recording(recording, v2_heads.rally.feature_config, FEATURE_CACHE)
        provenance = "training-dataset" if recording_id in training_ids else "evaluation-validation-test-only"
        all_items.append(Item(prepared, provenance, None, recording_id in labeled_records))
    for prepared, _, _ in ordered:
        partition = "fit" if prepared.recording.id in feedback_train_ids else "held-out"
        all_items.append(Item(prepared, "export-feedback", partition, True))

    base_predictions: dict[str, dict[str, tuple[Interval, ...]]] = {
        "old": {}, "v2": {}, "current": {}, "v3": {}
    }
    for index, entry in enumerate(all_items, start=1):
        recording_id = entry.prepared.recording.id
        print(f"Inferring {index}/{len(all_items)}: {recording_id}", flush=True)
        old = predict_heads(entry.prepared, old_heads)
        v2 = predict_heads(entry.prepared, v2_heads)
        v3 = predict_heads(entry.prepared, candidate1)
        base_predictions["old"][recording_id] = old
        base_predictions["v2"][recording_id] = v2
        base_predictions["current"][recording_id] = union_intervals(old, v2)
        base_predictions["v3"][recording_id] = v3

    development = [entry for entry in all_items if entry.prepared.recording.id in DEVELOPMENT_IDS]
    prod_config, prod_selection = tune_suppression(development, base_predictions["current"], suppression)
    v3_config, v3_selection = tune_suppression(development, base_predictions["v3"], suppression)
    suppression.decoder = v3_config
    suppression.training_summary.update({
        "productionEnsembleDecoderSelection": prod_selection,
        "v3CandidateDecoderSelection": v3_selection,
    })
    suppression.save(output / "models" / "v3-candidate2-four-head" / "suppression")

    candidate1_bundle = {
        "schemaVersion": 1,
        "experiment": EXPERIMENT_ID,
        "candidate": "v3-candidate1-three-head",
        "createdAt": datetime.now(UTC).isoformat(),
        "heads": {
            "rally": {"path": str(output / "models" / "v3-candidate1-three-head" / "rally"), "sha256": candidate1.rally.artifact_sha256},
            "serve": {"path": str(output / "models" / "v3-candidate1-three-head" / "serve"), "sha256": candidate1.serve.artifact_sha256},
            "deadState": {"path": str(output / "models" / "v3-candidate1-three-head" / "dead-state"), "sha256": candidate1.dead.artifact_sha256},
        },
    }
    candidate1_bundle["modelId"] = f"model-{candidate1.dead.artifact_sha256[:12]}"
    write_new_json(output / "models" / "v3-candidate1-three-head" / "bundle.json", candidate1_bundle)
    candidate2_digest = hashlib.sha256(
        (str(candidate1.dead.artifact_sha256) + str(suppression.artifact_sha256)).encode()
    ).hexdigest()
    candidate2_bundle = {
        **candidate1_bundle,
        "candidate": "v3-candidate2-four-head-suppression",
        "modelId": f"model-{candidate2_digest[:12]}",
        "heads": {
            **candidate1_bundle["heads"],
            "suppression": {
                "path": str(output / "models" / "v3-candidate2-four-head" / "suppression"),
                "sha256": suppression.artifact_sha256,
                "decoder": v3_config.to_dict(),
            },
        },
    }
    write_new_json(output / "models" / "v3-candidate2-four-head" / "bundle.json", candidate2_bundle)

    variants: dict[str, dict[str, tuple[Interval, ...]]] = {
        "current-production-ensemble": dict(base_predictions["current"]),
        "production-plus-suppression": {},
        "v3-candidate1-three-head": dict(base_predictions["v3"]),
        "v3-candidate2-four-head": {},
        "old-plus-v3-candidate1": {},
        "old-plus-v3-candidate2": {},
    }
    for entry in all_items:
        recording_id = entry.prepared.recording.id
        prod_cut = decode_suppression(entry.prepared, suppression, prod_config)
        v3_cut = decode_suppression(entry.prepared, suppression, v3_config)
        v3_suppressed = subtract_predictions(base_predictions["v3"][recording_id], v3_cut)
        variants["production-plus-suppression"][recording_id] = subtract_predictions(
            base_predictions["current"][recording_id], prod_cut
        )
        variants["v3-candidate2-four-head"][recording_id] = v3_suppressed
        variants["old-plus-v3-candidate1"][recording_id] = union_intervals(
            base_predictions["old"][recording_id], base_predictions["v3"][recording_id]
        )
        variants["old-plus-v3-candidate2"][recording_id] = union_intervals(
            base_predictions["old"][recording_id], v3_suppressed
        )

    inference_root = output / "inference"
    for variant, predictions in variants.items():
        for entry in all_items:
            recording_id = entry.prepared.recording.id
            write_new_json(inference_root / variant / f"{recording_id}.json", {
                "schemaVersion": 1,
                "experiment": EXPERIMENT_ID,
                "variant": variant,
                "recordingId": recording_id,
                "sourceFilename": entry.prepared.recording.raw.get("sourceFilename", entry.prepared.recording.video.name),
                "duration": entry.prepared.sequence.metadata.duration,
                "featureSource": "embedded-feedback-json" if entry.provenance == "export-feedback" else "existing-npz-cache",
                "ranges": interval_rows(predictions[recording_id]),
            })

    evaluable = [entry for entry in all_items if entry.labeled]
    groups = group_items(evaluable)
    variant_order = list(variants)
    metrics: dict[str, Any] = {}
    for scope, scoped_items in sorted(groups.items()):
        metrics[scope] = {}
        for variant in variant_order:
            padding_metrics = {
                f"{padding:g}": metric_for(scoped_items, variants[variant], padding)
                for padding in PADDING_CASES
            }
            baseline_target = metric_for(scoped_items, variants["current-production-ensemble"], TARGET_PADDING_SECONDS)
            metrics[scope][variant] = {
                "padding": padding_metrics,
                "deltaAtTargetVsCurrentProduction": delta(padding_metrics["2"], baseline_target),
            }

    display_scopes = [
        key for key in (
            "all-evaluable",
            "provenance:training-dataset",
            "provenance:evaluation-validation-test-only",
            "provenance:export-feedback",
            "feedback-partition:fit",
            "feedback-partition:held-out",
            "environment:grass",
            "environment:indoor",
            "environment:beach",
            "environment:unknown",
        ) if key in metrics
    ]
    report = {
        "schemaVersion": 1,
        "experiment": EXPERIMENT_ID,
        "createdAt": datetime.now(UTC).isoformat(),
        "targetProductPaddingSecondsBeforeAndAfter": TARGET_PADDING_SECONDS,
        "requiredPaddingCases": list(PADDING_CASES),
        "joinGapSecondsStrictlyLessThan": JOIN_GAP_SECONDS,
        "rankingMetric": "F1_padP_coreR",
        "splitPolicy": split_payload,
        "models": {"candidate1": candidate1_bundle, "candidate2": candidate2_bundle},
        "suppressionTargets": suppression_targets,
        "selection": {"productionPlusSuppression": prod_selection, "v3Candidate2": v3_selection},
        "variantOrder": variant_order,
        "variantLabels": {
            "current-production-ensemble": "Current production (old + all-labels v2)",
            "production-plus-suppression": "Current production + suppression specialist",
            "v3-candidate1-three-head": "v3 candidate 1 (three-head refit)",
            "v3-candidate2-four-head": "v3 candidate 2 (three heads + suppression)",
            "old-plus-v3-candidate1": "Old production + v3 candidate 1",
            "old-plus-v3-candidate2": "Old production + v3 candidate 2",
        },
        "displayScopes": display_scopes,
        "inference": {
            "recordings": len(all_items),
            "labeledAndScored": len(evaluable),
            "unlabeledInferenceOnly": sorted(entry.prepared.recording.id for entry in all_items if not entry.labeled),
            "feedbackFeatureGenerationPerformed": False,
        },
        "metrics": metrics,
    }
    write_new_json(output / "report.json", report)
    atomic_write_text(output / "report.md", markdown_report(report))
    print(json.dumps({
        "output": str(output),
        "candidate1ModelId": candidate1_bundle["modelId"],
        "candidate2ModelId": candidate2_bundle["modelId"],
        "report": str(output / "report.md"),
    }, indent=2))


if __name__ == "__main__":
    main()
