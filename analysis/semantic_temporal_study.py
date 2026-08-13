"""Leakage-safe study runner for the frozen-DINO temporal model.

This module owns sequence construction, fold-local targets/scalers, chunk
sampling, the deterministic interval decoder, and nested source-group
evaluation.  It intentionally does not load a visual backbone; the extractor
publishes validated :class:`DinoCache` objects before this module is called.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
import subprocess
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .config import DecoderConfig
from .decoder import decode_probabilities
from .dinov2_embeddings import DINO_EMBEDDING_DIMENSION, DINO_TOKEN_COUNT, DinoCache
from .features import FeatureSequence
from .metrics import aggregate_evaluations, aggregate_outcome_slices, evaluate_intervals
from .schema import Interval, Recording, labels_for_times, mask_for_times
from .semantic_temporal_model import (
    HEAD_NAMES,
    SemanticTemporalNetwork,
    TemporalModelConfig,
    fit_audiovisual_scaler,
    positive_weights,
    require_torch,
    seed_everything,
    standardize_audiovisual,
    trainable_parameter_count,
    weighted_temporal_loss,
)


SEMANTIC_TEMPORAL_SCHEMA_VERSION = 1
ANALYSIS_FPS = 4.0
TICK_SECONDS = 1.0 / ANALYSIS_FPS
CHUNK_LENGTH_TICKS = 256
CHUNK_STRIDE_TICKS = 128
BOUNDARY_SIGMA_SECONDS = 0.35
BOUNDARY_RADIUS_SECONDS = 1.0
DEVELOPMENT_SPLITS = frozenset({"train", "validation"})
SEED_VALUES = (3407, 1729, 20260812)


class SemanticTemporalStudyError(RuntimeError):
    """Raised when a temporal study would violate its data contract."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def objective(metrics: Mapping[str, Any]) -> float:
    return (
        0.55 * float(metrics.get("eventF1", 0.0))
        + 0.30 * float(metrics.get("timeIoU", 0.0))
        + 0.15 * float(metrics.get("liveTimeRecall", 0.0))
    )


@dataclass(frozen=True)
class SemanticTemporalExample:
    recording_id: str
    source_group: str
    split: str
    environment: str
    duration: float
    timestamps: np.ndarray
    dino_tokens: np.ndarray
    audiovisual: np.ndarray
    valid_mask: np.ndarray
    live_target: np.ndarray
    serve_target: np.ndarray
    end_target: np.ndarray
    hard_negative: np.ndarray
    rallies: tuple[Interval, ...]

    def validate(self) -> None:
        length = len(self.timestamps)
        if (
            self.timestamps.ndim != 1
            or not length
            or self.dino_tokens.shape != (length, DINO_TOKEN_COUNT, DINO_EMBEDDING_DIMENSION)
            or self.audiovisual.ndim != 2
            or self.audiovisual.shape[0] != length
            or self.valid_mask.shape != (length,)
            or self.live_target.shape != (length,)
            or self.serve_target.shape != (length,)
            or self.end_target.shape != (length,)
            or self.hard_negative.shape != (length,)
            or not np.all(np.isfinite(self.timestamps))
            or not np.all(np.isfinite(self.dino_tokens))
            or not np.all(np.isfinite(self.audiovisual))
        ):
            raise SemanticTemporalStudyError(
                f"{self.recording_id}: semantic temporal sequence shapes/values are invalid"
            )
        if len(self.timestamps) > 1 and not np.all(np.diff(self.timestamps) > 0):
            raise SemanticTemporalStudyError(f"{self.recording_id}: timestamps are not strictly increasing")
        if not math.isfinite(self.duration) or self.duration <= 0:
            raise SemanticTemporalStudyError(f"{self.recording_id}: duration is invalid")
        if self.audiovisual.shape[1] != 90:
            raise SemanticTemporalStudyError(
                f"{self.recording_id}: expected the existing 90 audiovisual signals"
            )


def _nearest_indexes(source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(source_times, target_times, side="left")
    positions = np.clip(positions, 0, len(source_times) - 1)
    previous = np.clip(positions - 1, 0, len(source_times) - 1)
    choose_previous = np.abs(source_times[previous] - target_times) < np.abs(
        source_times[positions] - target_times
    )
    return np.where(choose_previous, previous, positions).astype(np.int64)


def align_dino_to_audiovisual(
    dino_cache: DinoCache,
    target_times: np.ndarray,
    *,
    tolerance_seconds: float = TICK_SECONDS / 2.0,
) -> np.ndarray:
    if target_times.ndim != 1 or not len(target_times):
        raise ValueError("target_times must be a nonempty one-dimensional array")
    if dino_cache.timestamps.ndim != 1 or len(dino_cache.timestamps) == 0:
        raise SemanticTemporalStudyError("DINO cache has no timestamps")
    indexes = _nearest_indexes(dino_cache.timestamps, target_times)
    error = np.abs(dino_cache.timestamps[indexes] - target_times)
    if np.any(error > tolerance_seconds + 1e-9):
        index = int(np.argmax(error))
        raise SemanticTemporalStudyError(
            "DINO timestamps are not aligned to the audiovisual grid: "
            f"target={target_times[index]:.6f}, nearest={dino_cache.timestamps[indexes[index]]:.6f}"
        )
    return np.ascontiguousarray(dino_cache.tokens[indexes], dtype=np.float32)


def _boundary_pulse(times: np.ndarray, boundaries: Sequence[float]) -> np.ndarray:
    result = np.zeros(len(times), dtype=np.float32)
    for boundary in boundaries:
        distance = np.abs(times - float(boundary))
        values = np.exp(-0.5 * (distance / BOUNDARY_SIGMA_SECONDS) ** 2)
        values[distance > BOUNDARY_RADIUS_SECONDS] = 0.0
        result = np.maximum(result, values.astype(np.float32))
    return result


def _hard_negative_mask(recording: Recording, times: np.ndarray) -> np.ndarray:
    result = np.zeros(len(times), dtype=np.bool_)
    raw_intervals = recording.raw.get("hardNegatives", [])
    if not isinstance(raw_intervals, list):
        return result
    for item in raw_intervals:
        if not isinstance(item, Mapping):
            continue
        start = item.get("start")
        end = item.get("end")
        if isinstance(item.get("window"), Mapping):
            start = item["window"].get("start")
            end = item["window"].get("end")
        if isinstance(start, (int, float)) and not isinstance(start, bool) and isinstance(end, (int, float)) and not isinstance(end, bool):
            if math.isfinite(float(start)) and math.isfinite(float(end)) and float(end) > float(start):
                result |= (times >= float(start)) & (times < float(end))
    return result


def build_sequence_example(
    recording: Recording,
    dino_cache: DinoCache,
    audiovisual: FeatureSequence,
) -> SemanticTemporalExample:
    """Join validated caches and create fold-independent target arrays."""

    if audiovisual.values.ndim != 2 or audiovisual.values.shape[1] != 90:
        raise SemanticTemporalStudyError(
            f"{recording.id}: audiovisual cache must contain [T, 90] base signals"
        )
    times = np.asarray(audiovisual.times, dtype=np.float64)
    tokens = align_dino_to_audiovisual(dino_cache, times)
    if audiovisual.metadata.duration <= 0:
        raise SemanticTemporalStudyError(f"{recording.id}: audiovisual duration is invalid")
    example = SemanticTemporalExample(
        recording_id=recording.id,
        source_group=recording.source_group,
        split=recording.split,
        environment=recording.environment,
        duration=float(audiovisual.metadata.duration),
        timestamps=np.ascontiguousarray(times),
        dino_tokens=tokens,
        audiovisual=np.ascontiguousarray(audiovisual.values, dtype=np.float32),
        valid_mask=mask_for_times(times, recording.ignored_intervals),
        live_target=labels_for_times(times, recording.rallies),
        serve_target=_boundary_pulse(times, [item.start for item in recording.rallies]),
        end_target=_boundary_pulse(times, [item.end for item in recording.rallies]),
        hard_negative=_hard_negative_mask(recording, times),
        rallies=recording.rallies,
    )
    example.validate()
    return example


@dataclass(frozen=True)
class TemporalChunk:
    recording_id: str
    start_index: int
    length: int
    dino_tokens: np.ndarray
    audiovisual: np.ndarray
    live_target: np.ndarray
    serve_target: np.ndarray
    end_target: np.ndarray
    valid_mask: np.ndarray


def chunk_example(
    example: SemanticTemporalExample,
    *,
    chunk_length: int = CHUNK_LENGTH_TICKS,
    stride: int = CHUNK_STRIDE_TICKS,
) -> tuple[TemporalChunk, ...]:
    if chunk_length < 1 or stride < 1:
        raise ValueError("chunk_length and stride must be positive")
    length = len(example.timestamps)
    starts = list(range(0, max(1, length - chunk_length + 1), stride))
    final_start = max(0, length - chunk_length)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    result: list[TemporalChunk] = []
    for start in starts:
        end = min(length, start + chunk_length)
        actual = end - start
        dino = np.zeros((chunk_length, DINO_TOKEN_COUNT, DINO_EMBEDDING_DIMENSION), dtype=np.float32)
        audiovisual = np.zeros((chunk_length, example.audiovisual.shape[1]), dtype=np.float32)
        targets = [np.zeros(chunk_length, dtype=np.float32) for _ in range(3)]
        valid = np.zeros(chunk_length, dtype=np.bool_)
        dino[:actual] = example.dino_tokens[start:end]
        audiovisual[:actual] = example.audiovisual[start:end]
        targets[0][:actual] = example.live_target[start:end]
        targets[1][:actual] = example.serve_target[start:end]
        targets[2][:actual] = example.end_target[start:end]
        valid[:actual] = example.valid_mask[start:end]
        result.append(
            TemporalChunk(
                recording_id=example.recording_id,
                start_index=start,
                length=actual,
                dino_tokens=dino,
                audiovisual=audiovisual,
                live_target=targets[0],
                serve_target=targets[1],
                end_target=targets[2],
                valid_mask=valid,
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    dino_token_count: int
    use_audiovisual: bool
    decoder_mode: str

    def model_config(self) -> TemporalModelConfig:
        return TemporalModelConfig(
            dino_token_count=self.dino_token_count,
            use_audiovisual=self.use_audiovisual,
        )


def candidate_specs() -> tuple[CandidateSpec, ...]:
    return (
        CandidateSpec("audiovisual_tcn_control", 0, True, "live-only"),
        CandidateSpec("dino_class_only", 1, False, "live-only"),
        CandidateSpec("dino_class_plus_regions", 10, False, "live-only"),
        CandidateSpec("dino_regions_plus_audiovisual", 10, True, "live-only"),
        CandidateSpec("dino_regions_plus_audiovisual_boundary_heads", 10, True, "boundary"),
        CandidateSpec("dino_regions_plus_audiovisual_short_path", 10, True, "short"),
    )


@dataclass(frozen=True)
class SemanticTrainingConfig:
    max_epochs: int = 40
    patience: int = 6
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    seed: int = 3407

    def validate(self) -> None:
        if self.max_epochs < 1 or self.patience < 1 or self.batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("epoch, patience, batch, and accumulation values must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative")
        if not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "maxEpochs": self.max_epochs,
            "patience": self.patience,
            "batchSize": self.batch_size,
            "gradientAccumulationSteps": self.gradient_accumulation_steps,
            "learningRate": self.learning_rate,
            "weightDecay": self.weight_decay,
            "gradientClipNorm": self.gradient_clip_norm,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class SemanticDecoderConfig:
    live_decoder: DecoderConfig = DecoderConfig()
    use_boundary_heads: bool = False
    use_short_path: bool = False
    serve_threshold: float = 0.65
    end_threshold: float = 0.65
    short_serve_threshold: float = 0.75
    short_end_threshold: float = 0.75

    def validate(self) -> None:
        self.live_decoder.validate()
        for name, value in (
            ("serve_threshold", self.serve_threshold),
            ("end_threshold", self.end_threshold),
            ("short_serve_threshold", self.short_serve_threshold),
            ("short_end_threshold", self.short_end_threshold),
        ):
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "liveDecoder": self.live_decoder.to_dict(),
            "useBoundaryHeads": self.use_boundary_heads,
            "useShortPath": self.use_short_path,
            "serveThreshold": self.serve_threshold,
            "endThreshold": self.end_threshold,
            "shortServeThreshold": self.short_serve_threshold,
            "shortEndThreshold": self.short_end_threshold,
        }


def _model_inputs(
    chunks: Sequence[TemporalChunk],
    spec: CandidateSpec,
    mean: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, np.ndarray], np.ndarray]:
    if not chunks:
        raise ValueError("at least one chunk is required")
    dino: np.ndarray | None = None
    if spec.dino_token_count:
        dino = np.stack([item.dino_tokens[:, : spec.dino_token_count] for item in chunks]).astype(np.float32)
    audiovisual: np.ndarray | None = None
    if spec.use_audiovisual:
        audiovisual = np.stack(
            [standardize_audiovisual(item.audiovisual, mean, scale) for item in chunks]
        ).astype(np.float32)
    targets = {
        "live": np.stack([item.live_target for item in chunks]).astype(np.float32),
        "serve": np.stack([item.serve_target for item in chunks]).astype(np.float32),
        "end": np.stack([item.end_target for item in chunks]).astype(np.float32),
    }
    valid = np.stack([item.valid_mask for item in chunks]).astype(np.bool_)
    return dino, audiovisual, targets, valid


def _torch_batch(
    arrays: tuple[np.ndarray | None, np.ndarray | None, dict[str, np.ndarray], np.ndarray],
    device: Any,
) -> tuple[Any | None, Any | None, dict[str, Any], Any]:
    torch = require_torch()
    dino, audiovisual, targets, valid = arrays
    return (
        torch.from_numpy(dino).to(device) if dino is not None else None,
        torch.from_numpy(audiovisual).to(device) if audiovisual is not None else None,
        {name: torch.from_numpy(values).to(device) for name, values in targets.items()},
        torch.from_numpy(valid).to(device),
    )


def _probabilities_from_chunks(
    model: Any,
    example: SemanticTemporalExample,
    spec: CandidateSpec,
    mean: np.ndarray,
    scale: np.ndarray,
    device: Any,
) -> dict[str, np.ndarray]:
    torch = require_torch()
    chunks = chunk_example(example)
    total = len(example.timestamps)
    sums = {name: np.zeros(total, dtype=np.float64) for name in HEAD_NAMES}
    counts = np.zeros(total, dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for chunk in chunks:
            arrays = _model_inputs((chunk,), spec, mean, scale)
            dino, audiovisual, _, _ = _torch_batch(arrays, device)
            logits = model(dino, audiovisual)
            for name in HEAD_NAMES:
                values = torch.sigmoid(logits[name][0]).detach().cpu().numpy().astype(np.float64)
                sums[name][chunk.start_index : chunk.start_index + chunk.length] += values[: chunk.length]
            counts[chunk.start_index : chunk.start_index + chunk.length] += 1.0
    if np.any(counts <= 0):
        raise SemanticTemporalStudyError(f"{example.recording_id}: chunk inference left uncovered ticks")
    return {
        name: np.ascontiguousarray((sums[name] / counts).astype(np.float32))
        for name in HEAD_NAMES
    }


def _score_intervals(
    example: SemanticTemporalExample,
    probabilities: Mapping[str, np.ndarray],
    decoder: SemanticDecoderConfig,
) -> list[Interval]:
    decoder.validate()
    live = np.asarray(probabilities["live"], dtype=np.float32)
    if live.shape != example.timestamps.shape:
        raise SemanticTemporalStudyError(f"{example.recording_id}: live probabilities are misaligned")
    fps = 1.0 / float(np.median(np.diff(example.timestamps))) if len(example.timestamps) > 1 else ANALYSIS_FPS
    proposals, _ = decode_probabilities(
        example.timestamps,
        live,
        example.duration,
        decoder.live_decoder,
        fps,
    )
    intervals = [Interval(item.start, item.end) for item in proposals]
    if decoder.use_boundary_heads:
        serve = np.asarray(probabilities["serve"], dtype=np.float32)
        end = np.asarray(probabilities["end"], dtype=np.float32)
        refined: list[Interval] = []
        for interval in intervals:
            start = interval.start
            finish = interval.end
            start_indexes = np.flatnonzero(
                (example.timestamps >= max(0.0, interval.start - 2.0))
                & (example.timestamps <= min(example.duration, interval.start + 1.0))
                & (serve >= decoder.serve_threshold)
            )
            end_indexes = np.flatnonzero(
                (example.timestamps >= max(0.0, interval.end - 1.0))
                & (example.timestamps <= min(example.duration, interval.end + 3.0))
                & (end >= decoder.end_threshold)
            )
            if len(start_indexes):
                start = float(example.timestamps[start_indexes[int(np.argmax(serve[start_indexes]))]])
            if len(end_indexes):
                finish = float(example.timestamps[end_indexes[int(np.argmax(end[end_indexes]))]]) + TICK_SECONDS
            if finish > start:
                refined.append(Interval(max(0.0, start), min(example.duration, finish)))
        intervals = refined
    if decoder.use_short_path:
        serve = np.asarray(probabilities["serve"], dtype=np.float32)
        end = np.asarray(probabilities["end"], dtype=np.float32)
        serve_peaks = [
            index
            for index in range(len(serve))
            if serve[index] >= decoder.short_serve_threshold
            and (index == 0 or serve[index] >= serve[index - 1])
            and (index + 1 == len(serve) or serve[index] >= serve[index + 1])
        ]
        end_peaks = [
            index
            for index in range(len(end))
            if end[index] >= decoder.short_end_threshold
            and (index == 0 or end[index] >= end[index - 1])
            and (index + 1 == len(end) or end[index] >= end[index + 1])
        ]
        for serve_index in serve_peaks:
            end_index = next(
                (
                    item
                    for item in end_peaks
                    if 0.25 <= example.timestamps[item] - example.timestamps[serve_index] <= 3.0
                ),
                None,
            )
            if end_index is None:
                continue
            candidate = Interval(
                float(example.timestamps[serve_index]),
                min(example.duration, float(example.timestamps[end_index]) + TICK_SECONDS),
            )
            if candidate.end <= candidate.start:
                continue
            if not any(item.start < candidate.end and candidate.start < item.end for item in intervals):
                intervals.append(candidate)
    intervals.sort(key=lambda item: (item.start, item.end))
    merged: list[Interval] = []
    for item in intervals:
        if merged and item.start < merged[-1].end:
            merged[-1] = Interval(merged[-1].start, max(merged[-1].end, item.end))
        else:
            merged.append(item)
    return merged


def evaluate_probability_rows(
    examples: Sequence[SemanticTemporalExample],
    probability_rows: Sequence[Mapping[str, np.ndarray]],
    decoder: SemanticDecoderConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(examples) != len(probability_rows):
        raise ValueError("examples and probability rows are not aligned")
    rows: list[dict[str, Any]] = []
    for example, probabilities in zip(examples, probability_rows, strict=True):
        predictions = _score_intervals(example, probabilities, decoder)
        metrics = evaluate_intervals(example.rallies, predictions)
        metrics["id"] = example.recording_id
        metrics["sourceGroup"] = example.source_group
        metrics["environment"] = example.environment
        metrics["outcomeSlices"] = _outcome_slices(example.rallies, predictions)
        rows.append(metrics)
    if not rows:
        raise ValueError("cannot evaluate zero examples")
    aggregate = aggregate_evaluations(rows)
    aggregate["outcomeSlices"] = aggregate_outcome_slices([row["outcomeSlices"] for row in rows])
    aggregate["objective"] = objective(aggregate)
    return rows, aggregate


def _outcome_slices(truth: Sequence[Interval], predictions: Sequence[Interval]) -> dict[str, dict[str, Any]]:
    indexes = {
        "all": list(range(len(truth))),
        "shortAtMost3Seconds": [index for index, item in enumerate(truth) if item.end - item.start <= 3.0],
        "ace": [index for index, item in enumerate(truth) if "ace" in item.tags],
        "serviceFault": [index for index, item in enumerate(truth) if "service-fault" in item.tags],
        "ordinaryLong": [
            index for index, item in enumerate(truth)
            if item.end - item.start > 3.0 and not ({"ace", "service-fault"} & set(item.tags))
        ],
    }
    from .metrics import truth_slice_metrics

    return {
        name: truth_slice_metrics(truth, predictions, values)
        for name, values in indexes.items()
    }


def select_decoder(
    examples: Sequence[SemanticTemporalExample],
    probability_rows: Sequence[Mapping[str, np.ndarray]],
    *,
    mode: str,
) -> tuple[SemanticDecoderConfig, dict[str, Any]]:
    """Select thresholds from inner-fold predictions only."""

    if mode not in {"live-only", "boundary", "short"}:
        raise ValueError(f"unknown decoder mode {mode!r}")
    best: SemanticDecoderConfig | None = None
    best_metrics: dict[str, Any] | None = None
    best_score = -math.inf
    count = 0
    for smoothing in (0.5, 1.0, 1.5):
        for enter in (0.35, 0.45, 0.55, 0.65):
            for exit_delta in (0.05, 0.10):
                for minimum in (0.5, 1.0, 2.0):
                    exit_threshold = max(0.05, enter - exit_delta)
                    base = DecoderConfig(
                        smoothing_seconds=smoothing,
                        enter_threshold=enter,
                        exit_threshold=exit_threshold,
                        min_live_seconds=minimum,
                        bridge_gap_seconds=1.0,
                        short_event_min_seconds=0.5,
                        short_event_threshold=max(enter, 0.8),
                    )
                    for serve_threshold in ((0.55, 0.7) if mode != "live-only" else (0.65,)):
                        candidate = SemanticDecoderConfig(
                            live_decoder=base,
                            use_boundary_heads=mode in {"boundary", "short"},
                            use_short_path=mode == "short",
                            serve_threshold=serve_threshold,
                            end_threshold=serve_threshold,
                            short_serve_threshold=serve_threshold,
                            short_end_threshold=serve_threshold,
                        )
                        _, metrics = evaluate_probability_rows(examples, probability_rows, candidate)
                        score = objective(metrics)
                        count += 1
                        tie = (
                            float(metrics["eventF1"]),
                            float(metrics["liveTimePrecision"]),
                            -abs(int(metrics["predictedRallies"]) - int(metrics["trueRallies"])),
                            float(metrics["timeIoU"]),
                        )
                        old_tie = (
                            (float(best_metrics["eventF1"]), float(best_metrics["liveTimePrecision"]), -abs(int(best_metrics["predictedRallies"]) - int(best_metrics["trueRallies"])), float(best_metrics["timeIoU"]))
                            if best_metrics is not None else (-math.inf, -math.inf, -math.inf, -math.inf)
                        )
                        if score > best_score + 1e-9 or (abs(score - best_score) <= 1e-9 and tie > old_tie):
                            best_score = score
                            best = candidate
                            best_metrics = metrics
    if best is None or best_metrics is None:
        raise SemanticTemporalStudyError("decoder search selected no configuration")
    return best, {
        "candidateCount": count,
        "objective": "0.55*eventF1 + 0.30*timeIoU + 0.15*liveTimeRecall",
        "objectiveValue": best_score,
        "selected": best.to_dict(),
        "validationMetrics": best_metrics,
    }


@dataclass(frozen=True)
class FitResult:
    model: Any
    candidate: CandidateSpec
    mean: np.ndarray
    scale: np.ndarray
    positive_weights: Mapping[str, float]
    best_epoch: int
    epochs_completed: int
    trainable_parameters: int
    logs: tuple[Mapping[str, Any], ...]


def _fit_model(
    training: Sequence[SemanticTemporalExample],
    validation: Sequence[SemanticTemporalExample],
    candidate: CandidateSpec,
    config: SemanticTrainingConfig,
    *,
    device_name: str,
    epoch_cap: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> FitResult:
    torch = require_torch()
    config.validate()
    if not training:
        raise SemanticTemporalStudyError("cannot train a temporal model with no examples")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SemanticTemporalStudyError("CUDA was requested but is unavailable")
    seed_everything(config.seed)
    mean, scale = (
        fit_audiovisual_scaler(
            [item.audiovisual for item in training],
            [item.valid_mask for item in training],
        )
        if candidate.use_audiovisual
        else (np.zeros(90, dtype=np.float32), np.ones(90, dtype=np.float32))
    )
    train_chunks = tuple(chunk for item in training for chunk in chunk_example(item))
    validation_chunks = tuple(chunk for item in validation for chunk in chunk_example(item))
    target_arrays = {
        name: [np.asarray(getattr(item, f"{name}_target"), dtype=np.float32) for item in training]
        for name in HEAD_NAMES
    }
    weights = positive_weights(target_arrays, [item.valid_mask for item in training])
    model = SemanticTemporalNetwork(candidate.model_config()).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    maximum_epochs = min(config.max_epochs, epoch_cap) if epoch_cap is not None else config.max_epochs
    best_state: dict[str, Any] | None = None
    best_score = -math.inf
    best_epoch = 1
    stale = 0
    logs: list[Mapping[str, Any]] = []
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        order = np.random.default_rng(config.seed + epoch).permutation(len(train_chunks))
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for batch_number, offset in enumerate(range(0, len(order), config.batch_size), start=1):
            indexes = order[offset : offset + config.batch_size]
            rows = tuple(train_chunks[int(index)] for index in indexes)
            arrays = _model_inputs(rows, candidate, mean, scale)
            dino, audiovisual, targets, valid = _torch_batch(arrays, device)
            logits = model(dino, audiovisual)
            loss, _ = weighted_temporal_loss(logits, targets, valid, weights)
            (loss / config.gradient_accumulation_steps).backward()
            total_loss += float(loss.detach().cpu())
            should_step = batch_number % config.gradient_accumulation_steps == 0 or offset + len(indexes) >= len(order)
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if validation:
            probabilities = [_probabilities_from_chunks(model, item, candidate, mean, scale, device) for item in validation]
            _, validation_metrics = evaluate_probability_rows(
                validation,
                probabilities,
                SemanticDecoderConfig(),
            )
            score = objective(validation_metrics)
        else:
            # The outer refit deliberately has no validation set: its epoch
            # cap was selected from the inner folds.  Keep the log explicitly
            # JSON-safe instead of using NaN as a sentinel.
            validation_metrics = {"objective": None}
            score = -float(total_loss) / max(1, len(order))
        row = {
            "epoch": epoch,
            "trainLoss": total_loss / max(1, math.ceil(len(order) / config.batch_size)),
            "validationObjective": score,
            "validation": validation_metrics,
        }
        logs.append(row)
        if progress is not None:
            progress(f"{candidate.name}: epoch {epoch}/{maximum_epochs}, score={score:.5f}")
        if score > best_score + 1e-9:
            best_score = score
            best_epoch = epoch
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if validation and stale >= config.patience:
                break
    if best_state is None:
        raise SemanticTemporalStudyError("temporal training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return FitResult(
        model=model,
        candidate=candidate,
        mean=mean,
        scale=scale,
        positive_weights=weights,
        best_epoch=best_epoch,
        epochs_completed=len(logs),
        trainable_parameters=trainable_parameter_count(model),
        logs=tuple(logs),
    )


def _group_folds(examples: Sequence[SemanticTemporalExample]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    groups = tuple(sorted({item.source_group for item in examples}))
    if len(groups) < 2:
        raise SemanticTemporalStudyError("nested source-group study requires at least two groups")
    return tuple((held, tuple(group for group in groups if group != held)) for held in groups)


def _paired_gate(
    control_rows: Mapping[tuple[int, str], Mapping[str, Any]],
    candidate_rows: Mapping[tuple[int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    keys = sorted(set(control_rows) & set(candidate_rows))
    deltas = [objective(candidate_rows[key]) - objective(control_rows[key]) for key in keys]
    by_seed: dict[int, list[float]] = {}
    by_group: dict[str, list[float]] = {}
    for key, delta in zip(keys, deltas, strict=True):
        by_seed.setdefault(key[0], []).append(delta)
        by_group.setdefault(key[1], []).append(delta)
    seed_deltas = {str(seed): float(np.mean(values)) for seed, values in by_seed.items()}
    group_deltas = {group: float(np.mean(values)) for group, values in by_group.items()}
    return {
        "pairs": len(keys),
        "meanObjectiveDelta": float(np.mean(deltas)) if deltas else 0.0,
        "medianObjectiveDelta": float(np.median(deltas)) if deltas else 0.0,
        "positiveSeedDeltaEverySeed": bool(seed_deltas) and all(value > 0 for value in seed_deltas.values()),
        "positiveGroups": sum(value > 0 for value in group_deltas.values()),
        "seedDeltas": seed_deltas,
        "groupDeltas": group_deltas,
    }


def _code_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    paths = (
        root / "analysis" / "dinov2_embeddings.py",
        root / "analysis" / "semantic_temporal_model.py",
        root / "analysis" / "semantic_temporal_study.py",
    )
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        head, dirty = "unknown", True
    return {
        "gitHead": head,
        "gitDirty": dirty,
        "python": platform.python_version(),
        "filesSha256": {str(path.relative_to(root)): _sha256_file(path) for path in paths},
    }


def run_nested_development_study(
    examples: Sequence[SemanticTemporalExample],
    *,
    seeds: Sequence[int] = SEED_VALUES,
    device: str = "cuda",
    training_config: SemanticTrainingConfig | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run candidates 1–6 with nested leave-one-source-group-out development folds."""

    started = time.perf_counter()
    rows = tuple(examples)
    if not rows or any(item.split not in DEVELOPMENT_SPLITS for item in rows):
        raise SemanticTemporalStudyError("development study accepts train+validation recordings only")
    for item in rows:
        item.validate()
    if not seeds:
        raise ValueError("at least one seed is required")
    base_config = training_config or SemanticTrainingConfig()
    folds = _group_folds(rows)
    candidates = candidate_specs()
    candidate_reports: dict[str, Any] = {}
    result_rows: dict[str, dict[tuple[int, str], dict[str, Any]]] = {}
    total = len(candidates) * len(seeds) * len(folds)
    completed = 0
    for candidate in candidates:
        candidate_key_rows: dict[tuple[int, str], dict[str, Any]] = {}
        detailed: list[dict[str, Any]] = []
        for seed in seeds:
            for held_group, training_groups in folds:
                completed += 1
                if progress is not None:
                    progress(f"semantic study {completed}/{total}: {candidate.name}, seed={seed}, held={held_group}")
                outer_training = [item for item in rows if item.source_group in training_groups]
                held = [item for item in rows if item.source_group == held_group]
                inner_probabilities: list[Mapping[str, np.ndarray]] = []
                inner_examples: list[SemanticTemporalExample] = []
                inner_epochs: list[int] = []
                inner_groups = tuple(sorted(training_groups))
                for inner_validation_group in inner_groups:
                    inner_training = [item for item in outer_training if item.source_group != inner_validation_group]
                    inner_validation = [item for item in outer_training if item.source_group == inner_validation_group]
                    fit = _fit_model(
                        inner_training,
                        inner_validation,
                        candidate,
                        replace(base_config, seed=int(seed)),
                        device_name=device,
                        progress=None,
                    )
                    inner_epochs.append(fit.best_epoch)
                    inner_examples.extend(inner_validation)
                    inner_probabilities.extend(
                        _probabilities_from_chunks(fit.model, item, candidate, fit.mean, fit.scale, require_torch().device(device))
                        for item in inner_validation
                    )
                decoder, decoder_selection = select_decoder(
                    inner_examples,
                    inner_probabilities,
                    mode=candidate.decoder_mode,
                )
                epoch_cap = max(1, int(round(float(np.median(inner_epochs)))))
                outer_fit = _fit_model(
                    outer_training,
                    (),
                    candidate,
                    replace(base_config, seed=int(seed)),
                    device_name=device,
                    epoch_cap=epoch_cap,
                    progress=None,
                )
                torch = require_torch()
                held_probabilities = [
                    _probabilities_from_chunks(
                        outer_fit.model,
                        item,
                        candidate,
                        outer_fit.mean,
                        outer_fit.scale,
                        torch.device(device),
                    )
                    for item in held
                ]
                held_rows, held_metrics = evaluate_probability_rows(held, held_probabilities, decoder)
                for row in held_rows:
                    candidate_key_rows[(int(seed), str(row["id"]))] = row
                detailed.append(
                    {
                        "seed": int(seed),
                        "heldOutSourceGroup": held_group,
                        "outerTrainingSourceGroups": list(training_groups),
                        "innerBestEpochs": inner_epochs,
                        "epochCap": epoch_cap,
                        "decoder": decoder.to_dict(),
                        "decoderSelection": decoder_selection,
                        "trainableParameters": outer_fit.trainable_parameters,
                        "metrics": held_metrics,
                        "recordings": held_rows,
                        "trainingLogs": list(outer_fit.logs),
                    }
                )
        all_recordings = list(candidate_key_rows.values())
        aggregate = aggregate_evaluations(all_recordings)
        aggregate["outcomeSlices"] = aggregate_outcome_slices([row["outcomeSlices"] for row in all_recordings])
        aggregate["objective"] = objective(aggregate)
        candidate_reports[candidate.name] = {
            "candidate": candidate.__dict__,
            "aggregate": aggregate,
            "outerFolds": detailed,
            "bySeed": {
                str(seed): {
                    "aggregate": _aggregate_rows([row for key, row in candidate_key_rows.items() if key[0] == seed]),
                }
                for seed in seeds
            },
        }
        result_rows[candidate.name] = candidate_key_rows
    control_name = candidates[0].name
    comparisons: dict[str, Any] = {}
    for candidate in candidates[1:]:
        comparisons[candidate.name] = _paired_gate(result_rows[control_name], result_rows[candidate.name])
    selected_name = control_name
    gate_results: dict[str, Any] = {}
    control_aggregate = candidate_reports[control_name]["aggregate"]
    for candidate in candidates[1:]:
        candidate_aggregate = candidate_reports[candidate.name]["aggregate"]
        comparison = comparisons[candidate.name]
        short_control = control_aggregate.get("outcomeSlices", {}).get("shortAtMost3Seconds", {}).get("strictMatchRecall", 0.0)
        short_candidate = candidate_aggregate.get("outcomeSlices", {}).get("shortAtMost3Seconds", {}).get("strictMatchRecall", 0.0)
        fault_control = control_aggregate.get("outcomeSlices", {}).get("serviceFault", {}).get("strictMatchRecall", 0.0)
        fault_candidate = candidate_aggregate.get("outcomeSlices", {}).get("serviceFault", {}).get("strictMatchRecall", 0.0)
        passed = (
            float(candidate_aggregate["eventF1"]) >= float(control_aggregate["eventF1"]) + 0.02
            and float(candidate_aggregate["timeIoU"]) >= float(control_aggregate["timeIoU"]) + 0.02
            and float(candidate_aggregate["liveTimeRecall"]) >= float(control_aggregate["liveTimeRecall"]) - 0.01
            and comparison["positiveSeedDeltaEverySeed"]
            and comparison["positiveGroups"] >= max(1, len(folds) - 1)
            and short_candidate >= short_control
            and fault_candidate >= fault_control
        )
        gate_results[candidate.name] = {
            "passed": passed,
            "eventF1Delta": float(candidate_aggregate["eventF1"]) - float(control_aggregate["eventF1"]),
            "timeIoUDelta": float(candidate_aggregate["timeIoU"]) - float(control_aggregate["timeIoU"]),
            "liveTimeRecallDelta": float(candidate_aggregate["liveTimeRecall"]) - float(control_aggregate["liveTimeRecall"]),
            "shortRecallDelta": float(short_candidate) - float(short_control),
            "serviceFaultRecallDelta": float(fault_candidate) - float(fault_control),
            "paired": comparison,
        }
        if passed and float(candidate_aggregate["objective"]) > float(candidate_reports[selected_name]["aggregate"]["objective"]):
            selected_name = candidate.name
    report = {
        "schemaVersion": SEMANTIC_TEMPORAL_SCHEMA_VERSION,
        "kind": "volleycut-dinov2-temporal-study-development",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "freezeStatus": "frozen-development-selection",
        "assessmentRole": "development-only-nested-source-group-evaluation",
        "testLabelsUsed": False,
        "testRecordingsPrepared": False,
        "analysisFps": ANALYSIS_FPS,
        "chunking": {"lengthTicks": CHUNK_LENGTH_TICKS, "strideTicks": CHUNK_STRIDE_TICKS},
        "sourceGroups": sorted({item.source_group for item in rows}),
        "recordings": sorted(item.recording_id for item in rows),
        "seeds": [int(item) for item in seeds],
        "trainingConfig": base_config.to_dict(),
        "candidateOrder": [item.name for item in candidates],
        "candidateReports": candidate_reports,
        "pairedComparisonsAgainstControl": comparisons,
        "promotionGate": {
            "requirements": {
                "eventF1Delta": 0.02,
                "timeIoUDelta": 0.02,
                "liveRecallLossMaximum": 0.01,
                "positiveSeedDelta": True,
                "minimumPositiveOuterGroups": max(1, len(folds) - 1),
                "shortAndServiceFaultRecallNonDecreasing": True,
            },
            "candidates": gate_results,
        },
        "selectedCandidate": selected_name,
        "promotionDecision": {
            "promotionGatePassed": selected_name != control_name,
            "selectedArchitecture": selected_name,
            "decision": "promote-selected-candidate" if selected_name != control_name else "retain-audiovisual-tcn-control",
            "testOpened": False,
        },
        "runtime": {
            "wallClockSeconds": round(time.perf_counter() - started, 3),
            "platform": platform.platform(),
        },
        "codeProvenance": _code_provenance(),
        "protectedTest": {"opened": False, "reason": "development report does not access test"},
    }
    return report


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"recordings": 0, "eventF1": 0.0, "timeIoU": 0.0, "liveTimeRecall": 0.0, "objective": 0.0}
    result = aggregate_evaluations(list(rows))
    result["outcomeSlices"] = aggregate_outcome_slices([row["outcomeSlices"] for row in rows])
    result["objective"] = objective(result)
    return result


def write_report_no_replace(path: str | Path, report: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite semantic temporal report: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(report), indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary report path already exists: {temporary}")
    try:
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination
