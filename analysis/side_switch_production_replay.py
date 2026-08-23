"""Replay the two frozen production bundles over cached on-device F104 features."""

from __future__ import annotations

import base64
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from analysis.dead_ball_experiment import _prediction_inputs
from analysis.dead_state import DeadStateDecoderConfig
from analysis.dead_state_experiment import DeadStateRefinementConfig, _predictions_for
from analysis.features import FeatureSequence, VideoMetadata, contextualize
from analysis.model import LogisticModel, load_model
from analysis.pipeline import PreparedRecording
from analysis.schema import Recording
from analysis.serve import (
    ServeCompositionConfig,
    ServeDecoderConfig,
    decode_serve_probabilities,
)
from analysis.side_switch_production_state import (
    PRODUCTION_MODEL_SOURCES,
    ProductionTrace,
    ScoredTime,
    TimeRange,
)


@dataclass(frozen=True)
class ProductionHeads:
    rally: LogisticModel
    serve: LogisticModel
    dead: LogisticModel


def numeric_array(
    payload: Mapping[str, Any], dtype: str, shape: tuple[int, ...]
) -> np.ndarray:
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


def _recording(record: Mapping[str, Any], duration: float) -> Recording:
    roi = record.get("roi")
    roi_tuple = (
        (
            float(roi["x"]),
            float(roi["y"]),
            float(roi["width"]),
            float(roi["height"]),
        )
        if isinstance(roi, Mapping)
        else None
    )
    return Recording(
        id=str(record["recordingId"]),
        video=Path(str(record["videoPath"])),
        split="challenge",
        source_group=str(record.get("sourceGroup", "unknown")),
        environment=str(record.get("environment", "unknown")),
        game={},
        rallies=(),
        ignored_intervals=(),
        roi=roi_tuple,
        capture={},
        consent={"analyze": True, "train": False},
        content_sha256=None,
        raw={"duration": duration},
    )


def prepared_from_feedback(
    record: Mapping[str, Any], payload: Mapping[str, Any], template: LogisticModel
) -> PreparedRecording:
    features = payload["features"]
    rows, columns = int(features["rows"]), int(features["columns"])
    times = numeric_array(features["timestamps"], "float64", (rows,)).astype(np.float64)
    values = numeric_array(features["values"], "float32", (rows, columns)).astype(
        np.float32
    )
    names = tuple(str(item) for item in features["names"])
    media = payload["source"]["media"]
    duration = float(payload["source"]["gameWindow"]["end"])
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
    contextual_values, contextual_names = contextualize(sequence, template.feature_config)
    return PreparedRecording(
        recording=_recording(record, duration),
        sequence=sequence,
        contextual_values=contextual_values,
        contextual_names=contextual_names,
        labels=np.zeros(len(times), dtype=np.float32),
        sample_mask=np.ones(len(times), dtype=bool),
    )


def load_frozen_heads(
    v2_bundle: Path, old_model_root: Path, old_head_names: Mapping[str, str]
) -> tuple[ProductionHeads, ProductionHeads]:
    old = ProductionHeads(
        *(load_model(old_model_root / old_head_names[key]) for key in ("rally", "serve", "dead"))
    )
    bundle = json.loads(v2_bundle.read_text(encoding="utf-8"))["heads"]
    v2 = ProductionHeads(
        load_model(bundle["rally"]["path"]),
        load_model(bundle["serve"]["path"]),
        load_model(bundle["deadState"]["path"]),
    )
    signatures = {
        head.feature_names
        for group in (old, v2)
        for head in (group.rally, group.serve, group.dead)
    }
    configs = {
        json.dumps(head.feature_config.to_dict(), sort_keys=True)
        for group in (old, v2)
        for head in (group.rally, group.serve, group.dead)
    }
    if len(signatures) != 1 or len(configs) != 1:
        raise ValueError("production bundle feature contracts diverged")
    return old, v2


def _predict_ranges(
    item: PreparedRecording, heads: ProductionHeads
) -> tuple[TimeRange, ...]:
    serve_decoder = ServeDecoderConfig.from_dict(heads.serve.training_summary["serveDecoder"])
    composition = ServeCompositionConfig.from_dict(heads.serve.training_summary["composition"])
    dead_decoder = DeadStateDecoderConfig.from_dict(
        heads.dead.training_summary["selectedDeadStateDecoder"]
    )
    refinement = DeadStateRefinementConfig.from_dict(
        heads.dead.training_summary["selectedRefinement"]
    )
    inputs = _prediction_inputs(
        item, heads.rally, heads.serve, heads.dead, serve_decoder, composition
    )
    prediction = _predictions_for([inputs], dead_decoder, refinement)[0]
    return tuple(TimeRange(float(row.start), float(row.end)) for row in prediction.candidate)


def _scores(
    item: PreparedRecording, heads: ProductionHeads
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[ScoredTime, ...]]:
    rally = heads.rally.predict(item.contextual_values)
    serve = heads.serve.predict(item.contextual_values)
    dead = heads.dead.predict(item.contextual_values)
    decoder = ServeDecoderConfig.from_dict(heads.serve.training_summary["serveDecoder"])
    detections = decode_serve_probabilities(
        item.sequence.times,
        serve,
        decoder,
        duration=item.sequence.metadata.duration,
    )
    return (
        rally,
        serve,
        dead,
        tuple(ScoredTime(row.time, row.confidence) for row in detections),
    )


def replay_trace(
    item: PreparedRecording,
    old_heads: ProductionHeads,
    v2_heads: ProductionHeads,
) -> ProductionTrace:
    score_groups = {
        source: _scores(item, heads)
        for source, heads in zip(
            PRODUCTION_MODEL_SOURCES, (v2_heads, old_heads), strict=True
        )
    }
    trace = ProductionTrace(
        times=item.sequence.times,
        duration=item.sequence.metadata.duration,
        rally_scores={source: values[0] for source, values in score_groups.items()},
        serve_scores={source: values[1] for source, values in score_groups.items()},
        dead_state_scores={source: values[2] for source, values in score_groups.items()},
        ranges={
            source: _predict_ranges(item, heads)
            for source, heads in zip(
                PRODUCTION_MODEL_SOURCES, (v2_heads, old_heads), strict=True
            )
        },
        serves={source: values[3] for source, values in score_groups.items()},
    )
    trace.validate()
    return trace
