"""Reusable production serve-head evidence for candidate-level serve gating."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import FeatureConfig
from .serve import ServeDecoderConfig, ServeDetection, decode_serve_probabilities


GATE_VERSION = "production-dual-serve-head-anchor-window-v1"
ANCHOR_WINDOW_SECONDS = 1.0
GATE_AGGREGATION = "either-head-at-production-threshold"
HYBRID_GATE_VERSION = "production-serve-head-plus-both-rally-anchor-v2"
HYBRID_GATE_AGGREGATION = (
    "either-serve-head-or-anchor-contained-in-both-model-rally"
)


class ProductionServeGateError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProductionServeHead:
    model_id: str
    bundle_sha256: str
    feature_config: FeatureConfig
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    decoder: ServeDecoderConfig

    def predict(self, contextual_values: np.ndarray) -> np.ndarray:
        values = np.asarray(contextual_values, dtype=np.float32)
        expected = len(self.feature_names)
        if values.ndim != 2 or values.shape[1] != expected:
            raise ProductionServeGateError(
                f"{self.model_id} expects (*, {expected}) features, got {values.shape}"
            )
        normalized = (values - self.mean) / self.scale
        logits = np.clip(normalized @ self.weights + np.float32(self.bias), -30, 30)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


@dataclass(frozen=True)
class ServeHeadAnchorEvidence:
    model_id: str
    threshold: float
    peak_probability: float
    peak_time: float
    nearest_detection: ServeDetection | None

    def to_dict(self, anchor: float) -> dict[str, Any]:
        nearest = self.nearest_detection
        return {
            "modelId": self.model_id,
            "threshold": self.threshold,
            "peakProbability": self.peak_probability,
            "peakTime": self.peak_time,
            "crossesThreshold": self.peak_probability >= self.threshold,
            "nearestDetection": (
                {
                    "time": nearest.time,
                    "confidence": nearest.confidence,
                    "distanceSeconds": abs(nearest.time - anchor),
                }
                if nearest is not None
                else None
            ),
        }


@dataclass(frozen=True)
class ProductionRallyInterval:
    start: float
    end: float
    agreement: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "agreement": self.agreement,
        }


@dataclass(frozen=True)
class ProductionRallyAnchorEvidence:
    anchor_contained: bool
    interval: ProductionRallyInterval | None

    @property
    def both_models(self) -> bool:
        return self.interval is not None and self.interval.agreement == "both-models"

    @property
    def recovers_serve(self) -> bool:
        return self.anchor_contained and self.both_models

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchorContained": self.anchor_contained,
            "bothModels": self.both_models,
            "recoversServe": self.recovers_serve,
            "interval": self.interval.to_dict() if self.interval is not None else None,
        }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric_vector(
    value: Any, expected: int, label: str, *, positive: bool = False
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (expected,) or not np.isfinite(result).all():
        raise ProductionServeGateError(f"{label} must contain {expected} finite values")
    if positive and np.any(result <= 0):
        raise ProductionServeGateError(f"{label} must be positive")
    return result


def load_production_serve_head(path: str | Path) -> ProductionServeHead:
    bundle_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProductionServeGateError(
            f"cannot read production model bundle {bundle_path}: {error}"
        ) from error
    if not isinstance(payload, Mapping) or payload.get("schemaVersion") != 1:
        raise ProductionServeGateError("unsupported production model bundle")
    model_id = payload.get("modelId")
    names = payload.get("featureNames")
    serve = payload.get("serve")
    if (
        not isinstance(model_id, str)
        or not model_id
        or not isinstance(names, list)
        or not names
        or not all(isinstance(name, str) and name for name in names)
        or not isinstance(serve, Mapping)
        or not isinstance(serve.get("decoder"), Mapping)
        or not isinstance(payload.get("featureConfig"), Mapping)
    ):
        raise ProductionServeGateError("production serve-head contract is malformed")
    feature_names = tuple(names)
    expected = len(feature_names)
    bias = serve.get("bias")
    if not isinstance(bias, (int, float)) or not math.isfinite(float(bias)):
        raise ProductionServeGateError("production serve-head bias is invalid")
    return ProductionServeHead(
        model_id=model_id,
        bundle_sha256=sha256(bundle_path),
        feature_config=FeatureConfig.from_dict(dict(payload["featureConfig"])),
        feature_names=feature_names,
        mean=_numeric_vector(serve.get("mean"), expected, "serve mean"),
        scale=_numeric_vector(
            serve.get("scale"), expected, "serve scale", positive=True
        ),
        weights=_numeric_vector(serve.get("weights"), expected, "serve weights"),
        bias=float(bias),
        decoder=ServeDecoderConfig.from_dict(dict(serve["decoder"])),
    )


def anchor_evidence(
    head: ProductionServeHead,
    times: np.ndarray,
    probabilities: np.ndarray,
    detections: Sequence[ServeDetection],
    anchor: float,
    *,
    window_seconds: float = ANCHOR_WINDOW_SECONDS,
) -> ServeHeadAnchorEvidence:
    timestamps = np.asarray(times, dtype=np.float64)
    scores = np.asarray(probabilities, dtype=np.float32)
    if (
        timestamps.ndim != 1
        or scores.shape != timestamps.shape
        or not len(timestamps)
        or not np.isfinite(timestamps).all()
        or not np.isfinite(scores).all()
        or np.any((scores < 0) | (scores > 1))
    ):
        raise ProductionServeGateError("serve evidence needs aligned probability rows")
    if not math.isfinite(anchor) or anchor < 0:
        raise ProductionServeGateError("serve anchor must be finite and non-negative")
    if not math.isfinite(window_seconds) or window_seconds < 0:
        raise ProductionServeGateError("serve anchor window must be non-negative")
    distance = np.abs(timestamps - anchor)
    selected = np.flatnonzero(distance <= window_seconds + 1e-9)
    if not len(selected):
        selected = np.asarray([int(np.argmin(distance))])
    peak_index = int(selected[int(np.argmax(scores[selected]))])
    nearest = (
        min(detections, key=lambda detection: abs(detection.time - anchor))
        if detections
        else None
    )
    return ServeHeadAnchorEvidence(
        model_id=head.model_id,
        threshold=head.decoder.threshold,
        peak_probability=float(scores[peak_index]),
        peak_time=float(timestamps[peak_index]),
        nearest_detection=nearest,
    )


def infer_head(
    head: ProductionServeHead,
    times: np.ndarray,
    contextual_values: np.ndarray,
    *,
    duration: float,
) -> tuple[np.ndarray, list[ServeDetection]]:
    probabilities = head.predict(contextual_values)
    detections = decode_serve_probabilities(
        np.asarray(times, dtype=np.float64),
        probabilities,
        head.decoder,
        duration=duration,
    )
    return probabilities, detections


def dual_head_prediction(evidence: Sequence[ServeHeadAnchorEvidence]) -> str:
    if len(evidence) != 2:
        raise ProductionServeGateError("the production serve gate requires two heads")
    return (
        "serve"
        if any(item.peak_probability >= item.threshold for item in evidence)
        else "not-serve"
    )


def merge_production_rally_intervals(
    all_labels_v2: Sequence[Mapping[str, Any]],
    previous_production: Sequence[Mapping[str, Any]],
) -> tuple[ProductionRallyInterval, ...]:
    """Mirror production's overlap-union-disagreement-v1 interval merge."""

    candidates: list[tuple[float, float, str]] = []
    for source, rows in (
        ("all-labels-v2", all_labels_v2),
        ("previous-production", previous_production),
    ):
        for row in rows:
            if row.get("included", True) is not True:
                continue
            start = row.get("start")
            end = row.get("end")
            if (
                not isinstance(start, (int, float))
                or not isinstance(end, (int, float))
                or not math.isfinite(float(start))
                or not math.isfinite(float(end))
                or float(end) <= float(start)
            ):
                raise ProductionServeGateError(
                    "production rally intervals must have finite positive duration"
                )
            candidates.append((float(start), float(end), source))
    candidates.sort(key=lambda item: (item[0], item[1]))
    clusters: list[list[tuple[float, float, str]]] = []
    for candidate in candidates:
        current = clusters[-1] if clusters else None
        current_end = max(item[1] for item in current) if current else -math.inf
        if current is None or candidate[0] >= current_end:
            clusters.append([candidate])
        else:
            current.append(candidate)
    return tuple(
        ProductionRallyInterval(
            start=min(item[0] for item in cluster),
            end=max(item[1] for item in cluster),
            agreement=(
                "both-models"
                if len({item[2] for item in cluster}) == 2
                else (
                    "all-labels-v2-only"
                    if cluster[0][2] == "all-labels-v2"
                    else "previous-production-only"
                )
            ),
        )
        for cluster in clusters
    )


def production_rally_anchor_evidence(
    intervals: Sequence[ProductionRallyInterval], anchor: float
) -> ProductionRallyAnchorEvidence:
    if not math.isfinite(anchor) or anchor < 0:
        raise ProductionServeGateError("production rally anchor must be non-negative")
    containing = [item for item in intervals if item.start <= anchor < item.end]
    if len(containing) > 1:
        raise ProductionServeGateError("production rally intervals overlap at the anchor")
    interval = containing[0] if containing else None
    return ProductionRallyAnchorEvidence(interval is not None, interval)


def hybrid_gate_prediction(
    head_evidence: Sequence[ServeHeadAnchorEvidence],
    rally_evidence: ProductionRallyAnchorEvidence,
) -> tuple[str, str, bool]:
    """Return prediction, decision source, and whether human review is required."""

    head_prediction = dual_head_prediction(head_evidence)
    if head_prediction == "serve":
        return "serve", "serve-head", False
    if rally_evidence.recovers_serve:
        return "serve", "production-rally-recovery", True
    return "not-serve", "none", False


def gate_fingerprint(heads: Sequence[ProductionServeHead]) -> str:
    contract = {
        "version": GATE_VERSION,
        "anchorWindowSeconds": ANCHOR_WINDOW_SECONDS,
        "aggregation": GATE_AGGREGATION,
        "heads": [
            {
                "modelId": head.model_id,
                "bundleSha256": head.bundle_sha256,
                "threshold": head.decoder.threshold,
            }
            for head in heads
        ],
    }
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def hybrid_gate_fingerprint(heads: Sequence[ProductionServeHead]) -> str:
    contract = {
        "version": HYBRID_GATE_VERSION,
        "anchorWindowSeconds": ANCHOR_WINDOW_SECONDS,
        "aggregation": HYBRID_GATE_AGGREGATION,
        "reviewRequiredForRallyRecovery": True,
        "heads": [
            {
                "modelId": head.model_id,
                "bundleSha256": head.bundle_sha256,
                "threshold": head.decoder.threshold,
            }
            for head in heads
        ],
    }
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
