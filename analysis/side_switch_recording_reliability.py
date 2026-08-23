"""Recording-level reliability summaries and threshold-offset head."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from analysis.side_switch_full_union_ranker import decode_ranked_candidates
from analysis.side_switch_winner_experiment import (
    DECODER,
    PADDING,
    evaluate,
    logit,
    metric_rank,
    sigmoid,
    thresholds,
)


RELIABILITY_FEATURE_NAMES = (
    "cameraShiftP90",
    "alignmentResponseP10",
    "playerSideSeparationP10",
    "proposalCoverageP10",
    "paletteInstabilityP90",
    "globalAppearanceChangeP90",
    "serveAnchorErrorP90",
    "serveConfidenceP10",
    "candidateScoreMean",
    "candidateScoreStd",
    "candidateScoreP90",
    "fractionAboveBaseThreshold",
    "candidatesPerMinute",
)


@dataclass(frozen=True)
class ReliabilityHead:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    ridge: float

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float64)
        expected = (len(self.feature_names),)
        if values.ndim != 2 or values.shape[1:] != expected:
            raise ValueError("reliability feature matrix shape drifted")
        return ((values - self.mean) / self.scale) @ self.weights + self.bias

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "volleycut-side-switch-recording-reliability-ridge-v1",
            "featureNames": list(self.feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "ridge": self.ridge,
            "target": "per-recording optimal threshold logit minus fit-global threshold logit",
        }


def _values(rows: Sequence[Mapping[str, Any]], name: str) -> np.ndarray:
    result = np.asarray(
        [float(row.get("features", {}).get(name, math.nan)) for row in rows],
        dtype=np.float64,
    )
    return result[np.isfinite(result)]


def _quantile(rows: Sequence[Mapping[str, Any]], name: str, q: float) -> float:
    values = _values(rows, name)
    return float(np.quantile(values, q)) if len(values) else 0.0


def recording_summary(
    rows: Sequence[Mapping[str, Any]],
    scores: np.ndarray,
    base_threshold: float,
) -> dict[str, float]:
    values = np.asarray(scores, dtype=np.float64)
    if (
        not rows
        or values.shape != (len(rows),)
        or not np.isfinite(values).all()
        or np.any((values < 0) | (values > 1))
    ):
        raise ValueError("reliability summary needs finite aligned candidate scores")
    if not math.isfinite(base_threshold) or not 0 <= base_threshold <= 1:
        raise ValueError("reliability summary threshold must stay in [0, 1]")
    palette = np.asarray(
        [
            0.5
            * (
                float(row.get("features", {}).get("beforePlayerPaletteInstability", 0.0))
                + float(row.get("features", {}).get("afterPlayerPaletteInstability", 0.0))
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    anchor = np.asarray(
        [
            float(
                row.get("features", {}).get(
                    "productionMaximumAdjacentAnchorErrorSeconds", 0.0
                )
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    duration_minutes = max(float(row["transitionTime"]) for row in rows) / 60.0
    return {
        "cameraShiftP90": _quantile(rows, "v4MaximumCameraShift", 0.9),
        "alignmentResponseP10": _quantile(rows, "v4MinimumAlignmentResponse", 0.1),
        "playerSideSeparationP10": _quantile(rows, "minimumPlayerSideSeparation", 0.1),
        "proposalCoverageP10": _quantile(rows, "minimumProposalCoverage", 0.1),
        "paletteInstabilityP90": float(np.quantile(palette, 0.9)),
        "globalAppearanceChangeP90": _quantile(
            rows, "playerGlobalAppearanceChange", 0.9
        ),
        "serveAnchorErrorP90": float(np.quantile(anchor, 0.9)),
        "serveConfidenceP10": _quantile(
            rows, "productionMinimumAdjacentServeConfidence", 0.1
        ),
        "candidateScoreMean": float(np.mean(values)),
        "candidateScoreStd": float(np.std(values)),
        "candidateScoreP90": float(np.quantile(values, 0.9)),
        "fractionAboveBaseThreshold": float(np.mean(values >= base_threshold)),
        "candidatesPerMinute": len(rows) / max(duration_minutes, 1e-6),
    }


def summary_matrix(summaries: Sequence[Mapping[str, float]]) -> np.ndarray:
    matrix = np.asarray(
        [[float(value[name]) for name in RELIABILITY_FEATURE_NAMES] for value in summaries],
        dtype=np.float64,
    )
    if matrix.ndim != 2 or matrix.shape[1] != len(RELIABILITY_FEATURE_NAMES):
        raise ValueError("reliability summaries have the wrong shape")
    if not np.isfinite(matrix).all():
        raise ValueError("reliability summaries must be finite")
    return matrix


def fit_reliability_head(
    summaries: Sequence[Mapping[str, float]], targets: np.ndarray, ridge: float
) -> ReliabilityHead:
    matrix = summary_matrix(summaries)
    values = np.asarray(targets, dtype=np.float64)
    if values.shape != (len(summaries),) or not np.isfinite(values).all():
        raise ValueError("reliability targets must be finite and aligned")
    if ridge <= 0 or not math.isfinite(ridge):
        raise ValueError("reliability ridge must be finite and positive")
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (matrix - mean) / scale
    centered = values - float(np.mean(values))
    system = normalized.T @ normalized / len(values)
    system += ridge * np.eye(normalized.shape[1])
    weights = np.linalg.solve(system, normalized.T @ centered / len(values))
    return ReliabilityHead(
        RELIABILITY_FEATURE_NAMES,
        mean,
        scale,
        weights,
        float(np.mean(values)),
        ridge,
    )


def optimal_threshold_offset(
    rows: Sequence[Mapping[str, Any]],
    scores: np.ndarray,
    markers: Mapping[str, Any],
    base_threshold: float,
    maximum_absolute_offset: float = 2.0,
) -> float:
    best: dict[str, Any] | None = None
    candidates = {*thresholds(scores), float(base_threshold)}
    for threshold in candidates:
        predictions = decode_ranked_candidates(rows, scores, threshold, DECODER)
        metrics = evaluate(rows, predictions, markers, PADDING)
        offset = logit(float(threshold)) - logit(base_threshold)
        rank = (*metric_rank(metrics, threshold)[:-1], -abs(offset), threshold)
        value = {"rank": rank, "offset": offset}
        if best is None or value["rank"] > best["rank"]:
            best = value
    if best is None:
        raise AssertionError("recording threshold target selection failed")
    return float(np.clip(best["offset"], -maximum_absolute_offset, maximum_absolute_offset))


def adjust_scores(
    rows: Sequence[Mapping[str, Any]],
    scores: np.ndarray,
    offsets: Mapping[str, float],
    strength: float,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(rows),) or not np.isfinite(values).all():
        raise ValueError("reliability adjustment needs finite aligned scores")
    if strength < 0 or not math.isfinite(strength):
        raise ValueError("reliability strength must be finite and nonnegative")
    adjusted_logits = np.asarray(
        [
            logit(float(value))
            - strength * float(offsets[str(row["recordingId"])])
            for row, value in zip(rows, values, strict=True)
        ]
    )
    return sigmoid(adjusted_logits)
