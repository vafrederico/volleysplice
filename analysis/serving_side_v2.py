"""Court-relative temporal features and small models for serving-side v2.

The extractor deliberately works on short windows around reviewed serve anchors.
It removes global camera motion before measuring optical flow, divides evidence into
pre/contact/post phases, and measures connected motion occupancy in near/far service
zones.  Annotated service-zone anchors are preferred; the current corpus falls back
to fixed, ROI-relative end bands and records that limitation in every row.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .serving_side_specialist import ServingSideSpecialistError


FEATURE_VERSION = "serving-side-court-flow-v2"
OFFSETS_SECONDS = (-1.25, -0.75, -0.35, -0.10, 0.10, 0.30, 0.55, 0.85)
PHASE_PAIRS = {
    "pre": ((0, 1), (1, 2)),
    "contact": ((2, 3), (3, 4), (4, 5)),
    "post": ((5, 6), (6, 7)),
}
ZONES = ("near", "far")
PHASE_STATISTICS = (
    "flowMean",
    "flowP90",
    "activeFraction",
    "largestComponentFraction",
    "componentCountDensity",
    "centroidX",
    "centroidY",
    "flowX",
    "flowY",
)


def feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for phase in PHASE_PAIRS:
        for zone in ZONES:
            names.extend(
                f"{phase}:{zone}:{statistic}" for statistic in PHASE_STATISTICS
            )
        names.extend(
            f"{phase}:nearMinusFar:{statistic}"
            for statistic in PHASE_STATISTICS[:5]
        )
    for zone in ZONES:
        names.extend(
            (
                f"contactMinusPre:{zone}:flowMean",
                f"contactMinusPre:{zone}:activeFraction",
                f"contactMinusPre:{zone}:largestComponentFraction",
                f"postMinusContact:{zone}:flowMean",
                f"postMinusContact:{zone}:activeFraction",
            )
        )
    names.extend(
        (
            "contactDelta:nearMinusFar:flowMean",
            "contactDelta:nearMinusFar:activeFraction",
            "contactDelta:nearMinusFar:largestComponentFraction",
        )
    )
    return tuple(names)


FEATURE_NAMES = feature_names()


def _point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, Mapping):
        return None
    x, y = value.get("x"), value.get("y")
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None
    if not math.isfinite(float(x)) or not math.isfinite(float(y)):
        return None
    return float(x), float(y)


def _to_roi_point(
    point: tuple[float, float], roi: tuple[float, float, float, float]
) -> tuple[float, float]:
    x, y, width, height = roi
    return (point[0] - x) / width, (point[1] - y) / height


def service_zone_masks(
    height: int,
    width: int,
    *,
    roi: tuple[float, float, float, float],
    court_geometry: Mapping[str, Any] | None,
) -> tuple[dict[str, np.ndarray], str]:
    """Return near/far masks and their auditable geometry provenance."""
    if height <= 0 or width <= 0:
        raise ValueError("zone dimensions must be positive")
    geometry = court_geometry if isinstance(court_geometry, Mapping) else {}
    anchors = geometry.get("serviceZoneAnchors")
    anchors = anchors if isinstance(anchors, Mapping) else {}
    near = _point(anchors.get("near"))
    far = _point(anchors.get("far"))
    yy, xx = np.mgrid[0:height, 0:width]
    if near is not None and far is not None:
        # Anchors are normalized in the full frame; convert them to cropped ROI space.
        near_x, near_y = _to_roi_point(near, roi)
        far_x, far_y = _to_roi_point(far, roi)
        radius_x = max(0.12, abs(near_x - far_x) * 0.18)
        radius_y = max(0.10, abs(near_y - far_y) * 0.20)

        def ellipse(center_x: float, center_y: float) -> np.ndarray:
            normalized_x = (xx / max(width - 1, 1) - center_x) / radius_x
            normalized_y = (yy / max(height - 1, 1) - center_y) / radius_y
            return normalized_x**2 + normalized_y**2 <= 1.0

        masks = {"near": ellipse(near_x, near_y), "far": ellipse(far_x, far_y)}
        if all(np.any(mask) for mask in masks.values()):
            return masks, "annotated-service-zone-anchors"

    # The fallback is intentionally camera-relative, not claimed court calibration.
    band = max(1, round(height * 0.32))
    near_mask = np.zeros((height, width), dtype=bool)
    far_mask = np.zeros((height, width), dtype=bool)
    near_mask[height - band :, :] = True
    far_mask[:band, :] = True
    return {"near": near_mask, "far": far_mask}, "roi-relative-end-bands"


def _component_statistics(
    active: np.ndarray, magnitude: np.ndarray, flow: np.ndarray
) -> tuple[float, ...]:
    cv2 = _cv2()
    pixels = int(np.sum(active))
    if active.size == 0:
        return (0.0,) * len(PHASE_STATISTICS)
    active_fraction = pixels / active.size
    binary = active.astype(np.uint8)
    component_count, _, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if component_count > 1:
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
        largest = float(np.max(areas) / active.size)
        component_density = float(len(areas) / max(active.size / 1000.0, 1.0))
        largest_index = int(np.argmax(areas)) + 1
        centroid_x = float(centroids[largest_index, 0] / max(active.shape[1] - 1, 1))
        centroid_y = float(centroids[largest_index, 1] / max(active.shape[0] - 1, 1))
    else:
        largest = component_density = centroid_x = centroid_y = 0.0
    if pixels:
        flow_x = float(np.mean(flow[:, :, 0][active]) / max(active.shape[1], 1))
        flow_y = float(np.mean(flow[:, :, 1][active]) / max(active.shape[0], 1))
    else:
        flow_x = flow_y = 0.0
    diagonal = float(np.hypot(*active.shape))
    return (
        float(np.mean(magnitude) / diagonal),
        float(np.percentile(magnitude, 90) / diagonal),
        active_fraction,
        largest,
        component_density,
        centroid_x,
        centroid_y,
        flow_x,
        flow_y,
    )


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is required; run `npm run analysis:setup` first"
        ) from error
    return cv2


def extract_window_features(
    frames: Sequence[np.ndarray], masks: Mapping[str, np.ndarray]
) -> dict[str, float]:
    """Extract phase-aware residual-flow features from eight ordered BGR frames."""
    if len(frames) != len(OFFSETS_SECONDS):
        raise ValueError(f"expected {len(OFFSETS_SECONDS)} frames")
    cv2 = _cv2()
    gray = [
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame.ndim == 3
        else frame.astype(np.uint8, copy=False)
        for frame in frames
    ]
    output: dict[str, float] = {}
    phase_values: dict[str, dict[str, np.ndarray]] = {}
    for phase, pairs in PHASE_PAIRS.items():
        collected = {zone: [] for zone in ZONES}
        for before_index, after_index in pairs:
            flow = cv2.calcOpticalFlowFarneback(
                gray[before_index], gray[after_index], None, 0.5, 2, 13, 2, 5, 1.1, 0
            )
            global_motion = np.median(flow.reshape(-1, 2), axis=0)
            residual = flow - global_motion.astype(np.float32)
            magnitude = np.linalg.norm(residual, axis=2)
            active_full = magnitude >= 1.0
            active_full = cv2.morphologyEx(
                active_full.astype(np.uint8),
                cv2.MORPH_OPEN,
                np.ones((3, 3), dtype=np.uint8),
            ).astype(bool)
            for zone in ZONES:
                mask = np.asarray(masks[zone], dtype=bool)
                if mask.shape != magnitude.shape:
                    raise ValueError("zone mask and frame shapes differ")
                rows, columns = np.where(mask)
                if not len(rows):
                    collected[zone].append(np.zeros(len(PHASE_STATISTICS)))
                    continue
                y0, y1 = int(rows.min()), int(rows.max()) + 1
                x0, x1 = int(columns.min()), int(columns.max()) + 1
                local_mask = mask[y0:y1, x0:x1]
                local_magnitude = magnitude[y0:y1, x0:x1]
                local_flow = residual[y0:y1, x0:x1]
                local_active = active_full[y0:y1, x0:x1] & local_mask
                masked_magnitude = np.where(local_mask, local_magnitude, 0.0)
                collected[zone].append(
                    np.asarray(
                        _component_statistics(local_active, masked_magnitude, local_flow),
                        dtype=np.float64,
                    )
                )
        phase_values[phase] = {
            zone: np.mean(np.stack(values), axis=0)
            for zone, values in collected.items()
        }
        for zone in ZONES:
            for statistic, value in zip(
                PHASE_STATISTICS, phase_values[phase][zone], strict=True
            ):
                output[f"{phase}:{zone}:{statistic}"] = float(value)
        difference = phase_values[phase]["near"] - phase_values[phase]["far"]
        for statistic, value in zip(
            PHASE_STATISTICS[:5], difference[:5], strict=True
        ):
            output[f"{phase}:nearMinusFar:{statistic}"] = float(value)

    for zone in ZONES:
        pre, contact, post = (
            phase_values[phase][zone] for phase in ("pre", "contact", "post")
        )
        output[f"contactMinusPre:{zone}:flowMean"] = float(contact[0] - pre[0])
        output[f"contactMinusPre:{zone}:activeFraction"] = float(contact[2] - pre[2])
        output[f"contactMinusPre:{zone}:largestComponentFraction"] = float(
            contact[3] - pre[3]
        )
        output[f"postMinusContact:{zone}:flowMean"] = float(post[0] - contact[0])
        output[f"postMinusContact:{zone}:activeFraction"] = float(post[2] - contact[2])
    for statistic, index in (
        ("flowMean", 0),
        ("activeFraction", 2),
        ("largestComponentFraction", 3),
    ):
        contact_delta = phase_values["contact"]["near"][index] - phase_values["pre"]["near"][index]
        contact_delta -= phase_values["contact"]["far"][index] - phase_values["pre"]["far"][index]
        output[f"contactDelta:nearMinusFar:{statistic}"] = float(contact_delta)
    if tuple(output) != FEATURE_NAMES:
        raise AssertionError("serving-side v2 feature ordering changed")
    return output


@dataclass(frozen=True)
class BoostedStump:
    feature: int
    threshold: float
    polarity: int
    weight: float


@dataclass(frozen=True)
class BoostedStumpModel:
    impute: np.ndarray
    stumps: tuple[BoostedStump, ...]
    learning_rate: float
    threshold: float = 0.5

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        filled = np.where(np.isfinite(matrix), matrix, self.impute)
        scores = np.zeros(len(filled), dtype=np.float64)
        for stump in self.stumps:
            guesses = np.where(
                filled[:, stump.feature] >= stump.threshold, stump.polarity, -stump.polarity
            )
            scores += self.learning_rate * stump.weight * guesses
        return 1.0 / (1.0 + np.exp(np.clip(-2.0 * scores, -30.0, 30.0)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": "adaboost-decision-stumps",
            "impute": self.impute.tolist(),
            "learningRate": self.learning_rate,
            "threshold": self.threshold,
            "stumps": [
                {
                    "feature": stump.feature,
                    "threshold": stump.threshold,
                    "polarity": stump.polarity,
                    "weight": stump.weight,
                }
                for stump in self.stumps
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BoostedStumpModel":
        return cls(
            impute=np.asarray(value["impute"], dtype=np.float64),
            stumps=tuple(
                BoostedStump(
                    int(item["feature"]),
                    float(item["threshold"]),
                    int(item["polarity"]),
                    float(item["weight"]),
                )
                for item in value["stumps"]
            ),
            learning_rate=float(value["learningRate"]),
            threshold=float(value.get("threshold", 0.5)),
        )


def fit_boosted_stumps(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    estimators: int,
    learning_rate: float,
    quantiles: int = 15,
) -> BoostedStumpModel:
    """Fit deterministic class-balanced AdaBoost decision stumps."""
    matrix = np.asarray(values, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    if matrix.ndim != 2 or len(matrix) != len(truth) or set(truth.tolist()) != {0, 1}:
        raise ServingSideSpecialistError("boosting needs aligned rows with both sides")
    impute = np.asarray(
        [
            float(np.median(column[np.isfinite(column)]))
            if np.any(np.isfinite(column))
            else 0.0
            for column in matrix.T
        ]
    )
    filled = np.where(np.isfinite(matrix), matrix, impute)
    targets = np.where(truth == 1, 1, -1)
    positives, negatives = np.sum(truth == 1), np.sum(truth == 0)
    weights = np.where(truth == 1, 0.5 / positives, 0.5 / negatives).astype(np.float64)
    candidates = []
    probabilities = np.linspace(0.0, 1.0, quantiles + 2)[1:-1]
    for feature in range(filled.shape[1]):
        thresholds = np.unique(np.quantile(filled[:, feature], probabilities))
        candidates.append(thresholds)
    stumps: list[BoostedStump] = []
    for _ in range(estimators):
        best: tuple[float, int, float, int, np.ndarray] | None = None
        for feature, thresholds in enumerate(candidates):
            for threshold in thresholds:
                base = np.where(filled[:, feature] >= threshold, 1, -1)
                for polarity in (1, -1):
                    guesses = base * polarity
                    error = float(np.sum(weights[guesses != targets]))
                    candidate = (error, feature, float(threshold), polarity, guesses)
                    if best is None or candidate[:4] < best[:4]:
                        best = candidate
        assert best is not None
        error, feature, threshold, polarity, guesses = best
        error = float(np.clip(error, 1e-9, 1.0 - 1e-9))
        if error >= 0.5 - 1e-12:
            break
        alpha = 0.5 * math.log((1.0 - error) / error)
        stumps.append(BoostedStump(feature, threshold, polarity, alpha))
        weights *= np.exp(-learning_rate * alpha * targets * guesses)
        weights /= np.sum(weights)
    if not stumps:
        raise ServingSideSpecialistError("boosting found no useful stump")
    return BoostedStumpModel(impute, tuple(stumps), learning_rate)


@dataclass(frozen=True)
class LogisticModel:
    impute: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    l2: float
    threshold: float = 0.5

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        filled = np.where(np.isfinite(matrix), matrix, self.impute)
        logits = np.clip(((filled - self.mean) / self.scale) @ self.weights + self.bias, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": "class-balanced-logistic",
            "impute": self.impute.tolist(),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "l2": self.l2,
            "threshold": self.threshold,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LogisticModel":
        return cls(
            impute=np.asarray(value["impute"], dtype=np.float64),
            mean=np.asarray(value["mean"], dtype=np.float64),
            scale=np.asarray(value["scale"], dtype=np.float64),
            weights=np.asarray(value["weights"], dtype=np.float64),
            bias=float(value["bias"]),
            l2=float(value["l2"]),
            threshold=float(value.get("threshold", 0.5)),
        )


def fit_logistic(values: np.ndarray, labels: np.ndarray, *, l2: float) -> LogisticModel:
    matrix = np.asarray(values, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.float64)
    positives, negatives = int(np.sum(truth == 1)), int(np.sum(truth == 0))
    if matrix.ndim != 2 or len(matrix) != len(truth) or not positives or not negatives:
        raise ServingSideSpecialistError("logistic fitting needs aligned rows with both sides")
    impute = np.asarray([
        float(np.median(column[np.isfinite(column)])) if np.any(np.isfinite(column)) else 0.0
        for column in matrix.T
    ])
    filled = np.where(np.isfinite(matrix), matrix, impute)
    mean, scale = np.mean(filled, axis=0), np.std(filled, axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (filled - mean) / scale
    sample_weights = np.where(truth == 1, len(truth) / (2 * positives), len(truth) / (2 * negatives))
    weights = np.zeros(normalized.shape[1], dtype=np.float64)
    bias = 0.0
    design = np.column_stack((normalized, np.ones(len(normalized))))
    regularizer = np.diag(np.r_[np.full(len(weights), l2), 0.0])
    for _ in range(100):
        logits = np.clip(normalized @ weights + bias, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ ((probabilities - truth) * sample_weights) / np.sum(sample_weights)
        gradient[:-1] += l2 * weights
        curvature = probabilities * (1.0 - probabilities) * sample_weights
        hessian = (design.T * curvature) @ design / np.sum(sample_weights) + regularizer
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        weights -= step[:-1]
        bias -= float(step[-1])
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return LogisticModel(impute, mean, scale, weights, bias, l2)
