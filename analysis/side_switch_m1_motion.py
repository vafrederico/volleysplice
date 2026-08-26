"""Foreground side-exchange motion reductions for the M1 side-switch experiment."""

from __future__ import annotations

import math
from typing import Any, Sequence

import cv2
import numpy as np

from analysis.serving_side_flight import ResidualMotion, _affine_camera_flow


PAIR_COUNT = 6
MAX_PAIR_DELTA_SECONDS = 0.25
COORDINATED_FLUX_FLOOR = 0.001
M1_FEATURE_NAMES = (
    "nearToFarForegroundFluxMean",
    "farToNearForegroundFluxMean",
    "bidirectionalExchangeMinimum",
    "foregroundExchangeImbalance",
    "coordinatedExchangePairFraction",
    "coordinatedExchangeDurationSeconds",
    "netBandMotionFractionMean",
    "edgeStableMotionMinimum",
)


def sample_pairs(
    gap_start: float, gap_end: float
) -> tuple[tuple[float, float], ...]:
    """Return six local flow pairs distributed across the complete candidate gap."""

    if (
        not math.isfinite(gap_start)
        or not math.isfinite(gap_end)
        or gap_end <= gap_start
    ):
        raise ValueError("M1 candidate gap must be finite and positive")
    duration = gap_end - gap_start
    delta = min(MAX_PAIR_DELTA_SECONDS, duration / 2.0)
    midpoints = np.linspace(
        gap_start + delta / 2.0,
        gap_end - delta / 2.0,
        PAIR_COUNT,
    )
    pairs = tuple(
        (float(midpoint - delta / 2.0), float(midpoint + delta / 2.0))
        for midpoint in midpoints
    )
    if len(pairs) != PAIR_COUNT or any(
        left < gap_start - 1e-9
        or right > gap_end + 1e-9
        or right <= left
        for left, right in pairs
    ):
        raise AssertionError("M1 sample-pair contract changed")
    return pairs


def extract_residual_motion(before: np.ndarray, after: np.ndarray) -> ResidualMotion:
    """Calculate camera-compensated dense flow using the frozen M1 parameters."""

    first = np.asarray(before)
    second = np.asarray(after)
    if first.shape != second.shape or first.ndim not in {2, 3}:
        raise ValueError("M1 frames must be aligned images with the same shape")
    first_gray = (
        first.astype(np.uint8, copy=False)
        if first.ndim == 2
        else cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    )
    second_gray = (
        second.astype(np.uint8, copy=False)
        if second.ndim == 2
        else cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
    )
    height, width = first_gray.shape
    if min(height, width) < 32:
        raise ValueError("M1 frames are too small")
    flow = cv2.calcOpticalFlowFarneback(
        first_gray,
        second_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.1,
        0,
    )
    residual = flow - _affine_camera_flow(flow)
    flow_x = residual[:, :, 0] / max(width, 1)
    flow_y = residual[:, :, 1] / max(height, 1)
    magnitude = np.linalg.norm(residual, axis=2) / math.hypot(height, width)
    threshold = max(7.5e-4, float(np.percentile(magnitude, 90.0)))
    energy = np.maximum(magnitude - threshold, 0.0)
    derivative_x = np.gradient(flow_x, axis=1) * width
    derivative_y = np.gradient(flow_y, axis=0) * height
    return ResidualMotion(
        energy=np.ascontiguousarray(energy, dtype=np.float32),
        flow_x=np.ascontiguousarray(flow_x, dtype=np.float32),
        flow_y=np.ascontiguousarray(flow_y, dtype=np.float32),
        divergence=np.ascontiguousarray(derivative_x + derivative_y, dtype=np.float32),
    )


def _pair_reduction(motion: ResidualMotion) -> dict[str, float]:
    energy = np.asarray(motion.energy, dtype=np.float64)
    flow_y = np.asarray(motion.flow_y, dtype=np.float64)
    if (
        energy.ndim != 2
        or flow_y.shape != energy.shape
        or not np.isfinite(energy).all()
        or not np.isfinite(flow_y).all()
        or np.any(energy < 0.0)
    ):
        raise ValueError("M1 residual motion must be finite and aligned")
    height, width = energy.shape
    yy, xx = np.mgrid[0:height, 0:width]
    normalized_x = xx / max(width - 1, 1)
    normalized_y = yy / max(height - 1, 1)
    court = (
        (normalized_x >= 0.06)
        & (normalized_x <= 0.94)
        & (normalized_y >= 0.18)
        & (normalized_y <= 0.94)
    )
    restricted_energy = np.where(court, energy, 0.0)
    total = float(np.sum(restricted_energy))
    near_weight = np.clip((normalized_y - 0.45) / 0.45, 0.0, 1.0)
    far_weight = np.clip((0.55 - normalized_y) / 0.35, 0.0, 1.0)
    if total > 0.0:
        near_to_far = float(
            np.sum(
                restricted_energy * np.maximum(-flow_y, 0.0) * near_weight
            )
            / total
        )
        far_to_near = float(
            np.sum(
                restricted_energy * np.maximum(flow_y, 0.0) * far_weight
            )
            / total
        )
        net_band = (normalized_y >= 0.35) & (normalized_y <= 0.65)
        net_band_fraction = float(
            np.sum(restricted_energy[net_band]) / total
        )
    else:
        near_to_far = far_to_near = net_band_fraction = 0.0
    active_fraction = float(np.sum((restricted_energy > 0.0) & court) / np.sum(court))
    return {
        "nearToFar": near_to_far,
        "farToNear": far_to_near,
        "bidirectionalMinimum": min(near_to_far, far_to_near),
        "coordinated": float(
            near_to_far >= COORDINATED_FLUX_FLOOR
            and far_to_near >= COORDINATED_FLUX_FLOOR
        ),
        "netBandMotionFraction": net_band_fraction,
        "activeForegroundFraction": active_fraction,
    }


def m1_features(
    motions: Sequence[ResidualMotion], gap_duration: float
) -> tuple[dict[str, float], dict[str, Any]]:
    """Reduce six flow fields into the frozen eight-value M1 bundle."""

    if len(motions) != PAIR_COUNT:
        raise ValueError(f"M1 expects exactly {PAIR_COUNT} residual-motion pairs")
    if not math.isfinite(gap_duration) or gap_duration <= 0.0:
        raise ValueError("M1 gap duration must be finite and positive")
    pairs = [_pair_reduction(motion) for motion in motions]
    near_to_far = float(np.mean([value["nearToFar"] for value in pairs]))
    far_to_near = float(np.mean([value["farToNear"] for value in pairs]))
    coordinated_fraction = float(
        np.mean([value["coordinated"] for value in pairs])
    )
    first_stability = 1.0 - pairs[0]["activeForegroundFraction"]
    last_stability = 1.0 - pairs[-1]["activeForegroundFraction"]
    values = {
        "nearToFarForegroundFluxMean": near_to_far,
        "farToNearForegroundFluxMean": far_to_near,
        "bidirectionalExchangeMinimum": min(near_to_far, far_to_near),
        "foregroundExchangeImbalance": abs(near_to_far - far_to_near),
        "coordinatedExchangePairFraction": coordinated_fraction,
        "coordinatedExchangeDurationSeconds": gap_duration
        * coordinated_fraction,
        "netBandMotionFractionMean": float(
            np.mean([value["netBandMotionFraction"] for value in pairs])
        ),
        "edgeStableMotionMinimum": float(
            np.clip(min(first_stability, last_stability), 0.0, 1.0)
        ),
    }
    if tuple(values) != M1_FEATURE_NAMES or not all(
        math.isfinite(value) for value in values.values()
    ):
        raise AssertionError("M1 feature signature or values changed")
    return values, {
        "pairCount": PAIR_COUNT,
        "coordinatedFluxFloor": COORDINATED_FLUX_FLOOR,
        "pairs": pairs,
    }
