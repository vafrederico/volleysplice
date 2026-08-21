"""Explicit residual-motion component trajectories for serving-side inference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .serving_side_flight import OFFSETS_SECONDS, ResidualMotion


FEATURE_VERSION = "serving-side-residual-component-trajectory-v1"
PAIR_COUNT = len(OFFSETS_SECONDS) - 1
MAX_COMPONENTS_PER_PAIR = 12
MAX_LINK_DISTANCE = 0.22
MAX_LINK_COST = 2.5
_EPSILON = 1e-12


@dataclass(frozen=True)
class MotionComponent:
    pair_index: int
    centroid_x: float
    centroid_y: float
    area_fraction: float
    box_width: float
    box_height: float
    energy_mass: float
    energy_fraction: float
    flow_x: float
    flow_y: float
    divergence: float


@dataclass(frozen=True)
class ComponentTrack:
    components: tuple[MotionComponent, ...]
    link_errors: tuple[float, ...]


def feature_names() -> tuple[str, ...]:
    return (
        "components:countMean",
        "components:countMax",
        "components:topEnergyFractionMean",
        "tracks:count",
        "tracks:persistentCount",
        "tracks:multiPairFraction",
        "tracks:confidenceMax",
        "tracks:confidenceMeanTop3",
        "tracks:weightedDisplacementY",
        "tracks:weightedExpansion",
        "tracks:verticalDirectionConsensus",
        "best:persistence",
        "best:span",
        "best:confidence",
        "best:energyMean",
        "best:energyMax",
        "best:areaMean",
        "best:startCentroidX",
        "best:startCentroidY",
        "best:endCentroidX",
        "best:endCentroidY",
        "best:displacementX",
        "best:displacementY",
        "best:displacementMagnitude",
        "best:pathLength",
        "best:straightness",
        "best:velocityXMean",
        "best:velocityYMean",
        "best:speedMean",
        "best:speedStd",
        "best:accelerationMean",
        "best:accelerationMax",
        "best:flowXMean",
        "best:flowYMean",
        "best:flowAlignment",
        "best:logAreaChange",
        "best:logAreaSlope",
        "best:linkErrorMean",
        "best:linkErrorMax",
        "best:onsetPair",
        "best:peakEnergyPair",
        "best:launchEnergyFraction",
        "best:directionConsistency",
    )


FEATURE_NAMES = feature_names()


def _cv2():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is required; run `npm run analysis:setup` first"
        ) from error
    return cv2


def _weighted_mean(values: np.ndarray, weights: np.ndarray, total: float) -> float:
    return float(np.sum(values * weights) / total) if total > 0 else 0.0


def motion_components(
    motion: ResidualMotion,
    pair_index: int,
    *,
    max_components: int = MAX_COMPONENTS_PER_PAIR,
) -> tuple[MotionComponent, ...]:
    """Return strongest residual-energy components for one frame pair."""
    if not 0 <= pair_index < PAIR_COUNT:
        raise ValueError("pair index is outside the flight window")
    if max_components < 1:
        raise ValueError("max components must be positive")
    energy = np.asarray(motion.energy, dtype=np.float64)
    flow_x = np.asarray(motion.flow_x, dtype=np.float64)
    flow_y = np.asarray(motion.flow_y, dtype=np.float64)
    divergence = np.asarray(motion.divergence, dtype=np.float64)
    if (
        energy.ndim != 2
        or flow_x.shape != energy.shape
        or flow_y.shape != energy.shape
        or divergence.shape != energy.shape
        or not all(
            np.isfinite(value).all()
            for value in (energy, flow_x, flow_y, divergence)
        )
        or np.any(energy < 0)
    ):
        raise ValueError("residual-motion arrays must be finite aligned 2D values")
    active = energy > 0
    total_energy = float(np.sum(energy))
    if not np.any(active) or total_energy <= 0:
        return ()
    cv2 = _cv2()
    # A one-pixel dilation joins the two sides of a tiny moving object's flow edge.
    connected = cv2.dilate(
        active.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1
    )
    component_count, labels, _, _ = cv2.connectedComponentsWithStats(
        connected, connectivity=8
    )
    height, width = energy.shape
    yy, xx = np.mgrid[0:height, 0:width]
    components: list[MotionComponent] = []
    for label in range(1, component_count):
        mask = (labels == label) & active
        pixels = int(np.sum(mask))
        if not pixels:
            continue
        weights = energy[mask]
        mass = float(np.sum(weights))
        if mass <= 0:
            continue
        rows, columns = np.where(mask)
        components.append(
            MotionComponent(
                pair_index=pair_index,
                centroid_x=_weighted_mean(
                    xx[mask] / max(width - 1, 1), weights, mass
                ),
                centroid_y=_weighted_mean(
                    yy[mask] / max(height - 1, 1), weights, mass
                ),
                area_fraction=pixels / energy.size,
                box_width=(int(columns.max()) - int(columns.min()) + 1) / width,
                box_height=(int(rows.max()) - int(rows.min()) + 1) / height,
                energy_mass=mass,
                energy_fraction=mass / total_energy,
                flow_x=_weighted_mean(flow_x[mask], weights, mass),
                flow_y=_weighted_mean(flow_y[mask], weights, mass),
                divergence=_weighted_mean(divergence[mask], weights, mass),
            )
        )
    components.sort(
        key=lambda item: (
            -item.energy_mass,
            item.centroid_y,
            item.centroid_x,
            item.area_fraction,
        )
    )
    return tuple(components[:max_components])


def _link_measure(
    before: MotionComponent, after: MotionComponent
) -> tuple[float, float]:
    expected_x = before.centroid_x + before.flow_x
    expected_y = before.centroid_y + before.flow_y
    error = math.hypot(
        after.centroid_x - expected_x, after.centroid_y - expected_y
    )
    scale_change = abs(
        math.log((after.area_fraction + _EPSILON) / (before.area_fraction + _EPSILON))
    )
    flow_change = math.hypot(
        after.flow_x - before.flow_x, after.flow_y - before.flow_y
    )
    cost = error / 0.10 + 0.20 * scale_change + flow_change / 0.20
    return error, cost


def link_component_tracks(
    components_by_pair: Sequence[Sequence[MotionComponent]],
) -> tuple[ComponentTrack, ...]:
    """Greedily link components with deterministic one-to-one adjacent matches."""
    if len(components_by_pair) != PAIR_COUNT:
        raise ValueError("component sequence has the wrong number of frame pairs")
    tracks: list[list[MotionComponent]] = []
    errors: list[list[float]] = []
    active_track_indices: list[int] = []
    for pair_index, raw_components in enumerate(components_by_pair):
        components = list(raw_components)
        if any(component.pair_index != pair_index for component in components):
            raise ValueError("component pair identity is inconsistent")
        if pair_index:
            edges: list[tuple[float, float, int, int]] = []
            for track_index in active_track_indices:
                before = tracks[track_index][-1]
                for component_index, after in enumerate(components):
                    error, cost = _link_measure(before, after)
                    if error <= MAX_LINK_DISTANCE and cost <= MAX_LINK_COST:
                        edges.append((cost, error, track_index, component_index))
            edges.sort()
            used_tracks: set[int] = set()
            used_components: set[int] = set()
            next_active: list[int] = []
            for _, error, track_index, component_index in edges:
                if track_index in used_tracks or component_index in used_components:
                    continue
                tracks[track_index].append(components[component_index])
                errors[track_index].append(error)
                used_tracks.add(track_index)
                used_components.add(component_index)
                next_active.append(track_index)
            for component_index, component in enumerate(components):
                if component_index in used_components:
                    continue
                tracks.append([component])
                errors.append([])
                next_active.append(len(tracks) - 1)
            active_track_indices = next_active
        else:
            for component in components:
                tracks.append([component])
                errors.append([])
            active_track_indices = list(range(len(tracks)))
    result = [
        ComponentTrack(tuple(track), tuple(track_errors))
        for track, track_errors in zip(tracks, errors, strict=True)
    ]
    result.sort(
        key=lambda track: (
            -_track_confidence(track),
            -len(track.components),
            track.components[0].pair_index,
            track.components[0].centroid_y,
            track.components[0].centroid_x,
        )
    )
    return tuple(result)


def _track_vectors(track: ComponentTrack) -> np.ndarray:
    return np.asarray(
        [
            (
                after.centroid_x - before.centroid_x,
                after.centroid_y - before.centroid_y,
            )
            for before, after in zip(
                track.components[:-1], track.components[1:], strict=True
            )
        ],
        dtype=np.float64,
    )


def _straightness(vectors: np.ndarray) -> float:
    if not len(vectors):
        return 0.0
    path_length = float(np.sum(np.linalg.norm(vectors, axis=1)))
    displacement = float(np.linalg.norm(np.sum(vectors, axis=0)))
    return displacement / path_length if path_length > 0 else 0.0


def _direction_consistency(vectors: np.ndarray) -> float:
    if not len(vectors):
        return 0.0
    speeds = np.linalg.norm(vectors, axis=1)
    moving = speeds > _EPSILON
    if not np.any(moving):
        return 0.0
    directions = vectors[moving] / speeds[moving, None]
    return float(np.linalg.norm(np.mean(directions, axis=0)))


def _track_confidence(track: ComponentTrack) -> float:
    components = track.components
    if not components:
        return 0.0
    persistence = len(components) / PAIR_COUNT
    energy = float(np.mean([item.energy_fraction for item in components]))
    link_quality = (
        math.exp(-float(np.mean(track.link_errors)) / 0.08)
        if track.link_errors
        else 0.25
    )
    straightness = _straightness(_track_vectors(track))
    return persistence * math.sqrt(max(energy, 0.0)) * link_quality * (
        0.5 + 0.5 * straightness
    )


def _track_features(track: ComponentTrack) -> dict[str, float]:
    components = track.components
    first, last = components[0], components[-1]
    vectors = _track_vectors(track)
    speeds = np.linalg.norm(vectors, axis=1) if len(vectors) else np.zeros(0)
    accelerations = np.diff(vectors, axis=0) if len(vectors) > 1 else np.zeros((0, 2))
    acceleration_magnitudes = (
        np.linalg.norm(accelerations, axis=1) if len(accelerations) else np.zeros(0)
    )
    displacement_x = last.centroid_x - first.centroid_x
    displacement_y = last.centroid_y - first.centroid_y
    path_length = float(np.sum(speeds))
    flow = np.asarray([(item.flow_x, item.flow_y) for item in components])
    mean_vector = np.mean(vectors, axis=0) if len(vectors) else np.zeros(2)
    mean_flow = np.mean(flow, axis=0)
    alignment_denominator = float(np.linalg.norm(mean_vector) * np.linalg.norm(mean_flow))
    flow_alignment = (
        float(np.dot(mean_vector, mean_flow) / alignment_denominator)
        if alignment_denominator > _EPSILON
        else 0.0
    )
    areas = np.asarray([item.area_fraction for item in components])
    log_area_change = math.log(
        (float(areas[-1]) + _EPSILON) / (float(areas[0]) + _EPSILON)
    )
    span = last.pair_index - first.pair_index + 1
    energy = np.asarray([item.energy_fraction for item in components])
    energy_mass = np.asarray([item.energy_mass for item in components])
    total_mass = float(np.sum(energy_mass))
    launch_mass = float(
        np.sum(
            [
                item.energy_mass
                for item in components
                if item.pair_index <= 2
            ]
        )
    )
    link_errors = np.asarray(track.link_errors, dtype=np.float64)
    return {
        "persistence": len(components) / PAIR_COUNT,
        "span": span / PAIR_COUNT,
        "confidence": _track_confidence(track),
        "energyMean": float(np.mean(energy)),
        "energyMax": float(np.max(energy)),
        "areaMean": float(np.mean(areas)),
        "startCentroidX": first.centroid_x,
        "startCentroidY": first.centroid_y,
        "endCentroidX": last.centroid_x,
        "endCentroidY": last.centroid_y,
        "displacementX": displacement_x,
        "displacementY": displacement_y,
        "displacementMagnitude": math.hypot(displacement_x, displacement_y),
        "pathLength": path_length,
        "straightness": _straightness(vectors),
        "velocityXMean": float(np.mean(vectors[:, 0])) if len(vectors) else 0.0,
        "velocityYMean": float(np.mean(vectors[:, 1])) if len(vectors) else 0.0,
        "speedMean": float(np.mean(speeds)) if len(speeds) else 0.0,
        "speedStd": float(np.std(speeds)) if len(speeds) else 0.0,
        "accelerationMean": (
            float(np.mean(acceleration_magnitudes))
            if len(acceleration_magnitudes)
            else 0.0
        ),
        "accelerationMax": (
            float(np.max(acceleration_magnitudes))
            if len(acceleration_magnitudes)
            else 0.0
        ),
        "flowXMean": float(mean_flow[0]),
        "flowYMean": float(mean_flow[1]),
        "flowAlignment": flow_alignment,
        "logAreaChange": log_area_change,
        "logAreaSlope": log_area_change / max(span - 1, 1),
        "linkErrorMean": float(np.mean(link_errors)) if len(link_errors) else 0.0,
        "linkErrorMax": float(np.max(link_errors)) if len(link_errors) else 0.0,
        "onsetPair": first.pair_index / max(PAIR_COUNT - 1, 1),
        "peakEnergyPair": (
            components[int(np.argmax(energy_mass))].pair_index
            / max(PAIR_COUNT - 1, 1)
        ),
        "launchEnergyFraction": launch_mass / total_mass if total_mass > 0 else 0.0,
        "directionConsistency": _direction_consistency(vectors),
    }


def extract_trajectory_features(
    motions: Sequence[ResidualMotion],
) -> dict[str, float]:
    if len(motions) != PAIR_COUNT:
        raise ValueError("trajectory-motion sequence has the wrong number of pairs")
    components_by_pair = [
        motion_components(motion, pair_index)
        for pair_index, motion in enumerate(motions)
    ]
    tracks = link_component_tracks(components_by_pair)
    component_counts = np.asarray([len(items) for items in components_by_pair])
    top_energy = np.asarray(
        [max((item.energy_fraction for item in items), default=0.0) for items in components_by_pair]
    )
    multi_pair = [track for track in tracks if len(track.components) >= 2]
    persistent = [track for track in tracks if len(track.components) >= 3]
    best = tracks[0] if tracks else None
    top = tracks[:3]
    confidences = np.asarray([_track_confidence(track) for track in top])
    confidence_total = float(np.sum(confidences))
    weighted_displacement_y = (
        float(
            np.sum(
                [
                    confidence
                    * (
                        track.components[-1].centroid_y
                        - track.components[0].centroid_y
                    )
                    for confidence, track in zip(confidences, top, strict=True)
                ]
            )
            / confidence_total
        )
        if confidence_total > 0
        else 0.0
    )
    expansions = [
        math.log(
            (track.components[-1].area_fraction + _EPSILON)
            / (track.components[0].area_fraction + _EPSILON)
        )
        for track in top
    ]
    weighted_expansion = (
        float(np.dot(confidences, expansions) / confidence_total)
        if confidence_total > 0
        else 0.0
    )
    vertical_signs = [
        math.copysign(
            1.0,
            track.components[-1].centroid_y - track.components[0].centroid_y,
        )
        if abs(
            track.components[-1].centroid_y - track.components[0].centroid_y
        )
        > _EPSILON
        else 0.0
        for track in top
    ]
    vertical_consensus = (
        abs(float(np.dot(confidences, vertical_signs) / confidence_total))
        if confidence_total > 0
        else 0.0
    )
    output: dict[str, float] = {
        "components:countMean": float(np.mean(component_counts)),
        "components:countMax": float(np.max(component_counts)),
        "components:topEnergyFractionMean": float(np.mean(top_energy)),
        "tracks:count": float(len(tracks)),
        "tracks:persistentCount": float(len(persistent)),
        "tracks:multiPairFraction": len(multi_pair) / len(tracks) if tracks else 0.0,
        "tracks:confidenceMax": float(np.max(confidences)) if len(confidences) else 0.0,
        "tracks:confidenceMeanTop3": float(np.mean(confidences)) if len(confidences) else 0.0,
        "tracks:weightedDisplacementY": weighted_displacement_y,
        "tracks:weightedExpansion": weighted_expansion,
        "tracks:verticalDirectionConsensus": vertical_consensus,
    }
    best_values = _track_features(best) if best is not None else {
        name.removeprefix("best:"): 0.0
        for name in FEATURE_NAMES
        if name.startswith("best:")
    }
    output.update({f"best:{name}": value for name, value in best_values.items()})
    if tuple(output) != FEATURE_NAMES or not all(
        math.isfinite(value) for value in output.values()
    ):
        raise AssertionError("trajectory feature contract changed or became non-finite")
    return output
