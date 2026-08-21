"""Concentrated post-contact motion features for serving-side experiments.

The feature family is intentionally not a ball detector. It removes a robust
affine camera-flow field, keeps the strongest residual motion, and summarizes
where that energy is concentrated and how it moves through an image grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


FEATURE_VERSION = "serving-side-concentrated-flight-grid-v1"
OFFSETS_SECONDS = (-0.15, 0.05, 0.20, 0.35, 0.55, 0.80, 1.10, 1.40, 1.75)
PHASE_PAIRS: Mapping[str, tuple[int, ...]] = {
    "launch": (0, 1, 2),
    "early": (3, 4, 5),
    "late": (6, 7),
}
GLOBAL_STATISTICS = (
    "energyMean",
    "activeFraction",
    "centroidX",
    "centroidY",
    "spreadX",
    "spreadY",
    "entropy",
    "largestComponentFraction",
    "flowX",
    "flowY",
    "divergence",
    "bottomMinusTop",
    "smallComponentEnergyFraction",
    "smallComponentCentroidY",
    "smallComponentFlowY",
)
TRAJECTORY_STATISTICS = (
    "centroidY",
    "spreadY",
    "entropy",
    "flowY",
    "divergence",
    "bottomMinusTop",
    "smallComponentEnergyFraction",
    "smallComponentCentroidY",
    "smallComponentFlowY",
)


@dataclass(frozen=True)
class ResidualMotion:
    energy: np.ndarray
    flow_x: np.ndarray
    flow_y: np.ndarray
    divergence: np.ndarray


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is required; run `npm run analysis:setup` first"
        ) from error
    return cv2


def feature_names(grid_rows: int, grid_columns: int) -> tuple[str, ...]:
    if grid_rows < 2 or grid_columns < 2:
        raise ValueError("flight-motion grids need at least two rows and columns")
    names: list[str] = []
    for phase in PHASE_PAIRS:
        names.extend(
            f"{phase}:grid:r{row}:c{column}:energy"
            for row in range(grid_rows)
            for column in range(grid_columns)
        )
        names.extend(f"{phase}:row:r{row}:flowY" for row in range(grid_rows))
        names.extend(f"{phase}:{name}" for name in GLOBAL_STATISTICS)
    for transition in ("launchToEarly", "earlyToLate"):
        names.extend(
            f"trajectory:{transition}:{name}" for name in TRAJECTORY_STATISTICS
        )
        names.extend(
            f"trajectory:{transition}:row:r{row}:energy"
            for row in range(grid_rows)
        )
    return tuple(names)


def _gray(frame: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    value = np.asarray(frame)
    if value.ndim == 2:
        result = value.astype(np.uint8, copy=False)
    elif value.ndim == 3 and value.shape[2] == 3:
        result = cv2.cvtColor(value, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("flight frames must be grayscale or BGR images")
    if min(result.shape) < 32:
        raise ValueError("flight frames are too small")
    return result


def _affine_camera_flow(flow: np.ndarray) -> np.ndarray:
    """Fit translation, rotation, and zoom with one robust trimmed refit."""
    height, width = flow.shape[:2]
    stride = max(1, min(height, width) // 24)
    yy, xx = np.mgrid[0:height:stride, 0:width:stride]
    design = np.column_stack(
        (
            xx.reshape(-1) / max(width - 1, 1),
            yy.reshape(-1) / max(height - 1, 1),
            np.ones(xx.size),
        )
    )
    targets = flow[::stride, ::stride].reshape(-1, 2).astype(np.float64)
    coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]
    residual = np.linalg.norm(targets - design @ coefficients, axis=1)
    retained = residual <= np.percentile(residual, 75)
    if int(np.sum(retained)) >= 6:
        coefficients = np.linalg.lstsq(
            design[retained], targets[retained], rcond=None
        )[0]
    full_y, full_x = np.mgrid[0:height, 0:width]
    full_design = np.stack(
        (
            full_x / max(width - 1, 1),
            full_y / max(height - 1, 1),
            np.ones_like(full_x),
        ),
        axis=-1,
    )
    return (full_design @ coefficients).astype(np.float32)


def extract_motion_sequence(frames: Sequence[np.ndarray]) -> tuple[ResidualMotion, ...]:
    if len(frames) != len(OFFSETS_SECONDS):
        raise ValueError(f"expected {len(OFFSETS_SECONDS)} ordered frames")
    cv2 = _cv2()
    gray = [_gray(frame) for frame in frames]
    shape = gray[0].shape
    if any(frame.shape != shape for frame in gray):
        raise ValueError("flight frames must share one shape")
    height, width = shape
    diagonal = math.hypot(height, width)
    motions: list[ResidualMotion] = []
    for before, after in zip(gray[:-1], gray[1:], strict=True):
        flow = cv2.calcOpticalFlowFarneback(
            before,
            after,
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
        magnitude = np.linalg.norm(residual, axis=2) / diagonal
        threshold = max(7.5e-4, float(np.percentile(magnitude, 90)))
        energy = np.maximum(magnitude - threshold, 0.0)
        derivative_x = np.gradient(flow_x, axis=1) * width
        derivative_y = np.gradient(flow_y, axis=0) * height
        divergence = derivative_x + derivative_y
        motions.append(
            ResidualMotion(
                energy=np.ascontiguousarray(energy, dtype=np.float32),
                flow_x=np.ascontiguousarray(flow_x, dtype=np.float32),
                flow_y=np.ascontiguousarray(flow_y, dtype=np.float32),
                divergence=np.ascontiguousarray(divergence, dtype=np.float32),
            )
        )
    return tuple(motions)


def _weighted_mean(values: np.ndarray, weights: np.ndarray, total: float) -> float:
    return float(np.sum(values * weights) / total) if total > 0 else 0.0


def _pair_summary(
    motion: ResidualMotion, grid_rows: int, grid_columns: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cv2 = _cv2()
    energy = np.asarray(motion.energy, dtype=np.float64)
    height, width = energy.shape
    total = float(np.sum(energy))
    yy, xx = np.mgrid[0:height, 0:width]
    normalized_x = xx / max(width - 1, 1)
    normalized_y = yy / max(height - 1, 1)
    if total > 0:
        centroid_x = _weighted_mean(normalized_x, energy, total)
        centroid_y = _weighted_mean(normalized_y, energy, total)
        spread_x = math.sqrt(
            max(0.0, _weighted_mean((normalized_x - centroid_x) ** 2, energy, total))
        )
        spread_y = math.sqrt(
            max(0.0, _weighted_mean((normalized_y - centroid_y) ** 2, energy, total))
        )
        probabilities = energy[energy > 0] / total
        entropy = float(
            -np.sum(probabilities * np.log(probabilities))
            / max(math.log(energy.size), 1.0)
        )
    else:
        centroid_x = centroid_y = 0.5
        spread_x = spread_y = entropy = 0.0

    row_edges = np.linspace(0, height, grid_rows + 1, dtype=np.int64)
    column_edges = np.linspace(0, width, grid_columns + 1, dtype=np.int64)
    grid_energy = np.zeros((grid_rows, grid_columns), dtype=np.float64)
    row_flow_y = np.zeros(grid_rows, dtype=np.float64)
    for row in range(grid_rows):
        y0, y1 = int(row_edges[row]), int(row_edges[row + 1])
        row_energy = energy[y0:y1]
        row_total = float(np.sum(row_energy))
        if row_total > 0:
            row_flow_y[row] = _weighted_mean(
                motion.flow_y[y0:y1], row_energy, row_total
            )
        for column in range(grid_columns):
            x0, x1 = int(column_edges[column]), int(column_edges[column + 1])
            grid_energy[row, column] = float(np.sum(energy[y0:y1, x0:x1]))
    if total > 0:
        grid_energy /= total

    active = energy > 0
    largest_component_fraction = 0.0
    small_energy = small_centroid_y = small_flow_y = 0.0
    if np.any(active):
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            active.astype(np.uint8), connectivity=8
        )
        if component_count > 1:
            areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
            largest_component_fraction = float(np.max(areas) / energy.size)
            small_labels = np.flatnonzero(areas <= max(4.0, energy.size * 0.0025)) + 1
            small_mask = np.isin(labels, small_labels)
            small_total = float(np.sum(energy[small_mask]))
            if small_total > 0:
                small_energy = small_total / total
                small_centroid_y = _weighted_mean(
                    normalized_y[small_mask], energy[small_mask], small_total
                )
                small_flow_y = _weighted_mean(
                    motion.flow_y[small_mask], energy[small_mask], small_total
                )

    midpoint = height // 2
    bottom_minus_top = (
        float(np.sum(energy[midpoint:]) - np.sum(energy[:midpoint])) / total
        if total > 0
        else 0.0
    )
    statistics = np.asarray(
        (
            float(np.mean(energy)),
            float(np.mean(active)),
            centroid_x,
            centroid_y,
            spread_x,
            spread_y,
            entropy,
            largest_component_fraction,
            _weighted_mean(motion.flow_x, energy, total),
            _weighted_mean(motion.flow_y, energy, total),
            _weighted_mean(motion.divergence, energy, total),
            bottom_minus_top,
            small_energy,
            small_centroid_y,
            small_flow_y,
        ),
        dtype=np.float64,
    )
    return grid_energy.reshape(-1), row_flow_y, statistics


def summarize_motion(
    motions: Sequence[ResidualMotion], grid_rows: int, grid_columns: int
) -> dict[str, float]:
    if len(motions) != len(OFFSETS_SECONDS) - 1:
        raise ValueError("flight-motion sequence has the wrong number of pairs")
    names = feature_names(grid_rows, grid_columns)
    pair_values = [
        _pair_summary(motion, grid_rows, grid_columns) for motion in motions
    ]
    phases: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    output: dict[str, float] = {}
    for phase, indices in PHASE_PAIRS.items():
        grid = np.mean(np.stack([pair_values[index][0] for index in indices]), axis=0)
        rows = np.mean(np.stack([pair_values[index][1] for index in indices]), axis=0)
        statistics = np.mean(
            np.stack([pair_values[index][2] for index in indices]), axis=0
        )
        phases[phase] = grid, rows, statistics
        for row in range(grid_rows):
            for column in range(grid_columns):
                output[f"{phase}:grid:r{row}:c{column}:energy"] = float(
                    grid[row * grid_columns + column]
                )
        for row, value in enumerate(rows):
            output[f"{phase}:row:r{row}:flowY"] = float(value)
        for name, value in zip(GLOBAL_STATISTICS, statistics, strict=True):
            output[f"{phase}:{name}"] = float(value)

    statistic_indices = {
        name: GLOBAL_STATISTICS.index(name) for name in TRAJECTORY_STATISTICS
    }
    for transition, before, after in (
        ("launchToEarly", "launch", "early"),
        ("earlyToLate", "early", "late"),
    ):
        for name, index in statistic_indices.items():
            output[f"trajectory:{transition}:{name}"] = float(
                phases[after][2][index] - phases[before][2][index]
            )
        before_rows = phases[before][0].reshape(grid_rows, grid_columns).sum(axis=1)
        after_rows = phases[after][0].reshape(grid_rows, grid_columns).sum(axis=1)
        for row, value in enumerate(after_rows - before_rows):
            output[f"trajectory:{transition}:row:r{row}:energy"] = float(value)
    if tuple(output) != names or not all(math.isfinite(value) for value in output.values()):
        raise AssertionError("flight-motion feature contract changed or became non-finite")
    return output


def extract_flight_features(
    frames: Sequence[np.ndarray], grid_rows: int, grid_columns: int
) -> dict[str, float]:
    return summarize_motion(
        extract_motion_sequence(frames), grid_rows, grid_columns
    )
