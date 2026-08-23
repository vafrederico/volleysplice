"""Non-reanchored latent point-count prior for side-switch candidates."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from analysis.side_switch_winner_experiment import logit, sigmoid


_BOUNDARY = re.compile(r":boundary:R(\d+):R(\d+)$")
_RANGE = re.compile(r"^R(\d+)$")


@dataclass(frozen=True)
class PointTransition:
    redo_probability: float
    point_probability: float
    missed_point_probability: float
    kernel_sigma_points: float

    def __post_init__(self) -> None:
        probabilities = (
            self.redo_probability,
            self.point_probability,
            self.missed_point_probability,
        )
        if any(not math.isfinite(value) or value < 0 for value in probabilities):
            raise ValueError("point-transition probabilities must be finite and nonnegative")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-12):
            raise ValueError("point-transition probabilities must sum to one")
        if not math.isfinite(self.kernel_sigma_points) or self.kernel_sigma_points <= 0:
            raise ValueError("cadence kernel sigma must be finite and positive")

    def to_dict(self) -> dict[str, float]:
        return {
            "redoProbability": self.redo_probability,
            "pointProbability": self.point_probability,
            "missedPointProbability": self.missed_point_probability,
            "kernelSigmaPoints": self.kernel_sigma_points,
        }


def opportunity_index(row: Mapping[str, Any]) -> int:
    event_id = str(row.get("eventId", ""))
    boundary = _BOUNDARY.search(event_id)
    if boundary:
        left = int(boundary.group(1))
        right = int(boundary.group(2))
        if right != left + 1:
            raise ValueError(f"non-adjacent rally boundary: {event_id}")
        return left
    source_range = str(row.get("sourceRangeId", ""))
    match = _RANGE.match(source_range)
    if match:
        return int(match.group(1))
    internal = re.search(r":internal-dead-peak:R(\d+):", event_id)
    if internal:
        return int(internal.group(1))
    raise ValueError(f"candidate has no rally opportunity index: {event_id}")


def latent_count_distributions(
    maximum_observation: int, transition: PointTransition
) -> list[np.ndarray]:
    if maximum_observation < 0:
        raise ValueError("maximum observation must be nonnegative")
    distributions = [np.asarray([1.0], dtype=np.float64)]
    for _ in range(maximum_observation):
        previous = distributions[-1]
        current = np.zeros(len(previous) + 2, dtype=np.float64)
        current[: len(previous)] += transition.redo_probability * previous
        current[1 : len(previous) + 1] += transition.point_probability * previous
        current[2 : len(previous) + 2] += transition.missed_point_probability * previous
        current /= float(np.sum(current))
        distributions.append(current)
    return distributions


def _cadence_kernel(count: int, sigma: float) -> float:
    if count <= 0:
        return 0.0
    remainder = count % 7
    distance = min(remainder, 7 - remainder)
    return math.exp(-0.5 * (distance / sigma) ** 2)


def cadence_hazards(
    rows: Sequence[Mapping[str, Any]], transition: PointTransition
) -> np.ndarray:
    indexes = [opportunity_index(row) for row in rows]
    distributions = latent_count_distributions(max(indexes, default=0), transition)
    hazards = np.asarray(
        [
            sum(
                float(probability)
                * _cadence_kernel(count, transition.kernel_sigma_points)
                for count, probability in enumerate(distributions[index])
            )
            for index in indexes
        ],
        dtype=np.float64,
    )
    if not np.isfinite(hazards).all() or np.any((hazards < 0) | (hazards > 1)):
        raise ValueError("latent score prior produced an invalid hazard")
    return hazards


def centered_prior_logits(
    rows: Sequence[Mapping[str, Any]], transition: PointTransition
) -> np.ndarray:
    hazards = cadence_hazards(rows, transition)
    logits = np.asarray([logit(float(value)) for value in hazards])
    centered = np.full(len(rows), np.nan, dtype=np.float64)
    for recording_id in sorted({str(row["recordingId"]) for row in rows}):
        indexes = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if str(row["recordingId"]) == recording_id
            ]
        )
        centered[indexes] = logits[indexes] - float(np.median(logits[indexes]))
    if not np.isfinite(centered).all():
        raise ValueError("cadence prior centering left candidates invalid")
    return centered


def apply_soft_score_prior(
    scores: np.ndarray, prior_logits: np.ndarray, weight: float
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    prior = np.asarray(prior_logits, dtype=np.float64)
    if (
        values.shape != prior.shape
        or values.ndim != 1
        or not np.isfinite(values).all()
        or not np.isfinite(prior).all()
        or np.any((values < 0) | (values > 1))
    ):
        raise ValueError("soft score prior needs finite aligned inputs")
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("soft score prior weight must be finite and nonnegative")
    adjusted = np.asarray([logit(float(value)) for value in values]) + weight * prior
    return sigmoid(adjusted)
