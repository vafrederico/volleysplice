"""Independent-gap decoding for side-switch cadence ablations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from analysis.side_switch_v3 import V3Event, event_metrics


DECODER_KIND = "volleycut-side-switch-independent-gap-decoder-v1"


class SideSwitchNoCadenceError(ValueError):
    pass


@dataclass(frozen=True)
class IndependentGapDecoderSettings:
    """A decoder contract with no cadence, spacing, count cap, or re-anchoring."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": DECODER_KIND,
            "selectionRule": "select every gap whose classifier score is at least threshold",
            "usesCadence": False,
            "minimumGapSeparation": 0,
            "maximumPredictions": None,
            "reanchorOnSelection": False,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "IndependentGapDecoderSettings":
        expected = cls().to_dict()
        if dict(payload) != expected:
            raise SideSwitchNoCadenceError(
                "independent-gap decoder settings changed or contain a cadence constraint"
            )
        return cls()


def independent_gap_predictions(
    probabilities: np.ndarray, threshold: float
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise SideSwitchNoCadenceError("probabilities must be a finite vector")
    if np.any((values < 0.0) | (values > 1.0)):
        raise SideSwitchNoCadenceError("probabilities must stay in [0, 1]")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise SideSwitchNoCadenceError("threshold must stay in [0, 1]")
    return values >= threshold


def _thresholds(probabilities: np.ndarray) -> tuple[float, ...]:
    if not len(probabilities):
        raise SideSwitchNoCadenceError("threshold selection needs scored gaps")
    values = sorted({float(value) for value in probabilities}, reverse=True)
    above_maximum = min(1.0, math.nextafter(values[0], math.inf))
    return tuple(dict.fromkeys((above_maximum, *values)))


def _selection_rank(candidate: Mapping[str, Any]) -> tuple[float, ...]:
    metrics = candidate["metrics"]
    return (
        float(metrics["f1"] or 0.0),
        float(metrics["precision"] or 0.0),
        float(metrics["recall"] or 0.0),
        float(candidate["threshold"]),
    )


def select_independent_threshold(
    events: Sequence[V3Event], probabilities: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    """Select one independent score threshold by exact-gap validation metrics."""

    values = np.asarray(probabilities, dtype=np.float64)
    if len(events) != len(values):
        raise SideSwitchNoCadenceError("events and probabilities are not aligned")
    # Validate once before searching so malformed values cannot silently compare.
    independent_gap_predictions(values, 0.5)
    candidates: list[dict[str, Any]] = []
    for threshold in _thresholds(values):
        predictions = independent_gap_predictions(values, threshold)
        candidates.append(
            {
                "threshold": threshold,
                "metrics": event_metrics(events, predictions, tolerance=0),
            }
        )
    selected = max(candidates, key=_selection_rank)
    predictions = independent_gap_predictions(values, float(selected["threshold"]))
    return (
        {
            **selected,
            "candidateThresholds": len(candidates),
            "ranking": "exact F1, precision, recall, then higher threshold",
        },
        predictions,
    )


def prediction_structure(
    events: Sequence[V3Event], predictions: np.ndarray
) -> dict[str, Any]:
    """Describe proposal clustering without consulting labels."""

    selected = np.asarray(predictions, dtype=bool)
    if selected.ndim != 1 or len(events) != len(selected):
        raise SideSwitchNoCadenceError("events and predictions are not aligned")
    by_recording: dict[str, Any] = {}
    total_clusters = 0
    adjacent_pairs = 0
    maximum_run = 0
    all_spacings: list[int] = []
    for recording_id in sorted({event.recording_id for event in events}):
        gaps = sorted(
            event.gap_order
            for index, event in enumerate(events)
            if event.recording_id == recording_id and selected[index]
        )
        clusters: list[list[int]] = []
        for gap in gaps:
            if clusters and gap == clusters[-1][-1] + 1:
                clusters[-1].append(gap)
            else:
                clusters.append([gap])
        spacings = [right - left for left, right in zip(gaps, gaps[1:])]
        all_spacings.extend(spacings)
        local_adjacent_pairs = sum(value == 1 for value in spacings)
        local_maximum_run = max((len(cluster) for cluster in clusters), default=0)
        adjacent_pairs += local_adjacent_pairs
        total_clusters += len(clusters)
        maximum_run = max(maximum_run, local_maximum_run)
        by_recording[recording_id] = {
            "selectedGaps": gaps,
            "selectedCount": len(gaps),
            "consecutiveClusters": clusters,
            "consecutiveClusterCount": len(clusters),
            "adjacentSelectedPairs": local_adjacent_pairs,
            "maximumConsecutiveRun": local_maximum_run,
            "minimumSelectedSpacing": min(spacings) if spacings else None,
        }
    return {
        "selectedCount": int(np.sum(selected)),
        "consecutiveClusterCount": total_clusters,
        "adjacentSelectedPairs": adjacent_pairs,
        "maximumConsecutiveRun": maximum_run,
        "minimumSelectedSpacing": min(all_spacings) if all_spacings else None,
        "byRecording": by_recording,
    }


__all__ = [
    "DECODER_KIND",
    "IndependentGapDecoderSettings",
    "SideSwitchNoCadenceError",
    "independent_gap_predictions",
    "prediction_structure",
    "select_independent_threshold",
]
