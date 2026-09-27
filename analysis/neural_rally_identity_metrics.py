"""Identity-preserving rally and boundary metrics for the review proposal study.

This module scores unpadded event objects. It never unions prediction identities,
even when they overlap, touch, or their exports would join. Gold touched by ignored
time is excluded from event/boundary scoring. Ignored and censored-gold spans are
subtracted from each prediction's support, retaining its original identity and
original boundaries. Entirely masked predictions disappear from the denominator.

Rally-start localization is a serve-contact proxy under the annotation policy;
it is not a test of serving side, scoring outcome, or score reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .crop_evaluation import subtract_intervals
from .schema import Interval


BOUNDARY_TOLERANCES_SECONDS = (0.25, 0.5, 1.0, 2.0)
EVENT_IOU_THRESHOLD = 0.5
MATERIAL_OVERLAP_MAX_SECONDS = 0.5
MATERIAL_OVERLAP_GOLD_FRACTION = 0.1


@dataclass(frozen=True)
class _Identity:
    index: int
    interval: Interval
    support: tuple[Interval, ...]
    start_observed: bool = True
    end_observed: bool = True


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be finite and numeric")
    return float(value)


def _read(values: Iterable[Any], duration: float, name: str) -> list[_Identity]:
    if isinstance(values, (str, bytes, Mapping)):
        raise ValueError(f"{name} must be an interval iterable")
    result = []
    for index, item in enumerate(values):
        start_observed = end_observed = True
        if isinstance(item, Interval):
            start, end = item.start, item.end
        elif isinstance(item, Mapping):
            start, end = item.get("start"), item.get("end")
            start_observed, end_observed = item.get("startObserved", True), item.get("endObserved", True)
            if not isinstance(start_observed, bool) or not isinstance(end_observed, bool):
                raise ValueError("startObserved/endObserved must be booleans")
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 2:
            start, end = item
        else:
            raise ValueError(f"invalid {name}[{index}]")
        start, end = _number(start, "start"), _number(end, "end")
        if end <= start:
            raise ValueError(f"{name}[{index}] must have start < end")
        start, end = max(0.0, start), min(duration, end)
        if end > start:
            interval = Interval(start, end)
            result.append(_Identity(index, interval, (interval,), start_observed, end_observed))
    return sorted(result, key=lambda x: (x.interval.start, x.interval.end, x.index))


def _overlap(left: Interval, right: Interval) -> float:
    return max(0.0, min(left.end, right.end) - max(left.start, right.start))


def _maximum_weight_pairs(weights: np.ndarray, eligible: np.ndarray) -> list[tuple[int, int]]:
    """General bipartite cardinality-first matching, then maximum total weight.

Weights must be in [0, 1]. Rectangular Hungarian assignment has dummy unmatched
columns. A cardinality reward larger than any total weight difference ensures
that a low-weight extra match beats any smaller matching. Stable array order
breaks exact ties. No chronological constraint is imposed on nested predictions.
"""
    weights = np.asarray(weights, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=np.bool_)
    if weights.ndim != 2 or eligible.shape != weights.shape:
        raise ValueError("weights and eligible must have identical 2-D shapes")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0) or np.any(weights > 1):
        raise ValueError("matching weights must lie in [0, 1]")
    truth_count, prediction_count = weights.shape
    if not truth_count or not prediction_count:
        return []
    transposed = truth_count > prediction_count
    if transposed:
        weights, eligible = weights.T, eligible.T
    n, m = weights.shape
    costs = np.zeros((n, m + n), dtype=np.float64)
    costs[:, :m] = np.where(eligible, -(n + 1.0 + weights), 1.0)
    width = m + n
    u, v = np.zeros(n + 1), np.zeros(width + 1)
    p, way = np.zeros(width + 1, dtype=np.int64), np.zeros(width + 1, dtype=np.int64)
    for row in range(1, n + 1):
        p[0] = row
        j0 = 0
        minv = np.full(width + 1, np.inf)
        used = np.zeros(width + 1, dtype=np.bool_)
        while True:
            used[j0] = True
            i0 = p[j0]
            unused = np.flatnonzero(~used[1:]) + 1
            cur = costs[i0 - 1, unused - 1] - u[i0] - v[unused]
            improved = cur < minv[unused]
            targets = unused[improved]
            minv[targets] = cur[improved]
            way[targets] = j0
            j1 = int(unused[int(np.argmin(minv[unused]))])
            delta = minv[j1]
            u[p[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
    result = []
    for column in range(1, m + 1):
        if p[column] and eligible[p[column] - 1, column - 1]:
            pair = (int(p[column] - 1), column - 1)
            result.append((pair[1], pair[0]) if transposed else pair)
    return sorted(result)


def _rates(matched: int, predicted: int, true: int) -> dict[str, Any]:
    # No eligible truth is not evidence of perfect detection. Count statistics
    # remain available for pooling; precision can still be zero on false alarms.
    precision = matched / predicted if predicted else (0.0 if true else None)
    recall = matched / true if true else None
    f1 = 2.0 * matched / (predicted + true) if true else None
    return {"true": true, "predicted": predicted, "matched": matched,
            "falsePositive": predicted - matched, "falseNegative": true - matched,
            "precision": precision, "recall": recall, "f1": f1}


def _timestamp_matches(truth: list[float], predictions: list[float], tolerance: float) -> int:
    """Maximum-cardinality 1-D tolerance matching by earliest eligible pair."""
    truth, predictions = sorted(truth), sorted(predictions)
    i = j = matched = 0
    while i < len(truth) and j < len(predictions):
        if predictions[j] < truth[i] - tolerance:
            j += 1
        elif predictions[j] > truth[i] + tolerance:
            i += 1
        else:
            matched += 1
            i += 1
            j += 1
    return matched


def _boundary_available(time: float, excluded: Sequence[Interval], *, is_end: bool) -> bool:
    # Starts are approached from the right, ends from the left. An event ending
    # exactly at an ignored interval's start still has an observed end boundary.
    if is_end:
        return not any(span.start < time <= span.end for span in excluded)
    return not any(span.start <= time < span.end for span in excluded)


def _error_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "maeSeconds": None, "medianAbsoluteSeconds": None,
                "p95AbsoluteSeconds": None, "biasSeconds": None}
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "maeSeconds": float(np.mean(np.abs(array))),
            "medianAbsoluteSeconds": float(np.median(np.abs(array))),
            "p95AbsoluteSeconds": float(np.percentile(np.abs(array), 95)),
            "biasSeconds": float(np.mean(array))}


def _record(row: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("id", "sourceGroup"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"{key} must be nonempty")
    duration = _number(row.get("durationSeconds"), "durationSeconds")
    if duration <= 0:
        raise ValueError("durationSeconds must be positive")
    raw_truth = _read(row["rallies"], duration, "rallies")
    if any(right.interval.start < left.interval.end for left, right in zip(raw_truth, raw_truth[1:])):
        raise ValueError("gold identities must not overlap")
    raw_predictions = _read(row["predictions"], duration, "predictions")
    ignored = tuple(item.interval for item in _read(row.get("ignoredIntervals", ()), duration, "ignoredIntervals"))
    censored = [item for item in raw_truth if any(_overlap(item.interval, mask) > 0 for mask in ignored)]
    censored_indexes = {item.index for item in censored}
    truth = [item for item in raw_truth if item.index not in censored_indexes]
    excluded = (*ignored, *(item.interval for item in censored))
    predictions = []
    for item in raw_predictions:
        support = tuple(subtract_intervals(item.support, excluded))
        if support:
            predictions.append(_Identity(item.index, item.interval, support, item.start_observed, item.end_observed))
    intersections = np.zeros((len(truth), len(predictions)), dtype=np.float64)
    ious = np.zeros_like(intersections)
    for i, actual in enumerate(truth):
        for j, prediction in enumerate(predictions):
            shared = sum(_overlap(actual.interval, fragment) for fragment in prediction.support)
            intersections[i, j] = shared
            denominator = actual.interval.end - actual.interval.start + sum(x.end - x.start for x in prediction.support) - shared
            ious[i, j] = shared / denominator
    matches = _maximum_weight_pairs(ious, ious >= EVENT_IOU_THRESHOLD)
    overlap_edges = intersections > 0
    material_threshold = np.asarray([
        min(MATERIAL_OVERLAP_MAX_SECONDS, MATERIAL_OVERLAP_GOLD_FRACTION * (x.interval.end - x.interval.start))
        for x in truth
    ], dtype=np.float64)
    material_edges = intersections >= material_threshold[:, None]
    starts_available = [_boundary_available(x.interval.start, excluded, is_end=False) for x in predictions]
    ends_available = [_boundary_available(x.interval.end, excluded, is_end=True) for x in predictions]
    starts_observed = [available and x.start_observed for x, available in zip(predictions, starts_available)]
    ends_observed = [available and x.end_observed for x, available in zip(predictions, ends_available)]
    start_errors = [predictions[j].interval.start - truth[i].interval.start for i, j in matches if starts_available[j]]
    end_errors = [predictions[j].interval.end - truth[i].interval.end for i, j in matches if ends_available[j]]
    result: dict[str, Any] = {
        "id": row["id"], "sourceGroup": row["sourceGroup"],
        "originalTrueRallies": len(raw_truth), "ignoredTouchedTrueRallies": len(censored),
        "originalPredictedRallies": len(raw_predictions),
        "entirelyMaskedPredictions": len(raw_predictions) - len(predictions),
        "trueRallies": len(truth), "predictedRallies": len(predictions), "matchedRallies": len(matches),
        "completeMisses": int(np.sum(~np.any(overlap_edges, axis=1))),
        "mergedPredictions": int(np.sum(np.sum(overlap_edges, axis=0) > 1)),
        "splitTrueRallies": int(np.sum(np.sum(overlap_edges, axis=1) > 1)),
        "mergedPredictionsMaterial": int(np.sum(np.sum(material_edges, axis=0) > 1)),
        "splitTrueRalliesMaterial": int(np.sum(np.sum(material_edges, axis=1) > 1)),
        "unobservedPredictionStarts": sum(not x.start_observed for x in predictions),
        "unobservedPredictionEnds": sum(not x.end_observed for x in predictions),
        "startLocalization": {}, "endLocalization": {},
        "observedStartLocalization": {}, "observedEndLocalization": {}, "matchedEventBoundaries": {},
        "matchedStartErrorsSeconds": start_errors, "matchedEndErrorsSeconds": end_errors,
        "matches": [{"truthIndex": truth[i].index, "predictionIndex": predictions[j].index,
                     "iou": float(ious[i, j])} for i, j in matches],
    }
    for tolerance in BOUNDARY_TOLERANCES_SECONDS:
        key = format(tolerance, "g")
        for field, available, name in (
            ("start", starts_available, "startLocalization"), ("end", ends_available, "endLocalization"),
            ("start", starts_observed, "observedStartLocalization"), ("end", ends_observed, "observedEndLocalization"),
        ):
            actual = [getattr(x.interval, field) for x in truth]
            predicted = [getattr(x.interval, field) for x, valid in zip(predictions, available) if valid]
            result[name][key] = _rates(_timestamp_matches(actual, predicted, tolerance), len(predicted), len(actual))
        start_correct = sum(abs(x) <= tolerance for x in start_errors)
        end_correct = sum(abs(x) <= tolerance for x in end_errors)
        both_correct = sum(starts_available[j] and ends_available[j]
                           and abs(predictions[j].interval.start - truth[i].interval.start) <= tolerance
                           and abs(predictions[j].interval.end - truth[i].interval.end) <= tolerance
                           for i, j in matches)
        result["matchedEventBoundaries"][key] = {
            "startCorrect": start_correct, "endCorrect": end_correct, "bothCorrect": both_correct,
            "eligibleTrueRallies": len(truth),
        }
    return _summarize([result], preserve=result)


_COUNT_KEYS = (
    "originalTrueRallies", "ignoredTouchedTrueRallies", "originalPredictedRallies",
    "entirelyMaskedPredictions", "trueRallies", "predictedRallies", "matchedRallies",
    "completeMisses", "mergedPredictions", "splitTrueRallies",
    "mergedPredictionsMaterial", "splitTrueRalliesMaterial",
    "unobservedPredictionStarts", "unobservedPredictionEnds",
)


def _summarize(rows: Sequence[Mapping[str, Any]], *, preserve: dict[str, Any] | None = None) -> dict[str, Any]:
    result = dict(preserve or {})
    result.update({key: sum(row[key] for row in rows) for key in _COUNT_KEYS})
    rates = _rates(result["matchedRallies"], result["predictedRallies"], result["trueRallies"])
    result.update({"eventPrecision": rates["precision"], "eventRecall": rates["recall"], "eventF1": rates["f1"],
                   "falsePositiveRallies": rates["falsePositive"], "falseNegativeRallies": rates["falseNegative"]})
    result["matchedStartErrorsSeconds"] = [value for row in rows for value in row["matchedStartErrorsSeconds"]]
    result["matchedEndErrorsSeconds"] = [value for row in rows for value in row["matchedEndErrorsSeconds"]]
    result["matchedStartError"] = _error_summary(result["matchedStartErrorsSeconds"])
    result["matchedEndError"] = _error_summary(result["matchedEndErrorsSeconds"])
    for name in ("startLocalization", "endLocalization", "observedStartLocalization", "observedEndLocalization"):
        result[name] = {}
        for tolerance in BOUNDARY_TOLERANCES_SECONDS:
            key = format(tolerance, "g")
            counts = {field: sum(row[name][key][field] for row in rows) for field in ("matched", "predicted", "true")}
            result[name][key] = _rates(**counts)
    result["matchedEventBoundaries"] = {}
    for tolerance in BOUNDARY_TOLERANCES_SECONDS:
        key = format(tolerance, "g")
        counts = {field: sum(row["matchedEventBoundaries"][key][field] for row in rows)
                  for field in ("startCorrect", "endCorrect", "bothCorrect", "eligibleTrueRallies")}
        for field in ("start", "end", "both"):
            counts[field + "Recall"] = counts[field + "Correct"] / counts["eligibleTrueRallies"] if counts["eligibleTrueRallies"] else None
        result["matchedEventBoundaries"][key] = counts
    return result


def evaluate_rally_identities(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Pool raw event counts per seed; retain source-group and recording readouts.

Accepts id, sourceGroup, durationSeconds, rallies, predictions, ignoredIntervals.
Inputs can contain overlapping/nested predictions, but gold events cannot overlap.
Interval mappings, schema.Interval objects, and two-element sequences are accepted.
"""
    rows = [_record(row) for row in records]
    if not rows:
        raise ValueError("cannot evaluate an empty recording set")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("recording IDs must be unique; do not pool seeds as recordings")
    return {
        "metricContract": {
            "version": "rally-identity-v1", "eventIouThreshold": EVENT_IOU_THRESHOLD,
            "eventMatching": "general bipartite maximum cardinality then maximum total IoU; deterministic stable-order ties",
            "identityRule": "raw identities stay separate; no padding or union; masked fragments retain one parent identity",
            "ignoredRule": "exclude touched gold identities and their whole spans; subtract those spans and ignored time per prediction identity",
            "boundaryRule": "original clipped-video boundaries; no synthetic endpoints at ignored masks; start viewed from right, end from left",
            "localizationRule": "maximum-cardinality independent boundary timestamp matching within tolerance; no event IoU requirement",
            "observedLocalizationRule": "same timestamp matching but filter predictions by startObserved/endObserved flags; missing flags default true; gold denominator unchanged",
            "startInterpretation": "rally-start localization; serve-contact proxy, not serve-side or score correctness",
            "matchedBoundaryRule": "start/end/both correctness requires IoU-matched identity; recall denominator all eligible gold identities",
            "boundaryTolerancesSeconds": list(BOUNDARY_TOLERANCES_SECONDS),
            "mergeSplitRule": "any positive temporal overlap; also material sensitivity >= min(0.5 seconds, 10% gold duration)",
            "aggregation": "sum counts across recordings before P/R/F1; do not average recording F1",
            "emptyTruthRule": "recall and F1 unavailable if no eligible truth; counts retained for pooling",
        },
        "pooled": _summarize(rows),
        "sourceGroups": {group: _summarize([row for row in rows if row["sourceGroup"] == group])
                         for group in sorted({row["sourceGroup"] for row in rows})},
        "recordings": rows,
    }
