#!/usr/bin/env python3
"""Explain the promoted side-switch model on recording-held-out development data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.side_switch_full_union_ranker import (
    DERIVED_FEATURE_NAMES,
    UnionDecoderSettings,
    add_derived_features,
    decode_ranked_candidates,
    fit_weighted_logistic,
)
from analysis.side_switch_full_video import (
    event_metric_counts,
    monotonic_interval_match,
)
from analysis.side_switch_production_state import (
    SERVE_ANCHOR_FEATURE_NAMES,
    STATE_GATE_FEATURE_NAMES,
)
from analysis.side_switch_v3 import V3Event, average_precision
from analysis.side_switch_v5 import VISUAL_FEATURE_NAMES
from analysis.side_switch_v6 import V6Model, matrix_for


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
REPORTS = ROOT / "reports/side-switch"
DEFAULT_FEATURES = REPORTS / "side-switch-full-union-v5-state-features-v1.json"
DEFAULT_FULL_AUDIT = (
    REPORTS / "side-switch-full-video-marker-evaluation-2026-08-21-r2.json"
)
DEFAULT_MODEL = ROOT / "models/side-switch-hard-negative-mining-v1/model.json"
DEFAULT_EVALUATION = REPORTS / "side-switch-hard-negative-mining-v1-evaluation.json"
DEFAULT_PREVIOUS_MODEL = ROOT / "models/side-switch-v5-peak-cleanup-v1/model.json"
DEFAULT_OUTPUT = REPORTS / "side-switch-feature-importance-2026-08-24.json"

EXPECTED_SHA256 = {
    "features": "9763cb3e5cd9baada64f4bf54f06140dcff5068bb8cd74a1d485c677d1e6c551",
    "fullAudit": "142d6617aed7f20b51cdcdfc9b3e61c8c76beaf79a31fc2d7ee54fb0fdf27b89",
    "model": "c2570481c30dec62f56ac3284cd4028763ffd8e72a9ad07e9908fec10df387e3",
    "evaluation": "e67088b36d177d68c24587efcb186eab4201b5294532be1fdf978ef13aaacc4b",
    "previousModel": "82e64c69564d17c3dfbea9c5f4a276cdd161a0c400507d3df7589c019e44ee3f",
}

FEATURE_NAMES = (*VISUAL_FEATURE_NAMES, *STATE_GATE_FEATURE_NAMES, *DERIVED_FEATURE_NAMES)
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "courtBandAppearance": tuple(FEATURE_NAMES[0:6]),
    "playerAssignmentAppearance": (
        "playerSameAssignmentCost",
        "playerSwappedAssignmentCost",
        "playerSwapMargin",
        "playerOrientationFlipEvidence",
        "playerGlobalAppearanceChange",
    ),
    "playerObservationQuality": (
        "minimumPlayerSideSeparation",
        "playerSideSeparationChange",
        "beforePlayerPaletteInstability",
        "afterPlayerPaletteInstability",
        "minimumProposalCoverage",
        "proposalCoverageChange",
        "minimumProposalCount",
        "proposalCountChange",
        "minimumNearSupport",
        "minimumFarSupport",
        "sideSupportImbalanceChange",
    ),
    "productionAdjacentRallySupport": tuple(FEATURE_NAMES[22:26]),
    "productionGapState": tuple(FEATURE_NAMES[26:32]),
    "candidateProvenance": tuple(DERIVED_FEATURE_NAMES),
}
L2 = 0.1
CLASS_BALANCE_EXPONENT = 0.5
HARD_NEGATIVES_PER_RECORDING = 2
HARD_NEGATIVE_MULTIPLIER = 2.0
PADDING = 4.0
THRESHOLD_QUANTILES = 65
DECODER = UnionDecoderSettings(
    minimum_index_separation=2,
    free_predictions_per_recording=6,
    count_penalty_logit=0.5,
)


@dataclass
class FoldState:
    recording_id: str
    indexes: np.ndarray
    rows: list[Mapping[str, Any]]
    events: list[V3Event]
    model: V6Model
    threshold: float
    scores: np.ndarray
    predictions: np.ndarray
    trace: dict[str, Mapping[str, Any]]


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _proposal(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "eventId": str(row["eventId"]),
        "recordingId": str(row["recordingId"]),
        "kind": str(row["kind"]),
        "gapStart": float(row["gapStart"]),
        "gapEnd": float(row["gapEnd"]),
        "transitionTime": float(row["transitionTime"]),
    }


def _labels_and_targets(
    rows: Sequence[Mapping[str, Any]], markers: Mapping[str, Sequence[Mapping[str, Any]]]
) -> tuple[dict[str, int], dict[tuple[str, int], int]]:
    labels = {str(row["eventId"]): 0 for row in rows}
    targets: dict[tuple[str, int], int] = {}
    for recording_id, truth in markers.items():
        indexes = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if str(row["recordingId"]) == recording_id
            ],
            dtype=np.int64,
        )
        local = [rows[index] for index in indexes]
        matched = monotonic_interval_match(
            [_proposal(row) for row in local], truth, PADDING
        )
        for pair in matched.pairs:
            global_index = int(indexes[pair.proposal_index])
            labels[str(rows[global_index]["eventId"])] = 1
            targets[(recording_id, pair.marker_index)] = global_index
    return labels, targets


def _events(
    rows: Sequence[Mapping[str, Any]], labels: Mapping[str, int]
) -> list[V3Event]:
    order: dict[str, int] = {}
    for recording_id in sorted({str(row["recordingId"]) for row in rows}):
        local = sorted(
            (row for row in rows if str(row["recordingId"]) == recording_id),
            key=lambda row: (float(row["transitionTime"]), str(row["eventId"])),
        )
        order.update({str(row["eventId"]): index for index, row in enumerate(local, 1)})
    return [
        V3Event(
            str(row["eventId"]),
            str(row["recordingId"]),
            "opened-development",
            order[str(row["eventId"])],
            labels[str(row["eventId"])],
            row,
        )
        for row in rows
    ]


def _fit(
    events: Sequence[V3Event], feature_names: Sequence[str]
) -> tuple[V6Model, dict[str, list[str]]]:
    names = tuple(feature_names)
    initial = fit_weighted_logistic(
        events, L2, names, CLASS_BALANCE_EXPONENT
    )
    initial_scores = initial.predict_proba(matrix_for(events, names))
    multipliers = np.ones(len(events), dtype=np.float64)
    selected: dict[str, list[str]] = {}
    for recording_id in sorted({event.recording_id for event in events}):
        negatives = [
            index
            for index, event in enumerate(events)
            if event.recording_id == recording_id and event.label == 0
        ]
        chosen = sorted(
            negatives,
            key=lambda index: (-float(initial_scores[index]), events[index].event_id),
        )[:HARD_NEGATIVES_PER_RECORDING]
        multipliers[chosen] = HARD_NEGATIVE_MULTIPLIER
        selected[recording_id] = [events[index].event_id for index in chosen]
    return (
        fit_weighted_logistic(
            events,
            L2,
            names,
            CLASS_BALANCE_EXPONENT,
            multipliers,
        ),
        selected,
    )


def _crossfit(
    rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, int],
    feature_names: Sequence[str],
) -> np.ndarray:
    scores = np.full(len(rows), np.nan, dtype=np.float64)
    events = _events(rows, labels)
    for held_id in sorted({event.recording_id for event in events}):
        fit_indexes = np.asarray(
            [
                index
                for index, event in enumerate(events)
                if event.recording_id != held_id
            ],
            dtype=np.int64,
        )
        held_indexes = np.asarray(
            [
                index
                for index, event in enumerate(events)
                if event.recording_id == held_id
            ],
            dtype=np.int64,
        )
        model, _ = _fit([events[index] for index in fit_indexes], feature_names)
        scores[held_indexes] = model.predict_proba(
            matrix_for([events[index] for index in held_indexes], feature_names)
        )
    if not np.isfinite(scores).all():
        raise ValueError("cross-fitting left candidate rows unscored")
    return scores


def _evaluate(
    rows: Sequence[Mapping[str, Any]],
    predictions: np.ndarray,
    markers: Mapping[str, Sequence[Mapping[str, Any]]],
    padding: float = PADDING,
) -> dict[str, Any]:
    by_recording: dict[str, Any] = {}
    for recording_id, truth in markers.items():
        proposals = sorted(
            [
                _proposal(row)
                for row, selected in zip(rows, predictions, strict=True)
                if selected and str(row["recordingId"]) == recording_id
            ],
            key=lambda row: (row["transitionTime"], row["eventId"]),
        )
        match = monotonic_interval_match(proposals, truth, padding)
        by_recording[recording_id] = {
            "humanEvents": len(truth),
            "proposals": len(proposals),
            **event_metric_counts(match),
        }
    tp = sum(int(value["truePositives"]) for value in by_recording.values())
    fp = sum(int(value["falsePositives"]) for value in by_recording.values())
    fn = sum(int(value["falseNegatives"]) for value in by_recording.values())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "paddingSeconds": padding,
        "proposals": tp + fp,
        "truePositives": tp,
        "falsePositives": fp,
        "falseNegatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "macroPerVideoPrecision": float(
            np.mean([float(value["precision"] or 0.0) for value in by_recording.values()])
        ),
        "macroPerVideoRecall": float(
            np.mean([float(value["recall"] or 0.0) for value in by_recording.values()])
        ),
        "byRecording": by_recording,
    }


def _thresholds(scores: np.ndarray) -> tuple[float, ...]:
    quantiles = np.quantile(scores, np.linspace(0.0, 1.0, THRESHOLD_QUANTILES))
    above = min(1.0, math.nextafter(float(np.max(scores)), math.inf))
    return tuple(sorted({above, *(float(value) for value in quantiles)}, reverse=True))


def _rank(metrics: Mapping[str, Any], threshold: float) -> tuple[float, ...]:
    return (
        float(metrics["f1"]),
        float(metrics["precision"]),
        float(metrics["recall"]),
        -float(metrics["proposals"]),
        threshold,
    )


def _select_threshold(
    rows: Sequence[Mapping[str, Any]],
    scores: np.ndarray,
    markers: Mapping[str, Sequence[Mapping[str, Any]]],
) -> float:
    best: tuple[tuple[float, ...], float] | None = None
    for threshold in _thresholds(scores):
        predictions = decode_ranked_candidates(rows, scores, threshold, DECODER)
        metrics = _evaluate(rows, predictions, markers)
        value = (_rank(metrics, threshold), threshold)
        if best is None or value[0] > best[0]:
            best = value
    if best is None:
        raise AssertionError("threshold selection failed")
    return best[1]


def _logit(value: float) -> float:
    clipped = min(1.0 - 1e-9, max(1e-9, value))
    return math.log(clipped / (1.0 - clipped))


def _decode_trace(
    rows: Sequence[Mapping[str, Any]],
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Mapping[str, Any]]:
    chronological = sorted(
        range(len(rows)),
        key=lambda index: (
            float(rows[index]["transitionTime"]),
            str(rows[index]["eventId"]),
        ),
    )
    ordinal = {index: order for order, index in enumerate(chronological)}
    ranked = sorted(
        range(len(rows)),
        key=lambda index: (
            -float(probabilities[index]),
            float(rows[index]["transitionTime"]),
            str(rows[index]["eventId"]),
        ),
    )
    threshold_logit = _logit(threshold)
    selected: list[int] = []
    trace: dict[str, Mapping[str, Any]] = {}
    for rank, index in enumerate(ranked, 1):
        event_id = str(rows[index]["eventId"])
        blockers = [
            other
            for other in selected
            if abs(ordinal[index] - ordinal[other]) < DECODER.minimum_index_separation
        ]
        if blockers:
            trace[event_id] = {
                "selected": False,
                "reason": "localSuppression",
                "scoreRank": rank,
                "blockedBy": [str(rows[other]["eventId"]) for other in blockers],
            }
            continue
        excess = max(0, len(selected) + 1 - DECODER.free_predictions_per_recording)
        raw_margin = _logit(float(probabilities[index])) - threshold_logit
        adjusted_margin = raw_margin - DECODER.count_penalty_logit * excess
        if adjusted_margin >= -1e-12:
            selected.append(index)
            trace[event_id] = {
                "selected": True,
                "reason": "selected",
                "scoreRank": rank,
                "selectionOrder": len(selected),
                "rawLogitMargin": raw_margin,
                "adjustedLogitMargin": adjusted_margin,
                "softCountExcess": excess,
            }
        else:
            trace[event_id] = {
                "selected": False,
                "reason": "softCountPenalty" if raw_margin >= 0 and excess else "belowThreshold",
                "scoreRank": rank,
                "rawLogitMargin": raw_margin,
                "adjustedLogitMargin": adjusted_margin,
                "softCountExcess": excess,
            }
    return trace


def _fold_states(
    rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, int],
    evaluation: Mapping[str, Any],
) -> tuple[list[FoldState], np.ndarray, np.ndarray]:
    states: list[FoldState] = []
    all_scores = np.full(len(rows), np.nan, dtype=np.float64)
    all_predictions = np.zeros(len(rows), dtype=bool)
    for fold in evaluation["outerFolds"]:
        recording_id = str(fold["heldRecordingId"])
        selected = next(
            value
            for value in fold["leaderboard"]
            if value["variant"]["id"] == "union34-top2-x2"
        )
        threshold = float(selected["threshold"])
        fit_rows = [
            row for row in rows if str(row["recordingId"]) != recording_id
        ]
        indexes = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if str(row["recordingId"]) == recording_id
            ],
            dtype=np.int64,
        )
        held_rows = [rows[index] for index in indexes]
        held_events = _events(held_rows, labels)
        model, _ = _fit(_events(fit_rows, labels), FEATURE_NAMES)
        scores = model.predict_proba(matrix_for(held_events, FEATURE_NAMES))
        predictions = decode_ranked_candidates(
            held_rows, scores, threshold, DECODER
        )
        trace = _decode_trace(held_rows, scores, threshold)
        all_scores[indexes] = scores
        all_predictions[indexes] = predictions
        states.append(
            FoldState(
                recording_id,
                indexes,
                held_rows,
                held_events,
                model,
                threshold,
                scores,
                predictions,
                trace,
            )
        )
    if not np.isfinite(all_scores).all():
        raise ValueError("outer fold reconstruction left rows unscored")
    return states, all_scores, all_predictions


def _metric_summary(
    baseline: Mapping[str, Any], values: Sequence[Mapping[str, float]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("rowAveragePrecision", "precision", "recall", "f1", "proposals"):
        numeric = np.asarray([float(value[name]) for value in values])
        baseline_value = float(baseline[name])
        result[name] = {
            "baseline": baseline_value,
            "permutedMean": float(np.mean(numeric)),
            "permutedP05": float(np.quantile(numeric, 0.05)),
            "permutedP95": float(np.quantile(numeric, 0.95)),
            "importanceBaselineMinusMean": baseline_value - float(np.mean(numeric)),
        }
    return result


def _score_mutation(
    states: Sequence[FoldState],
    rows: Sequence[Mapping[str, Any]],
    labels_array: np.ndarray,
    markers: Mapping[str, Sequence[Mapping[str, Any]]],
    column_indexes: Sequence[int],
    *,
    neutralize: bool,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    samples: list[dict[str, float]] = []
    for _ in range(repeats):
        scores = np.full(len(rows), np.nan, dtype=np.float64)
        predictions = np.zeros(len(rows), dtype=bool)
        for state in states:
            values = matrix_for(state.events, FEATURE_NAMES).copy()
            if neutralize:
                values[:, column_indexes] = state.model.mean[list(column_indexes)]
            else:
                permutation = rng.permutation(len(values))
                values[:, column_indexes] = values[permutation][:, column_indexes]
            local_scores = state.model.predict_proba(values)
            local_predictions = decode_ranked_candidates(
                state.rows, local_scores, state.threshold, DECODER
            )
            scores[state.indexes] = local_scores
            predictions[state.indexes] = local_predictions
        metrics = _evaluate(rows, predictions, markers)
        samples.append(
            {
                "rowAveragePrecision": average_precision(labels_array, scores),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "proposals": float(metrics["proposals"]),
            }
        )
    return {"samples": samples}


def _nested_feature_view(
    rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, int],
    markers: Mapping[str, Sequence[Mapping[str, Any]]],
    feature_names: Sequence[str],
) -> dict[str, Any]:
    scores = np.full(len(rows), np.nan, dtype=np.float64)
    predictions = np.zeros(len(rows), dtype=bool)
    thresholds: dict[str, float] = {}
    for held_id in markers:
        fit_indexes = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if str(row["recordingId"]) != held_id
            ],
            dtype=np.int64,
        )
        held_indexes = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if str(row["recordingId"]) == held_id
            ],
            dtype=np.int64,
        )
        fit_rows = [rows[index] for index in fit_indexes]
        held_rows = [rows[index] for index in held_indexes]
        fit_markers = {key: value for key, value in markers.items() if key != held_id}
        crossfit_scores = _crossfit(fit_rows, labels, feature_names)
        threshold = _select_threshold(fit_rows, crossfit_scores, fit_markers)
        model, _ = _fit(_events(fit_rows, labels), feature_names)
        held_events = _events(held_rows, labels)
        held_scores = model.predict_proba(matrix_for(held_events, feature_names))
        held_predictions = decode_ranked_candidates(
            held_rows, held_scores, threshold, DECODER
        )
        thresholds[held_id] = threshold
        scores[held_indexes] = held_scores
        predictions[held_indexes] = held_predictions
    labels_array = np.asarray([labels[str(row["eventId"])] for row in rows])
    return {
        "featureNames": list(feature_names),
        "rowAveragePrecision": average_precision(labels_array, scores),
        "primary": _evaluate(rows, predictions, markers),
        "strict": _evaluate(rows, predictions, markers, 0.0),
        "thresholdsByRecording": thresholds,
    }


def _masked_serve_anchor_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    for row in rows:
        updated = dict(row)
        features = dict(row["features"])
        if str(row["kind"]) == "internal-dead-state-peak":
            for name in SERVE_ANCHOR_FEATURE_NAMES:
                features[name] = math.nan
        updated["features"] = features
        result.append(updated)
    return result


def _rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    positives = labels == 1
    positive_count = int(np.sum(positives))
    negative_count = len(labels) - positive_count
    u = float(np.sum(ranks[positives])) - positive_count * (positive_count + 1) / 2.0
    return u / (positive_count * negative_count)


def _univariate_crossfit(
    rows: Sequence[Mapping[str, Any]], labels: Mapping[str, int], name: str
) -> dict[str, Any]:
    scores = np.full(len(rows), np.nan, dtype=np.float64)
    signs: list[int] = []
    labels_array = np.asarray([labels[str(row["eventId"])] for row in rows])
    for held_id in sorted({str(row["recordingId"]) for row in rows}):
        fit = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if str(row["recordingId"]) != held_id
            ],
            dtype=np.int64,
        )
        held = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if str(row["recordingId"]) == held_id
            ],
            dtype=np.int64,
        )
        fit_values = np.asarray(
            [float(rows[index]["features"].get(name, math.nan)) for index in fit]
        )
        finite = fit_values[np.isfinite(fit_values)]
        impute = float(np.median(finite)) if len(finite) else 0.0
        fit_values = np.where(np.isfinite(fit_values), fit_values, impute)
        mean = float(np.mean(fit_values))
        scale = max(float(np.std(fit_values)), 1e-6)
        normalized = (fit_values - mean) / scale
        fit_labels = labels_array[fit]
        positive_ap = average_precision(fit_labels, normalized)
        negative_ap = average_precision(fit_labels, -normalized)
        sign = 1 if positive_ap >= negative_ap else -1
        signs.append(sign)
        held_values = np.asarray(
            [float(rows[index]["features"].get(name, math.nan)) for index in held]
        )
        held_values = np.where(np.isfinite(held_values), held_values, impute)
        scores[held] = sign * (held_values - mean) / scale
    finite_count = sum(
        math.isfinite(float(row["features"].get(name, math.nan))) for row in rows
    )
    auc = _rank_auc(labels_array, scores)
    return {
        "feature": name,
        "recordingHeldOutAveragePrecision": average_precision(labels_array, scores),
        "recordingHeldOutAuc": auc,
        "positiveDirectionFolds": int(sum(sign > 0 for sign in signs)),
        "negativeDirectionFolds": int(sum(sign < 0 for sign in signs)),
        "finiteRows": finite_count,
        "missingRows": len(rows) - finite_count,
    }


def _coefficient_rows(
    states: Sequence[FoldState],
    current_model: Mapping[str, Any],
    previous_model: Mapping[str, Any],
) -> list[dict[str, Any]]:
    classifier = current_model["classifier"]
    current_names = tuple(str(value) for value in classifier["featureNames"])
    current_weights = dict(zip(current_names, classifier["weights"], strict=True))
    previous_classifier = previous_model["primaryClassifier"]
    previous_weights = dict(
        zip(
            previous_classifier["featureNames"],
            previous_classifier["weights"],
            strict=True,
        )
    )
    weights = np.stack([state.model.weights for state in states])
    result: list[dict[str, Any]] = []
    for index, name in enumerate(FEATURE_NAMES):
        values = weights[:, index]
        nonnegative = int(np.sum(values >= 0))
        nonpositive = int(np.sum(values <= 0))
        result.append(
            {
                "feature": name,
                "group": next(
                    group for group, names in FEATURE_GROUPS.items() if name in names
                ),
                "finalStandardizedCoefficient": float(current_weights[name]),
                "previousWinnerStandardizedCoefficient": (
                    float(previous_weights[name]) if name in previous_weights else None
                ),
                "outerMeanCoefficient": float(np.mean(values)),
                "outerMedianCoefficient": float(np.median(values)),
                "outerMeanAbsoluteCoefficient": float(np.mean(np.abs(values))),
                "outerStandardDeviation": float(np.std(values)),
                "outerMinimumCoefficient": float(np.min(values)),
                "outerMaximumCoefficient": float(np.max(values)),
                "dominantSignAgreement": max(nonnegative, nonpositive) / len(values),
                "positiveFolds": nonnegative,
                "negativeFolds": nonpositive,
            }
        )
    return result


def _correlations(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    values = np.asarray(
        [
            [float(row["features"].get(name, math.nan)) for name in FEATURE_NAMES]
            for row in rows
        ],
        dtype=np.float64,
    )
    result: list[dict[str, Any]] = []
    for left in range(len(FEATURE_NAMES)):
        for right in range(left + 1, len(FEATURE_NAMES)):
            mask = np.isfinite(values[:, left]) & np.isfinite(values[:, right])
            if int(np.sum(mask)) < 3:
                continue
            left_values = values[mask, left]
            right_values = values[mask, right]
            if float(np.std(left_values)) < 1e-12 or float(np.std(right_values)) < 1e-12:
                continue
            correlation = float(np.corrcoef(left_values, right_values)[0, 1])
            if abs(correlation) >= 0.75:
                result.append(
                    {
                        "left": FEATURE_NAMES[left],
                        "right": FEATURE_NAMES[right],
                        "pearsonCorrelation": correlation,
                    }
                )
    return sorted(result, key=lambda value: -abs(value["pearsonCorrelation"]))


def _contribution_matrix(
    states: Sequence[FoldState], rows: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    contributions = np.full((len(rows), len(FEATURE_NAMES)), np.nan, dtype=np.float64)
    for state in states:
        values = matrix_for(state.events, FEATURE_NAMES)
        filled = np.where(np.isfinite(values), values, state.model.impute)
        contributions[state.indexes] = (
            (filled - state.model.mean) / state.model.scale
        ) * state.model.weights
    if not np.isfinite(contributions).all():
        raise ValueError("feature attribution left non-finite contributions")
    return contributions


def _top_contributors(
    values: np.ndarray, *, positive: bool, limit: int = 5
) -> list[dict[str, Any]]:
    indexes = sorted(
        range(len(values)),
        key=lambda index: float(values[index]),
        reverse=positive,
    )[:limit]
    return [
        {"feature": FEATURE_NAMES[index], "logitContribution": float(values[index])}
        for index in indexes
    ]


def _error_analysis(
    rows: Sequence[Mapping[str, Any]],
    markers: Mapping[str, Sequence[Mapping[str, Any]]],
    targets: Mapping[tuple[str, int], int],
    states: Sequence[FoldState],
    scores: np.ndarray,
    predictions: np.ndarray,
    contributions: np.ndarray,
) -> dict[str, Any]:
    state_by_recording = {state.recording_id: state for state in states}
    detected_target_indexes: list[int] = []
    missed_target_indexes: list[int] = []
    selected_tp_indexes: list[int] = []
    selected_fp_indexes: list[int] = []
    missed_events: list[dict[str, Any]] = []
    false_proposals: list[dict[str, Any]] = []
    by_recording: dict[str, Any] = {}
    reason_counts = {
        "noCandidate": 0,
        "belowThreshold": 0,
        "localSuppression": 0,
        "softCountPenalty": 0,
        "selectedButMatchingConflict": 0,
    }

    for recording_id, truth in markers.items():
        state = state_by_recording[recording_id]
        selected_local = [
            local_index
            for local_index, selected in enumerate(state.predictions)
            if selected
        ]
        selected_proposals = [_proposal(state.rows[index]) for index in selected_local]
        match = monotonic_interval_match(selected_proposals, truth, PADDING)
        matched_markers = {pair.marker_index for pair in match.pairs}
        matched_selected = {pair.proposal_index for pair in match.pairs}
        local_tp = [
            int(state.indexes[selected_local[index]]) for index in sorted(matched_selected)
        ]
        local_fp = [
            int(state.indexes[selected_local[index]])
            for index in match.unmatched_proposal_indices
        ]
        selected_tp_indexes.extend(local_tp)
        selected_fp_indexes.extend(local_fp)

        covered = sum((recording_id, marker_index) in targets for marker_index in range(len(truth)))
        by_recording[recording_id] = {
            "humanEvents": len(truth),
            "candidateCoveredEvents": covered,
            "candidateCoverageRecall": covered / len(truth),
            "threshold": state.threshold,
            "proposals": len(selected_local),
            **event_metric_counts(match),
        }

        for marker_index, marker in enumerate(truth):
            target_index = targets.get((recording_id, marker_index))
            if marker_index in matched_markers:
                if target_index is not None:
                    detected_target_indexes.append(target_index)
                continue
            if target_index is None:
                reason_counts["noCandidate"] += 1
                missed_events.append(
                    {
                        "recordingId": recording_id,
                        "markerTime": float(marker["time"]),
                        "candidateCovered": False,
                        "reason": "noCandidate",
                    }
                )
                continue
            missed_target_indexes.append(target_index)
            row = rows[target_index]
            trace = state.trace[str(row["eventId"])]
            reason = str(trace["reason"])
            if bool(trace["selected"]):
                reason = "selectedButMatchingConflict"
            reason_counts[reason] += 1
            values = contributions[target_index]
            missed_events.append(
                {
                    "recordingId": recording_id,
                    "markerTime": float(marker["time"]),
                    "candidateCovered": True,
                    "targetCandidateId": str(row["eventId"]),
                    "candidateKind": str(row["kind"]),
                    "candidateTransitionTime": float(row["transitionTime"]),
                    "candidateScore": float(scores[target_index]),
                    "threshold": state.threshold,
                    "reason": reason,
                    "decoderTrace": dict(trace),
                    "topNegativeContributors": _top_contributors(values, positive=False),
                    "topPositiveContributors": _top_contributors(values, positive=True),
                    "groupContributions": {
                        group: float(
                            np.sum(
                                [
                                    values[FEATURE_NAMES.index(name)]
                                    for name in names
                                ]
                            )
                        )
                        for group, names in FEATURE_GROUPS.items()
                    },
                }
            )

        for index in local_fp:
            values = contributions[index]
            row = rows[index]
            false_proposals.append(
                {
                    "recordingId": recording_id,
                    "candidateId": str(row["eventId"]),
                    "candidateKind": str(row["kind"]),
                    "transitionTime": float(row["transitionTime"]),
                    "score": float(scores[index]),
                    "threshold": state.threshold,
                    "topPositiveContributors": _top_contributors(values, positive=True),
                    "topNegativeContributors": _top_contributors(values, positive=False),
                    "groupContributions": {
                        group: float(
                            np.sum(
                                [
                                    values[FEATURE_NAMES.index(name)]
                                    for name in names
                                ]
                            )
                        )
                        for group, names in FEATURE_GROUPS.items()
                    },
                }
            )

    def contrast(
        left_indexes: Sequence[int], right_indexes: Sequence[int], label: str
    ) -> list[dict[str, Any]]:
        left = contributions[np.asarray(left_indexes, dtype=np.int64)]
        right = contributions[np.asarray(right_indexes, dtype=np.int64)]
        result = []
        for index, name in enumerate(FEATURE_NAMES):
            left_mean = float(np.mean(left[:, index]))
            right_mean = float(np.mean(right[:, index]))
            result.append(
                {
                    "feature": name,
                    "group": next(
                        group for group, names in FEATURE_GROUPS.items() if name in names
                    ),
                    f"mean{label}LeftContribution": left_mean,
                    f"mean{label}RightContribution": right_mean,
                    "leftMinusRightLogitContribution": left_mean - right_mean,
                }
            )
        return result

    return {
        "rootCauseCounts": reason_counts,
        "targetCandidateCounts": {
            "detectedCoveredTargets": len(detected_target_indexes),
            "missedCoveredTargets": len(missed_target_indexes),
        },
        "selectedProposalCounts": {
            "truePositives": len(selected_tp_indexes),
            "falsePositives": len(selected_fp_indexes),
        },
        "recallContributionContrast": contrast(
            detected_target_indexes, missed_target_indexes, "Recall"
        ),
        "precisionContributionContrast": contrast(
            selected_fp_indexes, selected_tp_indexes, "Precision"
        ),
        "missedEvents": missed_events,
        "falsePositiveProposals": false_proposals,
        "byRecording": by_recording,
    }


def _without_group(group: str) -> tuple[str, ...]:
    removed = set(FEATURE_GROUPS[group])
    return tuple(name for name in FEATURE_NAMES if name not in removed)


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "features": args.features.expanduser().resolve(),
        "fullAudit": args.full_audit.expanduser().resolve(),
        "model": args.model.expanduser().resolve(),
        "evaluation": args.evaluation.expanduser().resolve(),
        "previousModel": args.previous_model.expanduser().resolve(),
    }
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite feature-importance artifact: {output}")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if args.enforce_source_hash and hashes != EXPECTED_SHA256:
        raise ValueError(f"feature-importance source identity changed: {hashes}")

    feature_payload = _load(paths["features"])
    audit = _load(paths["fullAudit"])
    model_payload = _load(paths["model"])
    evaluation = _load(paths["evaluation"])
    previous_model = _load(paths["previousModel"])
    rows = [add_derived_features(row) for row in feature_payload["rows"]]
    recording_ids = tuple(str(value) for value in feature_payload["scope"]["recordingIds"])
    markers = {
        recording_id: [
            {"time": float(value)}
            for value in audit["scope"]["humanEventsByRecording"][recording_id]
        ]
        for recording_id in recording_ids
    }
    labels, targets = _labels_and_targets(rows, markers)
    labels_array = np.asarray([labels[str(row["eventId"])] for row in rows])
    if int(np.sum(labels_array)) != 46:
        raise ValueError("candidate label universe changed")

    states, scores, predictions = _fold_states(rows, labels, evaluation)
    baseline_metrics = _evaluate(rows, predictions, markers)
    expected = evaluation["fixedVariantOuterResults"]["union34-top2-x2"]["primary"]
    for name in ("proposals", "truePositives", "falsePositives", "falseNegatives"):
        if int(baseline_metrics[name]) != int(expected[name]):
            raise ValueError(f"baseline reconstruction drifted for {name}")
    baseline = {
        "rowAveragePrecision": average_precision(labels_array, scores),
        **{key: value for key, value in baseline_metrics.items() if key != "byRecording"},
        "byRecording": baseline_metrics["byRecording"],
    }

    coefficient_importance = _coefficient_rows(states, model_payload, previous_model)
    permutation: list[dict[str, Any]] = []
    neutralization: list[dict[str, Any]] = []
    for index, name in enumerate(FEATURE_NAMES):
        permuted = _score_mutation(
            states,
            rows,
            labels_array,
            markers,
            [index],
            neutralize=False,
            repeats=args.permutation_repeats,
            seed=args.seed + index * 1009,
        )
        neutralized = _score_mutation(
            states,
            rows,
            labels_array,
            markers,
            [index],
            neutralize=True,
            repeats=1,
            seed=args.seed,
        )
        permutation.append(
            {
                "feature": name,
                "group": next(
                    group for group, names in FEATURE_GROUPS.items() if name in names
                ),
                **_metric_summary(baseline, permuted["samples"]),
            }
        )
        neutralization.append(
            {
                "feature": name,
                "group": next(
                    group for group, names in FEATURE_GROUPS.items() if name in names
                ),
                "result": neutralized["samples"][0],
                "baselineMinusResult": {
                    key: float(baseline[key]) - float(neutralized["samples"][0][key])
                    for key in (
                        "rowAveragePrecision",
                        "precision",
                        "recall",
                        "f1",
                        "proposals",
                    )
                },
            }
        )

    group_permutation: list[dict[str, Any]] = []
    group_neutralization: list[dict[str, Any]] = []
    for group_index, (group, names) in enumerate(FEATURE_GROUPS.items()):
        indexes = [FEATURE_NAMES.index(name) for name in names]
        permuted = _score_mutation(
            states,
            rows,
            labels_array,
            markers,
            indexes,
            neutralize=False,
            repeats=args.permutation_repeats,
            seed=args.seed + 50000 + group_index * 1009,
        )
        neutralized = _score_mutation(
            states,
            rows,
            labels_array,
            markers,
            indexes,
            neutralize=True,
            repeats=1,
            seed=args.seed,
        )
        group_permutation.append(
            {
                "group": group,
                "featureNames": list(names),
                **_metric_summary(baseline, permuted["samples"]),
            }
        )
        group_neutralization.append(
            {
                "group": group,
                "featureNames": list(names),
                "result": neutralized["samples"][0],
                "baselineMinusResult": {
                    key: float(baseline[key]) - float(neutralized["samples"][0][key])
                    for key in (
                        "rowAveragePrecision",
                        "precision",
                        "recall",
                        "f1",
                        "proposals",
                    )
                },
            }
        )

    nested_ablations: list[dict[str, Any]] = []
    print("[nested 1/9] current union34", flush=True)
    nested_ablations.append(
        {
            "id": "currentUnion34",
            "change": "none",
            **_nested_feature_view(rows, labels, markers, FEATURE_NAMES),
        }
    )
    for position, group in enumerate(FEATURE_GROUPS, 2):
        print(f"[nested {position}/9] drop {group}", flush=True)
        nested_ablations.append(
            {
                "id": f"drop-{group}",
                "change": "dropFeatureGroup",
                "droppedGroup": group,
                **_nested_feature_view(
                    rows, labels, markers, _without_group(group)
                ),
            }
        )
    print("[nested 8/9] drop correlated internal-identity triplet", flush=True)
    internal_identity = {
        "productionGapLiveFraction",
        "candidateIsInternalDeadStatePeak",
        "candidateGeneratorScore",
    }
    nested_ablations.append(
        {
            "id": "drop-internalIdentityTriplet",
            "change": "dropOverlappingDiagnosticGroup",
            "droppedFeatureNames": sorted(internal_identity),
            "reason": (
                "these three inputs are statistically interchangeable candidate-kind "
                "indicators on the retained union"
            ),
            **_nested_feature_view(
                rows,
                labels,
                markers,
                tuple(name for name in FEATURE_NAMES if name not in internal_identity),
            ),
        }
    )
    print("[nested 9/9] add masked serve-anchor features", flush=True)
    masked_rows = _masked_serve_anchor_rows(rows)
    nested_ablations.append(
        {
            "id": "addMaskedServeAnchor10",
            "change": "addFeatureGroup",
            "addedFeatureNames": list(SERVE_ANCHOR_FEATURE_NAMES),
            "internalCandidatePolicy": "values masked to missing and fit-imputed",
            **_nested_feature_view(
                masked_rows,
                labels,
                markers,
                (*FEATURE_NAMES, *SERVE_ANCHOR_FEATURE_NAMES),
            ),
        }
    )

    univariate_rows = _masked_serve_anchor_rows(rows)
    univariate = [
        _univariate_crossfit(univariate_rows, labels, name)
        for name in (*FEATURE_NAMES, *SERVE_ANCHOR_FEATURE_NAMES)
    ]
    contributions = _contribution_matrix(states, rows)
    error_analysis = _error_analysis(
        rows,
        markers,
        targets,
        states,
        scores,
        predictions,
        contributions,
    )

    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-feature-importance-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "scope": {
            **feature_payload["scope"],
            "humanMarkers": sum(len(value) for value in markers.values()),
            "positiveCandidateLabels": int(np.sum(labels_array)),
            "status": "opened-development-only",
        },
        "methodology": {
            "model": "fixed promoted union34-top2-x2 hard-negative variant",
            "outerProtocol": (
                "each row uses a model fit without its recording and the exact fit-selected "
                "threshold stored in the promoted evaluation"
            ),
            "individualImportance": (
                "within-recording permutation and mean-neutralization of held rows; "
                "positive baseline-minus-mutated values mean the fitted model relies on the feature"
            ),
            "coefficientStability": "standardized coefficients from the 11 outer-fit models",
            "groupAblation": (
                "drop one complete feature family, refit top-2/2x hard-negative models, "
                "and reselect thresholds using grouped inner LOO on every outer fit scope"
            ),
            "errorAttribution": (
                "fold-specific standardized feature x coefficient logit contributions, "
                "with candidate-generation misses separated from classifier/decoder misses"
            ),
            "permutationRepeats": args.permutation_repeats,
            "permutationSeed": args.seed,
            "primaryEventPaddingSeconds": PADDING,
        },
        "baseline": baseline,
        "coefficientImportance": coefficient_importance,
        "individualPermutationImportance": permutation,
        "individualNeutralizationImportance": neutralization,
        "groupPermutationImportance": group_permutation,
        "groupNeutralizationImportance": group_neutralization,
        "nestedFeatureGroupAblations": nested_ablations,
        "recordingHeldOutUnivariateScreens": univariate,
        "highCorrelationPairs": _correlations(rows),
        "errorAnalysis": error_analysis,
        "sources": {
            name: {"path": str(paths[name]), "sha256": hashes[name]} for name in paths
        },
        "limitations": [
            "All 11 recordings and all 50 markers are opened development data, not an untouched test set.",
            "Permutation and neutralization explain the fitted linear heads; correlated features can share or substitute importance.",
            "Nested group ablations reselect thresholds but test feature families after this development scope was opened.",
            "Four human events have no candidate within the declared +/-4-second union and therefore have no classifier feature row to explain.",
            "The ten serve-anchor inputs are masked for internal candidates because their same-containing-range values are not semantically eligible.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    if output.exists():
        replacement = output.with_name(
            f".{output.stem}-{uuid.uuid4().hex}{output.suffix}"
        )
        try:
            atomic_write_text(replacement, serialized)
            replacement.replace(output)
        finally:
            replacement.unlink(missing_ok=True)
    else:
        atomic_write_text(output, serialized)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--full-audit", type=Path, default=DEFAULT_FULL_AUDIT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--previous-model", type=Path, default=DEFAULT_PREVIOUS_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--permutation-repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--enforce-source-hash", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.permutation_repeats < 1:
        raise ValueError("permutation repeats must be positive")
    payload = run(args)
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "baseline": {
                    key: value
                    for key, value in payload["baseline"].items()
                    if key != "byRecording"
                },
                "rootCauseCounts": payload["errorAnalysis"]["rootCauseCounts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
