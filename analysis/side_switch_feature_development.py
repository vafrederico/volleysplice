"""Controlled feature-profile experiments for the promoted side-switch ranker."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from analysis.side_switch_full_union_ranker import (
    DERIVED_FEATURE_NAMES,
    UnionDecoderSettings,
    add_derived_features,
    decode_ranked_candidates,
    fit_weighted_logistic,
)
from analysis.side_switch_full_video import event_metric_counts, monotonic_interval_match
from analysis.side_switch_production_state import STATE_GATE_FEATURE_NAMES
from analysis.side_switch_v3 import V3Event, average_precision
from analysis.side_switch_v5 import VISUAL_FEATURE_NAMES
from analysis.side_switch_v6 import V6Model, matrix_for


BASE_FEATURE_NAMES = (
    *VISUAL_FEATURE_NAMES,
    *STATE_GATE_FEATURE_NAMES,
    *DERIVED_FEATURE_NAMES,
)
INTERACTION_FEATURE_NAMES = (
    "jointSwapMarginMinimum",
    "positiveSwapMarginProduct",
    "swapMarginDisagreement",
    "playerSwapSpecificity",
    "courtSwapSpecificity",
    "playerQualityGatedSwap",
    "courtQualityGatedSwap",
)
L2 = 0.1
CLASS_BALANCE_EXPONENT = 0.5
HARD_NEGATIVES_PER_RECORDING = 2
HARD_NEGATIVE_MULTIPLIER = 2.0
PADDING_SECONDS = 4.0
THRESHOLD_QUANTILES = 65
DECODER = UnionDecoderSettings(
    minimum_index_separation=2,
    free_predictions_per_recording=6,
    count_penalty_logit=0.5,
)

FeatureBuilder = Callable[[Mapping[str, float]], Mapping[str, float]]


@dataclass(frozen=True)
class FeatureProfile:
    identifier: str
    feature_names: tuple[str, ...]
    hypothesis: str
    builder: FeatureBuilder | None = None

    def __post_init__(self) -> None:
        if not self.identifier or not self.hypothesis:
            raise ValueError("feature profile identity and hypothesis are required")
        if not self.feature_names or len(set(self.feature_names)) != len(
            self.feature_names
        ):
            raise ValueError("feature profile names must be non-empty and unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "featureNames": list(self.feature_names),
            "hypothesis": self.hypothesis,
        }


BASELINE_PROFILE = FeatureProfile(
    identifier="union34-v1",
    feature_names=BASE_FEATURE_NAMES,
    hypothesis="exact promoted 34-input control",
)


def swap_interaction_features(features: Mapping[str, float]) -> dict[str, float]:
    """Materialize the preregistered I1 swap-specific interaction bundle."""

    player = float(features["playerSwapMargin"])
    court = float(features["v4MeanSwapMargin"])
    player_change = float(features["playerGlobalAppearanceChange"])
    court_change = float(features["v4GlobalAppearanceChange"])
    separation = float(features["minimumPlayerSideSeparation"])
    coverage = float(features["minimumProposalCoverage"])
    alignment = float(features["v4MinimumAlignmentResponse"])
    values = (
        player,
        court,
        player_change,
        court_change,
        separation,
        coverage,
        alignment,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("I1 source features must be finite")

    def specificity(margin: float, change: float) -> float:
        return min(5.0, max(-5.0, margin / max(change, 0.02)))

    return {
        "jointSwapMarginMinimum": min(player, court),
        "positiveSwapMarginProduct": max(player, 0.0) * max(court, 0.0),
        "swapMarginDisagreement": abs(player - court),
        "playerSwapSpecificity": specificity(player, player_change),
        "courtSwapSpecificity": specificity(court, court_change),
        "playerQualityGatedSwap": player * separation * coverage,
        "courtQualityGatedSwap": court * alignment,
    }


INTERACTION_PROFILE = FeatureProfile(
    identifier="union34-plus-interactions-i1",
    feature_names=(*BASE_FEATURE_NAMES, *INTERACTION_FEATURE_NAMES),
    hypothesis=(
        "explicit swap agreement, specificity, and quality gates reduce generic "
        "high-player-change false positives without losing covered switches"
    ),
    builder=swap_interaction_features,
)


def apply_profile(
    source_rows: Sequence[Mapping[str, Any]], profile: FeatureProfile
) -> list[dict[str, Any]]:
    """Return copied rows with deterministic profile features materialized by name."""

    result: list[dict[str, Any]] = []
    for source in source_rows:
        row = add_derived_features(source)
        features = {
            str(name): float(value)
            for name, value in row.get("features", {}).items()
        }
        if profile.builder is not None:
            derived = profile.builder(features)
            for name, value in derived.items():
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ValueError(
                        f"profile {profile.identifier} produced non-finite {name}"
                    )
                features[str(name)] = numeric
        missing = [name for name in profile.feature_names if name not in features]
        if missing:
            raise ValueError(
                f"profile {profile.identifier} is missing features: {missing}"
            )
        updated = dict(row)
        updated["features"] = features
        result.append(updated)
    return result


def proposal(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "eventId": str(row["eventId"]),
        "recordingId": str(row["recordingId"]),
        "kind": str(row["kind"]),
        "gapStart": float(row["gapStart"]),
        "gapEnd": float(row["gapEnd"]),
        "transitionTime": float(row["transitionTime"]),
    }


def labels_for_rows(
    rows: Sequence[Mapping[str, Any]],
    markers: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, int]:
    labels = {str(row["eventId"]): 0 for row in rows}
    for recording_id, truth in markers.items():
        local = [row for row in rows if str(row["recordingId"]) == recording_id]
        match = monotonic_interval_match(
            [proposal(row) for row in local], truth, PADDING_SECONDS
        )
        for pair in match.pairs:
            labels[str(local[pair.proposal_index]["eventId"])] = 1
    return labels


def events_for_rows(
    rows: Sequence[Mapping[str, Any]], labels: Mapping[str, int]
) -> list[V3Event]:
    order: dict[str, int] = {}
    for recording_id in sorted({str(row["recordingId"]) for row in rows}):
        local = sorted(
            (row for row in rows if str(row["recordingId"]) == recording_id),
            key=lambda row: (float(row["transitionTime"]), str(row["eventId"])),
        )
        order.update(
            {str(row["eventId"]): index for index, row in enumerate(local, 1)}
        )
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


def fit_profile(
    events: Sequence[V3Event], feature_names: Sequence[str]
) -> tuple[V6Model, dict[str, list[str]]]:
    names = tuple(feature_names)
    initial = fit_weighted_logistic(events, L2, names, CLASS_BALANCE_EXPONENT)
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
    model = fit_weighted_logistic(
        events,
        L2,
        names,
        CLASS_BALANCE_EXPONENT,
        multipliers,
    )
    return model, selected


def crossfit_scores(
    rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, int],
    feature_names: Sequence[str],
) -> np.ndarray:
    events = events_for_rows(rows, labels)
    scores = np.full(len(rows), np.nan, dtype=np.float64)
    for held_id in sorted({event.recording_id for event in events}):
        fit = [index for index, event in enumerate(events) if event.recording_id != held_id]
        held = [index for index, event in enumerate(events) if event.recording_id == held_id]
        model, _ = fit_profile([events[index] for index in fit], feature_names)
        scores[held] = model.predict_proba(
            matrix_for([events[index] for index in held], feature_names)
        )
    if not np.isfinite(scores).all():
        raise ValueError("feature-profile cross-fit left rows unscored")
    return scores


def evaluate_predictions(
    rows: Sequence[Mapping[str, Any]],
    predictions: np.ndarray,
    markers: Mapping[str, Sequence[Mapping[str, Any]]],
    padding: float,
    *,
    inventory: bool = False,
) -> dict[str, Any]:
    selected = np.asarray(predictions, dtype=bool)
    if selected.shape != (len(rows),):
        raise ValueError("prediction mask is not aligned with candidate rows")
    by_recording: dict[str, Any] = {}
    for recording_id, truth in markers.items():
        proposals = sorted(
            [
                proposal(row)
                for row, keep in zip(rows, selected, strict=True)
                if keep and str(row["recordingId"]) == recording_id
            ],
            key=lambda row: (row["transitionTime"], row["eventId"]),
        )
        match = monotonic_interval_match(proposals, truth, padding)
        metrics = event_metric_counts(match)
        row: dict[str, Any] = {
            "humanEvents": len(truth),
            "proposals": len(proposals),
            **metrics,
            "missedHumanTimes": [
                float(truth[index]["time"])
                for index in match.unmatched_marker_indices
            ],
        }
        if inventory:
            row["proposalInventory"] = proposals
        by_recording[recording_id] = row
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


def threshold_candidates(scores: np.ndarray) -> tuple[float, ...]:
    values = np.asarray(scores, dtype=np.float64)
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, THRESHOLD_QUANTILES))
    above = min(1.0, math.nextafter(float(np.max(values)), math.inf))
    return tuple(sorted({above, *(float(value) for value in quantiles)}, reverse=True))


def _threshold_rank(metrics: Mapping[str, Any], threshold: float) -> tuple[float, ...]:
    return (
        float(metrics["f1"]),
        float(metrics["precision"]),
        float(metrics["recall"]),
        -float(metrics["proposals"]),
        threshold,
    )


def select_threshold(
    rows: Sequence[Mapping[str, Any]],
    scores: np.ndarray,
    markers: Mapping[str, Sequence[Mapping[str, Any]]],
) -> float:
    best: tuple[tuple[float, ...], float] | None = None
    for threshold in threshold_candidates(scores):
        predictions = decode_ranked_candidates(rows, scores, threshold, DECODER)
        metrics = evaluate_predictions(rows, predictions, markers, PADDING_SECONDS)
        candidate = (_threshold_rank(metrics, threshold), threshold)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        raise AssertionError("feature-profile threshold selection failed")
    return best[1]


def _metrics_without_recordings(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "byRecording"}


def evaluate_profile(
    source_rows: Sequence[Mapping[str, Any]],
    markers: Mapping[str, Sequence[Mapping[str, Any]]],
    profile: FeatureProfile,
) -> dict[str, Any]:
    rows = apply_profile(source_rows, profile)
    labels = labels_for_rows(rows, markers)
    labels_array = np.asarray(
        [labels[str(row["eventId"])] for row in rows], dtype=np.int64
    )
    scores = np.full(len(rows), np.nan, dtype=np.float64)
    predictions = np.zeros(len(rows), dtype=bool)
    outer_folds: list[dict[str, Any]] = []

    for held_id in sorted(markers):
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
        fit_scores = crossfit_scores(fit_rows, labels, profile.feature_names)
        threshold = select_threshold(fit_rows, fit_scores, fit_markers)
        model, hard_negatives = fit_profile(
            events_for_rows(fit_rows, labels), profile.feature_names
        )
        held_scores = model.predict_proba(
            matrix_for(events_for_rows(held_rows, labels), profile.feature_names)
        )
        held_predictions = decode_ranked_candidates(
            held_rows, held_scores, threshold, DECODER
        )
        scores[held_indexes] = held_scores
        predictions[held_indexes] = held_predictions
        outer_folds.append(
            {
                "heldRecordingId": held_id,
                "threshold": threshold,
                "fitHardNegativesByRecording": hard_negatives,
                "heldCandidateScores": [
                    {
                        "eventId": str(row["eventId"]),
                        "kind": str(row["kind"]),
                        "probability": float(score),
                        "selected": bool(selected),
                        "label": int(labels[str(row["eventId"])]),
                    }
                    for row, score, selected in zip(
                        held_rows, held_scores, held_predictions, strict=True
                    )
                ],
            }
        )

    if not np.isfinite(scores).all():
        raise ValueError("outer feature-profile evaluation left rows unscored")

    primary = evaluate_predictions(
        rows, predictions, markers, PADDING_SECONDS, inventory=True
    )
    strict = evaluate_predictions(rows, predictions, markers, 0.0, inventory=True)
    row_ap = average_precision(labels_array, scores)
    by_kind: dict[str, Any] = {}
    for kind in sorted({str(row["kind"]) for row in rows}):
        indexes = np.asarray(
            [index for index, row in enumerate(rows) if str(row["kind"]) == kind],
            dtype=np.int64,
        )
        kind_predictions = predictions.copy()
        kind_predictions[
            np.asarray(
                [index for index in range(len(rows)) if index not in set(indexes)],
                dtype=np.int64,
            )
        ] = False
        by_kind[kind] = {
            "candidates": len(indexes),
            "positiveCandidateLabels": int(np.sum(labels_array[indexes])),
            "rowAveragePrecision": average_precision(labels_array[indexes], scores[indexes]),
            "primary": _metrics_without_recordings(
                evaluate_predictions(rows, kind_predictions, markers, PADDING_SECONDS)
            ),
        }

    full_scores = crossfit_scores(rows, labels, profile.feature_names)
    full_threshold = select_threshold(rows, full_scores, markers)
    final_model, final_hard_negatives = fit_profile(
        events_for_rows(rows, labels), profile.feature_names
    )
    classifier = final_model.to_dict()
    classifier["threshold"] = full_threshold

    return {
        "profile": profile.to_dict(),
        "candidateRows": len(rows),
        "positiveCandidateLabels": int(np.sum(labels_array)),
        "candidateCoverage": {
            "coveredHumanEvents": int(np.sum(labels_array)),
            "humanEvents": sum(len(value) for value in markers.values()),
            "recall": int(np.sum(labels_array))
            / sum(len(value) for value in markers.values()),
        },
        "rowAveragePrecision": row_ap,
        "strict": strict,
        "primary": primary,
        "byCandidateKind": by_kind,
        "outerFolds": outer_folds,
        "fullDevelopment": {
            "crossfitRowAveragePrecision": average_precision(labels_array, full_scores),
            "selectedThreshold": full_threshold,
            "classifier": classifier,
            "hardNegativesByRecording": final_hard_negatives,
        },
    }


def metric_delta(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "rowAveragePrecision": float(candidate["rowAveragePrecision"])
        - float(baseline["rowAveragePrecision"]),
        "strictF1": float(candidate["strict"]["f1"])
        - float(baseline["strict"]["f1"]),
        "primaryPrecision": float(candidate["primary"]["precision"])
        - float(baseline["primary"]["precision"]),
        "primaryRecall": float(candidate["primary"]["recall"])
        - float(baseline["primary"]["recall"]),
        "primaryF1": float(candidate["primary"]["f1"])
        - float(baseline["primary"]["f1"]),
        "proposals": int(candidate["primary"]["proposals"])
        - int(baseline["primary"]["proposals"]),
        "truePositives": int(candidate["primary"]["truePositives"])
        - int(baseline["primary"]["truePositives"]),
        "falsePositives": int(candidate["primary"]["falsePositives"])
        - int(baseline["primary"]["falsePositives"]),
        "falseNegatives": int(candidate["primary"]["falseNegatives"])
        - int(baseline["primary"]["falseNegatives"]),
    }


def concise_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rowAveragePrecision": result["rowAveragePrecision"],
        "strict": _metrics_without_recordings(result["strict"]),
        "primary": _metrics_without_recordings(result["primary"]),
    }
